#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ri_kernels {

namespace ffi = xla::ffi;

/**
 * @brief Wrap an already-known result in a Future that is complete on return.
 *
 * Used on the validation paths of the CPU handlers, which have to produce the
 * same return type as the parallel path.
 */
inline ffi::Future completed_future(ffi::Error error) {
  ffi::Promise promise;
  ffi::Future future(promise); // must be created while use_count() == 2
  if (error.success()) {
    promise.SetAvailable();
  } else {
    promise.SetError(std::move(error));
  }
  return future;
}

/**
 * @brief Split [0, n_items) into contiguous chunks and run them on XLA's
 *        intra-op pool, without ever blocking the calling thread.
 *
 * The handler that calls this is running *on* an XLA intra-op worker. Waiting
 * there for work queued to that same pool is a thread-pool inversion: under
 * multi-device CPU execution each device also parks a worker inside the
 * collective rendezvous, and once every worker is parked there is nobody left
 * to drain the queued chunks. The kernels then never return and the rendezvous
 * never completes. Returning the Future instead lets the worker go straight
 * back to the queue; XLA resumes the thunk when the last chunk signals.
 *
 * Consequently @p body runs *after* the handler has returned. It must not touch
 * the XLA FFI call frame - `ffi::Buffer`/`ffi::Result` hold a pointer into it,
 * so `dimensions()` in particular is out of bounds. Snapshot the data pointers
 * and extents into value types (e.g. the Tensor*D views) before calling here
 * and capture those by value.
 *
 * @param thread_pool The pool from the FFI execution context.
 * @param n_items     Size of the iteration space.
 * @param body        Copyable callable invoked as `body(begin, end)` per chunk.
 * @return A Future that becomes available once every chunk has run.
 */
template <typename F>
ffi::Future parallel_for(ffi::ThreadPool &thread_pool, std::int64_t n_items,
                         F body) {
  if (n_items <= 0) {
    return completed_future(ffi::Error::Success());
  }

  const std::int64_t max_chunks =
      std::max<std::int64_t>(thread_pool.num_threads(), 1);
  const std::int64_t chunk_size = (n_items + max_chunks - 1) / max_chunks;
  const std::int64_t n_chunks = (n_items + chunk_size - 1) / chunk_size;

  // A single chunk - either a pool of one, or fewer items than workers - has
  // nothing to gain from a round trip through the queue. Running it inline is
  // also what keeps the degenerate one-schedulable-CPU case working.
  if (n_chunks == 1) {
    body(std::int64_t{0}, n_items);
    return completed_future(ffi::Error::Success());
  }

  ffi::CountDownPromise promise(n_chunks);
  ffi::Future future(promise);

  for (std::int64_t i_chunk = 0; i_chunk < n_chunks; ++i_chunk) {
    const std::int64_t begin = i_chunk * chunk_size;
    const std::int64_t end = std::min(begin + chunk_size, n_items);

    // `body` and `promise` are copied into every task: the enclosing frame is
    // gone by the time these run. ThreadPool::Schedule may also run the task
    // inline if no pool is available, which this is safe under.
    thread_pool.Schedule([promise, body, begin, end]() mutable {
      body(begin, end);
      promise.CountDown();
    });
  }

  return future;
}


/**
 * @brief Run several parallel loops one after another, without blocking.
 *
 * Each phase is split into chunks like parallel_for(); the chunks of phase
 * k + 1 are scheduled by whichever chunk of phase k finishes last, so a
 * phase sees the whole of the previous one and no worker ever waits. The
 * returned Future completes with the last chunk of the last phase. The same
 * lifetime rules as parallel_for() apply, and the thread pool is used from
 * inside its own tasks, the recursive scheduling ThreadPool::Schedule
 * allows.
 *
 * @param phases  A list of (n_items, body) pairs, bodies as for parallel_for().
 */
template <typename F>
ffi::Future parallel_phases(ffi::ThreadPool &thread_pool,
                            std::vector<std::pair<std::int64_t, F>> phases) {
  struct Chunk {
    std::int64_t phase, begin, end;
  };
  const std::int64_t max_chunks = std::max<std::int64_t>(thread_pool.num_threads(), 1);
  std::vector<std::vector<Chunk>> chunks(phases.size());
  std::int64_t total = 0;
  for (std::size_t k = 0; k < phases.size(); ++k) {
    const std::int64_t n_items = phases[k].first;
    if (n_items <= 0) continue;
    const std::int64_t chunk_size = (n_items + max_chunks - 1) / max_chunks;
    for (std::int64_t begin = 0; begin < n_items; begin += chunk_size)
      chunks[k].push_back({std::int64_t(k), begin, std::min(begin + chunk_size, n_items)});
    total += std::int64_t(chunks[k].size());
  }
  if (total == 0) return completed_future(ffi::Error::Success());
  if (max_chunks == 1) {  // one worker: everything inline, in order
    for (std::size_t k = 0; k < phases.size(); ++k)
      for (const Chunk &c : chunks[k]) phases[k].second(c.begin, c.end);
    return completed_future(ffi::Error::Success());
  }

  struct State {
    std::vector<std::pair<std::int64_t, F>> phases;
    std::vector<std::vector<Chunk>> chunks;
    std::vector<std::atomic<std::int64_t>> remaining;
    State(std::vector<std::pair<std::int64_t, F>> p, std::vector<std::vector<Chunk>> c)
        : phases(std::move(p)), chunks(std::move(c)), remaining(chunks.size()) {
      for (std::size_t k = 0; k < chunks.size(); ++k) remaining[k] = std::int64_t(chunks[k].size());
    }
  };
  auto state = std::make_shared<State>(std::move(phases), std::move(chunks));
  ffi::CountDownPromise promise(total);
  ffi::Future future(promise);

  // Schedules phase k's chunks; each chunk, when it is the last of its phase
  // to finish, schedules the next non-empty phase.
  struct Launch {
    ffi::ThreadPool pool;
    std::shared_ptr<State> state;
    ffi::CountDownPromise promise;
    // Non-const: ThreadPool::Schedule is a non-const member.
    void operator()(std::size_t k) {
      while (k < state->chunks.size() && state->chunks[k].empty()) ++k;
      if (k >= state->chunks.size()) return;
      for (const Chunk &c : state->chunks[k]) {
        Launch self = *this;
        pool.Schedule([self, c]() mutable {
          self.state->phases[c.phase].second(c.begin, c.end);
          const bool last = self.state->remaining[c.phase].fetch_sub(1) == 1;
          if (last) self(std::size_t(c.phase) + 1);
          self.promise.CountDown();
        });
      }
    }
  };
  Launch{thread_pool, state, promise}(0);
  return future;
}

} // namespace ri_kernels
