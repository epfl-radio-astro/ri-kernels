#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

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


} // namespace ri_kernels
