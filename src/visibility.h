#pragma once

// Symbol visibility control for the FFI entry points.
//
// The libraries are compiled with hidden default visibility and linked with
// --exclude-libs,ALL, so nothing leaves the dynamic symbol table unless it is
// annotated with RI_KERNELS_API. That keeps symbols pulled in from static
// archives (Highway, and the CUDA runtime when linked statically) private to
// the library. Otherwise they would be exported, and a CUDA runtime already
// loaded into the process by XLA could interpose on our calls -- or ours on
// its -- because the dynamic loader searches the global scope before a
// dlopen'd object's own definitions.
//
// The libraries are only ever dlopen'd (from ctypes in rfi_vis_op.py) and
// never linked against, so no dllimport variant is needed.
//
// Note that the attribute must be placed inside the extern "C" linkage
// specification. Written in front of it, as in
//
//     RI_KERNELS_API XLA_FFI_DECLARE_HANDLER_SYMBOL(calc_rfi_vis_cpu);
//
// GCC discards it with only a -Wattributes warning and the symbol stays
// hidden, so declare the handlers by spelling out the signature:
//
//     extern "C" RI_KERNELS_API XLA_FFI_Error *
//     calc_rfi_vis_cpu(XLA_FFI_CallFrame *call_frame);

#if defined(_WIN32) || defined(__CYGWIN__)
#define RI_KERNELS_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define RI_KERNELS_API __attribute__((visibility("default")))
#else
#define RI_KERNELS_API
#endif
