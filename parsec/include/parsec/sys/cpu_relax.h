/*
 * Copyright (c) 2026 NVIDIA Corporation.  All rights reserved.
 */

#ifndef PARSEC_CPU_RELAX_H_HAS_BEEN_INCLUDED
#define PARSEC_CPU_RELAX_H_HAS_BEEN_INCLUDED

/* This file only holds an inline function, so it does not need BEGIN_C_DECLS,
 * and it is architecture dependent: it should only be compiled within the
 * PaRSEC sources */

#if !defined(BUILDING_PARSEC)
#error "This file is architecture dependent and should not be included from a source file that does not belong to PaRSEC core code"
#endif

#include "parsec/parsec_config.h"

/**
 * @ingroup parsec_internal_runtime
 * @{
 */

/**
 * Hint to the CPU that the calling thread is in a busy-wait loop.
 *
 * @details
 *   Call this in every iteration of a loop that spins on a memory location
 *   another thread is going to change. It lowers the power used by the
 *   spinning thread, leaves the resources of a core to its other hardware
 *   threads, and on some architectures avoids the penalty of leaving the loop
 *   with a mis-speculated memory order. It does not give up the core: for a
 *   wait that can be long, e.g. because the thread waited for can be
 *   descheduled, give up the core after some iterations (sched_yield()).
 *
 *   It is also a compiler barrier, so the value spun upon is read again at
 *   each iteration, and it implies no memory ordering.
 */
static inline void parsec_cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__powerpc__) || defined(__ppc__)
    __asm__ __volatile__("or 27,27,27" ::: "memory");
#elif defined(__GNUC__)
    __asm__ __volatile__("" ::: "memory");
#endif
}

/** @} */

#endif  /* PARSEC_CPU_RELAX_H_HAS_BEEN_INCLUDED */
