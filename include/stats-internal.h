/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _statsinternal_h
#define _statsinternal_h

#ifndef stats_h
#error "Never include stats-internal.h directly: include stats.h"
#endif

#include "dague_config.h"

#if defined(DAGUE_STATS)

#if defined(DAGUE_STATS_C_DECLARE)

#include <stdint.h>
#define DECLARE_STAT(name)                                  \
    volatile uint64_t dague_stats_##name##_max = 0;       \
    volatile uint64_t dague_stats_##name##_current = 0

#define DECLARE_STATMAX(name)                           \
    volatile uint64_t dague_stats_##name##_max = 0

#define DECLARE_STATACC(name)                           \
    volatile uint64_t dague_stats_##name##_acc = 0

#elif defined(DAGUE_STATS_C_DUMP)

#define DECLARE_STAT(name) \
    fprintf(statfile, "%s%s%-40s\t(MAX)\t%13llu\n", prefix != NULL ? prefix : "", prefix != NULL ? " " : "", #name, (unsigned long long)dague_stats_##name##_max)
#define DECLARE_STATMAX(name) \
    fprintf(statfile, "%s%s%-40s\t(MAX)\t%13llu\n", prefix != NULL ? prefix : "", prefix != NULL ? " " : "", #name, (unsigned long long)dague_stats_##name##_max)
#define DECLARE_STATACC(name) \
    fprintf(statfile, "%s%s%-40s\t(ACC)\t%13llu\n", prefix != NULL ? prefix : "", prefix != NULL ? " " : "", #name, (unsigned long long)dague_stats_##name##_acc)

#else /* no C-magic */

#include <dague/sys/atomic.h>

#define DECLARE_STAT(name)                                   \
    extern volatile uint64_t dague_stats_##name##_max;     \
    extern volatile uint64_t dague_stats_##name##_current

#define DECLARE_STATMAX(name)                           \
    extern volatile uint64_t dague_stats_##name##_max

#define DECLARE_STATACC(name)                           \
    extern volatile uint64_t dague_stats_##name##_acc

#define DAGUE_STAT_INCREASE(name, value)                   \
    __dague_stat_increase(&dague_stats_##name##_current, \
                            &dague_stats_##name##_max,     \
                            value)

static inline void __dague_stat_increase(volatile uint64_t *current, volatile uint64_t *max, uint64_t value)
{
    uint64_t ov, nv;
    do {
        ov = *current;
        nv = ov + value;
        /* Conservative assumption: we take the max there. This max could never be in current
           (if another thread is  decreasing in parallel), but it's closer to the potential max */
        if( nv > *max ) {
            *max = nv;
        }
    } while( ! dague_atomic_cas_64b( current, ov, nv ) );
}

#define DAGUE_STAT_DECREASE(name, value)                      \
    __dague_stat_decrease(&dague_stats_##name##_current,    \
                            value)

static inline void __dague_stat_decrease(volatile uint64_t *current, uint64_t value)
{
    uint64_t ov, nv;
    do {
        ov = *current;
        nv = ov - value;
    } while( ! dague_atomic_cas_64b( current, ov, nv ) );
}

#define DAGUE_STATMAX_UPDATE(name, value) \
    __dague_statmax_update(&dague_stats_##name##_max, value)

static inline void __dague_statmax_update(volatile uint64_t *max, uint64_t value)
{
    uint64_t cv, nv;
    do {
        cv = *max;
        nv = cv < value ? value : cv;
    } while( !dague_atomic_cas_64b( max, cv, nv ) );
}

#define DAGUE_STATACC_ACCUMULATE(name, value) \
    __dague_statacc_accumulate(&dague_stats_##name##_acc, value)

static inline void __dague_statacc_accumulate(volatile uint64_t *acc, uint64_t value)
{
    uint64_t cv, nv;
    do {
        cv = *acc;
        nv = cv + value;
    } while( !dague_atomic_cas_64b( acc, cv, nv ) );
}

void dague_stats_dump(char *filename, char *prefix);

#define STAT_MALLOC_OVERHEAD (2*sizeof(void*))

#endif /* end of C-magic */

#else /* not defined DAGUE_STATS */

#define DECLARE_STAT(name)
#define DECLARE_STATMAX(name)
#define DECLARE_STATACC(name)
#define DAGUE_STAT_INCREASE(name, value)
#define DAGUE_STAT_DECREASE(name, value)
#define DAGUE_STATMAX_UPDATE(n, v)
#define DAGUE_STATACC_ACCUMULATE(n, v)
#define DAGUE_stats_dump(f, p)

#endif /* DAGUE_STATS */


#endif /* _statsinternal_h */
