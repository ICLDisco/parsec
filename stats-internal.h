#ifndef _statsinternal_h
#define _statsinternal_h

#ifndef stats_h
#error "Never include stats-internal.h directly: include stats.h"
#endif

#include "dplasma_config.h"

#if defined(DPLASMA_STATS)

#if defined(DPLASMA_STATS_C_DECLARE)

#include <stdint.h>
#define DECLARE_STAT(name)                                  \
    volatile uint64_t dplasma_stats_##name##_max = 0;       \
    volatile uint64_t dplasma_stats_##name##_current = 0

#define DECLARE_STATMAX(name)                           \
    volatile uint64_t dplasma_stats_##name##_max = 0

#elif defined(DPLASMA_STATS_C_DUMP)

#define DECLARE_STAT(name) \
    fprintf(statfile, "%s%s" #name "  MAX = %llu\n", prefix != NULL ? prefix : "", prefix != NULL ? " " : "", dplasma_stats_##name##_max)
#define DECLARE_STATMAX(name) \
    fprintf(statfile, "%s%s" #name "  MAX = %llu\n", prefix != NULL ? prefix : "", prefix != NULL ? " " : "", dplasma_stats_##name##_max)

#else /* no C-magic */

#include "atomic.h"
#include "os-spec-timing.h"

#define DECLARE_STAT(name)                                   \
    extern volatile uint64_t dplasma_stats_##name##_max;     \
    extern volatile uint64_t dplasma_stats_##name##_current

#define DECLARE_STATMAX(name)                           \
    extern volatile uint64_t dplasma_stats_##name##_max

#define DPLASMA_STAT_INCREASE(name, value)                   \
    __dplasma_stat_increase(&dplasma_stats_##name##_current, \
                            &dplasma_stats_##name##_max,     \
                            value)

static inline void __dplasma_stat_increase(volatile uint64_t *current, volatile uint64_t *max, uint64_t value)
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
    } while( ! dplasma_atomic_cas_64b( current, ov, nv ) );
}

#define DPLASMA_STAT_DECREASE(name, value)                      \
    __dplasma_stat_decrease(&dplasma_stats_##name##_current,    \
                            &dplasma_stats_##name##_max,        \
                            value)

static inline void __dplasma_stat_decrease(volatile uint64_t *current, volatile uint64_t *max, uint64_t value)
{
    uint64_t ov, nv;
    do {
        ov = *current;
        nv = ov - value;
    } while( ! dplasma_atomic_cas_64b( current, ov, nv ) );
}

#define DPLASMA_STATMAX_UPDATE(name, value) \
    __dplasma_statmax_update(&dplasma_stats_##name##_max, value)

static inline void __dplasma_statmax_update(volatile uint64_t *max, uint64_t value)
{
    uint64_t cv, nv;
    do {
        cv = *max;
        nv = cv < value ? value : cv;
    } while( !dplasma_atomic_cas_64b( max, cv, nv ) );
}

void dplasma_stats_dump(char *filename, char *prefix);

#define STAT_MALLOC_OVERHEAD (2*sizeof(void*))

#endif /* end of C-magic */

#else /* not defined DPLASMA_STATS */

#define DECLARE_STAT(name)
#define DECLARE_STATMAX(name)
#define DPLASMA_STAT_INCREASE(name, value)
#define DPLASMA_STAT_DECREASE(name, value)

#endif /* DPLASMA_STATS */


#endif /* _statsinternal_h */
