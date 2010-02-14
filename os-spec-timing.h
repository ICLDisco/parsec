#ifndef _os_spec_timing_h
#define _os_spec_timing_h

/** TIMING SYSTEM-SPECIFICS **/

#if defined(HAVE_CLOCK_GETTIME)
#include <unistd.h>
#include <time.h>
typedef struct timespec dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    clock_gettime(CLOCK_REALTIME, &ret);
    return ret;
}

#define TIMER_UNIT "nanosecond"
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000000 +
           (end.tv_nsec - start.tv_nsec);
    return diff;
}

static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start.tv_sec < end.tv_sec ||
        (start.tv_sec == end.tv_sec &&
         start.tv_nsec < end.tv_nsec);
}
#define ZERO_TIME {0,0}
#elif defined(__IA64)
typedef uint64_t dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    __asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
    return ret;
}
#define TIMER_UNIT "cycles"
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    return (end - start);
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start < end;
}
#define ZERO_TIME 0
#elif defined(__X86)
typedef uint64_t dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    __asm__ __volatile__("rdtsc" : "=A"(ret));
    return ret;
}
#define TIMER_UNIT "cycles"
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    return (end - start);
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start < end;
}
#define ZERO_TIME 0
#else
#include <sys/time.h>
typedef struct timeval dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    struct timeval tv;

    gettimeofday( &tv, NULL );
    return tv;
}
#define TIMER_UNIT "microseconds"
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000 +
           (end.tv_usec - start.tv_usec);
    return diff;
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start.tv_sec < end.tv_sec ||
        (start.tv_sec == end.tv_sec &&
         start.tv_usec < end.tv_usec);
}
#define ZERO_TIME {0,0}
#endif

#endif /* _os_spec_timing_h */
