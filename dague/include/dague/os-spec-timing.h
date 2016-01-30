/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _os_spec_timing_h
#define _os_spec_timing_h

#include <stdint.h>

/** TIMING SYSTEM-SPECIFICS **/

#if defined(DAGUE_HAVE_CLOCK_GETTIME)
#include <unistd.h>
#include <time.h>
typedef struct timespec dague_time_t;
static inline dague_time_t take_time(void)
{
    dague_time_t ret;
    clock_gettime(CLOCK_REALTIME, &ret);
    return ret;
}

#define TIMER_UNIT "nanosecond"
static inline uint64_t diff_time( dague_time_t start, dague_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000000 +
           (end.tv_nsec - start.tv_nsec);
    return diff;
}

static inline int time_less( dague_time_t start, dague_time_t end )
{
    return (start.tv_sec < end.tv_sec) ||
        ((start.tv_sec == end.tv_sec) &&
         (start.tv_nsec < end.tv_nsec));
}
#define ZERO_TIME {0,0}
#elif defined(__IA64)
typedef uint64_t dague_time_t;
static inline dague_time_t take_time(void)
{
    dague_time_t ret;
    __asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
    return ret;
}
#define TIMER_UNIT "cycles"
static inline uint64_t diff_time( dague_time_t start, dague_time_t end )
{
    return (end - start);
}
static inline int time_less( dague_time_t start, dague_time_t end )
{
    return start < end;
}
#define ZERO_TIME 0
#elif defined(__X86)
typedef uint64_t dague_time_t;
static inline dague_time_t take_time(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#define TIMER_UNIT "cycles"
static inline uint64_t diff_time( dague_time_t start, dague_time_t end )
{
    return (end - start);
}
static inline int time_less( dague_time_t start, dague_time_t end )
{
    return start < end;
}
#define ZERO_TIME 0
#elif defined(__bgp__)
#include <bpcore/ppc450_inlines.h>
typedef uint64_t dague_time_t;
static inline dague_time_t take_time(void)
{
    return _bgp_GetTimeBase();
}
#define TIMER_UNIT "cycles"
static inline uint64_t diff_time( dague_time_t start, dague_time_t end )
{
    return (end - start);
}
static inline int time_less( dague_time_t start, dague_time_t end )
{
    return start < end;
}
#define ZERO_TIME 0
#else
#include <sys/time.h>
typedef struct timeval dague_time_t;
static inline dague_time_t take_time(void)
{
    struct timeval tv;

    gettimeofday( &tv, NULL );
    return tv;
}
#define TIMER_UNIT "microseconds"
static inline uint64_t diff_time( dague_time_t start, dague_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000 +
           (end.tv_usec - start.tv_usec);
    return diff;
}
static inline int time_less( dague_time_t start, dague_time_t end )
{
    return (start.tv_sec < end.tv_sec) ||
        ((start.tv_sec == end.tv_sec) &&
         (start.tv_usec < end.tv_usec));
}
#define ZERO_TIME {0,0}
#endif

#endif /* _os_spec_timing_h */
