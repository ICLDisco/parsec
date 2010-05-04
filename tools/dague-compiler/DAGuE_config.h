#ifndef DAGuE_CONFIG_H_HAS_BEEN_INCLUDED
#define DAGuE_CONFIG_H_HAS_BEEN_INCLUDED

#define DAGuE_VERSION_MAJOR 0
#define DAGuE_VERSION_MINOR 1

/* Communication engine */
#define DAGuE_COLLECTIVE
#define DAGuE_MPI
/* Scheduling engine */
/* #undef HAVE_HWLOC */
/* #undef DAGuE_CACHE_AWARE */
/* debug */
/* #undef DAGuE_DEBUG */
/* #undef DAGuE_DEBUG_HISTORY */
/* profiling */
#define DAGuE_PROFILING
/* #undef DAGuE_STATS */
/* #undef DAGuE_GRAPHER */
/* #undef DAGuE_CALL_TRACE */
/* #undef DAGuE_DRY_RUN */
/* #undef HAVE_PAPI */
/* system */
#define HAVE_PTHREAD
/* #undef HAVE_SCHED_SETAFFINITY */
/* #undef HAVE_CLOCK_GETTIME */
#define HAVE_COMPARE_AND_SWAP_32
#define HAVE_COMPARE_AND_SWAP_64
#define HAVE_ASPRINTF
#define HAVE_VASPRINTF
#define HAVE_STDARG_H
/* #undef HAVE_VA_COPY */
/* #undef HAVE_UNDERSCORE_VA_COPY */
#define HAVE_GETOPT_LONG
#define HAVE_GETOPT_H
#define HAVE_ERRNO_H
#define ARCH_X86
/* #undef ARCH_X86_64 */
/* #undef ARCH_PPC */
#define MAC_OS_X

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif  /* !defined(_GNU_SOURCE) */

#ifdef ARCH_PPC
#define inline __inline__
#define restrict 
#endif

#endif  /*DAGuE_CONFIG_H_HAS_BEEN_INCLUDED */
