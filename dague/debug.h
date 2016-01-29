/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_H_HAS_BEEN_INCLUDED
#define DEBUG_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague/utils/output.h"

#include <stdlib.h>
#include <stdio.h>

/**
 * Control debug output and verbosity
 *   default output is 0 (stderr)
 *   DEBUG is compiled out if !defined(DAGUE_DEBUG_ENABLE)
 *   DEBUG2 and DEBUG3 are compiled out if DAGUE_DEBUG_VERBOSE=0
 *   default runtime debug verbosity is 0 (silent)
 *   debug history compiled in as soon as defined(DAGUE_DEBUG_HISTORY)
 *      independent of DAGUE_DEBUG_VERBOSE setting
 *      debug history verbosity follows dague_debug_verbose setting too
 */
extern int dague_debug_output;
extern int dague_debug_verbose;
extern int dague_debug_rank;
extern char dague_debug_hostname[];

void dague_debug_init(void);
void dague_debug_fini(void);

void dague_debug_backtrace_save(void);
void dague_debug_backtrace_dump(void);

#if defined(DAGUE_DEBUG_HISTORY) && defined(DAGUE_DEBUG_ENABLE)
    void dague_debug_history_add(const char *format, ...);
    void dague_debug_history_dump(void);
    void dague_debug_history_purge(void);
#   define _DAGUE_DEBUG_HISTORY(VERB, ...)                          \
    if( VERB >= dague_debug_verbose ) {                             \
        dague_debug_history_add(__VA_ARGS__);                       \
    }
#else
#   define dague_debug_history_add(...)
#   define dague_debug_history_dump()
#   define dague_debug_history_purge()    
#   define _DAGUE_DEBUG_HISTORY(...)
#endif

/* Use when encountering a FATAL condition. Will terminate the program. */
#define ERROR(FMT, ...) do {                                        \
    dague_output(dague_debug_output,                                \
        "X@%05d "FMT" @%s:%s:%5d %s:%5d", dague_debug_rank,         \
        ##__VA_ARGS__,                                              \
        __FILE__, __func__, __LINE__,                               \
        dague_debug_hostname, getpid());                            \
    abort();                                                        \
} while(0)

/* Use when encountering a SERIOUS condition. The program will continue
 * but a loud warning will always be issued on the default error output
 */
#define WARNING(FMT, ...) do {                                      \
    dague_output_verbose(1, dague_debug_output,                     \
        "W@%05d "FMT, dague_debug_rank, ##__VA_ARGS__);             \
} while(0)

/* Use when some INFORMATION can be usefull for the end-user. */
#define STATUS(FMT, ...) do {                                       \
    dague_output_verbose(2, dague_debug_output,                     \
        "i@%05d "FMT, dague_debug_rank, ##__VA_ARGS__);         \
} while(0)

#if defined(DAGUE_DEBUG_ENABLE)
/* Light debugging output, compiled in for all levels of
 * DAGUE_DEBUG_VERBOSE, so not to use in performance critical
 * routines. */
#define DEBUG(FMT, ...) do {                                        \
    dague_output_verbose(3, dague_debug_output,                     \
        "D@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__);                              \
    _DAGUE_DEBUG_HISTORY(3,                                         \
        "D@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__);                              \
} while(0)

/* Increasingly heavy debugging output (2,3). Compiled out when
 * DAGUE_DEBUG_VERBOSE and DAGUE_DEBUG_HISTORY are not enabled */
#define DEBUG2(FMT, ...) do {                                       \
    DAGUE_OUTPUT_VERBOSE((4, dague_debug_ouput,                     \
        "d@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__));                              \
    _DAGUE_DEBUG_HISTORY(4                                          \
        "d@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__);                              \
} while(0)

#define DEBUG3(FMT, ...) do {                                       \
    DAGUE_OUTPUT_VERBOSE((5, dague_debug_ouput,                     \
        "d@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__));                              \
    _DAGUE_DEBUG_HISTORY(5,                                         \
        "d@%05d "FMT" @%s:%s:%5d", dague_debug_rank, ##__VA_ARGS__, \
        __FILE__, __func__, __LINE__);                              \
} while(0)

#else
#define DEBUG(...)
#define DEBUG2(...)    
#define DEBUG3(...)
#endif /* defined(DAGUE_DEBUG_ENABLE) */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */

