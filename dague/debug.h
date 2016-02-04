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
 *   default output for errors, warnings and inform is 0 (stderr)
 *   warning is verbosity 1, inform is verbosity 2
 * 
 * Debug macros decorate the output, but the calling module has to 
 * select the output stream and verbosity.
 *   The dague_debug_output can be used for misc outputs.
 *
 * Guide for setting the debug verbosity:
 *   3-4: debug information (module initialized, available features etc).
 *   5-9: light debug output
 *   >=dague_debug_colorize: heavy debug output
 *
 * Debug history compiled in as soon as defined(DAGUE_DEBUG_HISTORY)
 *   independent of DAGUE_DEBUG_VERBOSE setting
 *   debug history verbosity follows dague_debug_history_verbose
 */
extern int dague_debug_output;
extern int dague_debug_verbose;
extern int dague_debug_colorize;
extern int dague_debug_rank;
extern char dague_debug_hostname[];

void dague_debug_init(void);
void dague_debug_fini(void);

void dague_debug_backtrace_save(void);
void dague_debug_backtrace_dump(void);

#if defined(DAGUE_DEBUG_HISTORY)
    extern int dague_debug_history_verbose;
    void dague_debug_history_add(const char *format, ...);
    void dague_debug_history_dump(void);
    void dague_debug_history_purge(void);
#   define _DAGUE_DEBUG_HISTORY(VERB, ...) do {                     \
        if( VERB >= dague_debug_verbose ) {                         \
            dague_debug_history_add(__VA_ARGS__);                   \
        }                                                           \
    } while(0)
#else
#   define dague_debug_history_add(...)
#   define dague_debug_history_dump()
#   define dague_debug_history_purge()
#   define _DAGUE_DEBUG_HISTORY(...)
#endif /* defined(DAGUE_DEBUG_HISTORY) */

/* Use when encountering a FATAL condition. Will terminate the program. */
#define dague_abort(FMT, ...) do {                                  \
    dague_output(0,                                                 \
        "%.*sX@%05d%.*s "FMT" %.*s@%.30s:%-5d%.*s",                 \
        dague_debug_colorize, "\x1B[1;37;41m", dague_debug_rank, dague_debug_colorize, "\033[0m", ##__VA_ARGS__,\
        dague_debug_colorize, "\x1B[36m", __func__, __LINE__, dague_debug_hostname, getpid(), dague_debug_colorize, "\033[0m");\
    abort();                                                        \
} while(0)

/* Use when encountering a SERIOUS condition. The program will continue
 * but a loud warning will always be issued on the default error output
 */
#define dague_warning(FMT, ...) do {                                \
    dague_output_verbose(1, 0,                                      \
        "%.*sW@%05d%.*s "FMT,                                       \
        dague_debug_colorize, "\x1B[1;37;43", dague_debug_rank, dague_debug_colorize, "\033[0m",        \
        ##__VA_ARGS__);                                             \
} while(0)

/* Use when some INFORMATION can be usefull for the end-user. */
#define dague_inform(FMT, ...) do {                                 \
    dague_output_verbose(2, 0,                                      \
        "%.*si@%05d%.*s "FMT,                                       \
        dague_debug_colorize, "\x1B[1;37;42", dague_debug_rank, dague_debug_colorize, "\033[0m",        \
        ##__VA_ARGS__);                                             \
} while(0)

/* Light debugging output, compiled in for all levels of
 * so not to use in performance critical routines. */
#define dague_debug_verbose(LVL, OUT, FMT, ...) do {                \
    dague_output_verbose(LVL, OUT,                                  \
        "%.*sD@%05d%.*s "FMT" %.*s@%.30s:%-5d%.*s",                 \
        dague_debug_colorize, "\x1B[0;37;44m", dague_debug_rank, dague_debug_colorize, "\033[0m", ##__VA_ARGS__,\
        dague_debug_colorize, "\x1B[36m", __func__, __LINE__, dague_debug_colorize, "\033[0m");         \
    _DAGUE_DEBUG_HISTORY(LVL,                                       \
        "D@%05d "FMT" @%.20s:%-5d", dague_debug_rank, ##__VA_ARGS__,\
        __func__, __LINE__);                                        \
} while(0)

#if defined(DAGUE_DEBUG_NOISIER)
/* Increasingly heavy debugging output. Compiled out when
 * DAGUE_DEBUG_VERBOSE is not enabled.
 * The entire history is logged as soon as debug_verbose >= 3
 */
#define DAGUE_DEBUG_VERBOSE(LVL, OUT, FMT, ...) do {                \
    dague_output_verbose(LVL, OUT,                                  \
        "%.*sd@%05d%.*s "FMT" %.*s@%.30s:%-5d%.*s",                 \
        dague_debug_colorize, "\x1B[0;37;44m", dague_debug_rank, dague_debug_colorize, "\033[0m", ##__VA_ARGS__,\
        dague_debug_colorize, "\x1B[36m", __func__, __LINE__, dague_debug_colorize, "\033[0m");         \
    _DAGUE_DEBUG_HISTORY(LVL,                                       \
        "d@%05d "FMT" @%.20s:%-5d", dague_debug_rank, ##__VA_ARGS__,\
        __func__, __LINE__);                                        \
} while(0)
#else
#define DAGUE_DEBUG_VERBOSE(...) do{} while(0)
#endif /* defined(DAGUE_DEBUG_VERBOSE) */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */

