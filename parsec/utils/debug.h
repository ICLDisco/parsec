/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_H_HAS_BEEN_INCLUDED
#define DEBUG_H_HAS_BEEN_INCLUDED

/** @defgroup parsec_internal_debug Internal Debugging System
 *  @ingroup parsec_internal
 *    Functions and macros of this group are used internally to
 *    check internal assertions and output debugging information
 *    when requested by the user.
 *  @addtogroup parsec_internal_debug
 *  @{
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/output.h"
#include <stdlib.h>

BEGIN_C_DECLS

/**
 * Control debug output and verbosity
 *   default output for errors, warnings and inform is 0 (stderr)
 *   warning is verbosity 1, inform is verbosity 2
 *
 * Debug macros decorate the output, but the calling module has to
 * select the output stream and verbosity.
 *   The parsec_debug_output can be used for misc outputs.
 *
 * Guide for setting the debug verbosity:
 *   3-4: debug information (module initialized, available features etc).
 *   5-9: light debug output
 *   >=10: heavy debug output
 *
 * Debug history compiled in as soon as defined(PARSEC_DEBUG_HISTORY)
 *   independent of PARSEC_DEBUG_VERBOSE setting
 *   debug history verbosity follows parsec_debug_history_verbose
 */
extern int parsec_debug_output;
extern int parsec_debug_verbose;
extern int parsec_debug_colorize;
extern int parsec_debug_rank;
extern int parsec_debug_coredump_on_fatal;
extern int parsec_debug_history_on_fatal;
extern const char* parsec_hostname;

void parsec_debug_init(void);
void parsec_debug_fini(void);

void parsec_debug_backtrace_save(void);
void parsec_debug_backtrace_dump(void);

#if defined(PARSEC_DEBUG_HISTORY)
    extern int parsec_debug_history_verbose;
    void parsec_debug_history_add(const char *format, ...);
    void parsec_debug_history_dump(void);
    void parsec_debug_history_purge(void);
    void parsec_debug_history_init(void);
    void parsec_debug_history_fini(void);
#   define _PARSEC_DEBUG_HISTORY(VERB, ...) do {                     \
        if( VERB <= parsec_debug_history_verbose ) {                 \
            parsec_debug_history_add(__VA_ARGS__);                   \
        }                                                            \
    } while(0)
#else
#   define parsec_debug_history_add(...)
#   define parsec_debug_history_dump()
#   define parsec_debug_history_purge()
#   define parsec_debug_history_init()
#   define parsec_debug_history_fini()
#   define _PARSEC_DEBUG_HISTORY(...)
#endif /* defined(PARSEC_DEBUG_HISTORY) */

/* The main prototype is in runtime.h, but in order to avoid creating
 * a dependencies between the header files we include this definition
 * here.
 */
extern void (*parsec_weaksym_exit)(int status);

/* Use when encountering a FATAL condition. Will terminate the program. */
#define parsec_fatal(FMT, ...) do {                                  \
    parsec_output(0,                                                 \
        "%.*sx@%05d%.*s " FMT " %.*s@%.30s:%-5d (%.30s:%-5d)%.*s",   \
        parsec_debug_colorize, "\x1B[1;37;41m", parsec_debug_rank,   \
        parsec_debug_colorize, "\033[0m", ##__VA_ARGS__,             \
        parsec_debug_colorize, "\x1B[36m", __func__, __LINE__, parsec_hostname, getpid(), \
        parsec_debug_colorize, "\033[0m");                           \
    if ( parsec_debug_history_on_fatal ) {                           \
        parsec_debug_history_dump();                                 \
    }                                                                \
    if ( parsec_debug_coredump_on_fatal ) {                          \
        abort();                                                     \
    }                                                                \
    parsec_weaksym_exit(-6);                                         \
} while(0)

/* Use when encountering a SERIOUS condition. The program will continue
 * but a loud warning will always be issued on the default error output
 */
#define parsec_warning(FMT, ...) do {                                \
    parsec_output_verbose(0, 0,                                      \
        "%.*sW@%05d%.*s " FMT,                                       \
        parsec_debug_colorize, "\x1B[1;37;43m", parsec_debug_rank,   \
        parsec_debug_colorize, "\033[0m", ##__VA_ARGS__);            \
} while(0)

/* Use when some INFORMATION can be usefull for the end-user. */
#define parsec_inform(FMT, ...) do {                                 \
    parsec_output_verbose(0, 0,                                      \
        "%.*si@%05d%.*s " FMT,                                       \
        parsec_debug_colorize, "\x1B[1;37;42m", parsec_debug_rank,   \
        parsec_debug_colorize, "\033[0m", ##__VA_ARGS__);            \
} while(0)

#if defined(PARSEC_DEBUG_HISTORY)
#define parsec_debug_verbose(LVL, OUT, FMT, ...) do {                \
    (void)(OUT);                                                     \
    _PARSEC_DEBUG_HISTORY(LVL,                                       \
        "D@%05d " FMT " @%.20s:%-5d", parsec_debug_rank,             \
        ##__VA_ARGS__, __func__, __LINE__);                          \
} while(0)
#else
/* Light debugging output, compiled in for all levels of
 * so not to use in performance critical routines. */
#define parsec_debug_verbose(LVL, OUT, FMT, ...) do {                \
    parsec_output_verbose(LVL, OUT,                                  \
        "%.*sD@%05d%.*s " FMT " %.*s@%.30s:%-5d%.*s",                \
        parsec_debug_colorize, "\x1B[0;37;44m", parsec_debug_rank,   \
        parsec_debug_colorize, "\033[0m", ##__VA_ARGS__,             \
        parsec_debug_colorize, "\x1B[36m", __func__, __LINE__,       \
        parsec_debug_colorize, "\033[0m");                           \
} while(0)
#endif

#if defined(PARSEC_DEBUG_NOISIER)
/* Increasingly heavy debugging output. Compiled out when
 * PARSEC_DEBUG_VERBOSE is not enabled.
 */
#if defined(PARSEC_DEBUG_HISTORY)
#define PARSEC_DEBUG_VERBOSE(LVL, OUT, FMT, ...) do {                \
    _PARSEC_DEBUG_HISTORY(LVL,                                       \
        "d@%05d " FMT " @%.20s:%-5d", parsec_debug_rank,             \
        ##__VA_ARGS__, __func__, __LINE__);                          \
} while(0)
#else
#define PARSEC_DEBUG_VERBOSE(LVL, OUT, FMT, ...) do {                \
    parsec_output_verbose(LVL, OUT,                                  \
        "%.*sd@%05d%.*s " FMT " %.*s@%.30s:%-5d%.*s",                \
        parsec_debug_colorize, "\x1B[0;37;44m", parsec_debug_rank,   \
        parsec_debug_colorize, "\033[0m", ##__VA_ARGS__,             \
        parsec_debug_colorize, "\x1B[36m", __func__, __LINE__,       \
        parsec_debug_colorize, "\033[0m");                           \
} while(0)
#endif
#else
#define PARSEC_DEBUG_VERBOSE(...) do{} while(0)
#endif /* defined(PARSEC_DEBUG_VERBOSE) */

/** $brief To check if any parsec function returned error.
 */
#define PARSEC_CHECK_ERROR(rc, MSG)                             \
    if( rc < 0 ) {                                              \
        parsec_fatal( "%s", MSG );                              \
    }                                                           \

END_C_DECLS

/** @} */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */
