/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_H_HAS_BEEN_INCLUDED
#define DEBUG_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#if defined(HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* define(HAVE_STDARG_H) */
#if !defined(HAVE_ASPRINTF)
int asprintf(char **ret, const char *format, ...);
#endif  /* !defined(HAVE_ASPRINTF) */
#if !defined(HAVE_VASPRINTF)
int vasprintf(char **ret, const char *format, va_list ap);
#endif  /* !defined(HAVE_VASPRINTF) */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void debug_save_stack_trace(void);
void debug_dump_stack_traces(void);

/* only one printf to avoid line breaks in the middle */
static inline char* arprintf(const char* fmt, ...)
{
    char* txt;
    va_list args;
    int rc;

    va_start(args, fmt);
    rc = vasprintf(&txt, fmt, args);
    va_end(args);

    (void)rc;
    return txt;
}

#if defined(DAGUE_DEBUG_HISTORY)
    void dague_debug_history_add(const char *format, ...);
#   define _DAGUE_DEBUG_HISTORY(ARG) dague_debug_history_add ARG
#else
#   define _DAGUE_DEBUG_HISTORY(ARG)
#endif

#ifdef HAVE_MPI
#   define _DAGUE_OUTPUT(PRFX, ARG) do { \
        char* __debug_str; \
        __debug_str = arprintf ARG ; \
        fprintf(stderr,  "[" PRFX "DAGuE % 5d]:\t%s", -1, __debug_str); \
        free(__debug_str); \
    } while(0)

#   define ABORT() MPI_Abort(MPI_COMM_SELF, -1)

#else /* HAVE_MPI */
#   define _DAGUE_OUTPUT(PRFX, ARG) do { \
        char* __debug_str; \
        __debug_str = arprintf ARG ; \
        fprintf(stderr, "[" PRFX "DAGuE]:\t%s", __debug_str); \
        free(__debug_str); \
    } while(0)

#   define ABORT() abort()

#endif /* HAVE_MPI */

#define STATUS(ARG) do { \
    _DAGUE_OUTPUT("..", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#define VERBOSE(ARG) do { \
    if(dague_verbose) \
        _DAGUE_OUTPUT("+.", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#define VERBOSE2(ARG) do { \
    if(dague_verbose > 1) \
        _DAGUE_OUTPUT("+^", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#define WARNING(ARG) do { \
    _DAGUE_OUTPUT("!.", ARG) ; \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)

#define ERROR(ARG) do { \
    _DAGUE_OUTPUT("X.", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
    ABORT(); \
} while(0)

#ifdef DAGUE_DEBUG_VERBOSE3
# define DEBUG3(ARG) do { \
    _DAGUE_OUTPUT("D^", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#else /*DEBUG3*/
# define DEBUG3(ARG) do { _DAGUE_DEBUG_HISTORY(ARG); } while(0)
#endif /*DEBUG3*/

#ifdef DAGUE_DEBUG_VERBOSE2
# define DEBUG2(ARG) do { \
    _DAGUE_OUTPUT("D.", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#else /*DEBUG2*/
# define DEBUG2(ARG) do { _DAGUE_DEBUG_HISTORY(ARG); } while(0)
#endif /*DEBUG2*/

#ifdef DAGUE_DEBUG_VERBOSE1
# define DEBUG(ARG) do { \
    _DAGUE_OUTPUT("d.", ARG); \
    _DAGUE_DEBUG_HISTORY(ARG); \
} while(0)
#else /*DEBUG1*/
# define DEBUG(ARG) do { _DAGUE_DEBUG_HISTORY(ARG); } while(0)
#endif /*DEBUG1*/



#ifdef DAGUE_DEBUG_HISTORY
#   ifndef DAGUE_DEBUG_VERBOSE1
#       define DAGUE_DEBUG_VERBOSE1
#   endif
#   ifndef DAGUE_DEBUG_VERBOSE2
#       define DAGUE_DEBUG_VERBOSE2
#   endif
#   ifndef DAGUE_DEBUG_VERBOSE3
#       define DAGUE_DEBUG_VERBOSE3
#   endif

struct dague_execution_context_t;
void debug_mark_exe(int th, int vp, const struct dague_execution_context_t *ctx);
#define DEBUG_MARK_EXE(th, vp, ctx) debug_mark_exe(th, vp, ctx)

struct remote_dep_wire_activate_t;
void debug_mark_ctl_msg_activate_sent(int to, const void *b, const struct remote_dep_wire_activate_t *m);
#define DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(to, buffer, message) debug_mark_ctl_msg_activate_sent(to, buffer, message)
void debug_mark_ctl_msg_activate_recv(int from, const void *b, const struct remote_dep_wire_activate_t *m);
#define DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, buffer, message) debug_mark_ctl_msg_activate_recv(from, buffer, message)

struct remote_dep_wire_get_t;
void debug_mark_ctl_msg_get_sent(int to, const void *b, const struct remote_dep_wire_get_t *m);
#define DEBUG_MARK_CTL_MSG_GET_SENT(to, buffer, message) debug_mark_ctl_msg_get_sent(to, buffer, message)
void debug_mark_ctl_msg_get_recv(int from, const void *b, const struct remote_dep_wire_get_t *m);
#define DEBUG_MARK_CTL_MSG_GET_RECV(from, buffer, message) debug_mark_ctl_msg_get_recv(from, buffer, message)

void debug_mark_dta_msg_start_send(int to, const void *b, int tag);
#define DEBUG_MARK_DTA_MSG_START_SEND(to, buffer, tag) debug_mark_dta_msg_start_send(to, buffer, tag)
void debug_mark_dta_msg_start_recv(int from, const void *b, int tag);
#define DEBUG_MARK_DTA_MSG_START_RECV(from, buffer, tag) debug_mark_dta_msg_start_recv(from, buffer, tag)
void debug_mark_dta_msg_end_send(int tag);
#define DEBUG_MARK_DTA_MSG_END_SEND(tag) debug_mark_dta_msg_end_send(tag)
void debug_mark_dta_msg_end_recv(int tag);
#define DEBUG_MARK_DTA_MSG_END_RECV(tag) debug_mark_dta_msg_end_recv(tag)

void debug_mark_display_history(void);
void debug_mark_purge_history(void);
void debug_mark_purge_all_history(void);

#else /* DAGUE_DEBUG_HISTORY */

#define DEBUG_MARK_EXE(th, vp, ctx)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_RECV(from, buffer, message)
#define DEBUG_MARK_DTA_MSG_START_SEND(to, buffer, tag) 
#define DEBUG_MARK_DTA_MSG_START_RECV(from, buffer, tag)
#define DEBUG_MARK_DTA_MSG_END_SEND(tag)
#define DEBUG_MARK_DTA_MSG_END_RECV(tag)
#define debug_mark_purge_history()
#define debug_mark_purge_all_history()

#endif /* DAGUE_DEBUG_HISTORY */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */

