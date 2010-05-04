/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_H_HAS_BEEN_INCLUDED
#define DEBUG_H_HAS_BEEN_INCLUDED

#include "DAGuE_config.h"

#if defined(HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* define(HAVE_STDARG_H) */
#if !defined(HAVE_ASPRINTF)
int asprintf(char **ret, const char *format, ...);
#endif  /* !defined(HAVE_ASPRINTF) */
#if !defined(HAVE_VASPRINTF)
int vasprintf(char **ret, const char *format, va_list ap);
#endif  /* !defined(HAVE_VASPRINTF) */

#ifdef DPLASMA_DEBUG

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#   ifdef USE_MPI
/* only one printf to avoid line breaks in the middle */

static inline char* arprintf(const char* fmt, ...)
{
    char* txt;
    va_list args;
    
    va_start(args, fmt);
    vasprintf(&txt, fmt, args);
    va_end(args);
    return txt;
}

#include <mpi.h>

#define DEBUG(ARG)  do { \
    int __debug_rank; \
    char* __debug_str; \
    MPI_Comm_rank(MPI_COMM_WORLD, &__debug_rank); \
    __debug_str = arprintf ARG ; \
    fprintf(stderr, "[%d]\t%s", __debug_rank, __debug_str); \
    free(__debug_str); \
} while(0)

#   else /* USE_MPI */

#define DEBUG(ARG) printf ARG

#   endif /* USE_MPI */

#else /* DPLASMA_DEBUG */

#define DEBUG(ARG)

#endif /* DPLASMA_DEBUG */


#ifdef DPLASMA_DEBUG_HISTORY

struct DAGuE_execution_context_t;
void debug_mark_exe(int core, const struct DAGuE_execution_context_t *ctx);
#define DEBUG_MARK_EXE(core, ctx) debug_mark_exe(core, ctx)

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

#else /* DPLASMA_DEBUG_HISTORY */

#define DEBUG_MARK_EXE(core, ctx)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_RECV(from, buffer, message)
#define DEBUG_MARK_DTA_MSG_START_SEND(to, buffer, tag) 
#define DEBUG_MARK_DTA_MSG_START_RECV(from, buffer, tag)
#define DEBUG_MARK_DTA_MSG_END_SEND(tag)
#define DEBUG_MARK_DTA_MSG_END_RECV(tag)

#endif /* DPLASMA_DEBUG_HISTORY */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */

