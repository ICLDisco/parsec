/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_MARKS_H_HAS_BEEN_INCLUDED
#define DEBUG_MARKS_H_HAS_BEEN_INCLUDED

/**  @addtogroup parsec_internal_debug
 *   @{
 */

#include "parsec/parsec_config.h"

#ifdef PARSEC_DEBUG_HISTORY

struct parsec_execution_context_s;
void debug_mark_exe(int th, int vp, const struct parsec_execution_context_s *ctx);
#define DEBUG_MARK_EXE(th, vp, ctx) debug_mark_exe(th, vp, ctx)

struct remote_dep_wire_activate_s;
void debug_mark_ctl_msg_activate_sent(int to, const void *b, const struct remote_dep_wire_activate_s *m);
#define DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(to, buffer, message) debug_mark_ctl_msg_activate_sent(to, buffer, message)
void debug_mark_ctl_msg_activate_recv(int from, const void *b, const struct remote_dep_wire_activate_s *m);
#define DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, buffer, message) debug_mark_ctl_msg_activate_recv(from, buffer, message)

struct remote_dep_wire_get_s;
void debug_mark_ctl_msg_get_sent(int to, const void *b, const struct remote_dep_wire_get_s *m);
#define DEBUG_MARK_CTL_MSG_GET_SENT(to, buffer, message) debug_mark_ctl_msg_get_sent(to, buffer, message)
void debug_mark_ctl_msg_get_recv(int from, const void *b, const struct remote_dep_wire_get_s *m);
#define DEBUG_MARK_CTL_MSG_GET_RECV(from, buffer, message) debug_mark_ctl_msg_get_recv(from, buffer, message)

void debug_mark_dta_msg_start_send(int to, const void *b, int tag);
#define DEBUG_MARK_DTA_MSG_START_SEND(to, buffer, tag) debug_mark_dta_msg_start_send(to, buffer, tag)
void debug_mark_dta_msg_start_recv(int from, const void *b, int tag);
#define DEBUG_MARK_DTA_MSG_START_RECV(from, buffer, tag) debug_mark_dta_msg_start_recv(from, buffer, tag)
void debug_mark_dta_msg_end_send(int tag);
#define DEBUG_MARK_DTA_MSG_END_SEND(tag) debug_mark_dta_msg_end_send(tag)
void debug_mark_dta_msg_end_recv(int tag);
#define DEBUG_MARK_DTA_MSG_END_RECV(tag) debug_mark_dta_msg_end_recv(tag)


#else /* PARSEC_DEBUG_HISTORY */

#define DEBUG_MARK_EXE(th, vp, ctx)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_SENT(to, buffer, message)
#define DEBUG_MARK_CTL_MSG_GET_RECV(from, buffer, message)
#define DEBUG_MARK_DTA_MSG_START_SEND(to, buffer, tag)
#define DEBUG_MARK_DTA_MSG_START_RECV(from, buffer, tag)
#define DEBUG_MARK_DTA_MSG_END_SEND(tag)
#define DEBUG_MARK_DTA_MSG_END_RECV(tag)

#endif /* PARSEC_DEBUG_HISTORY */

/** @} */

#endif /* DEBUG_MARKS_H_HAS_BEEN_INCLUDED */
