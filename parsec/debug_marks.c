/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec_config.h"

#if defined(PARSEC_DEBUG_HISTORY)
#include "parsec/debug_marks.h"
#include "parsec/debug.h"
#include "parsec_internal.h"
#include "parsec/remote_dep.h"

void debug_mark_exe(int th, int vp, const struct parsec_execution_context_s *ctx)
{
    int j, pos = 0, len = 512;
    char msg[512];

    pos += snprintf(msg+pos, len-pos, "%s(", ctx->function->name);
    for(j = 0; j < ctx->function->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s",
                        j, ctx->function->locals[j]->name, ctx->locals[j].value,
                        (j == ctx->function->nb_parameters-1) ? ")\n" : ", ");
    }

    parsec_debug_history_add("Mark: execution on thread %d of VP %d:\t%s",
                            th, vp, msg);
}

void debug_mark_ctl_msg_activate_sent(int to, const void *b, const struct remote_dep_wire_activate_s *m)
{
    int j, pos = 0, len = 512;
    char msg[512];
    parsec_handle_t *object;
    const parsec_function_t *f;

    pos += snprintf(msg+pos, len-pos, "Mark: emission of an activate message to %d\n", to);
    pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emision\n", b);
    object = parsec_handle_lookup( m->handle_id );
    f = object->functions_array[m->function_id];
    pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
    for(j = 0; j < f->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s",
                        j,
                        f->locals[j]->name, m->locals[j].value,
                        (j == f->nb_parameters - 1) ? ")\n" : ", ");
    }
    pos += snprintf(msg+pos, len-pos, "\toutput_mask = 0x%08x\n",
                    (uint32_t)m->output_mask);

    /* Do not use set_my_mark: msg is a stack-allocated buffer */
    parsec_debug_history_add("%s", msg);
}

void debug_mark_ctl_msg_activate_recv(int from, const void *b, const struct remote_dep_wire_activate_s *m)
{
    int j, pos = 0, len = 512;
    char msg[512];
    parsec_handle_t *object;
    const parsec_function_t *f;

    pos += snprintf(msg+pos, len-pos, "Mark: reception of an activate message from %d\n", from);
    pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n", b);
    object = parsec_handle_lookup( m->handle_id );
    f = object->functions_array[m->function_id];
    pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
    for(j = 0; j < f->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s",
                        j,
                        f->locals[j]->name, m->locals[j].value,
                        (j == f->nb_parameters - 1) ? ")\n" : ", ");
    }
    pos += snprintf(msg+pos, len-pos, "\toutput_mask = 0x%08x\n",
                    (uint32_t)m->output_mask);
    pos += snprintf(msg+pos, len-pos, "\t      deps = 0x%X\n",
                    (uint32_t)m->deps);

    /* Do not use set_my_mark: msg is a stack-allocated buffer */
    parsec_debug_history_add("%s", msg);
}

void debug_mark_ctl_msg_get_sent(int to, const void *b, const struct remote_dep_wire_get_s *m)
{
    parsec_debug_history_add("Mark: emission of a Get control message to %d\n"
                            "\t      Using buffer %p for emission\n"
                            "\t      deps requested = 0x%X\n"
                            "\t      which requested = 0x%08x\n"
                            "\t      tag for the reception of data = %d\n",
                            to, b, m->deps, (uint32_t)m->output_mask, m->tag);
}

void debug_mark_ctl_msg_get_recv(int from, const void *b, const struct remote_dep_wire_get_s *m)
{
    parsec_debug_history_add("Mark: reception of a Get control message from %d\n"
                            "\t      Using buffer %p for reception\n"
                            "\t      deps requested = 0x%X\n"
                            "\t      which requested = 0x%08x\n"
                            "\t      tag for the reception of data = %d\n",
                            from, b, m->deps, (uint32_t)m->output_mask, m->tag);
}

void debug_mark_dta_msg_start_send(int to, const void *b, int tag)
{
    parsec_debug_history_add("Mark: Start emitting data to %d\n"
                            "\t      Using buffer %p for emission\n"
                            "\t      tag for the emission of data = %d\n",
                            to, b, tag);
}

void debug_mark_dta_msg_end_send(int tag)
{
    parsec_debug_history_add("Mark: Done sending data of tag %d\n", tag);
}

void debug_mark_dta_msg_start_recv(int from, const void *b, int tag)
{
    parsec_debug_history_add("Mark: Start receiving data from %d\n"
                            "\t      Using buffer %p for reception\n"
                            "\t      tag for the reception of data = %d\n",
                            from, b, tag);
}

void debug_mark_dta_msg_end_recv(int tag)
{
    parsec_debug_history_add("Mark: Done receiving data with tag %d\n", tag);
}

#endif /* defined(PARSEC_DEBUG_HISTORY) */
