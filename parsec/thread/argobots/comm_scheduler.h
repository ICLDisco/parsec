/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"

typedef struct {
    parsec_context_t* context;
    int stop_cb_id;
    int add_cb_id;
} cb_data_t;

void comm_sched_init(parsec_context_t* context, parsec_thread_t* comm_thread);
void comm_sched_free(parsec_thread_t* comm_thread);
void comm_sched_register_callbacks(parsec_context_t* context);
void comm_sched_unregister_callbacks(cb_data_t* data);
void comm_check_status(void* arg);
