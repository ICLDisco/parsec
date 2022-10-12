/*
 * Copyright (c) 2021-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */


/**
 * @file
 *
 * User-Triggered termination detection: one task (and one task only)
 *  of the monitored taskpool will call set_nb_tasks to set it to 0;
 *  this denotes the termination of the entire taskpool. Pending actions
 *  are still internally counted, but that's the only counting going
 *  on.
 *
 */

#ifndef MCA_TERMDET_USER_TRIGGER_H
#define MCA_TERMDET_USER_TRIGGER_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/termdet/termdet.h"
#include "parsec/parsec_comm_engine.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_termdet_base_component_t parsec_termdet_user_trigger_component;
PARSEC_DECLSPEC extern const parsec_termdet_module_t parsec_termdet_user_trigger_module;
/* static accessor */
mca_base_component_t *termdet_user_trigger_static_component(void);

int parsec_termdet_user_trigger_msg_dispatch(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,  void *msg,
                                             size_t size, int src,  void *module);

typedef struct {
    uint32_t tp_id;
    int32_t  root;
} parsec_termdet_user_trigger_msg_t;

#define PARSEC_TERMDET_USER_TRIGGER_MAX_MSG_SIZE (sizeof(parsec_termdet_user_trigger_msg_t))

typedef struct {
    parsec_list_item_t list_item;
    unsigned char msg[PARSEC_TERMDET_USER_TRIGGER_MAX_MSG_SIZE];
    parsec_comm_engine_t *ce;
    void *module;
    long unsigned int tag;
    long unsigned int size;
    int src;
} parsec_termdet_user_trigger_delayed_msg_t;

extern parsec_list_t parsec_termdet_user_trigger_delayed_messages;

END_C_DECLS
#endif /* MCA_TERMDET_USER_TRIGGER_H */

