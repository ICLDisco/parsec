/*
 * Copyright (c) 2021-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  See general MCA comment on visibility.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/parsec_internal.h"

#include "parsec/mca/termdet/termdet.h"
#include "parsec/mca/termdet/user_trigger/termdet_user_trigger.h"
#include "parsec/remote_dep.h"
/*
 * Local function
 */
static int termdet_user_trigger_component_query(mca_base_module_t **module, int *priority);
static int termdet_user_trigger_component_close();

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_termdet_base_component_t parsec_termdet_user_trigger_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_TERMDET_BASE_VERSION_2_0_0,

        /* Component name, options and version */
        "user_trigger",
        "",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL, /*< No open: termdet_user_trigger is always available, no need to check at runtime */
        termdet_user_trigger_component_close,
        termdet_user_trigger_component_query,
        /*< specific query to return the module and add it to the list of available modules */
        NULL, /*< No register: no parameters to the absolute priority component */
        "", /*< no reserve */
    },
    {
        /* The component has no metada */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};

mca_base_component_t *termdet_user_trigger_static_component(void)
{
    return (mca_base_component_t *)&parsec_termdet_user_trigger_component;
}

/* set to 1 when the callback is registered -- workaround current MCA interface limitation */
static int parsec_termdet_user_trigger_msg_cb_registered = 0;
static int parsec_termdet_user_trigger_msg_cb_id = 0;

static void parsec_termdet_user_trigger_ce_up(parsec_comm_engine_t *ce, void *user_data)
{
    int rc = ce->tag_register(PARSEC_TERMDET_USER_TRIGGER_MSG_TAG,
                              parsec_termdet_user_trigger_msg_dispatch,
                              user_data,
                              PARSEC_TERMDET_USER_TRIGGER_MAX_MSG_SIZE);
    (void)rc;
}

static void parsec_termdet_user_trigger_ce_down(parsec_comm_engine_t *ce, void *user_data)
{
    (void)user_data;
    ce->tag_unregister(PARSEC_TERMDET_USER_TRIGGER_MSG_TAG);
}


static int termdet_user_trigger_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_termdet_user_trigger_module;
    *priority = 1;
    *module = (mca_base_module_t *)ptr;

    if( 0 == parsec_termdet_user_trigger_msg_cb_registered ) {
        assert(0 == parsec_termdet_user_trigger_msg_cb_id);
        parsec_termdet_user_trigger_msg_cb_id = 
            parsec_comm_engine_register_callback(parsec_termdet_user_trigger_ce_up, ptr,
                                                 parsec_termdet_user_trigger_ce_down, ptr);
        PARSEC_OBJ_CONSTRUCT(&parsec_termdet_user_trigger_delayed_messages, parsec_list_t);
    }
    parsec_termdet_user_trigger_msg_cb_registered++;

    return MCA_SUCCESS;
}

static int termdet_user_trigger_component_close()
{
    parsec_termdet_user_trigger_msg_cb_registered--;
    if( 0 == parsec_termdet_user_trigger_msg_cb_registered ) {
        parsec_comm_engine_unregister_callback(parsec_termdet_user_trigger_msg_cb_id);
        parsec_termdet_user_trigger_msg_cb_id = 0;
        PARSEC_OBJ_DESTRUCT(&parsec_termdet_user_trigger_delayed_messages);
    }
    return MCA_SUCCESS;
}
