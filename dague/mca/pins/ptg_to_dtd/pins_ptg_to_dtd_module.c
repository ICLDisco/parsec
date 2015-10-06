/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "pins_ptg_to_dtd.h"
#include "dague/profiling.h"
#include "dague/utils/mca_param.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

#include <stdio.h>

/* init functions */
static void pins_handle_init_ptg_to_dtd(struct dague_handle_s * handle);
static void pins_handle_fini_ptg_to_dtd(struct dague_handle_s * handle);


const dague_pins_module_t dague_pins_ptg_to_dtd_module = {
    &dague_pins_ptg_to_dtd_component,
    {
        NULL,
        NULL,
        pins_handle_init_ptg_to_dtd,
        pins_handle_fini_ptg_to_dtd,
        NULL,
        NULL
    }
};

static void pins_handle_init_ptg_to_dtd(dague_handle_t * handle)
{
    /* Adding code to instrument testing insert_task interface */
    testing_ptg_to_dtd = 99;

    /* I really don't get why the number of tasks is not handled through the original handle.
     Please, put a comment to explain and justify your choice */

    __dtd_handle = dague_dtd_new(handle->context, handle->nb_local_tasks);
    dague_handle_update_nbtask(handle, 1);
    copy_chores(handle, __dtd_handle);
    /* END */
}

static void pins_handle_fini_ptg_to_dtd(dague_handle_t * handle)
{
}
