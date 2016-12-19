/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <parsec_config.h>

#include <stdio.h>
#include <string.h>

#include "parsec/class/list.h"
#include <parsec/utils/mca_param_internal.h>
#include <parsec/utils/keyval_parse.h>

static char * file_being_read;

static void save_value(const char *name, const char *value)
{
    parsec_mca_param_file_value_t *fv;
    int found_elem = 0;

    /* First traverse through the list and ensure that we don't
       already have a param of this name.  If we do, just replace the
       value. */
    PARSEC_LIST_ITERATOR(&parsec_mca_param_file_values, item,
        {
            fv = (parsec_mca_param_file_value_t *) item;
            if (0 == strcmp(name, fv->mbpfv_param)) {
                if (NULL != fv->mbpfv_value ) {
                    free(fv->mbpfv_value);
                }
                if (NULL != value) {
                    fv->mbpfv_value = strdup(value);
                } else {
                    fv->mbpfv_value = NULL;
                }
                fv->mbpfv_file = strdup(file_being_read);
                found_elem = 1;
                break;  /* do not return, we hold the lock */
            }
        });
    if( found_elem ) return;

    /* We didn't already have the param, so append it to the list */
    fv = OBJ_NEW(parsec_mca_param_file_value_t);
    if (NULL != fv) {
        fv->mbpfv_param = strdup(name);
        if (NULL != value) {
            fv->mbpfv_value = strdup(value);
        } else {
            fv->mbpfv_value = NULL;
        }
        fv->mbpfv_file = strdup(file_being_read);
        parsec_list_append(&parsec_mca_param_file_values, (parsec_list_item_t*) fv);
    }
}

int parsec_mca_parse_paramfile(const char *paramfile)
{
    file_being_read = (char*)paramfile;
    return parsec_util_keyval_parse(paramfile, save_value);
}
