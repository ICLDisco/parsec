/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2007 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#include "parsec/parsec_config.h"

#include "parsec/class/parsec_value_array.h"

static void parsec_value_array_construct(parsec_value_array_t* array)
{
    array->array_items = NULL;
    array->array_size = 0;
    array->array_item_sizeof = 0;
    array->array_alloc_size = 0;
}

static void parsec_value_array_destruct(parsec_value_array_t* array)
{
    if (NULL != array->array_items)
        free(array->array_items);
}

PARSEC_OBJ_CLASS_INSTANCE(
    parsec_value_array_t,
    parsec_object_t,
    parsec_value_array_construct,
    parsec_value_array_destruct
);


int parsec_value_array_set_size(parsec_value_array_t* array, size_t size)
{
    if(array->array_item_sizeof == 0) {
        return PARSEC_ERR_BAD_PARAM;
    }

    if(size > array->array_alloc_size) {
        while(array->array_alloc_size < size)
            array->array_alloc_size <<= 1;
        array->array_items = (unsigned char *)realloc(array->array_items,
            array->array_alloc_size * array->array_item_sizeof);
        if (NULL == array->array_items)
            return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    array->array_size = size;
    return PARSEC_SUCCESS;
}

