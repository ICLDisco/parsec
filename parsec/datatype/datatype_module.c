/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/datatype_module.h"
#include "parsec/utils/debug.h"

#if !defined(PARSEC_HAVE_MPI)
extern const parsec_datatype_module_t parsec_datatype_basic_module;
static const parsec_datatype_module_t *parsec_datatype_selected_module = &parsec_datatype_basic_module;
#else
/*
 * MPI-enabled builds start without a datatype backend on purpose: the selected
 * communication component owns the datatype representation and installs the
 * matching module during parsec_comm_engine_init().
 */
static const parsec_datatype_module_t *parsec_datatype_selected_module = NULL;
#endif

static int
parsec_datatype_module_ready(void)
{
    if( NULL != parsec_datatype_selected_module ) {
        return 1;
    }

    parsec_warning("No datatype backend has been installed");
    return 0;
}

int
parsec_datatype_module_install(const parsec_datatype_module_t *module)
{
    if( NULL == module ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (NULL == module->size) ||
        (NULL == module->extent) ||
        (NULL == module->free) ||
        (NULL == module->create_contiguous) ||
        (NULL == module->create_vector) ||
        (NULL == module->create_hvector) ||
        (NULL == module->create_indexed) ||
        (NULL == module->create_indexed_block) ||
        (NULL == module->create_struct) ||
        (NULL == module->create_resized) ||
        (NULL == module->contiguous) ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    parsec_datatype_selected_module = module;
    return PARSEC_SUCCESS;
}

int
parsec_type_size(parsec_datatype_t type, int *size)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->size(type, size);
}

int
parsec_type_extent(parsec_datatype_t type, ptrdiff_t *lb, ptrdiff_t *extent)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->extent(type, lb, extent);
}

int
parsec_type_free(parsec_datatype_t *type)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->free(type);
}

int
parsec_type_create_contiguous(int count,
                              parsec_datatype_t oldtype,
                              parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_contiguous(count, oldtype, newtype);
}

int
parsec_type_create_vector(int count,
                          int blocklength,
                          int stride,
                          parsec_datatype_t oldtype,
                          parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_vector(count, blocklength, stride,
                                                          oldtype, newtype);
}

int
parsec_type_create_hvector(int count,
                           int blocklength,
                           ptrdiff_t stride,
                           parsec_datatype_t oldtype,
                           parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_hvector(count, blocklength, stride,
                                                           oldtype, newtype);
}

int
parsec_type_create_indexed(int count,
                           const int array_of_blocklengths[],
                           const int array_of_displacements[],
                           parsec_datatype_t oldtype,
                           parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_indexed(count, array_of_blocklengths,
                                                           array_of_displacements,
                                                           oldtype, newtype);
}

int
parsec_type_create_indexed_block(int count,
                                 int blocklength,
                                 const int array_of_displacements[],
                                 parsec_datatype_t oldtype,
                                 parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_indexed_block(count, blocklength,
                                                                 array_of_displacements,
                                                                 oldtype, newtype);
}

int
parsec_type_create_struct(int count,
                          const int array_of_blocklengths[],
                          const ptrdiff_t array_of_displacements[],
                          const parsec_datatype_t array_of_types[],
                          parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_struct(count, array_of_blocklengths,
                                                          array_of_displacements,
                                                          array_of_types, newtype);
}

int
parsec_type_create_resized(parsec_datatype_t oldtype,
                           ptrdiff_t lb,
                           ptrdiff_t extent,
                           parsec_datatype_t *newtype)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->create_resized(oldtype, lb, extent, newtype);
}

int
parsec_type_match(parsec_datatype_t dtt1, parsec_datatype_t dtt2)
{
#if defined(PARSEC_HAVE_MPI)
    return (dtt1 == dtt2 ? PARSEC_SUCCESS : PARSEC_ERROR);
#else
    (void)dtt1; (void)dtt2;
    return PARSEC_SUCCESS;
#endif
}

int
parsec_type_contiguous(parsec_datatype_t dtt)
{
    if( !parsec_datatype_module_ready() ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    return parsec_datatype_selected_module->contiguous(dtt);
}
