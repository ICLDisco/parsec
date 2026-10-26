/*
 * Copyright (c) 2015-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include "parsec/runtime.h"
#include "parsec/datatype_module.h"
#include <stdlib.h>

/**
 * Minimal datatype backend used when no communication component provides a
 * richer datatype implementation.  It recognizes PaRSEC's predefined scalar
 * datatypes and records the size and extent of simple derived datatypes.  This
 * is enough for communication backends that only need contiguous byte ranges,
 * while still failing through the public API if a caller asks for information
 * about an unknown datatype.
 */

typedef struct parsec_datatype_basic_desc_s {
    uint64_t magic;
    int size;
    ptrdiff_t lb;
    ptrdiff_t extent;
    int contiguous;
} parsec_datatype_basic_desc_t;

#define PARSEC_DATATYPE_BASIC_MAGIC 0x7061727365636474ULL

static int
parsec_datatype_basic_is_predefined(parsec_datatype_t type)
{
    return (type >= parsec_datatype_int_t) &&
           (type <= parsec_datatype_double_complex_t);
}

static parsec_datatype_basic_desc_t *
parsec_datatype_basic_get_desc(parsec_datatype_t type)
{
    parsec_datatype_basic_desc_t *desc;
    uintptr_t handle = (uintptr_t)type;

    if( parsec_datatype_basic_is_predefined(type) ||
        (PARSEC_DATATYPE_NULL == type) ) {
        return NULL;
    }
    if( handle < 4096 || 0 != (handle % sizeof(void *)) ) {
        return NULL;
    }
    desc = (parsec_datatype_basic_desc_t *)(intptr_t)type;
    if( PARSEC_DATATYPE_BASIC_MAGIC != desc->magic ) {
        return NULL;
    }
    return desc;
}

static int
parsec_datatype_basic_get_info(parsec_datatype_t type,
                               int *size,
                               ptrdiff_t *lb,
                               ptrdiff_t *extent,
                               int *contiguous)
{
    parsec_datatype_basic_desc_t *desc;
    int predefined_size;

    switch( type ) {
    case parsec_datatype_int_t:
        predefined_size = sizeof( int );
        break;
    case parsec_datatype_int8_t:
        predefined_size = sizeof( int8_t );
        break;
    case parsec_datatype_int16_t:
        predefined_size = sizeof( int16_t );
        break;
    case parsec_datatype_int32_t:
        predefined_size = sizeof( int32_t );
        break;
    case parsec_datatype_int64_t:
        predefined_size = sizeof( int64_t );
        break;
    case parsec_datatype_uint8_t:
        predefined_size = sizeof( uint8_t );
        break;
    case parsec_datatype_uint16_t:
        predefined_size = sizeof( uint16_t );
        break;
    case parsec_datatype_uint32_t:
        predefined_size = sizeof( uint32_t );
        break;
    case parsec_datatype_uint64_t:
        predefined_size = sizeof( uint64_t );
        break;
    case parsec_datatype_float_t:
        predefined_size = sizeof( float );
        break;
    case parsec_datatype_double_t:
        predefined_size = sizeof( double );
        break;
    case parsec_datatype_long_double_t:
        predefined_size = sizeof( long double );
        break;
    case parsec_datatype_complex_t:
        predefined_size = 2 * sizeof( float );
        break;
    case parsec_datatype_double_complex_t:
        predefined_size = 2 * sizeof( double );
        break;
    default:
        desc = parsec_datatype_basic_get_desc(type);
        if( NULL == desc ) {
            return PARSEC_ERR_NOT_SUPPORTED;
        }
        if( NULL != size ) *size = desc->size;
        if( NULL != lb ) *lb = desc->lb;
        if( NULL != extent ) *extent = desc->extent;
        if( NULL != contiguous ) *contiguous = desc->contiguous;
        return PARSEC_SUCCESS;
    }

    if( NULL != size ) *size = predefined_size;
    if( NULL != lb ) *lb = 0;
    if( NULL != extent ) *extent = predefined_size;
    if( NULL != contiguous ) *contiguous = 1;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_desc_create(int size,
                                  ptrdiff_t lb,
                                  ptrdiff_t extent,
                                  int contiguous,
                                  parsec_datatype_t *newtype)
{
    parsec_datatype_basic_desc_t *desc;

    desc = (parsec_datatype_basic_desc_t *)calloc(1, sizeof(*desc));
    if( NULL == desc ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    desc->magic = PARSEC_DATATYPE_BASIC_MAGIC;
    desc->size = size;
    desc->lb = lb;
    desc->extent = extent;
    desc->contiguous = contiguous;
    *newtype = (parsec_datatype_t)(intptr_t)desc;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_size(parsec_datatype_t type, int *size)
{
    return parsec_datatype_basic_get_info(type, size, NULL, NULL, NULL);
}

static int
parsec_datatype_basic_extent(parsec_datatype_t type, ptrdiff_t *lb, ptrdiff_t *extent)
{
    return parsec_datatype_basic_get_info(type, NULL, lb, extent, NULL);
}

static int
parsec_datatype_basic_free(parsec_datatype_t *type)
{
    parsec_datatype_basic_desc_t *desc;

    desc = parsec_datatype_basic_get_desc(*type);
    if( NULL != desc ) {
        desc->magic = 0;
        free(desc);
    }
    *type = PARSEC_DATATYPE_NULL;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_contiguous(int count,
                                        parsec_datatype_t oldtype,
                                        parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous;
    ptrdiff_t oldlb, oldextent;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, &oldlb,
                                        &oldextent, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    return parsec_datatype_basic_desc_create(count * oldsize, oldlb,
                                             count * oldextent,
                                             contiguous, newtype);
}

static int
parsec_datatype_basic_create_vector(int count,
                                    int blocklength,
                                    int stride,
                                    parsec_datatype_t oldtype,
                                    parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous;
    ptrdiff_t oldlb, oldextent, extent;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, &oldlb,
                                        &oldextent, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    extent = ((ptrdiff_t)(count - 1) * stride + blocklength) * oldextent;
    return parsec_datatype_basic_desc_create(count * blocklength * oldsize,
                                             oldlb, extent,
                                             contiguous && (extent == (count * blocklength * oldsize)),
                                             newtype);
}

static int
parsec_datatype_basic_create_hvector(int count,
                                     int blocklength,
                                     ptrdiff_t stride,
                                     parsec_datatype_t oldtype,
                                     parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous;
    ptrdiff_t oldlb, oldextent, extent;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, &oldlb,
                                        &oldextent, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    extent = ((ptrdiff_t)(count - 1) * stride) + blocklength * oldextent;
    return parsec_datatype_basic_desc_create(count * blocklength * oldsize,
                                             oldlb, extent,
                                             contiguous && (extent == (count * blocklength * oldsize)),
                                             newtype);
}

static int
parsec_datatype_basic_create_indexed(int count,
                                     const int array_of_blocklengths[],
                                     const int array_of_displacements[],
                                     parsec_datatype_t oldtype,
                                     parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous, size = 0;
    ptrdiff_t oldlb, oldextent, min_disp = 0, max_disp = 0;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, &oldlb,
                                        &oldextent, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    for(int i = 0; i < count; i++) {
        ptrdiff_t begin = (ptrdiff_t)array_of_displacements[i] * oldextent;
        ptrdiff_t end = begin + array_of_blocklengths[i] * oldextent;
        if( (0 == i) || (begin < min_disp) ) min_disp = begin;
        if( (0 == i) || (end > max_disp) ) max_disp = end;
        size += array_of_blocklengths[i] * oldsize;
    }
    return parsec_datatype_basic_desc_create(size, oldlb + min_disp,
                                             max_disp - min_disp, 0, newtype);
}

static int
parsec_datatype_basic_create_indexed_block(int count,
                                           int blocklength,
                                           const int array_of_displacements[],
                                           parsec_datatype_t oldtype,
                                           parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous, size;
    ptrdiff_t oldlb, oldextent, min_disp = 0, max_disp = 0;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, &oldlb,
                                        &oldextent, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    for(int i = 0; i < count; i++) {
        ptrdiff_t begin = (ptrdiff_t)array_of_displacements[i] * oldextent;
        ptrdiff_t end = begin + blocklength * oldextent;
        if( (0 == i) || (begin < min_disp) ) min_disp = begin;
        if( (0 == i) || (end > max_disp) ) max_disp = end;
    }
    size = count * blocklength * oldsize;
    return parsec_datatype_basic_desc_create(size, oldlb + min_disp,
                                             max_disp - min_disp, 0, newtype);
}

static int
parsec_datatype_basic_create_struct(int count,
                                    const int array_of_blocklengths[],
                                    const ptrdiff_t array_of_displacements[],
                                    const parsec_datatype_t array_of_types[],
                                    parsec_datatype_t *newtype)
{
    int rc, oldsize, size = 0;
    ptrdiff_t oldlb, oldextent, min_disp = 0, max_disp = 0;

    for(int i = 0; i < count; i++) {
        rc = parsec_datatype_basic_get_info(array_of_types[i], &oldsize,
                                            &oldlb, &oldextent, NULL);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        ptrdiff_t begin = array_of_displacements[i] + oldlb;
        ptrdiff_t end = begin + array_of_blocklengths[i] * oldextent;
        if( (0 == i) || (begin < min_disp) ) min_disp = begin;
        if( (0 == i) || (end > max_disp) ) max_disp = end;
        size += array_of_blocklengths[i] * oldsize;
    }
    return parsec_datatype_basic_desc_create(size, min_disp,
                                             max_disp - min_disp, 0, newtype);
}

static int
parsec_datatype_basic_create_resized(parsec_datatype_t oldtype,
                                     ptrdiff_t lb,
                                     ptrdiff_t extent,
                                     parsec_datatype_t *newtype)
{
    int oldsize, rc, contiguous;

    rc = parsec_datatype_basic_get_info(oldtype, &oldsize, NULL, NULL,
                                        &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    return parsec_datatype_basic_desc_create(oldsize, lb, extent,
                                             contiguous && (extent == oldsize),
                                             newtype);
}

static int
parsec_datatype_basic_contiguous(parsec_datatype_t dtt)
{
    int contiguous, rc;

    rc = parsec_datatype_basic_get_info(dtt, NULL, NULL, NULL, &contiguous);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    return contiguous ? PARSEC_SUCCESS : PARSEC_ERROR;
}

const parsec_datatype_module_t parsec_datatype_basic_module = {
    .size                 = parsec_datatype_basic_size,
    .extent               = parsec_datatype_basic_extent,
    .free                 = parsec_datatype_basic_free,
    .create_contiguous    = parsec_datatype_basic_create_contiguous,
    .create_vector        = parsec_datatype_basic_create_vector,
    .create_hvector       = parsec_datatype_basic_create_hvector,
    .create_indexed       = parsec_datatype_basic_create_indexed,
    .create_indexed_block = parsec_datatype_basic_create_indexed_block,
    .create_struct        = parsec_datatype_basic_create_struct,
    .create_resized       = parsec_datatype_basic_create_resized,
    .contiguous           = parsec_datatype_basic_contiguous,
};
