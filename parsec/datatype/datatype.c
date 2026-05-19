/*
 * Copyright (c) 2015-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include "parsec/runtime.h"
#include "parsec/datatype_module.h"

/**
 * Minimal datatype backend used when no communication component provides a
 * richer datatype implementation.  It recognizes PaRSEC's predefined scalar
 * datatypes and treats all constructors as no-op placeholders.
 */
static int
parsec_datatype_basic_size(parsec_datatype_t type, int *size)
{
    *size = 0;
    switch( type ) {
    case parsec_datatype_int_t:
        *size = sizeof( int ); break;
    case parsec_datatype_int8_t:
        *size = sizeof( int8_t ); break;
    case parsec_datatype_int16_t:
        *size = sizeof( int16_t ); break;
    case parsec_datatype_int32_t:
        *size = sizeof( int32_t ); break;
    case parsec_datatype_int64_t:
        *size = sizeof( int64_t ); break;
    case parsec_datatype_uint8_t:
        *size = sizeof( uint8_t ); break;
    case parsec_datatype_uint16_t:
        *size = sizeof( uint16_t ); break;
    case parsec_datatype_uint32_t:
        *size = sizeof( uint32_t ); break;
    case parsec_datatype_uint64_t:
        *size = sizeof( uint64_t ); break;
    case parsec_datatype_float_t:
        *size = sizeof( float ); break;
    case parsec_datatype_double_t:
        *size = sizeof( double ); break;
    case parsec_datatype_long_double_t:
        *size = sizeof( long double ); break;
    case parsec_datatype_complex_t:
        *size = 2 * sizeof( float ); break;
    case parsec_datatype_double_complex_t:
        *size = 2 * sizeof( double ); break;
    default:
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_extent(parsec_datatype_t type, ptrdiff_t *lb, ptrdiff_t *extent)
{
    int size, rc;

    rc = parsec_datatype_basic_size(type, &size);
    if( NULL != lb ) {
        *lb = 0;
    }
    *extent = size;
    return rc;
}

static int
parsec_datatype_basic_free(parsec_datatype_t *type)
{
    *type = PARSEC_DATATYPE_NULL;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_contiguous(int count,
                                        parsec_datatype_t oldtype,
                                        parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_vector(int count,
                                    int blocklength,
                                    int stride,
                                    parsec_datatype_t oldtype,
                                    parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_hvector(int count,
                                     int blocklength,
                                     ptrdiff_t stride,
                                     parsec_datatype_t oldtype,
                                     parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_indexed(int count,
                                     const int array_of_blocklengths[],
                                     const int array_of_displacements[],
                                     parsec_datatype_t oldtype,
                                     parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_indexed_block(int count,
                                           int blocklength,
                                           const int array_of_displacements[],
                                           parsec_datatype_t oldtype,
                                           parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)array_of_displacements; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_struct(int count,
                                    const int array_of_blocklengths[],
                                    const ptrdiff_t array_of_displacements[],
                                    const parsec_datatype_t array_of_types[],
                                    parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)array_of_types;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_create_resized(parsec_datatype_t oldtype,
                                     ptrdiff_t lb,
                                     ptrdiff_t extent,
                                     parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)lb; (void)extent; (void)oldtype;
    return PARSEC_SUCCESS;
}

static int
parsec_datatype_basic_contiguous(parsec_datatype_t dtt)
{
    (void)dtt;
    return PARSEC_SUCCESS;
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
