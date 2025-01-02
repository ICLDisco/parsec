/*
 * Copyright (c) 2015-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/runtime.h"
#include "parsec/datatype.h"

/**
 * Map the datatype creation to the well designed and well known MPI datatype
 * mainipulation. However, right now we only provide the most basic types and
 * functions to mix them together.
 *
 * However, this file contains only the support functions needed when MPI is not
 * available.
 */
int parsec_type_size( parsec_datatype_t type,
                     int *size )
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

int parsec_type_extent(parsec_datatype_t type, ptrdiff_t* lb, ptrdiff_t* extent) {
    int size, rc;
    rc = parsec_type_size(type, &size);
    *extent = size;
    return rc;
}

int parsec_type_free(parsec_datatype_t* type) {
    *type = PARSEC_DATATYPE_NULL;
    return PARSEC_SUCCESS;
}

int parsec_type_create_contiguous( int count,
                                  parsec_datatype_t oldtype,
                                  parsec_datatype_t* newtype )
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_create_vector( int count,
                              int blocklength,
                              int stride,
                              parsec_datatype_t oldtype,
                              parsec_datatype_t* newtype )
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_create_hvector( int count,
                               int blocklength,
                               ptrdiff_t stride,
                               parsec_datatype_t oldtype,
                               parsec_datatype_t* newtype )
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_create_indexed(int count,
                              const int array_of_blocklengths[],
                              const int array_of_displacements[],
                              parsec_datatype_t oldtype,
                              parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_create_indexed_block(int count,
                                    int blocklength,
                                    const int array_of_displacements[],
                                    parsec_datatype_t oldtype,
                                    parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)array_of_displacements; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_create_struct(int count,
                             const int array_of_blocklengths[],
                             const ptrdiff_t array_of_displacements[],
                             const parsec_datatype_t array_of_types[],
                             parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)array_of_types;
    return PARSEC_SUCCESS;
}

int parsec_type_create_resized(parsec_datatype_t oldtype,
                              ptrdiff_t lb,
                              ptrdiff_t extent,
                              parsec_datatype_t *newtype)
{
    *newtype = PARSEC_DATATYPE_NULL;
    (void)lb; (void)extent; (void)oldtype;
    return PARSEC_SUCCESS;
}

int parsec_type_match(parsec_datatype_t dtt1,
                      parsec_datatype_t dtt2){
    (void)dtt1; (void)dtt2;
    return PARSEC_SUCCESS;
}

int parsec_type_contiguous(parsec_datatype_t dtt)
{
    (void)dtt;
    return PARSEC_SUCCESS;
}
