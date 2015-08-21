/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague.h"
#include "dague/datatype.h"

/**
 * Map the datatype creation to the well designed and well known MPI datatype
 * mainipulation. However, right now we only provide the most basic types and
 * functions to mix them together.
 *
 * However, this file contains only the support functions needed when MPI is not
 * available.
 */
int dague_type_create_contiguous( int count,
                                  dague_datatype_t oldtype,
                                  dague_datatype_t* newtype )
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)oldtype;
    return DAGUE_SUCCESS;
}

int dague_type_create_vector( int count,
                              int blocklength,
                              int stride,
                              dague_datatype_t oldtype,
                              dague_datatype_t* newtype )
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return DAGUE_SUCCESS;
}

int dague_type_create_hvector( int count,
                               int blocklength,
                               ptrdiff_t stride,
                               dague_datatype_t oldtype,
                               dague_datatype_t* newtype )
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)stride; (void)oldtype;
    return DAGUE_SUCCESS;
}

int dague_type_create_indexed(int count,
                              const int array_of_blocklengths[],
                              const int array_of_displacements[],
                              dague_datatype_t oldtype,
                              dague_datatype_t *newtype)
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)oldtype;
    return DAGUE_SUCCESS;
}

int dague_type_create_indexed_block(int count,
                                    int blocklength,
                                    const int array_of_displacements[],
                                    dague_datatype_t oldtype,
                                    dague_datatype_t *newtype)
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)blocklength; (void)array_of_displacements; (void)oldtype;
    return DAGUE_SUCCESS;
}

int dague_type_create_struct(int count,
                             const int array_of_blocklengths[],
                             const ptrdiff_t array_of_displacements[],
                             const dague_datatype_t array_of_types[],
                             dague_datatype_t *newtype)
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)count; (void)array_of_blocklengths; (void)array_of_displacements; (void)array_of_types;
    return DAGUE_SUCCESS;
}

int dague_type_create_resized(dague_datatype_t oldtype,
                              ptrdiff_t lb,
                              ptrdiff_t extent,
                              dague_datatype_t *newtype)
{
    *newtype = DAGUE_DATATYPE_NULL;
    (void)lb; (void)extent; (void)oldtype;
    return DAGUE_SUCCESS;
}
