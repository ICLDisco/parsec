#ifndef DAGUE_DATATYPE_H_HAS_BEEN_INCLUDED
#define DAGUE_DATATYPE_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague.h"
#include "dague/constants.h"
#if defined(HAVE_MPI)
#include <mpi.h>

#define DAGUE_DATATYPE_NULL  MPI_DATATYPE_NULL
typedef MPI_Datatype dague_datatype_t;

#define dague_datatype_int8_t        MPI_INT8_T
#define dague_datatype_int16_t       MPI_INT16_T
#define dague_datatype_int32_t       MPI_INT32_T
#define dague_datatype_int64_t       MPI_INT64_T
#define dague_datatype_uint8_t       MPI_UINT8_T
#define dague_datatype_uint16_t      MPI_UINT16_T
#define dague_datatype_uint32_t      MPI_UINT32_T
#define dague_datatype_uint64_t      MPI_UINT64_T
#define dague_datatype_float_t       MPI_FLOAT
#define dague_datatype_double_t      MPI_DOUBLE
#define dague_datatype_long_double_t MPI_LONG_DOUBLE

#include "dague/datatype/datatype_mpi.h"

#else  /* !defined(HAVE_MPI) */

#define DAGUE_DATATYPE_NULL  NULL
typedef void* dague_datatype_t;

#define dague_datatype_int8_t        NULL
#define dague_datatype_int16_t       NULL
#define dague_datatype_int32_t       NULL
#define dague_datatype_int64_t       NULL
#define dague_datatype_uint8_t       NULL
#define dague_datatype_uint16_t      NULL
#define dague_datatype_uint32_t      NULL
#define dague_datatype_uint64_t      NULL
#define dague_datatype_float_t       NULL
#define dague_datatype_double_t      NULL
#define dague_datatype_long_double_t NULL

BEGIN_C_DECLS

/**
 * Map the datatype creation to the well designed and well known MPI datatype
 * mainipulation. However, right now we only provide the most basic types and
 * functions to mix them together.
 */
int dague_type_create_contiguous(int count,
                                 dague_datatype_t oldtype,
                                 dague_datatype_t* newtype );
int dague_type_create_vector(int count,
                             int blocklength,
                             int stride,
                             dague_datatype_t oldtype,
                              dague_datatype_t* newtype );
int dague_type_create_hvector(int count,
                              int blocklength,
                              ptrdiff_t stride,
                              dague_datatype_t oldtype,
                              dague_datatype_t* newtype );
int dague_type_create_indexed(int count,
                              const int array_of_blocklengths[],
                              const int array_of_displacements[],
                              dague_datatype_t oldtype,
                              dague_datatype_t *newtype);
int dague_type_create_indexed_block(int count,
                                    int blocklength,
                                    const int array_of_displacements[],
                                    dague_datatype_t oldtype,
                                    dague_datatype_t *newtype);
int dague_type_create_struct(int count,
                             const int array_of_blocklengths[],
                             const ptrdiff_t array_of_displacements[],
                             const dague_datatype_t array_of_types[],
                             dague_datatype_t *newtype);
int dague_type_create_resized(dague_datatype_t oldtype,
                              ptrdiff_t lb,
                              ptrdiff_t extent,
                              dague_datatype_t *newtype);

END_C_DECLS

#endif  /* defined(HAVE_MPI) */

#endif  /* DAGUE_DATATYPE_H_HAS_BEEN_INCLUDED */
