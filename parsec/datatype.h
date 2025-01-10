/*
 * Copyright (c) 2015-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DATATYPE_H_HAS_BEEN_INCLUDED
#define PARSEC_DATATYPE_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>

/**
 *  @defgroup parsec_internal_datatype Datatype portability
 *  @ingroup parsec_internal
 *    The Datatype portability group defines an abstract API that
 *    allows to define basic data types on top of a variety of
 *    communication and data movement engines.
 *  @{
 */

#define PARSEC_DATATYPE_NULL  MPI_DATATYPE_NULL
#define PARSEC_DATATYPE_PACKED  MPI_PACKED
typedef MPI_Datatype parsec_datatype_t;

#define parsec_datatype_int_t              MPI_INT
#define parsec_datatype_int8_t             MPI_INT8_T
#define parsec_datatype_int16_t            MPI_INT16_T
#define parsec_datatype_int32_t            MPI_INT32_T
#define parsec_datatype_int64_t            MPI_INT64_T
#define parsec_datatype_uint8_t            MPI_UINT8_T
#define parsec_datatype_uint16_t           MPI_UINT16_T
#define parsec_datatype_uint32_t           MPI_UINT32_T
#define parsec_datatype_uint64_t           MPI_UINT64_T
#define parsec_datatype_float_t            MPI_FLOAT
#define parsec_datatype_double_t           MPI_DOUBLE
#define parsec_datatype_long_double_t      MPI_LONG_DOUBLE
#define parsec_datatype_complex_t          MPI_C_COMPLEX
#define parsec_datatype_double_complex_t   MPI_C_DOUBLE_COMPLEX

#else  /* !defined(PARSEC_HAVE_MPI) */

#define PARSEC_DATATYPE_NULL  ((intptr_t)NULL)
#define PARSEC_DATATYPE_PACKED  ((intptr_t)NULL)
typedef intptr_t  parsec_datatype_t;

#define parsec_datatype_int_t              1
#define parsec_datatype_int8_t             2
#define parsec_datatype_int16_t            3
#define parsec_datatype_int32_t            4
#define parsec_datatype_int64_t            5
#define parsec_datatype_uint8_t            6
#define parsec_datatype_uint16_t           7
#define parsec_datatype_uint32_t           8
#define parsec_datatype_uint64_t           9
#define parsec_datatype_float_t            10
#define parsec_datatype_double_t           11
#define parsec_datatype_long_double_t      12
#define parsec_datatype_complex_t          13
#define parsec_datatype_double_complex_t   14

#endif  /* !defined(PARSEC_HAVE_MPI) */

BEGIN_C_DECLS

/**
 * Map the datatype creation to the well designed and well known MPI datatype
 * API. The datatype support remains extremely basic, providing API only for
 * basic datatypes and functions to mix them together.
 */
int parsec_type_size(parsec_datatype_t type,
                     int *size);
int parsec_type_extent(parsec_datatype_t type, ptrdiff_t *lb, ptrdiff_t *extent);

int parsec_type_free(parsec_datatype_t* type);
int parsec_type_create_contiguous(int count,
                                  parsec_datatype_t oldtype,
                                  parsec_datatype_t* newtype );
int parsec_type_create_vector(int count,
                              int blocklength,
                              int stride,
                              parsec_datatype_t oldtype,
                              parsec_datatype_t* newtype );
int parsec_type_create_hvector(int count,
                               int blocklength,
                               ptrdiff_t stride,
                               parsec_datatype_t oldtype,
                               parsec_datatype_t* newtype );
int parsec_type_create_indexed(int count,
                               const int array_of_blocklengths[],
                               const int array_of_displacements[],
                               parsec_datatype_t oldtype,
                               parsec_datatype_t *newtype);
int parsec_type_create_indexed_block(int count,
                                     int blocklength,
                                     const int array_of_displacements[],
                                     parsec_datatype_t oldtype,
                                     parsec_datatype_t *newtype);
int parsec_type_create_struct(int count,
                              const int array_of_blocklengths[],
                              const ptrdiff_t array_of_displacements[],
                              const parsec_datatype_t array_of_types[],
                              parsec_datatype_t *newtype);
int parsec_type_create_resized(parsec_datatype_t oldtype,
                               ptrdiff_t lb,
                               ptrdiff_t extent,
                               parsec_datatype_t *newtype);

/**
 * Routine to check if two datatypes represent the same data extraction.
 * @param[in] parsec_datatype_t datatype
 * @param[in] parsec_datatype_t datatype
 * @return PARSEC_SUCCESS if the two datatypes matches, PARSEC_ERROR otherwise.
 */
int parsec_type_match(parsec_datatype_t dtt1,
                      parsec_datatype_t dtt2);

/**
 * Routine to check if a datatype is contiguous.
 * @param[in] parsec_datatype_t datatype
 * @return PARSEC_SUCCESS if it was created with MPI_Type_contiguous, PARSEC_ERROR otherwise.
 */
int parsec_type_contiguous(parsec_datatype_t dtt);
END_C_DECLS

/** @} */

#endif  /* PARSEC_DATATYPE_H_HAS_BEEN_INCLUDED */
