#ifndef PARSEC_DATATYPE_MPI_H_HAS_BEEN_INCLUDED
#define PARSEC_DATATYPE_MPI_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * This file should only be directly included from parsec/datatype.h.
 */
#if !defined(PARSEC_DATATYPE_H_HAS_BEEN_INCLUDED)
#error __FILE__ should only be directly included from parsec/datatype.h. Include "parsec/datatype.h" instead.
#endif  /* !defined(PARSEC_DATATYPE_H_HAS_BEEN_INCLUDED) */

#if !defined(PARSEC_HAVE_MPI)
#error __FILE__ should only be used when MPI support is enabled.
#endif  /* !defined(PARSEC_HAVE_MPI) */

static inline int
parsec_type_size( parsec_datatype_t type, int *size )
{
    int rc = MPI_Type_size( type, size );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_contiguous( int count,
                              parsec_datatype_t oldtype,
                              parsec_datatype_t* newtype )
{
    int rc = MPI_Type_contiguous( count, oldtype, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_vector( int count,
                          int blocklength,
                          int stride,
                          parsec_datatype_t oldtype,
                          parsec_datatype_t* newtype )
{
    int rc = MPI_Type_vector( count, blocklength, stride,
                              oldtype, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_hvector( int count,
                           int blocklength,
                           ptrdiff_t stride,
                           parsec_datatype_t oldtype,
                           parsec_datatype_t* newtype )
{
    int rc = MPI_Type_create_hvector( count, blocklength, stride,
                                      oldtype, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_indexed( int count,
                           /* const */ int array_of_blocklengths[],
                           /* const */ int array_of_displacements[],
                           parsec_datatype_t oldtype,
                           parsec_datatype_t *newtype )
{
    int rc = MPI_Type_indexed( count,
                               array_of_blocklengths,
                               array_of_displacements,
                               oldtype, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_indexed_block( int count,
                                 int blocklength,
                                 /* const */ int array_of_displacements[],
                                 parsec_datatype_t oldtype,
                                 parsec_datatype_t *newtype )
{
    int rc = MPI_Type_create_indexed_block( count, blocklength,
                                            array_of_displacements,
                                            oldtype, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_struct( int count,
                          /* const */ int *array_of_blocklengths,
                          /* const */ ptrdiff_t *array_of_displacements,
                          /* const */ parsec_datatype_t *array_of_types,
                          parsec_datatype_t *newtype )
{
    int rc = MPI_Type_create_struct( count,
                                     array_of_blocklengths,
                                     array_of_displacements,
                                     array_of_types, newtype );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static inline int
parsec_type_create_resized(parsec_datatype_t oldtype,
                          ptrdiff_t lb,
                          ptrdiff_t extent,
                          parsec_datatype_t *newtype)
{
    int rc;
#if defined(PARSEC_HAVE_MPI_20)
    rc = MPI_Type_create_resized(oldtype, lb, extent, newtype);
#else
    int blocklens[] = {1, 1, 1};
    MPI_Aint indices[] = {lb, 0, extent};
    MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
    rc = MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

#endif  /* PARSEC_DATATYPE_MPI_H_HAS_BEEN_INCLUDED */
