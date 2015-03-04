#ifndef DAGUE_DATATYPE_MPI_H_HAS_BEEN_INCLUDED
#define DAGUE_DATATYPE_MPI_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * This file should only be directly included from dague/datatype.h.
 */
#if !defined(DAGUE_DATATYPE_H_HAS_BEEN_INCLUDED)
#error __FILE__ should only be directly included from dague/datatype.h. Include "dague/datatype.h" instead.
#endif  /* !defined(DAGUE_DATATYPE_H_HAS_BEEN_INCLUDED) */

#define dague_type_create_contiguous    MPI_Type_contiguous
#define dague_type_create_vector        MPI_Type_create_vector
#define dague_type_create_hvector       MPI_Type_create_hvector
#define dague_type_create_indexed       MPI_Type_create_indexed
#define dague_type_create_indexed_block MPI_Type_create_indexed_block
#define dague_type_create_struct        MPI_Type_sreate_struct

static inline int
dague_type_create_resized(dague_datatype_t oldtype,
                          ptrdiff_t lb,
                          ptrdiff_t extent,
                          dague_datatype_t *newtype)
{
    int rc;
#if defined(HAVE_MPI_20)
    rc = MPI_Type_create_resized(oldtype, lb, extent, newtype);
#else
    int blocklens[] = {1, 1, 1};
    MPI_Aint indices[] = {lb, 0, extent};
    MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
    rc = MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
#endif  /* defined(HAVE_MPI_20) */
    return (MPI_SUCCESS == rc ? DAGUE_SUCCESS : DAGUE_ERROR);
}

#endif  /* DAGUE_DATATYPE_MPI_H_HAS_BEEN_INCLUDED */
