/*
 * Copyright (c) 2015-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/datatype_module.h"

#if !defined(PARSEC_HAVE_MPI)
#error __FILE__ should only be used when MPI support is enabled.
#endif  /* !defined(PARSEC_HAVE_MPI) */

static int
comm_mpi_datatype_size(parsec_datatype_t type, int *size)
{
    int rc = MPI_Type_size( type, size );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_extent(parsec_datatype_t type, ptrdiff_t *lb, ptrdiff_t *extent)
{
    int rc;
    MPI_Aint mpi_extent, mpi_lb;
#if defined(PARSEC_HAVE_MPI_20)
    rc = MPI_Type_get_extent(type, &mpi_lb, &mpi_extent);
#else
    mpi_lb = 0;
    rc = MPI_Type_extent( type, &mpi_extent);
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    *lb = mpi_lb; *extent = mpi_extent;
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_free(parsec_datatype_t *type)
{
    int rc = MPI_Type_free(type);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_contiguous(int count,
                                    parsec_datatype_t oldtype,
                                    parsec_datatype_t *newtype)
{
    int rc = MPI_Type_contiguous( count, oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_vector(int count,
                                int blocklength,
                                int stride,
                                parsec_datatype_t oldtype,
                                parsec_datatype_t *newtype)
{
    int rc = MPI_Type_vector( count, blocklength, stride,
                              oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_hvector(int count,
                                 int blocklength,
                                 ptrdiff_t stride,
                                 parsec_datatype_t oldtype,
                                 parsec_datatype_t *newtype)
{
    int rc = MPI_Type_create_hvector( count, blocklength, stride,
                                      oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_indexed(int count,
                                 const int array_of_blocklengths[],
                                 const int array_of_displacements[],
                                 parsec_datatype_t oldtype,
                                 parsec_datatype_t *newtype)
{
    int rc = MPI_Type_indexed( count,
                               array_of_blocklengths,
                               array_of_displacements,
                               oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_indexed_block(int count,
                                       int blocklength,
                                       const int array_of_displacements[],
                                       parsec_datatype_t oldtype,
                                       parsec_datatype_t *newtype)
{
    int rc = MPI_Type_create_indexed_block( count, blocklength,
                                            array_of_displacements,
                                            oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_struct(int count,
                                const int *array_of_blocklengths,
                                const ptrdiff_t *array_of_displacements,
                                const parsec_datatype_t *array_of_types,
                                parsec_datatype_t *newtype)
{
    int rc = MPI_Type_create_struct( count,
                                     array_of_blocklengths,
                                     array_of_displacements,
                                     array_of_types, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_create_resized(parsec_datatype_t oldtype,
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
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

static int
comm_mpi_datatype_contiguous(parsec_datatype_t dtt)
{
    int rc;
    int num_integers, num_addresses, num_datatypes, combiner;
    rc = MPI_Type_get_envelope(dtt, &num_integers, &num_addresses, &num_datatypes, &combiner);
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    /* Weak: datatype may be contiguous but not created with MPI_Type_contiguous */
    if( combiner == MPI_COMBINER_CONTIGUOUS ){
        return PARSEC_SUCCESS;
    }
    return PARSEC_ERROR;
}

/**
 * Whether one element of dtt holds its data as a single run of bytes at
 * offsets [0, size), in typemap order (see parsec_type_dense()).
 *
 * Decode the constructors, and accept only predefined types without holes
 * and the types built from them by dup, contiguous, vector or hvector with
 * touching blocks, and resized. Anything else, indexed and struct types
 * included, is reported as not dense.
 */
static int
comm_mpi_datatype_element_is_dense(MPI_Datatype dtt, int depth)
{
    int ni, na, nd, combiner, size, ints[3], dense = 0;
    MPI_Aint lb, extent, addrs[2];
    MPI_Datatype old;

    if( MPI_DATATYPE_NULL == dtt || depth > 16 ) return 0;
    if( MPI_SUCCESS != MPI_Type_get_envelope(dtt, &ni, &na, &nd, &combiner) ) return 0;
    if( MPI_COMBINER_NAMED == combiner ) {
        /* Some predefined pair types (MPI_SHORT_INT) have holes. */
        return (MPI_SUCCESS == MPI_Type_size(dtt, &size)) &&
               (MPI_SUCCESS == MPI_Type_get_true_extent(dtt, &lb, &extent)) &&
               (0 < size) && (0 == lb) && ((MPI_Aint)size == extent);
    }
    if( (MPI_COMBINER_DUP != combiner) && (MPI_COMBINER_CONTIGUOUS != combiner) &&
        (MPI_COMBINER_VECTOR != combiner) && (MPI_COMBINER_HVECTOR != combiner) &&
        (MPI_COMBINER_RESIZED != combiner) ) return 0;
    if( (ni > 3) || (na > 2) || (1 != nd) ) return 0;
    if( MPI_SUCCESS != MPI_Type_get_contents(dtt, ni, na, nd, ints, addrs, &old) ) return 0;

    if( comm_mpi_datatype_element_is_dense(old, depth + 1) &&
        (MPI_SUCCESS == MPI_Type_size(old, &size)) &&
        (MPI_SUCCESS == MPI_Type_get_extent(old, &lb, &extent)) ) {
        /* Consecutive elements of old touch only if its extent is its size. */
        int touch = ((MPI_Aint)size == extent);
        switch( combiner ) {
        case MPI_COMBINER_CONTIGUOUS:  /* ints: count */
            dense = (ints[0] <= 1) || touch;
            break;
        case MPI_COMBINER_VECTOR:      /* ints: count, blocklength, stride in elements */
            dense = ((ints[0] <= 1) || (ints[2] == ints[1])) &&
                    (((int64_t)ints[0] * ints[1] <= 1) || touch);
            break;
        case MPI_COMBINER_HVECTOR:     /* ints: count, blocklength; addrs: stride in bytes */
            dense = ((ints[0] <= 1) || (addrs[0] == (MPI_Aint)ints[1] * extent)) &&
                    (((int64_t)ints[0] * ints[1] <= 1) || touch);
            break;
        default:  /* dup and resized keep the data of old where it was */
            dense = 1;
        }
    }
    /* get_contents hands back new handles for derived types; free those. */
    if( (MPI_SUCCESS == MPI_Type_get_envelope(old, &ni, &na, &nd, &combiner)) &&
        (MPI_COMBINER_NAMED != combiner) ) {
        MPI_Type_free(&old);
    }
    return dense;
}

static int
comm_mpi_datatype_dense(parsec_datatype_t dtt)
{
    return comm_mpi_datatype_element_is_dense(dtt, 0) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

const parsec_datatype_module_t parsec_comm_mpi_datatype_module = {
    .size                 = comm_mpi_datatype_size,
    .extent               = comm_mpi_datatype_extent,
    .free                 = comm_mpi_datatype_free,
    .create_contiguous    = comm_mpi_datatype_create_contiguous,
    .create_vector        = comm_mpi_datatype_create_vector,
    .create_hvector       = comm_mpi_datatype_create_hvector,
    .create_indexed       = comm_mpi_datatype_create_indexed,
    .create_indexed_block = comm_mpi_datatype_create_indexed_block,
    .create_struct        = comm_mpi_datatype_create_struct,
    .create_resized       = comm_mpi_datatype_create_resized,
    .contiguous           = comm_mpi_datatype_contiguous,
    .dense                = comm_mpi_datatype_dense,
};
