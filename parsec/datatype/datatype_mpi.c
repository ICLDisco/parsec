/*
 * Copyright (c) 2015-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/datatype.h"

#if !defined(PARSEC_HAVE_MPI)
#error __FILE__ should only be used when MPI support is enabled.
#endif  /* !defined(PARSEC_HAVE_MPI) */

int
parsec_type_size( parsec_datatype_t type, int *size )
{
    int rc = MPI_Type_size( type, size );
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_extent( parsec_datatype_t type, ptrdiff_t* lb, ptrdiff_t* extent)
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

int
parsec_type_free( parsec_datatype_t* type )
{
    int rc = MPI_Type_free(type);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_contiguous( int count,
                               parsec_datatype_t oldtype,
                               parsec_datatype_t* newtype )
{
    int rc = MPI_Type_contiguous( count, oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_vector( int count,
                           int blocklength,
                           int stride,
                           parsec_datatype_t oldtype,
                           parsec_datatype_t* newtype )
{
    int rc = MPI_Type_vector( count, blocklength, stride,
                              oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_hvector( int count,
                            int blocklength,
                            ptrdiff_t stride,
                            parsec_datatype_t oldtype,
                            parsec_datatype_t* newtype )
{
    int rc = MPI_Type_create_hvector( count, blocklength, stride,
                                      oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_indexed( int count,
                            const int array_of_blocklengths[],
                            const int array_of_displacements[],
                            parsec_datatype_t oldtype,
                            parsec_datatype_t *newtype )
{
    int rc = MPI_Type_indexed( count,
                               array_of_blocklengths,
                               array_of_displacements,
                               oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_indexed_block( int count,
                                  int blocklength,
                                  const int array_of_displacements[],
                                  parsec_datatype_t oldtype,
                                  parsec_datatype_t *newtype )
{
    int rc = MPI_Type_create_indexed_block( count, blocklength,
                                            array_of_displacements,
                                            oldtype, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_struct( int count,
                           const int *array_of_blocklengths,
                           const ptrdiff_t *array_of_displacements,
                           const parsec_datatype_t *array_of_types,
                           parsec_datatype_t *newtype )
{
    int rc = MPI_Type_create_struct( count,
                                     array_of_blocklengths,
                                     array_of_displacements,
                                     array_of_types, newtype );
    if( MPI_SUCCESS != rc ) return PARSEC_ERROR;
    rc = MPI_Type_commit(newtype);
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int
parsec_type_create_resized( parsec_datatype_t oldtype,
                            ptrdiff_t lb,
                            ptrdiff_t extent,
                            parsec_datatype_t *newtype )
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

int parsec_type_match(parsec_datatype_t dtt1,
                      parsec_datatype_t dtt2)
{
    (void)dtt1; (void)dtt2;
    return ( dtt1 == dtt2 ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int parsec_type_contiguous(parsec_datatype_t dtt)
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
