/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>
#include <dague/constants.h>
#include <dague/arena.h>
#include "data_dist/matrix/matrix.h"

#if defined(HAVE_MPI)
int dague_matrix_get_extent( MPI_Datatype dt, MPI_Aint* extent )
{
    int rc;
#if defined(HAVE_MPI_20)
    MPI_Aint lb = 0; (void)lb;
    rc = MPI_Type_get_extent(dt, &lb, extent);
#else
    rc = MPI_Type_extent( dt, extent);
#endif  /* defined(HAVE_MPI_20) */
    return (MPI_SUCCESS == rc ? DAGUE_SUCCESS : DAGUE_ERROR);
}

int dague_matrix_define_contiguous( dague_datatype_t oldtype,
                                    unsigned int nb_elem,
                                    int resized,
                                    dague_datatype_t* newtype )
{
    int oldsize, rc;

    /* Check if the type is valid and supported by the MPI library */
    rc = dague_type_size(oldtype, &oldsize);
    if( 0 == oldsize ) {
        return DAGUE_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = dague_type_create_contiguous(nb_elem, oldtype, newtype);
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        MPI_Datatype tmp = *newtype;
        rc = dague_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( DAGUE_SUCCESS != rc ) {
            return rc;
        }
        MPI_Type_free(&tmp);
    }
    MPI_Type_commit(newtype);
#if defined(HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "CONT %s*%4u", oldtype_name, nb_elem);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(HAVE_MPI_20) */
    return DAGUE_SUCCESS;
}

int dague_matrix_define_rectangle( dague_datatype_t oldtype,
                                   unsigned int mb,
                                   unsigned int nb,
                                   unsigned int ld,
                                   int resized,
                                   dague_datatype_t* newtype )
{
    int oldsize, rc;

    if ( mb == ld ) {
        return dague_matrix_define_contiguous( oldtype, ld*nb, resized, newtype );
    }

    /* Check if the type is valid and supported by the MPI library */
    dague_type_size(oldtype, &oldsize);
    if( 0 == oldsize ) {
        return DAGUE_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = dague_type_create_vector( nb, mb, ld, oldtype, newtype );
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        MPI_Datatype tmp = *newtype;
        rc = dague_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( DAGUE_SUCCESS != rc ) {
            return rc;
        }
        MPI_Type_free(&tmp);
    }
    MPI_Type_commit(newtype);
#if defined(HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "RECT %s*%4u*%4u", oldtype_name, mb, nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(HAVE_MPI_20) */
    return DAGUE_SUCCESS;
}

int dague_matrix_define_triangle( dague_datatype_t oldtype,
                                  int uplo, int diag,
                                  unsigned int m,
                                  unsigned int n,
                                  unsigned int ld,
                                  dague_datatype_t* newtype )
{
    int *blocklens, *indices, oldsize, rc;
    unsigned int i;
    MPI_Datatype tmp;
    unsigned int nmax;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( n * sizeof(int) );
    indices   = (int*)malloc( n * sizeof(int) );

    if ( uplo == matrix_Upper ) {
        nmax = n-diag;

        for( i = diag; i < n; i++ ) {
            unsigned int mm = i + 1 - diag;
            blocklens[i] = mm < m ? mm : m ;
            indices[i]   = i * ld;
        }
    }
    else if ( uplo == matrix_Lower ) {
        nmax = n >= (m-diag) ? m-diag : n;

        for( i = 0; i < nmax; i++ ) {
            blocklens[i] = m - i - diag;
            indices[i]   = i * ld + i + diag;
        }
        diag = 0;
    }
    else
        return DAGUE_ERR_BAD_PARAM;

    rc = dague_type_create_indexed(nmax, blocklens+diag, indices+diag, oldtype, &tmp );
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }
    dague_type_size(oldtype, &oldsize);
    dague_type_create_resized( tmp, 0, ld*n*oldsize, newtype );
#if defined(HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "UPPER %s*%4u*%4u", oldtype_name, m, n);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(HAVE_MPI_20) */
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
    return DAGUE_SUCCESS;
}

int dague_matrix_undefine_type(dague_datatype_t* type)
{
    return (MPI_SUCCESS == MPI_Type_free(type) ? DAGUE_SUCCESS : DAGUE_ERROR);
}

#endif /* defined(HAVE_MPI) */

int dague_matrix_add2arena( dague_arena_t *arena, dague_datatype_t oldtype,
                            int uplo, int diag,
                            unsigned int m, unsigned int n, unsigned int ld,
                            size_t alignment, int resized )
{
    dague_datatype_t newtype = NULL;
    MPI_Aint extent = 0;
    int rc;

#if defined(HAVE_MPI)
    switch( uplo ) {
    case matrix_Lower:
    case matrix_Upper:
        rc = dague_matrix_define_triangle( oldtype, uplo, diag, m, n, ld, &newtype );
        break;
    case matrix_UpperLower:
    default:
        if ( m == ld ) {
            rc = dague_matrix_define_contiguous( oldtype, ld * n, resized, &newtype );
        }
        else {
            rc = dague_matrix_define_rectangle( oldtype, m, n, ld, resized, &newtype );
        }
    }
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }

    rc = dague_matrix_get_extent(newtype, &extent);
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }
#else
    int oldsize = 0;
    (void)uplo; (void)diag; (void)m; (void)resized;
    dague_type_size( oldtype, &oldsize );
    extent = oldsize * n * ld;
#endif

    rc = dague_arena_construct(arena, extent, alignment, newtype);
    if( DAGUE_SUCCESS != rc ) {
        return rc;
    }

    return 0;
}

int dague_matrix_del2arena( dague_arena_t *arena )
{
    int rc = DAGUE_SUCCESS;
    (void)arena;

#if defined(HAVE_MPI)
    dague_matrix_undefine_type( &(arena->opaque_dtt) );
#endif
    return rc;
}
