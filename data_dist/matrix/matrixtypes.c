/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec.h"
#include "parsec/constants.h"
#include "parsec/arena.h"
#include "data_dist/matrix/matrix.h"

#if defined(PARSEC_HAVE_MPI)
int parsec_matrix_get_extent( MPI_Datatype dt, MPI_Aint* extent )
{
    int rc;
#if defined(PARSEC_HAVE_MPI_20)
    MPI_Aint lb = 0; (void)lb;
    rc = MPI_Type_get_extent(dt, &lb, extent);
#else
    rc = MPI_Type_extent( dt, extent);
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    return (MPI_SUCCESS == rc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

int parsec_matrix_define_contiguous( parsec_datatype_t oldtype,
                                    unsigned int nb_elem,
                                    int resized,
                                    parsec_datatype_t* newtype )
{
    int oldsize, rc;

    /* Check if the type is valid and supported by the MPI library */
    rc = parsec_type_size(oldtype, &oldsize);
    if( 0 == oldsize ) {
        return PARSEC_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = parsec_type_create_contiguous(nb_elem, oldtype, newtype);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        MPI_Datatype tmp = *newtype;
        rc = parsec_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        MPI_Type_free(&tmp);
    }
    MPI_Type_commit(newtype);
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "CONT %s*%4u", oldtype_name, nb_elem);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    return PARSEC_SUCCESS;
}

int parsec_matrix_define_rectangle( parsec_datatype_t oldtype,
                                   unsigned int mb,
                                   unsigned int nb,
                                   unsigned int ld,
                                   int resized,
                                   parsec_datatype_t* newtype )
{
    int oldsize, rc;

    if ( mb == ld ) {
        return parsec_matrix_define_contiguous( oldtype, ld*nb, resized, newtype );
    }

    /* Check if the type is valid and supported by the MPI library */
    parsec_type_size(oldtype, &oldsize);
    if( 0 == oldsize ) {
        return PARSEC_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = parsec_type_create_vector( nb, mb, ld, oldtype, newtype );
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        MPI_Datatype tmp = *newtype;
        rc = parsec_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        MPI_Type_free(&tmp);
    }
    MPI_Type_commit(newtype);
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "RECT %s*%4u*%4u", oldtype_name, mb, nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    return PARSEC_SUCCESS;
}

int parsec_matrix_define_triangle( parsec_datatype_t oldtype,
                                  int uplo, int diag,
                                  unsigned int m,
                                  unsigned int n,
                                  unsigned int ld,
                                  parsec_datatype_t* newtype )
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
        return PARSEC_ERR_BAD_PARAM;

    rc = parsec_type_create_indexed(nmax, blocklens+diag, indices+diag, oldtype, &tmp );
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    parsec_type_size(oldtype, &oldsize);
    parsec_type_create_resized( tmp, 0, ld*n*oldsize, newtype );
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "UPPER %s*%4u*%4u", oldtype_name, m, n);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
    return PARSEC_SUCCESS;
}

int parsec_matrix_undefine_type(parsec_datatype_t* type)
{
    return (MPI_SUCCESS == MPI_Type_free(type) ? PARSEC_SUCCESS : PARSEC_ERROR);
}

#endif /* defined(PARSEC_HAVE_MPI) */

int parsec_matrix_add2arena( parsec_arena_t *arena, parsec_datatype_t oldtype,
                            int uplo, int diag,
                            unsigned int m, unsigned int n, unsigned int ld,
                            size_t alignment, int resized )
{
    parsec_datatype_t newtype = PARSEC_DATATYPE_NULL;
    parsec_aint_t extent = 0;
    int rc;

#if defined(PARSEC_HAVE_MPI)
    switch( uplo ) {
    case matrix_Lower:
    case matrix_Upper:
        rc = parsec_matrix_define_triangle( oldtype, uplo, diag, m, n, ld, &newtype );
        break;
    case matrix_UpperLower:
    default:
        if ( m == ld ) {
            rc = parsec_matrix_define_contiguous( oldtype, ld * n, resized, &newtype );
        }
        else {
            rc = parsec_matrix_define_rectangle( oldtype, m, n, ld, resized, &newtype );
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }

    rc = parsec_matrix_get_extent(newtype, &extent);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
#else
    int oldsize = 0;
    (void)uplo; (void)diag; (void)m; (void)resized;
    parsec_type_size( oldtype, &oldsize );
    extent = oldsize * n * ld;
#endif

    rc = parsec_arena_construct(arena, extent, alignment, newtype);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }

    return 0;
}

int parsec_matrix_del2arena( parsec_arena_t *arena )
{
    int rc = PARSEC_SUCCESS;
    (void)arena;

#if defined(PARSEC_HAVE_MPI)
    parsec_matrix_undefine_type( &(arena->opaque_dtt) );
#endif
    return rc;
}
