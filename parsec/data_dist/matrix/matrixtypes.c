/*
 * Copyright (c) 2010-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include "parsec/constants.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_MPI)
/* just for the type names */
#include <mpi.h>
#endif

int parsec_matrix_define_contiguous( parsec_datatype_t oldtype,
                                     unsigned int nb_elem,
                                     int resized,
                                     parsec_datatype_t* newtype )
{
    int oldsize, rc;

    /* Check if the type is valid and supported by the MPI library */
    rc = parsec_type_size(oldtype, &oldsize);
    if( 0 == oldsize ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = parsec_type_create_contiguous(nb_elem, oldtype, newtype);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        parsec_datatype_t tmp = *newtype;
        rc = parsec_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        parsec_type_free(&tmp);
    }
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len, mpi_is_on;
        MPI_Initialized(&mpi_is_on);
        if(mpi_is_on) {
            MPI_Type_get_name(oldtype, oldtype_name, &len);
            len = snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "CONT %s*%4u", oldtype_name, nb_elem);
            if(len >= MPI_MAX_OBJECT_NAME) {
                parsec_debug_verbose(50, parsec_debug_output, "Type name %s truncated when deriving from %s", newtype_name, oldtype_name);
            }
            MPI_Type_set_name(*newtype, newtype_name);
        }
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
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    /**
     * Define the TILE type.
     */
    rc = parsec_type_create_vector( nb, mb, ld, oldtype, newtype );
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( resized >= 0 ) {
        parsec_datatype_t tmp = *newtype;
        rc = parsec_type_create_resized(tmp, 0, resized*oldsize, newtype);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        parsec_type_free(&tmp);
    }
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len, mpi_is_on;
        MPI_Initialized(&mpi_is_on);
        if(mpi_is_on) {
            MPI_Type_get_name(oldtype, oldtype_name, &len);
            len = snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "RECT %s*%4u*%4u", oldtype_name, mb, nb);
            if(len >= MPI_MAX_OBJECT_NAME) {
                parsec_debug_verbose(50, parsec_debug_output, "Type name %s truncated when deriving from %s", newtype_name, oldtype_name);
            }
            MPI_Type_set_name(*newtype, newtype_name);
        }
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
    parsec_datatype_t tmp;
    unsigned int nmax;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( n * sizeof(int) );
    indices   = (int*)malloc( n * sizeof(int) );

    if ( uplo == PARSEC_MATRIX_UPPER ) {
        nmax = n-diag;

        for( i = diag; i < n; i++ ) {
            unsigned int mm = i + 1 - diag;
            blocklens[i] = mm < m ? mm : m ;
            indices[i]   = i * ld;
        }
    } else if ( uplo == PARSEC_MATRIX_LOWER ) {
        nmax = n >= (m-diag) ? m-diag : n;

        for( i = 0; i < nmax; i++ ) {
            blocklens[i] = m - i - diag;
            indices[i]   = i * ld + i + diag;
        }
        diag = 0;
    } else {
        free(blocklens);
        free(indices);
        return PARSEC_ERR_BAD_PARAM;
    }

    rc = parsec_type_create_indexed(nmax, blocklens+diag, indices+diag, oldtype, &tmp );
    if( PARSEC_SUCCESS != rc ) {
        free(blocklens);
        free(indices);
        return rc;
    }
    parsec_type_size(oldtype, &oldsize);
    parsec_type_create_resized( tmp, 0, ld*n*oldsize, newtype );
#if defined(PARSEC_HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len, mpi_is_on;
        MPI_Initialized(&mpi_is_on);
        if(mpi_is_on) {
            MPI_Type_get_name(oldtype, oldtype_name, &len);
            len = snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "%s %s*%4u*%4u", (uplo==PARSEC_MATRIX_UPPER)?"UPPER":"LOWER", oldtype_name, m, n);
            if(len >= MPI_MAX_OBJECT_NAME) {
                parsec_debug_verbose(50, parsec_debug_output, "Type name %s truncated when deriving from %s", newtype_name, oldtype_name);
            }
            MPI_Type_set_name(*newtype, newtype_name);
        }
    }
#endif  /* defined(PARSEC_HAVE_MPI_20) */
    parsec_type_free(&tmp);
    free(blocklens);
    free(indices);
    return PARSEC_SUCCESS;
}

int parsec_matrix_define_datatype(parsec_datatype_t *newtype, parsec_datatype_t oldtype,
                                  parsec_matrix_uplo_t uplo, int diag,
                                  unsigned int m, unsigned int n, unsigned int ld,
                                  int resized,
                                  ptrdiff_t * extent)
{
    int rc;
    ptrdiff_t lb = 0;

    *extent = 0;
    *newtype = PARSEC_DATATYPE_NULL;
#if defined(PARSEC_HAVE_MPI)
    switch( uplo ) {
    case PARSEC_MATRIX_LOWER:
    case PARSEC_MATRIX_UPPER:
        rc = parsec_matrix_define_triangle( oldtype, uplo, diag, m, n, ld, newtype );
        break;
    case PARSEC_MATRIX_FULL:
    default:
        if ( m == ld ) {
            rc = parsec_matrix_define_contiguous( oldtype, ld * n, resized, newtype );
        }
        else {
            rc = parsec_matrix_define_rectangle( oldtype, m, n, ld, resized, newtype );
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }

    rc = parsec_type_extent(*newtype, &lb, extent);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
#else
    int oldsize = 0;
    (void)uplo; (void)diag; (void)m; (void)resized; (void)lb;
    parsec_type_size( oldtype, &oldsize );
    *extent = oldsize * n * ld;
#endif

    return PARSEC_SUCCESS;
}

/**
 * Arena-datatype management.
 */
/* high level interfaces */
int parsec_matrix_arena_datatype_define_type(parsec_arena_datatype_t *adt,
                                             parsec_datatype_t oldtype,
                                             parsec_matrix_uplo_t uplo, int diag,
                                             unsigned int m, unsigned int n, unsigned int ld,
                                             size_t alignment, int resized)
{
    ptrdiff_t extent = 0;
    int rc;
    parsec_datatype_t newtype;

    rc = parsec_matrix_define_datatype(&newtype, oldtype, uplo, diag,
                                       m, n, ld, resized, &extent);
    if( PARSEC_SUCCESS != rc ) return rc;

    parsec_arena_datatype_set_type(adt, extent, alignment, newtype);
    if( PARSEC_SUCCESS != rc ) return rc;

    return PARSEC_SUCCESS;
}

int parsec_matrix_arena_datatype_destruct_free_type(parsec_arena_datatype_t *adt)
{
    parsec_type_free( &adt->opaque_dtt );
    PARSEC_OBJ_DESTRUCT(adt);
    return PARSEC_SUCCESS;
}

/* shorthands of the above for the common types */
int parsec_matrix_adt_define_rect(parsec_arena_datatype_t *adt,
                                parsec_datatype_t oldtype,
                                unsigned int m, unsigned int n, unsigned int ld)
{
    return parsec_matrix_arena_datatype_define_type(adt, oldtype, PARSEC_MATRIX_FULL, 0, m, n, ld, PARSEC_ARENA_ALIGNMENT_SSE, -1);
}

int parsec_matrix_adt_define_upper(parsec_arena_datatype_t *adt,
                                 parsec_datatype_t oldtype,
                                 int diag, unsigned int m)
{
    return parsec_matrix_arena_datatype_define_type(adt, oldtype, PARSEC_MATRIX_UPPER, diag, m, m, m, PARSEC_ARENA_ALIGNMENT_SSE, -1);
}

int parsec_matrix_adt_define_lower(parsec_arena_datatype_t *adt,
                                 parsec_datatype_t oldtype,
                                 int diag, unsigned int m)
{
    return parsec_matrix_arena_datatype_define_type(adt, oldtype, PARSEC_MATRIX_LOWER, diag, m, m, m, PARSEC_ARENA_ALIGNMENT_SSE, -1);
}

int parsec_matrix_adt_define_square(parsec_arena_datatype_t *adt,
                                  parsec_datatype_t oldtype,
                                  unsigned int m)
{
    return parsec_matrix_arena_datatype_define_type(adt, oldtype, PARSEC_MATRIX_FULL, 0, m, m, m, PARSEC_ARENA_ALIGNMENT_SSE, -1);
}

parsec_arena_datatype_t *parsec_matrix_adt_new_rect(parsec_datatype_t oldtype,
                                unsigned int m, unsigned int n, unsigned int ld)
{
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    parsec_matrix_adt_define_rect(adt, oldtype, m, n, ld);
    return adt;
}

parsec_arena_datatype_t *parsec_matrix_adt_new_upper(parsec_datatype_t oldtype,
                                 int diag, unsigned int m)
{
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    parsec_matrix_adt_define_upper(adt, oldtype, diag, m);
    return adt;
}

parsec_arena_datatype_t *parsec_matrix_adt_new_lower(parsec_datatype_t oldtype,
                                 int diag, unsigned int m)
{
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    parsec_matrix_adt_define_lower(adt, oldtype, diag, m);
    return adt;
}

parsec_arena_datatype_t *parsec_matrix_adt_new_square(parsec_datatype_t oldtype,
                                  unsigned int m)
{
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    parsec_matrix_adt_define_square(adt, oldtype, m);
    return adt;
}

int parsec_matrix_adt_free(parsec_arena_datatype_t **padt)
{
    parsec_matrix_arena_datatype_destruct_free_type(*padt);
    free(*padt); /* OBJ_DESTRUCT done above */
    *padt = NULL;
    return PARSEC_SUCCESS;
}

