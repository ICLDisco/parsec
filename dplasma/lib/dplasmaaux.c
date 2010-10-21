/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmaaux.h"

int dplasma_aux_get_priority( char* function, const tiled_matrix_desc_t* ddesc )
{
    (void)function; (void)ddesc;  /* TODO */
    return 0;
}

int dplasma_aux_create_rectangle_type( dague_remote_dep_datatype_t oldtype,
                                       unsigned int tile_nb,
                                       unsigned int tile_mb,
                                       int resized,
                                       dague_remote_dep_datatype_t* newtype )
{
#if defined(USE_MPI)
    int oldsize;
    /**
     * Define the TILE type.
     */
    MPI_Type_contiguous(tile_nb * tile_mb, oldtype, newtype);
    MPI_Type_size(oldtype, &oldsize);
    if( resized >= 0 ) {
        MPI_Datatype tmp = *newtype;
        MPI_Type_create_resized(tmp, 0, resized*oldsize, newtype);
        MPI_Type_free(&tmp);
    }
    MPI_Type_commit(newtype);
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "TILE %s*%4u*%4u", oldtype_name, tile_nb, tile_mb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#else
    *newtype = NULL; (void)oldtype; (void)tile_nb; (void)tile_mb; (void)resized;
#endif  /* USE_MPI */
    return 0;
}

int dplasma_aux_create_tile_type( dague_remote_dep_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_remote_dep_datatype_t* newtype )
{
    return dplasma_aux_create_rectangle_type(oldtype, tile_nb, tile_nb, -1, newtype);
}

int dplasma_aux_create_upper_type( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb,
                                   dague_remote_dep_datatype_t* newtype )
{
#if defined(USE_MPI)
    int *blocklens, *indices, i, oldsize;
    MPI_Datatype tmp;

    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* UPPER_TILE with the diagonal */
    for( i = 0; i < tile_nb; i++ ) {
        blocklens[i] = i + 1;
        indices[i] = i * tile_nb;
    }
    MPI_Type_indexed(tile_nb, blocklens, indices, oldtype, &tmp);
    MPI_Type_size(oldtype, &oldsize);
    MPI_Type_create_resized(tmp, 0, tile_nb*tile_nb*oldsize, newtype);
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "UPPER %s*%4u", oldtype_name, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
#else
    *newtype = NULL; (void)oldtype; (void)tile_nb;
#endif  /* USE_MPI */
    return 0;
}

int dplasma_aux_create_lower_type( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb,
                                   dague_remote_dep_datatype_t* newtype )
{
#if defined(USE_MPI)
    int *blocklens, *indices, i, oldsize;
    MPI_Datatype tmp;

    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* LOWER_TILE without the diagonal */
    for( i = 0; i < tile_nb-1; i++ ) {
        blocklens[i] = tile_nb - i - 1;
        indices[i] = i * tile_nb + i + 1;
    }
    MPI_Type_indexed(tile_nb-1, blocklens, indices, oldtype, &tmp);
    MPI_Type_size(oldtype, &oldsize);
    MPI_Type_create_resized(tmp, 0, tile_nb*tile_nb*oldsize, newtype);
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "LOWER %s*%4u", oldtype_name, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
#else
    *newtype = NULL; (void)oldtype; (void)tile_nb;
#endif  /* USE_MPI */
    return 0;
}
