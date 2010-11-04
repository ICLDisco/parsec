/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"

#if defined(HAVE_MPI)
int dplasma_datatype_define_rectangle( dague_remote_dep_datatype_t oldtype,
                                       unsigned int tile_mb,
                                       unsigned int tile_nb,
                                       int resized,
                                       dague_remote_dep_datatype_t* newtype )
{
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
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "TILE %s*%4u*%4u", oldtype_name, tile_mb, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
    return 0;
}

int dplasma_datatype_define_tile( dague_remote_dep_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_remote_dep_datatype_t* newtype )
{
    return dplasma_datatype_define_rectangle(oldtype, tile_nb, tile_nb, -1, newtype);
}

int dplasma_datatype_define_upper( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_remote_dep_datatype_t* newtype )
{
    int *blocklens, *indices, oldsize;
    unsigned int i;
    MPI_Datatype tmp;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* UPPER_TILE with the diagonal */
    for( i = 0; i < tile_nb; i++ ) {
        blocklens[i] = i + diag;
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
    return 0;
}

int dplasma_datatype_define_lower( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_remote_dep_datatype_t* newtype )
{
    int *blocklens, *indices, oldsize;
    unsigned int i;
    MPI_Datatype tmp;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* LOWER_TILE without the diagonal */
    for( i = 0; i < tile_nb-1; i++ ) {
        blocklens[i] = tile_nb - i - diag;
        indices[i] = i * tile_nb + i + diag;
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
    return 0;
}
#endif 

