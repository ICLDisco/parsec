/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <core_blas.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"

#if defined(HAVE_MPI)

int dplasma_get_extent( MPI_Datatype dt, MPI_Aint* extent )
{
#if defined(HAVE_MPI_20)
    MPI_Aint lb = 0; (void)lb;
    return MPI_Type_get_extent(dt, &lb, extent);
#else
    return MPI_Type_extent( dt, extent);
#endif  /* defined(HAVE_MPI_20) */
}

int dplasma_add2arena_contiguous( dague_arena_t *arena,
                                  size_t elem_size,
                                  size_t alignment,
                                  dague_datatype_t oldtype,
                                  unsigned int nb_elem,
                                  int resized )
{
    dague_datatype_t newtype;
    MPI_Aint extent = 0;

    (void)elem_size;

    dplasma_datatype_define_contiguous(oldtype, nb_elem, resized, &newtype);
    dplasma_get_extent(newtype, &extent);
    dague_arena_construct(arena, extent, alignment, newtype);

    return 0;
}

int dplasma_add2arena_rectangle( dague_arena_t *arena,
                                 size_t elem_size,
                                 size_t alignment,
                                 dague_datatype_t oldtype,
                                 unsigned int tile_mb,
                                 unsigned int tile_nb,
                                 int resized )
{
    dague_datatype_t newtype;
    MPI_Aint extent = 0;

    (void)elem_size;

    dplasma_datatype_define_rectangle(oldtype, tile_mb, tile_nb, resized, &newtype);
    dplasma_get_extent(newtype, &extent);
    dague_arena_construct(arena, extent, alignment, newtype);

    return 0;
}

int dplasma_add2arena_tile( dague_arena_t *arena,
                            size_t elem_size,
                            size_t alignment,
                            dague_datatype_t oldtype,
                            unsigned int tile_mb )
{
    return dplasma_add2arena_rectangle( arena, elem_size, alignment,
                                        oldtype, tile_mb, tile_mb, -1);
}

int dplasma_add2arena_upper( dague_arena_t *arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t oldtype,
                             unsigned int tile_mb,  int diag )
{
    dague_datatype_t newtype;
    MPI_Aint extent = 0;
    (void)elem_size;

    dplasma_datatype_define_upper( oldtype, tile_mb, diag, &newtype);
    dplasma_get_extent(newtype, &extent);
    dague_arena_construct(arena, extent, alignment, newtype);
    return 0;
}

int dplasma_add2arena_lower( dague_arena_t *arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t oldtype,
                             unsigned int tile_mb, int diag )
{
    dague_datatype_t newtype;
    MPI_Aint extent = 0;
    (void)elem_size;

    dplasma_datatype_define_lower( oldtype, tile_mb, diag, &newtype);
    dplasma_get_extent(newtype, &extent);
    dague_arena_construct(arena, extent, alignment, newtype);
    return 0;
}

int dplasma_datatype_define_contiguous( dague_datatype_t oldtype,
                                        unsigned int nb_elem,
                                        int resized,
                                        dague_datatype_t* newtype )
{
    int oldsize;
    /**
     * Define the TILE type.
     */
    MPI_Type_contiguous(nb_elem, oldtype, newtype);
    MPI_Type_size(oldtype, &oldsize);
    if( resized >= 0 ) {
#if defined(HAVE_MPI_20)
        MPI_Datatype tmp = *newtype;
        MPI_Type_create_resized(tmp, 0, resized*oldsize, newtype);
        MPI_Type_free(&tmp);
#else
        int blocklens[] = {1, 1, 1};
        MPI_Aint indices[] = {0, 0, resized*oldsize};
        MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
        MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
#endif  /* defined(HAVE_MPI_20) */
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
    return 0;
}

int dplasma_datatype_define_rectangle( dague_datatype_t oldtype,
                                       unsigned int tile_mb,
                                       unsigned int tile_nb,
                                       int resized,
                                       dague_datatype_t* newtype )
{
    int oldsize;
    /**
     * Define the TILE type.
     */
    MPI_Type_contiguous(tile_nb * tile_mb, oldtype, newtype);
    MPI_Type_size(oldtype, &oldsize);
    if( resized >= 0 ) {
#if defined(HAVE_MPI_20)
        MPI_Datatype tmp = *newtype;
        MPI_Type_create_resized(tmp, 0, resized*oldsize, newtype);
        MPI_Type_free(&tmp);
#else
        int blocklens[] = {1, 1, 1};
        MPI_Aint indices[] = {0, 0, resized*oldsize};
        MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
        MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
#endif  /* defined(HAVE_MPI_20) */
    }
    MPI_Type_commit(newtype);
#if defined(HAVE_MPI_20)
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "RECT %s*%4u*%4u", oldtype_name, tile_mb, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#endif  /* defined(HAVE_MPI_20) */
    return 0;
}

int dplasma_datatype_define_tile( dague_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_datatype_t* newtype )
{
    return dplasma_datatype_define_rectangle(oldtype, tile_nb, tile_nb, -1, newtype);
}

int dplasma_datatype_define_upper( dague_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_datatype_t* newtype )
{
    int *blocklens, *indices, oldsize;
    unsigned int i;
    MPI_Datatype tmp;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* UPPER_TILE with the diagonal */
    for( i = diag; i < tile_nb; i++ ) {
        blocklens[i] = i + 1 - diag;
        indices[i] = i * tile_nb;
    }
    MPI_Type_indexed(tile_nb-diag, blocklens+diag, indices+diag, oldtype, &tmp);
    MPI_Type_size(oldtype, &oldsize);
#if defined(HAVE_MPI_20)
    MPI_Type_create_resized(tmp, 0, tile_nb*tile_nb*oldsize, newtype);
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "UPPER %s*%4u", oldtype_name, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#else
    {
        int blocklens[] = {1, 1, 1};
        MPI_Aint indices[] = {0, 0, tile_nb*tile_nb*oldsize};
        MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
        MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
    }
#endif  /* defined(HAVE_MPI_20) */
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
    return 0;
}

int dplasma_datatype_define_lower( dague_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_datatype_t* newtype )
{
    int *blocklens, *indices, oldsize;
    unsigned int i;
    MPI_Datatype tmp;

    diag = (diag == 0) ? 1 : 0;
    blocklens = (int*)malloc( tile_nb * sizeof(int) );
    indices = (int*)malloc( tile_nb * sizeof(int) );

    /* LOWER_TILE without the diagonal */
    for( i = 0; i < tile_nb-diag; i++ ) {
        blocklens[i] = tile_nb - i - diag;
        indices[i] = i * tile_nb + i + diag;
    }
    MPI_Type_indexed(tile_nb-diag, blocklens, indices, oldtype, &tmp);
    MPI_Type_size(oldtype, &oldsize);
#if defined(HAVE_MPI_20)
    MPI_Type_create_resized(tmp, 0, tile_nb*tile_nb*oldsize, newtype);
    {
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "LOWER %s*%4u", oldtype_name, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }
#else
    {
        int blocklens[] = {1, 1, 1};
        MPI_Aint indices[] = {0, 0, tile_nb*tile_nb*oldsize};
        MPI_Datatype old_types[] = {MPI_LB, oldtype, MPI_UB};
        MPI_Type_struct( 3, blocklens, indices, old_types, newtype );
    }
#endif  /* defined(HAVE_MPI_20) */
    MPI_Type_commit(newtype);
    MPI_Type_free(&tmp);
    free(blocklens);
    free(indices);
    return 0;
}

int dplasma_datatype_undefine_type(dague_datatype_t* type)
{
    return MPI_Type_free(type);
}

#else /* HAVE_MPI */

int dplasma_add2arena_contiguous( dague_arena_t *arena,
                                  size_t elem_size,
                                  size_t alignment,
                                  dague_datatype_t oldtype,
                                  unsigned int nb_elem,
                                  int resized )
{
    (void)oldtype;
    (void)nb_elem;
    (void)resized;

    dague_arena_construct(arena, elem_size, alignment, NULL);
    return 0;
}

int dplasma_add2arena_rectangle( dague_arena_t *arena,
                                 size_t elem_size,
                                 size_t alignment,
                                 dague_datatype_t oldtype,
                                 unsigned int tile_mb,
                                 unsigned int tile_nb,
                                 int resized )
{
    (void)oldtype;
    (void)tile_mb;
    (void)tile_nb;
    (void)resized;

    dague_arena_construct(arena, elem_size, alignment, NULL);
    return 0;
}

int dplasma_add2arena_tile( dague_arena_t *arena,
                            size_t elem_size,
                            size_t alignment,
                            dague_datatype_t oldtype,
                            unsigned int tile_mb )
{
    (void)oldtype;
    (void)tile_mb;

    dague_arena_construct(arena, elem_size, alignment, NULL);
    return 0;
}

int dplasma_add2arena_upper( dague_arena_t *arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t oldtype,
                             unsigned int tile_mb,  int diag )
{
    (void)oldtype;
    (void)tile_mb;
    (void)diag;

    dague_arena_construct(arena, elem_size, alignment, NULL);
    return 0;
}

int dplasma_add2arena_lower( dague_arena_t *arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t oldtype,
                             unsigned int tile_mb,  int diag )
{
    (void)oldtype;
    (void)tile_mb;
    (void)diag;

    dague_arena_construct(arena, elem_size, alignment, NULL);
    return 0;
}

#endif /* HAVE_MPI */

