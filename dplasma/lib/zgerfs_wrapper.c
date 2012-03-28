/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <lapacke.h>

#include "zgerfs.h"



dague_object_t*
dplasma_zgerfs_New(tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t* LU,
                   tiled_matrix_desc_t* B,
                   tiled_matrix_desc_t* R,
                   tiled_matrix_desc_t* X,
                   tiled_matrix_desc_t* Z)
{
    dague_object_t *dague_zgerfs = NULL;

    dague_zgerfs = (dague_object_t*)dague_zgerfs_new(*A,  (dague_ddesc_t*)A,
                                                     *LU, (dague_ddesc_t*)LU,
                                                     *B,  (dague_ddesc_t*)B,
                                                     *R,  (dague_ddesc_t*)R,
                                                     *X,  (dague_ddesc_t*)X,
                                                     *Z,  (dague_ddesc_t*)Z);

    dplasma_add2arena_tile(((dague_zgerfs_object_t*)dague_zgerfs)->arenas[DAGUE_zgerfs_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zgerfs;
}

void
dplasma_zgerfs_Destruct( dague_object_t *o )
{
    dague_zgerfs_object_t *dague_zgerfs = (dague_zgerfs_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zgerfs->arenas[DAGUE_zgerfs_DEFAULT_ARENA]->opaque_dtt) );
    dague_zgerfs_destroy(dague_zgerfs);
}

int dplasma_zgerfs_aux( dague_context_t     *dague,
                        tiled_matrix_desc_t *ddescA,
                        tiled_matrix_desc_t *ddescLU,
                        tiled_matrix_desc_t *ddescB,
                        tiled_matrix_desc_t *ddescR,
                        tiled_matrix_desc_t *ddescX,
                        tiled_matrix_desc_t *ddescZ)
{
    dague_object_t *dague_zgerfs = NULL;

    dague_zgerfs = dplasma_zgerfs_New(ddescA, ddescLU, ddescB, ddescR, ddescX, ddescZ);

    if ( dague_zgerfs != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgerfs);
        dplasma_progress(dague);
        dplasma_zgerfs_Destruct( dague_zgerfs );
    }

    return 0;
}


int dplasma_zgerfs( dague_context_t     *dague,
                    tiled_matrix_desc_t *ddescA,
                    tiled_matrix_desc_t *ddescLU,
                    tiled_matrix_desc_t *ddescB,
                    tiled_matrix_desc_t *ddescX)
{
    two_dim_block_cyclic_t ddescR, ddescZ;

    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        ddescR, two_dim_block_cyclic,
        (&ddescR, matrix_ComplexDouble, matrix_Tile, ddescB->super.nodes, ddescB->super.cores, ddescB->super.myrank,
         ddescB->mb, ddescB->nb, ddescB->mt, ddescB->nt, 0, 0, ddescB->m, ddescB->n,
         ((two_dim_block_cyclic_t*)ddescB)->grid.strows,
         ((two_dim_block_cyclic_t*)ddescB)->grid.stcols,
         ((two_dim_block_cyclic_t*)ddescB)->grid.rows));

    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        ddescZ, two_dim_block_cyclic,
        (&ddescZ, matrix_ComplexDouble, matrix_Tile, ddescB->super.nodes, ddescB->super.cores, ddescB->super.myrank,
         ddescB->mb, ddescB->nb, ddescB->mt, ddescB->nt, 0, 0, ddescB->m, ddescB->n,
         ((two_dim_block_cyclic_t*)ddescB)->grid.strows,
         ((two_dim_block_cyclic_t*)ddescB)->grid.stcols,
         ((two_dim_block_cyclic_t*)ddescB)->grid.rows));

    double eps = LAPACKE_dlamch_work('e');
    double Anorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescA);
    double Bnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescB);
    double Xnorm, Rnorm, Znorm;

    int nb_iter_ref = 0;
    int m = ddescB->m;
    double result;
    do
    {
        dplasma_zgerfs_aux(dague, ddescA, ddescLU, ddescB, (tiled_matrix_desc_t*) &ddescR, ddescX, (tiled_matrix_desc_t*) &ddescZ);
        Rnorm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t *)&ddescR);
        Xnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescX);
        Znorm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t *)&ddescZ);

        result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

        nb_iter_ref++;
         printf("Iter ref %d: ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", nb_iter_ref,result);
    }
    while(  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (( ( Anorm * Xnorm + Bnorm ) * m * eps ) < Rnorm) );

    dague_data_free(ddescR.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescR);
    dague_data_free(ddescZ.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescZ);

    printf("Solution refined in %d iterations\n", nb_iter_ref );

    return 0;
}
