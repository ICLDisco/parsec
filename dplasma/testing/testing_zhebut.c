/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "dplasma/cores/cuda_sgemm.h"
#endif

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;
    int uplo = PlasmaLower;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    /* TODO: compute flops for butterfly */
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDA, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, 1358);
    dplasma_zplrnt( dague,
                    (tiled_matrix_desc_t *)&ddescX, 3872);
    if(loud > 2) printf("Done\n");

    if(loud > 2) printf("+++ Computing Butterfly ... ");

    SYNC_TIME_START();
    TIME_START();
    dplasma_zhebut(dague, (tiled_matrix_desc_t *)&ddescA, butterfly_level);
    dplasma_zhetrf(dague, (tiled_matrix_desc_t *)&ddescA);
    dplasma_zhetrs(dague, (const tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescX);

    fprintf(stderr,"-- DONE\n");

    if(loud)
        TIME_PRINT(rank, ("zhebut computed %d tasks,\trate %f task/s\n",
                   nb_local_tasks,
                   nb_local_tasks/time_elapsed));
    SYNC_TIME_PRINT(rank, ("zhebut computation N= %d NB= %d : %f gflops\n", N, NB,
                    gflops = (flops/1e9)/(sync_time_elapsed)));
    (void)gflops;

    if(loud > 2) printf("Done.\n");

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- The butterfly failed (info = %d) ! \n", info);
        ret |= 1;
    }

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescA);

    return ret;
}

