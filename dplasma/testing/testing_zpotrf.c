/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dague.h"

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <lapacke.h>

#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#include "dplasma.h"

#include "testscommon.h"
#include "timing.h"

#define _FMULS_POTRF(N) ((N) * (1.0 / 6.0 * (N) + 0.5) * (N))
#define _FADDS_POTRF(N) ((N) * (1.0 / 6.0 * (N)      ) * (N))

#define _FMULS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) + 1.) ) )
#define _FADDS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) - 1.) ) )

#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
#include "gpu_data.h"
#include "cuda_sgemm.h"
#endif

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

int main(int argc, char ** argv)
{
    int iparam[IPARAM_INBPARAM];
    DagDouble_t flops;
    DagDouble_t gflops;
    dague_context_t* dague;

    sym_two_dim_block_cyclic_t ddescA;
    dague_object_t *dague_zpotrf = NULL;

    /* parsing arguments */
    runtime_init(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int N     = iparam[IPARAM_N];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int LDA   = iparam[IPARAM_LDA];
    //int NRHS  = iparam[IPARAM_NRHS];
    int GRIDrows = iparam[IPARAM_GDROW];
    int info;

    //#ifdef VTRACE 
    //    VT_ON();
    //#endif
    
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
    if( iparam[IPARAM_NGPUS] > 0 ) {
        if( 0 != dague_gpu_init( &iparam[IPARAM_NGPUS], 0 ) ) {
            fprintf(stderr, "Unable to initialize the CUDA environment.\n");
            exit(1);
        }
    }
#endif

    /* initializing matrix structure */
    sym_two_dim_block_cyclic_init(&ddescA, matrix_ComplexDouble, nodes, cores, rank,
                                  MB, NB, N, N, 0, 0, LDA, N, GRIDrows);
#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
#endif
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);

    /* Initialize DAGuE */
    TIME_START();
    dague = setup_dague(&argc, &argv, iparam);
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));

    
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
    if( iparam[IPARAM_NGPUS] > 0 ) {
        if( 0 != spotrf_cuda_init( (tiled_matrix_desc_t *) &ddescA ) ) {
            fprintf(stderr, "Unable to load GEMM operations.\n");
            exit(1);
        }
    }
#endif

    if ( iparam[IPARAM_CHECK] == 0 ) {
        PLASMA_enum uplo = PlasmaLower;
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
        flops = 2.*_FADDS_POTRF(N) + 6.*_FMULS_POTRF(N);
#else
        flops = _FADDS_POTRF(N) + _FMULS_POTRF(N);
#endif

        /* matrix generation */
        printf("Generate matrices ... ");
        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
        printf("Done\n");
        
        /* Create TRMM DAGuE */
        printf("Generate ZPOTRF DAG ... ");
        SYNC_TIME_START();
#if defined(LLT_LL)
        dague_zpotrf = dplasma_zpotrf_ll_New(uplo, (tiled_matrix_desc_t*)&ddescA, &info);
#else
        dague_zpotrf = dplasma_zpotrf_rl_New(uplo, (tiled_matrix_desc_t*)&ddescA, &info);
#endif
        dague_enqueue( dague, (dague_object_t*)dague_zpotrf);
        printf("Done\n");
        printf("Total nb tasks to run: %u\n", dague->taskstodo);

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        TIME_PRINT(("Dague proc %d:\ttasks: %u\t%f task/s\n",
                    rank, dague_zpotrf->nb_local_tasks,
                    dague_zpotrf->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB,
                         gflops = flops/(sync_time_elapsed)));
        (void) gflops;
        TIME_PRINT(("Dague priority change at position \t%u\n", ddescA.super.nt - iparam[IPARAM_PRIORITY]));
    }
#if 0 /* Future check */
    else {

    }
#else
    /* Old checking by comparison: will be remove because doesn't check anything */
    if ( iparam[IPARAM_CHECK] == 1 ) {
        char fname[20];
        sprintf(fname , "zposv_r%d", rank );
        printf("writing matrix to file\n");
        data_write((tiled_matrix_desc_t *) &ddescA, fname);
    } 
    else if( iparam[IPARAM_CHECK] == 2 ) {
        char fname[20];
        sym_two_dim_block_cyclic_t ddescB;

        sym_two_dim_block_cyclic_init(&ddescB, matrix_ComplexDouble, nodes, cores, rank,
                                      MB, NB, N, N, 0, 0, LDA, N, GRIDrows);
        ddescB.mat = dague_data_allocate((size_t)ddescB.super.nb_local_tiles * (size_t)ddescB.super.bsiz * (size_t)ddescB.super.mtype);
#if defined(DAGUE_PROFILING)
        ddescB.super.super.key = strdup("B");
#endif

        sprintf(fname , "sposv_r%d", rank );
        printf("reading matrix from file\n");
        data_read((tiled_matrix_desc_t *) &ddescB, fname);
        
        matrix_scompare_dist_data((tiled_matrix_desc_t *) &ddescA, (tiled_matrix_desc_t *) &ddescB);

        dague_data_free(ddescB.mat);
#if defined(DAGUE_PROFILING)
        free(ddescB.super.super.key);
#endif
    }
#endif

    dague_data_free(ddescA.mat);
#if defined(DAGUE_PROFILING)
        free(ddescA.super.super.key);
#endif

    cleanup_dague(dague, "zpotrf");
    /*** END OF DAGUE COMPUTATION ***/

    runtime_fini();
    return EXIT_SUCCESS;
}
