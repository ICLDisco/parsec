/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include "dague.h"
#ifdef USE_MPI
#include "remote_dep.h"
#include <mpi.h>
#endif  /* defined(USE_MPI) */

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
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dplasma.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

#include "testscommon.h"
#include "timing.h"

#define _FMULS_GETRF(M, N) ( 0.5 * (DagDouble_t)(N) * ( (DagDouble_t)(N) * ((DagDouble_t)(M) - (1. / 3.) * (DagDouble_t)(N)) - (DagDouble_t)(N) ) )
#define _FADDS_GETRF(M, N) ( 0.5 * (DagDouble_t)(N) * ( (DagDouble_t)(N) * ((DagDouble_t)(M) - (1. / 3.) * (DagDouble_t)(N))                    ) )

int main(int argc, char ** argv)
{
    int iparam[IPARAM_INBPARAM];
    DagDouble_t flops;
    DagDouble_t gflops;
    dague_context_t* dague;

    /* parsing arguments */
    runtime_init(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int IB    = iparam[IPARAM_IB];
    int LDA   = iparam[IPARAM_LDA];
    int nrst  = iparam[IPARAM_STM];
    int ncst  = iparam[IPARAM_STN];
    int GRIDrows = iparam[IPARAM_GDROW];
    int mt = (M%MB==0) ? (M/MB) : (M/MB+1);
    int info;

    two_dim_block_cyclic_t ddescA;
    two_dim_block_cyclic_t ddescLIPIV;

    dague_object_t *dague_zgetrf = NULL;
    
    /* initializing matrix structure */
    two_dim_block_cyclic_init(&ddescA,     matrix_ComplexDouble, nodes, cores, rank, MB,   NB, M,         N, 0, 0, LDA,       N, nrst, ncst, GRIDrows);
    /* In each tile we store IPIV followed by L */
    two_dim_block_cyclic_init(&ddescLIPIV, matrix_ComplexDouble, nodes, cores, rank, IB+1, NB, mt*(IB+1), N, 0, 0, mt*(IB+1), N, nrst, ncst, GRIDrows);

    ddescA.mat     = dague_data_allocate((size_t)ddescA.super.nb_local_tiles    *(size_t)ddescA.super.bsiz    *(size_t)ddescA.super.mtype    );
    ddescLIPIV.mat = dague_data_allocate((size_t)ddescLIPIV.super.nb_local_tiles*(size_t)ddescLIPIV.super.bsiz*(size_t)ddescLIPIV.super.mtype);

    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescLIPIV);

    /* Initialize DAGuE */
    TIME_START();
    dague = setup_dague(&argc, &argv, iparam);
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));

    if ( iparam[IPARAM_CHECK] == 0 ) {
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
        flops = 2.*_FADDS_GETRF(M, N) + 6.*_FMULS_GETRF(M, N);
#else
        flops = _FADDS_GETRF(M, N) + _FMULS_GETRF(M, N);
#endif
	
	/* Create GETRF DAGuE */
        printf("Generate GETRF DAG ... ");
        SYNC_TIME_START();
	dague_zgetrf = (dague_object_t*)dplasma_zgetrf_sd_New( (tiled_matrix_desc_t*)&ddescA,
							       (tiled_matrix_desc_t*)&ddescLIPIV,
							       &info );
	dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
        printf("Done\n");
        printf("Total nb tasks to run: %u\n", dague->taskstodo);

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        TIME_PRINT(("Dague proc %d:\ttasks: %u\t%f task/s\n",
                    rank, dague_zgetrf->nb_local_tasks, dague_zgetrf->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB,
                         gflops = flops/(sync_time_elapsed)));
        (void) gflops;
        TIME_PRINT(("Dague priority change at position \t%u\n", ddescA.super.nt - iparam[IPARAM_PRIORITY]));
    }
    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescLIPIV.mat);

    cleanup_dague(dague, "zgetrf");
    /*** END OF DAGUE COMPUTATION ***/

    runtime_fini();
    return EXIT_SUCCESS;
}
