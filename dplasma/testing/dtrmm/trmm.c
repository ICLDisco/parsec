/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dague.h"
#ifdef USE_MPI
#include "remote_dep.h"
extern dague_arena_t DAGUE_DEFAULT_DATA_TYPE;
#endif  /* defined(USE_MPI) */

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/* Plasma and math libs */
#include <cblas.h>
#include <plasma.h>
#include <lapack.h>
#include <core_blas.h>

#include "timing.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dplasma.h"
#include "testscommon.h"

#define _FMULS(side, M, N) ( side == PlasmaLeft ? ( 0.5 * (N) * (M) * ((M)+1) ) : ( 0.5 * (M) * (N) * ((N)+1) ) )
#define _FADDS(side, M, N) ( side == PlasmaLeft ? ( 0.5 * (N) * (M) * ((M)-1) ) : ( 0.5 * (M) * (N) * ((N)-1) ) )

/*******************************
 * globals and argv set values *
 *******************************/
double time_elapsed;
double sync_time_elapsed;

two_dim_block_cyclic_t ddescA;
two_dim_block_cyclic_t ddescB;
two_dim_block_cyclic_t ddescC;

static dague_object_t *dague_trmm = NULL;

#if defined(USE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* USE_MPI */

int   side[2]  = { PlasmaLeft,    PlasmaRight };
int   uplo[2]  = { PlasmaUpper,   PlasmaLower };
int   diag[2]  = { PlasmaNonUnit, PlasmaUnit  };
int   trans[3] = { PlasmaNoTrans, PlasmaTrans, PlasmaConjTrans };

char *sidestr[2]  = { "Left ", "Right" };
char *uplostr[2]  = { "Upper", "Lower" };
char *diagstr[2]  = { "NonUnit", "Unit   " };
char *transstr[3] = { "N", "T", "H" };

/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                          double alpha, two_dim_block_cyclic_t *ddescA, two_dim_block_cyclic_t *ddescB, two_dim_block_cyclic_t *ddescC )
{
    int info_solution;
    double Anorm, Binitnorm, Bdaguenorm, Blapacknorm, Rnorm, result;
    double *A, *B, *C;
    int M = ddescB->super.m;
    int N = ddescB->super.n;
    int LDA = ddescA->super.lm;
    int LDB = ddescB->super.lm;
    double eps = lapack_dlamch(lapack_eps);
    double *work = (double *)malloc(max(M, N)* sizeof(double));
    int Am, An;
    double mdone = (double)-1.0;

    M = ddescB->super.m;
    N = ddescB->super.n;
    if (side == PlasmaLeft) {
        Am = M; An = M;
    } else {
        Am = N; An = N;
    }

    A = (double *)malloc((ddescA->super.mt)*(ddescA->super.nt)*(ddescA->super.bsiz)*sizeof(double));
    B = (double *)malloc((ddescB->super.mt)*(ddescB->super.nt)*(ddescB->super.bsiz)*sizeof(double));
    C = (double *)malloc((ddescC->super.mt)*(ddescC->super.nt)*(ddescC->super.bsiz)*sizeof(double));

    twoDBC_to_lapack_double( ddescA, A, LDA );
    twoDBC_to_lapack_double( ddescB, B, LDB );
    twoDBC_to_lapack_double( ddescC, C, LDB );

    Anorm      = lapack_dlantr( lapack_inf_norm, (enum lapack_uplo_type)uplo, (enum lapack_diag_type)diag, Am, An, A, LDA, work );
    Binitnorm  = lapack_dlange( lapack_inf_norm, M, N, B, LDB, work );
    Bdaguenorm = lapack_dlange( lapack_inf_norm, M, N, C, LDB, work );

    cblas_dtrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                (CBLAS_DIAG)diag, M, N, alpha, A, LDA, B, LDB);
    Blapacknorm = lapack_dlange(lapack_inf_norm, M, N, B, LDB, work);

    cblas_daxpy(LDB * N, mdone, C, 1, B, 1);
    Rnorm = lapack_dlange(lapack_inf_norm, M, N, B, LDB, work);

    if (getenv("PLASMA_TESTING_VERBOSE"))
	printf("Rnorm %e, Anorm %e, Binitnorm %e, Bdaguenorm %e, Blapacknorm %e\n",
	       Rnorm, Anorm, Binitnorm, Bdaguenorm, Blapacknorm);

    result = Rnorm / ((Anorm + Blapacknorm) * max(M,N) * eps);
    if (  isinf(Blapacknorm) || isinf(Bdaguenorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
	printf("-- The solution is suspicious ! \n");
	info_solution = 1;
    }
    else{
        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }
    
    free(work);
    free(A);
    free(B);
    free(C);

    return info_solution;
}

int main(int argc, char ** argv)
{
    int iparam[IPARAM_INBPARAM];
    double gflops;
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
    int LDA   = iparam[IPARAM_LDA];
    int NRHS  = iparam[IPARAM_NRHS];
    int LDB   = iparam[IPARAM_LDB];
    int nrst  = iparam[IPARAM_STM];
    int ncst  = iparam[IPARAM_STN];
    int GRIDrows = iparam[IPARAM_GDROW];

    /* Initialize DAGuE */
    TIME_START();
    dague = setup_dague(&argc, &argv, iparam, PlasmaRealDouble);
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));
	
    if ( iparam[IPARAM_CHECK] == 0 ) {
	int s = PlasmaLeft;

	/* initializing matrix structure */
	two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, MB, NB, 0, M, N,    0, 0, LDA, N,    nrst, ncst, GRIDrows);
	two_dim_block_cyclic_init(&ddescB, matrix_RealDouble, nodes, cores, rank, MB, NB, 0, M, NRHS, 0, 0, LDB, NRHS, nrst, ncst, GRIDrows);
	
	/* matrix generation */
	printf("Generate matrices ... ");
	generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
	generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
	printf("Done\n");
	
	/* Create TRMM DAGuE */
	printf("Generate TRMM DAG ... ");
	SYNC_TIME_START();
	dague_trmm = DAGUE_dtrmm_New(s, PlasmaLower, PlasmaNoTrans, PlasmaUnit, (double)1.0, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescB);
	dague_enqueue( dague, (dague_object_t*)dague_trmm);
	printf("Done\n");
	printf("Total nb tasks to run: %u\n", dague->taskstodo);
	
	/* lets rock! */
	SYNC_TIME_START();
	TIME_START();
	dague_progress(dague);
	TIME_PRINT(("Dague proc %d:\ttasks: %u\t%f task/s\n", 
		    rank, dague_trmm->nb_local_tasks, 
		    dague_trmm->nb_local_tasks/time_elapsed));
	SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB, 
			 gflops = (_FADDS(s, M, NRHS) + _FMULS(s, M, NRHS))/(sync_time_elapsed)));
	(void) gflops;
	TIME_PRINT(("Dague priority change at position \t%u\n", ddescA.super.nt - iparam[IPARAM_PRIORITY]));

	twoDBC_free(&ddescA);
	twoDBC_free(&ddescB);
    }
    else {
	int s, u, t, d;
	int info_solution;
	double alpha = 1.0;

	for (s=0; s<2; s++) {
	    for (u=0; u<2; u++) {
/* #ifdef COMPLEX */
/* 		for (t=0; t<3; t++) { */
/* #else */
		for (t=0; t<2; t++) {
/*#endif*/
		    for (d=0; d<2; d++) {

			printf("***************************************************\n");
			printf(" ----- TESTING DTRMM (%s, %s, %s, %s) -------- \n",
				   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
			/* initializing matrix structure */
			two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, MB, NB, 0, M, N,    0, 0, LDA, N,    nrst, ncst, GRIDrows);
			two_dim_block_cyclic_init(&ddescB, matrix_RealDouble, nodes, cores, rank, MB, NB, 0, M, NRHS, 0, 0, LDB, NRHS, nrst, ncst, GRIDrows);
			two_dim_block_cyclic_init(&ddescC, matrix_RealDouble, nodes, cores, rank, MB, NB, 0, M, NRHS, 0, 0, LDB, NRHS, nrst, ncst, GRIDrows);
			
			/* matrix generation */
			printf("Generate matrices ... ");
			generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
			generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
			generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescC, 200);
			printf("Done\n");
	
			/* Create TRMM DAGuE */
			SYNC_TIME_START();
			printf("Generate TRMM DAG ... ");
			dague_trmm = DAGUE_dtrmm_New(side[s], uplo[u], trans[t], diag[d], 
						     (double)alpha, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescC);
			dague_enqueue( dague, (dague_object_t*)dague_trmm);
			printf("Done\n");
			printf("Total nb tasks to run: %u\n", dague->taskstodo);
			TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));
			
			/* lets rock! */
			SYNC_TIME_START();
			TIME_START();
			dague_progress(dague);
			TIME_PRINT(("Dague proc %d:\ttasks: %u\t%f task/s\n", rank, 
				    dague_trmm->nb_local_tasks, 
				    dague_trmm->nb_local_tasks/time_elapsed));
			SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB, 
					 gflops = (_FADDS(side[s], M, NRHS) + _FMULS(side[s], M, NRHS))/(sync_time_elapsed)));
			(void) gflops;

			/* Check the solution */
			info_solution = check_solution(side[s], uplo[u], trans[t], diag[d],
						       alpha, &ddescA, &ddescB, &ddescC);
			
			if (info_solution == 0) {
			    printf(" ---- TESTING DTRMM (%s, %s, %s, %s) ...... PASSED !\n",
				   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
			}
			else {
			    printf(" ---- TESTING DTRMM (%s, %s, %s, %s) ... FAILED !\n",
				   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
			}
			printf("***************************************************\n");


			twoDBC_free(&ddescA);
			twoDBC_free(&ddescB);
			twoDBC_free(&ddescC);
		    }
		}
#ifdef __UNUSED__
		}
#endif
	    }
	}
    }
    
    /*data_dump((tiled_matrix_desc_t *) &ddescA);*/
    cleanup_dague(dague, "dtrmm");
    /*** END OF DAGUE COMPUTATION ***/
    
    runtime_fini();
    return 0;
}
