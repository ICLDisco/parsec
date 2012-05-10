/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };


int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N))

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, cores, rank, MB, 1, MB*P, dague_imin(MT, NT), 0, 0,
                               MB*P, dague_imin(MT, NT), SMB, SNB, P));


    PASTE_CODE_ALLOCATE_MATRIX(ddescAl, check,
        two_dim_block_cyclic, (&ddescAl, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescLUl, check,
        two_dim_block_cyclic, (&ddescLUl, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, 1));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIVl, check,
        two_dim_block_cyclic, (&ddescIPIVl, matrix_Integer, matrix_Lapack,
                               1, cores, rank, MB, 1, MB*P, dague_imin(MT, NT), 0, 0,
                               MB*P, dague_imin(MT, NT), SMB, SNB, 1));


    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 7657);


    if ( check )
    {
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescAl );
	if(P == 1 && Q == 1) {
	  int i,j,k;
	  printf("Matrix A(%d,%d) with Lapack\n",MB,LDA);
/* 	  for(k=0; k<MT; k++) */
	    {
/* 	      int tempkm = ((k)==(MT-1)) ? (M-(k*MB)) : (MB); */
/* 	      int tempkn = ((k)==(NT-1)) ? (N-(k*NB)) : (NB); */
	      if(((dague_ddesc_t*) &ddescAl)->rank_of(((dague_ddesc_t*) &ddescAl), 0, 0)  == ((dague_ddesc_t*) &ddescAl)->myrank) {
		Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescAl)->data_of(((dague_ddesc_t*) &ddescAl), 0, 0);
		for(i=0; i<M; i++) {
		  printf("%d:\t",i);
		  for(j=0; j<N; j++)
		    printf("%e\t",mat[MB*j+i]);
		  printf("\n");
		}
	      }
	    }

	  printf("Matrix A with DPLASMA\n");
	  for(k=0; k<MT; k++)
	    {
	      int tempkm = ((k)==(MT-1)) ? (M-(k*MB)) : (MB);
	      int tempkn = ((k)==(NT-1)) ? (N-(k*NB)) : (NB);
	      if(((dague_ddesc_t*) &ddescA)->rank_of(((dague_ddesc_t*) &ddescA), k, 0)  == ((dague_ddesc_t*) &ddescA)->myrank) {
		Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescA)->data_of(((dague_ddesc_t*) &ddescA), k, 0);
		for(i=0; i<tempkm; i++) {
		  printf("%d:\t",i+k*MB);
		  for(j=0; j<tempkn; j++)
		    printf("%e\t",mat[MB*j+i]);
		  printf("\n");
		}
	      }
	    }
	}
    }
    if(loud > 2) printf("Done\n");

    /* Create DAGuE */
    if(loud > 2) printf("+++ Computing getrf ... ");
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_panel,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescIPIV,
                               P,
                               Q,
                               &info));
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgetrf_panel);
    dplasma_zgetrf_panel_Destruct( DAGUE_zgetrf_panel );


    if(loud > 2) printf("Done.\n");

    if ( check && info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
      
      if( (((dague_ddesc_t*) &ddescA)->myrank) == 0)
	LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, M, N, 
			    (Dague_Complex64_t*)(ddescAl.mat), MB*MT, 
			    (int *)(ddescIPIVl.mat));
      /* Check ipiv */
      int success = 1;
      int i,j;
      for(i=0; i<dague_imin(MT, NT); i++)
	{
	  if(((dague_ddesc_t*) &ddescIPIV)->rank_of(((dague_ddesc_t*) &ddescIPIV), 0, i)  == ((dague_ddesc_t*) &ddescIPIV)->myrank)
	    for(j=0; j<MB; j++) {
	      int *ipiv = ((dague_ddesc_t*) &ddescIPIV)->data_of(((dague_ddesc_t*) &ddescIPIV), 0, i);
	      if( ipiv[j] != ((int *)(ddescIPIVl.mat))[j+i*MB] ) {
		fprintf(stderr, "\nDPLASMA (ipiv[%d] = %d) / LAPACK (ipiv[%d] = %d)\n",
			j, ipiv[j],
			j+i*MB, ((int *)(ddescIPIVl.mat))[j+i*MB]); 
		success = 0;
		break;
	      }/* else */
/* 		printf("\nDPLASMA (ipiv[%d] = %d)\n", */
/* 		       j, ipiv[j]); */
	    }
	}   
   
/*       int k; */
/*       printf("LU decomposition of A with Lapack\n"); */
/*       for(k=0; k<MT; k++) */
/* 	{ */
/* 	  int tempkm = ((k)==(MT-1)) ? (M-(k*MB)) : (MB); */
/* 	  int tempkn = ((k)==(NT-1)) ? (N-(k*NB)) : (NB); */
/* 	  if(((dague_ddesc_t*) &ddescAl)->rank_of(((dague_ddesc_t*) &ddescAl), k, 0)  == ((dague_ddesc_t*) &ddescAl)->myrank) { */
/* 	    Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescAl)->data_of(((dague_ddesc_t*) &ddescAl), k, 0); */
/* 	    for(i=0; i<tempkm; i++) { */
/* 	      printf("%d:\t",i+k*MB); */
/* 	      for(j=0; j<tempkn; j++) */
/* 		printf("%e\t",mat[LDA*j+i]); */
/* 	      printf("\n"); */
/* 	    } */
/* 	  } */
/* 	} */

/*       printf("LU decomposition of A with DPLASMA\n"); */
/*       for(k=0; k<MT; k++) */
/* 	{ */
/* 	  int tempkm = ((k)==(MT-1)) ? (M-(k*MB)) : (MB); */
/* 	  int tempkn = ((k)==(NT-1)) ? (N-(k*NB)) : (NB); */
/* 	  if(((dague_ddesc_t*) &ddescA)->rank_of(((dague_ddesc_t*) &ddescA), k, 0)  == ((dague_ddesc_t*) &ddescA)->myrank) { */
/* 	    Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescA)->data_of(((dague_ddesc_t*) &ddescA), k, 0); */
/* 	    for(i=0; i<tempkm; i++) { */
/* 	      printf("%d:\t",i+k*MB); */
/* 	      for(j=0; j<tempkn; j++) */
/* 		printf("%e\t",mat[MB*j+i]); */
/* 	      printf("\n"); */
/* 	      } */
/* 	    } */
/* 	} */


      if(success) {
	dplasma_zlacpy( dague, PlasmaUpperLower,
			(tiled_matrix_desc_t *)&ddescA,
			(tiled_matrix_desc_t *)&ddescLUl );
	double alpha = -1.;
	dplasma_zgeadd(dague, PlasmaUpperLower,  CBLAS_SADDR(alpha), (tiled_matrix_desc_t *)&ddescLUl, (tiled_matrix_desc_t *)&ddescAl);
	double norm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t *)&ddescAl);
	if( (((dague_ddesc_t*) &ddescAl)->myrank) == 0)
	  printf("The norm is %e\n",norm);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      if((((dague_ddesc_t*) &ddescA)->myrank) == 0) {
	dague_data_free(ddescAl.mat);
	dague_data_free(ddescLUl.mat);
	dague_data_free(ddescIPIVl.mat);
      }
      dague_ddesc_destroy( (dague_ddesc_t*)&ddescAl);
      dague_ddesc_destroy( (dague_ddesc_t*)&ddescLUl);
      dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIVl);

    }

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    cleanup_dague(dague, iparam);

    return ret;
}
