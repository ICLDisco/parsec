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

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

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

    if ( M != N && check ) {
        fprintf(stderr, "Check cannot be perfomed with M != N\n");
        check = 0;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, cores, rank, MB, 1, MB*P, dague_imin(MT, NT), 0, 0,
                               MB*P, dague_imin(MT, NT), SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

/*     PASTE_CODE_ALLOCATE_MATRIX(ddescAl, check, */
/*         two_dim_block_cyclic, (&ddescAl, matrix_ComplexDouble, matrix_Lapack, */
/*                                1, cores, rank, MB, NB, LDA, N, 0, 0, */
/*                                M, N, SMB, SNB, 1)); */

/*     PASTE_CODE_ALLOCATE_MATRIX(ddescIPIVl, check, */
/*         two_dim_block_cyclic, (&ddescIPIVl, matrix_Integer, matrix_Lapack, */
/*                                1, cores, rank, MB, 1, MB*P, dague_imin(MT, NT), 0, 0, */
/*                                MB*P, dague_imin(MT, NT), SMB, SNB, 1)); */

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 7657);

    /* Increase diagonale to avoid pivoting */
    {
        tiled_matrix_desc_t *descA = (tiled_matrix_desc_t *)&ddescA;
        int minmnt = dague_imin( descA->mt, descA->nt );
        int minmn  = dague_imin( descA->m,  descA->n );
        int t, e;

        for(t = 0; t < minmnt; t++ ) {
	  if(((dague_ddesc_t*) &ddescA)->rank_of(((dague_ddesc_t*) &ddescA), t, t)  == ((dague_ddesc_t*) &ddescA)->myrank)
	    {
	      Dague_Complex64_t *tab = ((dague_ddesc_t*) &ddescA)->data_of(((dague_ddesc_t*) &ddescA), t, t);
	      for(e = 0; e < descA->mb; e++)
                tab[e * descA->mb + e] += (Dague_Complex64_t)minmn;
	    }
        }
    }

    if ( check )
    {
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescA0 );
/*         dplasma_zlacpy( dague, PlasmaUpperLower, */
/*                         (tiled_matrix_desc_t *)&ddescA, */
/*                         (tiled_matrix_desc_t *)&ddescAl ); */
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,
                        (tiled_matrix_desc_t *)&ddescX );
    }
    if(loud > 2) printf("Done\n");

    /* Create DAGuE */
    if(loud > 2) printf("+++ Computing getrf ... ");
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_fusion,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescIPIV,
                               P,
                               Q,
                               &info));
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgetrf_fusion);
    dplasma_zgetrf_fusion_Destruct( DAGUE_zgetrf_fusion );
    if(loud > 2) printf("Done.\n");

    if ( check && info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
/*       if( (((dague_ddesc_t*) &ddescA)->myrank) == 0) */
/* 	LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, M, N,  */
/* 			    (Dague_Complex64_t*)(ddescAl.mat), MB*MT,  */
/* 			    (int *)(ddescIPIVl.mat)); */
/*       int i, j, it, jt; */
/*       printf("LU decomposition of A with Lapack\n"); */
/*       for(it=0; it<MT; it++) */
/* 	{ */
/* 	  int tempkm = ((it)==(MT-1)) ? (M-(it*MB)) : (MB); */
/*           for(i=0; i<tempkm; i++) { */
/*             printf("%d:\t",i+it*MB); */

/*             for(jt=0; jt<NT; jt++) */
/*               { */
/*                 int tempkn = ((jt)==(NT-1)) ? (N-(jt*NB)) : (NB); */
/*                 if(((dague_ddesc_t*) &ddescAl)->rank_of(((dague_ddesc_t*) &ddescAl), it, jt)  == ((dague_ddesc_t*) &ddescAl)->myrank) { */
/*                   Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescAl)->data_of(((dague_ddesc_t*) &ddescAl), it, jt); */
/*                   for(j=0; j<tempkn; j++) */
/*                     printf("%e\t",mat[MT*MB*j+i]); */
/*                 } */
/*               } */
/*             printf("\n"); */
/*           } */
/*         } */

/*       printf("LU decomposition of A with DPLASMA\n"); */
/*       for(it=0; it<MT; it++) */
/* 	{ */
/* 	  int tempkm = ((it)==(MT-1)) ? (M-(it*MB)) : (MB); */
/*           for(i=0; i<tempkm; i++) { */
/*             printf("%d:\t",i+it*MB); */

/*             for(jt=0; jt<NT; jt++) */
/*               { */
/*                 int tempkn = ((jt)==(NT-1)) ? (N-(jt*NB)) : (NB); */
/*                 if(((dague_ddesc_t*) &ddescAl)->rank_of(((dague_ddesc_t*) &ddescA), it, jt)  == ((dague_ddesc_t*) &ddescA)->myrank) { */
/*                   Dague_Complex64_t *mat = ((dague_ddesc_t*) &ddescA)->data_of(((dague_ddesc_t*) &ddescA), it, jt); */
/*                   for(j=0; j<tempkn; j++) */
/*                     printf("%e\t",mat[MB*j+i]); */
/*                 } */
/* 	    } */
/*             printf("\n"); */
/*           } */
/*         } */

        dplasma_ztrsm(dague, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                      1.0, (tiled_matrix_desc_t *)&ddescA,
                           (tiled_matrix_desc_t *)&ddescX);
        dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      1.0, (tiled_matrix_desc_t *)&ddescA,
                           (tiled_matrix_desc_t *)&ddescX);

        /* Check the solution */
        ret |= check_solution( dague, (rank == 0) ? loud : 0,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);
    }

    if ( check ) {
        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0);
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    }

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    return ret;
}



static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = ddescB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
