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
#if defined(HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
#endif

//#define MYDEBUG 1
static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int dplasma_iprint( dague_context_t *dague,
                    PLASMA_enum uplo,
                    tiled_matrix_desc_t *A);

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };
static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };

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
#if defined(HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N));

#ifndef MYDEBUG
    if ( M != N && check ) {
        fprintf(stderr, "Check cannot be perfomed with M != N\n");
        check = 0;
    }
#endif
    LDA = dague_imax( LDA, MT * MB );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, rank, 1, NB, P, dague_imin(M, N), 0, 0,
                               P, dague_imin(M, N), SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

#ifdef MYDEBUG
    PASTE_CODE_ALLOCATE_MATRIX(ddescAl, check,
                               two_dim_block_cyclic, (&ddescAl, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, 1));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIVl, check,
                               two_dim_block_cyclic, (&ddescIPIVl, matrix_Integer, matrix_Lapack,
                                                      1, rank, 1, NB, 1, dague_imin(M, N), 0, 0,
                                                      1, dague_imin(M, N), SMB, SNB, 1));
#endif

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 7657);

    /* Increase diagonale to avoid pivoting */
    if (0)
    {
        tiled_matrix_desc_t *descA = (tiled_matrix_desc_t *)&ddescA;
        int minmnt = dague_imin( descA->mt, descA->nt );
        int minmn  = dague_imin( descA->m,  descA->n );
        int t, e;

        for(t = 0; t < minmnt; t++ ) {
          if(((dague_ddesc_t*) &ddescA)->rank_of(((dague_ddesc_t*) &ddescA), t, t)  == ((dague_ddesc_t*) &ddescA)->myrank)
            {
              dague_data_t* data = ((dague_ddesc_t*) &ddescA)->data_of(((dague_ddesc_t*) &ddescA), t, t);
              dague_data_copy_t* copy = dague_data_get_copy(data, 0);
              dague_complex64_t *tab = (dague_complex64_t*)dague_data_copy_get_ptr(copy);
              for(e = 0; e < descA->mb; e++)
                tab[e * descA->mb + e] += (dague_complex64_t)minmn;
            }
        }
    }

    if ( check )
    {
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescA0 );
#ifdef MYDEBUG
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescAl );
#endif
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,
                        (tiled_matrix_desc_t *)&ddescX );
    }
    if(loud > 2) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0)
        {
            if(loud > 3) printf("+++ Load GPU kernel ... ");
            if(0 != gpu_kernel_init_zgemm(dague))
                {
                    printf("XXX Unable to load GPU kernel.\n");
                    exit(3);
                }
            dague_gpu_data_register(dague,
                                    (dague_ddesc_t*)&ddescA,
                                    MT*NT, MB*NB*sizeof(dague_complex64_t) );
            if(loud > 3) printf("Done\n");
        }
#endif

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
#ifdef MYDEBUG
        if( rank  == 0 ) {
            int i;
            LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, M, N,
                                (dague_complex64_t*)(ddescAl.mat), LDA,
                                (int *)(ddescIPIVl.mat));

            printf("The Lapack swap are :\n");
            for(i=0; i < dague_imin(M, N); i++) {
                if ( i%NB == 0 )
                    printf("\n(%d, %d) ", 0, i/NB );
                printf( "%d ", ((int *)ddescIPIVl.mat)[i] );
            }
            printf("\n");
        }
/*         dplasma_iprint(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&ddescIPIV);  */
        dplasma_zprint(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&ddescA);
        dplasma_zprint(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&ddescAl);

        dplasma_zgeadd( dague, PlasmaUpperLower, -1.0,
                        (tiled_matrix_desc_t*)&ddescA,
                        (tiled_matrix_desc_t*)&ddescAl );
        dplasma_zprint(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&ddescAl);

        dague_data_free(ddescAl.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescAl);
        dague_data_free(ddescIPIVl.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescIPIVl);
#else
        dplasma_ztrsmpl_fusion(dague,
                               (tiled_matrix_desc_t *)&ddescA,
                               (tiled_matrix_desc_t *)&ddescIPIV,
                               (tiled_matrix_desc_t *)&ddescX);

        dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      1.0, (tiled_matrix_desc_t *)&ddescA,
                           (tiled_matrix_desc_t *)&ddescX);

        /* Check the solution */
        ret |= check_solution( dague, (rank == 0) ? loud : 0,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);
#endif
    }

    if ( check ) {
        dague_data_free(ddescA0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
        dague_data_free(ddescB.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
        dague_data_free(ddescX.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);
    }

#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister();
        dague_gpu_kernel_fini(dague, "zgemm");
    }
#endif
    dague_data_free(ddescA.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA);
    dague_data_free(ddescIPIV.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescIPIV);

    cleanup_dague(dague, iparam);

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

    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);

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
