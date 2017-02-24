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

int *IPIV_Lapack = NULL;

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int i, ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize Parsec */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if ( N > NB ) {
        fprintf(stderr, "This test performs the factorization of only one panel, N is set to NB\n");
        N = NB;
    }

    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = dplasma_imax( LDA, MT * MB );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, rank, 1, NB, P, dplasma_imin(M, N), 0, 0,
                               P, dplasma_imin(M, N), SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescAl, check,
                               two_dim_block_cyclic, (&ddescAl, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, 1));

    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIVl, check,
                               two_dim_block_cyclic, (&ddescIPIVl, matrix_Integer, matrix_Lapack,
                                                      1, rank, 1, NB, 1, dplasma_imin(M, N), 0, 0,
                                                      1, dplasma_imin(M, N), SMB, SNB, 1));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 7657);

    /* Increase diagonale to avoid pivoting */
    if (0) {
        tiled_matrix_desc_t *descA = (tiled_matrix_desc_t *)&ddescA;
        int minmnt = dplasma_imin( descA->mt, descA->nt );
        int minmn  = dplasma_imin( descA->m,  descA->n );
        int t, e;

        for(t = 0; t < minmnt; t++ ) {
          if(((parsec_ddesc_t*) &ddescA)->rank_of(((parsec_ddesc_t*) &ddescA), t, t)  == ((parsec_ddesc_t*) &ddescA)->myrank)
            {
              parsec_data_t* data = ((parsec_ddesc_t*) &ddescA)->data_of(((parsec_ddesc_t*) &ddescA), t, t);
              parsec_data_copy_t* copy = parsec_data_get_copy(data, 0);
              parsec_complex64_t *tab = (parsec_complex64_t*)parsec_data_copy_get_ptr(copy);
              for(e = 0; e < descA->mb; e++)
                tab[e * descA->mb + e] += (parsec_complex64_t)minmn;
            }
        }
    }

    dplasma_zlacpy( parsec, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA,
                    (tiled_matrix_desc_t *)&ddescA0 );

    if ( check ) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescAl );
        K = 0;
    }
    if(loud > 2) printf("Done\n");

    /* Create Parsec */
    /* Startup */
    info = dplasma_zgetrf_fusion(parsec,
                                 (tiled_matrix_desc_t*)&ddescA,
                                 (tiled_matrix_desc_t*)&ddescIPIV);

    for(i=0; i<K; i++) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA0,
                        (tiled_matrix_desc_t *)&ddescA );

        if(loud > 2) printf("+++ Computing getrf ... ");
        PASTE_CODE_ENQUEUE_KERNEL(parsec, zgetrf_fusion,
                                  ((tiled_matrix_desc_t*)&ddescA,
                                   (tiled_matrix_desc_t*)&ddescIPIV,
                                   P,
                                   Q,
                                   &info));
        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(parsec, zgetrf_fusion);
        dplasma_zgetrf_fusion_Destruct( PARSEC_zgetrf_fusion );
        if(loud > 2) printf("Done.\n");
    }


    if ( check && info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        int ipivok = 1;
        double eps, Anorm, Rnorm, result;

        eps = LAPACKE_dlamch_work('e');

        Anorm = dplasma_zlange( parsec, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA0 );
        if( rank  == 0 ) {
            parsec_complex64_t *dA, *lA;
            int *dplasma_piv;
            int *lapack_piv;

            dA = (parsec_complex64_t*)( ((parsec_ddesc_t*) &ddescA )->data_of(((parsec_ddesc_t*) &ddescA),  0, 0) );
            lA = (parsec_complex64_t*)( ((parsec_ddesc_t*) &ddescAl)->data_of(((parsec_ddesc_t*) &ddescAl), 0, 0) );
            dplasma_piv = (int*)( ((parsec_ddesc_t*) &ddescIPIV )->data_of(((parsec_ddesc_t*) &ddescIPIV),  0, 0) );
            lapack_piv  = (int*)( ((parsec_ddesc_t*) &ddescIPIVl)->data_of(((parsec_ddesc_t*) &ddescIPIVl), 0, 0) );

            LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, M, N, lA, LDA, lapack_piv );

            /* Check IPIV */
            if (loud > 2 )
                printf("--- Check IPIV ---\n");
            for(i=0; i < dplasma_imin(M, N); i++) {
                if ( dplasma_piv[i] != lapack_piv[i] ) {
                    ipivok = 0;
                    if ( loud > 2 ) {
                        printf( "IPIV[%d] = (%d / %d) (%e / %e)\n",
                                i, dplasma_piv[i], lapack_piv[i],
                                cabs(dA[ i * (NB+1) ]), cabs(lA[ i * (ddescA.super.lm+1) ]) );
                    }
                }
            }
            if (loud > 2 )
                printf("------------------\n");
        }

#if defined(HAVE_MPI)
        MPI_Bcast( &ipivok, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        if ( ipivok ) {
            dplasma_zgeadd( parsec, PlasmaNoTrans,
                            -1.0, (tiled_matrix_desc_t*)&ddescA,
                             1.0, (tiled_matrix_desc_t*)&ddescAl );

            Rnorm = dplasma_zlange( parsec, PlasmaMaxNorm, (tiled_matrix_desc_t*)&ddescAl );
            result = Rnorm / (Anorm * max(M,N) * eps);

            if ( rank == 0 && loud > 2 ) {
                printf("  ||A||_inf = %e, ||lA - dA||_max = %e, ||lA-dA||/(||A|| * M * eps) = %e\n",
                       Anorm, Rnorm, result );
            }
        } else {
            printf("-- Solution cannot be checked ! \n");
            ret |= 1;
        }

        if ( isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
            if( rank == 0 && loud ) printf("-- Solution is suspicious ! \n");
            ret |= 1;
        } else {
            if( rank == 0 && loud ) printf("-- Solution is CORRECT ! \n");
        }

        parsec_data_free(ddescAl.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescAl);
        parsec_data_free(ddescIPIVl.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescIPIVl);
    }

    parsec_data_free(ddescA.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA);
    parsec_data_free(ddescIPIV.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescIPIV);

    cleanup_parsec(parsec, iparam);

    return ret;
}
