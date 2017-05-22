/*
 * Copyright (c) 2011-2014 The University of Tennessee and The University
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
#include "data_dist/matrix/diag_band_to_rect.h"
#include "dplasma/lib/dplasmatypes.h"

/* Including the bulge chassing */
#define FADDS_ZHEEV(__n) (((__n) * (-8.0 / 3.0 + (__n) * (1.0 + 2.0 / 3.0 * (__n)))) - 4.0)
#define FMULS_ZHEEV(__n) (((__n) * (-1.0 / 6.0 + (__n) * (5.0 / 2.0 + 2.0 / 3.0 * (__n)))) - 15.0)

#undef PRINTF_HEAVY

static int check_solution(int N, double *E1, double *E2, double eps);

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int j;
    int rc;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 120, 120);

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHEEV, FMULS_ZHEEV, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SMB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SMB, P));

    /* Fill A with randomness */
    dplasma_zplghe( parsec, (double)N, uplo,
                    (tiled_matrix_desc_t *)&ddescA, 3872);
#ifdef PRINTF_HEAVY
    printf("########### A (initial, tile storage)\n");
    dplasma_zprint( parsec, uplo, (tiled_matrix_desc_t *)&ddescA );
#endif

    /* Step 1 - Reduction A to band matrix */
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zherbt,
                              (uplo, IB,
                               (tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT));
    PASTE_CODE_PROGRESS_KERNEL(parsec, zherbt);
#ifdef PRINTF_HEAVY
    printf("########### A (reduced to band form)\n");
    dplasma_zprint( parsec, uplo, &ddescA);
#endif

goto fin;

    /* Step 2 - Conversion of the tiled band to 1D band storage */
    PASTE_CODE_ALLOCATE_MATRIX(ddescBAND, 1,
        two_dim_block_cyclic, (&ddescBAND, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB+1, NB+2, MB+1, (NB+2)*(NT+1), 0, 0,
                               MB+1, (NB+2)*(NT+1), 1, SNB, 1 /* 1D cyclic */ ));
    SYNC_TIME_START();
    parsec_diag_band_to_rect_taskpool_t* PARSEC_diag_band_to_rect = parsec_diag_band_to_rect_new((sym_two_dim_block_cyclic_t*)&ddescA, &ddescBAND,
                                                                                            MT, NT, MB, NB, sizeof(parsec_complex64_t));
    parsec_arena_t* arena = PARSEC_diag_band_to_rect->arenas[PARSEC_diag_band_to_rect_DEFAULT_ARENA];
    dplasma_add2arena_tile(arena,
                           MB*NB*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, MB);
    rc = parsec_enqueue(parsec, (parsec_taskpool_t*)PARSEC_diag_band_to_rect);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    SYNC_TIME_PRINT(rank, ( "diag_band_to_rect N= %d NB = %d : %f s\n", N, NB, sync_time_elapsed));
#ifdef PRINTF_HEAVY
    printf("########### BAND (converted from A)\n");
    dplasma_zprint(parsec, PlasmaUpperLower, &ddescBAND);
#endif

    /* Step 3 - Reduce band to bi-diag form */
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zhbrdt, ((tiled_matrix_desc_t*)&ddescBAND));
    PASTE_CODE_PROGRESS_KERNEL(parsec, zhbrdt);

    if( check ) {
        PLASMA_Complex64_t *A0  = (PLASMA_Complex64_t *)malloc(LDA*N*sizeof(PLASMA_Complex64_t));
        double *W0              = (double *)malloc(N*sizeof(double));
        PLASMA_Complex64_t* band;
        double *D               = (double *)malloc(N*sizeof(double));
        double *E               = (double *)malloc(N*sizeof(double));
        int INFO;

        /* COMPUTE THE EIGENVALUES FROM DPLASMA (with LAPACK) */
        SYNC_TIME_START();
        if( P*Q > 1 ) {
            /* We need to gather the distributed band on rank0 */
#if 0
            /* LAcpy doesn't taskpool differing tile sizes, so lets get simple here */
            PASTE_CODE_ALLOCATE_MATRIX(ddescW, 1,
                                       two_dim_block_cyclic, (&ddescW, matrix_ComplexDouble, matrix_Tile,
                                                              nodes, rank, 2, N, 1, 1, 0, 0, 2, N, 1, 1, 1)); /* on rank 0 only */
#else
            PASTE_CODE_ALLOCATE_MATRIX(ddescW, 1,
                                       two_dim_block_cyclic, (&ddescW, matrix_ComplexDouble, matrix_Tile,
                                                              1, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 0, 0,
                                                              MB+1, (NB+2)*NT, 1, 1, 1 /* rank0 only */ ));
#endif
            dplasma_zlacpy(parsec, PlasmaUpperLower, &ddescBAND.super, &ddescW.super);
            band = ddescW.mat;
        }
        else {
            band = ddescBAND.mat;
        }

        if( 0 == rank ) {
            /* Extract the D (diagonal) and E (subdiag) vectors from the band */
            int k, sizearena = (MB+1)*(NB+2);
            /* store resulting diag and lower diag D and E*/
            for( k=0; k<NT-1; k++ ) {
                for( j=0; j<NB; j++ ) {
                    if( (k*NB)+j >= N ) break;
                    D[(k*NB)+j] = creal(band[(k*sizearena)+ (MB+1)*j]);
                    E[(k*NB)+j] = creal(band[(k*sizearena)+ (MB+1)*j+1]);
                }
            }
            for( j=0; j<NB-1; j++ ) {
                if( (k*NB)+j >= N ) break;
                D[(k*NB)+j] = creal(band[(k*sizearena)+ (MB+1)*j]);
                E[(k*NB)+j] = creal(band[(k*sizearena)+ (MB+1)*j+1]);
            }
            if( (k*NB)+j < N ) D[(k*NB)+j] = creal(band[(k*sizearena)+ (MB+1)*j]);

#ifdef PRINTF_HEAVY
            printf("############################\n"
                   "D= ");
            for(int i = 0; i < N; i++) {
                printf("% 11.4g ", D[i]);
            }
            printf("\nE= ");
            for(int i = 0; i < N-1; i++) {
                printf("% 11.4g ", E[i]);
            }
            printf("\n");
#endif
            /* call eigensolver */
            dsterf_( &N, D, E, &INFO);
            assert( 0 == INFO );
        }
        SYNC_TIME_PRINT( rank, ("Dplasma Stage3: dsterf\n"));

        /* COMPUTE THE EIGENVALUES WITH LAPACK */
        /* Regenerate A (same random generator) into A0 */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0t, 1,
                                   two_dim_block_cyclic, (&ddescA0t, matrix_ComplexDouble, matrix_Tile,
                                                          1, rank, MB, NB, LDA, N, 0, 0,
                                                          N, N, 1, 1, 1));
        /* Fill A0 again */
        dplasma_zlaset( parsec, PlasmaUpperLower, 0.0, 0.0, &ddescA0t.super);
        dplasma_zplghe( parsec, (double)N, uplo, (tiled_matrix_desc_t *)&ddescA0t, 3872);
        /* Convert into Lapack format */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1, 
                                   two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Lapack,
                                                          1, rank, MB, NB, LDA, N, 0, 0,
                                                          N, N, 1, 1, 1));
        dplasma_zlacpy( parsec, uplo, &ddescA0t.super, &ddescA0.super);
        parsec_data_free(ddescA0t.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0t);
#ifdef PRINTF_HEAVY
        printf("########### A0 (initial, lapack storage)\n");
        dplasma_zprint( parsec, uplo, &ddescA0 );
#endif
        if( 0 == rank ) {
            A0 = ddescA0.mat;
            /* Compute eigenvalues directly */
            TIME_START();
            LAPACKE_zheev( LAPACK_COL_MAJOR,
                           lapack_const(PlasmaNoVec), lapack_const(uplo),
                           N, A0, LDA, W0);
            TIME_PRINT(rank, ("LAPACK HEEV\n"));
        }
#ifdef PRINTF_HEAVY
        printf("########### A0 (after LAPACK direct eignesolver)\n");
        dplasma_zprint( parsec, uplo, &ddescA0 );
#endif
        parsec_data_free(ddescA0.mat);
        tiled_matrix_desc_destroy( &ddescA0.super );
        if( 0 == rank ) {
#ifdef PRINTF_HEAVY
            printf("\n###############\nDPLASMA Eignevalues\n");
            for(int i = 0; i < N; i++) {
                printf("% .14e", D[i]);
            }
            printf("\nLAPACK Eigenvalues\n");
            for(int i = 0; i < N; i++) {
                printf("% .14e ", W0[i]);
            }
            printf("\n");
#endif
            double eps = LAPACKE_dlamch_work('e');
            printf("\n");
            printf("------ TESTS FOR PLASMA ZHEEV ROUTINE -------  \n");
            printf("        Size of the Matrix %d by %d\n", N, N);
            printf("\n");
            printf(" The matrix A is randomly generated for each test.\n");
            printf("============\n");
            printf(" The relative machine precision (eps) is to be %e \n",eps);
            printf(" Computational tests pass if scaled residuals are less than 60.\n");

            /* Check the eigen solutions */
            int info_solution = check_solution(N, W0, D, eps);

            if (info_solution == 0) {
                printf("***************************************************\n");
                printf(" ---- TESTING ZHEEV ..................... PASSED !\n");
                printf("***************************************************\n");
            }
            else {
                printf("************************************************\n");
                printf(" - TESTING ZHEEV ..................... FAILED !\n");
                printf("************************************************\n");
            }
        }
        free(W0); free(D); free(E);
    }

    dplasma_zherbt_Destruct( PARSEC_zherbt );
    PARSEC_INTERNAL_TASKPOOL_DESTRUCT( PARSEC_diag_band_to_rect );
    dplasma_zhbrdt_Destruct( PARSEC_zhbrdt );

    parsec_data_free(ddescBAND.mat);
    parsec_data_free(ddescA.mat);
    parsec_data_free(ddescT.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescBAND);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescT);
fin:
    cleanup_parsec(parsec, iparam);

    return EXIT_SUCCESS;
}


#include "math.h"

/*------------------------------------------------------------
 *  Check the eigenvalues
 */
static int check_solution(int N, double *E1, double *E2, double eps)
{
    int info_solution, i;
    double resid;
    double maxtmp;
    double maxel = fabs( fabs(E1[0]) - fabs(E2[0]) );
    double maxeig = fmax( fabs(E1[0]), fabs(E2[0]) );
    for (i = 1; i < N; i++){
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = fmax(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = fmax(maxtmp, maxeig);
        maxel  = fmax(resid,  maxel );
    }

    maxel = maxel / (maxeig * N * eps);
    printf(" ======================================================\n");
    printf(" | D - eigcomputed | / (|D| * N * eps) : %15.3E \n",  maxel );
    printf(" ======================================================\n");

    if ( isnan(maxel) || isinf(maxel) || (maxel > 100) ) {
        printf("-- The eigenvalues are suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The eigenvalues are CORRECT ! \n");
        info_solution = 0;
    }
    return info_solution;
}

