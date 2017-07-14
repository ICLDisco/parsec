/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

double check_zlanm2(int M, int N, parsec_complex64_t *A, int LDA, int *info )
{
    double            *DX  = (double*)malloc(N * sizeof(double));
    parsec_complex64_t *ZX  = (parsec_complex64_t*)malloc(N * sizeof(parsec_complex64_t));
    parsec_complex64_t *ZSX = (parsec_complex64_t*)malloc(M * sizeof(parsec_complex64_t));
    parsec_complex64_t zone  = 1.;
    parsec_complex64_t zzero = 0.;
    parsec_complex64_t alpha;
    double normx, normsx, e0, e1, tol;
    int maxiter, i = 0;

    memset( DX, 0, N * sizeof(double) );
    CORE_dzasum(PlasmaColumnwise, PlasmaUpperLower,
                M, N, A, LDA, DX);

    normx = cblas_dnrm2( N, DX, 1 );
    normsx = 0.;
    tol = 3.e-1;
    e0 = 0.;
    e1 = normx;
    maxiter = dplasma_imin(100, N);

#if defined(PRECISION_z) || defined(PRECISION_c)
    CORE_dlag2z( 1, N, DX, 1, ZX, 1 );
#else
    CORE_zlacpy( PlasmaUpperLower, 1, N, DX, 1, ZX, 1 );
#endif

    while( (i < maxiter) &&
           (fabs(e1 - e0) > (tol * e1)))
    {
#if defined(VERBOSE)
            printf( "LAP[0] ZLANM2[%d] normx=%e / normsx=%e / e0=%e / e1=%e\n",
                    i, normx, normsx, e0, e1 );
#endif
        alpha = 1. / normx;
        cblas_zscal( N, CBLAS_SADDR(alpha), ZX, 1 );

        cblas_zgemv( CblasColMajor, CblasNoTrans,   M, N,
                     CBLAS_SADDR(zone), A, LDA, ZX,  1, CBLAS_SADDR(zzero), ZSX, 1 );
        cblas_zgemv( CblasColMajor, CblasConjTrans, M, N,
                     CBLAS_SADDR(zone), A, LDA, ZSX, 1, CBLAS_SADDR(zzero), ZX,  1 );

        normx  = cblas_dznrm2( N, ZX,  1 );
        normsx = cblas_dznrm2( M, ZSX, 1 );

        e0 = e1;
        e1 = normx / normsx;
        i++;
    }

#if defined(VERBOSE)
        printf( "LAP[0] ZLANM2[%d] normx=%e / normsx=%e / e0=%e / e1=%e\n",
                i, normx, normsx, e0, e1 );
#endif

    *info = i;

    free(DX);
    free(ZX);
    free(ZSX);

    return e1;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    double result;
    double normlap = 0.0;
    double normdag = 0.0;
    double eps = LAPACKE_dlamch_work('e');
    int iparam[IPARAM_SIZEOF];
    int An, ret = 0;
    int infolap, infodag;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
    /* Initialize Parsec */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    An = dplasma_imax(M, N);
    LDA = max( LDA, M );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(dcA0, 1,
        two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               M, An, SMB, SNB, 1));

    /*
     * General cases LANGE
     */
    {
        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   M, N, SMB, SNB, P));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA0, 3872);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA,  3872);
        if(loud > 2) printf("Done\n");

        if ( rank == 0 ) {
            printf("***************************************************\n");
        }
        if(loud > 2) printf("+++ Computing 2-norm ... \n");
        normdag = dplasma_zlanm2(parsec,
                                 (parsec_tiled_matrix_dc_t *)&dcA,
                                 &infodag);

        if ( rank == 0 ) {
            normlap = check_zlanm2(M, N, (parsec_complex64_t*)(dcA0.mat), dcA0.super.lm, &infolap );
        }
        if(loud > 2) printf("Done.\n");

        if ( loud > 3 ) {
            printf( "%d: The 2-norm of A is %e [%d]\n",
                    rank, normdag, infodag);
        }

        if ( rank == 0 ) {
            result = fabs(normdag - normlap) / (normlap * eps * dplasma_imax(M, N));

            if ( loud > 3 ) {
                printf( "%d: The 2-norm of A is %e [%d] (LAPACK)\n",
                        rank, normlap, infolap);
            }

            if ( result < 1. && infolap == infodag ) {
                printf(" ----- TESTING ZLANM2 ... SUCCESS !\n");
            } else {
                printf("       Ndag = %e, Nlap = %e\n", normdag, normlap );
                printf("       | Ndag - Nlap | / eps = %e\n", result);
                printf("       #iterations  lapack=%d, dplasma=%d", infolap, infodag);
                printf(" ----- TESTING ZLANM2 ... FAILED !\n");
                ret |= 1;
            }
        }

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    }

    if ( rank == 0 ) {
        printf("***************************************************\n");
    }
    parsec_data_free(dcA0.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);

    cleanup_parsec(parsec, iparam);

    return ret;
}
