/**
 *
 * @file dplasma_cuda_ztsmqr.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.7.1
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @precisions normal z -> c d s
 *
 **/
#include <cblas.h>
#include <cublas.h>
#include "dplasma.h"
#include "dplasma_cores.h"
#include "dplasma_zcores.h"

#define max( _a_, _b_ ) (((_a_) < (_b_)) ? (_b_) : (_a_))
#define min( _a_, _b_ ) (((_a_) > (_b_)) ? (_b_) : (_a_))

int
dplasma_cuda_zparfb(PLASMA_enum side, PLASMA_enum trans,
                    PLASMA_enum direct, PLASMA_enum storev,
                    int M1, int N1,
                    int M2, int N2,
                    int K, int L,
                    dague_complex64_t *A1, int LDA1,
                    dague_complex64_t *A2, int LDA2,
                    const dague_complex64_t *V, int LDV,
                    const dague_complex64_t *T, int LDT,
                    dague_complex64_t *WORK, int LDWORK,
                    dague_complex64_t *WORKC, int LDWORKC,
                    CUstream stream)
{
#if defined(PRECISION_z) || defined(PRECISION_c)
    cuDoubleComplex zzero = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex zone  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex mzone = make_cuDoubleComplex(-1.0, 0.0);
#else
    double zzero = 0.0;
    double zone  = 1.0;
    double mzone = -1.0;
#endif /* defined(PRECISION_z) || defined(PRECISION_c) */

    int j;
    (void)L;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        return -1;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        return -2;
    }
    if ((direct != PlasmaForward) && (direct != PlasmaBackward)) {
        return -3;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        return -4;
    }
    if (M1 < 0) {
        return -5;
    }
    if (N1 < 0) {
        return -6;
    }
    if ((M2 < 0) ||
        ( (side == PlasmaRight) && (M1 != M2) ) ) {
        return -7;
    }
    if ((N2 < 0) ||
        ( (side == PlasmaLeft) && (N1 != N2) ) ) {
        return -8;
    }
    if (K < 0) {
        return -9;
    }

    /* Quick return */
    if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0))
        return PLASMA_SUCCESS;

    if (direct == PlasmaForward) {

        if (side == PlasmaLeft) {

            /*
             * Column or Rowwise / Forward / Left
             * ----------------------------------
             *
             * Form  H * A  or  H' * A  where  A = ( A1 )
             *                                     ( A2 )
             */

            /*
             * W = A1 + V' * A2:
             *      W = A1
             *      W = W + V' * A2
             *
             */
            cudaMemcpy2DAsync( WORK, LDWORK * sizeof(cuDoubleComplex),
                               A1,   LDA1   * sizeof(cuDoubleComplex),
                               K * sizeof(cuDoubleComplex), N1,
                               cudaMemcpyDeviceToDevice, stream );

            cublasZgemm(lapack_const(PlasmaConjTrans), 'N',
                        K, N1, M2,
                        zone,
                        (cuDoubleComplex*)V     /* K*M2  */, LDV,
                        (cuDoubleComplex*)A2    /* M2*N1 */, LDA2,
                        zone,
                        (cuDoubleComplex*)WORK  /* K*N1  */, LDWORK);

            if (WORKC == NULL) {
                /* W = op(T) * W */
                cublasZtrmm( 'L', 'U',
                             lapack_const(trans), 'N',
                             K, N2,
                             zone,
                             (cuDoubleComplex*)T, LDT,
                             (cuDoubleComplex*)WORK, LDWORK);


                /* A1 = A1 - W = A1 - op(T) * W */
                for(j = 0; j < N1; j++) {
                    cublasZaxpy(K, mzone,
                                (cuDoubleComplex*)(WORK + LDWORK*j), 1,
                                (cuDoubleComplex*)(A1 + LDA1*j), 1);
                }

                /* A2 = A2 - op(V) * W  */
                cublasZgemm('N', 'N',
                            M2, N2, K,
                            mzone,
                            (cuDoubleComplex*)V     /* M2*K  */, LDV,
                            (cuDoubleComplex*)WORK  /* K*N2  */, LDWORK,
                            zone,
                            (cuDoubleComplex*)A2    /* m2*N2 */, LDA2);

            } else {
                /* Wc = V * op(T) */
                cublasZgemm( 'N', lapack_const(trans),
                             M2, K, K,
                             zone,  (cuDoubleComplex*)V,     LDV,
                                    (cuDoubleComplex*)T,     LDT,
                             zzero, (cuDoubleComplex*)WORKC, LDWORKC );

                /* A1 = A1 - opt(T) * W */
                cublasZgemm( lapack_const(trans), 'N',
                             K, N1, K,
                             mzone, (cuDoubleComplex*)T,    LDT,
                                    (cuDoubleComplex*)WORK, LDWORK,
                             zone,  (cuDoubleComplex*)A1,   LDA1 );

                /* A2 = A2 - Wc * W */
                cublasZgemm( 'N', 'N',
                             M2, N2, K,
                             mzone, (cuDoubleComplex*)WORKC, LDWORKC,
                                    (cuDoubleComplex*)WORK,  LDWORK,
                             zone,  (cuDoubleComplex*)A2,    LDA2 );
            }
        }
        else {
            /*
             * Column or Rowwise / Forward / Right
             * -----------------------------------
             *
             * Form  H * A  or  H' * A  where A  = ( A1 A2 )
             *
             */
            fprintf(stderr, "Not implemented (Column or Rowwise / Forward / Right)");
            return DAGUE_NOT_SUPPORTED;
        }
    }
    else {
        fprintf(stderr, "Not implemented (Backward / Left or Right)");
        return DAGUE_NOT_SUPPORTED;
    }

    return DAGUE_SUCCESS;
}

int
dplasma_cuda_ztsmqr( PLASMA_enum side, PLASMA_enum trans,
                     int M1, int N1,
                     int M2, int N2,
                     int K, int IB,
                     dague_complex64_t *A1, int LDA1,
                     dague_complex64_t *A2, int LDA2,
                     const dague_complex64_t *V, int LDV,
                     const dague_complex64_t *T, int LDT,
                     dague_complex64_t *WORK, int LDWORK,
                     dague_complex64_t *WORKC, int LDWORKC,
                     CUstream stream)
{
    int i, i1, i3;
    int NQ, NW;
    int kb;
    int ic = 0;
    int jc = 0;
    int mi = M1;
    int ni = N1;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        return -1;
    }

    /* NQ is the order of Q */
    if (side == PlasmaLeft) {
        NQ = M2;
        NW = IB;
    }
    else {
        NQ = N2;
        NW = M1;
    }

    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        return -2;
    }
    if (M1 < 0) {
        return -3;
    }
    if (N1 < 0) {
        return -4;
    }
    if ( (M2 < 0) ||
         ( (M2 != M1) && (side == PlasmaRight) ) ){
        return -5;
    }
    if ( (N2 < 0) ||
         ( (N2 != N1) && (side == PlasmaLeft) ) ){
        return -6;
    }
    if ((K < 0) ||
        ( (side == PlasmaLeft)  && (K > M1) ) ||
        ( (side == PlasmaRight) && (K > N1) ) ) {
        return -7;
    }
    if (IB < 0) {
        return -8;
    }
    if (LDA1 < max(1,M1)){
        return -10;
    }
    if (LDA2 < max(1,M2)){
        return -12;
    }
    if (LDV < max(1,NQ)){
        return -14;
    }
    if (LDT < max(1,IB)){
        return -16;
    }
    if (LDWORK < max(1,NW)){
        return -18;
    }

    /* Quick return */
    if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0) || (IB == 0))
        return DAGUE_SUCCESS;

    if (((side == PlasmaLeft)  && (trans != PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans == PlasmaNoTrans))) {
        i1 = 0;
        i3 = IB;
    }
    else {
        i1 = ((K-1) / IB)*IB;
        i3 = -IB;
    }

    for(i = i1; (i > -1) && (i < K); i += i3) {
        kb = min(IB, K-i);

        if (side == PlasmaLeft) {
            /*
             * H or H' is applied to C(i:m,1:n)
             */
            mi = M1 - i;
            ic = i;
        }
        else {
            /*
             * H or H' is applied to C(1:m,i:n)
             */
            ni = N1 - i;
            jc = i;
        }
        /*
         * Apply H or H' (NOTE: CORE_zparfb used to be CORE_ztsrfb)
         */
        dplasma_cuda_zparfb( side, trans, PlasmaForward, PlasmaColumnwise,
                             mi, ni, M2, N2, kb, 0,
                             A1 + LDA1*jc+ic, LDA1,
                             A2, LDA2,
                             V + LDV*i, LDV,
                             T + LDT*i, LDT,
                             WORK, LDWORK, WORKC, LDWORKC, stream );
    }
    return DAGUE_SUCCESS;
}
