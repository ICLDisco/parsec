
/***************************************************************************//**
 *
 * 
 *
 *  CORE_zparfb applies a complex upper triangular block reflector H
 *  or its transpose H' to a complex rectangular matrix formed by
 *  coupling two tiles A1 and A2. Matrix V is:
 *
 *          COLUMNWISE                    ROWWISE
 *
 *         |     K     |                 |      N2-L     |   L  |
 *      __ _____________ __           __ _________________        __
 *         |    |      |                 |               | \
 *         |    |      |                 |               |   \    L
 *    M2-L |    |      |              K  |_______________|_____\  __
 *         |    |      | M2              |                      |
 *      __ |____|      |                 |                      | K-L
 *         \    |      |              __ |______________________| __
 *       L   \  |      |
 *      __     \|______| __              |          N2          |
 *
 *         | L |  K-L  |
 *
 *******************************************************************************
 *
 * @param[in] side
 *         @arg PlasmaLeft  : apply Q or Q**H from the Left;
 *         @arg PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         @arg PlasmaNoTrans   : No transpose, apply Q;
 *         @arg PlasmaConjTrans : ConjTranspose, apply Q**H.
 *
 * @param[in] direct
 *         Indicates how H is formed from a product of elementary
 *         reflectors
 *         @arg PlasmaForward  : H = H(1) H(2) . . . H(k) (Forward)
 *         @arg PlasmaBackward : H = H(k) . . . H(2) H(1) (Backward)
 *
 * @param[in] storev
 *         Indicates how the vectors which define the elementary
 *         reflectors are stored:
 *         @arg PlasmaColumnwise
 *         @arg PlasmaRowwise
 *
 * @param[in] M1
 *         The number of columns of the tile A1. M1 >= 0.
 *
 * @param[in] N1
 *         The number of rows of the tile A1. N1 >= 0.
 *
 * @param[in] M2
 *         The number of columns of the tile A2. M2 >= 0.
 *
 * @param[in] N2
 *         The number of rows of the tile A2. N2 >= 0.
 *
 * @param[in] K
 *         The order of the matrix T (= the number of elementary
 *         reflectors whose product defines the block reflector).
 *
 * @param[in] L
 *         The size of the triangular part of V
 *
 * @param[in,out] A1
 *         On entry, the M1-by-N1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] LDA1
 *         The leading dimension of the array A1. LDA1 >= max(1,N1).
 *
 * @param[in,out] A2
 *         On entry, the M2-by-N2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] LDA2
 *         The leading dimension of the tile A2. LDA2 >= max(1,N2).
 *
 * @param[in] V
 *         (LDV,K) if STOREV = 'C'
 *         (LDV,M2) if STOREV = 'R' and SIDE = 'L'
 *         (LDV,N2) if STOREV = 'R' and SIDE = 'R'
 *         Matrix V.
 *
 * @param[in] LDV
 *         The leading dimension of the array V.
 *         If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M2);
 *         if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N2);
 *         if STOREV = 'R', LDV >= K.
 *
 * @param[out] T
 *         The triangular K-by-K matrix T in the representation of the
 *         block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] LDT
 *         The leading dimension of the array T. LDT >= K.
 *
 * @param[in,out] WORK
 *
 * @param[in] LDWORK
 *         The dimension of the array WORK.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 * @precisions normal z -> s d c
 *
 ******************************************************************************/
/* This kernel is never traced so return type on previous line for convert2eztrace.pl script */

#include "data_dist/matrix/precision.h"
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <plasma.h>

#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif  /* defined(PRECISION_z) || defined(PRECISION_c) */


///////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_SM_VERSION_KERNEL_NAME_I(func, version)  zgemm_##func##_SM##version
#define GENERATE_SM_VERSION_KERNEL_NAME_I2(func, version) GENERATE_SM_VERSION_KERNEL_NAME_I(func, version)
#define GENERATE_SM_VERSION_KERNEL_NAME(func)             GENERATE_SM_VERSION_KERNEL_NAME_I2(func, CUDA_SM_VERSION)

#define GENERATE_SM_VERSION_NAME_I(func, version)  magmablas_##func##_SM##version
#define GENERATE_SM_VERSION_NAME_I2(func, version) GENERATE_SM_VERSION_NAME_I(func, version)
#define GENERATE_SM_VERSION_NAME(func)             GENERATE_SM_VERSION_NAME_I2(func, CUDA_SM_VERSION)

///////////////////////////////////////////////////////////////////////////////////////////////////

inline static char PLASMA_TRANS_TO_CUBLAS_TRANS(PLASMA_enum trans)
{
    if (trans == PlasmaNoTrans) {
        return 'N';
    } else if (trans == PlasmaTrans) {
        return 'T';
    } else {
        return 'C';
    }
}


extern "C" void
GENERATE_SM_VERSION_NAME(ZPARFB)(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
	            int M1, int N1, int M2, int N2, int K, int L,
    	              dague_complex64_t *A1, int LDA1,
    	              dague_complex64_t *A2, int LDA2,
    	        const dague_complex64_t *V, int LDV,
    	        const dague_complex64_t *T, int LDT,
    	              dague_complex64_t *WORK, int LDWORK,
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

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        fprintf(stderr, "Illegal value of side");
        return;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        fprintf(stderr, "Illegal value of trans");
        return;
    }
    if ((direct != PlasmaForward) && (direct != PlasmaBackward)) {
        fprintf(stderr, "Illegal value of direct");
        return;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        fprintf(stderr, "Illegal value of storev");
        return;
    }
    if (M1 < 0) {
        fprintf(stderr, "Illegal value of M1");
        return;
    }
    if (N1 < 0) {
        fprintf(stderr, "Illegal value of N1");
        return;
    }
    if ((M2 < 0) ||
        ( (side == PlasmaRight) && (M1 != M2) ) ) {
        fprintf(stderr, "Illegal value of M2");
        return;
    }
    if ((N2 < 0) ||
        ( (side == PlasmaLeft) && (N1 != N2) ) ) {
        fprintf(stderr, "Illegal value of N2");
        return;
    }
    if (K < 0) {
        fprintf(stderr, "Illegal value of K");
        return;
    }

    /* Quick return */
    if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0))
        return;


    if (direct == PlasmaForward) {

        if (side == PlasmaLeft) {

            /*
             * Column or Rowwise / Forward / Left
             * ----------------------------------
             *
             * Form  H * A  or  H' * A  where  A = ( A1 )
             *                                     ( A2 )
             */

            /* W = A1 + op(V) * A2 */
          /*  CORE_zpamm(
                    PlasmaW, PlasmaLeft, storev,
                    K, N1, M2, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK); */

             /* W = W + op(V) * A2  op = Trans */
            cublasSetKernelStream( stream );
            cublasZgemm('T', 'N',
                        K, N1, M2,
                        zone,
                        (cuDoubleComplex*)V     /* K*M2  */ , LDV,
                        (cuDoubleComplex*)A2    /* M2*N1 */, LDA2,
                        zzero,
                        (cuDoubleComplex*)WORK  /* K*N1  */, LDWORK);

            /* W = W + A1*/
            cublasSetKernelStream( stream );
            for(j = 0; j < N1; j++) {
                cublasZaxpy(
                        K, zone,
                        (cuDoubleComplex*)(&A1[LDA1*j]), 1,
                        (cuDoubleComplex*)(&WORK[LDWORK*j]), 1);
            }

            /* W = op(T) * W */
           /* cblas_ztrmm(
                CblasColMajor, CblasLeft, CblasUpper,
                (CBLAS_TRANSPOSE)trans, CblasNonUnit, K, N2,
                CBLAS_SADDR(zone), T, LDT, WORK, LDWORK);*/
            cublasSetKernelStream( stream );
            cublasZtrmm( 'L', 'U',
                        PLASMA_TRANS_TO_CUBLAS_TRANS(trans), 'N',
                        K, N2,
                        zone, 
                        (cuDoubleComplex*)T, LDT,
                        (cuDoubleComplex*)WORK, LDWORK);

            /* A1 = A1 - W */
            cublasSetKernelStream( stream );
            for(j = 0; j < N1; j++) {
                /*cblas_zaxpy(
                        K, CBLAS_SADDR(mzone),
                        &WORK[LDWORK*j], 1,
                        &A1[LDA1*j], 1);*/
                cublasZaxpy(K, mzone,
                            (cuDoubleComplex*)(&WORK[LDWORK*j]), 1,
                            (cuDoubleComplex*)(&A1[LDA1*j]), 1);
            }

            /* A2 = A2 - op(V) * W  */
            /* W also changes: W = V * W, A2 = A2 - W */
           /* CORE_zpamm(
                    PlasmaA2, PlasmaLeft, storev,
                    M2, N2, K, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK);*/
            cublasSetKernelStream( stream );
            cublasZgemm('N', 'N',
                        M2, N2, K,
                        mzone,
                        (cuDoubleComplex*)V     /* M2*K  */, LDV,
                        (cuDoubleComplex*)WORK  /* K*N2  */, LDWORK,
                        zone,
                        (cuDoubleComplex*)A2    /* m2*N2 */, LDA2);
        }
        else {
            /*
             * Column or Rowwise / Forward / Right
             * -----------------------------------
             *
             * Form  H * A  or  H' * A  where A  = ( A1 A2 )
             *
             */

            /* W = A1 + A2 * op(V) */
           /* CORE_zpamm(
                    PlasmaW, PlasmaRight, storev,
                    M1, K, N2, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK); */

            /* W = W + A2 * op(V) op = NoTrans */
            cublasZgemm(CUBLAS_OP_N, CUBLAS_OP_N,
                        M1, K, N2,
                        zone,
                        (cuDoubleComplex*)A2    /* M1*N2 */, LDA2,
                        (cuDoubleComplex*)V     /* N2*K  */ , LDV,
                        zzero,
                        (cuDoubleComplex*)WORK  /* M1*K  */, LDWORK);

            /* W = W + A1 */
            for(j = 0; j < N1; j++) {
                cublasZaxpy(
                        K, zone,
                        (cuDoubleComplex*)(&A1[LDA1*j]), 1,
                        (cuDoubleComplex*)(&WORK[LDWORK*j]), 1);
            }

            /* W = W * op(T) */
          /*  cblas_ztrmm(
                CblasColMajor, CblasRight, CblasUpper,
                (CBLAS_TRANSPOSE)trans, CblasNonUnit, M2, K,
                CBLAS_SADDR(zone), T, LDT, WORK, LDWORK);*/
            cublasZtrmm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                         PLASMA_TRANS_TO_CUBLAS_TRANS(trans), CUBLAS_DIAG_NON_UNIT, 
                         M2, K,
                         zone, 
                         (cuDoubleComplex*)T, LDT, 
                         (cuDoubleComplex*)WORK, LDWORK);
             

            /* A1 = A1 - W */
            for(j = 0; j < K; j++) {
              /*  cblas_zaxpy(
                        M1, CBLAS_SADDR(mzone),
                        &WORK[LDWORK*j], 1,
                        &A1[LDA1*j], 1);*/
                cublasZaxpy(
                        M1, mzone,
                        (cuDoubleComplex*)(&WORK[LDWORK*j]), 1,
                        (cuDoubleComplex*)(&A1[LDA1*j]), 1);
            }

            /* A2 = A2 - W * op(V) */
            /* W also changes: W = W * V', A2 = A2 - W */
         /*   CORE_zpamm(
                    PlasmaA2, PlasmaRight, storev,
                    M2, N2, K, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK);*/
            cublasZgemm(CUBLAS_OP_N, CUBLAS_OP_T,
                        M2, N2, K,
                        mzone,
                        (cuDoubleComplex*)WORK  /* M2*K  */, LDWORK,
                        (cuDoubleComplex*)V     /* op(V) K*N2  */, LDV,
                        zone,
                        (cuDoubleComplex*)A2    /* M2*N2 */, LDA2);
        }
    }
    else {
        fprintf(stderr, "Not implemented (Backward / Left or Right)");
        return;
    }

    return;
}
