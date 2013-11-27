#include "common.h"

#pragma PARSEC_INVARIANT A.mt == A.nt
#pragma PARSEC_INVARIANT A.nt == B.nt

#define A(m,n) BLKADDR(A, PLASMA_Complex64_t, m, n)
#define B(m,n) BLKADDR(B, PLASMA_Complex64_t, m, n)

void plasma_pztrmm_quark(PLASMA_enum side, PLASMA_enum uplo,
                         PLASMA_enum trans, PLASMA_enum diag,
                         PLASMA_Complex64_t alpha, PLASMA_desc A, PLASMA_desc B)
{
    int k, m, n;
    int lda, ldak, ldb, ldbk;
    int tempkm, tempkn, tempmm, tempnn;

    for (n = 0; n < B.nt; n++) {
        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
        lda = BLKLDD(A, n);
        for (m = 0; m < B.mt; m++) {
            tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
            ldb = BLKLDD(B, m);
            QUARK_CORE_ztrmm(
                plasma->quark, &task_flags,
                side, uplo, trans, diag,
                tempmm, tempnn, A.mb,
                alpha, A(n, n), lda,  /* lda * tempkm */
                B(m, n), ldb); /* ldb * tempnn */

            for (k = n+1; k < A.mt; k++) {
                tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                QUARK_CORE_zgemm(
                    plasma->quark, &task_flags,
                    PlasmaNoTrans, trans,
                    tempmm, tempnn, tempkn, A.mb,
                    alpha, B(m, k), ldb,
                    A(n, k), lda,
                    1.,  B(m, n), ldb);
            }
        }
    }
}
