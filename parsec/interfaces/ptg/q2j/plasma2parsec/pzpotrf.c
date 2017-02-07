#include "common.h"

#define A(m,n) BLKADDR(A, PLASMA_Complex64_t, m, n)

void plasma_pzpotrf_quark(PLASMA_enum uplo, PLASMA_desc A,
                          PLASMA_sequence *sequence, PLASMA_request *request)
{
    plasma_context_t *plasma;
    Quark_Task_Flags task_flags = Quark_Task_Flags_Initializer;

    int k, m, n;
    int ldak, ldam;
    int tempkm, tempmm;

    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t) 1.0;
    PLASMA_Complex64_t mzone = (PLASMA_Complex64_t)-1.0;

    plasma = plasma_context_self();
    if (sequence->status != PLASMA_SUCCESS)
        return;
    QUARK_Task_Flag_Set(&task_flags, TASK_SEQUENCE, (intptr_t)sequence->quark_sequence);

    for (k = 0; k < A.mt; k++) {
        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
        ldak = BLKLDD(A, k);
        QUARK_CORE_zpotrf(
            plasma->quark, &task_flags,
            PlasmaLower, tempkm, A.mb,
            A(k, k), ldak,
            sequence, request, A.nb*k);

        for (m = k+1; m < A.mt; m++) {
            tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            ldam = BLKLDD(A, m);
            QUARK_CORE_ztrsm(
                plasma->quark, &task_flags,
                PlasmaRight, PlasmaLower, PlasmaConjTrans, PlasmaNonUnit,
                tempmm, A.mb, A.mb,
                zone, A(k, k), ldak,
                A(m, k), ldam);
        }
        for (m = k+1; m < A.mt; m++) {
            tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            ldam = BLKLDD(A, m);
            QUARK_CORE_zherk(
                plasma->quark, &task_flags,
                PlasmaLower, PlasmaNoTrans,
                tempmm, A.mb, A.mb,
                -1.0, A(m, k), ldam,
                1.0, A(m, m), ldam);

            for (n = k+1; n < m; n++) {
                QUARK_CORE_zgemm(
                    plasma->quark, &task_flags,
                    PlasmaNoTrans, PlasmaConjTrans,
                    tempmm, A.mb, A.mb, A.mb,
                    mzone, A(m, k), ldam,
                    A(n, k), A.mb,
                    zone,  A(m, n), ldam);
            }
        }
    }
}
