#pragma CORE_zaxpy A B
#pragma CORE_zgelqt A T
#pragma CORE_zgemm A B C
#pragma CORE_zgeqrt A T
#pragma CORE_zgessm IPIV L A
#pragma CORE_zgetrf A IPIV
#pragma CORE_zhemm A B C
#pragma CORE_zher2k A B C
#pragma CORE_zherk A C
#pragma CORE_zlacpy A B
#pragma CORE_zlag2c A B
#pragma CORE_clag2z A B
#pragma CORE_zlange A result
#pragma CORE_zlange_f1 A result
#pragma CORE_zlanhe A result
#pragma CORE_zlanhe_f1 A result
#pragma CORE_zlansy A result
#pragma CORE_zlansy_f1 A result
#pragma CORE_zlauum A
#pragma CORE_zpotrf A
#pragma CORE_zssssm A1 A2 L1 L2 IPIV
#pragma CORE_zsymm A B C
#pragma CORE_zsyr2k A B C
#pragma CORE_zsyrk A C
#pragma CORE_ztrmm A B
#pragma CORE_ztrsm A B
#pragma CORE_ztrtri A
#pragma CORE_ztslqt A1 A2 T
#pragma CORE_ztsmlq A1 A2 V T
#pragma CORE_ztsmqr A1 A2 V T
#pragma CORE_ztsqrt A1 A2 T
#pragma CORE_ztstrf U A L IPIV
#pragma CORE_zttmqr A1 A2 V T
#pragma CORE_zttqrt A1 A2 T
#pragma CORE_zunmlq A T C
#pragma CORE_zunmqr A T C
#pragma CORE_zherfb1 A T C
#pragma CORE_ztsmlq1 A1 A2 V T
#pragma CORE_ztsmlqrl A1 A2 A3 V T
#pragma CORE_ztsmqrlr A1 A2 A3 V T
#pragma CORE_ztsmqr1 A1 A2 V T

#pragma PARSEC_DATA_TYPE_MODIFIER T LITTLE_T
#pragma PARSEC_DATA_COLOCATED T A

void 
plasma_pzherbt_quark(PLASMA_enum uplo,
		     PLASMA_desc A, PLASMA_desc T,
		     PLASMA_sequence * sequence, PLASMA_request * request)
{


	int             k, m, n, i, j;
	int             ldak, ldam, ldan, ldaj, ldai;
	int             tempkn, tempmm, tempnn, tempjj;
	int             ib;
	//static PLASMA_Complex64_t zzero = 0.0;

	plasma = plasma_context_self();
	if (sequence->status != PLASMA_SUCCESS)
		return;

	QUARK_Task_Flag_Set(&task_flags, TASK_SEQUENCE, (intptr_t) sequence->quark_sequence);

	ib = PLASMA_IB;
	for (k = 0; k < A.nt - 1; k++) {
		tempkn = k + 1 == A.nt - 1 ? A.n - (k + 1) * A.nb : A.nb;
		ldak = BLKLDD(A, k + 1);
#pragma PARSEC_DATA_ACCESS_MASK A 0b10
#pragma PARSEC_TASK_START CORE_zgeqrt A:INOUT:UP, T:OUTPUT, W:SCRATCH, Q:SCRATCH

		QUARK_Insert_Task((plasma->quark), CORE_zgeqrt_quark, (&task_flags),
                        sizeof(int), &(tempkn), VALUE,
                        sizeof(int), &(A.nb), VALUE,
                        sizeof(int), &(ib), VALUE,
                        sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[k + 1][k]), INOUT,
                        sizeof(int), &(ldak), VALUE,
                        sizeof(PLASMA_Complex64_t) * ib * T.nb, (T[k + 1][k]), OUTPUT,
                        sizeof(int), &(T.mb), VALUE,
                        sizeof(PLASMA_Complex64_t) * T.nb, (NULL), SCRATCH,
                        sizeof(PLASMA_Complex64_t) * ib * T.nb, (NULL), SCRATCH, 0);

		//LEFT and RIGHT on the symmetric diagonal block
#pragma PARSEC_DATA_ACCESS_MASK A 0b11
		QUARK_Insert_Task((plasma->quark), CORE_zherfb1_quark, (&task_flags),
                    sizeof(PLASMA_enum), &(PlasmaLower), VALUE,
                    sizeof(int), &(tempkn), VALUE,
                    sizeof(int), &(tempkn), VALUE,
                    sizeof(int), &(ib), VALUE,
                    sizeof(int), &(T.nb), VALUE,
                    sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[k + 1][k], INPUT,
                    sizeof(int), &(ldak), VALUE,
                    sizeof(PLASMA_Complex64_t) * ib * T.nb, T[k + 1][k], INPUT,
                    sizeof(int), &(T.mb), VALUE,
                    sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[k + 1][k + 1], INOUT,
                    sizeof(int), &(ldak), VALUE,
                    sizeof(PLASMA_Complex64_t) * 2 * T.nb * T.nb, NULL, SCRATCH,
                    sizeof(int), &(T.nb), VALUE, 0);

		//RIGHT on the remaining tiles until the bottom
		for (m = k + 2; m < A.mt; m++) {
			tempmm = m == A.mt - 1 ? A.m - m * A.mb : A.mb;
			ldam = BLKLDD(A, m);
#pragma PARSEC_DATA_ACCESS_MASK A 0b01
			QUARK_Insert_Task((plasma->quark), CORE_zunmqr_quark, (&task_flags),
                		sizeof(PLASMA_enum), &(PlasmaRight), VALUE,
                		sizeof(PLASMA_enum), &(PlasmaNoTrans), VALUE,
                		sizeof(int), &(tempmm), VALUE,
                		sizeof(int), &(A.nb), VALUE,
                		sizeof(int), &(tempkn), VALUE,
                		sizeof(int), &(ib), VALUE,
                		sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[k + 1][k]), INPUT,
                		sizeof(int), &(ldak), VALUE,
                		sizeof(PLASMA_Complex64_t) * ib * T.nb, (T[k + 1][k]), INPUT,
                		sizeof(int), &(T.mb), VALUE,
                		sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[m][k + 1]), INOUT,
                		sizeof(int), &(ldam), VALUE,
                		sizeof(PLASMA_Complex64_t) * ib * T.nb, (NULL), SCRATCH,
                		sizeof(int), &(T.nb), VALUE, 0);
		}

		for (m = k + 2; m < A.mt; m++) {
			int ldwork = T.nb;
			tempmm = m == A.mt - 1 ? A.m - m * A.mb : A.mb;
			ldam = BLKLDD(A, m);
			QUARK_Insert_Task((plasma->quark), CORE_ztsqrt_quark, (&task_flags),
                            sizeof(int), &(tempmm), VALUE,
                            sizeof(int), &(A.nb), VALUE,
                            sizeof(int), &(ib), VALUE,
                            sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[k + 1][k]), INOUT,
                            sizeof(int), &(ldak), VALUE,
                            sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[m][k]), INOUT,
                            sizeof(int), &(ldam), VALUE,
                            sizeof(PLASMA_Complex64_t) * ib * T.nb, (T[m][k]), OUTPUT,
                            sizeof(int), &(T.mb), VALUE,
                            sizeof(PLASMA_Complex64_t) * T.nb, (NULL), SCRATCH,
                            sizeof(PLASMA_Complex64_t) * ib * T.nb, (NULL), SCRATCH, 0);

			//LEFT
			for (i = k + 2; i < m; i++) {
				int ldwork = PlasmaLeft == PlasmaLeft ? ib : T.nb;
				ldai = BLKLDD(A, i);
				QUARK_Insert_Task(plasma->quark, CORE_ztsmqr1_quark, &task_flags,
					sizeof(PLASMA_enum), &PlasmaLeft, VALUE,
					sizeof(PLASMA_enum), &PlasmaConjTrans, VALUE,
					sizeof(int), &A.mb, VALUE,
					sizeof(int), &A.nb, VALUE,
					sizeof(int), &tempmm, VALUE,
					sizeof(int), &A.nb, VALUE,
					sizeof(int), &A.nb, VALUE,
					sizeof(int), &ib, VALUE,
					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[i][k + 1], INOUT,
					sizeof(int), &ldai, VALUE,
					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[m][i], INOUT,
					sizeof(int), &ldam, VALUE,
					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[m][k], INPUT,
					sizeof(int), &ldam, VALUE,
					sizeof(PLASMA_Complex64_t) * ib * T.nb, T[m][k], INPUT,
					sizeof(int), &T.mb, VALUE,
					sizeof(PLASMA_Complex64_t) * ib * T.nb, NULL, SCRATCH,
					sizeof(int), &ldwork, VALUE, 0);
			}

			//RIGHT
			for (j = m + 1; j < A.mt; j++) {
				int ldwork = PlasmaRight == PlasmaLeft ? ib : T.nb;
				tempjj = j == A.mt - 1 ? A.m - j * A.mb : A.mb;
				ldaj = BLKLDD(A, j);
				QUARK_Insert_Task((plasma->quark), CORE_ztsmqr_quark, (&task_flags),
    					sizeof(PLASMA_enum), &(PlasmaRight), VALUE,
    					sizeof(PLASMA_enum), &(PlasmaNoTrans), VALUE,
    					sizeof(int), &(tempjj), VALUE,
    					sizeof(int), &(A.nb), VALUE,
    					sizeof(int), &(tempjj), VALUE,
    					sizeof(int), &(tempmm), VALUE,
    					sizeof(int), &(A.nb), VALUE,
    					sizeof(int), &(ib), VALUE,
    					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[j][k + 1]), INOUT,
    					sizeof(int), &(ldaj), VALUE,
    					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[j][m]), INOUT,
    					sizeof(int), &(ldaj), VALUE,
    					sizeof(PLASMA_Complex64_t) * T.nb * T.nb, (A[m][k]), INPUT,
    					sizeof(int), &(ldam), VALUE,
    					sizeof(PLASMA_Complex64_t) * ib * T.nb, (T[m][k]), INPUT,
    					sizeof(int), &(T.mb), VALUE,
    					sizeof(PLASMA_Complex64_t) * ib * T.nb, (NULL), SCRATCH,
    					sizeof(int), &(ldwork), VALUE, 0);
			}

			//LEFT->RIGHT
			QUARK_Insert_Task(plasma->quark, CORE_ztsmqrlr_quark, &task_flags,
    				sizeof(int), &(A.nb), VALUE,
    				sizeof(int), &(A.nb), VALUE,
    				sizeof(int), &(tempmm), VALUE,
    				sizeof(int), &(A.nb), VALUE,
    				sizeof(int), &(tempmm), VALUE,
    				sizeof(int), &(tempmm), VALUE,
    				sizeof(int), &(A.nb), VALUE,
    				sizeof(int), &(ib), VALUE,
    				sizeof(int), &(T.nb), VALUE,
    				sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[k + 1][k + 1], INOUT,
    				sizeof(int), &(ldak), VALUE,
    				sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[m][k + 1], INOUT,
    				sizeof(int), &(ldam), VALUE,
    				sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[m][m], INOUT,
    				sizeof(int), &(ldam), VALUE,
    				sizeof(PLASMA_Complex64_t) * T.nb * T.nb, A[m][k], INPUT,
    				sizeof(int), &(ldam), VALUE,
    				sizeof(PLASMA_Complex64_t) * ib * T.nb, T[m][k], INPUT,
    				sizeof(int), &(T.mb), VALUE,
    				sizeof(PLASMA_Complex64_t) * 4 * T.nb * T.nb, NULL, SCRATCH,
    				sizeof(int), &(ldwork), VALUE, 0);

			//Clean up the Vs on the entire tile of the sub blocks
				// QUARK_CORE_zlaset2(
					       //plasma->quark, &task_flags,
				    //PlasmaUpperLower, tempmm, A.nb, zzero,
						      //A[m][k], ldam);
		}
		//Clean up the Vs on the lower part of the sub diagonal blocks
			// QUARK_CORE_zlaset2(
					      //plasma->quark, &task_flags,
					 //PlasmaLower, tempkn, A.nb, zzero,
					      //A[k + 1][k], ldak);
	}
}
