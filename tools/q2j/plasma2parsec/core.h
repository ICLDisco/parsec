#define QUARK_CORE_zbrdalg1(quark, task_flags, uplo, n, nb, A, lda, VQ, TAUQ, VP, TAUP, Vblksiz, wantz, i, sweepid, m, grsiz, PCOL, ACOL, MCOL) {\
    QUARK_Insert_Task((quark), CORE_zbrdalg1_quark, (task_flags),\
        sizeof(int),                   &(uplo), VALUE,\
        sizeof(int),                      &(n), VALUE,\
        sizeof(int),                     &(nb), VALUE,\
        sizeof(PLASMA_Complex64_t),        (A),    NODEP,\
        sizeof(int),                    &(lda), VALUE,\
        sizeof(PLASMA_Complex64_t),       (VQ),    NODEP,\
        sizeof(PLASMA_Complex64_t),     (TAUQ),    NODEP,\
        sizeof(PLASMA_Complex64_t),       (VP),    NODEP,\
        sizeof(PLASMA_Complex64_t),     (TAUP),    NODEP,\
        sizeof(int),                &(Vblksiz), VALUE,\
        sizeof(int),                  &(wantz), VALUE,\
        sizeof(int),                      &(i), VALUE,\
        sizeof(int),                &(sweepid), VALUE,\
        sizeof(int),                      &(m), VALUE,\
        sizeof(int),                  &(grsiz), VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,  (NULL),    SCRATCH,\
        sizeof(int),                    (PCOL),    INPUT,\
        sizeof(int),                    (ACOL),    INPUT,\
        sizeof(int),                    (MCOL),    OUTPUT | LOCALITY,\
        0);}
#pragma zbrdalg1 PCOL ACOL MCOL

#define QUARK_CORE_zgeadd(quark, task_flags, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_zgeadd_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             INOUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma zgeadd A B

#define QUARK_CORE_zgelqt(quark, task_flags, m, n, ib, nb, A, lda, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zgelqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma zgelqt A T

#define QUARK_CORE_zgemm(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zgemm A B C

#define QUARK_CORE_zgemm2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT | LOCALITY | GATHERV,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zgemm2 A B C

#define QUARK_CORE_zgemm_f2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task((quark), CORE_zgemm_f2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT | LOCALITY,\
        sizeof(int),                        &(ldc),       VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),             flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2),             flag2,\
        0);}
#pragma zgemm_f2 A B C

#define QUARK_CORE_zgemm_p2(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*ldc*nb,    (C),                 INOUT | LOCALITY,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zgemm_p2 A B C

#define QUARK_CORE_zgemm_p3(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p3_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*ldb*nb,   (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t*),         (C),                 INOUT | LOCALITY,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zgemm_p3 A B C

#define QUARK_CORE_zgemm_p2f1(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc, fake1, szefake1, flag1) {\
    QUARK_Insert_Task((quark), CORE_zgemm_p2f1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(transB),    VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*ldc*nb,    (C),                 INOUT | LOCALITY,\
        sizeof(int),                        &(ldc),       VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),             flag1,\
        0);}
#pragma zgemm_p2f1 A B C

#define QUARK_CORE_zgemm_tile(quark, task_flags, transA, transB, m, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc, Alock, Block, Clock) {\
    QUARK_Insert_Task((quark), CORE_zgemm_tile_quark, (task_flags),\
        sizeof(PLASMA_enum),              &(transA), VALUE,\
        sizeof(PLASMA_enum),              &(transB), VALUE,\
        sizeof(int),                      &(m),      VALUE,\
        sizeof(int),                      &(n),      VALUE,\
        sizeof(int),                      &(k),      VALUE,\
        sizeof(PLASMA_Complex64_t),       (alpha),           INPUT,\
        sizeof(PLASMA_Complex64_t)*nb*nb, (A),               NODEP,          /* input; see Alock */\
        sizeof(int),                      &(lda),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb, (B),               NODEP,          /* input; see Block */\
        sizeof(int),                      &(ldb),    VALUE,\
        sizeof(PLASMA_Complex64_t),       (beta),            INPUT,\
        sizeof(PLASMA_Complex64_t)*nb*nb, (C),                       NODEP,  /* inout; see Clock */\
        sizeof(int),                      &(ldc),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb, (Alock),           INPUT,\
        sizeof(PLASMA_Complex64_t)*nb,    (Block),           INPUT,\
        sizeof(PLASMA_Complex64_t)*nb,    (Clock),                   INOUT,\
        0);}
#pragma zgemm_tile alpha beta Alock Block Clock

#define QUARK_CORE_zgemv(quark, task_flags, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) {\
    QUARK_Insert_Task((quark), CORE_zgemv_quark, (task_flags),\
        sizeof(PLASMA_enum),             &(trans),  VALUE,\
        sizeof(int),                     &(m),      VALUE,\
        sizeof(int),                     &(n),      VALUE,\
        sizeof(PLASMA_Complex64_t),      &(alpha),  VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n,  (A),               INPUT,\
        sizeof(int),                     &(lda),    VALUE,\
        sizeof(PLASMA_Complex64_t)*n,    (x),               INPUT,\
        sizeof(int),                     &(incx),   VALUE,\
        sizeof(PLASMA_Complex64_t),      &(beta),   VALUE,\
        sizeof(PLASMA_Complex64_t)*m,    (y),               INOUT,\
        sizeof(int),                     &(incy),   VALUE,\
        0);}
#pragma zgemv A x y

#define QUARK_CORE_zgemv_tile(quark, task_flags, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, Alock, xlock, ylock) {\
    /* Quick return. Bad things happen if sizeof(...)*m*n is zero in QUARK_Insert_Task */\
    if ( m == 0 || n == 0 )\
        return;\
\
    QUARK_Insert_Task((quark), CORE_zgemv_tile_quark, (task_flags),\
        sizeof(PLASMA_enum),             &(trans),  VALUE,\
        sizeof(int),                     &(m),      VALUE,\
        sizeof(int),                     &(n),      VALUE,\
        sizeof(PLASMA_Complex64_t),      (alpha),           INPUT,\
        sizeof(PLASMA_Complex64_t)*m*n,  (A),               NODEP,          /* input; see Alock */\
        sizeof(int),                     &(lda),    VALUE,\
        sizeof(PLASMA_Complex64_t)*n,    (x),               NODEP,          /* input; see xlock */\
        sizeof(int),                     &(incx),   VALUE,\
        sizeof(PLASMA_Complex64_t),      (beta),            INPUT,\
        sizeof(PLASMA_Complex64_t)*m,    (y),                       NODEP,  /* inout; see ylock */\
        sizeof(int),                     &(incy),   VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n,  (Alock),           INPUT,\
        sizeof(PLASMA_Complex64_t)*n,    (xlock),           INPUT,\
        sizeof(PLASMA_Complex64_t)*m,    (ylock),                   INOUT,\
        0);}
#pragma zgemv_tile alpha beta Alock xlock ylock

#define QUARK_CORE_zgeqp3_tntpiv(quark, task_flags, m, n, nb, A, lda, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgeqp3_tntpiv_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_Complex64_t)*min(m,n),    (NULL),               SCRATCH,\
        sizeof(int)*n,                          (NULL),               SCRATCH,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma zgeqp3_tntpiv A IPIV

#define QUARK_CORE_zgeqp3_update(quark, task_flags, Ajj, lda1, Ajk, lda2, Fk, ldf, joff, k, koff, nb, norms1, norms2, info ) {\
    QUARK_Insert_Task(\
        quark, (CORE_zgeqp3_update_quark), task_flags,\
        sizeof(PLASMA_Complex64_t)*nb*nb,  (Ajj),             INPUT,\
        sizeof(int),                       &(lda1),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,  (Ajk),                     INOUT,\
        sizeof(int),                       &(lda2),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,  (Fk),              INPUT,\
        sizeof(int),                       &(ldf),    VALUE,\
        sizeof(int),                       &(joff),   VALUE,\
        sizeof(int),                       &(k),      VALUE,\
        sizeof(int),                       &(koff),   VALUE,\
        sizeof(int),                       &(nb),     VALUE,\
        sizeof(double)*nb,                 (norms1),                  INOUT,\
        sizeof(double)*nb,                 norms2,                  (NODEP),  /* INOUT, but implied by norms1 */\
        sizeof(int),                       (info),                    OUTPUT,\
        0);}
#pragma zgeqp3_update Ajj Ajk Fk norms1 info

#define QUARK_CORE_zgeqrt(quark, task_flags, m, n, ib, nb, A, lda, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zgeqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma zgeqrt A T

#define QUARK_CORE_zgessm(quark, task_flags, m, n, k, ib, nb, IPIV, L, ldl, A, lda) {\
    QUARK_Insert_Task((quark), CORE_zgessm_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int)*nb,                      (IPIV),          INPUT,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (L),             INPUT | QUARK_REGION_L,\
        sizeof(int),                        &(ldl),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        0);}
#pragma zgessm IPIV A

#define QUARK_CORE_zgetrf(quark, task_flags, m, n, nb, A, lda, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma zgetrf A IPIV

#define QUARK_CORE_zgetrf_incpiv(quark, task_flags, m, n, ib, nb, A, lda, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_incpiv_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(int),                        &(ib),            VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma zgetrf_incpiv A IPIV

#define QUARK_CORE_zgetrf_nopiv(quark, task_flags, m, n, ib, nb, A, lda, sequence, request, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_nopiv_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(int),                        &(ib),            VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma zgetrf_nopiv A

#define QUARK_CORE_zgetrf_reclap(quark, task_flags, m, n, nb, A, lda, IPIV, sequence, request, check_info, iinfo, nbthread) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_reclap_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        sizeof(int),                        &(nbthread),      VALUE,\
        0);}
#pragma zgetrf_reclap A IPIV

#define QUARK_CORE_zgetrf_rectil(quark, task_flags, A, Amn, size, IPIV, sequence, request, check_info, iinfo, nbthread) {\
    QUARK_Insert_Task((quark), CORE_zgetrf_rectil_quark, (task_flags),\
        sizeof(PLASMA_desc),                &(A),             VALUE,\
        sizeof(PLASMA_Complex64_t)*size,     (Amn),               INOUT,\
        sizeof(int)*A.n,                     (IPIV),              OUTPUT,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        sizeof(int),                        &(nbthread),      VALUE,\
        0);}
#pragma zgetrf_rectil Amn IPIV

#define QUARK_CORE_zgetrip(quark, task_flags, m, n, A, szeA) {\
    QUARK_Insert_Task((quark), CORE_zgetrip_quark, (task_flags),\
        sizeof(int),                     &(m),   VALUE,\
        sizeof(int),                     &(n),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*szeA, (NULL),     SCRATCH,\
        0);}
#pragma zgetrip A

#define QUARK_CORE_zhemm(quark, task_flags, side, uplo, m, n, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zhemm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),    VALUE,\
        sizeof(PLASMA_enum),                &(uplo),    VALUE,\
        sizeof(int),                        &(m),       VALUE,\
        sizeof(int),                        &(n),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),               INPUT,\
        sizeof(int),                        &(lda),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),               INPUT,\
        sizeof(int),                        &(ldb),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),               INOUT,\
        sizeof(int),                        &(ldc),     VALUE,\
        0);}
#pragma zhemm A B C

#define QUARK_CORE_zher2k(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zher2k_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(double),                     &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zher2k A B C

#define QUARK_CORE_zherfb(quark, task_flags, uplo, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    /* TODO: Understand why A needs to be INOUT and not INPUT */\
    QUARK_Insert_Task(\
        quark, (CORE_zherfb_quark), task_flags,\
        sizeof(PLASMA_enum),                     &(uplo),  VALUE,\
        sizeof(int),                             &(n),     VALUE,\
        sizeof(int),                             &(k),     VALUE,\
        sizeof(int),                             &(ib),    VALUE,\
        sizeof(int),                             &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,        (A),          uplo == PlasmaUpper ? INOUT|QUARK_REGION_U : INOUT|QUARK_REGION_L,\
        sizeof(int),                             &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,        (T),          INPUT,\
        sizeof(int),                             &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,        (C),          uplo == PlasmaUpper ? INOUT|QUARK_REGION_D|QUARK_REGION_U : INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                             &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*2*nb*nb,    (NULL),         SCRATCH,\
        sizeof(int),                             &(nb),    VALUE,\
        0);}
#pragma zherfb T

#define QUARK_CORE_zherk(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zherk_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(double),                     &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(double),                     &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zherk A C

#define QUARK_CORE_zlacpy(quark, task_flags, uplo, m, n, nb, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_zlacpy_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             OUTPUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma zlacpy A B

#define QUARK_CORE_zlacpy_f1(quark, task_flags, uplo, m, n, nb, A, lda, B, ldb, fake1, szefake1, flag1) {\
    if ( fake1 == B ) {\
        QUARK_Insert_Task((quark), CORE_zlacpy_quark, (task_flags),\
            sizeof(PLASMA_enum),                &(uplo),  VALUE,\
            sizeof(int),                        &(m),     VALUE,\
            sizeof(int),                        &(n),     VALUE,\
            sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
            sizeof(int),                        &(lda),   VALUE,\
            sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             OUTPUT | flag1,\
            sizeof(int),                        &(ldb),   VALUE,\
            0);}
#pragma zlacpy_f1 A

#define QUARK_CORE_zlacpy_pivot(quark, task_flags, descA, direct, k1, k2, ipiv, rankin, rankout, A, lda, pos, init) {\
    QUARK_Insert_Task((quark), CORE_zlacpy_pivot_quark, (task_flags),\
        sizeof(PLASMA_desc),                    &(descA),         VALUE,\
        sizeof(PLASMA_enum),                    &(direct),        VALUE,\
        sizeof(int),                            &(k1),            VALUE,\
        sizeof(int),                            &(k2),            VALUE,\
        sizeof(int)*lda,                         (ipiv),                INPUT,\
        sizeof(int)*lda,                         (rankin),              INOUT,\
        sizeof(int)*lda,                         (rankout),             OUTPUT | GATHERV,\
        sizeof(PLASMA_Complex64_t)*lda*descA.nb, (A),                   INOUT | GATHERV,\
        sizeof(int),                            &(lda),           VALUE,\
        sizeof(int),                            &(pos),           VALUE,\
        sizeof(int),                            &(init),          VALUE,\
        0);}
#pragma zlacpy_pivot ipiv rankin rankout A

#define QUARK_CORE_zlag2c(quark, task_flags, m, n, nb, A, lda, B, ldb, sequence, request) {\
    QUARK_Insert_Task((quark), CORE_zlag2c_quark, (task_flags),\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex32_t)*nb*nb,    (B),                 OUTPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma zlag2c A B

#define QUARK_CORE_clag2z(quark, task_flags, m, n, nb, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_clag2z_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex32_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             INOUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma clag2z A B

#define QUARK_CORE_zlange(quark, task_flags, norm, M, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlange_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        0);}
#pragma zlange A result

#define QUARK_CORE_zlange_f1(quark, task_flags, norm, M, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
\
    if ( result == fake ) {\
        QUARK_Insert_Task((quark), CORE_zlange_quark, (task_flags),\
            sizeof(PLASMA_enum),                &(norm),  VALUE,\
            sizeof(int),                        &(M),     VALUE,\
            sizeof(int),                        &(N),     VALUE,\
            sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
            sizeof(int),                        &(LDA),   VALUE,\
            sizeof(double)*szeW,                 (NULL),          SCRATCH,\
            sizeof(double),                      (result),        OUTPUT | GATHERV,\
            0);}
#pragma zlange_f1 A result

#define QUARK_CORE_zlanhe(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlanhe_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                     (result),         OUTPUT,\
        0);}
#pragma zlanhe A result

#define QUARK_CORE_zlanhe_f1(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
\
    if ( result == fake ) {\
        QUARK_Insert_Task((quark), CORE_zlanhe_quark, (task_flags),\
            sizeof(PLASMA_enum),                &(norm),  VALUE,\
            sizeof(PLASMA_enum),                &(uplo),  VALUE,\
            sizeof(int),                        &(N),     VALUE,\
            sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
            sizeof(int),                        &(LDA),   VALUE,\
            sizeof(double)*szeW,                 (NULL),          SCRATCH,\
            sizeof(double)*szeF,                 (result),        OUTPUT | GATHERV,\
            0);}
#pragma zlanhe_f1 A result

#define QUARK_CORE_zlansy(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result) {\
    szeW = max(1, szeW);\
    QUARK_Insert_Task((quark), CORE_zlansy_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(norm),  VALUE,\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        sizeof(double)*szeW,                 (NULL),          SCRATCH,\
        sizeof(double),                      (result),        OUTPUT,\
        0);}
#pragma zlansy A result

#define QUARK_CORE_zlansy_f1(quark, task_flags, norm, uplo, N, A, LDA, szeA, szeW, result, fake, szeF) {\
    szeW = max(1, szeW);\
\
    if ( result == fake ) {\
        QUARK_Insert_Task((quark), CORE_zlansy_quark, (task_flags),\
            sizeof(PLASMA_enum),                &(norm),  VALUE,\
            sizeof(PLASMA_enum),                &(uplo),  VALUE,\
            sizeof(int),                        &(N),     VALUE,\
            sizeof(PLASMA_Complex64_t)*szeA,     (A),             INPUT,\
            sizeof(int),                        &(LDA),   VALUE,\
            sizeof(double)*szeW,                 (NULL),          SCRATCH,\
            sizeof(double)*szeF,                 (result),        OUTPUT | GATHERV,\
            0);}
#pragma zlansy_f1 A result

#define QUARK_CORE_zlaset2(quark, task_flags, uplo, M, N, alpha, A, LDA) {\
    QUARK_Insert_Task((quark), CORE_zlaset2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t)*M*N,     (A),      OUTPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        0);}
#pragma zlaset2 A

#define QUARK_CORE_zlaset(quark, task_flags, uplo, M, N, alpha, beta, A, LDA) {\
    QUARK_Insert_Task((quark), CORE_zlaset_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(M),     VALUE,\
        sizeof(int),                        &(N),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha), VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),  VALUE,\
        sizeof(PLASMA_Complex64_t)*LDA*N,    (A),      OUTPUT,\
        sizeof(int),                        &(LDA),   VALUE,\
        0);}
#pragma zlaset A

#define QUARK_CORE_zlaswp(quark, task_flags, n, A, lda, i1, i2, ipiv, inc) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_quark), task_flags,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n,  (A),        INOUT | LOCALITY,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(i1),   VALUE,\
        sizeof(int),                      &(i2),   VALUE,\
        sizeof(int)*n,                     (ipiv),     INPUT,\
        sizeof(int),                      &(inc),  VALUE,\
        0);}
#pragma zlaswp A ipiv

#define QUARK_CORE_zlaswp_f2(quark, task_flags, n, A, lda, i1, i2, ipiv, inc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_f2_quark), task_flags,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n,    (A),         INOUT | LOCALITY,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(int),                        &(i1),    VALUE,\
        sizeof(int),                        &(i2),    VALUE,\
        sizeof(int)*n,                       (ipiv),      INPUT,\
        sizeof(int),                        &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1),     flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2),     flag2,\
        0);}
#pragma zlaswp_f2 A ipiv

#define QUARK_CORE_zlaswp_ontile(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, fakepanel) {\
    if (fakepanel == Aij) {\
        QUARK_Insert_Task(\
            quark, (CORE_zlaswp_ontile_quark), task_flags,\
            sizeof(PLASMA_desc),              &(descA),     VALUE,\
            sizeof(PLASMA_Complex64_t)*1,      (Aij),           INOUT | LOCALITY,\
            sizeof(int),                      &(i1),        VALUE,\
            sizeof(int),                      &(i2),        VALUE,\
            sizeof(int)*(i2-i1+1)*abs(inc),   (ipiv),           INPUT,\
            sizeof(int),                      &(inc),       VALUE,\
            sizeof(PLASMA_Complex64_t)*1,      (fakepanel),     SCRATCH,\
            0);}
#pragma zlaswp_ontile Aij ipiv

#define QUARK_CORE_zlaswp_ontile_f2(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, fake1, szefake1, flag1, fake2, szefake2, flag2) {\
    QUARK_Insert_Task(\
        quark, (CORE_zlaswp_ontile_f2_quark), task_flags,\
        sizeof(PLASMA_desc),                &(descA), VALUE,\
        sizeof(PLASMA_Complex64_t)*1,        (Aij),       INOUT | LOCALITY,\
        sizeof(int),                        &(i1),    VALUE,\
        sizeof(int),                        &(i2),    VALUE,\
        sizeof(int)*(i2-i1+1)*abs(inc),      (ipiv),      INPUT,\
        sizeof(int),                        &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*szefake1, (fake1), flag1,\
        sizeof(PLASMA_Complex64_t)*szefake2, (fake2), flag2,\
        0);}
#pragma zlaswp_ontile_f2 Aij ipiv

#define QUARK_CORE_zswptr_ontile(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, Akk, ldak) {\
    QUARK_Insert_Task(\
        quark, (CORE_zswptr_ontile_quark), task_flags,\
        sizeof(PLASMA_desc),              &(descA), VALUE,\
        sizeof(PLASMA_Complex64_t)*1,      (Aij),       INOUT | LOCALITY,\
        sizeof(int),                      &(i1),    VALUE,\
        sizeof(int),                      &(i2),    VALUE,\
        sizeof(int)*(i2-i1+1)*abs(inc),    (ipiv),      INPUT,\
        sizeof(int),                      &(inc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ldak,   (Akk),       INPUT,\
        sizeof(int),                      &(ldak),  VALUE,\
        0);}
#pragma zswptr_ontile Aij ipiv Akk

#define QUARK_CORE_zlaswpc_ontile(quark, task_flags, descA, Aij, i1, i2, ipiv, inc, fakepanel) {\
    if (fakepanel == Aij) {\
        QUARK_Insert_Task(\
            quark, (CORE_zlaswpc_ontile_quark), task_flags,\
            sizeof(PLASMA_desc),              &(descA),     VALUE,\
            sizeof(PLASMA_Complex64_t)*1,      (Aij),           INOUT | LOCALITY,\
            sizeof(int),                      &(i1),        VALUE,\
            sizeof(int),                      &(i2),        VALUE,\
            sizeof(int)*(i2-i1+1)*abs(inc),   (ipiv),           INPUT,\
            sizeof(int),                      &(inc),       VALUE,\
            sizeof(PLASMA_Complex64_t)*1,      (fakepanel),     SCRATCH,\
            0);}
#pragma zlaswpc_ontile Aij ipiv

#define QUARK_CORE_zlatro(quark, task_flags, uplo, trans, m, n, nb, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_zlatro_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             OUTPUT,\
        sizeof(int),                        &(ldb),   VALUE,\
        0);}
#pragma zlatro A B

#define QUARK_CORE_zlatro_f1(quark, task_flags, uplo, trans, m, n, nb, A, lda, B, ldb, fake1, szefake1, flag1) {\
    if ( fake1 == B ) {\
        QUARK_Insert_Task((quark), CORE_zlatro_quark, (task_flags),\
            sizeof(PLASMA_enum),                &(uplo),  VALUE,\
            sizeof(PLASMA_enum),                &(trans), VALUE,\
            sizeof(int),                        &(m),     VALUE,\
            sizeof(int),                        &(n),     VALUE,\
            sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
            sizeof(int),                        &(lda),   VALUE,\
            sizeof(PLASMA_Complex64_t)*nb*nb,    (B),             OUTPUT,\
            sizeof(int),                        &(ldb),   VALUE,\
            0);}
#pragma zlatro_f1 A B

#define QUARK_CORE_zlauum(quark, task_flags, uplo, n, nb, A, lda) {\
    QUARK_Insert_Task((quark), CORE_zlauum_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),  VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INOUT,\
        sizeof(int),                        &(lda),   VALUE,\
        0);}
#pragma zlauum A

#define QUARK_CORE_zpamm(quark, task_flags, op, side, storev, m, n, k, l, A1, lda1, A2, lda2, V, ldv, W, ldw) {\
    QUARK_Insert_Task((quark), CORE_zpamm_quark, (task_flags),\
        sizeof(int),                        &(op),      VALUE,\
        sizeof(PLASMA_enum),                &(side),    VALUE,\
        sizeof(PLASMA_enum),                &(storev),  VALUE,\
        sizeof(int),                        &(m),       VALUE,\
        sizeof(int),                        &(n),       VALUE,\
        sizeof(int),                        &(k),       VALUE,\
        sizeof(int),                        &(l),       VALUE,\
        sizeof(PLASMA_Complex64_t)*m*k,     (A1),           INPUT,\
        sizeof(int),                        &(lda1),    VALUE,\
        sizeof(PLASMA_Complex64_t)*k*n,     (A2),           INOUT,\
        sizeof(int),                        &(lda2),    VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n,     (V),            INPUT,\
        sizeof(int),                        &(ldv),     VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n,     (W),            INOUT,\
        sizeof(int),                        &(ldw),     VALUE,\
        0);}
#pragma zpamm A1 A2 V W

#define QUARK_CORE_zplghe(quark, task_flags, bump, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplghe_quark, (task_flags),\
        sizeof(double),                   &(bump), VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma zplghe A

#define QUARK_CORE_zplgsy(quark, task_flags, bump, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplgsy_quark, (task_flags),\
        sizeof(PLASMA_Complex64_t),       &(bump), VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma zplgsy A

#define QUARK_CORE_zplrnt(quark, task_flags, m, n, A, lda, bigM, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zplrnt_quark, (task_flags),\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),         OUTPUT,\
        sizeof(int),                      &(lda),  VALUE,\
        sizeof(int),                      &(bigM), VALUE,\
        sizeof(int),                      &(m0),   VALUE,\
        sizeof(int),                      &(n0),   VALUE,\
        sizeof(unsigned long long int),   &(seed), VALUE,\
        0);}
#pragma zplrnt A

#define QUARK_CORE_zpltmg(quark, task_flags, mtxtype, m, n, A, lda, gM, gN, m0, n0, seed) {\
    QUARK_Insert_Task((quark), CORE_zpltmg_quark, (task_flags),\
        sizeof(int),                      &(mtxtype), VALUE,\
        sizeof(int),                      &(m),       VALUE,\
        sizeof(int),                      &(n),       VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*n, (A),            OUTPUT,\
        sizeof(int),                      &(lda),     VALUE,\
        sizeof(int),                      &(gM),      VALUE,\
        sizeof(int),                      &(gN),      VALUE,\
        sizeof(int),                      &(m0),      VALUE,\
        sizeof(int),                      &(n0),      VALUE,\
        sizeof(unsigned long long int),   &(seed),    VALUE,\
        0);}
#pragma zpltmg A


#define QUARK_CORE_zpotrf(quark, task_flags, uplo, n, nb, A, lda, sequence, request, iinfo) {\
    QUARK_Insert_Task((quark), CORE_zpotrf_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INOUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(int),                        &(iinfo),     VALUE,\
        0);}
#pragma zpotrf A

#define QUARK_CORE_zshiftw(quark, task_flags, s, cl, m, n, L, A, W) {\
    QUARK_Insert_Task((quark), CORE_zshiftw_quark, (task_flags),\
        sizeof(int),                      &(s),   VALUE,\
        sizeof(int),                      &(cl),  VALUE,\
        sizeof(int),                      &(m),   VALUE,\
        sizeof(int),                      &(n),   VALUE,\
        sizeof(int),                      &(L),   VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n*L, (A),        INOUT,\
        sizeof(PLASMA_Complex64_t)*L,     (W),        INPUT,\
        0);}
#pragma zshiftw A W

#define QUARK_CORE_zshift(quark, task_flags, s, m, n, L, A) {\
    QUARK_Insert_Task((quark), CORE_zshift_quark, (task_flags),\
        sizeof(int),                      &(s),    VALUE,\
        sizeof(int),                      &(m),    VALUE,\
        sizeof(int),                      &(n),    VALUE,\
        sizeof(int),                      &(L),    VALUE,\
        sizeof(PLASMA_Complex64_t)*m*n*L, (A),        INOUT | GATHERV,\
        sizeof(PLASMA_Complex64_t)*L,     (NULL),     SCRATCH,\
        0);}
#pragma zshift A

#define QUARK_CORE_zssssm(quark, task_flags, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, L1, ldl1, L2, ldl2, IPIV) {\
    QUARK_Insert_Task((quark), CORE_zssssm_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L1),            INPUT,\
        sizeof(int),                        &(ldl1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L2),            INPUT,\
        sizeof(int),                        &(ldl2),  VALUE,\
        sizeof(int)*nb,                      (IPIV),          INPUT,\
        0);}
#pragma zssssm A1 A2 L1 L2 IPIV

#define QUARK_CORE_zswpab(quark, task_flags, i, n1, n2, A, szeA) {\
    QUARK_Insert_Task(\
        quark, (CORE_zswpab_quark), task_flags,\
        sizeof(int),                           &(i),   VALUE,\
        sizeof(int),                           &(n1),  VALUE,\
        sizeof(int),                           &(n2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*szeA,       (A),            INOUT,\
        sizeof(PLASMA_Complex64_t)*min(n1,n2), (NULL),         SCRATCH,\
        0);}
#pragma zswpab A

#define QUARK_CORE_zsymm(quark, task_flags, side, uplo, m, n, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsymm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),    VALUE,\
        sizeof(PLASMA_enum),                &(uplo),    VALUE,\
        sizeof(int),                        &(m),       VALUE,\
        sizeof(int),                        &(n),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),               INPUT,\
        sizeof(int),                        &(lda),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),               INPUT,\
        sizeof(int),                        &(ldb),     VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),               INOUT,\
        sizeof(int),                        &(ldc),     VALUE,\
        0);}
#pragma zsymm A B C

#define QUARK_CORE_zsyr2k(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, B, ldb, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsyr2k_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INPUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zsyr2k A B C

#define QUARK_CORE_zsyrk(quark, task_flags, uplo, trans, n, k, nb, alpha, A, lda, beta, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zsyrk_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(trans),     VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(int),                        &(k),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t),         &(beta),      VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),                 INOUT,\
        sizeof(int),                        &(ldc),       VALUE,\
        0);}
#pragma zsyrk A C

#define QUARK_CORE_ztrdalg1(quark, task_flags, n, nb, A, lda, V, TAU, Vblksiz, wantz, i, sweepid, m, grsiz, PCOL, ACOL, MCOL) {\
    QUARK_Insert_Task((quark), CORE_ztrdalg1_quark,   (task_flags),\
        sizeof(int),                      &(n), VALUE,\
        sizeof(int),                     &(nb), VALUE,\
        sizeof(PLASMA_Complex64_t),        (A),   NODEP,\
        sizeof(int),                    &(lda), VALUE,\
        sizeof(PLASMA_Complex64_t),        (V),   NODEP,\
        sizeof(PLASMA_Complex64_t),      (TAU),   NODEP,\
        sizeof(int),                &(Vblksiz), VALUE,\
        sizeof(int),                  &(wantz), VALUE,\
        sizeof(int),                      &(i), VALUE,\
        sizeof(int),                &(sweepid), VALUE,\
        sizeof(int),                      &(m), VALUE,\
        sizeof(int),                  &(grsiz), VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,  (NULL),   SCRATCH,\
        sizeof(int),                    (PCOL), INPUT,\
        sizeof(int),                    (ACOL), INPUT,\
        sizeof(int),                    (MCOL), OUTPUT | LOCALITY,\
        0);}
#pragma ztrdalg1 PCOL ACOL MCOL

#define QUARK_CORE_ztrmm(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrmm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma ztrmm A B

#define QUARK_CORE_ztrmm_p2(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrmm_p2_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*lda*nb,   (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t*),         (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma ztrmm_p2 A B

#define QUARK_CORE_ztrsm(quark, task_flags, side, uplo, transA, diag, m, n, nb, alpha, A, lda, B, ldb) {\
    QUARK_Insert_Task((quark), CORE_ztrsm_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),      VALUE,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(transA),    VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(m),         VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t),         &(alpha),     VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INPUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (B),                 INOUT,\
        sizeof(int),                        &(ldb),       VALUE,\
        0);}
#pragma ztrsm A B

#define QUARK_CORE_ztrtri(quark, task_flags, uplo, diag, n, nb, A, lda, sequence, request, iinfo) {\
    QUARK_Insert_Task(\
        quark, (CORE_ztrtri_quark), task_flags,\
        sizeof(PLASMA_enum),                &(uplo),      VALUE,\
        sizeof(PLASMA_enum),                &(diag),      VALUE,\
        sizeof(int),                        &(n),         VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                 INOUT,\
        sizeof(int),                        &(lda),       VALUE,\
        sizeof(int),                        &(iinfo),     VALUE,\
        0);}
#pragma ztrtri A

#define QUARK_CORE_ztslqt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_ztslqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma ztslqt A2 T

#define QUARK_CORE_ztsmlq(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmlq A1 A2 V T

#define QUARK_CORE_ztsmlq_corner(quark, task_flags, m1, n1, m2, n2, m3, n3, k, ib, nb, A1, lda1, A2, lda2, A3, lda3, V, ldv, T, ldt) {\
    int ldwork = nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_corner_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(m3),    VALUE,\
        sizeof(int),                        &(n3),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int),                        &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A3),            INOUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(lda3),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*4*nb*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmlq_corner A2 V T

#define QUARK_CORE_ztsmlq_hetra1(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmlq_hetra1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_U|QUARK_REGION_D,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmlq_hetra1 A2 V T

#define QUARK_CORE_ztsmqr(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmqr A1 A2 V T

#define QUARK_CORE_ztsmqr_corner(quark, task_flags, m1, n1, m2, n2, m3, n3, k, ib, nb, A1, lda1, A2, lda2, A3, lda3, V, ldv, T, ldt) {\
    int ldwork = nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_corner_quark, (task_flags),\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(m3),    VALUE,\
        sizeof(int),                        &(n3),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(int),                        &(nb),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A3),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda3),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*4*nb*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmqr_corner A2 V T

#define QUARK_CORE_ztsmqr_hetra1(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_ztsmqr_hetra1_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_L|QUARK_REGION_D,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork), VALUE,\
        0);}
#pragma ztsmqr_hetra1 A2 V T

#define QUARK_CORE_ztsqrt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_ztsqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT | QUARK_REGION_D | QUARK_REGION_U,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT | LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma ztsqrt A2 T

#define QUARK_CORE_ztstrf(quark, task_flags, m, n, ib, nb, U, ldu, A, lda, L, ldl, IPIV, sequence, request, check_info, iinfo) {\
    QUARK_Insert_Task((quark), CORE_ztstrf_quark, (task_flags),\
        sizeof(int),                        &(m),             VALUE,\
        sizeof(int),                        &(n),             VALUE,\
        sizeof(int),                        &(ib),            VALUE,\
        sizeof(int),                        &(nb),            VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (U),                     INOUT | QUARK_REGION_D | QUARK_REGION_U,\
        sizeof(int),                        &(ldu),           VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),                     INOUT | LOCALITY,\
        sizeof(int),                        &(lda),           VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (L),                     OUTPUT,\
        sizeof(int),                        &(ldl),           VALUE,\
        sizeof(int)*nb,                      (IPIV),                  OUTPUT,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),                  SCRATCH,\
        sizeof(int),                        &(nb),            VALUE,\
        sizeof(PLASMA_bool),                &(check_info),    VALUE,\
        sizeof(int),                        &(iinfo),         VALUE,\
        0);}
#pragma ztstrf A L IPIV

#define QUARK_CORE_zttlqt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zttlqt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_L,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT|QUARK_REGION_D|QUARK_REGION_L|LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma zttlqt T

#define QUARK_CORE_zttmlq(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_zttmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork),    VALUE,\
        0);}
#pragma zttmlq A1 A2 V T

#define QUARK_CORE_zttmqr(quark, task_flags, side, trans, m1, n1, m2, n2, k, ib, nb, A1, lda1, A2, lda2, V, ldv, T, ldt) {\
    int ldwork = side == PlasmaLeft ? ib : nb;\
\
    QUARK_Insert_Task((quark), CORE_zttmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m1),    VALUE,\
        sizeof(int),                        &(n1),    VALUE,\
        sizeof(int),                        &(m2),    VALUE,\
        sizeof(int),                        &(n2),    VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (V),             INPUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(ldv),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(ldwork),    VALUE,\
        0);}
#pragma zttmqr A1 A2 V T

#define QUARK_CORE_zttqrt(quark, task_flags, m, n, ib, nb, A1, lda1, A2, lda2, T, ldt) {\
    QUARK_Insert_Task((quark), CORE_zttqrt_quark, (task_flags),\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A1),            INOUT|QUARK_REGION_D|QUARK_REGION_U,\
        sizeof(int),                        &(lda1),  VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A2),            INOUT|QUARK_REGION_D|QUARK_REGION_U|LOCALITY,\
        sizeof(int),                        &(lda2),  VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             OUTPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb,       (NULL),          SCRATCH,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        0);}
#pragma zttqrt A1 A2 T

#define QUARK_CORE_zunmlq(quark, task_flags, side, trans, m, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zunmlq_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (A),             INPUT,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (T),             INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,    (C),             INOUT,\
        sizeof(int),                        &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,    (NULL),          SCRATCH,\
        sizeof(int),                        &(nb),    VALUE,\
        0);}
#pragma zunmlq A T C

#define QUARK_CORE_zunmqr(quark, task_flags, side, trans, m, n, k, ib, nb, A, lda, T, ldt, C, ldc) {\
    QUARK_Insert_Task((quark), CORE_zunmqr_quark, (task_flags),\
        sizeof(PLASMA_enum),                &(side),  VALUE,\
        sizeof(PLASMA_enum),                &(trans), VALUE,\
        sizeof(int),                        &(m),     VALUE,\
        sizeof(int),                        &(n),     VALUE,\
        sizeof(int),                        &(k),     VALUE,\
        sizeof(int),                        &(ib),    VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,   (A),      INPUT | QUARK_REGION_L,\
        sizeof(int),                        &(lda),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,   (T),      INPUT,\
        sizeof(int),                        &(ldt),   VALUE,\
        sizeof(PLASMA_Complex64_t)*nb*nb,   (C),      INOUT,\
        sizeof(int),                        &(ldc),   VALUE,\
        sizeof(PLASMA_Complex64_t)*ib*nb,   (NULL),   SCRATCH,\
        sizeof(int),                        &(nb),    VALUE,\
        0);}
#pragma zunmqr T C

