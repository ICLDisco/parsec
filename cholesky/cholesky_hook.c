
/**
 * PLASMA include for defined and constants.
 */
#include <plasma.h>
#include <core_dblas.h>

#include "dplasma.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef DPLASMA_PROFILING
#include "profiling.h"
int POTRF_start_key, POTRF_end_key;
int SYRK_start_key, SYRK_end_key;
int GEMM_start_key, GEMM_end_key;
int TRSM_start_key, TRSM_end_key;
#define TAKE_TIME(KEY)  dplasma_profiling_trace((KEY))
#else
#define TAKE_TIME(KEY)
#endif  /* DPLASMA_PROFILING */

extern PLASMA_desc descA;
int PLASMA_INFO;

#define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]

#if 0
#define OUTPUT(ARG)  printf ARG
#else
#define OUTPUT(ARG)
#endif

/* Define it to shortcut the lookup for the local variables. */
#define DPLASMA_HOOK_OPTIMIZED

static int NB;

int POTRF_hook(const dplasma_execution_context_t* exec_context)
{
    int k, rc = 0;

    TAKE_TIME(POTRF_start_key);

#ifndef DPLASMA_HOOK_OPTIMIZED
    {
        assignment_t* pk;
        rc = dplasma_find_assignment("k", exec_context->locals, MAX_LOCAL_COUNT, &pk);
        if( 0 != rc ) {
            return rc;
        }
        k = pk->value;
    }
#else
    k = exec_context->locals[0].value;
#endif

#ifdef DPLASMA_EXECUTE
    CORE_dpotrf( PlasmaLower,
                 NB, /*k == A.nt-1 ? A.n-k*A.nb : A.nb,*/
                 A(k, k), NB, /*A.nb,*/
                 &PLASMA_INFO); 
#else
    OUTPUT(( "CORE_dpotrf( %s, %d, A(%d,%d), %d)\n",
             "PlasmaLower",
             NB, /*k == A.nt-1 ? A.n-k*A.nb : A.nb,*/
             k, k, NB /*A.nb,*/));
#endif  /* DPLASMA_EXECUTE */

    TAKE_TIME(POTRF_end_key);
    return rc;
}

int SYRK_hook(const dplasma_execution_context_t* exec_context)
{
    int k, n, rc = 0;

    TAKE_TIME(SYRK_start_key);
#ifndef DPLASMA_HOOK_OPTIMIZED
    {
        assignment_t *pk, *pn;
        rc = dplasma_find_assignment("k", exec_context->locals, MAX_LOCAL_COUNT, &pk);
        if( 0 != rc ) {
            return rc;
        }
        rc = dplasma_find_assignment("n", exec_context->locals, MAX_LOCAL_COUNT, &pn);
        if( 0 != rc ) {
            return rc;
        }
        k = pk->value;
        n = pn->value;
    }
#else
    k = exec_context->locals[0].value;
    n = exec_context->locals[1].value;
#endif

#ifdef DPLASMA_EXECUTE
    CORE_dsyrk( PlasmaLower, PlasmaNoTrans,
                NB, /*k == A.nt-1 ? A.n-k*A.nb : A.nb,*/
                NB, /*A.nb,*/
                -1.0, A(k, n), NB, /*A.nb,*/
                1.0, A(k, k), NB /*A.nb*/);
#else
    OUTPUT(("CORE_dsyrk( %s, %s, %d, %d, %f, A(%d,%d), %d, %f, A(%d,%d), %d)\n",
            "PlasmaLower", "PlasmaNoTrans",
            NB, /*k == A.nt-1 ? A.n-k*A.nb : A.nb,*/
            NB, /*A.nb,*/
            -1.0, k, n, NB, /*A.nb,*/
            1.0, k, k, NB /*A.nb*/));
#endif  /* DPLASMA_EXECUTE */

    TAKE_TIME(SYRK_end_key);
    return rc;
}

int GEMM_hook(const dplasma_execution_context_t* exec_context)
{
    int k, m, n, rc = 0;

    TAKE_TIME(GEMM_start_key);
#ifndef DPLASMA_HOOK_OPTIMIZED
    {
        assignment_t *pk, *pm, *pn;
        rc = dplasma_find_assignment("k", exec_context->locals, MAX_LOCAL_COUNT, &pk);
        if( 0 != rc ) {
            return rc;
        }
        rc = dplasma_find_assignment("m", exec_context->locals, MAX_LOCAL_COUNT, &pm);
        if( 0 != rc ) {
            return rc;
        }
        rc = dplasma_find_assignment("n", exec_context->locals, MAX_LOCAL_COUNT, &pn);
        if( 0 != rc ) {
            return rc;
        }
        k = pk->value;
        m = pm->value;
        n = pn->value;
    }
#else
    k = exec_context->locals[0].value;
    m = exec_context->locals[1].value;
    n = exec_context->locals[2].value;
#endif

#ifdef DPLASMA_EXECUTE
    CORE_dgemm( PlasmaNoTrans, PlasmaTrans,
                NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
                NB, /*A.nb,*/
                NB, /*A.nb,*/
                -1.0, A(m, n), NB, /*A.nb,*/
                A(k, n), NB, /*A.nb,*/
                1.0, A(m, k), NB /*A.nb*/);
#else
    OUTPUT(("CORE_dgemm( %s, %s, %d, %d, %d, %f, A(%d,%d), %d, A(%d,%d), %d, %f, A(%d,%d), %d)\n",
            "PlasmaNoTrans", "PlasmaTrans",
            NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
            NB, /*A.nb,*/
            NB, /*A.nb,*/
            -1.0, m, n, NB, /*A.nb,*/
            k, n, NB, /*A.nb,*/
            1.0, m, k, NB /*A.nb*/));
#endif  /* DPLASMA_EXECUTE */

    TAKE_TIME(GEMM_end_key);
    return rc;
}

int TRSM_hook(const dplasma_execution_context_t* exec_context)
{
    int k, m, rc = 0;

    TAKE_TIME(TRSM_start_key);
#ifndef DPLASMA_HOOK_OPTIMIZED
    {
        assignment_t *pk, *pm;
        rc = dplasma_find_assignment("k", exec_context->locals, MAX_LOCAL_COUNT, &pk);
        if( 0 != rc ) {
            return rc;
        }
        rc = dplasma_find_assignment("m", exec_context->locals, MAX_LOCAL_COUNT, &pm);
        if( 0 != rc ) {
            return rc;
        }
        k = pk->value;
        m = pm->value;
    }
#else
    k = exec_context->locals[0].value;
    m = exec_context->locals[1].value;
#endif

#ifdef DPLASMA_EXECUTE
    CORE_dtrsm( PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
                NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
                NB, /*A.nb,*/
                1.0, A(k, k), NB, /*A.nb,*/
                A(m, k), NB /*A.nb*/);
#else
    OUTPUT(( "CORE_dtrsm( %s, %s, %s, %s, %d, %d, %f, A(%d,%d), %d, A(%d,%d), %d)\n",
             "PlasmaRight", "PlasmaLower", "PlasmaTrans", "PlasmaNonUnit",
             NB, /*m == A.nt-1 ? A.n-m*A.nb : A.nb,*/
             NB, /*A.nb,*/
             1.0, k, k, NB, /*A.nb,*/
             m, k, NB /*A.nb*/));
#endif  /* DPLASMA_EXECUTE */

    TAKE_TIME(TRSM_end_key);
    return rc;
 }

int load_dplasma_hooks( void )
{
    dplasma_t* object;
    const symbol_t* pNB;
    int rc;

    object = (dplasma_t*)dplasma_find("POTRF");
    object->hook = POTRF_hook;

    object = (dplasma_t*)dplasma_find("SYRK");
    object->hook = SYRK_hook;

    object = (dplasma_t*)dplasma_find("GEMM");
    object->hook = GEMM_hook;

    object = (dplasma_t*)dplasma_find("TRSM");
    object->hook = TRSM_hook;

    /* This is a constant, only look for it once. */
    pNB = dplasma_search_global_symbol("NB");
    rc = expr_eval( pNB->min, NULL, 0, &NB );
    if( 0 != rc ) {
        return rc;
    }

#ifdef DPLASMA_PROFILING
    dplasma_profiling_init(1024);
    dplasma_profiling_add_dictionary_keyword( "POTRF", "fill:rgb(255,0,0);stroke:rgb(0,0,0)",
                                              &POTRF_start_key, &POTRF_end_key);
    dplasma_profiling_add_dictionary_keyword( "SYRK",  "fill:rgb(0,255,0);stroke:rgb(0,0,0)",
                                              &SYRK_start_key, &SYRK_end_key);
    dplasma_profiling_add_dictionary_keyword( "GEMM",  "fill:rgb(0,0,255);stroke:rgb(0,0,0)",
                                              &GEMM_start_key, &GEMM_end_key);
    dplasma_profiling_add_dictionary_keyword( "TRSM",  "fill:rgb(128,128,0);stroke:rgb(0,0,0)",
                                              &TRSM_start_key, &TRSM_end_key);
#endif  /* DPLASMA_PROFILING */

    return 0;
}
