
/**
 * PLASMA include for defined and constants.
 */
#include <plasma.h>
#include <core_dblas.h>

#include "dplasma.h"
#include <stdlib.h>
#include <stdio.h>

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

    return rc;
}

int SYRK_hook(const dplasma_execution_context_t* exec_context)
{
    int k, n, rc = 0;

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

    return rc;
}

int GEMM_hook(const dplasma_execution_context_t* exec_context)
{
    int k, m, n, rc = 0;

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

    return rc;
}

int TRSM_hook(const dplasma_execution_context_t* exec_context)
{
    int k, m, rc = 0;

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

    return 0;
}
