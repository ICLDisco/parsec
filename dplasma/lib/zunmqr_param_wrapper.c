/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

//#include "zunmqr_param_LN.h"
#include "zunmqr_param_LC.h"
//#include "zunmqr_param_RN.h"
//#include "zunmqr_param_RC.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_param_New - Overwrites the general M-by-N matrix B with Q*B, where
 *  Q is an orthogonal matrix (unitary in the complex case) defined as the
 *  product of elementary reflectors returned by dplasma_zgeqrf_param. Q is of order M.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Intended usage:
 *          = PlasmaLeft:  apply Q or Q**H from the left;
 *          = PlasmaRight: apply Q or Q**H from the right.
 *          Currently only PlasmaLeft is supported.
 *
 * @param[in] trans
 *          Intended usage:
 *          = PlasmaNoTrans:   no transpose, apply Q;
 *          = PlasmaConjTrans: conjugate transpose, apply Q**H.
 *          Currently only PlasmaConjTrans is supported.
 *
 * @param[in] qrtree
 *          Structure describing the trees used to factorize A. It has to be the
 *          same than the one used during call to dplasma_zgeqrf_param.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param in the first k columns of its array
 *          argument A.
 *
 * @param[in] TS
 *          Descriptor of the auxiliary factorization data, computed
 *          by dplasma_zgeqrf_param.
 *
 * @param[in] TT
 *          Descriptor of the auxiliary factorization data, computed
 *          by dplasma_zgeqrf_param.
 *
 * @param[out] B
 *          Descriptor of the M-by-N matrix B returned.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague object which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_param_Destruct
 * @sa dplasma_zunmqr_param
 * @sa dplasma_cunmqr_param_New
 * @sa dplasma_dormqr_param_New
 * @sa dplasma_sormqr_param_New
 * @sa dplasma_zgeqrf_param_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zunmqr_param_New( PLASMA_enum side, PLASMA_enum trans,
                          dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *B)
{
    dague_object_t* object = NULL;
    int Am, ib = TS->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(B) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal B descriptor"); */
    /*     return NULL; */
    /* } */
    if ( side == PlasmaLeft ) {
        Am = B->m;
    } else {
        Am = B->n;
    }

    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->n");
        return NULL;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->m");
        return NULL;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */
    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            /* object = (dague_object_t*)dague_zunmqr_param_LN_new( side, trans, */
            /*                                                      (dague_ddesc_t*)A, */
            /*                                                      (dague_ddesc_t*)B, */
            /*                                                      (dague_ddesc_t*)TS, */
            /*                                                      (dague_ddesc_t*)TT, */
            /*                                                      *qrtree, */
            /*                                                      NULL); */
        } else {
            object = (dague_object_t*)dague_zunmqr_param_LC_new( side, trans,
                                                                 (dague_ddesc_t*)A,
                                                                 (dague_ddesc_t*)B,
                                                                 (dague_ddesc_t*)TS,
                                                                 (dague_ddesc_t*)TT,
                                                                 *qrtree,
                                                                 NULL);
        }
    } else {
        /* if ( trans == PlasmaNoTrans ) { */
        /*     object = (dague_object_t*)dague_zunmqr_param_RN_new( side, trans, */
        /*                                                          (dague_ddesc_t*)A, */
        /*                                                          (dague_ddesc_t*)B, */
        /*                                                          (dague_ddesc_t*)TS, */
        /*                                                          (dague_ddesc_t*)TT, */
        /*                                                          *qrtree, */
        /*                                                          NULL); */
        /* } else { */
        /*     object = (dague_object_t*)dague_zunmqr_param_RC_new( side, trans, */
        /*                                                          (dague_ddesc_t*)A, */
        /*                                                          (dague_ddesc_t*)B, */
        /*                                                          (dague_ddesc_t*)TS, */
        /*                                                          (dague_ddesc_t*)TT, */
        /*                                                          *qrtree, */
        /*                                                          NULL); */
        /* } */
    }

    ((dague_zunmqr_param_LC_object_t*)object)->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( ((dague_zunmqr_param_LC_object_t*)object)->p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((dague_zunmqr_param_LC_object_t*)object)->arenas[DAGUE_zunmqr_param_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( ((dague_zunmqr_param_LC_object_t*)object)->arenas[DAGUE_zunmqr_param_LC_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( ((dague_zunmqr_param_LC_object_t*)object)->arenas[DAGUE_zunmqr_param_LC_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( ((dague_zunmqr_param_LC_object_t*)object)->arenas[DAGUE_zunmqr_param_LC_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, TS->mb, TS->nb, -1);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_param_Destruct - Clean the data structures associated to a
 *  zunmqr_param dague object.
 *
 *******************************************************************************
 *
 * @param[in] object
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_param_New
 * @sa dplasma_zunmqr_param
 * @sa dplasma_cunmqr_param_Destruct
 * @sa dplasma_dormqr_param_Destruct
 * @sa dplasma_sormqr_param_Destruct
 *
 ******************************************************************************/
void
dplasma_zunmqr_param_Destruct( dague_object_t *object )
{
    dague_zunmqr_param_LC_object_t *dague_zunmqr_param = (dague_zunmqr_param_LC_object_t *)object;

    dplasma_datatype_undefine_type( &(dague_zunmqr_param->arenas[DAGUE_zunmqr_param_LC_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmqr_param->arenas[DAGUE_zunmqr_param_LC_LITTLE_T_ARENA  ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmqr_param->arenas[DAGUE_zunmqr_param_LC_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmqr_param->arenas[DAGUE_zunmqr_param_LC_UPPER_TILE_ARENA]->opaque_dtt) );

    dague_private_memory_fini( dague_zunmqr_param->p_work );
    free( dague_zunmqr_param->p_work );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zunmqr_param);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_param - Synchronous version of dplasma_zunmqr_param_New
 *
 *******************************************************************************
 *
 * @param[in] dague
 *          Dague context to which submit the DAG object.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 if success
 *          \retval < 0 if one of the parameter had an illegal value.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_param_Destroy
 * @sa dplasma_zunmqr_param_New
 * @sa dplasma_cunmqr_param
 * @sa dplasma_dormqr_param
 * @sa dplasma_sormqr_param
 * @sa dplasma_zgeqrf_param
 *
 ******************************************************************************/
int
dplasma_zunmqr_param( dague_context_t *dague,
                      PLASMA_enum side, PLASMA_enum trans,
                      dplasma_qrtree_t    *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *B )
{
    dague_object_t *dague_zunmqr_param = NULL;
    int Am;

    if (dague == NULL) {
        dplasma_error("dplasma_zunmqr_param", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        Am = B->m;
    } else {
        Am = B->n;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->m");
        return -3;
    }
    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->n");
        return -5;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return -20;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return -20;
    }

    if (dague_imin(B->m, dague_imin(B->n, A->n)) == 0)
        return 0;

    dague_zunmqr_param = dplasma_zunmqr_param_New(side, trans, qrtree, A, TS, TT, B);

    dague_enqueue(dague, (dague_object_t*)dague_zunmqr_param);
    dplasma_progress(dague);

    dplasma_zunmqr_param_Destruct( dague_zunmqr_param );

    return 0;
}
