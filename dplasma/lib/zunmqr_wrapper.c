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

/*#include "zunmqr_LN.h"*/
#include "zunmqr_LC.h"
#include "zunmqr_RN.h"
/*#include "zunmqr_RC.h"*/

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_New - Overwrites the general M-by-N matrix B with Q*B, where
 *  Q is an orthogonal matrix (unitary in the complex case) defined as the
 *  product of elementary reflectors returned by PLASMA_zgeqrf. Q is of order M.
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
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf in the first k columns of its array
 *          argument A.
 *
 * @param[in] T
 *          Descriptor of the auxiliary factorization data, computed
 *          by dplasma_zgeqrf.
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
 * @sa dplasma_zunmqr_Destruct
 * @sa dplasma_zunmqr
 * @sa dplasma_cunmqr_New
 * @sa dplasma_dorgqr_New
 * @sa dplasma_sorgqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zunmqr_New( PLASMA_enum side, PLASMA_enum trans,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *B)
{
    dague_object_t* object;
    int Am, ib = T->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(B) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal B descriptor"); */
    /*     return NULL; */
    /* } */
    if ( side == PlasmaLeft ) {
        Am = B->m;
    } else {
        Am = B->n;
    }

    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->n");
        return NULL;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->m");
        return NULL;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */
    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            fprintf(stderr, "zunmqr( Left, NoTrans ) is not implemented\n");
            return NULL;

            /* object = (dague_object_t*)dague_zunmqr_LN_new( side, trans, */
            /*                                                (dague_ddesc_t*)A, */
            /*                                                (dague_ddesc_t*)B, */
            /*                                                (dague_ddesc_t*)T, */
            /*                                                NULL); */
        } else {
            object = (dague_object_t*)dague_zunmqr_LC_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)B,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)dague_zunmqr_RN_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)B,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        } else {
            fprintf(stderr, "zunmqr( Left, NoTrans ) is not implemented\n");
            return NULL;
            /* object = (dague_object_t*)dague_zunmqr_RC_new( side, trans, */
            /*                                                (dague_ddesc_t*)A, */
            /*                                                (dague_ddesc_t*)B, */
            /*                                                (dague_ddesc_t*)T, */
            /*                                                NULL); */
        }
    }

    ((dague_zunmqr_LC_object_t*)object)->pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( ((dague_zunmqr_LC_object_t*)object)->pool_0, ib * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((dague_zunmqr_LC_object_t*)object)->arenas[DAGUE_zunmqr_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( ((dague_zunmqr_LC_object_t*)object)->arenas[DAGUE_zunmqr_LC_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( ((dague_zunmqr_LC_object_t*)object)->arenas[DAGUE_zunmqr_LC_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_Destruct - Clean the data structures associated to a
 *  zunmqr dague object.
 *
 *******************************************************************************
 *
 * @param[in] object
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_New
 * @sa dplasma_zunmqr
 * @sa dplasma_cunmqr_Destruct
 * @sa dplasma_dorgqr_Destruct
 * @sa dplasma_sorgqr_Destruct
 *
 ******************************************************************************/
void
dplasma_zunmqr_Destruct( dague_object_t *object )
{
    dague_zunmqr_LC_object_t *dague_zunmqr = (dague_zunmqr_LC_object_t *)object;

    dplasma_datatype_undefine_type( &(dague_zunmqr->arenas[DAGUE_zunmqr_LC_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmqr->arenas[DAGUE_zunmqr_LC_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmqr->arenas[DAGUE_zunmqr_LC_LITTLE_T_ARENA  ]->opaque_dtt) );

    dague_private_memory_fini( dague_zunmqr->pool_0 );
    free( dague_zunmqr->pool_0 );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zunmqr);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr - Synchronous version of dplasma_zunmqr_New
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
 * @sa dplasma_zunmqr_Destroy
 * @sa dplasma_zunmqr_New
 * @sa dplasma_cunmqr
 * @sa dplasma_dorgqr
 * @sa dplasma_sorgqr
 * @sa dplasma_zgeqrf
 *
 ******************************************************************************/
int
dplasma_zunmqr( dague_context_t *dague,
                PLASMA_enum side, PLASMA_enum trans,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T,
                tiled_matrix_desc_t *B )
{
    dague_object_t *dague_zunmqr = NULL;
    int Am;

    if (dague == NULL) {
        dplasma_error("dplasma_zunmqr", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        Am = B->m;
    } else {
        Am = B->n;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->m");
        return -3;
    }
    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->n");
        return -5;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_New", "illegal size of T (T should have as many tiles as A)");
        return -20;
    }

    if (dague_imin(B->m, dague_imin(B->n, A->n)) == 0)
        return 0;

    dague_zunmqr = dplasma_zunmqr_New(side, trans, A, T, B);

    dague_enqueue(dague, (dague_object_t*)dague_zunmqr);
    dplasma_progress(dague);

    dplasma_zunmqr_Destruct( dague_zunmqr );

    return 0;
}
