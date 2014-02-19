/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"


#include "ztsmqr_LC.h"


/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_New - Generates the dague object that overwrites the general
 *  M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'C':      Q**H * C       C * Q**H
 *
 *  where Q is a unitary matrix defined as the product of k elementary
 *  reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by dplasma_zgeqrf(). Q is of order M if side = PlasmaLeft
 *  and of order N if side = PlasmaRight.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          @arg PlasmaLeft:  apply Q or Q**H from the left;
 *          @arg PlasmaRight: apply Q or Q**H from the right.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   no transpose, apply Q;
 *          @arg PlasmaConjTrans: conjugate transpose, apply Q**H.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K if side == PlasmaLeft, or
 *          N-by-K if side == PlasmaRight factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
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
 * @sa dplasma_dormqr_New
 * @sa dplasma_sormqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_ztsmqr_New( PLASMA_enum side, PLASMA_enum trans,
                    tiled_matrix_desc_t *A1,
                    tiled_matrix_desc_t *A2,
                    tiled_matrix_desc_t *V,
                    tiled_matrix_desc_t *T )
{
    dague_handle_t* object;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(C) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal C descriptor"); */
    /*     return NULL; */
    /* } */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of side");
        return NULL;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of trans");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            return NULL;
        } else {
            object = (dague_handle_t*)dague_ztsmqr_LC_new( side, trans,
                                                           (dague_ddesc_t*)A1,
                                                           (dague_ddesc_t*)A2,
                                                           (dague_ddesc_t*)V,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            dplasma_error("dplasma_zunmqr_New", "PlasmaRight not support yet\n");
            return NULL;
        } else {
            dplasma_error("dplasma_zunmqr_New", "PlasmaRight not support yet\n");
            return NULL;
        }
    }

    ((dague_ztsmqr_LC_handle_t*)object)->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( ((dague_ztsmqr_LC_handle_t*)object)->p_work, T->nb * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_rectangle( ((dague_ztsmqr_LC_handle_t*)object)->arenas[DAGUE_ztsmqr_LC_DEFAULT_ARENA],
                            A1->mb*A1->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A1->mb , A1->nb, -1);

    /* Little T */
    dplasma_add2arena_rectangle( ((dague_ztsmqr_LC_handle_t*)object)->arenas[DAGUE_ztsmqr_LC_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zunmqr_Destruct - Free the data structure associated to an object
 *  created with dplasma_zunmqr_New().
 *
 *******************************************************************************
 *
 * @param[in,out] object
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_New
 * @sa dplasma_zunmqr
 *
 ******************************************************************************/
void
dplasma_ztsmqr_Destruct( dague_handle_t *object )
{
    dague_ztsmqr_LC_handle_t *dague_ztsmqr = (dague_ztsmqr_LC_handle_t *)object;

    dplasma_datatype_undefine_type( &(dague_ztsmqr->arenas[DAGUE_ztsmqr_LC_DEFAULT_ARENA ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_ztsmqr->arenas[DAGUE_ztsmqr_LC_LITTLE_T_ARENA]->opaque_dtt) );

    dague_private_memory_fini( dague_ztsmqr->p_work );
    free( dague_ztsmqr->p_work );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_ztsmqr);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmqr_New - Overwrites the general M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'C':      Q**H * C       C * Q**H
 *
 *  where Q is a unitary matrix defined as the product of k elementary
 *  reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by dplasma_zgeqrf(). Q is of order M if side = PlasmaLeft
 *  and of order N if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] side
 *          @arg PlasmaLeft:  apply Q or Q**H from the left;
 *          @arg PlasmaRight: apply Q or Q**H from the right.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   no transpose, apply Q;
 *          @arg PlasmaConjTrans: conjugate transpose, apply Q**H.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K if side == PlasmaLeft, or
 *          N-by-K if side == PlasmaRight factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_Destruct
 * @sa dplasma_zunmqr
 * @sa dplasma_cunmqr_New
 * @sa dplasma_dormqr_New
 * @sa dplasma_sormqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
int
dplasma_ztsmqr( dague_context_t *dague,
                PLASMA_enum side, PLASMA_enum trans,
                tiled_matrix_desc_t *A1,
                tiled_matrix_desc_t *A2,
                tiled_matrix_desc_t *V,
                tiled_matrix_desc_t *T )
{
    dague_handle_t *dague_ztsmqr = NULL;

    if (dague == NULL) {
        dplasma_error("dplasma_zunmqr", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr", "illegal value of trans");
        return -2;
    }


    dague_ztsmqr = dplasma_ztsmqr_New(side, trans, A1, A2, V, T);

    if ( dague_ztsmqr != NULL ){
        dague_enqueue(dague, (dague_handle_t*)dague_ztsmqr);
        dplasma_progress(dague);
        dplasma_ztsmqr_Destruct( dague_ztsmqr );
    }
    return 0;
}
