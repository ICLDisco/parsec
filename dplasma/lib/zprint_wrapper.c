/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zprint.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zprint - Print a matrix tile by tile.
 *
 * Rk: This function doesn't use the dplasma_map() function because it has extra
 * control to synchronize the outputs.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of the matrix is printed
 *          = PlasmaUpper: Upper part of A;
 *          = PlasmaLower: Lower part of A;
 *          = PlasmaUpperLower: ALL elements of A.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A to generate. Any tiled matrix
 *          descriptor can be used.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_cprint
 * @sa dplasma_dprint
 * @sa dplasma_sprint
 *
 ******************************************************************************/
int dplasma_zprint( parsec_context_t *parsec,
                    PLASMA_enum uplo,
                    const tiled_matrix_desc_t *A)
{
    parsec_zprint_handle_t* handle;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zplghe", "illegal value of type");
        return -3;
    }

    handle = parsec_zprint_new( uplo, A);

    if (handle != NULL) {
        /* Default type */
        dplasma_add2arena_tile( handle->arenas[PARSEC_zprint_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(parsec_complex64_t),
                                PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_complex_t, A->mb );

        parsec_enqueue(parsec, (parsec_handle_t*)handle);
        dplasma_progress(parsec);

        parsec_matrix_del2arena( handle->arenas[PARSEC_zprint_DEFAULT_ARENA] );
        PARSEC_INTERNAL_HANDLE_DESTRUCT( handle );
        return 0;
    }
    return -101;
}
