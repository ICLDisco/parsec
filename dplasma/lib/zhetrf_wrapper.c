/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <math.h>
#include <stdlib.h>
#include <core_blas.h>
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dague/private_mempool.h"

#include "dplasma/lib/zhetrf.h"
#include "dplasma/lib/ztrmdm.h"
/*#include <lapacke.h>*/

/*
 * dplasma_zhetrf_New()
 */
dague_handle_t*
dplasma_zhetrf_New( tiled_matrix_desc_t *A, int *INFO)
{
    int ldwork, lwork, ib;
    dague_handle_t *dague_zhetrf = NULL;
    dague_memory_pool_t *pool_0, *pool_1;

    ib = A->mb;

    /* ldwork and lwork are necessary for the macros zhetrf_pool_0_SIZE and zhetrf_pool_1_SIZE */
    ldwork = (A->nb+1)*ib;
    lwork = (A->mb+1)*A->nb + ib*ib;

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, zhetrf_pool_0_SIZE );

    pool_1 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_1, zhetrf_pool_1_SIZE );

    dague_zhetrf = (dague_handle_t *)dague_zhetrf_new(PlasmaLower, *A, (dague_ddesc_t *)A, ib, pool_1, pool_0, INFO);

    dplasma_add2arena_tile(((dague_zhetrf_handle_t*)dague_zhetrf)->arenas[DAGUE_zhetrf_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, A->mb);

    return dague_zhetrf;
}

void
dplasma_zhetrf_Destruct( dague_handle_t *handle )
{
    dague_zhetrf_handle_t *obut = (dague_zhetrf_handle_t *)handle;

    dague_matrix_del2arena( obut->arenas[DAGUE_zhetrf_DEFAULT_ARENA] );

    handle->destructor(handle);
}


/*
 * dplasma_ztrmdm_New()
 */
dague_handle_t*
dplasma_ztrmdm_New( tiled_matrix_desc_t *A)
{
    dague_handle_t *dague_ztrmdm = NULL;


    dague_ztrmdm = (dague_handle_t *)dague_ztrmdm_new(*A, (dague_ddesc_t *)A);

    dplasma_add2arena_tile(((dague_ztrmdm_handle_t*)dague_ztrmdm)->arenas[DAGUE_ztrmdm_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, A->mb);

    return dague_ztrmdm;
}

void
dplasma_ztrmdm_Destruct( dague_handle_t *handle )
{
    dague_ztrmdm_handle_t *obut = (dague_ztrmdm_handle_t *)handle;

    dague_matrix_del2arena( obut->arenas[DAGUE_ztrmdm_DEFAULT_ARENA] );

    //dague_ztrmdm_destroy(obut);
    handle->destructor(handle);
}

/*
 * Blocking Interface
 */

int dplasma_zhetrf(dague_context_t *dague, tiled_matrix_desc_t *A)
{
    dague_handle_t *dague_zhetrf/*, *dague_ztrmdm*/;
    int info = 0, ginfo = 0;

    dague_zhetrf = dplasma_zhetrf_New(A, &info);
    dague_enqueue(dague, (dague_handle_t *)dague_zhetrf);
    dplasma_progress(dague);
    dplasma_zhetrf_Destruct(dague_zhetrf);

    /*
    dague_ztrmdm = dplasma_ztrmdm_New(A);
    dague_enqueue(dague, (dague_handle_t *)dague_ztrmdm);
    dplasma_progress(dague);
    dplasma_ztrmdm_Destruct(dague_ztrmdm);
    */

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
