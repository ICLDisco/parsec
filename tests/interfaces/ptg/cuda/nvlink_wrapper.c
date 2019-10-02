/**
 * Copyright (c) 2019-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/execution_stream.h"
#include "parsec/class/info.h"

#if defined(PARSEC_HAVE_CUDA)
#include <cublas_v2.h>
#endif

#include "nvlink.h"

#if defined(PARSEC_HAVE_CUDA)
static void destruct_cublas_handle(void *p)
{
    cublasHandle_t handle = (cublasHandle_t)p;
    cublasStatus_t status;
    if(NULL != handle) {
        status = cublasDestroy(handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
}
#endif

parsec_taskpool_t* testing_nvlink_New( parsec_context_t *ctx, int depth, int mb )
{
    parsec_nvlink_taskpool_t* testing_handle = NULL;
    parsec_arena_t *arena;
    int *dev_index, nb, dev, i;
    two_dim_block_cyclic_t *dcA;

    /** Find all CUDA devices */
    nb = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
        fprintf(stderr, "This test requires at least one CUDA device per node -- no CUDA device found on rank %d on %s\n",
                ctx->my_rank, hostname);
        return NULL;
    }
    dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb++] = device->device_index;
        }
    }

#if defined(PARSEC_HAVE_CUDA)
    parsec_info_id_t CuHI = parsec_info_register(&parsec_per_device_infos, "CUBLAS::HANDLE", NULL);
    assert(CuHI != -1);
#else
    int CuHI = -1;
#endif

    dcA = (two_dim_block_cyclic_t*)calloc(1, sizeof(two_dim_block_cyclic_t));
    two_dim_block_cyclic_init(dcA, matrix_RealDouble, matrix_Tile,
                              ctx->nb_nodes, ctx->my_rank,
                              mb, mb,
                              depth*mb, ctx->nb_nodes*mb,
                              0, 0,
                              depth*mb, ctx->nb_nodes*mb,
                              1, 1,
                              1);
    dcA->mat = parsec_data_allocate((size_t)dcA->super.nb_local_tiles *
                                    (size_t)dcA->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA->super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)dcA, "A");

    for(i = 0; i < dcA->super.nb_local_tiles * mb * mb; i++)
        ((double*)dcA->mat)[i] = (double)rand() / (double)RAND_MAX;

    testing_handle = parsec_nvlink_new(dcA, ctx->nb_nodes, CuHI, nb, dev_index);

    arena = testing_handle->arenas[PARSEC_nvlink_DEFAULT_ARENA];
    parsec_matrix_add2arena( arena, parsec_datatype_double_complex_t,
                             matrix_UpperLower, 1, mb, mb, mb,
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return &testing_handle->super;
}

void testing_nvlink_Destruct( parsec_taskpool_t *tp )
{
    parsec_nvlink_taskpool_t *nvlink_taskpool = (parsec_nvlink_taskpool_t *)tp;
    two_dim_block_cyclic_t *dcA;
    if (nvlink_taskpool->arenas[PARSEC_nvlink_DEFAULT_ARENA])
        parsec_matrix_del2arena( nvlink_taskpool->arenas[PARSEC_nvlink_DEFAULT_ARENA] );
    parsec_data_free(nvlink_taskpool->_g_descA->mat);
    parsec_info_unregister(&parsec_per_device_infos, nvlink_taskpool->_g_CuHI, NULL);
    dcA = nvlink_taskpool->_g_descA;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)nvlink_taskpool->_g_descA );
    parsec_taskpool_free(tp);
    free(dcA);
}
