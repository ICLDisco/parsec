
/**
 * Copyright (c) 2019-2023 The University of Tennessee and The University
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

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cublas_v2.h>
#endif

#include "nvlink.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
static void destruct_cublas_handle(void *p)
{
    cublasHandle_t handle = (cublasHandle_t)p;
    cublasStatus_t status;
    if(NULL != handle) {
        status = cublasDestroy(handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
        (void)status;
    }
}

static void *create_cublas_handle(void *obj, void *p)
{
    cublasHandle_t handle;
    cublasStatus_t status;
    parsec_cuda_exec_stream_t *stream = (parsec_cuda_exec_stream_t *)obj;
    (void)p;
    /* No need to call cudaSetDevice, as this has been done by PaRSEC before calling the task body */
    status = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == status);
    status = cublasSetStream(handle, stream->cuda_stream);
    assert(CUBLAS_STATUS_SUCCESS == status);
    (void)status;
    return (void*)handle;
}
#endif

static void destroy_cublas_handle(void *_h, void *_n)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cublasHandle_t cublas_handle = (cublasHandle_t)_h;
    cublasDestroy_v2(cublas_handle);
#endif
    (void)_n;
    (void)_h;
}

static void
__parsec_nvlink_destructor( parsec_nvlink_taskpool_t* nvlink_taskpool)
{
    int g, dev;
    parsec_matrix_block_cyclic_t *userM;
    parsec_matrix_block_cyclic_t *dcA;
    parsec_del2arena( & nvlink_taskpool->arenas_datatypes[PARSEC_nvlink_DEFAULT_ADT_IDX] );
    parsec_data_free(nvlink_taskpool->_g_descA->mat);
    parsec_info_unregister(&parsec_per_stream_infos, nvlink_taskpool->_g_CuHI, NULL);
    dcA = nvlink_taskpool->_g_descA;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)nvlink_taskpool->_g_descA );

    userM = nvlink_taskpool->_g_userM;
    for(g = 0, dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == cuda_device->super.super.type ) {
            parsec_data_t *dta = ((parsec_dc_t*)userM)->data_of((parsec_dc_t*)userM, g, userM->super.super.myrank);
            parsec_data_copy_t *gpu_copy = parsec_data_get_copy(dta, cuda_device->super.super.device_index);
            cudaError_t status = cudaSetDevice( cuda_device->cuda_index );
            PARSEC_CUDA_CHECK_ERROR( "(nvlink_wrapper) cudaSetDevice ", status, {} );
            status = (cudaError_t)cudaFree( gpu_copy->device_private );
            PARSEC_CUDA_CHECK_ERROR( "(nvlink_wrapper) cudaFree ", status, {} );
            gpu_copy->device_private = NULL;
            parsec_data_copy_detach(dta, gpu_copy, cuda_device->super.super.device_index);
            PARSEC_OBJ_RELEASE(gpu_copy);
            g++;
        }
    }
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)nvlink_taskpool->_g_userM );
    
    free(dcA);
    free(userM);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_nvlink_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_nvlink_destructor);

parsec_taskpool_t* testing_nvlink_New( parsec_context_t *ctx, int depth, int mb )
{
    parsec_nvlink_taskpool_t* testing_handle = NULL;
    int *dev_index, nb, dev, i;
    parsec_matrix_block_cyclic_t *dcA;
    parsec_matrix_block_cyclic_t *userM;

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

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    parsec_info_id_t CuHI = parsec_info_register(&parsec_per_stream_infos, "CUBLAS::HANDLE",
                                                 destroy_cublas_handle, NULL,
                                                 create_cublas_handle, NULL,
                                                 NULL);
    assert(CuHI != -1);
#else
    int CuHI = -1;
#endif

    /* A is used READ-ONLY by both GEMM1 and GEMM2 */
    dcA = (parsec_matrix_block_cyclic_t*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                              ctx->my_rank,
                              mb, mb,
                              depth*mb, ctx->nb_nodes*mb,
                              0, 0,
                              depth*mb, ctx->nb_nodes*mb,
                              1, ctx->nb_nodes, 1, 1,
                              0, 0);
    dcA->mat = parsec_data_allocate((size_t)dcA->super.nb_local_tiles *
                                    (size_t)dcA->super.bsiz *
                                    (size_t)parsec_datadist_getsizeoftype(dcA->super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)dcA, "A");

    for(i = 0; i < dcA->super.nb_local_tiles * mb * mb; i++)
        ((double*)dcA->mat)[i] = (double)rand() / (double)RAND_MAX;

    /* GEMM1 tasks will create one data copy per GPU, and work on those.
     * see nvlink.jdf:MAKE_C tasks */
    
    /* userM is a user-managed matrix: the user creates the data copies
     * only on the GPU they want the GEMM2 to run. To simplify the code,
     * we use parsec_matrix_block_cyclic that requires to also have a CPU data
     * copy, then for each data, we allocate a GPU data copy */
    userM = (parsec_matrix_block_cyclic_t*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(userM, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                              ctx->my_rank,
                              mb, mb,
                              nb*mb, ctx->nb_nodes*mb,
                              0, 0,
                              nb*mb, ctx->nb_nodes*mb,
                              1, 1,
                              1, 1,
                              0, 0);
    size_t userMlsize = (size_t)userM->super.nb_local_tiles *
        (size_t)userM->super.bsiz *
        (size_t)parsec_datadist_getsizeoftype(userM->super.mtype);
    userM->mat = parsec_data_allocate(userMlsize);
    memset(userM->mat, 0, userMlsize);

    /* Now, we create a GPU version of each tile. As these tiles will be accessed RW
     * in the JDF, this also pins the task on the GPU that we chose to host the tile */
    for(int g = 0, dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == cuda_device->super.super.type ) {
            /* We get the data from the data collection */
            parsec_data_t *dta = ((parsec_dc_t*)userM)->data_of((parsec_dc_t*)userM, g, ctx->my_rank);
            /* The corresponding data copy on CPU RAM */
            parsec_data_copy_t *cpu_copy = parsec_data_get_copy(dta, 0);
            /* And we create a new data copy on GPU */
            parsec_data_copy_t *gpu_copy = PARSEC_OBJ_NEW(parsec_data_copy_t);
            /* We chose the GPU */
            cudaError_t status = cudaSetDevice( cuda_device->cuda_index );
            PARSEC_CUDA_CHECK_ERROR( "(nvlink_wrapper) cudaSetDevice ", status, {return NULL;} );
            /* Allocate memory on it, for one tile */
            status = (cudaError_t)cudaMalloc( &gpu_copy->device_private, mb*mb*parsec_datadist_getsizeoftype(PARSEC_MATRIX_DOUBLE) );
            PARSEC_CUDA_CHECK_ERROR( "(nvlink_wrapper) cudaMalloc ", status, {return NULL;} );
            /* Attach this copy to the data, on the corresponding device */
            parsec_data_copy_attach(dta, gpu_copy, cuda_device->super.super.device_index);
            /* We also need to tell PaRSEC that the owner of this data is the GPU, or the
             * GPU might not be selected to work on that data */
            parsec_data_transfer_ownership_to_copy(dta, cuda_device->super.super.device_index, PARSEC_FLOW_ACCESS_RW);
            /* And copy the tile from CPU to GPU */
            status = (cudaError_t)cudaMemcpy( gpu_copy->device_private,
                                              cpu_copy->device_private,
                                              dta->nb_elts,
                                              cudaMemcpyHostToDevice );
            PARSEC_CUDA_CHECK_ERROR( "(nvlink_wrapper) cudaMemcpy ", status, {return NULL;} );
            g++;
        }
    }
    
    testing_handle = parsec_nvlink_new(dcA, userM, ctx->nb_nodes, CuHI, nb, dev_index);

    parsec_add2arena( &testing_handle->arenas_datatypes[PARSEC_nvlink_DEFAULT_ADT_IDX],
                             parsec_datatype_double_complex_t,
                             PARSEC_MATRIX_FULL, 1, mb, mb, mb,
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );
    
    return &testing_handle->super;
}

