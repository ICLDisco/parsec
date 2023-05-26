#include "parsec.h"
#include "parsec/execution_stream.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "stress.h"

static void __parsec_stress_destructor( parsec_taskpool_t *tp )
{
    parsec_stress_taskpool_t *stress_taskpool = (parsec_stress_taskpool_t *)tp;
    parsec_matrix_block_cyclic_t *dcA;
    parsec_del2arena( & stress_taskpool->arenas_datatypes[PARSEC_stress_DEFAULT_ADT_IDX] );
    parsec_data_free(stress_taskpool->_g_descA->mat);
    dcA = stress_taskpool->_g_descA;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)stress_taskpool->_g_descA );
    free(dcA);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_stress_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_stress_destructor);

parsec_taskpool_t* testing_stress_New( parsec_context_t *ctx, int depth, int mb )
{
    parsec_stress_taskpool_t* testing_handle = NULL;
    int *dev_index, nb, dev, i;
    parsec_matrix_block_cyclic_t *dcA;

    /** Find all CUDA devices */
    nb = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }
    if(nb == 0) {
        /* We just simulate a run on CPUs, with an arbitrary number of pseudo-GPUs */
        nb = 8;
        dev_index = (int*)malloc(nb * sizeof(int));
        memset(dev_index, -1, nb*sizeof(int));
        fprintf(stderr, "Simulating %d GPUs for sanity checking in stress test\n", nb);
    } else {
        dev_index = (int*)malloc(nb * sizeof(int));
        nb = 0;
        for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
            parsec_device_module_t *device = parsec_mca_device_get(dev);
            if( PARSEC_DEV_CUDA == device->type ) {
                dev_index[nb++] = device->device_index;
            }
        }
    }

    dcA = (parsec_matrix_block_cyclic_t*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                              ctx->my_rank,
                              mb, mb,
                              depth*mb, ctx->nb_nodes*mb,
                              0, 0,
                              depth*mb, ctx->nb_nodes*mb,
                              1, ctx->nb_nodes,
                              1, 1,
                              0, 0);
    dcA->mat = parsec_data_allocate((size_t)dcA->super.nb_local_tiles *
                                    (size_t)dcA->super.bsiz *
                                    (size_t)parsec_datadist_getsizeoftype(dcA->super.mtype));
    assert(NULL != dcA->mat);
    parsec_data_collection_set_key((parsec_data_collection_t*)dcA, "A");

    for(i = 0; i < dcA->super.nb_local_tiles * mb * mb; i++)
        ((double*)dcA->mat)[i] = (double)rand() / (double)RAND_MAX;

    testing_handle = parsec_stress_new(dcA, ctx->nb_nodes, nb, dev_index);

    parsec_add2arena( &testing_handle->arenas_datatypes[PARSEC_stress_DEFAULT_ADT_IDX],
                             parsec_datatype_double_complex_t,
                             PARSEC_MATRIX_FULL, 1, mb, mb, mb,
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );
    return &testing_handle->super;
}

