#ifndef _pingpong_gpu_h_
#define _pingpong_gpu_h_

#include <dague.h>
#include <data_distribution.h>
#include <data.h>
#include <dague/devices/cuda/dev_cuda.h>
#include <dague/devices/device_malloc.h>
#include <dague/utils/output.h>
#include <fifo.h>
#include "scheduling.h"

extern int dague_cuda_output_stream;

static int
gpu_kernel_push_bandwidth( gpu_device_t            *gpu_device,
                           dague_gpu_context_t     *gpu_task,
                           dague_gpu_exec_stream_t *gpu_stream)
{
    int i, ret, space_needed = 0;
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_data_t              *original;
    dague_data_copy_t         *data, *local;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;

        this_task->data[i].data_out = NULL;  /* TODO: clean this up to segfault */
        data = this_task->data[i].data_in;
        original = data->original;
        if( NULL != (local = dague_data_get_copy(original, gpu_device->super.device_index)) ) {
            this_task->data[i].data_out = local;
            /* Check the most up2date version of the data */
            if( data->device_index != gpu_device->super.device_index ) {
                if(data->version <= local->version) {
                    if(data->version == local->version) continue;
                    /* Trouble: there are two versions of this data coexisting in same
                     * time, one using a read-only path and one that has been updated.
                     * We don't handle this case yet!
                     * TODO:
                     */
                    assert(0);
                }
            }
            continue;  /* space available on the device */
        }
        /* If the data is needed as an input load it up */
        if(this_task->function->in[i]->access_type & ACCESS_READ)
            space_needed++;
    }

    if( 0 != space_needed ) { /* Try to reserve enough room for all data */
        ret = dague_gpu_data_reserve_device_space( gpu_device,
                                                   this_task,
                                                   space_needed );
        if( ret < 0 ) {
            goto release_and_return_error;
        }
    }

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->in[i]) continue;
        assert( NULL != dague_data_copy_get_ptr(this_task->data[i].data_in) );

        DAGUE_OUTPUT_VERBOSE((2, dague_cuda_output_stream,
                              "GPU[%1d]:\tIN  Data of %s <%x> on GPU\n",
                              gpu_device->cuda_index, this_task->function->in[i]->name,
                              (int)this_task->data[i].data_out->original->key));
        ret = dague_gpu_data_stage_in( gpu_device, this_task->function->in[i]->access_type,
                                       &(this_task->data[i]), gpu_stream->cuda_stream );
        if( ret < 0 ) {
            return ret;
        }
    }

  release_and_return_error:
    return ret;
}

static int
gpu_kernel_pop_bandwidth( gpu_device_t        *gpu_device,
                          dague_gpu_context_t *gpu_task,
                          dague_gpu_exec_stream_t* gpu_stream)
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    const dague_flow_t        *flow;
    int return_code = 0, how_many = 0, i, first = 1;
    cudaError_t status;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        flow = this_task->function->in[i];
        if(NULL == flow)
            flow = this_task->function->out[i];

        original = this_task->data[i].data_out->original;
        gpu_copy = this_task->data[i].data_out;
        assert(original == this_task->data[i].data_in->original);
        if( flow->access_type & ACCESS_READ ) {
            gpu_copy->readers--; assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->access_type & ACCESS_WRITE) ) {
                dague_list_item_ring_chop((dague_list_item_t*)gpu_copy);
                DAGUE_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
            }
        }
        if( flow->access_type & ACCESS_WRITE ) {
            assert( gpu_copy == dague_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );
            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((dague_list_item_t*)gpu_copy)->list_next == (dague_list_item_t*)gpu_copy );
            assert( ((dague_list_item_t*)gpu_copy)->list_prev == (dague_list_item_t*)gpu_copy );

            DAGUE_OUTPUT_VERBOSE((2, dague_cuda_output_stream,
                                  "GPU[%1d]:\tOUT Data of %s key %d\n", gpu_device->cuda_index,
                                  this_task->function->out[i]->name, this_task->data[i].data_out->original->key));
            if(first) {
                DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_device->super.profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                                    this_task->function->function_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                first = 0;
            }
            /* TODO: Move the data back into main memory, but not always on the first device (!) */
            status = (cudaError_t)cuMemcpyDtoHAsync( original->device_copies[0]->device_private,
                                                     (CUdeviceptr)gpu_copy->device_private,
                                                     original->nb_elts, gpu_stream->cuda_stream );
            DAGUE_CUDA_CHECK_ERROR( "cuMemcpyDtoHAsync from device ", status,
                                    { WARNING(("data %s <<%p>> -> <<%p>>\n", this_task->function->out[i]->name,
                                               gpu_copy->device_private, original->device_copies[0]->device_private));
                                        return_code = -2;
                                        goto release_and_return_error;} );
            gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
            how_many++;
        }
    }

  release_and_return_error:
    return (return_code < 0 ? return_code : how_many);
}

static int
gpu_kernel_epilog_bandwidth( gpu_device_t        *gpu_device,
                             dague_gpu_context_t *gpu_task )
{
    dague_execution_context_t *this_task = gpu_task->ec;
    dague_gpu_data_copy_t     *gpu_copy;
    dague_data_t              *original;
    int i;

    for( i = 0; i < this_task->function->nb_parameters; i++ ) {
        if(NULL == this_task->function->out[i]) continue;
        if(!(this_task->function->out[i]->access_type & ACCESS_WRITE)) continue;

        gpu_copy = this_task->data[this_task->function->out[i]->flow_index].data_out;
        original = gpu_copy->original;
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        /*original->coherency_state = DATA_COHERENCY_SHARED;*/
        original->owner_device = 0;
        original->device_copies[0]->coherency_state = DATA_COHERENCY_SHARED;

        /* Use the CPU version now that we tranferred the ownership */
        this_task->data[i].data_out = original->device_copies[0];

        dague_ulist_fifo_push(&gpu_device->gpu_mem_lru, (dague_list_item_t*)gpu_copy);
    }
    return 0;
}

static int
gpu_kernel_submit_bandwidth( gpu_device_t            *gpu_device,
                             dague_gpu_context_t     *gpu_task,
                             dague_gpu_exec_stream_t *gpu_stream )
{
    dague_execution_context_t *this_task = gpu_task->ec;

    DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                             gpu_device->super.profiling,
                             (-1 == gpu_stream->prof_event_key_start ?
                              DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                        this_task->function->function_id) :
                              gpu_stream->prof_event_key_start),
                             this_task);
    (void)gpu_device; (void)gpu_stream; (void)this_task;
    return 0;
}

#define KERNEL_NAME bandwidth
#include <dague/devices/cuda/cuda_scheduling.h>

static inline
int bandwidth_cuda(dague_execution_unit_t* eu_context,
                   dague_execution_context_t* this_task)
{
    dague_gpu_context_t* gpu_task;

    gpu_task = (dague_gpu_context_t*)malloc(sizeof(dague_gpu_context_t));
    OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
    gpu_task->ec = this_task;

    return gpu_kernel_scheduler_bandwidth( eu_context, gpu_task, 1 );
}

#endif /* _bandwidth_gpu_h_ */
