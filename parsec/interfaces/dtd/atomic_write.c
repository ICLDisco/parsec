/**
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* **************************************************************************** */
/**
 * @file atomic_write.c
 *
 * Support for the PARSEC_ATOMIC_WRITE access type of the DTD interface.
 *
 * A run is a sequence of consecutive ATOMIC_WRITE accesses on a tile, all placed on one rank. Towards
 * the other accesses a run behaves like a write; among themselves its tasks are chained like readers,
 * so they run concurrently. Empty INOUT tasks placed on the rank of the run provide the first part:
 * an INOUT waits for all the readers before it, and what follows it is activated once it completes.
 *
 *  - the prologue goes in front of the first task of a run that follows readers;
 *  - the epilogue goes after the last task of a run. It waits for the whole run and becomes the last
 *    writer of the tile: whatever follows, on any rank, has the right parent and gets the data from
 *    the rank where the run executed.
 *
 * Two runs on different ranks are ordered by the epilogue of the first, so they need no prologue.
 *
 * The fences are tasks of classes created on first use, inserted like any other task. The decisions
 * only depend on the accesses inserted and on the placement of the tasks, which every rank sees
 * identically, so every rank takes the same ones (state in parsec_dtd_tile_t::atomic_write).
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>

#include "parsec/parsec_internal.h"
#include "parsec/runtime.h"
#include "parsec/mca/device/device.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

/* **************************************************************************** */
/* The empty tasks that delimit the runs                                         */
/* **************************************************************************** */

/* Two bodies, so that the two fences are two task classes with their own names in the traces */
int
parsec_dtd_atomic_write_prologue_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_dtd_atomic_write_epilogue_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

/* The tasks of a run write concurrently and cannot push the data out, so the epilogue has a GPU chore
 * for that: it computes nothing, the pushout is what the runtime does after the body. */
static int
parsec_dtd_atomic_write_epilogue_gpu_body(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
                                          parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_task;
    (void)gpu_stream;
    return PARSEC_HOOK_RETURN_DONE;
}

typedef enum {
    PARSEC_DTD_ATOMIC_FENCE_NONE = 0,
    PARSEC_DTD_ATOMIC_FENCE_PROLOGUE,
    PARSEC_DTD_ATOMIC_FENCE_EPILOGUE
} parsec_dtd_atomic_fence_t;

/* The task class of a fence, created on first use (the chores of the epilogue depend on the devices of
 * the taskpool). Both take the data (INOUT) and the rank to run on (a value used as affinity). */
static parsec_task_class_t *
parsec_dtd_atomic_write_fence_class(parsec_dtd_taskpool_t *dtd_tp, parsec_dtd_atomic_fence_t fence)
{
    const int is_epilogue = (PARSEC_DTD_ATOMIC_FENCE_EPILOGUE == fence);
    parsec_hook_t *body = is_epilogue ? parsec_dtd_atomic_write_epilogue_body
                                      : parsec_dtd_atomic_write_prologue_body;
    uint64_t fkey = parsec_dtd_task_class_cache_key((uintptr_t)body, 1);
    parsec_task_class_t *tc = (parsec_task_class_t *)parsec_dtd_find_task_class(dtd_tp, fkey);

    if( NULL != tc ) {
        return tc;
    }
    tc = parsec_dtd_create_task_class(&dtd_tp->super,
                                      is_epilogue ? "ATOMIC_WRITE_epilogue" : "ATOMIC_WRITE_prologue",
                                      PASSED_BY_REF, PARSEC_INOUT,
                                      sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                      PARSEC_DTD_ARG_END);
    if( PARSEC_SUCCESS != parsec_dtd_task_class_add_chore(&dtd_tp->super, tc, PARSEC_DEV_CPU, body) ) {
        parsec_fatal("Could not create the task class of the fences of the PARSEC_ATOMIC_WRITE runs\n");
    }
    if( is_epilogue ) {
        const int gpu_types[] = { PARSEC_DEV_CUDA, PARSEC_DEV_HIP, PARSEC_DEV_LEVEL_ZERO };
        for( size_t i = 0; i < sizeof(gpu_types) / sizeof(gpu_types[0]); i++ ) {
            if( parsec_dtd_taskpool_supports_device_type(&dtd_tp->super, gpu_types[i], 1) ) {
                (void)parsec_dtd_task_class_add_chore(&dtd_tp->super, tc, gpu_types[i],
                                                      parsec_dtd_atomic_write_epilogue_gpu_body);
            }
        }
    }
    ((parsec_dtd_task_class_t *)tc)->atomic_write_fence = 1;
    parsec_dtd_register_task_class(&dtd_tp->super, fkey, tc);  /* the taskpool releases it with the others */
    return tc;
}

/**
 * Which fence, on which rank, does the access about to be inserted on a tile need in front of it?
 *
 *  - An ATOMIC_WRITE after readers: a prologue on its rank, as the readers are activated together
 *    and it must not start before they are done.
 *  - An ATOMIC_WRITE on another rank than the open run: an epilogue on the rank of the run, which is
 *    also where the new run gets its data from.
 *  - Any other access after a run: the epilogue of the run. A reader would be chained with the run,
 *    and a task on another rank would be sent the data before the run ran. The rule is the same
 *    wherever the task is placed.
 */
static parsec_dtd_atomic_fence_t
parsec_dtd_atomic_write_needs_fence(const parsec_dtd_tile_t *tile, int access, int task_rank, int *fence_rank)
{
    const parsec_dtd_tile_atomic_write_t *atomic_write = &tile->atomic_write;

    if( PARSEC_ATOMIC_WRITE == access ) {
        if( PARSEC_INPUT == atomic_write->last_access ) {
            *fence_rank = task_rank;
            return PARSEC_DTD_ATOMIC_FENCE_PROLOGUE;
        }
        if( PARSEC_ATOMIC_WRITE == atomic_write->last_access && atomic_write->rank != task_rank ) {
            *fence_rank = atomic_write->rank;
            return PARSEC_DTD_ATOMIC_FENCE_EPILOGUE;
        }
        return PARSEC_DTD_ATOMIC_FENCE_NONE;
    }
    if( PARSEC_ATOMIC_WRITE == atomic_write->last_access ) {
        *fence_rank = atomic_write->rank;
        return PARSEC_DTD_ATOMIC_FENCE_EPILOGUE;
    }
    return PARSEC_DTD_ATOMIC_FENCE_NONE;
}

/**
 * Does what follows a run need the data out of the device the run executed on? The epilogue can push
 * it out, the tasks of the run cannot. It is not needed when the follower writes the data (INOUT,
 * OUTPUT), only has GPU chores and is on the rank of the run: it takes the data from the device, and
 * its own PARSEC_PUSHOUT decides what happens next. It is needed for a reader, for a task on another
 * rank (sent the host copy) and for a task that can execute on the CPU.
 */
static int
parsec_dtd_atomic_write_follower_needs_pushout(const parsec_dtd_task_t *follower, int access, int run_rank)
{
    const parsec_task_class_t *tc = follower->super.task_class;
    int nb_chores = 0;

    if( follower->rank != run_rank || (PARSEC_INOUT != access && PARSEC_OUTPUT != access) ) {
        return 1;
    }
    for( int chore = 0; NULL != tc->incarnations[chore].hook; chore++ ) {
        if( !(follower->super.chore_mask & (1 << chore)) ) continue;
        if( !PARSEC_DEV_IS_GPU(tc->incarnations[chore].type) ) {
            return 1;
        }
        nb_chores++;
    }
    return 0 == nb_chores;
}

/* The fences are placed on the rank of the run, whoever owns the data */
static void
parsec_dtd_atomic_write_insert_fence(parsec_dtd_taskpool_t *dtd_tp, parsec_dtd_tile_t *tile, int region,
                                     parsec_dtd_atomic_fence_t fence, int fence_rank, int pushout)
{
    assert(PARSEC_DTD_ATOMIC_FENCE_NONE != fence && fence_rank >= 0);
    if( PARSEC_DTD_ATOMIC_FENCE_EPILOGUE == fence && pushout ) {
        region |= PARSEC_PUSHOUT;
    }
    parsec_dtd_insert_task_with_task_class(&dtd_tp->super, parsec_dtd_atomic_write_fence_class(dtd_tp, fence),
                                           0, PARSEC_DEV_ANY_TYPE,
                                           region, tile,
                                           PARSEC_DTD_EMPTY_FLAG, &fence_rank,
                                           PARSEC_DTD_ARG_END);
}

/* **************************************************************************** */
/* Task insertion                                                                */
/* **************************************************************************** */

void
parsec_dtd_atomic_write_check_task(const parsec_dtd_task_t *this_task)
{
    const parsec_task_class_t *tc = this_task->super.task_class;
    int nb_atomic_flows = 0;

    /* The tasks of a run share one device, so a task is part of one run only: one ATOMIC_WRITE flow, and
     * its data is not used by another flow of the task. */
    for( int flow_index = 0; flow_index < (int)tc->nb_flows; flow_index++ ) {
        parsec_dtd_flow_info_t *flow = FLOW_OF(this_task, flow_index);
        if( NULL == flow->tile || (flow->op_type & PARSEC_DONT_TRACK) ||
            PARSEC_ATOMIC_WRITE != (flow->op_type & PARSEC_GET_OP_TYPE) ) {
            continue;
        }
        nb_atomic_flows++;
        if( flow->op_type & PARSEC_PUSHOUT ) {
            parsec_fatal("DTD task '%s' has PARSEC_PUSHOUT on its PARSEC_ATOMIC_WRITE flow %d. The tasks of a run write "
                         "the same copy of the data concurrently, so none of them can push it out: use PARSEC_PUSHOUT "
                         "on the exclusive access that follows the run.\n",
                         tc->name, flow_index);
        }
        for( int other = 0; other < (int)tc->nb_flows; other++ ) {
            if( other != flow_index && FLOW_OF(this_task, other)->tile == flow->tile ) {
                parsec_fatal("DTD task '%s' uses the same data in flow %d, which is PARSEC_ATOMIC_WRITE, and in flow %d. "
                             "The data of a PARSEC_ATOMIC_WRITE flow cannot be used by another flow of the task.\n",
                             tc->name, flow_index, other);
            }
        }
    }
    if( nb_atomic_flows > 1 ) {
        parsec_fatal("DTD task '%s' has %d PARSEC_ATOMIC_WRITE flows, but a task can have only one.\n",
                     tc->name, nb_atomic_flows);
    }
}

void
parsec_dtd_atomic_write_order(parsec_dtd_taskpool_t *dtd_tp, const parsec_dtd_task_t *this_task,
                              parsec_dtd_tile_t *tile, int tile_op_type)
{
    const int access = tile_op_type & PARSEC_GET_OP_TYPE;
    int fence_rank = -1;
    parsec_dtd_atomic_fence_t fence;

    /* A fence needs no fence */
    if( parsec_dtd_is_atomic_write_fence((const parsec_dtd_task_class_t *)this_task->super.task_class) ) {
        return;
    }
    fence = parsec_dtd_atomic_write_needs_fence(tile, access, this_task->rank, &fence_rank);
    if( PARSEC_DTD_ATOMIC_FENCE_NONE != fence ) {
        parsec_dtd_atomic_write_insert_fence(dtd_tp, tile, tile_op_type & PARSEC_GET_REGION_INFO,
                                             fence, fence_rank,
                                             parsec_dtd_atomic_write_follower_needs_pushout(this_task, access, fence_rank));
    }
}

void
parsec_dtd_atomic_write_track(parsec_dtd_task_t *this_task, int flow_index, parsec_dtd_tile_t *tile, int access)
{
    parsec_dtd_tile_atomic_write_t *atomic_write = &tile->atomic_write;
    const parsec_task_class_t *tc = this_task->super.task_class;
    const int run_is_open = (PARSEC_ATOMIC_WRITE == atomic_write->last_access);
    int in_run;

    if( PARSEC_ATOMIC_WRITE == access ) {
        if( !run_is_open ) {  /* a new run starts */
            if( 0 == ++atomic_write->run ) {  /* 0 is for the tasks that are not part of a run */
                atomic_write->run = 1;
            }
            atomic_write->gpu = 0;
        }
        atomic_write->rank = this_task->rank;
        for( int chore = 0; NULL != tc->incarnations[chore].hook; chore++ ) {
            if( (this_task->super.chore_mask & (1 << chore)) && PARSEC_DEV_IS_GPU(tc->incarnations[chore].type) ) {
                atomic_write->gpu = 1;
            }
        }
        in_run = 1;
    } else {
        /* The epilogue of the run executes on the device of the run, to push the data out of it.
         * Any other access ends the run. */
        in_run = run_is_open && parsec_dtd_is_atomic_write_fence((const parsec_dtd_task_class_t *)tc);
        atomic_write->rank = -1;
        atomic_write->gpu = 0;
    }
    if( in_run && parsec_dtd_task_is_local(this_task) ) {
        this_task->atomic_write.run = atomic_write->run;
        this_task->atomic_write.flow = flow_index;
    }
    atomic_write->last_access = access;
}

/**
 * Close the open run of a tile about to be flushed. The flush copies the data of the last writer to
 * the owner, which is the epilogue after a run, but the epilogue is not always needed: not when the
 * run left the data in the host memory of the owner, that is when
 *  - the run executed on the owner, after a writer (if any) that was on the owner too: the tasks
 *    updated that writer's copy, and the flush waits for them like for any write;
 *  - no task of the run can execute on a GPU: otherwise only the epilogue can push the result out.
 */
void
parsec_dtd_atomic_write_close_run(parsec_dtd_taskpool_t *dtd_tp, parsec_dtd_tile_t *tile)
{
    parsec_dtd_tile_user_t last_writer;

    if( PARSEC_ATOMIC_WRITE != tile->atomic_write.last_access ) {
        return;
    }
    parsec_dtd_last_user_lock(&(tile->last_user));
    READ_FROM_TILE(last_writer, tile->last_writer);
    parsec_dtd_last_user_unlock(&(tile->last_user));

    if( tile->atomic_write.rank == tile->rank && !tile->atomic_write.gpu &&
        (NULL == last_writer.task || last_writer.task->rank == tile->rank) ) {
        return;
    }
    parsec_dtd_atomic_write_insert_fence(dtd_tp, tile, tile->arena_index,
                                         PARSEC_DTD_ATOMIC_FENCE_EPILOGUE, tile->atomic_write.rank, 1);
}

/* **************************************************************************** */
/* Execution                                                                     */
/* **************************************************************************** */

/* The tasks of a run execute concurrently on the copy and each one is a write: the version bump is atomic */
void
parsec_dtd_atomic_write_cpu_update(parsec_data_copy_t *copy)
{
    int8_t data_owner_dev;

    (void)parsec_atomic_fetch_inc_int32((volatile int32_t *)&copy->version);
    data_owner_dev = copy->original->owner_device;
    assert(data_owner_dev < parsec_mca_device_enabled());
    if( data_owner_dev >= 0 && parsec_mca_device_is_gpu(data_owner_dev) ) {
        /* Pure write, no read hold taken. The ownership transfer is
         * idempotent, so concurrent tasks may all go through it. */
        parsec_data_transfer_ownership_to_copy(copy->original, 0, PARSEC_FLOW_ACCESS_WRITE);
    }
}

/**
 * The tasks of a run write one copy, so they must all execute on the same device: another one would
 * make its own copy and lose the writes. The first task of the run to be selected picks the device:
 * like for any WRITE flow, the preferred device of the data or the GPU that owns it, else the lowest
 * index (the CPU when the task can run there).
 */
static int32_t
parsec_dtd_atomic_write_pick_device(parsec_task_t *this_task)
{
    parsec_dtd_task_t *dtd_task = (parsec_dtd_task_t *)this_task;
    parsec_taskpool_t *tp = this_task->taskpool;
    unsigned int valid_types = 0;

    for( int chore = 0; NULL != this_task->task_class->incarnations[chore].hook; chore++ ) {
        if( this_task->chore_mask & (1 << chore) ) {
            valid_types |= this_task->task_class->incarnations[chore].type & PARSEC_DEV_ANY_TYPE;
        }
    }

    if( NULL != this_task->data[dtd_task->atomic_write.flow].data_in ) {
        parsec_device_module_t *device =
            parsec_select_device_for_write_data(tp, this_task->data[dtd_task->atomic_write.flow].data_in->original,
                                                valid_types);
        if( NULL != device ) {
            return device->device_index;
        }
    }

    for( int index = 0; index < (int)parsec_nb_devices; index++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(index);
        if( NULL == device || (device->type & PARSEC_DEV_RECURSIVE) ) continue;
        if( !(tp->devices_index_mask & (1 << index)) ) continue;
        if( device->type & valid_types ) {
            return index;
        }
    }
    return -1;
}

/*
 * The device of a run is latched in its tile: the run and the device index in one 64-bit word, so the
 * first task that picks wins with a single CAS. Nothing is allocated and nothing is reset: the tasks
 * of a run are only selected once the whole previous run is over (whatever follows a run waits for
 * all its tasks), so a latch holding another run is stale, and nobody needing it is left.
 */
static inline int64_t
parsec_dtd_atomic_write_latch_pack(int32_t run, int32_t device)
{
    return (int64_t)(((uint64_t)(uint32_t)run << 32) | (uint32_t)device);
}

static inline int32_t
parsec_dtd_atomic_write_latch_run(int64_t latch)
{
    return (int32_t)((uint64_t)latch >> 32);
}

static inline int32_t
parsec_dtd_atomic_write_latch_device(int64_t latch)
{
    return (int32_t)((uint64_t)latch & 0xffffffffu);
}

/**
 * Evaluate hook of every chore of the DTD task classes, which keeps the tasks of a run on one device.
 * The device selection calls it once per chore, in order, until one selects a device: this hook
 * selects the device of the run, or rejects the chore (clearing its bit in the chore mask), so the
 * chore under evaluation is the lowest one left. The epilogue of a run follows its device but never
 * picks it: the tasks of the run were selected first.
 */
parsec_hook_return_t
parsec_dtd_atomic_write_evaluate_chore(const parsec_task_t *task)
{
    parsec_task_t *this_task = (parsec_task_t *)task;  /* selecting the device is what an evaluate hook is for */
    parsec_dtd_task_t *dtd_task = (parsec_dtd_task_t *)this_task;
    int chore;

    if( 0 == dtd_task->atomic_write.run ) {
        return PARSEC_HOOK_RETURN_DONE;  /* the common case: nothing to enforce */
    }

    parsec_dtd_flow_info_t *flow = FLOW_OF(dtd_task, dtd_task->atomic_write.flow);
    volatile int64_t *latch = &flow->tile->atomic_write.latch;
    const int32_t run = dtd_task->atomic_write.run;
    const int picks = (PARSEC_ATOMIC_WRITE == (flow->op_type & PARSEC_GET_OP_TYPE));

    for( chore = 0; !(this_task->chore_mask & (1 << chore)); chore++ ) /* nothing */;

    int64_t current = parsec_atomic_fetch_add_int64(latch, 0);
    while( parsec_dtd_atomic_write_latch_run(current) != run ) {
        if( !picks ) {
            return PARSEC_HOOK_RETURN_DONE;
        }
        int32_t picked = parsec_dtd_atomic_write_pick_device(this_task);
        if( picked < 0 ) {
            parsec_fatal("DTD task '%s' has a PARSEC_ATOMIC_WRITE flow, but no device the taskpool can use has a chore for it\n",
                         this_task->task_class->name);
        }
        /* The first one wins: when the swap fails another task of the run did it */
        if( parsec_atomic_cas_int64(latch, current, parsec_dtd_atomic_write_latch_pack(run, picked)) ) {
            current = parsec_dtd_atomic_write_latch_pack(run, picked);
        } else {
            current = parsec_atomic_fetch_add_int64(latch, 0);
        }
    }

    int32_t device_index = parsec_dtd_atomic_write_latch_device(current);
    parsec_device_module_t *device = parsec_mca_device_get(device_index);
    if( NULL != device && (this_task->task_class->incarnations[chore].type & device->type) &&
        (this_task->taskpool->devices_index_mask & (1 << device_index)) ) {
        this_task->selected_device = device;
        return PARSEC_HOOK_RETURN_DONE;
    }
    if( 0 == (this_task->chore_mask & ~((2 << chore) - 1)) ) {
        parsec_fatal("DTD task '%s' has a PARSEC_ATOMIC_WRITE flow, which makes it execute on device %d (%s) like the other "
                     "tasks that write the same data concurrently, but it has no chore for that device\n",
                     this_task->task_class->name, device_index, (NULL == device) ? "unknown" : device->name);
    }
    return PARSEC_HOOK_RETURN_NEXT;
}
