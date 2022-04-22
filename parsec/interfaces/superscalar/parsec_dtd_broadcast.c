/**
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/mca_param.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#ifdef PARSEC_DTD_DIST_COLLECTIVES

void
populate_remote_deps(int* data_ptr, parsec_remote_deps_t* remote_deps)
{
    struct remote_dep_output_param_s* output = &remote_deps->output[0];
    int _array_pos, _array_mask;
    uint32_t dest_rank_idx;
    /* TODO: don't assume the length of data_ptr */
    int num_dest_ranks = data_ptr[600];
    for(dest_rank_idx = 0; dest_rank_idx < (uint32_t)num_dest_ranks; ++dest_rank_idx) {
        uint32_t dest_rank = data_ptr[600+dest_rank_idx+1];
        _array_pos = dest_rank / (8 * sizeof(uint32_t));
        _array_mask = 1 << (dest_rank % (8 * sizeof(uint32_t)));

        if( !(output->rank_bits[_array_pos] & _array_mask) ) {
            output->rank_bits[_array_pos] |= _array_mask;
            output->count_bits++;
        }
    }
}

/* when the comm_coll_bcast is 1 we use the chain topology, get the successor's rank */
int
get_chain_successor(parsec_execution_stream_t*es, parsec_task_t* task, parsec_remote_deps_t* remote_deps)
{
    (void)task;
    int my_idx, idx, current_mask;
    unsigned int array_index, count, bit_index;
    uint32_t boffset;
    uint32_t dep_fw_mask[es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof];
    memset(dep_fw_mask, 0, es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
    memcpy(&dep_fw_mask, remote_deps->remote_dep_fw_mask, es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
    struct remote_dep_output_param_s* output = &remote_deps->output[0];
    boffset = remote_deps->root / (8 * sizeof(uint32_t));
    dep_fw_mask[boffset] |= ((uint32_t)1) << (remote_deps->root % (8 * sizeof(uint32_t)));
    my_idx = (remote_deps->root == es->virtual_process->parsec_context->my_rank) ? 0 : -1;
    idx = 0;
    for(array_index = count = 0; count < remote_deps->output[0].count_bits; array_index++) {
        current_mask = output->rank_bits[array_index];
        if( 0 == current_mask ) continue;
        for( bit_index = 0; current_mask != 0; bit_index++ ) {
            if( !(current_mask & (1 << bit_index)) ) continue;
            int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
            current_mask ^= (1 << bit_index);
            count++;

            boffset = rank / (8 * sizeof(uint32_t));
            idx++;
            if(my_idx == -1) {
                if(rank == es->virtual_process->parsec_context->my_rank) {
                    my_idx = idx;
                }
                boffset = rank / (8 * sizeof(uint32_t));
                dep_fw_mask[boffset] |= ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
                continue;
            }
            if(my_idx != -1){
                if(idx == my_idx+1)
                {
                    return rank;
                }
            }
        }
    }
    return -1;
}

void
parsec_dtd_bcast_key_iterate_successors(parsec_execution_stream_t *es,
        const parsec_task_t *this_task,
        uint32_t action_mask,
        parsec_ontask_function_t *ontask,
        void *ontask_arg)
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    int current_dep;
    parsec_dtd_tile_t *tile;

    parsec_dep_t deps;
    parsec_dep_data_description_t data;
    int vpid_dst=0;

    /* finding for which flow we need to iterate successors of */
    int flow_mask = action_mask;
    int my_rank = current_task->super.taskpool->context->my_rank;
    int successor = -1;

    int rc; /* retrive the mca number for comm_coll_bcast */
    int comm_coll_bcast; /* retrive the value set for comm_coll_bcast */
    if (0 < (rc = parsec_mca_param_find("runtime", NULL, "comm_coll_bcast")) ) {
        parsec_mca_param_lookup_int(rc, &comm_coll_bcast);
    }
    for( current_dep = 0; current_dep < current_task->super.task_class->nb_flows; current_dep++ ) {
        if( (flow_mask & (1<<current_dep)) ) {
            if (action_mask & PARSEC_ACTION_COMPLETE_LOCAL_TASK) {
                /* root of the bcast key */
                parsec_remote_deps_t *deps = NULL;
                PARSEC_ALLOCATE_REMOTE_DEPS_IF_NULL(deps, this_task, MAX_PARAM_COUNT);
                deps->root = my_rank;
                deps->outgoing_mask |= (1 << 0); /* only 1 flow */
                deps->max_priority  = 0;

                struct remote_dep_output_param_s* output = &deps->output[0];
                output->data.data   = current_task->super.data[0].data_out;//NULL;
                output->data.arena  = parsec_dtd_arenas_datatypes[15].arena;
                output->data.layout = parsec_dtd_arenas_datatypes[15].opaque_dtt;
                output->data.count  = 1;
                output->data.displ  = 0;
                output->priority    = 0;
                int* data_ptr = (int*)parsec_data_copy_get_ptr(current_task->super.data[0].data_out);
                //successor = get_chain_successor(es, (parsec_task_t*)current_task, deps);
                //current_task->super.locals[0].value = current_task->ht_item.key = ((1<<29) |(my_rank << 20) | *(data_ptr+1+successor));
                tile = FLOW_OF(current_task, current_dep)->tile;
                parsec_dtd_tile_retain(tile);
                populate_remote_deps(data_ptr, deps);
                parsec_remote_dep_activate(
                        es, (parsec_task_t *)current_task,
                        deps,
                        deps->outgoing_mask);
                //current_task->deps_out = NULL;
                /* decrease the count as in the data flush */
                parsec_dtd_release_local_task( current_task );
            } else if (action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
                /* a node in the key array propagation */
                parsec_release_dep_fct_arg_t* arg = (parsec_release_dep_fct_arg_t*)ontask_arg;
                parsec_remote_deps_t* deps = arg->remote_deps;
                int root = deps->root;
                int my_rank = current_task->super.taskpool->context->my_rank;

                int _array_pos, _array_mask;
                struct remote_dep_output_param_s* output;
                output = &deps->output[0];
                _array_pos = my_rank / (8 * sizeof(uint32_t));
                _array_mask = 1 << (my_rank % (8 * sizeof(uint32_t)));

                /* We are part of the broadcast, forward message */
                int* data_ptr = (int*)parsec_data_copy_get_ptr(current_task->super.data[0].data_out);
                int* buffer = malloc(sizeof(int)*50*50);
                memcpy(buffer, data_ptr, sizeof(int)*50*50);
                populate_remote_deps(data_ptr, deps);
                //successor = get_chain_successor(es, (parsec_task_t*)current_task, current_task->deps_out);
                if(successor == -1) {
                    //current_task->deps_out->outgoing_mask = 0;
                }
                //current_task->super.locals[0].value = current_task->ht_item.key = ((1<<29) | (root << 20) | *(data_ptr+1+successor));
                assert(NULL != current_task->super.data[current_dep].data_out);

                //current_task->deps_out->output[0].data.data = current_task->super.data[0].data_out;
                //parsec_dtd_retain_data_copy(current_task->super.data[current_dep].data_out);
                parsec_remote_dep_activate(
                        es, (parsec_task_t *)current_task,
                        deps,
                        deps->outgoing_mask);
                //current_task->deps_out = NULL;
                /* update the BCAST DATA task or dep with the global ID that we know now */
                uint64_t key = ((uint64_t)(1<<28 | (root << 18 ) | data_ptr[es->virtual_process->parsec_context->my_rank+1])<<32) | (1U<<0);
                uint64_t key2 = ((uint64_t)(data_ptr[0])<<32) | (1U<<0);

                parsec_dtd_task_t* dtd_task = NULL;
                parsec_dtd_taskpool_t *tp = (parsec_dtd_taskpool_t *)current_task->super.taskpool;
                parsec_hash_table_lock_bucket(tp->task_hash_table, (parsec_key_t)key);
                dtd_task = parsec_dtd_find_task(tp, key);
               
                // store the meta data info into the key hash table
                dtd_hash_table_pointer_item_t *item = (dtd_hash_table_pointer_item_t *)parsec_thread_mempool_allocate(tp->hash_table_bucket_mempool->thread_mempools);
                parsec_hash_table_t *hash_table = tp->keys_hash_table;
                item->ht_item.key   = (parsec_key_t)key;
                item->mempool_owner = tp->hash_table_bucket_mempool->thread_mempools;
                item->value         = (void *)buffer;
                parsec_hash_table_insert( hash_table, &item->ht_item );
               
                //parsec_dtd_tile_t* bcast_keys = (parsec_dtd_tile_t *) parsec_thread_mempool_allocate( parsec_bcast_keys_tile_mempool->thread_mempools );
                parsec_dtd_tile_t* bcast_keys = (parsec_dtd_tile_t *)malloc(sizeof(parsec_dtd_tile_t));
                bcast_keys->ht_item.key   = (parsec_key_t)((uintptr_t)buffer[0]);
                bcast_keys->data_copy = PARSEC_OBJ_NEW(parsec_data_copy_t);
                bcast_keys->data_copy->device_private = (void *)buffer;
                parsec_hash_table_insert( parsec_bcast_keys_hash, &bcast_keys->ht_item );
                parsec_mfence(); /* Write */
                //fprintf(stderr, "insert into parsec_bcast_keys_hash the item %p key %d with value pointer %p on rank %d\n", bcast_keys, buffer[0], buffer, es->virtual_process->parsec_context->my_rank);

                if(dtd_task != NULL) {
                    parsec_hash_table_lock_bucket(tp->task_hash_table, (parsec_key_t)key2);
                    parsec_remote_deps_t *dep = parsec_dtd_find_task(tp, key2);

                    //populate_remote_deps(data_ptr, dtd_task->deps_out);
                    parsec_dtd_untrack_task(tp, key);
                    if(dep == NULL){
                        dtd_task->super.locals[0].value = data_ptr[0];
                        parsec_dtd_track_task(tp, key2, dtd_task);
                    }else{

                        dtd_task->super.locals[0].value = data_ptr[0];
                        parsec_dtd_untrack_remote_dep(tp, key2);
                        parsec_dtd_track_task(tp, key2, dtd_task);
                        remote_dep_dequeue_delayed_dep_release(dep);
                    }
                    parsec_hash_table_unlock_bucket(tp->task_hash_table, (parsec_key_t)key2);
                }
                parsec_hash_table_unlock_bucket(tp->task_hash_table, (parsec_key_t)key);
                tile = FLOW_OF(current_task, current_dep)->tile;
                parsec_dtd_tile_retain(tile);
            } else {
                /* on the receiver side, get datatype to aquire datatype, arena etc info */
                data.data   = current_task->super.data[current_dep].data_out;
                data.arena  = parsec_dtd_arenas_datatypes[FLOW_OF(current_task, current_dep)->arena_index].arena;
                data.layout = parsec_dtd_arenas_datatypes[FLOW_OF(current_task, current_dep)->arena_index].opaque_dtt;
                data.count  = 1;
                data.displ  = 0;
                deps.cond            = NULL;
                deps.ctl_gather_nb   = NULL;
                deps.flow            = current_task->super.task_class->out[current_dep];
                deps.dep_index       = 0;
                deps.belongs_to      = current_task->super.task_class->out[current_dep];
                deps.direct_data     = NULL;
                deps.dep_datatype_index = current_dep;
                ontask( es, (parsec_task_t *)current_task, (parsec_task_t *)current_task,
                        &deps, &data, current_task->rank, my_rank, vpid_dst, ontask_arg );
            }
        }
    }
}

/* **************************************************************************** */
/**
 * Body of bcast key task we insert that will propagate the key array
 * empty body!
 *
 * @param   context, this_task
 *
 * @ingroup DTD_INTERFACE_INTERNAL
 */
int
parsec_dtd_bcast_key_fn( parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    //fprintf(stderr, "executed the body of bcast key fn\n");
    return PARSEC_HOOK_RETURN_DONE;
}

/* **************************************************************************** */
/**
 * Body of bcast task we insert that will propagate the data tile we are broadcasting
 * empty body!
 *
 * @param   context, this_task
 *
 * @ingroup DTD_INTERFACE_INTERNAL
 */
int
parsec_dtd_bcast_data_fn( parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    //fprintf(stderr, "executed the body of bcast data fn\n");
    return PARSEC_HOOK_RETURN_DONE;
}

#endif
