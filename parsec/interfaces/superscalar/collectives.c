/**
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/class/lifo.h"
#include "parsec/parsec_internal.h"
#include "parsec/parsec_config.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/interfaces/superscalar/insert_function.h"

#ifdef PARSEC_DTD_DIST_COLLECTIVES

/**
 * Create and return `parsec_remote_deps_t` structure associated with
 * the broadcast of the a data to all the nodes set in the
 * `dest_ranks` array.
 **/
parsec_remote_deps_t* parsec_dtd_create_remote_deps(
      int myrank, int root, parsec_data_copy_t *data_copy,
      parsec_arena_datatype_t *arenas_datatype, 
      int* dest_ranks, int num_dest_ranks) {

   parsec_remote_deps_t *deps = (parsec_remote_deps_t*)remote_deps_allocate(&parsec_remote_dep_context.freelist);

   assert(NULL != deps);
   assert(NULL == deps->taskpool);
         
   deps->root = root;
   deps->outgoing_mask |= (1 << 0); /* only 1 flow */
   deps->max_priority  = 0;

   struct remote_dep_output_param_s* output = &deps->output[0];
   output->data.data   = NULL;
   output->data.arena  = arenas_datatype->arena;
   output->data.layout = arenas_datatype->opaque_dtt;
   output->data.count  = 1;
   output->data.displ  = 0;
   output->priority    = 0;

   if (myrank == root) {
      // if my rank corresponds to the root for this broadcast then we
      // add `data_copy` to the remote deps information
      // `data.data`. Otherwise, we leave it to NULL.
      output->data.data = data_copy;
   }
   
   int _array_pos, _array_mask;
   uint32_t dest_rank_idx;
   if(myrank == root) {
       // Loop through destination ranks in `dest_rank` array 
       for (dest_rank_idx = 0; dest_rank_idx < (uint32_t)num_dest_ranks; ++dest_rank_idx) {

           // Get rank from `dest_rank` array
           uint32_t dest_rank = dest_ranks[dest_rank_idx];

           // Skip if we are root
           if(deps->root == dest_rank) continue;

           _array_pos = dest_rank / (8 * sizeof(uint32_t));
           _array_mask = 1 << (dest_rank % (8 * sizeof(uint32_t)));

           if( !(output->rank_bits[_array_pos] & _array_mask) ) {
               output->rank_bits[_array_pos] |= _array_mask;
               output->deps_mask |= (1 << 0); /* not used by DTD? */
               output->count_bits++;
           }
       }
   } else{
       _array_pos = myrank / (8 * sizeof(uint32_t));
       _array_mask = 1 << (myrank % (8 * sizeof(uint32_t)));

       output->rank_bits[_array_pos] |= _array_mask;
       output->deps_mask |= (1 << 0); /* not used by DTD? */
       output->count_bits++;
   }
    
   return deps;
}

/**
 * Perform a broadcast for of the dtd tile `dtd_tile_root` from the
 * root node associated with the rank `root` to the nodes with ranks
 * set in the `dest_ranks` array.
 **/
void parsec_dtd_broadcast(
      parsec_taskpool_t *taskpool, int root,
      parsec_dtd_tile_t* dtd_tile_root, int arena_index,
      //parsec_dtd_tile_t* bcast_keys_root, int bcast_arena_index,
      int* dest_ranks, int num_dest_ranks) {
 
    parsec_dtd_tile_t* bcast_keys_root = NULL;
    int bcast_arena_index = 15;


    parsec_data_copy_t *parsec_data_copy;
    int *data_ptr;
    int key;
    int bcast_id;
    int myrank = taskpool->context->my_rank;
    parsec_dtd_taskpool_t *dtd_tp = (parsec_dtd_taskpool_t *)taskpool;
    
    //bcast_keys_root = (parsec_dtd_tile_t *) parsec_thread_mempool_allocate( parsec_bcast_keys_tile_mempool->thread_mempools );
    bcast_keys_root = (parsec_dtd_tile_t *) malloc(sizeof(parsec_dtd_tile_t));
    SET_LAST_ACCESSOR(bcast_keys_root);
    bcast_keys_root->dc = NULL;
    bcast_keys_root->arena_index = -1;
    bcast_keys_root->key = (uint64_t) bcast_id;
    bcast_keys_root->rank = root;
    bcast_keys_root->flushed = NOT_FLUSHED;
    parsec_data_copy_t* new_data_copy = PARSEC_OBJ_NEW(parsec_data_copy_t);

    new_data_copy->coherency_state = PARSEC_DATA_COHERENCY_OWNED;
    new_data_copy->device_private = malloc(sizeof(int)*2500);
    bcast_keys_root->data_copy = new_data_copy;
    
    if(myrank == root) {
        bcast_id = ( (1<<30) | (root << 18) | dtd_tp->bcast_id);
        dtd_tp->bcast_id++;
       
        
        parsec_data_copy = bcast_keys_root->data_copy;
        data_ptr = (int*)parsec_data_copy_get_ptr(parsec_data_copy);
        data_ptr[0] = bcast_id;
        data_ptr[600] = num_dest_ranks;
        for(int i = 0; i < num_dest_ranks; i++) {
            data_ptr[dest_ranks[i]+1] = dtd_tp->send_task_id[dest_ranks[i]]++;
            //pack the ranks at the end of the tiles as well
            data_ptr[600+i+1] = dest_ranks[i];
        }
        bcast_keys_root->ht_item.key = (parsec_key_t)((uintptr_t)data_ptr[0]);
        
        //fprintf(stderr, "on rank %d inserting key tile into bcast_keys_hash with key %ld num dest ranks %d\n", myrank, bcast_keys_root->ht_item.key, data_ptr[400]); 
        parsec_hash_table_insert(parsec_bcast_keys_hash, &bcast_keys_root->ht_item);
        parsec_mfence(); /* Write */
    }

    // Retrieve DTD tile's data_copy
    //parsec_data_copy_t *data_copy = dtd_tile_root->data_copy;
    //parsec_data_copy_t *key_copy = bcast_keys_root->data_copy;

    // Create remote deps corresponding to the braodcast
    /*
    parsec_remote_deps_t *deps_0 = parsec_dtd_create_remote_deps(
            myrank, root, data_copy, &parsec_dtd_arenas_datatypes[arena_index],
            dest_ranks, num_dest_ranks);
    parsec_remote_deps_t *deps_1 = parsec_dtd_create_remote_deps(
            myrank, root, key_copy, &parsec_dtd_arenas_datatypes[bcast_arena_index],
            dest_ranks, num_dest_ranks);
    */
    parsec_task_t *bcast_task_root = parsec_dtd_taskpool_create_task(
            taskpool, parsec_dtd_bcast_data_fn, 0, "bcast_data_fn",
            PASSED_BY_REF, dtd_tile_root, PARSEC_INOUT | arena_index,
            sizeof(int), &root, PARSEC_VALUE | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END);

    parsec_dtd_task_t *dtd_bcast_task_root = (parsec_dtd_task_t *)bcast_task_root;

    // Set broadcast topology info
    //deps_0->pending_ack = 0;
    //dtd_bcast_task_root->deps_out = deps_0;

    if(myrank == root) {
        dtd_bcast_task_root->ht_item.key = bcast_id;
        dtd_bcast_task_root->super.locals[0].value = dtd_bcast_task_root->ht_item.key;
    }else{
        bcast_id = ( (1<<28)  | (root << 18) | dtd_tp->recv_task_id[root]++);
        dtd_bcast_task_root->ht_item.key =  bcast_id;
        dtd_bcast_task_root->super.locals[0].value = dtd_bcast_task_root->ht_item.key;
    }

    parsec_task_t *bcast_key_root = parsec_dtd_taskpool_create_task(
            taskpool, parsec_dtd_bcast_key_fn, 0, "bcast_key_fn",
            PASSED_BY_REF, bcast_keys_root, PARSEC_INOUT | bcast_arena_index,
            sizeof(int), &root, PARSEC_VALUE | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END);
    parsec_dtd_task_t *dtd_bcast_key_root = (parsec_dtd_task_t *)bcast_key_root;
    //deps_1->pending_ack = 0;
    //dtd_bcast_key_root->deps_out = deps_1;
    if(myrank == root) {
        /* nothing here since the key is stored in the key array and will be updated before remote_dep_activate */
    }else{
        bcast_id = ( (1<<29)  | (root << 18) | (dtd_tp->recv_task_id[root] -1));
        dtd_bcast_key_root->ht_item.key =  bcast_id;
        dtd_bcast_key_root->super.locals[0].value = dtd_bcast_key_root->ht_item.key;
    }
    /* Post the bcast of keys and ranks array */
    /* Post the bcast tasks for the actual data */
    parsec_insert_dtd_task(dtd_bcast_task_root);
    //sleep(1);
    parsec_insert_dtd_task(dtd_bcast_key_root);
}

#endif
