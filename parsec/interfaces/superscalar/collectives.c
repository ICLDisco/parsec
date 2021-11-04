/**
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/class/lifo.h"
#include "parsec/parsec_config.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#ifdef PARSEC_DTD_DIST_COLLECTIVES

/* static parsec_lifo_t parsec_dep_lifo; */

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

       //if( !(output->rank_bits[_array_pos] & _array_mask) ) {
           output->rank_bits[_array_pos] |= _array_mask;
           output->deps_mask |= (1 << 0); /* not used by DTD? */
           output->count_bits++;
       //}
   }
    
   return deps;
}

/**
 * Free remote deps if it does not involve any participants.
 **/
static
int remote_deps_free_if_empty(parsec_remote_deps_t* deps) {

   // Return 1 if the remote_deps has no participants, 0 otherwise.
   int ret = 0;

   struct remote_dep_output_param_s* output = &deps->output[0];

   // TODO: loop through the whole output array are use max_priority
   // instead
   if (output->count_bits <= 0) {
      // No participants

      deps->pending_ack = 0;
      deps->incoming_mask = 0;
      deps->outgoing_mask = 0;
      remote_deps_free(deps);
      
      // Indicate that remote deps is empty 
      ret = 1;
   }
   
   return ret;
}

/**
 * Perform a broadcast for of the dtd tile `dtd_tile_root` from the
 * root node associated with the rank `root` to the nodes with ranks
 * set in the `dest_ranks` array.
 **/
void parsec_dtd_broadcast(
      parsec_taskpool_t *taskpool, int myrank, int root,
      parsec_dtd_tile_t* dtd_tile_root, int arena_index,
      parsec_dtd_tile_t* bcast_keys_root, int bcast_arena_index,
      int* dest_ranks, int num_dest_ranks) {
  
    parsec_data_copy_t *parsec_data_copy;
    int *data_ptr;
    int key;
    int bcast_id;
    parsec_dtd_taskpool_t *dtd_tp = (parsec_dtd_taskpool_t *)taskpool;
    
    if(myrank == root) {
        bcast_id = ( (1<<30) | (root << 18) | dtd_tp->bcast_id);
        dtd_tp->bcast_id++;
        
        parsec_data_copy = bcast_keys_root->data_copy;
        data_ptr = (int*)parsec_data_copy_get_ptr(parsec_data_copy);
        data_ptr[0] = bcast_id;
        data_ptr[400] = num_dest_ranks;
        for(int i = 0; i < num_dest_ranks; i++) {
            data_ptr[dest_ranks[i]+1] = dtd_tp->send_task_id[dest_ranks[i]]++;
            //pack the ranks at the end of the tiles as well
            data_ptr[400+i+1] = dest_ranks[i];
        }
    }
    //fprintf(stderr, "finished bcast key packing\n");
    // Retrieve DTD tile's data_copy
    parsec_data_copy_t *data_copy = dtd_tile_root->data_copy;
    parsec_data_copy_t *key_copy = bcast_keys_root->data_copy;

    // Create remote deps corresponding to the braodcast
    parsec_remote_deps_t *deps_0 = parsec_dtd_create_remote_deps(
            myrank, root, data_copy, &parsec_dtd_arenas_datatypes[arena_index],
            dest_ranks, num_dest_ranks);
    parsec_remote_deps_t *deps_1 = parsec_dtd_create_remote_deps(
            myrank, root, key_copy, &parsec_dtd_arenas_datatypes[bcast_arena_index],
            dest_ranks, num_dest_ranks);

    parsec_task_t *bcast_task_root = parsec_dtd_taskpool_create_task(
            taskpool, parsec_dtd_bcast_data_fn, 0, "bcast_data_fn",
            PASSED_BY_REF, dtd_tile_root, PARSEC_INOUT | arena_index,
            sizeof(int), &root, PARSEC_VALUE | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END);

    parsec_dtd_task_t *dtd_bcast_task_root = (parsec_dtd_task_t *)bcast_task_root;

    // Set broadcast topology info
    deps_0->pending_ack = 0;
    dtd_bcast_task_root->deps_out = deps_0;

    if(myrank == root) {
        dtd_bcast_task_root->ht_item.key = bcast_id;
        dtd_bcast_task_root->super.locals[0].value = dtd_bcast_task_root->ht_item.key;
    }else{
        bcast_id = ( (1<<28)  | (root << 20) | dtd_tp->recv_task_id[root]++);
        dtd_bcast_task_root->ht_item.key =  bcast_id;
        dtd_bcast_task_root->super.locals[0].value = dtd_bcast_task_root->ht_item.key;
    }

    parsec_task_t *bcast_key_root = parsec_dtd_taskpool_create_task(
            taskpool, parsec_dtd_bcast_key_fn, 0, "bcast_key_fn",
            PASSED_BY_REF, bcast_keys_root, PARSEC_INOUT | bcast_arena_index,
            sizeof(int), &root, PARSEC_VALUE | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END);
    parsec_dtd_task_t *dtd_bcast_key_root = (parsec_dtd_task_t *)bcast_key_root;
    deps_1->pending_ack = 0;
    dtd_bcast_key_root->deps_out = deps_1;
    if(myrank == root) {
        /* nothing here since the key is stored in the key array and will be updated before remote_dep_activate */
    }else{
        bcast_id = ( (1<<29)  | (root << 20) | (dtd_tp->recv_task_id[root] -1));
        //bcast_id = ( (1<<29)  | (dtd_tp->recv_task_id[root] ));
        dtd_bcast_key_root->ht_item.key =  bcast_id;
        dtd_bcast_key_root->super.locals[0].value = dtd_bcast_key_root->ht_item.key;
    }
    /* Post the bcast of keys and ranks array */
    parsec_insert_dtd_task(dtd_bcast_key_root);
    

    if(myrank == root) {
        //for (int dest_rank = 0; dest_rank < num_dest_ranks; ++dest_rank) {
        //    parsec_task_t *retrieve_task = parsec_dtd_taskpool_create_task(
        //            dtd_tp, parsec_dtd_aux_fn, 0, "retrieve_task",
        //            PASSED_BY_REF, bcast_keys_root, PARSEC_INPUT | bcast_arena_index,
        //            sizeof(int), &dest_ranks[dest_rank], PARSEC_VALUE | PARSEC_AFFINITY,
        //            PARSEC_DTD_ARG_END);
        //    parsec_dtd_task_t *dtd_retrieve_task = (parsec_dtd_task_t *)retrieve_task;
        //    parsec_insert_dtd_task(dtd_retrieve_task);
        //}
    }else {
        parsec_task_t *retrieve_task = parsec_dtd_taskpool_create_task(
                dtd_tp, parsec_dtd_bcast_key_recv, 0, "retrieve_task",
                PASSED_BY_REF, bcast_keys_root, PARSEC_INPUT | bcast_arena_index,
                sizeof(int), &myrank, PARSEC_VALUE | PARSEC_AFFINITY,
                PARSEC_DTD_ARG_END);
        parsec_dtd_task_t *dtd_retrieve_task = (parsec_dtd_task_t *)retrieve_task;
        //parsec_insert_dtd_task(dtd_retrieve_task);
    }
    
    /* Post the bcast tasks for the actual data */
    parsec_insert_dtd_task(dtd_bcast_task_root);
}

#endif
