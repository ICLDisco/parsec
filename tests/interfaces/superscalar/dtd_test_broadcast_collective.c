#include "mpi.h"

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/remote_dep.h"
#include "parsec/data_internal.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/interfaces/superscalar/insert_function.h"

#include <stdlib.h>
#include <stdio.h>

enum regions
   {
    TILE_FULL,
    TILE_BCAST
   };

parsec_tiled_matrix_dc_t *create_and_distribute_data(int rank, int world, int mb, int mt)
{
   two_dim_block_cyclic_t *m = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
   two_dim_block_cyclic_init(m, matrix_ComplexDouble, matrix_Tile,
                             rank,
                             mb, 1,
                             mt*mb, 1,
                             0, 0,
                             mt*mb, 1,
                             world, 1,
                             1, 1,
                             0, 0);

   m->mat = parsec_data_allocate((size_t)m->super.nb_local_tiles *
                                 (size_t)m->super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(m->super.mtype));

   return (parsec_tiled_matrix_dc_t*)m;
}

void free_data(parsec_tiled_matrix_dc_t *d)
{
    parsec_matrix_destroy_data(d);
    parsec_data_collection_destroy(&d->super);
    free(d);
}

// Read data
int read_task_fn(
      parsec_execution_stream_t  *es,
      parsec_task_t *this_task ) {
   (void)es;

   // INPUT data
   int *val_in;
   // Task rank 
   int dest_rank;

   parsec_dtd_unpack_args(this_task, &val_in, &dest_rank);

   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   printf("[read_task] rank = %d, val_in = %d\n", myrank, *val_in);

   return PARSEC_HOOK_RETURN_DONE;
}

// Write data
int write_task_fn(
      parsec_execution_stream_t  *es,
      parsec_task_t *this_task) {
   (void)es;

   // INOUT data
   int *val_out;
   // Value to set the data to
   int data_value;
   // Task rank 
   int dest_rank;
   
   parsec_dtd_unpack_args(this_task, &val_out, &data_value, &dest_rank);

   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   printf("[write_task] rank = %d, data_value = %d\n", myrank, data_value);

   *val_out = data_value;
      
   return PARSEC_HOOK_RETURN_DONE;
}

// For debugging purpose
void busy_wait() {
   // Debug
   int stop = 1;
   while (stop) {}
}


// Retrieve value associated with input data_copy for verification.
int retrieve_task_fn(
      parsec_execution_stream_t  *es,
      parsec_task_t *this_task ) {
   (void)es;

   int myrank = -1;
   // INPUT data
   int *val_in;
   // Task rank 
   int dest_rank;

   int *val_out;

   parsec_dtd_unpack_args(this_task, &val_in, &dest_rank, &val_out);

   /* int myrank; */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

    printf("[read_task] rank = %d, val_in = %d\n", myrank, *val_in);

   *val_out = *val_in;
   
   return PARSEC_HOOK_RETURN_DONE;
}


int dummy_task_fn(
      parsec_execution_stream_t *es,
      parsec_task_t *this_task) {
   (void)es;(void)this_task;

   return PARSEC_HOOK_RETURN_DONE;   
}

int test_broadcast_mixed(
      int world, int myrank, parsec_context_t* parsec_context, int root) {

   // Test return value:
   // - 0: success
   // - Failure otherwise
   int ret = 0;

   // Error code return by parsec routines
   int perr;
   
   // Tile size
   int nb = 1;
   int nb_bcast = 200;
   // Total number of tiles
   int nt = 1;
   int data_value = 0;

   //sleep(40);
   // One element per tile
   nb = 1;
   // few tiles per node 
   nt = world*5; 
   
   parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

   parsec_matrix_add2arena_rect(
         &parsec_dtd_arenas_datatypes[TILE_FULL],
         parsec_datatype_int32_t,
         nb, 1, nb);

   parsec_matrix_add2arena_rect(
         &parsec_dtd_arenas_datatypes[TILE_BCAST],
         parsec_datatype_int32_t,
         nb_bcast, 1, nb_bcast);
   // Initial value on the root node. All node should have this value
   // at the end of the operation.
   int data_root = 55;

   // Final value received on non-root nodes.
   int *data_value_out = (int*) calloc(1, sizeof(int)); 
   *data_value_out = -33;
   
   if( root == myrank ) {
      data_value = data_root;
   }
   else {
      data_value = -10-myrank;
   }

   parsec_tiled_matrix_dc_t *dcB;
   dcB = create_and_distribute_data(myrank, world, nb_bcast, nt);
   parsec_data_collection_set_key((parsec_data_collection_t *)dcB, "B");

   parsec_data_collection_t *B = (parsec_data_collection_t *)dcB;
   parsec_dtd_data_collection_init(B);

   parsec_tiled_matrix_dc_t *dcA;
   dcA = create_and_distribute_data(myrank, world, nb, nt);
   parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

   parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
   parsec_dtd_data_collection_init(A);
   // Initialize tiles

   parsec_data_copy_t *parsec_data_copy;
   parsec_data_t *parsec_data;
   // Pointer to local tile data
   int *data_ptr;
   // Local tile key
   int key;

   key = A->data_key(A, myrank, 0);
   parsec_data = A->data_of_key(A, key);
   parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
   data_ptr = (int*)parsec_data_copy_get_ptr(parsec_data_copy);
   if (root == myrank) {
      *data_ptr = data_value; 
   }
   else {
      // Initialise this value with rubbish. It should be equal to
      // `data_value` after the execution on even-indexed processes.
      data_value_out = data_ptr;
   }
   //parsec_output(0, "Initial data, node: %d A At key[%d]: %d\n", myrank, key, *data_ptr);

   // Registering the dtd_handle with PARSEC context
   perr = parsec_context_add_taskpool( parsec_context, dtd_tp );
   PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");
   
   perr = parsec_context_start(parsec_context);
   PARSEC_CHECK_ERROR(perr, "parsec_context_start");
   
   fprintf(stderr, "parsec context started\n");
   // Key of tile associated with root node
   int key_root = key = A->data_key(A, root, 0);
   parsec_dtd_tile_t* dtd_tile_root = PARSEC_DTD_TILE_OF_KEY(A, key_root);
   key_root = B->data_key(B, root, 0);
   parsec_dtd_tile_t* bcast_keys_root = PARSEC_DTD_TILE_OF_KEY(B, key_root);
   
   // Create array of destination ranks
   int num_dest_ranks = 0;
   int *dest_ranks = (int*)malloc(world*sizeof(int));

   // Destination rank index
   int dest_rank_idx = 0 ;

   // Put odd rank indexes into `dest_ranks` array except for the root
   // node. VALID ONLY ON THE ROOT NODE
   for (int rank = 0; rank < world; ++rank) {
      if (rank % 2 == 0 || rank == root) continue;
      
      dest_ranks[dest_rank_idx] = rank;
      ++dest_rank_idx;
   }
   num_dest_ranks = dest_rank_idx;

   //
   // Perform Broadcast
   //
   if(myrank % 2 == 1 || myrank == root) {
       fprintf(stderr, "perform bcast from rank %d\n", myrank);
       parsec_dtd_broadcast(
               dtd_tp, myrank, root,
               dtd_tile_root, TILE_FULL,
               bcast_keys_root, TILE_BCAST,
               dest_ranks, num_dest_ranks);
   }

   //
   // Retrieve value of broadcasted data
   //
   if(myrank % 2 == 1 || myrank == root) {
       for (int rank = 0; rank < world; ++rank) {

           if (rank % 2 == 0 || rank == root) continue;

           parsec_task_t *retrieve_task = parsec_dtd_taskpool_create_task(
                   dtd_tp, retrieve_task_fn, 0, "retrieve_task",
                   PASSED_BY_REF, dtd_tile_root, PARSEC_INPUT | TILE_FULL,
                   sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
                   sizeof(int*), &data_value_out, PARSEC_VALUE,
                   PARSEC_DTD_ARG_END);
           //parsec_dtd_task_t *dtd_retrieve_task = (parsec_dtd_task_t *)retrieve_task;
           //parsec_insert_dtd_task(retrieve_task);

       }
   }
for(int iter=1; iter <= 0; iter++) {
   // Second round of broadcast, create another array of keys for this bcast
   key_root = B->data_key(B, root+iter*world, 0);
   //key_root = B->data_key(B, root, 0);
   bcast_keys_root = PARSEC_DTD_TILE_OF_KEY(B, key_root);
 
   //sleep(5);
   int new_value = -1;
   if (root == myrank) {
      //*data_ptr = 1998;
       new_value = 1998+iter;
   }
   else {
      //data_value_out = data_ptr;
   }
   
   parsec_dtd_taskpool_insert_task(dtd_tp, 
           write_task_fn, 0, "write_task",
           PASSED_BY_REF, dtd_tile_root, PARSEC_INOUT | TILE_FULL,
           sizeof(int), &new_value, PARSEC_VALUE,
           sizeof(int), &root, PARSEC_VALUE | PARSEC_AFFINITY,
           PARSEC_DTD_ARG_END);
   
   // Put all rank indexes into `dest_ranks` array except for the root
   // node.
   dest_rank_idx = 0;
   for (int rank = 0; rank < world; ++rank) {
      if (rank == root) continue;
      dest_ranks[dest_rank_idx] = rank;
      ++dest_rank_idx;
   }
   num_dest_ranks = dest_rank_idx;
   
   //
   // Perform Broadcast AGAIN
   //
   parsec_dtd_broadcast(
           dtd_tp, myrank, root,
           dtd_tile_root, TILE_FULL,
           bcast_keys_root, TILE_BCAST,
           dest_ranks, num_dest_ranks);

   //
   // Retrieve value of broadcasted data
   //
   for (int rank = 0; rank < world; ++rank) {
       if ( rank == root) continue;
       parsec_task_t *retrieve_task = parsec_dtd_taskpool_create_task(
               dtd_tp, retrieve_task_fn, 0, "retrieve_task",
               PASSED_BY_REF, dtd_tile_root, PARSEC_INPUT | TILE_FULL,
               sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
               sizeof(int*), &data_value_out, PARSEC_VALUE,
               PARSEC_DTD_ARG_END);
       //parsec_dtd_task_t *dtd_retrieve_task = (parsec_dtd_task_t *)retrieve_task;
       //parsec_insert_dtd_task(retrieve_task);

   }
} 
   parsec_dtd_data_flush_all( dtd_tp, A );
   //parsec_dtd_data_flush_all( dtd_tp, B );
 
   // Wait for task completion
   perr = parsec_dtd_taskpool_wait( dtd_tp );
   PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

   perr = parsec_context_wait(parsec_context);
   PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

   // Check whether we obtained the correct value on the current node
   // at the end of the test. Odd processes should have received the
   // value form the root and other processes should have kept their
   // original value
   if ((myrank == root) ||
       ((myrank % 2 == 1) && (data_root == *data_value_out)) ||
       ((myrank % 2 == 0) && (data_value == *data_ptr))) {
      // Data received as expected
      ret = 0;
   }
   else {
      // Error
      ret = -1;
   }
   
//   parsec_output( 0, "Checking result, node: %d, data_value_out: %d\n", myrank, *data_value_out );
   
   // Cleanup data and parsec data structures
   parsec_type_free(&parsec_dtd_arenas_datatypes[TILE_FULL].opaque_dtt);
   PARSEC_OBJ_RELEASE(parsec_dtd_arenas_datatypes[TILE_FULL].arena);
   parsec_type_free(&parsec_dtd_arenas_datatypes[TILE_BCAST].opaque_dtt);
   PARSEC_OBJ_RELEASE(parsec_dtd_arenas_datatypes[TILE_BCAST].arena);
   parsec_dtd_data_collection_fini( A );
   parsec_dtd_data_collection_fini( B );
   free_data(dcA);
   free_data(dcB);

   parsec_taskpool_free( dtd_tp );
   
   return ret;

}

int main(int argc, char **argv) {

   int ret;
   parsec_context_t* parsec_context = NULL;

   int rank, world;

   {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
   }
   MPI_Comm_size(MPI_COMM_WORLD, &world);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
   /* int ncores = 1; */
   int ncores = 2;
   parsec_context = parsec_init(ncores, &argc, &argv);

   // Root node for the broadcast operation

   sleep(30);
   //
   // Simple broadcast
   
   // Testing trimming with a mixed destinations of receivers for broadcast
   ret = test_broadcast_mixed(world, rank, parsec_context, 0);


   parsec_fini(&parsec_context);

   MPI_Finalize();
   (void)ret;
   return 0;
}
