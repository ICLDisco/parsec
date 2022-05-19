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
   double *val_in;
   // Value to set the data to
   double data_value;
   // Task rank 
   int dest_rank;
   
   parsec_dtd_unpack_args(this_task, &val_in, &dest_rank, &data_value);

   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   //sleep(1);
   //printf("[write_task] rank = %d, data_value = %f\n", myrank, data_value);

   *val_in = data_value;
      
   return PARSEC_HOOK_RETURN_DONE;
}

// Retrieve value associated with input data_copy for verification.
int retrieve_task_fn(
      parsec_execution_stream_t  *es,
      parsec_task_t *this_task ) {
   (void)es;

   int myrank = -1;
   // INPUT data
   double *val_in;
   // Task rank 
   int dest_rank;

   double *val_out;

   parsec_dtd_unpack_args(this_task, &val_in, &dest_rank, &val_out);

   /* int myrank; */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

   //printf("[read_task] rank = %d, val_in = %f\n", myrank, *val_in);

   //*val_out = *val_in;
   
   return PARSEC_HOOK_RETURN_DONE;
}


int dummy_task_fn(
      parsec_execution_stream_t *es,
      parsec_task_t *this_task) {
   (void)es;(void)this_task;

   return PARSEC_HOOK_RETURN_DONE;   
}

int test_broadcast_mixed(
      int world, int myrank, parsec_context_t* parsec_context, int root, int num_elem) {

   // Test return value:
   // - 0: success
   // - Failure otherwise
   int ret = 0;

   // Error code return by parsec routines
   int perr;
   
   // Tile size
   int nb = 1;
   int nb_bcast = 30;
   // Total number of tiles
   int nt = 1;
   int data_value = 0;

   //sleep(40);
   //number of elements per tile
   nb = num_elem;
   // few tiles per node 
   nt = world; 
   double starttime, endtime;
   
   parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

   parsec_matrix_add2arena_rect(
         &parsec_dtd_arenas_datatypes[TILE_FULL],
         parsec_datatype_double_t,
         nb, 1, nb);
   
   // Initial value on the root node. All node should have this value
   // at the end of the operation.
   double data_root = 55;

   

   parsec_tiled_matrix_dc_t *dcA;
   dcA = create_and_distribute_data(myrank, world, nb, nt);
   parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

   parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
   two_dim_block_cyclic_t *__dcA = dcA;
   parsec_dtd_data_collection_init(A);

   parsec_data_copy_t *parsec_data_copy;
   parsec_data_t *parsec_data;
   // Pointer to local tile data
   double *data_ptr;
   // Local tile key
   int key;

   key = A->data_key(A, myrank, 0);
   parsec_data = A->data_of_key(A, key);
   parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
   data_ptr = (double*)parsec_data_copy_get_ptr(parsec_data_copy);

   // Registering the dtd_handle with PARSEC context
   perr = parsec_context_add_taskpool( parsec_context, dtd_tp );
   PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");
   
   perr = parsec_context_start(parsec_context);
   PARSEC_CHECK_ERROR(perr, "parsec_context_start");
   
   MPI_Barrier(MPI_COMM_WORLD);
   starttime = MPI_Wtime();
   // Initialize tiles
   if( root == myrank ) {
       parsec_task_t *root_task = parsec_dtd_taskpool_create_task(
               dtd_tp, write_task_fn, 0, "root_task",
               PASSED_BY_REF, PARSEC_DTD_TILE_OF(A, myrank, 0), PARSEC_INOUT | TILE_FULL,
               sizeof(int), &myrank, PARSEC_VALUE | PARSEC_AFFINITY,
               sizeof(double*), &data_root, PARSEC_VALUE,
               PARSEC_DTD_ARG_END);
       parsec_dtd_task_t *dtd_root_task = (parsec_dtd_task_t *)root_task;
       parsec_insert_dtd_task(dtd_root_task);
   }
   
   // Key of tile associated with root node
   int key_root;
   parsec_dtd_tile_t* dtd_tile_root;
   
   key_root = key = A->data_key(A, root, 0);
   dtd_tile_root = PARSEC_DTD_TILE_OF_KEY(A, key_root);

   // Create array of destination ranks
   int num_dest_ranks = 0;
   int *dest_ranks = (int*)malloc(world*sizeof(int));

   // Destination rank index
   int dest_rank_idx = 0 ;

   // VALID ONLY ON THE ROOT NODE
   for (int rank = 0; rank < world; ++rank) {
      if (rank == root) continue;
      dest_ranks[dest_rank_idx] = rank;
      ++dest_rank_idx;
   }
   num_dest_ranks = dest_rank_idx;

   //
   // Perform Broadcast
   //
   //fprintf(stderr, "perform bcast from rank %d\n", myrank);
   parsec_dtd_broadcast(
           dtd_tp, root,
           dtd_tile_root, TILE_FULL,
           dest_ranks, num_dest_ranks);

   //
   // Retrieve value of broadcasted data
   //
   double* data_value_out = -1;
   if(1) {
       parsec_task_t *retrieve_task = parsec_dtd_taskpool_create_task(
               dtd_tp, retrieve_task_fn, 0, "retrieve_task",
               PASSED_BY_REF, dtd_tile_root, PARSEC_INPUT | TILE_FULL,
               sizeof(int), &myrank, PARSEC_VALUE | PARSEC_AFFINITY,
               sizeof(double*), &data_value_out, PARSEC_VALUE,
               PARSEC_DTD_ARG_END);
       parsec_dtd_task_t *dtd_retrieve_task = (parsec_dtd_task_t *)retrieve_task;
       parsec_insert_dtd_task(retrieve_task);
   }
   parsec_dtd_data_flush_all( dtd_tp, A );
 
   // Wait for task completion
   perr = parsec_dtd_taskpool_wait( dtd_tp );
   PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

   perr = parsec_context_wait(parsec_context);
   PARSEC_CHECK_ERROR(perr, "parsec_context_wait");
   MPI_Barrier(MPI_COMM_WORLD);
   endtime   = MPI_Wtime();
   if(myrank==0)printf("That took %f seconds\n",endtime-starttime);
   
   // Cleanup data and parsec data structures
   parsec_type_free(&parsec_dtd_arenas_datatypes[TILE_FULL].opaque_dtt);
   PARSEC_OBJ_RELEASE(parsec_dtd_arenas_datatypes[TILE_FULL].arena);
   parsec_dtd_data_collection_fini( A );
   free_data(dcA);
   parsec_taskpool_free( dtd_tp );
   
   return ret;
}

int main(int argc, char **argv) {

   int ret;
   parsec_context_t* parsec_context = NULL;
   int rank, world;

   char *p;
   int nt = strtol(argv[1], &p, 10);
   nt = nt*nt;
   {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
   }
   MPI_Comm_size(MPI_COMM_WORLD, &world);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
   /* int ncores = 1; */
   int ncores = 2;
   parsec_context = parsec_init(ncores, &argc, &argv);
   
   // Testing trimming with a mixed destinations of receivers for broadcast
   //MPI_Barrier(MPI_COMM_WORLD);
   //starttime = MPI_Wtime();
   ret = test_broadcast_mixed(world, rank, parsec_context, 0, nt);
   //MPI_Barrier(MPI_COMM_WORLD);
   //endtime   = MPI_Wtime();
   //if(rank==0)printf("That took %f seconds\n",endtime-starttime);

   parsec_fini(&parsec_context);

   MPI_Finalize();
   (void)ret;
   return 0;
}
