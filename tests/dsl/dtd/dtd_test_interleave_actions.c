/*
 * Copyright (c) 2020-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if defined(PARSEC_HAVE_MPI)
#include "mpi.h"
#endif  /* defined(PARSEC_HAVE_MPI) */

#include <getopt.h>

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "tests/tests_data.h"

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int recv_data_kernel(
      parsec_execution_stream_t  *es,
      parsec_task_t *this_task ) {
   (void)es;
   int *data_in;
   int rank;
   
   parsec_dtd_unpack_args(this_task, &data_in, &rank);

   printf("[recv_data_kernel] rank = %d, data_in = %d\n", rank, *data_in);
    
   return PARSEC_HOOK_RETURN_DONE;
}

void busy_wait() {

   // Debug
   int stop = 1;
   while (stop) {}

}

#define DELAY_ADD_TASKPOOL  (1<<0)
#define DELAY_INSERT_TASK   (1<<1)
#define DELAY_FLUSH_ALL     (1<<2)
#define DELAY_TASKPOOL_WAIT (1<<3)

int main(int argc, char **argv) {

   int ret;
   parsec_context_t* parsec_context = NULL;
   unsigned int action_mask = DELAY_ADD_TASKPOOL
                              | DELAY_INSERT_TASK
                              | DELAY_FLUSH_ALL
                              | DELAY_TASKPOOL_WAIT;

    int rank, world;
   int pargc;
   char **pargv;
   int c;
   parsec_arena_datatype_t *adt;

   while (1) {
       int option_index = 0;
       static struct option long_options[] = {
                {"add-taskpool", no_argument, 0,  'a' },
                {"insert-task",  no_argument, 0,  'i' },
                {"flush-all",    no_argument, 0,  'f' },
                {"taspool-wait", no_argument, 0,  'w' },
                {"help",         no_argument, 0,  'h'},
                {0,         0,                0,  0 }
        };

        c = getopt_long(argc, argv, "aifwh",
                        long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            case 'a':
                action_mask &= ~DELAY_ADD_TASKPOOL;
                break;
            case 'i':
                action_mask &= ~DELAY_INSERT_TASK;
                break;
            case 'f':
                action_mask &= ~DELAY_FLUSH_ALL;
                break;
            case 'w':
                action_mask &= ~DELAY_TASKPOOL_WAIT;
                break;
            case 'h':
            case '?':
                fprintf(stderr, "Usage %s [-a|-i|-f|-w] [-- <parsec options>]\n"
                                " Insert delays on ranks > 0 to stress less used paths in the code\n"
                                "   -a: do not add delay before add taskpool\n"
                                "   -i: do not add delay before insert task\n"
                                "   -f: do not add delay before flush all\n"
                                "   -w: do not add delay before taskpool wait\n"
                                "   -h: print this help\n",
                        argv[0]);

                break;
        }
    }
    pargc = argc - optind;
    pargv = argv + optind;

    #if defined(PARSEC_HAVE_MPI)
   {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
   }
   MPI_Comm_size(MPI_COMM_WORLD, &world);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
   world = 1;
   rank = 0;
#endif

   int nb = 1;
   int nt = 1;

   nb = 1;
   nt = 1;

   int ncores = -1;
   parsec_context = parsec_init(ncores, &pargc, &pargv);

   if(world == 1) {
       parsec_warning("*** This test only makes sense with at least two nodes");
   }

   parsec_tiled_matrix_t *dcA;
   dcA = create_and_distribute_data(rank, world, nb, nt);
   parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

   parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
   parsec_dtd_data_collection_init(A);

   parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

   adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
   parsec_add2arena_rect( adt,
         parsec_datatype_int32_t,
         nb, 1, nb);

   if( 0 == rank ) {
      parsec_data_copy_t *parsec_data_copy;
      parsec_data_t *parsec_data;
      int *data_ptr;
      int key;

      key = A->data_key(A, rank, 0, 0);
      parsec_data = A->data_of_key(A, key);
      parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
      data_ptr = (int*)parsec_data_copy_get_ptr(parsec_data_copy);
      *data_ptr = 1; 
      parsec_output( 0, "Initial data, node: %d A At key[%d]: %d\n", rank, key, *data_ptr );
   }

   ret = parsec_context_start(parsec_context);
   PARSEC_CHECK_ERROR(ret, "parsec_context_start");

   if( (DELAY_ADD_TASKPOOL & action_mask) && rank != 0) {
      parsec_output( 0, "Node: %d waiting for rank 0 to register the taskpool\n", rank);
      // The following sleep statememt ensure that rank `0` has enough
      // time to send data to other processes before adding the taskpool.
      sleep(1);
   }   
      
   // Registering the dtd_handle with PARSEC context
   ret = parsec_context_add_taskpool( parsec_context, dtd_tp );
   PARSEC_CHECK_ERROR(ret, "parsec_context_add_taskpool");

   if( (DELAY_INSERT_TASK & action_mask) && rank != 0) {
      parsec_output( 0, "Node: %d waiting for rank 0 to insert tasks\n", rank);
      // The following sleep statememt ensure that rank `0` has enough
      // time to send data to other processes before adding the taskpool.
      sleep(1);
   }

   for (int i = 1; i < world; ++i) {
      parsec_dtd_insert_task(
            dtd_tp, recv_data_kernel, 0, PARSEC_DEV_CPU, "RecvData",
            PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INPUT | TILE_FULL,
            sizeof(int), &i, PARSEC_VALUE | PARSEC_AFFINITY,
            PARSEC_DTD_ARG_END);
   }

   if( (DELAY_FLUSH_ALL & action_mask) && rank != 0) {
      parsec_output( 0, "Node: %d waiting for rank 0 to flush\n", rank);
      // The following sleep statememt ensure that rank `0` has enough
      // time to send data to other processes before adding the taskpool.
      sleep(1);
   }

   parsec_dtd_data_flush_all( dtd_tp, A );

    if( (DELAY_TASKPOOL_WAIT & action_mask) && rank != 0) {
        parsec_output( 0, "Node: %d waiting for rank 0 to complete first\n", rank);
        // The following sleep statememt ensure that rank `0` has enough
        // time to send data to other processes before adding the taskpool.
        sleep(1);
    }

    // Wait for task completion
   ret = parsec_taskpool_wait(dtd_tp);
   PARSEC_CHECK_ERROR(ret, "parsec_taskpool_wait");
   // return 0;

   ret = parsec_context_wait(parsec_context);
   PARSEC_CHECK_ERROR(ret, "parsec_context_wait");

    // Cleanup data and parsec data structures
   parsec_del2arena(adt);
   PARSEC_OBJ_RELEASE(adt->arena);
   parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);
   parsec_dtd_data_collection_fini( A );
   free_data(dcA);

   parsec_taskpool_free( dtd_tp );

   parsec_fini(&parsec_context);

#ifdef PARSEC_HAVE_MPI
   MPI_Finalize();
#endif

   return 0;
}
