#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"
#ifdef PARSEC_VTRACE
#include "parsec/vt_user.h"
#endif

#include "common_data.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

uint32_t count = 0;

enum regions {
               TILE_FULL,
             };

int
call_to_kernel_type_read( parsec_execution_stream_t *es,
                          parsec_task_t *this_task )
{
    (void)es; (void)this_task;
    int *data;

    parsec_dtd_unpack_args(this_task,
                           UNPACK_DATA,  &data);
    if( *data > 1 ) {
        (void)parsec_atomic_inc_32b(&count);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int
call_to_kernel_type_write( parsec_execution_stream_t    *es,
                           parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task,
                           UNPACK_DATA,  &data);
    *data += 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores;
    int nb, nt;
    parsec_tiled_matrix_dc_t *dcA;

    cores = 8;
    int i, j;
    int no_of_tasks, no_of_read_tasks = 5, key;

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

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    no_of_tasks = world;
    nb = 1; /* tile_size */
    nt = no_of_tasks; /* total no. of tiles */

    parsec = parsec_init( cores, &argc, &argv );

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    #if 0
    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data;
    for( i = 0; i < no_of_tasks; i++ ) {
        key = A->data_key(A, i, 0);
        if( rank == A->rank_of_key(A, key) ) {
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = 0;
            parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
        }
    }
    #endif

    #if 0
    char hostname[1024];
    gethostname(hostname, 1024);
    printf("ssh %s module \tgdb -p %d\n", hostname, getpid());
    int ls = 1;
    while(ls) {
    }
    #endif

    /* Registering the dtd_taskpool with PARSEC context */
    parsec_enqueue( parsec, dtd_tp );

    parsec_context_start(parsec);

    for( i = 0; i < no_of_tasks; i++ ) {
        key = A->data_key(A, i, 0);
        parsec_dtd_taskpool_insert_task( dtd_tp, call_to_kernel_type_write,    0,  "Write_Task",
                           PASSED_BY_REF,    TILE_OF_KEY(A, key),   INOUT | TILE_FULL | AFFINITY,
                           0 );
        for( j = 0; j < no_of_read_tasks; j++ ) {
            parsec_dtd_taskpool_insert_task( dtd_tp, call_to_kernel_type_read,   0,   "Read_Task",
                               PASSED_BY_REF,    TILE_OF_KEY(A, key),   INPUT | TILE_FULL | AFFINITY,
                               0 );
        }
        parsec_dtd_taskpool_insert_task( dtd_tp, call_to_kernel_type_write,    0,  "Write_Task",
                           PASSED_BY_REF,    TILE_OF_KEY(A, key),   INOUT | TILE_FULL | AFFINITY,
                           0 );
    }

    parsec_dtd_data_flush_all( dtd_tp, A );

    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    parsec_context_wait(parsec);

    if( count > 0 ) {
        parsec_fatal( "Write after Read dependencies are not bsing satisfied properly\n\n" );
    } else {
        #if 0
        for( i = 0; i < no_of_tasks; i++ ) {
            key = A->data_key(A, i, 0);
            if( rank == A->rank_of_key(A, key) ) {
                data = A->data_of_key(A, key);
                gdata = data->device_copies[0];
                real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
                parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
            }
        }
        #endif
        parsec_output( 0, "\nWAR test passed\n\n" );
    }

    parsec_taskpool_free( dtd_tp );

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
