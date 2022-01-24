/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

static volatile int32_t count_war_error = 0;
static volatile int32_t count_raw_error = 0;

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
call_to_kernel_type_read( parsec_execution_stream_t *es,
                          parsec_task_t *this_task )
{
    (void)es; (void)this_task;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);
    if( *data < 1 ) {
        (void)parsec_atomic_fetch_inc_int32(&count_raw_error);
    }
    if( *data > 1 ) {
        (void)parsec_atomic_fetch_inc_int32(&count_war_error);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int
call_to_kernel_type_write( parsec_execution_stream_t    *es,
                           parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);
    *data += 1;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, nt, rc;
    parsec_tiled_matrix_t *dcA;

    int i, j;
    int no_of_tasks, no_of_read_tasks = 5, key;
    parsec_arena_datatype_t *adt;

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

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena_rect( adt,
                                  parsec_datatype_int32_t,
                                  nb, 1, nb);

    dcA = create_and_distribute_data(rank, world, nb, nt);
    memset(((parsec_matrix_block_cyclic_t *)dcA)->mat,
            0,
            (size_t)dcA->nb_local_tiles *
            (size_t)dcA->bsiz *
            (size_t)parsec_datadist_getsizeoftype(dcA->mtype));
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    /* Registering the dtd_taskpool with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( i = 0; i < no_of_tasks; i++ ) {
        key = A->data_key(A, i, 0);
        parsec_dtd_insert_task(dtd_tp, call_to_kernel_type_write, 0, PARSEC_DEV_CPU, "Write_Task",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),   PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
        for( j = 0; j < no_of_read_tasks; j++ ) {
            parsec_dtd_insert_task(dtd_tp, call_to_kernel_type_read, 0, PARSEC_DEV_CPU, "Read_Task",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),   PARSEC_INPUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END );
        }
        parsec_dtd_insert_task(dtd_tp, call_to_kernel_type_write, 0, PARSEC_DEV_CPU, "Write_Task",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),   PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END );
    }

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_dtd_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( count_war_error > 0 ) {
        parsec_fatal( "Write after Read dependencies are not being satisfied properly\n\n" );
    }
    if( count_raw_error > 0 ) {
        parsec_fatal( "Read after Write dependencies are not being satisfied properly\n\n" );
    }
    if( count_raw_error == 0 && count_war_error == 0 ) {
        parsec_output( 0, "WAR test passed\n\n" );
    }

    parsec_taskpool_free( dtd_tp );

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
