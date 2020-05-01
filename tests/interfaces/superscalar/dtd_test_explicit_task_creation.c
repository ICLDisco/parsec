/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "common_data.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

enum regions {
               TILE_FULL,
             };

struct my_datatype {
    int a;
    int b;
    int c;
};

void *ref_check;

int
call_to_kernel_type_write( parsec_execution_stream_t    *es,
                           parsec_task_t *this_task )
{
    int data1, data2;
    double data3;
    struct my_datatype data4;
    int *data5;
    void *ref;

    parsec_dtd_unpack_args(this_task, &data1, &data2, &data3, &data4, &data5, &ref);

    assert(data1 == 10);
    assert(data2 == 20);
    assert(data3 == 10.05);
    assert(data4.a == 1);
    assert(data4.b == 2);
    assert(data4.c == 3);
    assert(*data5 == 30);
    assert(ref == ref_check);

    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rc, rank = 0, world = 1, cores = -1;
    int nb, nt, i, no_of_tasks, key;
    parsec_tiled_matrix_dc_t *dcA;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if( world != 1 ) {
        parsec_fatal( "Nope! world is not right, we need exactly one MPI process. "
                      "Try with \"mpirun -np 1 .....\"\n" );
    }

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

    ref_check = (void *)A;

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data;
    for( i = 0; i < no_of_tasks; i++ ) {
        key = A->data_key(A, i, 0);
        if( rank == (int)A->rank_of_key(A, key) ) {
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = 30;
            parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
        }
    }

    /* Registering the dtd_taskpool with PARSEC context */
    int data1 = 10, data2 = 20;
    double data3 = 10.05;
    struct my_datatype data4;
    data4.a = 1;
    data4.b = 2;
    data4.c = 3;

    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_task_t *task = parsec_dtd_taskpool_create_task(dtd_tp,
                                    call_to_kernel_type_write,    0,  "Write_Task",
                                    sizeof(int), &data1, VALUE,
                                    sizeof(int), &data2, VALUE | AFFINITY,
                                    sizeof(double),  &data3, VALUE,
                                    sizeof(struct my_datatype), &data4, VALUE,
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, 0),   INOUT | TILE_FULL,
                                    sizeof(void *),  A, REF,
                                    PARSEC_DTD_ARG_END);
    parsec_insert_dtd_task(task);

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_dtd_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp );

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
