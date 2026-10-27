/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/* Naive star-based reduce-bcast allreduce; just an example, so keep it
 * simple... */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/arena.h"
#include "parsec/data_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

static int verbose = 0;

/* IDs for the Arena Datatypes */
static int TILE_FULL;

int
fill_data( parsec_execution_stream_t    *es,
           parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int scalar;
    int rank;

    parsec_dtd_unpack_args(this_task, &data, &scalar);

    /* initialize with some value */
    rank = this_task->taskpool->context->my_rank;
    *data = scalar;
    if(verbose) printf( "My rank: %d, producing data %d\n", rank, *data );

    return PARSEC_HOOK_RETURN_DONE;
}

int
reduce_accumulate( parsec_execution_stream_t    *es,
                   parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);

    if(verbose) printf( "My rank: %d, reducing %d += %d\n", this_task->taskpool->context->my_rank, *data, *second_data );

    *data += *second_data;

    return PARSEC_HOOK_RETURN_DONE;
}

int
bcast_recv( parsec_execution_stream_t    *es,
            parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);

    printf( "My rank: %d, bcast recv data: %d\n", this_task->taskpool->context->my_rank, *data );

    *second_data = *data;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    parsec_arena_datatype_t *adt;
    int rc, nb, nt = 0;
    int rank, world, cores = -1, root = 0;
    int i;
    parsec_tiled_matrix_t *dcA;

    nb = 1; /* tile_size */
    verbose = 0;

    int pargc = 0; char **pargv = NULL;
    for( i = 1; i < argc; i++) {
        if( 0 == strncmp(argv[i], "--", 3) ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
        if( 0 == strncmp(argv[i], "-n=", 3) ) {
            nt = strtol(argv[i]+3, NULL, 10);
            if( 0 >= nt ) nt = 0;  /* set to default value after rank discovery */
            continue;
        }
        if( 0 == strncmp(argv[i], "-v", 2) ) {
            verbose = 1;
            continue;
        }
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &pargc, &pargv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( 0 >= nt ) {
        nt = world*10; /* total no. of tiles */
    }

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    adt = parsec_matrix_adt_new_rect(
            parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    parsec_matrix_block_cyclic_t *m = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(m, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                              rank,
                              nb, 1,
                              nt*nb, 1,
                              0, 0,
                              nt*nb, 1,
                              world, 1,
                              1, 1,
                              0, 0);

    m->mat = parsec_data_allocate((size_t)m->super.nb_local_tiles *
                                  (size_t)m->super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(m->super.mtype));
    dcA = (parsec_tiled_matrix_t*)m;

    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

//Reduce:
// *********************
    key = A->data_key(A, rank, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = rank;

    for( i = 0; i < nt; i ++ ) {
        parsec_dtd_insert_task( dtd_tp, fill_data, 0, PARSEC_DEV_CPU, "fill_data",
                                PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                sizeof(int), &i, PARSEC_VALUE,
                                PARSEC_DTD_ARG_END );
        if(i != root) {
            parsec_dtd_insert_task( dtd_tp, reduce_accumulate, 0,  PARSEC_DEV_CPU, "reduce_accumulate",
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, root), PARSEC_INOUT | TILE_FULL,
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, i),    PARSEC_INPUT | TILE_FULL | PARSEC_AFFINITY, /* not the best affinity, but it exercises more PaRSEC subsystems that way */
                                    PARSEC_DTD_ARG_END );
        }
    }


//Test force data flush back to root; note that other than for the printf below,
//this would be unnecessary, PaRSEC would do the correct thing in bcast anyway
//and merge the reduce/bcast steps
//*********************
    rc = parsec_dtd_data_flush_all( dtd_tp, A );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_data_flush_all(A)");
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    if( rank == root) {
        printf("Root: %d; value=%d\n\n", root, ((int*)m->mat)[0] );
    }

    //Broadcast:
    // *********************

    for( i = 0; i < world; i++ ) {
        if( i != root ) {
            parsec_dtd_insert_task( dtd_tp, bcast_recv, 0,  PARSEC_DEV_CPU, "bcast_recv",
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, root),  PARSEC_INPUT | TILE_FULL,
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, i),     PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                    PARSEC_DTD_ARG_END );
        }
    }
//******************

    rc = parsec_dtd_data_flush_all( dtd_tp, A );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_data_flush_all(A)");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_taskpool_free( dtd_tp );

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    parsec_data_free(m->mat);
    parsec_tiled_matrix_destroy(dcA);
    free(dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
