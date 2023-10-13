/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/* Naive star-based reduce-bcast allreduce; just an example, so keep it
 * simple... */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#if PARSEC_VERSION_MAJOR < 4
#include "parsec/interfaces/superscalar/insert_function.h"
#else
#include "parsec/interfaces/dtd/insert_function.h"
#endif
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/arena.h"
#include "parsec/data_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

static int verbose = 0;

#if PARSEC_VERSION_MAJOR < 4
enum regions {
               TILE_FULL,
             };
#else
/* IDs for the Arena Datatypes */
static int TILE_FULL;
#endif

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
    int rc, nb, nt;
    int rank, world, cores = -1, root = 0;
    int i;
    parsec_tiled_matrix_t *dcA;

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

    nb = 1; /* tile_size */
    nt = world*10; /* total no. of tiles */
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
            if( 0 >= nt ) nt = world*10;  /* set to default value */
            continue;
        }
        if( 0 == strncmp(argv[i], "-v", 2) ) {
            verbose = 1;
            continue;
        }
        i++;  /* skip this one */
        continue;
    }

    parsec = parsec_init( cores, &pargc, &pargv );
    if( NULL == parsec ) {
        return -1;
    }

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );
#if PARSEC_VERSION_MAJOR < 4
    parsec_add2arena_rect(&parsec_dtd_arenas_datatypes[TILE_FULL],
#else
    parsec_arena_datatype_t *adt;
    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena_rect( adt,
#endif
                                 parsec_datatype_int32_t,
                                 nb, 1, nb);

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


    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

//Reduce:
// *********************

    for( i = 0; i < nt; i ++ ) {
        parsec_dtd_taskpool_insert_task( dtd_tp, fill_data, 0,  "fill_data",
                            PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, i), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                            sizeof(int),      &i, PARSEC_VALUE,
                            PARSEC_DTD_ARG_END );
        if(i != root) {
            parsec_dtd_taskpool_insert_task( dtd_tp, reduce_accumulate, 0,  "reduce_accumulate",
                            PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, root), PARSEC_INOUT | TILE_FULL,
                            PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(A, i),    PARSEC_INPUT | TILE_FULL | PARSEC_AFFINITY, /* not the best affinity, but it exercises more PaRSEC subsystems that way */
                            PARSEC_DTD_ARG_END );
        }
    }

//Test force data flush back to root; note that other than for the printf below,
//this would be unecessary, PaRSEC would do the correct thing in bcast anyway
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
            parsec_dtd_taskpool_insert_task( dtd_tp, bcast_recv, 0,  "bcast_recv",
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

#if PARSEC_VERSION_MAJOR < 4
    PARSEC_OBJ_RELEASE(parsec_dtd_arenas_datatypes[0].arena);
#else
    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
#endif
    parsec_dtd_data_collection_fini( A );
    parsec_tiled_matrix_destroy_data(dcA);
    parsec_data_collection_destroy(&dcA->super);
    free(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
