/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/execution_stream.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <string.h>

#define TYPE  PARSEC_MATRIX_INTEGER

static parsec_matrix_block_cyclic_t dcA;
static int N = 100;
static int block = 10;

static int
parsec_operator_print_id( struct parsec_execution_stream_s *es,
                          const void* src,
                          void* dst,
                          void* op_data, ... )
{
    va_list ap;
    int m, n, rank = 0;

#if defined(PARSEC_HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    va_start(ap, op_data);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "%s: tile (%d, %d) -> %p:%p thread %d of VP %d, process %d\n",
            (char*)op_data, m, n, src, dst, es->th_id, es->virtual_process->vp_id, rank );
    (void)es; (void)src; (void)dst; (void)op_data;
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char* argv[])
{
    parsec_context_t* parsec;
    parsec_taskpool_t *tp1, *tp2, *tp3;
    int nodes, rank, rc, i = 0;

#if defined(PARSEC_HAVE_MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &nodes);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif

    int pargc = 0; char **pargv = NULL;
    for( i = 1; i < argc; i++) {
        if( 0 == strncmp(argv[i], "--", 3) ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
        if( 0 == strncmp(argv[i], "-b=", 3) ) {
            block = strtol(argv[i]+3, NULL, 10);
            if( 0 >= block ) block = 10;
            continue;
        }
        if( 0 == strncmp(argv[i], "-n=", 3) ) {
            N = strtol(argv[i]+3, NULL, 10);
            if( 0 >= N ) N = 100;
            continue;
        }
        if( 0 == strncmp(argv[i], "-h", 2) ) {
            printf("-h: help\n"
                   "-b=<nb> the number of elements on each block\n"
                   "-n=<nb> the number of elements on a dimension\n"
                   "-v=<nb> the verbosity level\n");
            exit(0);
        }
    }

    parsec = parsec_init(1, &pargc, &pargv);
    assert( NULL != parsec );

    parsec_matrix_block_cyclic_init( &dcA, TYPE, PARSEC_MATRIX_TILE,
                               rank,
                               block, 1, N, 1,
                               0, 0, N, 1, nodes, 1, 1, 1, 0, 0);
    parsec_data_collection_set_key(&dcA.super.super, "A");
    dcA.mat = parsec_data_allocate( N * parsec_datadist_getsizeoftype(TYPE) );
    for( int i = 0; i < N; ((int*)dcA.mat)[i++] = 1);

    tp1 = parsec_map_operator_New((parsec_tiled_matrix_t*)&dcA,
                                   NULL,
                                   parsec_operator_print_id,
                                   "tp1");
    tp2 = parsec_map_operator_New((parsec_tiled_matrix_t*)&dcA,
                                   NULL,
                                   parsec_operator_print_id,
                                   "tp2");

    tp3 = parsec_compose((parsec_taskpool_t *)tp1, (parsec_taskpool_t *)tp2);

    rc = parsec_context_add_taskpool(parsec, tp3);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(tp1);
    parsec_taskpool_free(tp2);
    parsec_taskpool_free(tp3);

    parsec_fini(&parsec);

    free(dcA.mat);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}

