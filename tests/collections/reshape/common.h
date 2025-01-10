/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _reshape_h
#define _reshape_h

#include "parsec.h"
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/parsec_config.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

int reshape_set_matrix_value(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args);

int reshape_set_matrix_value_count(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args);

int reshape_set_matrix_value_count_lower2upper_matrix(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args);

int reshape_set_matrix_value_lower_tile(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args);


int reshape_set_matrix_value_position(parsec_execution_stream_t *es,
                                      const parsec_tiled_matrix_t *descA,
                                      void *_A, parsec_matrix_uplo_t uplo,
                                      int m, int n, void *args);

int reshape_set_matrix_value_position_swap(parsec_execution_stream_t *es,
                                      const parsec_tiled_matrix_t *descA,
                                      void *_A, parsec_matrix_uplo_t uplo,
                                      int m, int n, void *args);

int check_matrix_equal(parsec_matrix_block_cyclic_t dcA, parsec_matrix_block_cyclic_t dcA_check);

int reshape_print(parsec_execution_stream_t *es,
                  const parsec_tiled_matrix_t *descA,
                  void *_A, parsec_matrix_uplo_t uplo,
                  int m, int n, void *args);

#if defined(PARSEC_HAVE_MPI)
#define BARRIER MPI_Barrier(MPI_COMM_WORLD);
#else
#define BARRIER
#endif

#if defined(PARSEC_HAVE_MPI)
  #define DO_INIT_MPI()                                                \
      int provided;                                                    \
      int requested = m ? MPI_THREAD_MULTIPLE : MPI_THREAD_SERIALIZED; \
      MPI_Init_thread(&argc, &argv, requested, &provided);             \
      MPI_Comm_size(MPI_COMM_WORLD, &nodes);                           \
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);                            \
      if( requested > provided ) {                                     \
          fprintf(stderr, "#XXXXX User requested %s but the implementation returned a lower thread\n", requested==MPI_THREAD_MULTIPLE? "MPI_THREAD_MULTIPLE": "MPI_THREAD_SERIALIZED");\
          exit(2);                                                     \
      }
#else
  #define DO_INIT_MPI()                                                \
      nodes = 1;                                                       \
      rank = 0;
#endif

#define DO_INIT()                                                                        \
    char *name;                                                                          \
    int do_sleep = 0;                                                                    \
    int pargc = 0;                                                                       \
    char **pargv;                                                                        \
    while ((ch = getopt(argc, argv, "m:M:N:t:T:s:S:P:Q:c:I:R:h:w")) != -1) {             \
        switch (ch) {                                                                    \
            case 'm': m = atoi(optarg); break;                                           \
            case 'N': N = M = atoi(optarg); break;                                       \
            case 't': MB = NB = atoi(optarg); break;                                     \
            case 'T': MB = NB = atoi(optarg); break;                                     \
            case 'P': P = atoi(optarg); break;                                           \
            case 'c': cores = atoi(optarg); break;                                       \
            case 'w': do_sleep = 1; break;                                               \
            case '?': case 'h': default:                                                 \
                fprintf(stderr,                                                          \
                        "-m : initialize MPI_THREAD_MULTIPLE (default: 0/no)\n"          \
                        "-N : rowxcolumn dimension (N, M) of the matrices (default: 8)\n"\
                        "-t : row dimension (MB) of the tiles (default: 4)\n"            \
                        "-T : column dimension (NB) of the tiles (default: 4)\n"         \
                        "-P : rows (P) in the PxQ process grid (default: 1)\n"           \
                        "-c : number of cores used (default: -1)\n"                      \
                        "-w : sleep (default off) \n"                                    \
                        "\n");                                                           \
                 exit(1);                                                                \
        }                                                                                \
    }                                                                                    \
    DO_INIT_MPI();                                                                       \
    if(do_sleep) sleep(10);                                                              \
    /* Initialize PaRSEC */                                                              \
    pargc = 0; pargv = NULL;                                                             \
    for(int i = 1; i < argc; i++) {                                                      \
        if( strcmp(argv[i], "--") == 0 ) {                                               \
            pargc = argc - i;                                                            \
            pargv = argv + i;                                                            \
            break;                                                                       \
        }                                                                                \
    }                                                                                    \
    parsec = parsec_init(cores, &pargc, &pargv);                                         \
    if( NULL == parsec ) {                                                               \
        /* Failed to correctly initialize. In a correct scenario report*/                \
         /* upstream, but in this particular case bail out.*/                            \
        exit(-1);                                                                        \
    }                                                                                    \
    (void)name;


#define DO_INI_DATATYPES()                                                \
    parsec_arena_datatype_t adt_default;                                  \
    parsec_arena_datatype_t adt_lower;                                    \
    parsec_arena_datatype_t adt_upper;                                    \
    parsec_add2arena( &adt_default,                                \
                            parsec_datatype_int_t,                        \
                            PARSEC_MATRIX_FULL,                     \
                            1, MB, NB, MB,                                \
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );             \
                                                                          \
    parsec_add2arena( &adt_lower,                                  \
                             parsec_datatype_int_t,                       \
                             PARSEC_MATRIX_LOWER, 1, MB, NB, MB,          \
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );            \
                                                                          \
    parsec_add2arena( &adt_upper,                                  \
                             parsec_datatype_int_t,                       \
                             PARSEC_MATRIX_UPPER, 1, MB, NB, MB,          \
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );            \
    (void)adt_default; (void)adt_lower; (void)adt_upper;

#define DO_FINI_DATATYPES()                                              \
    parsec_del2arena(&adt_default);                               \
    parsec_del2arena(&adt_lower);                                 \
    parsec_del2arena(&adt_upper);


#define DO_RUN(ctp) do {                                                 \
    tp = (parsec_taskpool_t*)ctp;                                        \
    parsec_context_add_taskpool(parsec, tp);                             \
    parsec_context_start(parsec);                                        \
    parsec_context_wait(parsec);                                         \
    parsec_taskpool_free((parsec_taskpool_t*)tp);                        \
} while(0)


#define DO_CHECK(NAME, dc, dc_check) do {                                \
    cret = check_matrix_equal(dc, dc_check );                            \
    if(rank==0)                                                          \
        printf("Test " #NAME " %s\n", (cret > 0)? "FAILED" : "PASSED");  \
    ret |= cret;                                                         \
    BARRIER;                                                             \
} while(0)


#define RUN(NAME, ...) do {                                              \
    tp = NAME##_new(__VA_ARGS__);                                        \
    parsec_context_add_taskpool(parsec, tp);                             \
    parsec_context_start(parsec);                                        \
    parsec_context_wait(parsec);                                         \
    parsec_taskpool_free((parsec_taskpool_t*)tp);                        \
                                                                         \
    cret = check_matrix_equal(dcA, dcA_check );                          \
    if(rank==0)                                                          \
        printf("Test " #NAME " %s\n", (cret > 0)? "FAILED" : "PASSED");  \
    ret |= cret;                                                         \
    BARRIER;                                                             \
} while(0)

#endif
