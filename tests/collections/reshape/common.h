/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
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
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

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

int check_matrix_equal(parsec_context_t *parsec,
                       parsec_matrix_block_cyclic_t dcA,
                       parsec_matrix_block_cyclic_t dcA_check);

int reshape_print(parsec_execution_stream_t *es,
                  const parsec_tiled_matrix_t *descA,
                  void *_A, parsec_matrix_uplo_t uplo,
                  int m, int n, void *args);

#define BARRIER do {                                                                    \
    int _barrier_rc = parsec_tests_barrier(parsec);                                     \
    if( (PARSEC_SUCCESS != _barrier_rc) &&                                             \
        (PARSEC_ERR_NOT_IMPLEMENTED != _barrier_rc) ) {                                \
        PARSEC_CHECK_ERROR(_barrier_rc, "parsec_tests_barrier");                       \
    }                                                                                  \
} while(0)

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
                        "-m : request multiple-thread support from the test runtime (default: 0/no)\n"\
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
    pargc = 0; pargv = NULL;                                                             \
    for(int i = 1; i < argc; i++) {                                                      \
        if( strcmp(argv[i], "--") == 0 ) {                                               \
            pargc = argc - i;                                                            \
            pargv = argv + i;                                                            \
            break;                                                                       \
        }                                                                                \
    }                                                                                    \
    int _init_rc = parsec_tests_context_init(cores,                                      \
                                             m ? PARSEC_TEST_THREAD_MULTIPLE :           \
                                                 PARSEC_TEST_THREAD_SERIALIZED,          \
                                             &pargc, &pargv, &parsec, &rank, &nodes);   \
    PARSEC_CHECK_ERROR(_init_rc, "parsec_tests_context_init");                           \
    if(do_sleep) sleep(10);                                                              \
    (void)name;


#define DO_INI_DATATYPES()                                               \
    parsec_arena_datatype_t adt_default;                                 \
    parsec_arena_datatype_t adt_lower;                                   \
    parsec_arena_datatype_t adt_upper;                                   \
    PARSEC_OBJ_CONSTRUCT(&adt_default, parsec_arena_datatype_t);         \
    PARSEC_OBJ_CONSTRUCT(&adt_lower, parsec_arena_datatype_t);           \
    PARSEC_OBJ_CONSTRUCT(&adt_upper, parsec_arena_datatype_t);           \
    parsec_matrix_adt_define_rect( &adt_default,                         \
                            parsec_datatype_int_t, MB, NB, MB );         \
    parsec_matrix_adt_define_lower( &adt_lower,                          \
                            parsec_datatype_int_t, 1, MB );              \
    parsec_matrix_adt_define_upper( &adt_upper,                          \
                            parsec_datatype_int_t, 1, MB );              \
    (void)adt_default; (void)adt_lower; (void)adt_upper;

#define DO_FINI_DATATYPES()                                              \
    parsec_matrix_arena_datatype_destruct_free_type(&adt_default);       \
    parsec_matrix_arena_datatype_destruct_free_type(&adt_lower);         \
    parsec_matrix_arena_datatype_destruct_free_type(&adt_upper);


#define DO_RUN(ctp) do {                                                 \
    tp = (parsec_taskpool_t*)ctp;                                        \
    parsec_context_add_taskpool(parsec, tp);                             \
    parsec_context_start(parsec);                                        \
    parsec_context_wait(parsec);                                         \
    parsec_taskpool_free((parsec_taskpool_t*)tp);                        \
} while(0)


#define DO_CHECK(NAME, dc, dc_check) do {                                \
    cret = check_matrix_equal(parsec, dc, dc_check );                    \
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
    cret = check_matrix_equal(parsec, dcA, dcA_check );                  \
    if(rank==0)                                                          \
        printf("Test " #NAME " %s\n", (cret > 0)? "FAILED" : "PASSED");  \
    ret |= cret;                                                         \
    BARRIER;                                                             \
} while(0)

#endif
