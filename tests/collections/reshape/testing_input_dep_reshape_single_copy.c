/*
 * Copyright (c) 2017-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <string.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "common.h"

#include "input_dep_single_copy_reshape.h"

/* Program to test the different reshaping functionalities
 * Each different test is commented on the main program.
 */

/* TASK_A pairs up t=0 with t=1 through a condition variable, so the test
 * cannot run on fewer worker threads than this. */
#define NBTHREADS_NEEDED 2

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch;
    int ret = 0, cret;
    int op_args = 1, op_args2[2] = {1, 0};
    parsec_matrix_block_cyclic_t dcA;
    parsec_matrix_block_cyclic_t dcA_check;
    parsec_taskpool_t * tp;

    /* Default */
    int m = 0;
    int M = 8;
    int N = 8;
    int MB = 4;
    int NB = 4;
    int P = 1;
    int KP = 1;
    int KQ = 1;
    int cores = NBTHREADS_NEEDED;

    DO_INIT();

    /* TASK_A(m,k,0) and TASK_A(m,k,1) rendezvous through a condition
     * variable, so both have to be resident at once or the first one waits
     * for a signal that can never come. Ask the context what it actually
     * gave us rather than trusting what we requested: anything that caps the
     * thread count turns this test into a deadlock instead of a failure. */
    int available = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES);
    if( cores < NBTHREADS_NEEDED || available < NBTHREADS_NEEDED ) {
        fprintf(stderr,
                "%s needs %d worker threads, was configured for %d and "
                "PaRSEC provided %d. Check the -c flag and the "
                "`runtime_num_cores` parameter.\n",
                argv[0], NBTHREADS_NEEDED, cores, available);
        return 77;  /* ctest: skipped, rather than a deadlock */
    }

    DO_INI_DATATYPES();

    /* Matrix allocation */
    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                              rank, MB, NB, M, N, 0, 0,
                              M, N, P, nodes/P, KP, KQ, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "dcA");

    parsec_matrix_block_cyclic_init(&dcA_check, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                              rank, MB, NB, M, N, 0, 0,
                              M, N, P, nodes/P, KP, KQ, 0, 0);
    dcA_check.mat = parsec_data_allocate((size_t)dcA_check.super.nb_local_tiles *
                                   (size_t)dcA_check.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA_check.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA_check, "dcA_check");


    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, &op_args);

    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_lower_tile, op_args2);

    parsec_input_dep_single_copy_reshape_taskpool_t *ctp = NULL;
    ctp = parsec_input_dep_single_copy_reshape_new((parsec_tiled_matrix_t *)&dcA, cores );

    ctp->arenas_datatypes[PARSEC_input_dep_single_copy_reshape_DEFAULT_ADT_IDX]    = adt_default;
    ctp->arenas_datatypes[PARSEC_input_dep_single_copy_reshape_LOWER_TILE_ADT_IDX] = adt_lower;

    DO_RUN(ctp);
    DO_CHECK(input_dep_single_copy_reshape, dcA, dcA_check);

    /* Clean up */
    DO_FINI_DATATYPES();

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);

    parsec_data_free(dcA_check.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA_check);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
