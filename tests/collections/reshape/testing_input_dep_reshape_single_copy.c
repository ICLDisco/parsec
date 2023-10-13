/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <string.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "common.h"

#include "input_dep_single_copy_reshape.h"

/* Program to test the different reshaping functionalities
 * Each different test is comented on the main program.
 */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch;
    int ret = 0, cret;
    int *op_args;
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
    int cores = 2;

    DO_INIT();

    assert(cores == 2);

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


    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 1;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, op_args);

    op_args = (int *)malloc(sizeof(int)*2);
    op_args[0] = 1;
    op_args[1] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_lower_tile, op_args);

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
