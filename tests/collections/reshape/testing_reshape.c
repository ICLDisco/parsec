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

#include "local_no_reshape.h"
#include "local_read_reshape.h"
#include "local_output_reshape.h"
#include "local_input_reshape.h"
#include "remote_read_reshape.h"
#include "remote_no_re_reshape.h"
#include "local_input_LU_LL.h"

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
    int cores = -1;

    DO_INIT();

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


    /*******************
     * No local reshape
     * When only type_remote is used on the dependencies, the pointer to the origianl
     * matrix tiles is passed to the succesors tasks. Thus, the full original tiles are
     * set to 0.
     *******************/
    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 1;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, op_args);

    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, op_args);

    {
      parsec_local_no_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_local_no_reshape_new((parsec_tiled_matrix_t *)&dcA );

      ctp->arenas_datatypes[PARSEC_local_no_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_local_no_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(local_no_reshape, dcA, dcA_check);
    }

    /*************************
     * Local reshape on read
     * When using [type] when reading from the matrix, a new datacopy with the correct
     * shape is created and passed to successors.
     * Thus, only the lower part of original tiles is set to 0.
     *************************/
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

    {
      parsec_local_read_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_local_read_reshape_new((parsec_tiled_matrix_t *)&dcA);

      ctp->arenas_datatypes[PARSEC_local_read_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_local_read_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(local_read_reshape, dcA, dcA_check);
    }

    /************************
     * Local reshape on output
     * When using [type] on an output dependency, a new datacopy with the correct
     * shape is created and passed to successors.
     * Thus, only the lower part of original tiles is set to 0.
     ************************/
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

    {
      parsec_local_output_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_local_output_reshape_new((parsec_tiled_matrix_t *)&dcA);

      ctp->arenas_datatypes[PARSEC_local_output_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_local_output_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(local_output_reshape, dcA, dcA_check);
    }

    /*************************
     * Local reshape on input
     * When using [type] on an input dependency, a new datacopy with the correct
     * shape is created and passed to successors.
     * Thus, only the lower part of original tiles is set to 0.
     ************************/

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

    {
      parsec_local_input_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_local_input_reshape_new((parsec_tiled_matrix_t *)&dcA);

      ctp->arenas_datatypes[PARSEC_local_input_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_local_input_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(local_input_reshape, dcA, dcA_check);
    }

    /*************************
     * Remote reshape on read
     * When using [type] when reading from the matrix, a new datacopy with the correct
     * shape is created and passed to successors.
     * Thus, only the lower part of original tiles is set to 0.
     *************************/
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

    {
      parsec_remote_read_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_remote_read_reshape_new((parsec_tiled_matrix_t *)&dcA);

      ctp->arenas_datatypes[PARSEC_remote_read_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_remote_read_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(remote_read_reshape, dcA, dcA_check);
    }

    /*******************************
     * Remote reshape no re-reshape
     * Reshape during output before packing.
     *******************************/
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

    {
      parsec_remote_no_re_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_remote_no_re_reshape_new((parsec_tiled_matrix_t *)&dcA, cores );

      ctp->arenas_datatypes[PARSEC_remote_no_re_reshape_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_remote_no_re_reshape_LOWER_TILE_ADT_IDX] = adt_lower;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);

      DO_RUN(ctp);
      DO_CHECK(remote_no_re_reshape, dcA, dcA_check);
    }



    /*******************************
     * Local reshape 2 different types LU -> LL
     * Get tile lower and transform into upper
     *******************************/

    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, op_args);

    op_args = (int *)malloc(sizeof(int)*2);
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count_lower2upper_matrix, op_args);
    {
      parsec_local_input_LU_LL_taskpool_t *ctp = NULL;
      ctp = parsec_local_input_LU_LL_new((parsec_tiled_matrix_t *)&dcA);

      ctp->arenas_datatypes[PARSEC_local_input_LU_LL_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_local_input_LU_LL_LOWER_TILE_ADT_IDX] = adt_lower;
      ctp->arenas_datatypes[PARSEC_local_input_LU_LL_UPPER_TILE_ADT_IDX] = adt_upper;

      DO_RUN(ctp);
      DO_CHECK(local_input_LU_LL, dcA, dcA_check);
    }

    // name=strdup("OUT dcA");
    // parsec_apply( parsec, PARSEC_MATRIX_FULL,
    //               (parsec_tiled_matrix_t *)&dcA,
    //               (parsec_tiled_matrix_unary_op_t)reshape_print, name);

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
