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

#include "remote_multiple_outs_same_pred_flow.h"
#include "remote_multiple_outs_same_pred_flow_multiple_deps.h"

/* Program to test one single output flow producing multiple output
 * datacopies that are received by the different successors tasks on the
 * same process. Thus, checking workaround to avoid the fake
 * predecessor repo not being overwrite during reception.
 *
 * Note this test doesn't work with runtime_comm_short_limit != 0.
 * This test sends two ouput flows on a task with different shapes,
 * and received then on a remote successor with two different shapes.
 * Currently, PaRSEC doesn't support this scenario using SHORT.
 * In this case, two datas are included on the activation message, and
 * after reception on the receiver, the predecessor task is faked and
 * iterate_sucessors of the predecessor task is run only ONCE, therefore,
 * for one sucessors the flow will contain incorrect data.
 *
 * When SHORT is not used, PaRSEC runs iterate_sucessors for each data received,
 * thus, always the correct data is used.
 * This gives the chance to the reshaping mechanism to put the first data on
 * the predecessor's repo, and any subsequent data on the sucessors repo, avoiding
 * any overwrites.
 */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch;
    int ret = 0, cret;
    int *op_args;
    parsec_matrix_block_cyclic_t dcA;
    parsec_matrix_block_cyclic_t dcA_check;
    parsec_matrix_block_cyclic_t dcV;
    parsec_taskpool_t * tp;
    /* Default */
    int m = 0;
    int M = 6;
    int N = 6;
    int MB = 2;
    int NB = 2;
    int P = 1;
    int KP = 1;
    int KQ = 1;
    int cores = 1;

    DO_INIT();

    DO_INI_DATATYPES();

    /* Matrix allocation */
    //P = nodes;
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

    //P = 1;
    parsec_matrix_block_cyclic_init(&dcV, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                              rank, MB, NB, M, NB, 0, 0,
                              M, NB, P, nodes/P, KP, KQ, 0, 0);
    dcV.mat = parsec_data_allocate((size_t)dcV.super.nb_local_tiles *
                                   (size_t)dcV.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcV.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcV, "dcV");


    /*************************
     * Multiple outs by flow & multiple deps
     *************************/

    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, op_args);

    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcV,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, NULL);

    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, NULL);

    {
      parsec_remote_multiple_outs_same_pred_flow_taskpool_t *ctp = NULL;
      ctp = parsec_remote_multiple_outs_same_pred_flow_new((parsec_tiled_matrix_t *)&dcA,
            (parsec_tiled_matrix_t *)&dcV );

      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_LOWER_TILE_ADT_IDX] = adt_lower;
      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_UPPER_TILE_ADT_IDX] = adt_upper;
      PARSEC_OBJ_RETAIN(adt_default.arena);
      PARSEC_OBJ_RETAIN(adt_lower.arena);
      PARSEC_OBJ_RETAIN(adt_upper.arena);

      DO_RUN(ctp);
      DO_CHECK(remote_multiple_outs_same_pred_flow, dcA, dcA_check);

    }
    /*************************
     * Multiple outs by flow & multiple deps
     *************************/

    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value, op_args);

    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcV,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, NULL);

    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, NULL);

    {
      parsec_remote_multiple_outs_same_pred_flow_multiple_deps_taskpool_t *ctp = NULL;
      ctp = parsec_remote_multiple_outs_same_pred_flow_multiple_deps_new((parsec_tiled_matrix_t *)&dcA,
            (parsec_tiled_matrix_t *)&dcV );

      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_multiple_deps_DEFAULT_ADT_IDX]    = adt_default;
      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_multiple_deps_LOWER_TILE_ADT_IDX] = adt_lower;
      ctp->arenas_datatypes[PARSEC_remote_multiple_outs_same_pred_flow_multiple_deps_UPPER_TILE_ADT_IDX] = adt_upper;

      DO_RUN(ctp);
      DO_CHECK(remote_multiple_outs_same_pred_flow_multiple_deps, dcA, dcA_check);
    }

    // name=strdup("dcA");
    // parsec_apply( parsec, PARSEC_MATRIX_FULL,
    //               (parsec_tiled_matrix_t *)&dcA,
    //               (parsec_tiled_matrix_unary_op_t)reshape_print, name);

    // name=strdup("dcA_check");
    // parsec_apply( parsec, PARSEC_MATRIX_FULL,
    //               (parsec_tiled_matrix_t *)&dcA_check,
    //               (parsec_tiled_matrix_unary_op_t)reshape_print, name);

    /* Clean up */
    DO_FINI_DATATYPES();

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);
    parsec_data_free(dcA_check.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA_check);

    parsec_data_free(dcV.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcV);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
