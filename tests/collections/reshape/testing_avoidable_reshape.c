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

#include "avoidable_reshape.h"

/* Program to test the different reshaping functionalities
 * Logical equal types on DEFAULT_ADT and data collection result
 * on unnecessary reshapes. Solution: reuse datatype for DEFAULT_ADT.
 */
// #define AVOID_UNNECESSARY_RESHAPING

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

    assert(nodes == 1);

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
     * Doing avoidable reshape becasue dc datatype differs from default ADT.
     *******************/
    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, op_args);

    op_args = (int *)malloc(sizeof(int));
    op_args[0] = 0;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA_check,
                  (parsec_tiled_matrix_unary_op_t)reshape_set_matrix_value_count, op_args);

    {
      parsec_avoidable_reshape_taskpool_t *ctp = NULL;
      ctp = parsec_avoidable_reshape_new((parsec_tiled_matrix_t *)&dcA );

      ctp->arenas_datatypes[PARSEC_avoidable_reshape_DEFAULT_ADT_IDX]    = adt_default;

#ifdef AVOID_UNNECESSARY_RESHAPING
      /* Can be avoided by setting the datacollection type as the default adt*/
      parsec_datatype_t tmp = adt_default.opaque_dtt;
      ctp->arenas_datatypes[PARSEC_avoidable_reshape_DEFAULT_ADT_IDX].opaque_dtt = dcA.super.super.default_dtt;
#endif
      PARSEC_OBJ_RETAIN(adt_default.arena);

      DO_RUN(ctp);
      DO_CHECK(avoidable_reshape, dcA, dcA_check);

#ifdef AVOID_UNNECESSARY_RESHAPING
      adt_default.opaque_dtt = tmp;
#endif

    }


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
