/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic_band.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic_band.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include "two_dim_band_test.h"
#include <string.h>

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch;
    int pargc = 0, i;
    char **pargv = NULL;
    parsec_matrix_uplo_t uplo = PARSEC_MATRIX_UPPER; //PARSEC_MATRIX_LOWER
    parsec_matrix_uplo_t full = PARSEC_MATRIX_FULL;
    int rc;
    /* Super */
    int N = 16, NB = 4, P = 1, KP = 1, KQ = 1;
    /* Band */
    int P_BAND = 1, KP_BAND = 1, KQ_BAND = 1, BAND_SIZE = 1;

    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }

    /* Initialize PaRSEC */
    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_MULTIPLE,
                                   &pargc, &pargv,
                                   &parsec, &rank, &nodes);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    while ((ch = getopt(argc, argv, "N:T:s:S:P:p:f:F:b:h")) != -1) {
        switch (ch) {
            case 'N': N = atoi(optarg); break;
            case 'T': NB = atoi(optarg); break;
            case 's': KP = atoi(optarg); break;
            case 'S': KQ = atoi(optarg); break;
            case 'P': P = atoi(optarg); break;
            case 'p': P_BAND = atoi(optarg); break;
            case 'f': KP_BAND = atoi(optarg); break;
            case 'F': KQ_BAND = atoi(optarg); break;
            case 'b': BAND_SIZE = atoi(optarg); break;
            case '?': case 'h': default:
                fprintf(stderr,
                        "SUPER:\n"
                        "-N : dimension (N) of the matrices (default: 16)\n"
                        "-T : dimension (NB) of the tiles (default: 4)\n"
                        "-s : rows of tiles in a k-cyclic distribution (default: 1)\n"
                        "-S : columns of tiles in a k-cyclic distribution (default: 1)\n"
                        "-P : rows (P) in the PxQ process grid (default: 1)\n"
                        "BAND:\n"
                        "-p : rows (p) in the pxq process grid (default: 1)\n"
                        "-f : rows of tiles in a k-cyclic distribution (default: 1)\n"
                        "-F : columns of tiles in a k-cyclic distribution (default: 1)\n"
                        "-b : band size (default: 1)\n"
                        "\n");
            exit(1);
        }
    }

    /* dcY initializing matrix structure */
    /* Init Off_band */
    parsec_matrix_block_cyclic_band_t dcY;
    parsec_matrix_block_cyclic_init(&dcY.off_band, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                rank, NB, NB, N, N, 0, 0,
                                N, N,
                                P, nodes/P, KP, KQ, 0, 0);
    /* Init band */
    parsec_matrix_block_cyclic_init(&dcY.band, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                rank, NB, NB, NB*(2*BAND_SIZE-1), N, 0, 0,
                                NB*(2*BAND_SIZE-1), N,
                                P_BAND, nodes/P_BAND, KP_BAND, KQ_BAND, 0, 0);
    /* Init parsec_matrix_block_cyclic_band_t structure */
    parsec_matrix_block_cyclic_band_init( &dcY, nodes, rank, BAND_SIZE );
    /* set key needs dcY to be initialized already */
    parsec_data_collection_set_key(&dcY.off_band.super.super, "dcY off_band");
    parsec_data_collection_set_key(&dcY.band.super.super, "dcY band");

    /* YP */
    parsec_matrix_sym_block_cyclic_band_t dcYP;
    /* Init Off_band */
    parsec_matrix_sym_block_cyclic_init(&dcYP.off_band, PARSEC_MATRIX_DOUBLE,
                                rank, NB, NB, N, N, 0, 0,
                                N, N, P, nodes/P, uplo);
    /* Init band */
    parsec_matrix_block_cyclic_init(&dcYP.band, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                rank, NB, NB, NB*BAND_SIZE, N, 0, 0,
                                NB*BAND_SIZE, N,
                                P_BAND, nodes/P_BAND, KP_BAND, KQ_BAND, 0, 0);
    /* Init parsec_matrix_block_cyclic_band_t structure */
    parsec_matrix_sym_block_cyclic_band_init( &dcYP, nodes, rank, BAND_SIZE );
    /* set key needs dcYP to be initialized already */
    parsec_data_collection_set_key(&dcYP.off_band.super.super, "dcYP off_band");
    parsec_data_collection_set_key(&dcYP.band.super.super, "dcYP band");

    /* Allocate memory and set value */
    parsec_two_dim_band_test(parsec, (parsec_tiled_matrix_t *)&dcY, full);

    if( 0 == rank )
        printf("Y  Init \tSUPER: PxQ= %3d %-3d, KPxKQ=%3d %-3d, N= %7d, NB= %4d; BAND: PxQ= %3d %-3d KPxKQ=%3d %-3d, BAND_SIZE=%3d, M= %7d, N= %4d\n",
               P, nodes/P, KP, KQ, N, NB,
               P_BAND, nodes/P_BAND, KP_BAND, KQ_BAND, BAND_SIZE, NB*(2*BAND_SIZE-1), N);


    /* Allocate memory and set value */
    parsec_two_dim_band_test(parsec, (parsec_tiled_matrix_t *)&dcYP, uplo);

    if( 0 == rank )
        printf("YP Init \tSUPER: PxQ= %3d %-3d, KPxKQ=%3d %-3d, N= %7d, NB= %4d; BAND: PxQ= %3d %-3d KPxKQ=%3d %-3d, BAND_SIZE=%3d, M= %7d, N= %4d\n",
               P, nodes/P, KP, KQ, N, NB,
               P_BAND, nodes/P_BAND, KP_BAND, KQ_BAND, BAND_SIZE, NB*BAND_SIZE, N);

    /* Free memory */
    parsec_two_dim_band_free(parsec, (parsec_tiled_matrix_t *)&dcY, full);
    parsec_two_dim_band_free(parsec, (parsec_tiled_matrix_t *)&dcYP, uplo);
    parsec_tiled_matrix_destroy(&dcY.super);
    parsec_tiled_matrix_destroy(&dcY.band.super);
    parsec_tiled_matrix_destroy(&dcY.off_band.super);
    parsec_tiled_matrix_destroy(&dcYP.super);
    parsec_tiled_matrix_destroy(&dcYP.band.super);
    parsec_tiled_matrix_destroy(&dcYP.off_band.super);

    /* Clean up parsec*/
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
