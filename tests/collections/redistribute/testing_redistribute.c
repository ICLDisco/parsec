/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "redistribute_test.h"
#include "common.h"

/**
 * @brief Test example of redistribute
 *
 * @detail
 * parsec_redistribute: PTG, redistribute from ANY distribution
 * to ANY distribution, with ANY displacement
 *
 * parsec_redistribute_dtd: DTD, redistribute from ANY distribution
 * to ANY distribution, with ANY displacement
 *
 * parsec_redistribute_check: check the result correctness of
 * two submatrix, if correct, print "Redistribute Result is CORRECT";
 * otherwise print the first detected location and values where values
 * are different.
 *
 * parsec_redistribute_init: init dcY to 0 or numbers based on index
 *
 * @exemple testing_redistribute.c
 */
int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    double dparam[IPARAM_SIZEOF];
    int MMB, NNB, MMBR, NNBR;
    double time_ptg = 0.0, time_dtd = 0.0;

    /* Source */
    iparam[IPARAM_P] = 1;
    iparam[IPARAM_Q] = 1;
    iparam[IPARAM_M] = 4;
    iparam[IPARAM_N] = 4;
    iparam[IPARAM_MB] = 4;
    iparam[IPARAM_NB] = 4;
    iparam[IPARAM_DISI] = 0;
    iparam[IPARAM_DISJ] = 0;

    /* Target/redistribute */
    iparam[IPARAM_P_R] = 1;
    iparam[IPARAM_Q_R] = 1;
    iparam[IPARAM_M_R] = 4;
    iparam[IPARAM_N_R] = 4;
    iparam[IPARAM_MB_R] = 4;
    iparam[IPARAM_NB_R] = 4;
    iparam[IPARAM_DISI_R] = 0;
    iparam[IPARAM_DISJ_R] = 0;

    /* Matrix common */
    iparam[IPARAM_RADIUS] = 0;
    iparam[IPARAM_M_SUB] = 4;
    iparam[IPARAM_N_SUB] = 4;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam, dparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];

    /* Source */
    int P       = iparam[IPARAM_P];
    int Q       = iparam[IPARAM_Q];
    int M       = iparam[IPARAM_M];
    int N       = iparam[IPARAM_N];
    int MB      = iparam[IPARAM_MB];
    int NB      = iparam[IPARAM_NB];
    int SMB     = iparam[IPARAM_SMB];
    int SNB     = iparam[IPARAM_SNB];
    int disi_Y  = iparam[IPARAM_DISI];
    int disj_Y  = iparam[IPARAM_DISJ];

    /* Target/redistribute */
    int PR       = iparam[IPARAM_P_R];
    int QR       = iparam[IPARAM_Q_R];
    int MR       = iparam[IPARAM_M_R];
    int NR       = iparam[IPARAM_N_R];
    int MBR      = iparam[IPARAM_MB_R];
    int NBR      = iparam[IPARAM_NB_R];
    int SMBR     = iparam[IPARAM_SMB_R];
    int SNBR     = iparam[IPARAM_SNB_R];
    int disi_T  = iparam[IPARAM_DISI_R];
    int disj_T  = iparam[IPARAM_DISJ_R];

    /* Matrix common */
    int R = 0;  // R = 0 by default
    int size_row = iparam[IPARAM_M_SUB];
    int size_col = iparam[IPARAM_N_SUB];

    /* Others */
    int check = iparam[IPARAM_CHECK];
    int time = iparam[IPARAM_GETTIME];
    double network_bandwidth  = dparam[DPARAM_NETWORK_BANDWIDTH];
    double memcpy_bandwidth  = dparam[DPARAM_MEMCPY_BANDWIDTH];
    int num_runs = iparam[IPARAM_NUM_RUNS];
    int thread_type = iparam[IPARAM_THREAD_MULTIPLE];
    int no_optimization_version = iparam[IPARAM_NO_OPTIMIZATION_VERSION];

    /* Used for ghost region */
    MMB = (int)(ceil((double)M/MB));
    NNB = (int)(ceil((double)N/NB));
    MMBR = (int)(ceil((double)MR/MBR));
    NNBR = (int)(ceil((double)NR/NBR));

    /* Allocate memory for results */
    double *results = (double *)calloc(8, sizeof(double));

    /* Initializing matrix structure */
    /* Initializing dcY */
    parsec_matrix_block_cyclic_t dcY;
    parsec_matrix_block_cyclic_init(&dcY, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                              rank, MB+2*R, NB+2*R, M+2*R*MMB, N+2*R*NNB, 0, 0,
                              M+2*R*MMB, N+2*R*NNB, P, nodes/P, SMB, SNB, 0, 0 );
    dcY.mat = parsec_data_allocate((size_t)dcY.super.nb_local_tiles *
                                   (size_t)dcY.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcY.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcY, "dcY");

    /* Initializing dcT */
    parsec_matrix_block_cyclic_t dcT;
    parsec_matrix_block_cyclic_init(&dcT, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                              rank, MBR+2*R, NBR+2*R, MR+2*R*MMBR, NR+2*R*NNBR, 0, 0,
                              MR+2*R*MMBR, NR+2*R*NNBR, PR, nodes/PR, SMBR, SNBR, 0, 0 );
    dcT.mat = parsec_data_allocate((size_t)dcT.super.nb_local_tiles *
                                   (size_t)dcT.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcT.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcT, "dcT");

    for(int i = 0; i < num_runs; i++) {
#if RUN_PTG
         /*
         * Init dcY not including ghost region; if initvalue is 0,
         * init to 0; otherwise init to numbers based on index
         */
        int *op_args = (int *)malloc(sizeof(int));
        *op_args = 1;
        parsec_apply( parsec, PARSEC_MATRIX_FULL,
                      (parsec_tiled_matrix_t *)&dcY,
                      (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, op_args);

        /* Timer start */
        SYNC_TIME_START();

        /* Main part, call parsec_redistribute; double is default, which could be
         * changed in parsec/data_dist/matrix/redistribute/redistribute_internal.h
         */
        if( no_optimization_version )
            parsec_redistribute_no_optimization(parsec, (parsec_tiled_matrix_t *)&dcY,
                                                (parsec_tiled_matrix_t *)&dcT,
                                                size_row, size_col, disi_Y, disj_Y,
                                                disi_T, disj_T);
        else
            parsec_redistribute(parsec, (parsec_tiled_matrix_t *)&dcY,
                                (parsec_tiled_matrix_t *)&dcT,
                                size_row, size_col, disi_Y, disj_Y,
                                disi_T, disj_T);

        /* Timer end */
        if( time ) {
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_PTG\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
            time_ptg = sync_time_elapsed;
        }

        /* Check result */
        if( check ){
            if( 0 == rank )
                printf("Checking result for PTG:");

#if COPY_TO_1NODE
            parsec_redistribute_check(parsec, (parsec_tiled_matrix_t *)&dcY,
                                      (parsec_tiled_matrix_t *)&dcT,
                                      size_row, size_col, disi_Y, disj_Y,
                                      disi_T, disj_T);
#else
            /* Init dcY to 0 */
            int *op_args = (int *)malloc(sizeof(int));
            *op_args = 0;
            parsec_apply( parsec, PARSEC_MATRIX_FULL,
                          (parsec_tiled_matrix_t *)&dcY,
                          (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, op_args);

            /* Redistribute back from dcT to dcY */
            parsec_redistribute(parsec, (parsec_tiled_matrix_t *)&dcT,
                                (parsec_tiled_matrix_t *)&dcY,
                                size_row, size_col, disi_T, disj_T,
                                disi_Y, disj_Y);

            parsec_redistribute_check2(parsec, (parsec_tiled_matrix_t *)&dcY,
                                       size_row, size_col, disi_Y, disj_Y);
#endif /* COPY_TO_1NODE */
        }
#endif /* RUN_PTG */

#if RUN_DTD
        /*
         * Init dcT to 0.0 for DTD
         */
        int *op_args_dtd = (int *)malloc(sizeof(int));
        *op_args_dtd = 0;
        parsec_apply( parsec, PARSEC_MATRIX_FULL,
                      (parsec_tiled_matrix_t *)&dcT,
                      (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, op_args_dtd);

        /* Timer start */
        SYNC_TIME_START();

        /* Main part, call parsec_redistribute_dtd; double is default, which could be
         * changed in parsec/data_dist/matrix/redistribute/redistribute_internal.h
         */
        parsec_redistribute_dtd(parsec, (parsec_tiled_matrix_t *)&dcY,
                                (parsec_tiled_matrix_t *)&dcT,
                                size_row, size_col, disi_Y, disj_Y,
                                disi_T, disj_T);

        /* Timer end */
        if( time ) {
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_DTD\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
            time_dtd = sync_time_elapsed;
        }

        /* Check result */
        if( check ){
            if( 0 == rank )
                printf("Checking result for DTD:");

#if COPY_TO_1NODE
            parsec_redistribute_check(parsec, (parsec_tiled_matrix_t *)&dcY,
                                      (parsec_tiled_matrix_t *)&dcT,
                                      size_row, size_col, disi_Y, disj_Y,
                                      disi_T, disj_T);
#else
            /* Init dcY to 0 */
            int *op_args = (int *)malloc(sizeof(int));
            *op_args = 0;
            parsec_apply( parsec, PARSEC_MATRIX_FULL,
                          (parsec_tiled_matrix_t *)&dcY,
                          (parsec_tiled_matrix_unary_op_t)redistribute_init_ops, op_args);

            /* Redistribute back from dcT to dcY */
            parsec_redistribute_dtd(parsec, (parsec_tiled_matrix_t *)&dcT,
                                    (parsec_tiled_matrix_t *)&dcY,
                                    size_row, size_col, disi_T, disj_T,
                                    disi_Y, disj_Y);

            parsec_redistribute_check2(parsec, (parsec_tiled_matrix_t *)&dcY,
                                       size_row, size_col, disi_Y, disj_Y);
#endif /* COPY_TO_1NODE */
        }
#endif /* RUN_DTD */

        if( time ) {
            /* Timer start */
            SYNC_TIME_START();

            /* Call parsec_redistribute_bound to get time bound */
            results = parsec_redistribute_bound(parsec, (parsec_tiled_matrix_t *)&dcY,
                                                (parsec_tiled_matrix_t *)&dcT,
                                                size_row, size_col, disi_Y, disj_Y,
                                                disi_T, disj_T);

            /* Timer end */
#if PRINT_MORE
            SYNC_TIME_PRINT(rank, ("\"testing_redistribute_bound\""
                            "\tRedistributed Size: m= %d n= %d"
                            "\tSource: P= %d Q= %d M= %d N= %d MB= %d NB= %d I= %d J=%d SMB= %d SNB= %d"
                            "\tTarget: PR= %d QR= %d MR= %d NR= %d MBR= %d NBR= %d i= %d j=%d SMBR= %d SNBR= %d"
                            "\tCores: %d\n\n",
                            size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                            PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores));
#else
            SYNC_TIME_STOP();
#endif
        }

        /* Print info to draw figures */
        if( 0 == rank && time ) {
            double ratio_remote = results[7] / results[2];
            double input_bandwidth_mix = network_bandwidth && memcpy_bandwidth ? 
                    network_bandwidth * memcpy_bandwidth / ((ratio_remote + 1) * network_bandwidth + memcpy_bandwidth) / 1.0e9 : 0.0;
            double input_bandwidth_worst = network_bandwidth && memcpy_bandwidth ? 
                    network_bandwidth * memcpy_bandwidth / ((ratio_remote + 2) * network_bandwidth + memcpy_bandwidth) / 1.0e9 : 0.0;
#if PRINT_MORE
            printf("'Time_PTG', 'Time_DTD', 'm', 'n', 'P', 'Q', 'M', 'N', 'MB', 'NB', 'I', 'J', 'SMB', 'SNB', "
                    "'PR', 'QR', 'MR', 'NR', 'MBR', 'NBR', 'i', 'j', 'SMBR', 'SNBR', 'cores', 'nodes', "
                    "'ratio_remote', 'thread_multiple', 'no_optimization_version', "
                    "'Total_message_remote_bits', 'Total_message_local_bits', "
                    "'Max_send_or_receive_message_each_rank_bits', 'Max_local_message_each_rank_bits', "
                    "'Number_of_message_remote', 'Number_of_message_local', 'Max_remote_message_each_rank_bits', "
                    "'Max_local_related_remote', 'Input_network_bandwidth_Gbits', 'Input_memcpy_bandwidth_Gbits', "
                    "'Input_bandwidth_mix_Gbits', 'Input_bandwidth_worst_Gbits', "
                    "'Output_network_bandwidth_ptg_Gbits', 'Output_network_bandwidth_ptg_bidir_Gbits', "
                    "'Output_memcpy_bandwidth_ptg_Gbits', 'Output_network_bandwidth_dtd_Gbits', "
                    "'Output_network_bandwidth_dtd_bidir_Gbits', 'Output_memcpy_bandwidth_dtd_Gbits' "
                    "\n\n");
#endif
            printf("OUTPUT %lf %lf %d %d %d %d %d %d %d %d %d %d %d %d "
                   "%d %d %d %d %d %d %d %d %d %d %d %d %.2lf %d %d "
                   "%.10e %.10e %.10e %.10e %.2lf %.2lf %.10e %.10e "
                   "%.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %.2lf\n",
                   time_ptg, time_dtd, size_row, size_col, P, Q, M, N, MB, NB, disi_Y, disj_Y, SMB, SNB,
                   PR, QR, MR, NR, MBR, NBR, disi_T, disj_T, SMBR, SNBR, cores, nodes, ratio_remote, 
                   thread_type, no_optimization_version,
                   results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7],
                   network_bandwidth / 1.0e9, memcpy_bandwidth / 1.0e9,
                   input_bandwidth_mix, input_bandwidth_worst,
                   (time_ptg ? results[2] / 1.0e9 / time_ptg : 0.0),
                   (time_ptg ? results[6] / 1.0e9 / time_ptg : 0.0),
                   (time_ptg ? (results[2] + results[3]) / 1.0e9 / time_ptg : 0.0),
                   (time_dtd ? results[2] / 1.0e9 / time_dtd: 0.0),
                   (time_dtd ? results[6] / 1.0e9 / time_ptg : 0.0),
                   (time_dtd ? (results[2] + results[3]) / 1.0e9 / time_dtd : 0.0));
        }

    }

    /* Free memory */
    parsec_data_free(dcY.mat);
    parsec_data_free(dcT.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcY);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcT);

    cleanup_parsec(parsec, iparam, dparam);
    return 0;
}
