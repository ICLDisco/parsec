/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "common.h"

int reshape_set_matrix_value(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args)
{
    int *A = (int *)_A;
    int value = ((int *)args)[0];

    for(int i = 0; i < descA->mb; i++){
        for(int j = 0; j < descA->nb; j++){
            A[j*descA->mb + i] = value;
        }
    }
    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}

int reshape_set_matrix_value_count(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args)
{
    int *A = (int *)_A;
    int count=1;
    for(int j = 0; j < descA->nb; j++){
        for(int i = 0; i < descA->mb; i++){
            A[j*descA->mb + i] = count +10*(m);
            count++;
        }
    }
    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}

int reshape_set_matrix_value_count_lower2upper_matrix(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args)
{
    int *A = (int *)_A;
    int count=1;
    for(int j = 0; j < descA->nb; j++){
        for(int i = 0; i < descA->mb; i++){
            A[j*descA->mb + i] = count +10*(m);
            count++;
        }
    }

    count=1;
    int auxcount = 0;
    int increase = 0;
    for(int j = 0; j < descA->nb; j++){
        for(int i = 0; i < descA->mb; i++){
            if( j >= i ){
                A[j*descA->mb + i] = count +10*(m);
                count++;
                auxcount++;
                if( (auxcount+increase) == descA->mb ){
                    auxcount = 0;
                    count+=1+increase;
                    increase++;
                }

            }
        }
    }
    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}


int reshape_set_matrix_value_lower_tile(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args)
{
    int *A = (int *)_A;
    int value_upper = ((int *)args)[0];
    int value_lower = ((int *)args)[1];

    for(int i = 0; i < descA->mb; i++){
        for(int j = 0; j < descA->nb; j++){
            if( j <= i){
                A[j*descA->mb + i] = value_lower;
            }else{
                A[j*descA->mb + i] = value_upper;
            }
        }
    }

    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}

int reshape_set_matrix_value_position(parsec_execution_stream_t *es,
                                      const parsec_tiled_matrix_t *descA,
                                      void *_A, parsec_matrix_uplo_t uplo,
                                      int m, int n, void *args)
{
    int *A = (int *)_A;
    int value = 0;
    for(int i = 0; i < descA->mb; i++){
        for(int j = 0; j < descA->nb; j++){
            A[j*descA->mb + i] = value;
            value++;
        }
    }
    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}

int reshape_set_matrix_value_position_swap(parsec_execution_stream_t *es,
                                      const parsec_tiled_matrix_t *descA,
                                      void *_A, parsec_matrix_uplo_t uplo,
                                      int m, int n, void *args)
{
    int *A = (int *)_A;
    int value = 0;
    //int all = ((int *)args)[0];

    int type = 0;
    if(n%2 == 0){//lower
        type = 0;
    }else{//upper
        type = 1;
    }
    for(int i = 0; i < descA->mb; i++){
        for(int j = 0; j < descA->nb; j++){

            // if(all == 0){
            //     if(m==n){
            //         A[j*descA->mb + i] = value;
            //     }
            // }
            if(type==0){//lower
                if( j <= i){
                    A[j*descA->mb + i] = value;
                }
            }else{//upper
                if( j >= i){
                    A[j*descA->mb + i] = value;
                }
            }
            value++;
        }
    }
    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}

int check_matrix_equal(parsec_matrix_block_cyclic_t dcA, parsec_matrix_block_cyclic_t dcA_check){
    int ret = 0;
    for(int i=0; i < dcA_check.super.nb_local_tiles * dcA_check.super.bsiz; i++){
        if( ((int*)dcA.mat)[i] != ((int*)dcA_check.mat)[i]){
            ret = 1;
            break;
        }
    }
#if defined(PARSEC_HAVE_MPI)
    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    return ret;
}


int reshape_print(parsec_execution_stream_t *es,
                  const parsec_tiled_matrix_t *descA,
                  void *_A, parsec_matrix_uplo_t uplo,
                  int m, int n, void *args)
{
    int *A = (int *)_A;
    char *name = (char*)args;
    printf("%2d %10s (%2d, %2d) ", getpid(), name, m, n);
    for(int i = 0; i<descA->mb; i++){
        for(int j = 0; j<descA->nb; j++){
            printf("%3d ", A[j * descA->mb + i]);
        }
        printf("\n\t\t\t  ");
    }
    printf("\n");

    (void)es; (void)uplo; (void)m; (void)n; (void) args;
    return 0;
}
