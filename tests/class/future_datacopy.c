/*
 * Copyright (c) 2018-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdarg.h>
#include "parsec/parsec_config.h"
#include "parsec/bindthread.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"
#include "parsec/utils/mca_param.h"

#include "parsec/class/parsec_future.h"

/* Test for future_datacopy
 * t1) Same tracked data on all threads.
 * t2) Nested future.
 */

parsec_datacopy_future_t **fut_array;
int *data;
int ***data_check_out;
int cores = 4;
int ncopy = 10;


void cb_fulfill(parsec_base_future_t * future){
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    int * data_in = ((int*)d_fut->cb_fulfill_data_in);
    int * data = (int*)malloc(sizeof(int));
    *data = *data_in;
    parsec_future_set(future, data);
}   

int cb_match(parsec_base_future_t * future, void * t1, void * t2){
    (void)future;
    return (((int*)t1) == ((int*)t2));
}

void cb_cleanup(parsec_base_future_t * future){
    parsec_datacopy_future_t* d_fut = (parsec_datacopy_future_t*)future;
    int * data = (int*)d_fut->super.tracked_data;
    int * specs = (int*)d_fut->cb_match_data_in;
    free(data);
    free(specs);
}

void cb_nested(parsec_base_future_t ** future, void * tracked_data, void * data_in){
    parsec_datacopy_future_t** d_fut = (parsec_datacopy_future_t**)future;
    int * data = (int*)tracked_data;
    int * specs = (int*)data_in;
    *specs += *data; 
    *d_fut = PARSEC_OBJ_NEW(parsec_datacopy_future_t);
    parsec_future_init( *d_fut, 
                        cb_fulfill, 
                        specs,
                        cb_match,
                        specs,
                        cb_cleanup); 
}


static void *do_test_no_nested(void* _param){
    int *param = (int*)_param;
    int id = param[0]; //thread id

    for(int i = 0; i < ncopy; i++){
        while( (data_check_out[i][id] = 
                    parsec_future_get_or_trigger(fut_array[i],
                                                 NULL, NULL, /* nested data */
                                                 NULL, NULL /*callback not using es, tp, task */
                                                 ) ) == NULL ){}
    }
    return NULL;
}

static void *do_test_nested(void* _param){
    int *param = (int*)_param;
    int id = param[0]; //thread id
    int *specs; 
    if(id % 2 == 0) {
        for(int i = 0; i < ncopy; i++){
            while( (data_check_out[i][id] = 
                        parsec_future_get_or_trigger(fut_array[i],
                                                     NULL, NULL, /* nested data */
                                                     NULL, NULL /*callback not using es, tp, task */
                                                     ) ) == NULL ){}
        }
    }else{
        for(int i = 0; i < ncopy; i++){
            specs = (int*)malloc(sizeof(int));
            *specs = ncopy;
            while( (data_check_out[i][id] = 
                        parsec_future_get_or_trigger(fut_array[i],
                                                     cb_nested, specs, /* nested data */
                                                     NULL, NULL /*callback not using es, tp, task */
                                                     ) ) == NULL ){}
        }
    }   
    return NULL;
}

static void usage(const char *name, const char *msg)
{
    if( NULL != msg ) {
        fprintf(stderr, "%s\n", msg);
    }
    fprintf(stderr,
            "Usage: \n"
            "   %s [-c cores|-n ncopy|-h]\n"
            " where\n"
            "   -c cores:   cores (integer >0) defines the number of cores to test, even number (default %d) \n"
            "   -n ncopy:   ncopy (integer >0) defines the number of copies of test (default %u)\n",
            name,
            cores,
            ncopy);
    exit(1);
}

int main(int argc, char* argv[])
{  
    int i, j, ch;
    void * retval;
    int flag = 0;
    char *m;
    pthread_t * threads;
    int * data_in;

    /* Read in the parameters */ 
    while( (ch = getopt(argc, argv, "c:n:h")) != -1 ) {
        switch(ch) {
            case 'c':
                cores = strtol(optarg, &m, 0);
                if( (cores <= 1) || (cores %2 !=  0 ) || (m[0] != '\0') ) {
                    usage(argv[0], "invalid -c value, must be positive and even number ");
                }
                break;
            case 'n':
                ncopy = strtol(optarg, &m, 0);
                if( (ncopy <= 0) || (m[0] != '\0') ) {
                    usage(argv[0], "invalid -n value");
                }
                break;
            case 'h':
            default:
                usage(argv[0], NULL);
                break;
        }
    }
    
    printf("running with %d cores and %d copies\n", cores, ncopy);
    threads = calloc(sizeof(pthread_t), cores);

    fut_array = malloc(ncopy*sizeof(parsec_datacopy_future_t*));
    data = malloc(cores*ncopy*sizeof(int));
    int *ids = malloc(cores*sizeof(int));
    

    data_check_out = malloc(ncopy*sizeof(int**));
    for(i=0; i< ncopy; i++){
        data_check_out[i] = malloc(cores*sizeof(int*));
        data[i] = i;
        data_in = malloc(cores*sizeof(int*));
        *data_in = data[i];
        fut_array[i] = PARSEC_OBJ_NEW(parsec_datacopy_future_t);
        parsec_future_init( fut_array[i], 
                            cb_fulfill, 
                            data_in,
                            cb_match,
                            data_in,
                            cb_cleanup); 
    }

    for(i=0; i< cores; i++){
        ids[i] = i;
        pthread_create(&threads[i], NULL, do_test_no_nested, &ids[i]);
    }
 
    for(i=0; i< cores; i++){
        flag += pthread_join(threads[i], &retval);
    }

    flag = 0;
    for(i=0; i< ncopy; i++){
        for(j=1; j< cores; j++){
            if( data_check_out[i][j-1] != data_check_out[i][j] ){
                flag = 1; 
                break;
            }
        }
        PARSEC_OBJ_RELEASE(fut_array[i]);
    }

    if(flag == 0) {
        printf("No-nested parsec_datacopy_future validated\n");
    }

    for(i=0; i< ncopy; i++){
        data[i] = i;
        data_in = malloc(cores*sizeof(int*));
        *data_in = data[i];
        fut_array[i] = PARSEC_OBJ_NEW(parsec_datacopy_future_t);
        parsec_future_init( fut_array[i], 
                            cb_fulfill, 
                            data_in,
                            cb_match,
                            data_in,
                            cb_cleanup); 
    }

    for(i=0; i< cores; i++){
        ids[i] = i;
        pthread_create(&threads[i], NULL, do_test_nested, &ids[i]);
    }
 
    for(i=0; i< cores; i++){
        flag += pthread_join(threads[i], &retval);
    }

    flag = 0;
    for(i=0; i< ncopy; i++){
        for(j=2; j< cores; j++){
            if(j % 2 == 0) {
                if( data_check_out[i][j-2] != data_check_out[i][j] ){
                    flag = 1; 
                    break;
                }
            }else{
                if( (data_check_out[i][j-2] == data_check_out[i][j])
                    || (*data_check_out[i][j-2] != *data_check_out[i][j]) 
                    || (*data_check_out[i][j-2] != (data[i]+ncopy)) ){
                    flag = 1; 
                    break;
                }
            }
        }
        PARSEC_OBJ_RELEASE(fut_array[i]);
    }

    if(flag == 0) {
        printf("Nested parsec_datacopy_future validated\n");
    }
    free(threads);    
    free(ids);
    free(data);
    free(fut_array);
    for(i=0; i< ncopy; i++){
        free(data_check_out[i]);
    }
    free(data_check_out);
}
