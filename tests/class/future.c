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

//Global variables
parsec_base_future_t **fut_array;
int *data;
parsec_countable_future_t *c_fut;
int cores = 4;
int ncopy = 100;

void test_cb(parsec_base_future_t* future){
    int* temp = (int*)parsec_future_get(future);
    printf("callback returning %d\n", *temp);
}


static void *do_test(void* _param){
    int * param = (int*)_param;
    int id = param[0]; //thread id
    int *res;
    int validate = 0;
    /* Even number's thread will set the future to be ready, 
     * Odd number thread will read the value set by previous thread
     */
    if(id % 2 == 0) {
        for(int i = 0; i < ncopy; i++)
            parsec_future_set(fut_array[id+i*cores], &data[id+i*cores]);
    } else {
        for(int i = 0; i < ncopy; i++) {
            res = parsec_future_get(fut_array[id+i*cores-1]);
            if(*res != data[id+i*cores-1]) {
                printf("thread %d index %d with value %d, expected %d \n", 
                        id, id+i*cores-1, *((int*)res), data[id+i*cores-1]);validate++;
            }
        }
        if(validate != 0 ) printf("Get validation failed\n");
    }
    return NULL;
}

static void *do_test2(void* _param){
    int * param = (int*)_param;
    int id = param[0]; /* thread id */
 
    data[id] += 1;
    parsec_future_set(fut_array[id], &data[id]);
    /* Decrement count from a thread */
    parsec_future_set(c_fut, param); /* param is ignored by the set function */
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
    int i, ch;
    void * retval;
    int flag = 0;
    char *m;
    pthread_t * threads;

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
    
    c_fut = PARSEC_OBJ_NEW(parsec_countable_future_t);
    parsec_future_init(c_fut, NULL, cores);
    printf("running with %d cores and %d copies\n", cores, ncopy);
    threads = calloc(sizeof(pthread_t), cores);

    fut_array = malloc(cores*ncopy*sizeof(parsec_base_future_t*));
    data = malloc(cores*ncopy*sizeof(int));
    int *ids = malloc(cores*sizeof(int));
    
    /* Test base future get  functionality */
    for(i=0; i< cores*ncopy; i++){
        data[i] = i;
        fut_array[i] = PARSEC_OBJ_NEW(parsec_base_future_t);
        /*fut_array[i]->future_class->future_init(fut_array[i], NULL); */
    }
    for(i=0; i< cores; i++){
        ids[i] = i;
        pthread_create(&threads[i], NULL, do_test, &ids[i]);
    }
 
    for(i=0; i< cores; i++){
        flag += pthread_join(threads[i], &retval);
    }
    if(flag == 0) {
        printf("Base future get function validated\n");
    }
    for(i=0; i < cores*ncopy; i++){
        PARSEC_OBJ_RELEASE(fut_array[i]);
    }
    
    /* Test base future callback functionality */
    for(i=0; i< cores; i++){
        data[i] = i;
        fut_array[i] = PARSEC_OBJ_NEW(parsec_base_future_t);
        /* Initialize the callback function */
        parsec_future_init(fut_array[i], test_cb);
    }
    
    for(i=0; i< cores; i++){
        ids[i] = i;
        pthread_create(&threads[i], NULL, do_test2, &ids[i]);
    }
    for(i=0; i< cores; i++){
        flag += pthread_join(threads[i], &retval);
    }
    if(flag == 0) {
        printf("Base future callback function validated\n");
    }
    for(i=0; i < cores; i++){
        PARSEC_OBJ_RELEASE(fut_array[i]);
    }
    
    free(threads);
    if(parsec_future_is_ready(c_fut)){
        printf("countable future successfully triggered\n");
    }
    parsec_future_set(c_fut, NULL); /* trigger a warning of setting a ready future */

    free(fut_array);
    free(data);
    PARSEC_OBJ_RELEASE(c_fut);
}
