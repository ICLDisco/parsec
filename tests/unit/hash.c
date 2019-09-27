/*
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "parsec/class/barrier.h"
#include "parsec/bindthread.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"
#include "parsec/utils/mca_param.h"

#include "parsec/class/parsec_hash_table.h"

#define NB_TESTS 30000
#define NB_LOOPS 300
#define START_BASE 4
#define START_MASK (0xFFFFFFFF >> (32-START_BASE))

static parsec_hash_table_t hash_table;
static parsec_barrier_t barrier1;
static parsec_barrier_t barrier2;

typedef struct {
    parsec_hash_table_item_t ht_item;
    int               thread_id; /* The id of the thread that last inserted that item */
    int               nbthreads; /* How many threads there were when this item was inserted */
} empty_hash_item_t;

static parsec_key_fn_t key_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static void *do_test(void *_param)
{
    int *param = (int*)_param;
    int id = param[0];
    int nbthreads = param[1];
    int nbtests = NB_TESTS / nbthreads + (id < (NB_TESTS % nbthreads));
    int limit = (id*72 + 573) % nbtests;
    parsec_time_t t0, t1;
    int l, t;
    void *rc;
    uint64_t duration;
    empty_hash_item_t *item_array;
    
    parsec_bindthread(id, 0);

    if( id == 0 ) {
        parsec_hash_table_init(&hash_table, offsetof(empty_hash_item_t, ht_item), 3, key_functions, NULL);
    }

    item_array = malloc(sizeof(empty_hash_item_t)*nbtests);
    for(t = 0; t < nbtests; t++) {
        item_array[t].ht_item.key = (parsec_key_t)((uint64_t)((nbthreads+1) * t + id));
        item_array[t].thread_id = id;
        item_array[t].nbthreads = nbthreads;
    }

    parsec_barrier_wait(&barrier1);
    
    t0 = take_time();
    for(l = 0; l < NB_LOOPS; l++) {
        for(t = 0; t < limit; t++) {
            parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
            rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            if( NULL != rc ) {
                fprintf(stderr,
                        "Error in implementation of the hash table: item with key %"PRIu64" has not been inserted yet, but it is found in the hash table\n"
                        "Thread %d is supposed to have inserted it when running with %d threads, and I am thread %d, running with %d threads\n",
                        (uint64_t)item_array[t].ht_item.key,
                        ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads,
                        id, nbthreads);
                //raise(SIGABRT);
            }
            parsec_hash_table_nolock_insert(&hash_table, &item_array[t].ht_item);
            parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
        }
        for(t = 0; t < limit; t++) {
            parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
            rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
            }
            parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
        }
        for(t = 0; t < limit; t++) {
            rc = parsec_hash_table_remove(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
                //raise(SIGABRT);
            }
        }

        for(t = limit; t < nbtests; t++) {
            parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
            rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            if( NULL != rc ) {
                fprintf(stderr,
                        "Error in implementation of the hash table: item with key %"PRIu64" has not been inserted yet, but it is found in the hash table\n"
                        "Thread %d is supposed to have inserted it when running with %d threads, and I am thread %d, running with %d threads\n",
                        (uint64_t)item_array[t].ht_item.key,
                        ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads,
                        id, nbthreads);
                //raise(SIGABRT);
            }
            parsec_hash_table_nolock_insert(&hash_table, &item_array[t].ht_item);
            parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
        }
        for(t = limit; t < nbtests; t++) {
            parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
            rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
            }
            parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
        }
        for(t = limit; t < nbtests; t++) {
            rc = parsec_hash_table_remove(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
                //raise(SIGABRT);
            }
        }
    }
    t1 = take_time();

    duration = diff_time(t0, t1);

    parsec_barrier_wait(&barrier2);
    if( id == 0 ) {
        parsec_hash_table_fini(&hash_table);
    }
    free(item_array);

    return (void*)(uintptr_t)duration;
}

int main(int argc, char *argv[])
{
    pthread_t *threads;
    int ch;
    char *m;
    uintptr_t e, minthreads = 0, maxthreads = 0, nbthreads;
    uint64_t maxtime;
    void *retval;
    int *params;
    int mc_hint_index = -1;
    int mc_tuning_min = -1;
    int mc_tuning_max = -1;
    int mc_tuning_inc = 1;
    int mc_tuning;
    int md_hint_index = -1;
    int md_tuning_min = -1;
    int md_tuning_max = -1;
    int md_tuning_inc = 1;
    int md_tuning;

    parsec_hwloc_init();
    parsec_mca_param_init();
    parsec_hash_tables_init();

    mc_hint_index = parsec_mca_param_find("parsec", NULL, "hash_table_max_collisions_hint");
    md_hint_index = parsec_mca_param_find("parsec", NULL, "hash_table_max_table_nb_bits");
    if( mc_hint_index == PARSEC_ERROR ||
        md_hint_index == PARSEC_ERROR ) {
        fprintf(stderr, "Warning: unable to find the hash table hint, tuning behavior will be disabled\n");
    }
    
    while( (ch = getopt(argc, argv, "c:m:M:t:T:i:d:D:I:h?")) != -1 ) {
        switch(ch) {
        case 'c':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -c value");
            }
            minthreads  = (uintptr_t)ch;
            maxthreads = minthreads+1;
            break;
        case 'm':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -m value");
            }
            minthreads = (uintptr_t)ch;
            break;
        case 'M':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -M value");
            }
            maxthreads = (uintptr_t)ch;
            break;
        case 't':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -t value");
            }
            mc_tuning_min = ch;
            break;
        case 'T':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -T value");
            }
            mc_tuning_max = ch;
            break;
        case 'i':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -i value");
            }
            mc_tuning_inc = ch;
            break;
        case 'd':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -t value");
            }
            md_tuning_min = ch;
            break;
        case 'D':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -T value");
            }
            md_tuning_max = ch;
            break;
        case 'I':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -i value");
            }
            md_tuning_inc = ch;
            break;
        case 'h':
        case '?':
        default:
            fprintf(stderr,
                    "Usage: %s [-c nbthreads|-m minthreads -M maxthreads]\n"
                    "          [-t max_coll_min -T max_coll_max -i max_coll_inc]\n"
                    "          [-d max_table_depth_min -D max_table_depth_max -I max_table_depth_inc]\n", argv[0]);
            exit(1);
            break;
        }
    }

    if( (maxthreads < minthreads) ) {
        fprintf(stderr,
                "Error: max threads < min threads.\n"
                "Usage: %s [-c nbthreads|-m minthreads -M maxthreads]\n", argv[0]);
        exit(1);
    }

    if( mc_tuning_min > 0 ) {
        if( mc_tuning_max < mc_tuning_min ||
            mc_hint_index < 0 ||
            mc_tuning_inc <= 0 ) {
            fprintf(stderr, "Impossible to do a tuning run (max collisions = %d, max = %d, min = %d, inc = %d, see -h)\n",
                    mc_hint_index, mc_tuning_max, mc_tuning_min, mc_tuning_inc);
            exit(1);
        }
    } else {
        if( mc_hint_index < 0 ) {
            /* This does not matter, since we cannot set it, just define a non-zero range */
            mc_tuning_min = 0;
            mc_tuning_max = 1;
        } else {
            parsec_mca_param_lookup_int(mc_hint_index, &mc_tuning);
            mc_tuning_min = mc_tuning;
            mc_tuning_max = mc_tuning+1;
        }
    }
    
    if( md_tuning_min > 0 ) {
        if( md_tuning_max < md_tuning_min ||
            md_hint_index < 0 ||
            md_tuning_inc <= 0 ) {
            fprintf(stderr, "Impossible to do a tuning run (max table depth = %d, max = %d, min = %d, inc = %d, see -h)\n",
                    md_hint_index, md_tuning_max, md_tuning_min, md_tuning_inc);
            exit(1);
        }
    } else {
        if( md_hint_index < 0 ) {
            /* This does not matter, since we cannot set it, just define a non-zero range */
            md_tuning_min = 0;
            md_tuning_max = 1;
        } else {
            parsec_mca_param_lookup_int(md_hint_index, &md_tuning);
            md_tuning_min = md_tuning;
            md_tuning_max = md_tuning+1;
        }
    }

    threads = calloc(sizeof(pthread_t), maxthreads);
    params = calloc(sizeof(int), 2*(maxthreads+1));

    for(md_tuning = md_tuning_min; md_tuning < md_tuning_max; md_tuning += md_tuning_inc) {
        for(mc_tuning = mc_tuning_min; mc_tuning < mc_tuning_max; mc_tuning += mc_tuning_inc) {
            if(mc_hint_index > 0) {
                parsec_mca_param_set_int(mc_hint_index, mc_tuning);
            }
            if(md_hint_index > 0) {
                parsec_mca_param_set_int(md_hint_index, md_tuning);
            }

            for( nbthreads = minthreads; nbthreads < maxthreads; nbthreads++) {
                for(e = 0; e < nbthreads+1; e++) {
                    params[2*e] = e;
                    params[2*e+1] = nbthreads+1;
                }

                parsec_barrier_init(&barrier1, NULL, nbthreads+1);    
                parsec_barrier_init(&barrier2, NULL, nbthreads+1);    
                for(e = 0; e < nbthreads; e++) {
                    pthread_create(&threads[e], NULL, do_test, &params[2*e]);
                }
                maxtime = (uint64_t)do_test(&params[2*nbthreads]);
                for(e = 0; e < nbthreads; e++) {
                    pthread_join(threads[e], &retval);
                    if( (uint64_t)retval > maxtime )
                        maxtime = (uint64_t)retval;
                }
                parsec_barrier_destroy(&barrier1);
                parsec_barrier_destroy(&barrier2);
                printf("%lu threads %"PRIu64" "TIMER_UNIT" max_coll %d max_table_depth %d\n", nbthreads+1, maxtime, mc_tuning, md_tuning);
                fflush(stdout);
            }
        }
    }
    free(threads);
    free(params);
}
