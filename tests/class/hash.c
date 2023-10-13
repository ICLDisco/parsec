/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
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
#include <sys/time.h>
#include <math.h>
#include "parsec/class/barrier.h"
#include "parsec/bindthread.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"

#include "parsec/class/parsec_hash_table.h"

#define START_BASE 4
#define START_MASK (0xFFFFFFFF >> (32-START_BASE))

static parsec_hash_table_t hash_table;
static parsec_barrier_t barrier1;
static parsec_barrier_t barrier2;
static int nbcores;

typedef struct {
    parsec_hash_table_item_t ht_item;
    int               thread_id;  /* The id of the thread that last inserted that item */
    int               nbthreads;  /* How many threads there were when this item was inserted */
    int               thread_key; /* which of the keys is that key */
} empty_hash_item_t;

static parsec_key_fn_t key_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

typedef struct {
    int id;
    int nbthreads;
    int new_table_each_time;
    int nb_loops;
    int nb_tests;
    bool use_handle;
    uint64_t *keys;
} param_t;

static void *do_perf_test(void *_param)
{
    param_t *param = (param_t*)_param;
    int id = param->id;
    int nbthreads = param->nbthreads;
    int nbtests = param->nb_tests / nbthreads + (id < (param->nb_tests % nbthreads));
    parsec_time_t t0, t1;
    int l, t;
    uint64_t duration, max_duration = 0;
    empty_hash_item_t *item_array;

    parsec_bindthread(id%nbcores, 0);

    item_array = malloc(sizeof(empty_hash_item_t)*nbtests);
    for(t = 0; t < nbtests; t++) {
        assert(nbthreads * t + id < param->nb_tests);
        item_array[t].ht_item.key = param->keys[nbthreads * t + id];
        item_array[t].thread_id = id;
        item_array[t].nbthreads = nbthreads;
        item_array[t].thread_key = nbthreads * t + id;
    }

    for(l = 0; l < param->nb_loops; l++) {
        if( id == 0 && (l == 0 || param->new_table_each_time)) {
            parsec_hash_table_init(&hash_table, offsetof(empty_hash_item_t, ht_item), 3, key_functions, NULL);
        }

        parsec_barrier_wait(&barrier1);
        t0 = take_time();
        for(t = 0; t < nbtests; t++) {
            assert(item_array[t].ht_item.key != 0);
            parsec_hash_table_insert(&hash_table, &item_array[t].ht_item);
        }
        t1 = take_time();
        duration = diff_time(t0, t1);
        if(duration > max_duration)
            max_duration = duration;
        parsec_barrier_wait(&barrier1);
        printf("Time to do %d insertions on thread %d: %"PRIu64" ns\n", nbtests, id, duration);
        if(0 == id)
            parsec_hash_table_stat(&hash_table);
        parsec_barrier_wait(&barrier1);
        t0 = take_time();
        for(t = 0; t < nbtests; t++) {
            empty_hash_item_t *rc;
            rc = parsec_hash_table_remove(&hash_table, item_array[t].ht_item.key);
            assert(rc == &item_array[t]); (void)rc;
        }
        t1 = take_time();
        duration = diff_time(t0, t1);
        if(duration > max_duration)
            max_duration = duration;
        parsec_barrier_wait(&barrier1);
        printf("Time to do %d removals on thread %d: %"PRIu64" ns\n", nbtests, id, duration);

        if( id == 0 && (l == param->nb_loops-1 || param->new_table_each_time) ) {
            parsec_hash_table_fini(&hash_table);
        }
    }
    return (void*)(uintptr_t)max_duration;
}

static void *do_test(void *_param)
{
    param_t *param = (param_t*)_param;
    int id = param->id;
    int nbthreads = param->nbthreads;
    int nbtests = param->nb_tests / nbthreads + (id < (param->nb_tests % nbthreads));
    int limit = (id*72 + 573) % nbtests;
    bool use_handle = param->use_handle;
    parsec_key_handle_t kh;
    parsec_time_t t0, t1;
    int l, t;
    void *rc;
    uint64_t duration;
    empty_hash_item_t *item_array;

    parsec_bindthread(id%nbcores, 0);

    item_array = malloc(sizeof(empty_hash_item_t)*nbtests);
    for(t = 0; t < nbtests; t++) {
        item_array[t].ht_item.key = param->keys[nbthreads * t + id];
        item_array[t].thread_id = id;
        item_array[t].nbthreads = nbthreads;
        item_array[t].thread_key = nbthreads * t + id;
    }

    parsec_barrier_wait(&barrier1);

    t0 = take_time();
    for(l = 0; l < param->nb_loops; l++) {
        if( l==0 || param->new_table_each_time ) {
            if(0 == id) {
                parsec_hash_table_init(&hash_table, offsetof(empty_hash_item_t, ht_item), 3, key_functions, NULL);
            }
            parsec_barrier_wait(&barrier2);
        }

        for(t = 0; t < limit; t++) {
            if (use_handle) {
                parsec_hash_table_lock_bucket_handle(&hash_table, item_array[t].ht_item.key, &kh);
                rc = parsec_hash_table_nolock_find_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
                rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            }
            if( NULL != rc ) {
                fprintf(stderr,
                        "Error in implementation of the hash table: item with key %"PRIu64" has not been inserted yet, but it is found in the hash table\n"
                        "Thread %d is supposed to have inserted it when running with %d threads, and I am thread %d, running with %d threads\n",
                        (uint64_t)item_array[t].ht_item.key,
                        ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads,
                        id, nbthreads);
                //raise(SIGABRT);
            }
            if (use_handle) {
                parsec_hash_table_nolock_insert_handle(&hash_table, &kh, &item_array[t].ht_item);
                parsec_hash_table_unlock_bucket_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_nolock_insert(&hash_table, &item_array[t].ht_item);
                parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
            }
        }
        for(t = 0; t < limit; t++) {
            if (use_handle) {
                parsec_hash_table_lock_bucket_handle(&hash_table, item_array[t].ht_item.key, &kh);
                rc = parsec_hash_table_nolock_find_handle(&hash_table, &kh);
                parsec_hash_table_unlock_bucket_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
                rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
                parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
            }
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table 3: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
            }
        }
        if(0 == id)
            parsec_hash_table_stat(&hash_table);
        for(t = 0; t < limit; t++) {
            rc = parsec_hash_table_remove(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table 4: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
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
            if (use_handle) {
                parsec_hash_table_lock_bucket_handle(&hash_table, item_array[t].ht_item.key, &kh);
                rc = parsec_hash_table_nolock_find_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
                rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
            }
            if( NULL != rc ) {
                fprintf(stderr,
                        "Error in implementation of the hash table: item with key %"PRIu64" has not been inserted yet, but it is found in the hash table\n"
                        "Thread %d is supposed to have inserted it when running with %d threads, and I am thread %d, running with %d threads\n",
                        (uint64_t)item_array[t].ht_item.key,
                        ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads,
                        id, nbthreads);
                //raise(SIGABRT);
            }
            if (use_handle) {
                parsec_hash_table_nolock_insert_handle(&hash_table, &kh, &item_array[t].ht_item);
                parsec_hash_table_unlock_bucket_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_nolock_insert(&hash_table, &item_array[t].ht_item);
                parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
            }
        }
        for(t = limit; t < nbtests; t++) {
            if (use_handle) {
                parsec_hash_table_lock_bucket_handle(&hash_table, item_array[t].ht_item.key, &kh);
                rc = parsec_hash_table_nolock_find_handle(&hash_table, &kh);
                parsec_hash_table_unlock_bucket_handle(&hash_table, &kh);
            } else {
                parsec_hash_table_lock_bucket(&hash_table, item_array[t].ht_item.key);
                rc = parsec_hash_table_nolock_find(&hash_table, item_array[t].ht_item.key);
                parsec_hash_table_unlock_bucket(&hash_table, item_array[t].ht_item.key);
            }
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table 1: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
                            (uint64_t)item_array[t].ht_item.key);
                } else {
                    fprintf(stderr, "Error in implementation of the hash table: Should have found item with key %"PRIu64" as inserted by thread %d/%d, but found it as inserted from thread %d/%d\n",
                            (uint64_t)item_array[t].ht_item.key,
                            id, nbthreads,
                            ((empty_hash_item_t*)rc)->thread_id, ((empty_hash_item_t*)rc)->nbthreads);
                }
            }
        }
        for(t = limit; t < nbtests; t++) {
            rc = parsec_hash_table_remove(&hash_table, item_array[t].ht_item.key);
            if( rc != &item_array[t] ) {
                if( NULL == rc ) {
                    fprintf(stderr, "Error in implementation of the hash table 2: item with key %"PRIu64" is not to be found in the hash table, but it was not removed yet\n",
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

        if( l==param->nb_loops-1 || param->new_table_each_time ) {
            parsec_barrier_wait(&barrier1);
            if(0 == id) {
                parsec_hash_table_fini(&hash_table);
            }
        }
    }
    t1 = take_time();

    duration = diff_time(t0, t1);

    parsec_barrier_wait(&barrier2);
    free(item_array);

    return (void*)(uintptr_t)duration;
}

typedef struct node_s {
    uint64_t value;
    struct node_s *smaller;
    struct node_s *bigger;
} node_t;

int node_add(node_t **tree, uint64_t value)
{
    node_t *temp = NULL;
    if(!(*tree)) {
        temp = (node_t *)malloc(sizeof(node_t));
        temp->smaller = temp->bigger = NULL;
        temp->value = value;
        *tree = temp;
        return 1;
    }
    if(value == (*tree)->value)
        return 0;
    if(value < (*tree)->value) {
        return node_add(&(*tree)->smaller, value);
    } else {
        return node_add(&(*tree)->bigger, value);
    }
}

void free_tree(node_t **tree)
{
    if(!(*tree)) return;
    free_tree(&(*tree)->bigger);
    free_tree(&(*tree)->smaller);
    free(*tree);
    *tree = NULL;
}

static  void init_keys(uint64_t *keys, int nbkeys, int64_t seed, int structured_keys)
{
    struct timeval start, end, delta;
    int print_end = 0;
    gettimeofday(&start, NULL);

    int cur_sec = 0;
    if (!structured_keys) {
        node_t *tree = NULL;
        if (seed == 0) {
            seed = start.tv_sec * 1000000 + start.tv_usec;
        }
        srand(seed);
        for(int t = 0; t < nbkeys; t++) {
            uint64_t c;
            do {
                c = rand();
            } while( !node_add(&tree, c));
            keys[t] = c;
            gettimeofday(&end, NULL);
            timersub(&end, &start, &delta);
            if(delta.tv_sec > cur_sec) {
                if(cur_sec == 0) {
                    fprintf(stderr, "### Building an array with %d unique items %5.1f%% done", nbkeys, 100.0*t/(double)nbkeys); fflush(stderr);
                    print_end=1;
                } else {
                    fprintf(stderr, "\r### Building an array with %d unique items %5.1f%% done", nbkeys, 100.0*t/(double)nbkeys); fflush(stderr);
                }
                cur_sec = delta.tv_sec;
            }
        }
        free_tree(&tree);
    } else {
        uint64_t l0, l1, l2;
        int count = 0;
        l0 = l1 = 1 + (int)cbrt(nbkeys);
        l2 = (nbkeys + (l1*l0) -1) / (l1*l0);
        for (uint64_t i = 0; i < l0; ++i) {
            for (uint64_t j = 0; j < l1; ++j) {
                for (uint64_t k = 0; k < l2; ++k) {
                    if (count == nbkeys) goto gen_done;
                    keys[count] = (i << 42) + (j << 21) + k;
                    count++;
                }
            }
        }
    }
gen_done:
    gettimeofday(&end, NULL);
    timersub(&end, &start, &delta);
    if(print_end)
        fprintf(stderr, "\r### Building an array with %d unique items. done in %d.%06ds\n", nbkeys, (int)delta.tv_sec, (int)delta.tv_usec);
}

int main(int argc, char *argv[])
{
    pthread_t *threads;
    int ch;
    char *m;
    int e, minthreads = 0, maxthreads = 0, nbthreads;
    uint64_t maxtime;
    void *retval;
    param_t *params;
    uint64_t *keys;
    int64_t seed = -1;
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
    int simple_perf = 0;
    bool use_handle = 0;
    int nb_tests = 30000;
    int nb_loops = 300;
    int new_table_each_time = 0;
    int structured_keys = 0;

    parsec_debug_init();
    parsec_hwloc_init();
    parsec_mca_param_init();
    parsec_hash_tables_init();

    /* set some default for the hardware */
    nbcores = parsec_hwloc_nb_real_cores();
    minthreads = nbcores - 1;
    maxthreads = minthreads + 1;

    mc_hint_index = parsec_mca_param_find("parsec", NULL, "hash_table_max_collisions_hint");
    md_hint_index = parsec_mca_param_find("parsec", NULL, "hash_table_max_table_nb_bits");
    if( mc_hint_index == PARSEC_ERROR ||
        md_hint_index == PARSEC_ERROR ) {
        fprintf(stderr, "Warning: unable to find the hash table hint, tuning behavior will be disabled\n");
    }

    while( (ch = getopt(argc, argv, "c:m:M:t:T:i:d:D:I:#:s:r:3hnpH?")) != -1 ) {
        switch(ch) {
        case 'c':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -c value");
            }
            minthreads  = ch - 1;
            maxthreads = minthreads + 1;
            break;
        case 'm':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -m value");
            }
            minthreads = ch - 1;
            break;
        case 'M':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -M value");
            }
            maxthreads = ch;
            break;
        case 't':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -t value");
            }
            mc_tuning_min = ch;
            break;
        case 'T':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -T value");
            }
            mc_tuning_max = ch;
            break;
        case 'i':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -i value");
            }
            mc_tuning_inc = ch;
            break;
        case 'd':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -t value");
            }
            md_tuning_min = ch;
            break;
        case 'D':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -T value");
            }
            md_tuning_max = ch;
            break;
        case 'I':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -i value");
            }
            md_tuning_inc = ch;
            break;
        case 'n':
            new_table_each_time = 1;
            break;
        case '#':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -# value");
            }
            nb_tests = ch;
            break;
        case 's':
            seed = strtoll(optarg, &m, 0);
            if( (seed < -1) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -s value");
                seed = -1;
            }
            break;
        case 'r':
            ch = strtol(optarg, &m, 0);
            if( (ch <= 0) || (m[0] != '\0') ) {
                fprintf(stderr, "%s: %s\n", argv[0], "invalid -r value");
            }
            nb_loops = ch;
            break;
        case 'p':
            simple_perf = 1;
            break;
        case 'H':
            use_handle = true;
            break;
        case '3':
            structured_keys = 1;
            break;
        case 'h':
        case '?':
        default:
            fprintf(stderr,
                    "Usage: %s [-c nbthreads|-m minthreads -M maxthreads]\n"
                    "          [-t max_coll_min -T max_coll_max -i max_coll_inc]\n"
                    "          [-d max_table_depth_min -D max_table_depth_max -I max_table_depth_inc]\n"
                    "          [-# number of items to insert][-r number of loops of the test][-n use a new hash table for each test]\n"
                    "          [-p (run simple performance test)]\n"
                    "          [-s key generator seed (default: -1, random)]\n"
                    "          [-3 use structured 3D key space instead of random keys (false)]\n"
                    "          [-H (use key handles for locking buckets)]\n", argv[0]);
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
    if( maxthreads > nbcores ) {
        fprintf(stderr,
                "Warning: max threads (%d) > #physical cores (%d).\n",
                maxthreads, nbcores);
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
    params = calloc(sizeof(param_t), maxthreads+1);
    keys = calloc(sizeof(uint64_t), nb_tests);
    init_keys(keys, nb_tests, seed, structured_keys);

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
                    params[e].id = e;
                    params[e].nbthreads = nbthreads+1;
                    params[e].keys = keys;
                    params[e].nb_tests = nb_tests;
                    params[e].nb_loops = nb_loops;
                    params[e].new_table_each_time = new_table_each_time;
                    params[e].use_handle = use_handle;
                }

                parsec_barrier_init(&barrier1, NULL, nbthreads+1);
                parsec_barrier_init(&barrier2, NULL, nbthreads+1);

                if( simple_perf ) {
                    for(e = 0; e < nbthreads; e++) {
                        pthread_create(&threads[e], NULL, do_perf_test, &params[e]);
                    }
                    maxtime = (uint64_t)do_perf_test(&params[nbthreads]);
                } else {
                    for(e = 0; e < nbthreads; e++) {
                        pthread_create(&threads[e], NULL, do_test, &params[e]);
                    }
                    maxtime = (uint64_t)do_test(&params[nbthreads]);
                }
                for(e = 0; e < nbthreads; e++) {
                    pthread_join(threads[e], &retval);
                    if( (uint64_t)retval > maxtime )
                        maxtime = (uint64_t)retval;
                }
                parsec_barrier_destroy(&barrier1);
                parsec_barrier_destroy(&barrier2);
                printf("%lu threads %"PRIu64" "TIMER_UNIT" max_coll %d max_table_depth %d\n",
                       (long)(nbthreads+1), maxtime, mc_tuning, md_tuning);
                fflush(stdout);
            }
        }
    }
    free(threads);
    free(keys);
    free(params);
}
