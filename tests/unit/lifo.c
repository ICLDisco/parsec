/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#undef NDEBUG
#include <pthread.h>
#include <stdarg.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

#define PARSEC_LIFO_ALIGNMENT_DEFAULT 5
#include "parsec/class/lifo.h"
#include "parsec/os-spec-timing.h"
#include "parsec/bindthread.h"

static unsigned int NBELT = 8192;
static unsigned int NBTIMES = 1000000;

static void fatal(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    vprintf(format, va);
    va_end(va);
    raise(SIGABRT);
}

static parsec_lifo_t lifo1;
static parsec_lifo_t lifo2;

typedef struct {
    parsec_list_item_t list;
    unsigned int base;
    unsigned int nbelt;
    unsigned int elts[1];
} elt_t;

static elt_t *create_elem(parsec_lifo_t* lifo, int base)
{
    elt_t *elt;
    size_t r;
    unsigned int j;

    r = rand() % 1024;
    PARSEC_LIFO_ITEM_ALLOC( lifo, elt, r * sizeof(unsigned int) + sizeof(elt_t) );
    elt->base = base;
    elt->nbelt = r;
    for(j = 0; j < r; j++)
        elt->elts[j] = elt->base + j;

    return elt;
}

static void check_elt(elt_t *elt)
{
    unsigned int j;
    for(j = 0; j < elt->nbelt; j++)
        if( elt->elts[j] != elt->base + j )
            fatal(" ! Error: element number %u of elt with base %u is corrupt\n", j, elt->base);
}

static void check_translate_outoforder(parsec_lifo_t *l1,
                                    parsec_lifo_t *l2,
                                    const char *lifo1name,
                                    const char *lifo2name)
{
    static unsigned char *seen = NULL;
    unsigned int e;
    elt_t *elt;

    printf(" - pop them from %s, check they are ok, push them back in %s, and check they are all there\n",
           lifo1name, lifo2name);

    if( NULL == seen )
        seen = (unsigned char *)calloc(1, NBELT);
    else
        memset(seen, 0, NBELT);

    for(e = 0; e < NBELT; e++) {
        elt = (elt_t *)parsec_lifo_pop( l1 );
        if( NULL == elt )
            fatal(" ! Error: there are only %u elements in %s -- expecting %u\n", e+1, lifo1name, NBELT);
        check_elt( elt );
        parsec_lifo_push( l2, (parsec_list_item_t *)elt );
        if( elt->base >= NBELT )
            fatal(" ! Error: base of the element %u of %s is outside boundaries\n", e, lifo1name);
        if( seen[elt->base] == 1 )
            fatal(" ! Error: the element %u appears at least twice in %s\n", elt->base, lifo1name);
        seen[elt->base] = 1;
    }
    /* No need to check that seen[e] == 1 for all e: this is captured by if (NULL == elt) */
    if( (elt = (elt_t*)parsec_lifo_pop( l1 )) != NULL ) 
        fatal(" ! Error: unexpected element of base %u in %s: it should be empty\n",
              elt->base, lifo1name);
}

static void check_translate_inorder(parsec_lifo_t *l1,
                                    parsec_lifo_t *l2,
                                    const char *lifo1name,
                                    const char *lifo2name)
{
    unsigned int e;
    elt_t *elt;
    printf(" - pop them from %s, check they are ok, and push them back in %s\n",
           lifo1name, lifo2name);

    elt = (elt_t *)parsec_lifo_pop( l1 );
    if( NULL == elt )
        fatal(" ! Error: expecting a full list in %s, got an empty one...\n", lifo1name);
    if( elt->base == 0 ) {
        check_elt( elt );
        parsec_lifo_push( l2, (parsec_list_item_t *)elt );
        for(e = 1; e < NBELT; e++) {
            elt = (elt_t *)parsec_lifo_pop( l1 );
            if( NULL == elt )
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, lifo1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            parsec_lifo_push( l2, (parsec_list_item_t *)elt );
        }
    } else if( elt->base == NBELT-1 ) {
        check_elt( elt );
        parsec_lifo_push( l2, (parsec_list_item_t *)elt );
        for(e = NBELT-2; ; e--) {
            elt = (elt_t *)parsec_lifo_pop( l1 );
            if( NULL == elt )
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, lifo1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            parsec_lifo_push( l2, (parsec_list_item_t *)elt );
            if( 0 == e )
                break;
        }
    } else {
        fatal(" ! Error: the lifo %s does not start with 0 or %u\n", lifo1name, NBELT-1);
    }
}



static pthread_mutex_t heavy_synchro_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  heavy_synchro_cond = PTHREAD_COND_INITIALIZER;
static unsigned int    heavy_synchro = 0;

static void *translate_elements_random(void *params)
{
    unsigned int i;
    parsec_list_item_t *e;
    uint64_t *p = (uint64_t*)params;
    parsec_time_t start, end;

    pthread_mutex_lock(&heavy_synchro_lock);
    while( heavy_synchro == 0 ) {
        pthread_cond_wait(&heavy_synchro_cond, &heavy_synchro_lock);
    }
    pthread_mutex_unlock(&heavy_synchro_lock);

    i = 0;
    start = take_time();
    while( i < heavy_synchro ) {
        if( (rand() % 2) == 0 ) {
            e = parsec_lifo_pop( &lifo1 );
            if(NULL != e) {
                parsec_lifo_push(&lifo2, e);
                i++;
            }
        } else {
            e = parsec_lifo_pop( &lifo2 );
            if(NULL != e) {
                parsec_lifo_push(&lifo1, e);
                i++;
            }
        }
    }
    end = take_time();
    *p = diff_time(start, end);

    return NULL;
}

static void usage(const char *name, const char *msg)
{
    if( NULL != msg ) {
        fprintf(stderr, "%s\n", msg);
    }
    fprintf(stderr,
            "Usage: \n"
            "   %s [-c cores|-n nbelt|-h|-?]\n"
            " where\n"
            "   -c cores:   cores (integer >0) defines the number of cores to test\n"
            "   -n nbelt:   nbelt (integer >0) defines the number of elements to use (default %u)\n"
            "   -N nbtimes: nbtimes (integer >0) defines the number of times elements must be moved from one list to another (default %u)\n",
            name,
            NBELT,
            NBTIMES);
    exit(1);
}

int main(int argc, char *argv[])
{
    unsigned int e;
    elt_t *elt, *p;
    pthread_t *threads;
    uint64_t *times;
    uint64_t min_time, max_time, sum_time;
    long int nbthreads = 1;
    int ch;
    char *m;

    min_time = 0;
    max_time = 0xffffffff;
#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
#endif
    while( (ch = getopt(argc, argv, "c:n:N:h?")) != -1 ) {
        switch(ch) {
        case 'c':
            nbthreads = strtol(optarg, &m, 0);
            if( (nbthreads <= 0) || (m[0] != '\0') ) {
                usage(argv[0], "invalid -c value");
            }
            break;
        case 'n':
            NBELT = strtol(optarg, &m, 0);
            if( (NBELT <= 0) || (m[0] != '\0') ) {
                usage(argv[0], "invalid -n value");
            }
            break;
        case 'N':
            NBTIMES = strtol(optarg, &m, 0);
            if( (NBTIMES <= 0) || (m[0] != '\0') ) {
                usage(argv[0], "invalid -N value");
            }
            break;
        case 'h':
        case '?':
        default:
            usage(argv[0], NULL);
            break;
        }
    }

    threads = (pthread_t*)calloc(sizeof(pthread_t), nbthreads);
    times = (uint64_t*)calloc(sizeof(uint64_t), nbthreads);

    OBJ_CONSTRUCT(&lifo1, parsec_lifo_t);
    OBJ_CONSTRUCT(&lifo2, parsec_lifo_t);

    printf("Sequential test.\n");

    printf(" - create %u random elements and push them in lifo1\n", NBELT);
    for(e = 0; e < NBELT; e++) {
        elt = create_elem(&lifo1, e);
        parsec_lifo_push( &lifo1, (parsec_list_item_t *)elt );
    }

    check_translate_inorder(&lifo1, &lifo2, "lifo1", "lifo2");
    check_translate_inorder(&lifo2, &lifo1, "lifo2", "lifo1");

    printf("Parallel test.\n");

    printf(" - translate elements from lifo1 to lifo2 or from lifo2 to lifo1 (random), %u times on %ld threads\n",
           NBTIMES, nbthreads);
    for(e = 0; e < nbthreads; e++) {
        times[e] = e;
        pthread_create(&threads[e], NULL, translate_elements_random, &times[e]);
    }

    pthread_mutex_lock(&heavy_synchro_lock);
    heavy_synchro = NBTIMES;
    pthread_cond_broadcast(&heavy_synchro_cond);
    pthread_mutex_unlock(&heavy_synchro_lock);

    sum_time = 0;
    for(e = 0; e < nbthreads; e++) {
        pthread_join(threads[e], NULL);
        if( sum_time == 0 ) {
            min_time = times[e];
            max_time = times[e];
        } else {
            if( min_time > times[e] ) min_time = times[e];
            if( max_time < times[e] ) max_time = times[e];
        }
        sum_time += times[e];
    }
    printf("== Time to move %u times per thread for %ld threads from l1 to l2 or l2 to l1 randomly:\n"
           "== MIN %"PRIu64" %s\n"
           "== MAX %"PRIu64" %s\n"
           "== AVG %g %s\n",
           NBTIMES, nbthreads,
           min_time, TIMER_UNIT,
           max_time, TIMER_UNIT,
           (double)sum_time / (double)nbthreads, TIMER_UNIT);

    printf(" - move all elements to lifo1\n");
    p = NULL;
    ch = 0;
    while( !parsec_lifo_is_empty( &lifo2 ) ) {
        elt = (elt_t*)parsec_lifo_pop( &lifo2 );
        if( elt == NULL ) 
            fatal(" ! Error: list lifo2 is supposed to be non empty, but it is!\n");
        if( elt == p )
            fatal(" ! I keep poping the same element in the list at element %u... It is now officially a frying pan\n",
                  ch);
        ch++;
        p = elt;
        parsec_lifo_push( &lifo1, (parsec_list_item_t*)elt );
    }

    check_translate_outoforder(&lifo1, &lifo2, "lifo1", "lifo2");

    printf(" - pop all elements from lifo1, and free them\n");
    while( !parsec_lifo_is_empty( &lifo1 ) ) {
        elt = (elt_t*)parsec_lifo_pop( &lifo1 );
        free(elt);
    }
    printf(" - pop all elements from lifo2, and free them\n");
    while( !parsec_lifo_is_empty( &lifo2 ) ) {
        elt = (elt_t*)parsec_lifo_pop( &lifo2 );
        free(elt);
    }

    free(threads);

    printf(" - all tests passed\n");

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalized(&ch);
#endif
    return 0;
}
