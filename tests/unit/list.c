#include "dague_config.h"
#undef NDEBUG
#include <pthread.h>
#include <stdarg.h>
#include <signal.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#if defined(HAVE_MPI)
#include <mpi.h>
#endif
#if defined(HAVE_HWLOC)
#include "dague_hwloc.h"
#endif
#include "dague/class/list.h"
#include "os-spec-timing.h"
#include "bindthread.h"

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

static dague_list_t l1;
static dague_list_t l2;

typedef struct {
    dague_list_item_t list;
    unsigned int base;
    unsigned int nbelt;
    unsigned int elts[1];
} elt_t;

static elt_t *create_elem(int base)
{
    elt_t *elt;
    size_t r;
    unsigned int j;

    r = rand() % 1024;
    elt = (elt_t*)malloc(r * sizeof(unsigned int) + sizeof(elt_t));
    OBJ_CONSTRUCT(&elt->list, dague_list_item_t);
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

static void check_lifo_translate_outoforder(dague_list_t *l1,
                                    dague_list_t *l2,
                                    const char *l1name,
                                    const char *l2name)
{
    static unsigned char *seen = NULL;
    unsigned int e;
    elt_t *elt;

    printf(" - pop them from %s, check they are ok, push them back in %s, and check they are all there\n",
           l1name, l2name);

    if( NULL == seen ) 
        seen = (unsigned char *)calloc(1, NBELT);
    else
        memset(seen, 0, NBELT);

    for(e = 0; e < NBELT; e++) {
        elt = (elt_t *)dague_list_lifo_pop( l1 );
        if( NULL == elt ) 
            fatal(" ! Error: there are only %u elements in %s -- expecting %u\n", e+1, l1name, NBELT);
        check_elt( elt );
        dague_list_lifo_push( l2, (dague_list_item_t *)elt );
        if( elt->base >= NBELT )
            fatal(" ! Error: base of the element %u of %s is outside boundaries\n", e, l1name);
        if( seen[elt->base] == 1 ) 
            fatal(" ! Error: the element %u appears at least twice in %s\n", elt->base, l1name);
        seen[elt->base] = 1;
    }
    /* No need to check that seen[e] == 1 for all e: this is captured by if (NULL == elt) */
    if( (elt = (elt_t*)dague_list_lifo_pop( l1 )) != NULL ) 
        fatal(" ! Error: unexpected element of base %u in %s: it should be empty\n", 
              elt->base, l1name);
}

static void check_lifo_translate_inorder(dague_list_t *l1,
                                        dague_list_t *l2,
                                    const char *l1name,
                                    const char *l2name)
{
    unsigned int e;
    elt_t *elt;
    printf(" - pop them from %s, check they are ok, and push them back in %s\n",
           l1name, l2name);

    elt = (elt_t *)dague_ulist_lifo_pop( l1 );
    if( NULL == elt ) 
        fatal(" ! Error: expecting a full list in %s, got an empty one...\n", l1name);
    if( elt->base == 0 ) {
        check_elt( elt );
        dague_ulist_lifo_push( l2, (dague_list_item_t *)elt );
        for(e = 1; e < NBELT; e++) {
            elt = (elt_t *)dague_ulist_lifo_pop( l1 );
            if( NULL == elt ) 
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, l1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            dague_ulist_lifo_push( l2, (dague_list_item_t *)elt );
        }
    } else if( elt->base == NBELT-1 ) {
        check_elt( elt );
        dague_ulist_lifo_push( l2, (dague_list_item_t *)elt );
        for(e = NBELT-2; ; e--) {
            elt = (elt_t *)dague_ulist_lifo_pop( l1 );
            if( NULL == elt ) 
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, l1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            dague_ulist_lifo_push( l2, (dague_list_item_t *)elt );
            if( 0 == e )
                break;
        }
    } else {
        fatal(" ! Error: the lifo %s does not start with 0 or %u\n", l1name, NBELT-1);
    }
}

#if 0
    /* usefull code snippet */
    DAGUE_LIST_ITERATOR(l2, item, {
        printf(" %04d ", ((elt_t*)item)->base);
    });
    printf("\n");
#endif

#define elt_comparator offsetof(elt_t, base)

static void check_list_sort(dague_list_t* l1, dague_list_t* l2)
{
    printf(" - sort empty list l2\n");
    dague_ulist_sort(l2, elt_comparator);

    printf(" - sort already sorted list l1, check it is in order\n"); 
    dague_ulist_sort(l1, elt_comparator);
    check_lifo_translate_inorder(l1,l2,"l1","l2");
        
    printf(" - sort reverse sorted list l2, check it is in order\n");
    dague_ulist_sort(l2, elt_comparator);
    check_lifo_translate_inorder(l2,l1,"l2","l1");
    
    printf(" - randomize list l1 into l2, sort l2, check it is in order\n");
    elt_t* e;
    while(NULL != (e = (elt_t*)dague_ulist_fifo_pop(l1)))
    {
        int choice = rand()%3; /* do not care for true random*/
        switch(choice)
        {
            case 0:
                dague_ulist_fifo_push(l1, &e->list); /* return in l1, for later */
                break;
            case 1:
                dague_ulist_push_front(l2, &e->list);
                break;
            case 2:
                dague_ulist_push_back(l2, &e->list);
                break;
        }
    }
    dague_list_sort(l2, elt_comparator);
    check_lifo_translate_inorder(l2,l1,"l2","l1");
}

static pthread_mutex_t heavy_synchro_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  heavy_synchro_cond = PTHREAD_COND_INITIALIZER;
static unsigned int    heavy_synchro = 0;

static void *lifo_translate_elements_random(void *params)
{
    unsigned int i;
    dague_list_item_t *e;
    uint64_t *p = (uint64_t*)params;
    dague_time_t start, end;

    dague_bindthread( (int)*p, -1 );

    pthread_mutex_lock(&heavy_synchro_lock);
    while( heavy_synchro == 0 ) {
        pthread_cond_wait(&heavy_synchro_cond, &heavy_synchro_lock);
    }
    pthread_mutex_unlock(&heavy_synchro_lock);

    i = 0;
    start = take_time();
    while( i < heavy_synchro ) {
        if( rand() % 2 == 0 ) {
            e = dague_list_lifo_pop( &l1 );
            if(NULL != e) {
                dague_list_lifo_push(&l2, e);
                i++;
            }
        } else {
            e = dague_list_lifo_pop( &l2 );
            if(NULL != e) {
                dague_list_lifo_push(&l1, e);
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

#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
#endif
#if defined(HAVE_HWLOC)
    dague_hwloc_init();
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

    OBJ_CONSTRUCT( &l1, dague_list_t );
    OBJ_CONSTRUCT( &l2, dague_list_t );

    printf("Sequential test.\n");

    printf(" - create %u random elements and push them in l1\n", NBELT);
    for(e = 0; e < NBELT; e++) {
        elt = create_elem(e);
        dague_ulist_lifo_push( &l1, (dague_list_item_t *)elt );
    }

    check_lifo_translate_outoforder(&l1, &l2, "l1", "l2");
    check_lifo_translate_inorder(&l2, &l1, "l2", "l1");

    check_list_sort(&l1, &l2);


    printf("Parallel test.\n");

    printf(" - translate elements from l1 to l2 or from l2 to l1 (random), %u times on %ld threads\n",
           NBTIMES, nbthreads);
    for(e = 0; e < nbthreads; e++) {
        times[e] = e;
        pthread_create(&threads[e], NULL, lifo_translate_elements_random, &times[e]);
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
    
    printf(" - move all elements to l1\n");
    p = NULL;
    ch = 0;
    while( !dague_ulist_is_empty( &l2 ) ) {
        elt = (elt_t*)dague_ulist_lifo_pop( &l2 );
        if( elt == NULL ) 
            fatal(" ! Error: list l2 is supposed to be non empty, but it is!\n");
        if( elt == p ) 
            fatal(" ! I keep poping the same element in the list at element %u... It is now officially a frying pan\n",
                  ch);
        ch++;
        p = elt;
        dague_ulist_lifo_push( &l1, (dague_list_item_t*)elt );
    }
    
    check_lifo_translate_outoforder(&l1, &l2, "l1", "l2");



    printf(" - pop all elements from l1, and free them\n");
    while( !dague_ulist_is_empty( &l1 ) ) {
        elt = (elt_t*)dague_list_lifo_pop( &l1 );
        free(elt);
    }
    printf(" - pop all elements from l2, and free them\n");
    while( !dague_ulist_is_empty( &l2 ) ) {
        elt = (elt_t*)dague_list_lifo_pop( &l2 );
        free(elt);
    }

    free(threads);

    printf(" - all tests passed\n");

#if defined(HAVE_HWLOC)
    dague_hwloc_fini();
#endif  /* HAVE_HWLOC_BITMAP */
#if defined(HAVE_MPI)
    MPI_Finalized(&ch);
#endif
    return 0;
}
