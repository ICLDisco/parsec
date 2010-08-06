#include <pthread.h>
#include <stdarg.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include "lifo.h"

#define NBELT 8192

static void fatal(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    vprintf(format, va);
    va_end(va);
    raise(SIGABRT);
}

static dague_atomic_lifo_t lifo1;
static dague_atomic_lifo_t lifo2;

typedef struct {
    dague_list_item_t list;
    unsigned int base;
    unsigned int nbelt;
    unsigned int elts[1];
} elt_t;

static elt_t *create_elem(int base)
{
    elt_t *elt;
    size_t s;
    unsigned int j;

    s = sizeof(int) * (rand() % (1024 - sizeof(elt_t))) + sizeof(elt_t);
    elt = (elt_t*)malloc( s );
    elt->base = base;
    elt->nbelt = s;
    for(j = 0; j < s; j++)
        elt->elts[j] = elt->base + j;
    DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t *)elt );
    
    return elt;
}

static void check_elt(elt_t *elt)
{
    unsigned int j;
    for(j = 0; j < elt->nbelt; j++)
        if( elt->elts[j] != elt->base + j ) 
            fatal(" ! Error: element number %u of elt with base %u is corrupt\n", j, elt->base);
}

static void check_translate(dague_atomic_lifo_t *l1,
                            dague_atomic_lifo_t *l2,
                            const char *lifo1name,
                            const char *lifo2name)
{
    unsigned int e;
    elt_t *elt;
    printf(" - pop them from %s, check they are ok, and push them back in %s\n",
           lifo1name, lifo2name);

    elt = (elt_t *)dague_atomic_lifo_pop( l1 );
    if( elt->base == 0 ) {
        check_elt( elt );
        dague_atomic_lifo_push( l2, (dague_list_item_t *)elt );
        for(e = 1; e < NBELT; e++) {
            elt = (elt_t *)dague_atomic_lifo_pop( l1 );
            if( NULL == elt ) 
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, lifo1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            dague_atomic_lifo_push( l2, (dague_list_item_t *)elt );
        }
    } else if( elt->base == NBELT-1 ) {
        check_elt( elt );
        dague_atomic_lifo_push( l2, (dague_list_item_t *)elt );
        for(e = NBELT-2; ; e--) {
            elt = (elt_t *)dague_atomic_lifo_pop( l1 );
            if( NULL == elt ) 
                fatal(" ! Error: element number %u was not found at its position in %s\n", e, lifo1name);
            if( elt->base != e )
                fatal(" ! Error: element number %u has its base corrupt\n", e);
            check_elt( elt );
            dague_atomic_lifo_push( l2, (dague_list_item_t *)elt );
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

typedef struct {
    int base;
    int max;
} translating_params_t;

static void *translate_elements(void *params)
{
    translating_params_t *p = (translating_params_t*)params;

    pthread_mutex_lock(&heavy_synchro_lock);
    while( heavy_synchro == 0 ) {
        pthread_cond_wait(&heavy_synchro_cond, &heavy_synchro_lock);
    }
    pthread_mutex_unlock(&heavy_synchro_lock);

    (void)p;

    return NULL;
}


int main(int argc, char *argv[])
{
    unsigned int e;
    elt_t *elt;

    (void)argc;
    (void)argv;

    dague_atomic_lifo_construct( &lifo1 );
    dague_atomic_lifo_construct( &lifo2 );

    printf("Sequential test.\n");

    printf(" - create %u random elements and push them in lifo1\n", NBELT);
    for(e = 0; e < 8192; e++) {
        elt = create_elem(e);
        dague_atomic_lifo_push( &lifo1, (dague_list_item_t *)elt );
    }

    check_translate(&lifo1, &lifo2, "lifo1", "lifo2");
    check_translate(&lifo2, &lifo1, "lifo2", "lifo1");

    return 0;
}
