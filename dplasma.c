#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "dplasma.h"

static const dplasma_t** dplasma_array = NULL;
static int dplasma_array_size = 0, dplasma_array_count = 0;

void dplasma_dump(const dplasma_t *d, const char *prefix)
{
    int i;
    char *pref2 = malloc(strlen(prefix)+3);

    sprintf(pref2, "%s  ", prefix);
    printf("%sDplasma Function: %s\n", prefix, d->name);

    printf("%s Parameter Variables:\n", prefix);
    for(i = 0; NULL != d->locals[i] && i < MAX_LOCAL_COUNT; i++) {
        symbol_dump(d->locals[i], pref2);
    }

    printf("%s Predicates:\n", prefix);
    for(i = 0; NULL != d->preds[i] && i < MAX_PRED_COUNT; i++) {
        printf("%s", pref2);
        expr_dump(d->preds[i]);
        printf("\n");
    }

    printf("%s Parameters and Dependencies:\n", prefix);
    for(i = 0; NULL != d->params[i] && i < MAX_PARAM_COUNT; i++) {
        param_dump(d->params[i], pref2);
    }

    printf("%s Body:\n", prefix);
    printf("%s  %s\n", prefix, d->body);

    free(pref2);
}

void dplasma_dump_all( void )
{
    int i;

    for( i = 0; i < dplasma_array_count; i++ ) {
        printf("/**\n * dplasma_t object named %s index %d\n */\n", dplasma_array[i]->name, i );
        dplasma_dump( dplasma_array[i], "" );
    }
}

int dplasma_push( const dplasma_t* d )
{
    if( dplasma_array_count >= dplasma_array_size ) {
        if( 0 == dplasma_array_size ) {
            dplasma_array_size = 4;
        } else {
            dplasma_array_size *= 2;
        }
        dplasma_array = (const dplasma_t**)realloc( dplasma_array, dplasma_array_size * sizeof(dplasma_t*) );
        if( NULL == dplasma_array ) {
            return -1;  /* No more available memory */
        }
    }
    dplasma_array[dplasma_array_count] = d;
    dplasma_array_count++;
    return 0;
}

const dplasma_t* dplasma_find( const char* name )
{
    int i;
    const dplasma_t* object;

    for( i = 0; i < dplasma_array_count; i++ ) {
        object = dplasma_array[i];
        if( 0 == strcmp( object->name, name ) ) {
            return object;
        }
    }
    return NULL;
}

int dplasma_unroll( const dplasma_t* object )
{
    return 0;
}
