#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symbol.h"

extern int dplasma_lineno;

static const symbol_t** dplasma_symbol_array = NULL;
static int dplasma_symbol_array_count = 0,
    dplasma_symbol_array_size = 0;

void symbol_dump(const symbol_t *s, const char *prefix)
{
    if( NULL == s->name ) {
        return;
    }

    if( s->min == s->max ) {
        printf("%s%s = ", prefix, s->name);
        expr_dump(s->min);
        printf("\n" );
    } else {
        printf("%s%s = [", prefix, s->name);
        expr_dump(s->min);
        printf(" .. ");
        expr_dump(s->max);
        printf("]\n");
    }
}

void symbol_dump_all( const char* prefix )
{
    const symbol_t* symbol;
    int i;

    for( i = 0; i < dplasma_symbol_array_count; i++ ) {
        symbol = dplasma_symbol_array[i];
        symbol_dump( symbol, prefix );
    }
}

int dplasma_add_global_symbol( const char* name, const expr_t* expr )
{
    symbol_t* symbol;

    if( NULL != dplasma_search_global_symbol(name) ) {
        printf( "Overwrite an already defined symbol %s at line %d\n",
                name, dplasma_lineno );
        return -1;
    }

    if( dplasma_symbol_array_count >= dplasma_symbol_array_size ) {
        if( 0 == dplasma_symbol_array_size ) {
            dplasma_symbol_array_size = 4;
        } else {
            dplasma_symbol_array_size *= 2;
        }
        dplasma_symbol_array = (const symbol_t**)realloc( dplasma_symbol_array,
                                                          dplasma_symbol_array_size * sizeof(symbol_t*) );
        if( NULL == dplasma_symbol_array ) {
            return -1;  /* No more available memory */
        }
    }

    symbol = (symbol_t*)calloc(1, sizeof(symbol_t));
    symbol->name = strdup(name);
    symbol->min = expr;
    symbol->max = expr;

    dplasma_symbol_array[dplasma_symbol_array_count] = symbol;
    dplasma_symbol_array_count++;
    return 0;
}

const symbol_t* dplasma_search_global_symbol( const char* name )
{
    int i;
    const symbol_t* symbol;

    for( i = 0; i < dplasma_symbol_array_count; i++ ) {
        symbol = dplasma_symbol_array[i];
        if( 0 == strcmp(symbol->name, name) ) {
            return symbol;
        }
    }
    return NULL;
}
