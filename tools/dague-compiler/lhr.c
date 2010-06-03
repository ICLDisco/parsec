#include "dague.h"
#include "cholesky.h"

static dague_ontask_iterate_t printit(dague_execution_unit_t *_eu, const dague_execution_context_t *context, void *_)
{
    if( !strcmp(context->function->name, "POTRF") ) {
        printf(" -> %s(%d)\n", context->function->name, 
               context->locals[0].value);
    }
    if( !strcmp(context->function->name, "SYRK") ) {
        printf(" -> %s(%d, %d)\n", context->function->name, 
               context->locals[0].value, 
               context->locals[1].value);
    }
    if( !strcmp(context->function->name, "TRSM") ) {
        printf(" -> %s(%d, %d)\n", context->function->name, 
               context->locals[0].value, 
               context->locals[1].value);
    }
    if( !strcmp(context->function->name, "GEMM") ) {
        printf(" -> %s(%d, %d, %d)\n", context->function->name, 
               context->locals[0].value, 
               context->locals[1].value, 
               context->locals[2].value);
    }

    context->function->iterate_successors(NULL, context, printit, NULL);

    return DAGUE_ITERATE_CONTINUE;
}

int main(int argc, char *argv[])
{
    dague_object_t *o = dague_cholesky_new(NULL, 4, 10);
    dague_execution_context_t init;

    init.dague_object = o;
    init.function = (dague_t*)o->functions_array[3];
    init.locals[0].value = 0;

    printit(NULL, &init, NULL);
    
    o->functions_array[3]->iterate_successors(NULL, &init, printit, NULL);

    return 0;
}
