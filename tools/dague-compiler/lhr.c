#include "dague.h"
#include "cholesky.h"

static DAGuE_ontask_iterate_t printit(DAGuE_execution_unit_t *_eu, const DAGuE_execution_context_t *context, void *_)
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

    return DAGuE_ITERATE_CONTINUE;
}

int main(int argc, char *argv[])
{
    DAGuE_object_t *o = DAGuE_cholesky_new(NULL, 4, 10);
    DAGuE_execution_context_t init;

    init.DAGuE_object = o;
    init.function = (DAGuE_t*)o->functions_array[3];
    init.locals[0].value = 0;

    printit(NULL, &init, NULL);
    
    o->functions_array[3]->iterate_successors(NULL, &init, printit, NULL);

    return 0;
}
