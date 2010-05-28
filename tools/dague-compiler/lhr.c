#include "dague.h"
#include "cholesky.h"

static DAGuE_ontask_iterate_t printit(DAGuE_execution_unit_t *_eu, const DAGuE_execution_context_t *context, int depth, void *_)
{
    if( !strcmp(context->function->name, "POTRF") ) {
        printf("%*s -> %s(%d)\n", depth, "  ", context->function->name, context->locals[0].value);
    }
    if( !strcmp(context->function->name, "SYRK") ) {
        printf("%*s -> %s(%d, %d)\n", depth, "  ", context->function->name, context->locals[0].value, context->locals[1].value);
    }
    if( !strcmp(context->function->name, "TRSM") ) {
        printf("%*s -> %s(%d, %d)\n", depth, "  ", context->function->name, context->locals[0].value, context->locals[2].value);
    }
    if( !strcmp(context->function->name, "GEMM") ) {
        printf("%*s -> %s(%d, %d, %d)\n", depth, "  ", context->function->name, context->locals[0].value, context->locals[1].value, context->locals[2].value);
    }
    return DAGuE_TRAVERSE_CONTINUE;
}

int main(int argc, char *argv[])
{
    DAGuE_object_t *o = DAGuE_cholesky_new(NULL, 4, 10);
    DAGuE_execution_context_t init;

    init.DAGuE_object = o;
    init.function = (DAGuE_t*)o->functions_array[3];
    init.locals[0].value = 0;
    init.locals[1].value = 0;
    init.locals[2].value = 0;
    init.locals[3].value = 0;
    
    o->functions_array[3]->preorder(NULL, &init, 0, printit, NULL);

    return 0;
}
