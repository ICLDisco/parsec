#include "dague.h"
#include "cholesky.h"

static dague_ontask_iterate_t printit(struct dague_execution_unit_t *eu, 
                                      dague_execution_context_t *newcontext, 
                                      dague_execution_context_t *oldcontext, 
                                      int param_index, int outdep_index, 
                                      int rank_src, int rank_dst,
                                      void *param)
{
    (void)eu;
    (void)oldcontext;
    (void)param_index;
    (void)outdep_index;
    (void)rank_src;
    (void)rank_dst;
    (void)param;
    if( !strcmp(newcontext->function->name, "POTRF") ) {
        printf(" -> %s(%d)\n", newcontext->function->name, 
               newcontext->locals[0].value);
    }
    if( !strcmp(newcontext->function->name, "SYRK") ) {
        printf(" -> %s(%d, %d)\n", newcontext->function->name, 
               newcontext->locals[0].value, 
               newcontext->locals[1].value);
    }
    if( !strcmp(newcontext->function->name, "TRSM") ) {
        printf(" -> %s(%d, %d)\n", newcontext->function->name, 
               newcontext->locals[0].value, 
               newcontext->locals[1].value);
    }
    if( !strcmp(newcontext->function->name, "GEMM") ) {
        printf(" -> %s(%d, %d, %d)\n", newcontext->function->name, 
               newcontext->locals[0].value, 
               newcontext->locals[1].value, 
               newcontext->locals[2].value);
    }

    newcontext->function->iterate_successors(NULL, newcontext, printit, NULL);

    return DAGUE_ITERATE_CONTINUE;
}

int main(int argc, char *argv[])
{
    dague_object_t *o = dague_cholesky_new(NULL, 4, 10);
    dague_execution_context_t init;

    (void)argc;
    (void)argv;

    init.dague_object = o;
    init.function = (dague_t*)o->functions_array[3];
    init.locals[0].value = 0;

    printit(NULL, &init, NULL, -1, -1, -1, -1, NULL);
    
    o->functions_array[3]->iterate_successors(NULL, &init, printit, NULL);

    return 0;
}
