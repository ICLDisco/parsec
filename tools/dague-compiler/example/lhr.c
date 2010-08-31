#include "dague.h"
#include "cholesky.h"
#include "data_dist/data_distribution.h"

static dague_ontask_iterate_t printit(struct dague_execution_unit_t *eu, 
                                      dague_execution_context_t *newcontext, 
                                      dague_execution_context_t *oldcontext, 
                                      int param_index, int outdep_index, 
                                      int rank_src, int rank_dst,
                                      void *param)
{
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

    newcontext->function->iterate_successors(eu, newcontext, printit, NULL);

    return DAGUE_ITERATE_CONTINUE;
}

static uint32_t rank_of( struct dague_ddesc* mat, ... )
{
    (void)mat;
    return 0;  /* Always me */
}

static void* data_of( struct dague_ddesc* mat, ... )
{
    (void)mat;
    return NULL;  /* Nothing relevant */
}

int main(int argc, char *argv[])
{
    dague_ddesc_t empty_data;
    dague_object_t *o;
    dague_execution_context_t init;
    dague_context_t *dague;

    (void)argc;
    (void)argv;

    dague = dague_init(1, &argc, &argv, 10);

    empty_data.myrank = 0;
    empty_data.cores = 1;
    empty_data.nodes = 1;
    empty_data.rank_of = &rank_of;
    empty_data.data_of = &data_of;
    o = (dague_object_t*)dague_cholesky_new(&empty_data, 10, 4, 1.0);

    init.dague_object = o;
    init.function = (dague_t*)o->functions_array[3];
    init.locals[0].value = 0;

    printit(dague->execution_units[0], &init, NULL, -1, -1, -1, -1, NULL);
    
    o->functions_array[3]->iterate_successors(dague->execution_units[0], &init, printit, NULL);

    return 0;
}
