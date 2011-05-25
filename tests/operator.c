#include "dague.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

static int dague_operator_print_id( void* data, void* op_data, ... )
{
    va_list ap;
    int k, n;

    va_start(ap, op_data);
    k = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "tile %s(%d, %d) -> %p:%p\n", (char*)op_data, k, n, data, op_data );
    return 0;
}

int main( int argc, char* argv[] )
{
    dague_context_t* dague;
    struct dague_object_t* object;
    two_dim_block_cyclic_t ddescA;
    int cores = 1, world = 1, rank = 0;
    int mb = 120, nb = 120;
    int lm = 10000, ln = 10000;
    int rows = 1;

#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    dague = dague_init(cores, &argc, &argv);
    
    two_dim_block_cyclic_init( &ddescA, matrix_RealFloat,
                               world, cores, rank, mb, nb, lm, ln, 0, 0, lm, ln, 1, 1, rows );

    dague_ddesc_set_key(&ddescA.super, "A");
    object = dague_apply_operator_new((tiled_matrix_desc_t*)&ddescA,
                                      dague_operator_print_id,
                                      "A");
    dague_enqueue(dague, (dague_object_t*)object);

    dague_progress(dague);

    dague_apply_operator_destroy( object );

    dague_fini(&dague);

    return 0;
}
