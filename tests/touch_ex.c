#include "dague.h"
#include "arena.h" /* get rid of this */
#include "data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"

#define BLOCK 10
#define N     100

extern int touch_finalize(void);
extern dague_handle_t* touch_initialize(int block, int n);

int main( int argc, char** argv )
{
    dague_context_t* dague;
    dague_handle_t* handle;

    (void)argc; (void)argv;
    dague = dague_init(1, &argc, &argv);
    assert( NULL != dague );

    handle = touch_initialize(BLOCK, N);

    dague_enqueue( dague, handle );

    dague_context_wait(dague);

    dague_fini( &dague);

    touch_finalize();
    return 0;
}
