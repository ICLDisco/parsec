#include "dague.h"
#include "arena.h" /* get rid of this */
#include "data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"

#define BLOCK 10
#define N     100

extern int touch_finalize(void);
extern dague_object_t* touch_initialize(int block, int n);

int main( int argc, char** argv )
{
    dague_context_t* dague;
    dague_object_t* object;

    (void)argc; (void)argv;
    dague = dague_init(1, NULL, NULL);
    assert( NULL != dague );

    object = touch_initialize(BLOCK, N);

    dague_enqueue( dague, (dague_object_t*)object );

    dague_progress(dague);

    dague_fini( &dague);

    touch_finalize();
    return 0;
}
