#ifndef _BT_reduction_h_
#define _BT_reduction_h_
#include <dague.h>
#include <dague/constants.h>
#include <data_distribution.h>
#include <data.h>
#include <debug.h>
#include <dague/ayudame.h>
#include <dague/devices/device.h>
#include <assert.h>

BEGIN_C_DECLS
#define DAGUE_BT_reduction_DEFAULT_ARENA    0
#define DAGUE_BT_reduction_ARENA_INDEX_MIN 1
    typedef struct dague_BT_reduction_handle {
    dague_handle_t super;
    /* The list of globals */
    struct tiled_matrix_desc_t *dataA /* data dataA */ ;
    int NB;
    int NT;
    /* The array of datatypes (DEFAULT and co.) */
    dague_arena_t **arenas;
    int arenas_size;
} dague_BT_reduction_handle_t;

extern dague_BT_reduction_handle_t *dague_BT_reduction_new(struct tiled_matrix_desc_t *dataA /* data dataA */ , int NB,
							   int NT);

END_C_DECLS
#endif /* _BT_reduction_h_ */
