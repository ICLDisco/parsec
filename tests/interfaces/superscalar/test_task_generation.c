#include "dague_config.h"
/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* dague things */
#include "dague.h"
#include "dague/profiling.h"

#ifdef DAGUE_VTRACE
#include "dague/vt_user.h"
#endif

#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"
#include "dplasma/testing/common_timing.h"

double time_elapsed = 0.0;

int
call_to_kernel(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    /* Does nothing */
    return 0;
}


int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int ncores = 8, kk, k, uplo = 1, info;
    int no_of_tasks = 8;
    int size = 1;

    if(argv[1] != NULL){
        no_of_tasks = atoi(argv[1]);
        if(argv[2] != NULL){
            ncores = atoi(argv[2]);
        }
    }

    int i;

    dague = dague_init(ncores, &argc, &argv);


    two_dim_block_cyclic_t ddescDATA;
    two_dim_block_cyclic_init(&ddescDATA, matrix_Integer, matrix_Tile, 1/*nodes*/, 0/*rank*/, 1, 1,/* tile_size*/
                              size, size, /* Global matrix size*/ 0, 0, /* starting point */ size, size, 1, 1, 1);

    ddescDATA.mat = calloc((size_t)ddescDATA.super.nb_local_tiles * (size_t) ddescDATA.super.bsiz,
                           (size_t) dague_datadist_getsizeoftype(ddescDATA.super.mtype));
    dague_ddesc_set_key ((dague_ddesc_t *)&ddescDATA, "ddescDATA");


    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (dague, 1); /* 4 = task_class_count, 1 = arena_count */

    two_dim_block_cyclic_t *__ddescDATA = &ddescDATA;
    dague_ddesc_t *ddesc = &(ddescDATA.super.super);


    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);


    TIME_START();

    int total = ddescDATA.super.mt;
    dague_context_start(dague);

    for(kk = 0; kk< no_of_tasks; kk++) {
        for( k = 0; k < total; k++ ) {
            insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel,     "Task",
                                     0);
        }
    }

    increment_task_counter(DAGUE_dtd_handle);
    dague_context_wait(dague);

    dague_fini(&dague);
    return 0;
}
