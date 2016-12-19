#include "parsec_config.h"
/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"

#ifdef PARSEC_VTRACE
#include "parsec/vt_user.h"
#endif

#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "dplasma/testing/common_timing.h"

double time_elapsed = 0.0;

int
call_to_kernel(parsec_execution_unit_t *context, parsec_execution_context_t *this_task)
{
    (void)context; (void)this_task;
    /* Does nothing */
    return 0;
}


int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int ncores = 8, kk, k;
    int no_of_tasks = 8;
    int size = 1;

    if(argv[1] != NULL){
        no_of_tasks = atoi(argv[1]);
        if(argv[2] != NULL){
            ncores = atoi(argv[2]);
        }
    }

    parsec = parsec_init(ncores, &argc, &argv);
    two_dim_block_cyclic_t ddescDATA;
    two_dim_block_cyclic_init(&ddescDATA, matrix_Integer, matrix_Tile, 1/*nodes*/, 0/*rank*/, 1, 1,/* tile_size*/
                              size, size, /* Global matrix size*/ 0, 0, /* starting point */ size, size, 1, 1, 1);

    ddescDATA.mat = calloc((size_t)ddescDATA.super.nb_local_tiles * (size_t) ddescDATA.super.bsiz,
                           (size_t) parsec_datadist_getsizeoftype(ddescDATA.super.mtype));
    parsec_ddesc_set_key ((parsec_ddesc_t *)&ddescDATA, "ddescDATA");

    parsec_dtd_init();
    parsec_dtd_handle_t* PARSEC_dtd_handle = parsec_dtd_handle_new (parsec);

    parsec_enqueue(parsec, (parsec_handle_t*) PARSEC_dtd_handle);


    TIME_START();

    int total = ddescDATA.super.mt;
    parsec_context_start(parsec);

    for(kk = 0; kk< no_of_tasks; kk++) {
        for( k = 0; k < total; k++ ) {
            parsec_insert_task(PARSEC_dtd_handle, call_to_kernel,     "Task",
                                     0);
        }
    }

    parsec_dtd_handle_wait( parsec, PARSEC_dtd_handle );
    parsec_dtd_handle_destruct(PARSEC_dtd_handle);

    parsec_dtd_fini();
    parsec_fini(&parsec);
    return 0;
}
