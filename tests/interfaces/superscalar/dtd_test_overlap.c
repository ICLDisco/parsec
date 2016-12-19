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
call_to_kernel_type( parsec_execution_unit_t    *context,
                     parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;
    return 0;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;

    int ncores = 8, m, n;
    int no_of_tasks = 20;

    if(argv[1] != NULL){
        ncores = atoi(argv[1]);
    }
    if(argv[2] != NULL){
        no_of_tasks = atoi(argv[2]);
    }

    parsec = parsec_init(ncores, &argc, &argv);

    two_dim_block_cyclic_t ddescDATA;
    two_dim_block_cyclic_init(&ddescDATA, matrix_Integer, matrix_Tile, 1/*nodes*/, 0/*rank*/, 1, 1,/* tile_size*/
                              no_of_tasks, no_of_tasks, /* Global matrix size*/ 0, 0, /* starting point */ no_of_tasks, no_of_tasks, 1, 1, 1);

    parsec_ddesc_set_key ((parsec_ddesc_t *)&ddescDATA, "ddescDATA");

    parsec_dtd_init();

    two_dim_block_cyclic_t *__ddescDATA = &ddescDATA;

    TIME_START();

    parsec_dtd_handle_t* PARSEC_dtd_handle = parsec_dtd_handle_new (parsec); /* 1 = arena_count */
    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue(parsec, (parsec_handle_t*) PARSEC_dtd_handle);
#if defined (OVERLAP)
    parsec_context_start(parsec);
#endif

    for( m = 0; m < no_of_tasks; m++ ) {
        for( n = 0; n < no_of_tasks; n++ ) {
            parsec_insert_task(PARSEC_dtd_handle, call_to_kernel_type,     "Test_Task",
                                     PASSED_BY_REF,    TILE_OF(PARSEC_dtd_handle, DATA, m, n),   INOUT | REGION_FULL,
                                     0);
        }
    }

    parsec_dtd_handle_wait( parsec, PARSEC_dtd_handle );
    parsec_dtd_context_wait_on_handle( parsec, PARSEC_dtd_handle );

    parsec_dtd_handle_destruct(PARSEC_dtd_handle);

    TIME_STOP();

    printf("Time Elapsed:\t");
    printf("\n%lf\n", time_elapsed);

    parsec_dtd_fini();
    parsec_fini(&parsec);

    return 0;
}
