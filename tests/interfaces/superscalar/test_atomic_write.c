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
    (void)context;
    dague_data_copy_t *gDATA;

    dague_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &gDATA
                          );

    uint32_t *data = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *) gDATA);

    dague_atomic_inc_32b(data);

    return 0;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int ncores = 32, kk, k;
    int no_of_tasks = 8;
    int size = 1;

    if(argv[1] != NULL){
        no_of_tasks = atoi(argv[1]);
        if(argv[2] != NULL){
            size = atoi(argv[2]);
        }
    }

    dague = dague_init(ncores, &argc, &argv);

    two_dim_block_cyclic_t ddescDATA;
    two_dim_block_cyclic_init(&ddescDATA, matrix_Integer, matrix_Tile, 1/*nodes*/, 0/*rank*/, 1, 1,/* tile_size*/
                              size, size, /* Global matrix size*/ 0, 0, /* starting point */ size, size, 1, 1, 1);

    ddescDATA.mat = calloc((size_t)ddescDATA.super.nb_local_tiles * (size_t) ddescDATA.super.bsiz,
                           (size_t) dague_datadist_getsizeoftype(ddescDATA.super.mtype));
    dague_ddesc_set_key ((dague_ddesc_t *)&ddescDATA, "ddescDATA");


    dague_dtd_init();
    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_handle_new (dague);

    two_dim_block_cyclic_t *__ddescDATA = &ddescDATA;


    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);


    TIME_START();

    int total = ddescDATA.super.mt;

    dague_context_start(dague);

    for(kk = 0; kk< no_of_tasks; kk++) {
        for( k = 0; k < total; k++ ) {
            insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel,     "Task",
                                     PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, DATA, k, k),   ATOMIC_WRITE | REGION_FULL,
                                     0);
        }
    }

    dague_dtd_handle_wait( dague, DAGUE_dtd_handle );
    dague_dtd_handle_destruct(DAGUE_dtd_handle);

    /*printf("Finally \n");
     for (k = 0; k < total; k++){
     dague_data_copy_t *gdata = ddesc->data_of_key(ddesc, ddesc->data_key(ddesc,k,k))->device_copies[0];
     int *data = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *) gdata);
     printf("At index %d:\t%d\n", k, *data);
     } */

    TIME_STOP();

    printf("Time Elapsed:\t");
    printf("\n%lf\n",time_elapsed);

    dague_dtd_fini();
    dague_fini(&dague);
    return 0;
}
