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

int count = 0;

int
call_to_kernel_type_read( dague_execution_unit_t    *context,
                          dague_execution_context_t *this_task )
{
    (void)context; (void)this_task;
    int *data;

    dague_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data
                          );
    if( *data > 1 ) {
        dague_atomic_inc_32b(&count);
    }

    return 0;
}

int
call_to_kernel_type_write( dague_execution_unit_t    *context,
                           dague_execution_context_t *this_task )
{
    (void)context;
    int *data;

    dague_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data
                          );
    *data += 1;

    return 0;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;

    int ncores = 8, m, n;
    int no_of_tasks = 2;

    if(argv[1] != NULL){
        ncores = atoi(argv[1]);
    }
    if(argv[2] != NULL){
        no_of_tasks = atoi(argv[2]);
    }

    dague = dague_init(ncores, &argc, &argv);

    two_dim_block_cyclic_t ddescDATA;
    two_dim_block_cyclic_init( &ddescDATA, matrix_Integer, matrix_Tile, 1/*nodes*/, 0/*rank*/,
                                1, 1,/* tile_size*/ no_of_tasks, no_of_tasks,
                                /* Global matrix size*/ 0, 0, /* starting point */ no_of_tasks,
                                no_of_tasks, 1, 1, 1);

    ddescDATA.mat = calloc((size_t)ddescDATA.super.nb_local_tiles * (size_t) ddescDATA.super.bsiz,
                           (size_t) dague_datadist_getsizeoftype(ddescDATA.super.mtype));
    dague_ddesc_set_key ((dague_ddesc_t *)&ddescDATA, "ddescDATA");

    dague_ddesc_t *ddesc = &(ddescDATA.super.super);

    printf("---Starting--- \n");
    for( m = 0; m < no_of_tasks; m++ ) {
        for( n = 0; n < no_of_tasks; n++ ) {
            dague_data_copy_t *gdata = ddesc->data_of_key(ddesc, ddesc->data_key(ddesc, m, n))->device_copies[0];
            int *data = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *) gdata);
            printf("At index [%d, %d]:\t%d\n", m, n, *data);
        }
    }

    dague_dtd_init();

    two_dim_block_cyclic_t *__ddescDATA = &ddescDATA;

    TIME_START();

    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_handle_new (dague); /* 1 = arena_count */
    /* Registering the dtd_handle with DAGUE context */
    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);
#if defined (OVERLAP)
    dague_context_start(dague);
#endif

    int no_of_read_tasks = 5, k;

    for( m = 0; m < no_of_tasks; m++ ) {
        for( n = 0; n < no_of_tasks; n++ ) {
            dague_insert_task( DAGUE_dtd_handle, call_to_kernel_type_write,     "Write_Task",
                                   PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, DATA, m, n),   INOUT | REGION_FULL,
                                   0 );
            for( k = 0; k < no_of_read_tasks; k++ ) {
                dague_insert_task( DAGUE_dtd_handle, call_to_kernel_type_read,     "Read_Task",
                                       PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, DATA, m, n),   INPUT | REGION_FULL,
                                       0 );
            }
            dague_insert_task( DAGUE_dtd_handle, call_to_kernel_type_write,     "Write_Task",
                                   PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, DATA, m, n),   INOUT | REGION_FULL,
                                   0 );
        }
    }

    dague_dtd_handle_wait( dague, DAGUE_dtd_handle );
    dague_dtd_context_wait_on_handle( dague, DAGUE_dtd_handle );

    dague_dtd_handle_destruct(DAGUE_dtd_handle);

    TIME_STOP();

    printf("---Finally--- \n");
    for( m = 0; m < no_of_tasks; m++ ) {
        for( n = 0; n < no_of_tasks; n++ ) {
            dague_data_copy_t *gdata = ddesc->data_of_key(ddesc, ddesc->data_key(ddesc, m, n))->device_copies[0];
            int *data = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *) gdata);
            printf("At index [%d, %d]:\t%d\n", m, n, *data);
        }
    }

    printf("Total count of wrong read: %d\n", count);

    printf("Time Elapsed:\t");
    printf("\n%lf\n", time_elapsed);

    dague_data_free(ddescDATA.mat); ddescDATA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescDATA);

    dague_dtd_fini();
    dague_fini(&dague);

    return 0;
}
