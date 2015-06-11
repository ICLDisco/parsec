#include "dague_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* dague things */
#include "dague.h"
#include "dague/scheduling.h"
#include "dague/profiling.h"
//#include "common_timing.h"
#ifdef DAGUE_VTRACE
#include "dague/vt_user.h"
#endif



#include "dague/interfaces/superscalar/insert_function_internal.h"
#include "dplasma/testing/common_timing.h"

double time_elapsed = 0.0;

int
call_to_kernel(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{   
    int *uplo, *uplo2, *uplo1;
    double *uplo3;
    dague_data_copy_t *data;

     dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &uplo2,
                          UNPACK_VALUE, &uplo3,
                          UNPACK_DATA,  &data,
                          UNPACK_VALUE, &uplo1 
                        );

    printf("Executing Task: %d\n",((dtd_task_t *)this_task)->task_id+1);
    printf("Parameter 1: %d\n", *uplo);
    printf("Parameter 2: %d\n", *uplo2);
    printf("Parameter 3: %d\n", *uplo1);
    printf("Parameter 4: %p\n", data);
    printf("Parameter 5: %lf\n", *uplo3);
    return 0;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int ncores = 8, k, uplo = 1, info;
    int no_of_tasks = 52;

    if(argv[1] != NULL){
        no_of_tasks = atoi(argv[1]);
    }

    int uplo1=2, uplo2=3;
    double uplo3 = 88.56;
    dague_ddesc_t *ddesc = malloc(sizeof(dague_ddesc_t));

    dague = dague_init(ncores, &argc, &argv);

    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (dague, 4, 1, &info); /* 4 = task_class_count, 1 = arena_count */

    TIME_START();

    for( k = 0; k < no_of_tasks; k++ ) {
        insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel,     "Task",
                                 sizeof(int),      &uplo,               VALUE,
                                 sizeof(int),      &uplo2,              VALUE,
                                 sizeof(double),   NULL,                SCRATCH,
                                 PASSED_BY_REF,    ddesc,               INOUT | REGION_FULL,
                                 sizeof(int),      &uplo1,              VALUE,
                                 0);
    }

    printf("Test: %d\n",dague_context_test(dague));
    printf("Local_task_count: %d\n",DAGUE_dtd_handle->super.nb_local_tasks );
    increment_task_counter(DAGUE_dtd_handle); 
    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);
    dague_context_wait(dague);  

    printf("Time Elapsed:\t");
    printf("\n%lf\n",no_of_tasks/time_elapsed);
    
    dague_fini(&dague);
    return 0;
}
