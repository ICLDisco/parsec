#include "dague_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* dague things */
#include "dague.h"
#include "scheduling.h"
#include "profiling.h"
//#include "common_timing.h"
#ifdef DAGUE_VTRACE
#include "vt_user.h"
#endif



#include "dague/interfaces/superscalar/insert_function_internal.h"
#include "dplasma/testing/common_timing.h"

double time_elapsed = 0.0;

int
call_to_kernel(dague_execution_context_t * this_task)
{
    //printf("Executing Task\n");
    fflush(stdout); 
    return 0;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int ncores = 8, k, uplo, info;
    int no_of_tasks = 100000;
    
    dague = dague_init(ncores, &argc, &argv);

    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (4, 1, &info); /* 4 = task_class_count, 1 = arena_count */


    TIME_START();
    
    for(k=0;k<no_of_tasks;k++){
        insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel, "Task",
                                 sizeof(int),      &uplo,              VALUE,
                                 PASSED_BY_REF,    NULL, INOUT, DEFAULT,
                                 0);
    } 


    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);
    dague_progress(dague);

    printf("Time Elapsed:\t");
    printf("\n%lf\n",no_of_tasks/time_elapsed);    

    dague_fini(&dague);
    return 0;
}
