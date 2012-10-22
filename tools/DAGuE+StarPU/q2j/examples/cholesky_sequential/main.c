#include <starpu.h>
#include "dague_config.h"
#include "dague.h"

#include <plasma.h>

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif


#include "dague_prof_grapher.h"
#include "schedulers.h"
#include "cuda.h"
#include "cholesky_data.h"
#include "cholesky_wrapper.h"
#include "cholesky.h"

int
main (int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores, nodes;
    int info;
    PLASMA_enum uplo = PlasmaLower;
    dague_ddesc_t *ddescA;
    dague_object_t *cholesky;
    dague_execution_context_t *startup_list = NULL;
    struct timeval start;
    struct timeval end;

    if(argc <1)
    {
	fprintf(stderr, "Error : Provide matrix rank\n");
	return 1;
    }
/*
    POTRF_COUNT = 0;
    TRSM_COUNT = 0;
    HERK_COUNT = 0;
    GEMM_COUNT = 0;
*/
#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    world = 1;
    rank = 0;
    cores = 1;
    

    /*
    fprintf(stdout, "PLASMA_Complex64_t : %lu\nDouble : %lu\n"
	            "cuDoubleComplex : %lu\nmagma_int_t : %lu\n", 
	            sizeof(PLASMA_Complex64_t), sizeof(double), sizeof(cuDoubleComplex), sizeof(magma_int_t));

    fprintf(stderr, "starpu_init...\n");
    */


    if(starpu_init(NULL) == -ENODEV)
      return 1;

/*    starpu_mpi_initialize_extended(&rank, &nodes);
      starpu_helper_cublas_init();*/

//    fprintf(stderr, "initialisation starpu done\n");
//    fflush(stderr);

    dague = dague_init(cores, 0, &argv);
//      fprintf(stderr, "initialisation dague done\n");

    
    
    dague_set_scheduler( dague, dague_schedulers_array[ 0 ] );


    ddescA = create_and_distribute_data(rank, world, cores, atoi(argv[1]), atoi(argv[2]));
//    fprintf(stderr, "create & distribute done\n");
  

    dague_ddesc_set_key(ddesc, "A");
      
    cholesky = cholesky_new(ddescA, BLOCKSIZE, matrix_rank/BLOCKSIZE, uplo, &info);

	
    dague_progress(dague);
  
    

    
    if( cholesky->nb_local_tasks > 0 ) {
        /* Update the number of pending dague choleskys */
        dague_atomic_inc_32b( &(dague->active_objects) );

        if( NULL != cholesky->startup_hook ) {
            cholesky->startup_hook(dague, cholesky, &startup_list);
//	    fprintf(stderr, "name : %s\n", startup_list->function->name);
            if( NULL != startup_list ) {
		/* We should add these tasks on the system queue */
		// old :               __dague_schedule( dague->execution_units[0], startup_list );
		gettimeofday(&start, NULL);
		generic_scheduling_func(dague->execution_units[0],(dague_list_item_t*) startup_list);
 
            }
        }
    }

    while((dague->active_objects != 0));
	
    starpu_task_wait_for_all();
    gettimeofday(&end, NULL);
    	

    double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

    double flop = (1.0f*(matrix_rank/BLOCKSIZE*BLOCKSIZE)*(matrix_rank/BLOCKSIZE*BLOCKSIZE)*(matrix_rank/BLOCKSIZE*BLOCKSIZE))/3.0f;


    fprintf(stdout, "---------------------\n"
	            "End of computations :\n"
	            "Matrix rank         : %d\n"
	            "Execution time (ms) : %5.2f\n"
	            "GFlop               : %2.2f\n"
	            "GFlop/s             : %2.2f\n"
         	    "---------------------\n",
	    (matrix_rank/BLOCKSIZE*BLOCKSIZE), timing/1000.0f, flop/1000000000.0f,(flop/timing)/1000.0f);
	    

    dague_fini(&dague);

    fflush(stderr);
/*
    fprintf(stderr, "POTRF : %d\n"
	            "TRSM  : %d\n"
	            "HERK  : %d\n"
	            "GEMM  : %d\n",
	    POTRF_COUNT,  TRSM_COUNT, HERK_COUNT, GEMM_COUNT);
*/


    free_data(ddescA);

//    starpu_helper_cublas_shutdown();
    starpu_shutdown();
    fflush(stderr);

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;

}
