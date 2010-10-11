#include "dague.h"
#include "scheduling.h"
#include "profiling.h"

#ifdef USE_MPI
#include "remote_dep.h"
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <lapack.h>

#include "TSQR.h"

/* globals and argv set values */
int nbtasks = -1;
int rank = 0, count = 1;

static dague_object_t         *dague_QR = NULL;
static two_dim_block_cyclic_t  rtop;
#ifdef USE_MPI
MPI_Datatype RTILE_T;
#endif

static dague_context_t *setup_tsqr( int* pargc
                                    , char** pargv[] )
{
    dague_context_t   *dague;
    int totalCore = 1;
    int treeHeight = 4;

#ifdef USE_MPI
    MPI_Init( NULL, NULL );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &count );
    if ( rank >= count )
    {
        fprintf( stderr, "%i> /!\\ %i is shutting down, not in good partition size\n"
               , rank, count );
        return NULL;
    }
#endif  /* USE_MPI */
    sscanf( (*pargv)[1], "%i", &totalCore);
    sscanf( (*pargv)[2], "%i", &treeHeight );

    PLASMA_Init( 1 );
    printf( "%i> Launching TSQR cores:%i treeHeight:%i\n"
          , rank
          , totalCore, treeHeight );

    int powerCount = 255;
    int height = powerCount * (1 << treeHeight);

    dague = dague_init( totalCore
                      , pargc
                      , pargv
                      , powerCount*powerCount*sizeof(double)/*NB*/);

    printf( "%i> procCount:%i powerCount:%i height:%i\n"
          , rank
          , count
          , powerCount
          , height
          );

    two_dim_block_cyclic_init
            ( &rtop
            , matrix_RealDouble
            , count
            , totalCore // 8 pour dancer (int cores)
            , rank // int myrank
            , powerCount // int mb
            , powerCount // int nb
            , 0 // innerBlocking
                    // 1/5
                    // 0 si n'a pas de inner blocking
            , height        // lm == m
            , powerCount    // ln == n
            , 0          // int i (in whole grand scheme)
            , 0          // int j
            , height     // m
            , powerCount // n
            , height / (powerCount * count) // int nrst
            , 1 // int ncst
            , count // int process_GridRows
            );

    generate_tiled_random_mat((tiled_matrix_desc_t *)&rtop, 100);

    dague_QR = (dague_object_t*)
        dague_TSQR_new( (dague_ddesc_t*)&rtop
                      , treeHeight // info->process2Power
                      );
    dague_enqueue( dague, (dague_object_t*)dague_QR);

    printf( "%i> Task count:%u\n"
          , rank
          , dague->taskstodo );
    return dague;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;

    /*** THIS IS THE DPLASMA COMPUTATION ***/
    dague = setup_tsqr(&argc, &argv);
    printf( "%i> Progress!\n", rank );


    double qrBegin = 0; /*MPI_Wtime();*/
    dague_progress(dague);
    double qrEnd = 1; /*MPI_Wtime();*/

    printf("%i> FullProgress:%f\n", rank, qrEnd - qrBegin );

    /*** END OF DPLASMA COMPUTATION ***/
    dague_fini( &dague );
    PLASMA_Finalize();
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif  /* USE_MPI */

    return 0;
}
