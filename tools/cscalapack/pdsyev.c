
// /usr/local/bin/mpirun -np 8 ./scalapackChInv -p 2 -q 4 -n 8000 -nb 200

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "myscalapack.h"

int main(int argc, char **argv) {
    int iam, nprocs;
    int myrank_mpi, nprocs_mpi;
    int ictxt, nprow, npcol, myrow, mycol;
    int nb, n, mloc, nloc;
    int i, j, k, info_facto, info, iseed, verif;
    int my_info_facto;
    int descA[9];
    double *A=NULL, *W=NULL;
/**/
    double elapsed, GFLOPS;
    double my_elapsed;
/**/

    MPI_Init( &argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    n = 100; nprow = 1; npcol = 1; nb = 64; verif = 1;
    for( i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n      = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-p" ) == 0 ) {
            nprow  = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-q" ) == 0 ) {
            npcol  = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nb" ) == 0 ) {
            nb     = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-verif" ) == 0 ) {
            verif  = atoi(argv[i+1]);
            i++;
        }
    }

    /* no idea why I have problem with Cblacs on my computer, I am using blacsF77 interface here .... */
    blacs_pinfo_( &iam, &nprocs ) ;
    { int im1 = -1; int i0 = 0; blacs_get_( &im1, &i0, &ictxt ); }
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    { int i0=0; mloc = numroc_( &n, &nb, &myrow, &i0, &nprow ); }
    { int i0=0; nloc = numroc_( &n, &nb, &mycol, &i0, &npcol ); }

    { int i0=0; descinit_( descA, &n, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info ); }

    A = (double *)malloc(mloc*nloc*sizeof(double)) ;
    iseed = iam*mloc*nloc; srand(iseed);
    k = 0;
    for (i = 0; i < mloc; i++) {
        for (j = 0; j < nloc; j++) {
            A[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5 ;
            k++;    
        }
    }

    /* not that smart..., could used pdgeadd and pdlaset as well or pdmatgen */
    for (i = 1; i <= n; i++) {
        int idum1, idum2, iloc, jloc, i0=0;
        if ( ( myrow == indxg2p_( &i, &nb, &idum1, &i0, &nprow ) )
        &&   ( mycol == indxg2p_( &i, &nb, &idum1, &i0, &npcol ) ) ){
            iloc = indxg2l_( &i, &nb, &idum1, &idum2, &nprow );
            jloc = indxg2l_( &i, &nb, &idum1, &idum2, &npcol );
            A[ (jloc-1)*mloc + (iloc-1) ] += ((double) n);
        }
        
    }
    W = (double *)malloc(n*sizeof(double));

    my_elapsed =- MPI_Wtime();
    { int i1=1; double fwork; double *work; int lwork = -1;
      pdsyev_( "N", "L", &n, A, &i1, &i1, descA, W, NULL, NULL, NULL, NULL, &fwork, &lwork, &my_info_facto ); 
      lwork = (int)fwork;
      work = malloc(lwork * sizeof(double));
      pdsyev_( "N", "L", &n, A, &i1, &i1, descA, W, NULL, NULL, NULL, NULL, work, &lwork, &my_info_facto ); 
      free(work);
    }
    my_elapsed += MPI_Wtime();

    MPI_Allreduce( &my_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce( &my_info_facto, &info_facto, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    //GFLOPS = (((double) n)*((double) n)*((double) n))/1e+9/elapsed/3;
    GFLOPS = NAN;

    if ( iam == 0 ){
        printf("********************** N * NB * NP * P * Q *      T * Gflops * R *\n");
        printf("SCAL HEEV         % 6d % 4d % 4d % 3d % 3d % 8.2f % 8.2lf % 3d\n", n, nb, nprocs, nprow, npcol, elapsed, GFLOPS, info_facto);
    }

    free( A ); free(W);

    { int i0=0; blacs_gridexit_( &i0 ); }
    //{ int i0=0; blacs_exit_( &i0 ); } // OK, so that should be done, nevermind ...
    MPI_Finalize();
    return 0;
}
