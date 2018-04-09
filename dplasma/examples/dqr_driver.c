/*
 * Copyright (c) 2016-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *  Compile with $CC $(pkg-config --cflags dplasma) -c dqr_driver.c
 *  Link with $CC $(pkg-config --libs dplasma) -o dqr_driver dqr_driver.o
 *
 */
#include "dplasma.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <getopt.h>
#include <math.h>
#include <sys/time.h>

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world;
    int M = 0, N = 0;
    int mb = 200, nb = 200;
    int ib = 32;
    int P = 0;
    int ch;
    int cores = -1;
    two_dim_block_cyclic_t dcA;
    two_dim_block_cyclic_t dcWork;

    static struct option longopts[] = {
        { "M",      required_argument,            NULL,           'M' },
        { "N",      required_argument,            NULL,           'N' },
        { "mb",     required_argument,            NULL,           'm' },
        { "nb",     required_argument,            NULL,           'm' },
        { "P",      required_argument,            NULL,           'P' },
        { "cores",  required_argument,            NULL,           'c' },
        { "help",         no_argument,            NULL,           'h' },
        {  NULL,                    0,            NULL,           0 }
     };

    /** If compiled in, setup MPI */
#ifdef PARSEC_HAVE_MPI
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

     while ((ch = getopt_long(argc, argv, "M:N:m:n:P:c:h", longopts, NULL)) != -1)
             switch (ch) {
             case 'M':
                 M = atoi(optarg);
                 break;
             case 'N':
                 N = atoi(optarg);
                 break;
             case 'm':
                 mb = atoi(optarg);
                 break;
             case 'n':
                 nb = atoi(optarg);
                 break;
             case 'P':
                 P = atoi(optarg);
                 break;
             case 'c':
                 cores = atoi(optarg);
                 break;
             case 'h':
             default:
                 goto usage;
     }
     argc -= optind;
     argv += optind;

     /** Sanity check: the matrix must be defined */
     if( 0 == N ) {
         if( 0 == rank )
             fprintf(stderr, "Undefined matrix size\n");
         goto usage;
     } else {
         if( 0 == M ) {
             M = N;
         }
     }

     /** Sanity check: the process grid must be defined */
     if( 0 == P ) {
         P = (int)floor(sqrt(world));
     }
     if( (world % P) != 0 ) {
         if( 0 == rank )
             fprintf(stderr, "P = %d does not divide the world (%d): cannot define a process grid\n",
                     P, world);
        goto usage;
    }


    /** Initialize PaRSEC with the required number of cores,
     *  and pass all arguments after '--' to PaRSEC, if there are some */
    parsec = parsec_init(cores, &argc, &argv);

    /** Declare a Matrix A as a (4, 1)-2D-Tile cyclic matrix of size M x N real
     *  doubles, tiled in tiles of mb x nb, and distributed over PxQ processes,
     *  where Q = world / P
     *  Let P = 2, world = 6 (hence Q = 2): tiles will be distributed as follows
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *   | 1 | 3 | 5 | 1 | 3 |
     *   +---+---+---+---+---+
     *   | 1 | 3 | 5 | 1 | 3 |
     *   +---+---+---+---+---+
     *   | 1 | 3 | 5 | 1 | 3 |
     *   +---+---+---+---+---+
     *   | 1 | 3 | 5 | 1 | 3 |
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *   | 0 | 2 | 4 | 0 | 2 |
     *   +---+---+---+---+---+
     *  See $DPLASMA_INSTALL_DIR/include/data_dist/matrix/two_dim_rectangle_cyclic.h
     *  for documentation on all parameters.
     */
    two_dim_block_cyclic_init(&dcA, matrix_RealDouble, matrix_Tile,
                              world, rank, mb, nb, M, N, 0, 0,
                              M, N, 4, 1, P);
    /** Give a name to this matrix, for debugging and tracing purposes */
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "A");
    /** Allocate memory for the local view of the matrix.
     *  Equivalent to malloc( (M * N) / (P * Q) * sizeof(double))
     *  but manages limits at the borders
     */
    dcA.mat = malloc((size_t)dcA.super.nb_local_tiles *
                        (size_t)dcA.super.bsiz *
                        (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));

    /** Declare matrix Work, which serves as workspace memory for the QR
     *  factorization (see http://www.netlib.org/lapack/explore-3.1.1-html/dgeqrf.f.html)
     *  That matrix is distributed over the same process grid as A.
     */
    two_dim_block_cyclic_init(&dcWork, matrix_RealDouble, matrix_Tile,
                              world, rank, ib, nb, dcA.super.lmt*ib, N, 0, 0,
                              dcA.super.lmt*ib, N, 4, 1, P);
    /** Give a name to this matrix, for debugging and tracing purposes */
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcWork, "Work");
    /** Allocate memory for the workspace */
    dcWork.mat = malloc((size_t)dcWork.super.nb_local_tiles *
                           (size_t)dcWork.super.bsiz *
                           (size_t)parsec_datadist_getsizeoftype(dcWork.super.mtype));

    /* Initialize the matrix A with random values, using the
     * blocking DPLASMA interface
     * See $DPLASMA_INSTALL_DIR/include/dplasma.h for all possible
     * functions
     */
    dplasma_dplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, 3872);

    /** Perform the QR operation using the non-blocking interface
     *  Measure the time taken, reduce the max, and compute the
     *  performance in GFlop/s
     */
    {
        struct timeval start, end, diff;
        double my_duration, duration;

#if defined(PARSEC_HAVE_MPI)
        /** Simple synchronization of timing */
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&start, NULL);

        /** Create a handle for the QR operation on A and Work */
        parsec_taskpool_t* PARSEC_dgeqrf = dplasma_dgeqrf_New( (parsec_tiled_matrix_dc_t*)&dcA,
                                                           (parsec_tiled_matrix_dc_t*)&dcWork );

        /** Schedule the QR operation */
        parsec_enqueue(parsec, PARSEC_dgeqrf);

        /** Allows PaRSEC threads to start computing */
        parsec_context_start(parsec);

        /** Wait for the completion of all scheduled operations */
        parsec_context_wait(parsec);

        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);

        my_duration = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
#if defined(PARSEC_HAVE_MPI)
        MPI_Allreduce(&my_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
        duration = my_duration;
#endif

        if( rank == 0 ) {
            double gflops =
                ((((M > N) ?
                  ((double)N * ((double)N * (  0.5-(1./3.) * (double)N + (double)M) +    (double)M + 23. / 6.)) :
                  ((double)M * ((double)M * ( -0.5-(1./3.) * (double)M + (double)N) + 2.*(double)N + 23. / 6.)) )) +
                (((M > N) ?
                  ((double)N * ((double)N * (  0.5-(1./3.) * (double)N + (double)M)                    +  5. / 6.)) :
                  ((double)M * ((double)M * ( -0.5-(1./3.) * (double)M + (double)N) +    (double)N +  5. / 6.)) )))/1e9;
            printf("Computed %g GFlops in %g seconds: Performance is %g GFlop/s\n",
                   gflops, duration, gflops/duration);
        }
    }

    /** Free matrices allocated for the operation */
    free(dcA.mat);
    free(dcWork.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcWork);

    /** Release resources used by the PaRSEC engine */
    parsec_fini(&parsec);

#if defined(PARSEC_HAVE_MPI)
    /** Finalize MPI */
    MPI_Finalize();
#endif

    return 0;

  usage:
    if( rank == 0 ) {
        fprintf(stderr,
                "Usage: %s -M M [ -N N -m mb -n nb -c cores | -h ]\n"
                " Compute the QR factorization of a random matrix, and displays the performance\n"
                " Where:\n"
                "   --M | -M              Set the number of rows in the matrix (required)\n"
                "   --N | -N              Set the number of columns in the matrix (default: same value is M)\n"
                "   --mb | -m             Set the number of rows in a tile (default: %d)\n"
                "   --nb | -n             Set the number of columns in a tile (default: %d)\n"
                "   --P | -P              Set the number of process rows in the process grid (default: sqrt(number of nodes))\n"
                "   --cores | -c          Set the number of cores per node to use (default: all of them)\n",
                argv[1],
                mb, nb);
    }
    MPI_Finalize();
    exit(1);
}
