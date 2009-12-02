/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "mpi.h"
#include <getopt.h>
#include "../../dplasma.h"
#include "plasma.h"
#include "data_management.h"




int main(int argc, char ** argv){
    /* local variables*/

    int cores = 1;
    int nodes;
    int N = 0;
    int LDA = 0;
    int NRHS = 1;
    int LDB = 0;
    double eps;
    PLASMA_enum uplo;
    int info;
    int info_solution, info_factorization;
    int NminusOne; /* = N-1;*/
    int LDBxNRHS; /* = LDB*NRHS;*/
    
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    double *WORK;
    double *D;

    PLASMA_desc descA;
    DPLASMA_desc main_desc;

    MPI_Request * requests;
    int req_count;

    struct option long_options[] =
        {
            {"lda", required_argument,  0, 'a'},
            {"matrix-size", required_argument, 0, 'n'},
            {"nrhs", required_argument,       0, 'r'},
            {"ldb",  required_argument,       0, 'b'},
            {"grid-rows",  required_argument, 0, 'g'},
            {"stile-size",  required_argument, 0, 's'},
            {"help",  no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

    int option_index = 0;
    int c;
    
    /* mpi init */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &main_desc.mpi_rank); 

    /* plasma initialization */
    PLASMA_Init(cores);
    
    /* parse arguments */
    if (main_desc.mpi_rank == 0)
        {
            main_desc.GRIDrows = 1;
            main_desc.nrst = 1;
            main_desc.ncst = 1;
            printf("parsing arguments\n");
            while (1)
                {
                    c = getopt_long (argc, argv, "a:n:r:b:g:s:h",
                                     long_options, &option_index);
         
                    /* Detect the end of the options. */
                    if (c == -1)
                        break;
                    
                    switch (c)
                        {
                        case 'a':
                            LDA = atoi(optarg);
                            printf("LDA set to %d\n", LDA);
                            break;
                            
                        case 'n':
                            N = atoi(optarg);
                            printf("matrix size set to %d\n", N);
                            break;
                            
                        case 'r':
                            NRHS  = atoi(optarg);
                            printf("number of RHS set to %d\n", NRHS);
                            break;
                            
                        case 'b':
                            LDB  = atoi(optarg);
                            printf("LDB set to %d\n", LDB);
                            break;
                            
                        case 'g':
                            main_desc.GRIDrows = atoi(optarg);
                            printf("%d rows od processes in the process grid\n", main_desc.GRIDrows);
                            break;
                        case 's':
                            main_desc.nrst = atoi(optarg);
                            main_desc.ncst = main_desc.nrst;
                            printf("processes receives tiles by blocks of %dx%d\n", main_desc.nrst, main_desc.ncst);
                            break;
                            
                        case '?': /* getopt_long already printed an error message. */
                        case 'h':
                        default:
                            printf("must provide : -n, --matrix-size : the size of the matrix \n Optional arguments are:\n -a --lda : leading dimension of the matrix A (equal matrix size by default) \n -r --nrhs : number of RHS (default: 1) \n -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1) \n -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n");
                            MPI_Abort( MPI_COMM_WORLD, 2);
                        }
                    
                }
            
            if (N == 0)
                {
                    printf("must provide : -n, --matrix-size : the size of the matrix \n Optional arguments are:\n -a --lda : leading dimension of the matrix A (equal matrix size by default) \n -r --nrhs : number of RHS (default: 1) \n -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1) \n -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n");
                    MPI_Abort( MPI_COMM_WORLD, 2 );

                }
            if(LDA <= 0)
                LDA = N;
            if (LDB <= 0)
                LDB = N;
            
            if (main_desc.ncst <= 0)
                {
                    printf("select a positive value for super tile size\n");
                    MPI_Abort( MPI_COMM_WORLD, 2 );
                }
            
            if ((nodes % main_desc.GRIDrows) != 0 )
                {
                    printf("GRIDrows %d does not devide the total number of nodes %d\n", main_desc.GRIDrows, nodes);
                    MPI_Abort( MPI_COMM_WORLD, 2 );
                }
            
            main_desc.GRIDcols = nodes / main_desc.GRIDrows ;
            
            A1   = (double *)malloc(LDA*N*sizeof(double));
            A2   = (double *)malloc(LDA*N*sizeof(double));
            B1   = (double *)malloc(LDB*NRHS*sizeof(double));
            B2   = (double *)malloc(LDB*NRHS*sizeof(double));
            WORK = (double *)malloc(2*LDA*sizeof(double));
            D    = (double *)malloc(LDA*sizeof(double));
            
            NminusOne = N-1;
            LDBxNRHS = LDB*NRHS;
           
            /* generating a random matrix */
            //printf("generating matrix on rank 0\n");
            generate_matrix(N, A1, A2,  B1, B2,  WORK, D, LDA, NRHS, LDB);
            
            // printf("tiling matrix\n");
            tiling(&uplo, N, A2, LDA, &descA);
            //printf("structure initialization\n");
            dplasma_desc_init(&descA, &main_desc);
            printf("Data distribution\n");
            distribute_data(&descA, &main_desc, &requests, &req_count);
        }
    else
        { /* prepare data for block reception  */
            /* initialize main tiles description structure (Bcast inside) */
            dplasma_desc_init(NULL, &main_desc);
            distribute_data(NULL, &main_desc, &requests, &req_count);
        }
    
    /* distributing initial matrix */


    /* parsing jdf */

    /* checking local data ready */
    is_data_distributed(&main_desc, requests, req_count);
    data_dist_verif(&descA, &main_desc);
    /* start execution */
    PLASMA_Finalize();
    MPI_Finalize();
    return 0;
}



