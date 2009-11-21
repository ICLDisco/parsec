/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "mpi.h"
#include "../dplasma.h"
#include "plasma.h"
#include "data_management.h"




int main(int argc, char ** argv){
    /* local variables*/

    int cores = 1;
    int nodes;
    int N;
    int LDA;
    int NRHS;
    int LDB;
    double eps;
    PLASMA_enum uplo;
    int info;
    int info_solution, info_factorization;
    int i,j;
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
    /* mpi init */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &main_desc.mpi_rank); 

    /* plasma initialization */
    PLASMA_Init(cores);
    
    /* parse arguments */
    if (main_desc.mpi_rank == 0)
        {
            if (argc != 6){
                printf(" Proper Usage is : ./%s  N LDA NRHS LDB GRIDrows with \n - N : the size of the matrix \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the RHS B\n - GRIDrows : number of processes row in the process grid (must divide the total number of processes  \n", (char*)argv[0]);
                exit(1);
            }
                        
            N     = atoi(argv[1]);
            LDA   = atoi(argv[2]);
            NRHS  = atoi(argv[3]);
            LDB   = atoi(argv[4]);
            main_desc.GRIDrows = atoi(argv[5]);

            if ((nodes % main_desc.GRIDrows) != 0 )
                {
                    printf("GRIDrows %d does not devide the total number of nodes %d\n", main_desc.GRIDrows, nodes);
                    exit(2);
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
            printf("generating matrix on rank 0\n");
            generate_matrix(N, A1, A2,  B1, B2,  WORK, D, LDA, NRHS, LDB);
            
            printf("tiling matrix\n");
            tiling(&uplo, N, A2, LDA, &descA);
            dplasma_desc_init(&descA, &main_desc);
            distribute_data(&descA, &main_desc, &requests);
        }
    else
        { /* prepare data for block reception  */
            /* initialize main tiles description structure (Bcast inside) */
            dplasma_desc_init(NULL, &main_desc);
            distribute_data(NULL, &main_desc, &requests);
        }
    
    /* distributing initial matrix */


    /* parsing jdf */

    /* checking local data ready */
    is_data_distributed(&main_desc, requests);
    
    /* start execution */
    PLASMA_Finalize();
    MPI_Finalize();
    return 0;
}



