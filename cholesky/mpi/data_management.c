/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cblas.h"
#include <math.h>

#include "plasma.h"
#include "../src/common.h"
#include "../src/lapack.h"
#include "../src/context.h"
#include "../src/allocate.h"
#include "mpi.h"
#include "data_management.h"

//#define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
static void * plasma_A(PLASMA_desc * Pdesc, int m, int n)
{
    return &((double*)Pdesc->mat)[Pdesc->bsiz*(m)+Pdesc->bsiz*Pdesc->lmt*(n)];

}


int dplasma_desc_init(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc)
{

    int * tmp_ints;
    int i, j;
    int nbt;
    int nb_elem_c;
    int nb_elem_r;

        
    tmp_ints = malloc(sizeof(int)*16);
    if (tmp_ints == NULL)
        {
            printf("memory allocation failed\n");
            exit(2);
        }
    if (Ddesc->mpi_rank == 0) /* send data */
        {
            Ddesc->dtyp= tmp_ints[0] = Pdesc->dtyp;
            Ddesc->mb = tmp_ints[1] = Pdesc->mb;
            Ddesc->nb = tmp_ints[2] = Pdesc->nb;
            Ddesc->bsiz  = tmp_ints[3] = Pdesc->bsiz ;
            Ddesc->lm  = tmp_ints[4] = Pdesc->lm ;
            Ddesc->ln  = tmp_ints[5] = Pdesc->ln ;
            Ddesc->lmt  = tmp_ints[6] = Pdesc->lmt ;
            Ddesc->lnt  = tmp_ints[7] = Pdesc->lnt ;
            Ddesc->i = tmp_ints[8] = Pdesc->i ;
            Ddesc->j = tmp_ints[9] = Pdesc->j ;
            Ddesc->m = tmp_ints[10] = Pdesc->m  ;
            Ddesc->n = tmp_ints[11] = Pdesc->n ;
            Ddesc->mt  = tmp_ints[12] = Pdesc->mt ;
            Ddesc->nt  = tmp_ints[13] = Pdesc->nt ;
            tmp_ints[14] = Ddesc->GRIDrows ;
            tmp_ints[15] = Ddesc->GRIDcols ;

            MPI_Bcast(tmp_ints, 16, MPI_INT, 0, MPI_COMM_WORLD);

        }
    else /* rank != 0, receive data */
        {
            MPI_Bcast(tmp_ints, 16, MPI_INT, 0, MPI_COMM_WORLD);

            Ddesc->dtyp= tmp_ints[0];
            Ddesc->mb = tmp_ints[1];
            Ddesc->nb = tmp_ints[2];
            Ddesc->bsiz  = tmp_ints[3];
            Ddesc->lm  = tmp_ints[4];
            Ddesc->ln  = tmp_ints[5];
            Ddesc->lmt  = tmp_ints[6];
            Ddesc->lnt  = tmp_ints[7];
            Ddesc->i = tmp_ints[8];
            Ddesc->j = tmp_ints[9];
            Ddesc->m = tmp_ints[10];
            Ddesc->n = tmp_ints[11];
            Ddesc->mt  = tmp_ints[12];
            Ddesc->nt  = tmp_ints[13];
            Ddesc->GRIDrows  = tmp_ints[14];
            Ddesc->GRIDcols  = tmp_ints[15];
        }
    free(tmp_ints);

    
    /* computing colRANK and rowRANK */
    Ddesc->colRANK = 0;
    Ddesc->rowRANK = 0;
    i = Ddesc->mpi_rank;

    /* find rowRANK */
    while ( i >= Ddesc->GRIDcols)
        {
            Ddesc->rowRANK = Ddesc->rowRANK + 1;
            i = i - Ddesc->GRIDcols;
        }
    /* affect colRANK */
    Ddesc->colRANK = i;

    /* allocate memory for tiles data */
    nbt = Ddesc->lmt * Ddesc->lnt; /* total number of tiles */
    if ( Ddesc->GRIDrows > Ddesc->lmt || Ddesc->GRIDcols > Ddesc->lnt)
        {
            printf("The process gris choosen is %dx%d, tiling is %d, %d\n", Ddesc->GRIDrows, Ddesc->GRIDcols, Ddesc->lmt, Ddesc->lnt);
            exit(2);
        }
    /* find the number of tiles this process will handle */
    nb_elem_r = Ddesc->lmt / Ddesc->GRIDrows;
    j = Ddesc->lmt % Ddesc->GRIDrows;
    if (Ddesc->rowRANK < j)
        {
            nb_elem_r++;
        }
    nb_elem_c = Ddesc->lnt / Ddesc->GRIDcols;
    j =  Ddesc->lnt % Ddesc->GRIDcols;
    if (Ddesc->colRANK < j)
        {
            nb_elem_c++;
        }
    
    Ddesc->mat = malloc(sizeof(double) * nb_elem_c * nb_elem_r * Ddesc->bsiz);
    //    printf("This is process rank %d (%d,%d) in a Process grid %dx%d, will handle %dx%d tiles\n", Ddesc->mpi_rank, Ddesc->rowRANK, Ddesc->colRANK, Ddesc->GRIDrows, Ddesc->GRIDcols, nb_elem_c, nb_elem_r);
    return 0;
}



int generate_matrix(int N, double * A1, double * A2, double * B1, double * B2, double * WORK, double * D,int LDA, int NRHS, int LDB)
{
    
    int i, j;
    int IONE=1;
    int info;
    int ISEED[4] = {0,0,0,1};   /* initial seed for dlarnv() */
    int LDBxNRHS = LDB*NRHS;
    
    int NminusOne = N-1;
    /* Initialize A1 and A2 for Symmetric Positive Matrix */
    dlarnv(&IONE, ISEED, &LDA, D);
    dlagsy(&N, &NminusOne, D, A1, &LDA, ISEED, WORK, &info);
    for ( i = 0; i < N; i++)
        for (  j = 0; j < N; j++)
            A2[LDA*j+i] = A1[LDA*j+i];
    
    for ( i = 0; i < N; i++){
        A1[LDA*i+i] = A1[LDA*i+i]+ N ;
        A2[LDA*i+i] = A1[LDA*i+i];
    }
    
    /* Initialize B1 and B2 */
    dlarnv(&IONE, ISEED, &LDBxNRHS, B1);
    for ( i = 0; i < N; i++)
        for ( j = 0; j < NRHS; j++)
            B2[LDB*j+i] = B1[LDB*j+i];
    return 0;
}


int tiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA)
{
    int NB, NT;
    int status;
    double *Abdl;
    plasma_context_t *plasma;

    /* Plasma routines */
    *uplo=PlasmaLower;
    
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (*uplo != PlasmaUpper && *uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf", "illegal value of uplo");
        return -1;
    }
    if (N < 0) {
        plasma_error("PLASMA_dpotrf", "illegal value of N");
        return -2;
    }
    if (LDA < max(1, N)) {
        plasma_error("PLASMA_dpotrf", "illegal value of LDA");
        return -4;
    }
    /* Quick return */
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;

    /* Tune NB depending on M, N & NRHS; Set NBNBSIZE */
    status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
        return status;
    }

    /* Set NT */
    NB = PLASMA_NB;
    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    /* Allocate memory for matrices in block layout */
    Abdl = (double *)plasma_shared_alloc(plasma, NT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble);
    if (Abdl == NULL) {
        plasma_error("PLASMA_dpotrf", "plasma_shared_alloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }

    /*PLASMA_desc*/ *descA = plasma_desc_init(
                                             Abdl, PlasmaRealDouble,
                                             PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE,
                                             N, N, 0, 0, N, N);

    plasma_parallel_call_3(plasma_lapack_to_tile,
                           double*, A,
                           int, LDA,
                           PLASMA_desc, *descA);

    printf("matrix tiled in %dx%d\n", descA->lmt, descA->lnt);
    return 0;
    
}


int dplasma_get_rank_for_tile(DPLASMA_desc * Ddesc, int m, int n)
{
    int cr;
    int rr;
    int res;

    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    
    rr = m % Ddesc->GRIDrows;
    cr = n % Ddesc->GRIDcols;
    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * Ddesc->GRIDcols + cr;
    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
/*            m, n, res, rr, cr, Ddesc->GRIDrows, Ddesc->GRIDcols); */
    return res;
}

void * dplasma_get_tile(DPLASMA_desc * Ddesc, int m, int n)
{
    int res;
    int nb_elem_r;
    int j;
    if (Ddesc->mpi_rank != dplasma_get_rank_for_tile(Ddesc, m, n))
        {
            printf("Tile (%d,%d) does not belong to %d\n", m, n ,Ddesc->mpi_rank);
            return NULL;
        }


    /* find the number of tiles this process will handle */
    nb_elem_r = Ddesc->lmt / Ddesc->GRIDrows;
    j = Ddesc->lmt % Ddesc->GRIDrows;
    if (Ddesc->rowRANK < j)
        {
            nb_elem_r++;
        }
    
    res = ((n / Ddesc->GRIDcols) * nb_elem_r * Ddesc->bsiz) +
        ((m / Ddesc->GRIDrows) * Ddesc->bsiz) ;
    return &(((double *) Ddesc->mat)[res]);
}



int dplasma_set_tile(DPLASMA_desc * Ddesc, int m, int n, void * buff)
{
    int res;
    int nb_elem_r;
    int j;
    if (Ddesc->mpi_rank != dplasma_get_rank_for_tile(Ddesc, m, n))
        {
            return -1;
        }
    
    
    /* find the number of tiles this process will handle */
    nb_elem_r = Ddesc->lmt / Ddesc->GRIDrows;
    j = Ddesc->lmt % Ddesc->GRIDrows;
    if (Ddesc->rowRANK < j)
        {
            nb_elem_r++;
        }
    
    res = ((n / Ddesc->GRIDcols) * nb_elem_r * Ddesc->bsiz) +
        ((m / Ddesc->GRIDrows) * Ddesc->bsiz) ;
    memcpy(&(((double *)Ddesc->mat)[res]), buff, Ddesc->bsiz*sizeof(double));
    
    return 0;
}




int distribute_data(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc, MPI_Request ** reqs)
{
    int i, j, k, pos, rank;
    int nb_reqs;
    int nb_elem_r, nb_elem_c;
    pos = 0;
    k = 0;
    /* find the number of tiles this process will handle */
    nb_elem_r = Ddesc->lmt / Ddesc->GRIDrows;
    j = Ddesc->lmt % Ddesc->GRIDrows;
    if (Ddesc->rowRANK < j)
        {
            nb_elem_r++;
        }
    nb_elem_c = Ddesc->lnt / Ddesc->GRIDcols;
    j =  Ddesc->lnt % Ddesc->GRIDcols;
    if (Ddesc->colRANK < j)
        {
            nb_elem_c++;
        }
    if (Ddesc->mpi_rank == 0)
        {
            nb_reqs = ((Pdesc->lmt)*(Pdesc->lnt)) - (nb_elem_c*nb_elem_r);
            *reqs = (MPI_Request *)malloc(nb_reqs * sizeof(MPI_Request));
            for (i = 0 ; i < Pdesc->lmt; i++)
                for (j = 0 ; j < Pdesc->lnt ; j++)
                    {
                        rank = dplasma_get_rank_for_tile(Ddesc, i, j);
                        if (rank == 0) /* this tile belongs to me */
                            {
                                /*  printf("tile (%d, %d) for self, memcpy\n", i, j);*/
                                memcpy(&(((double *)Ddesc->mat)[pos]), plasma_A(Pdesc, i, j), Ddesc->bsiz*(sizeof(double)));
                                pos = pos + Ddesc->bsiz;
                                continue;
                            }
                        MPI_Isend(plasma_A(Pdesc, i, j), Ddesc->bsiz, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &((*reqs)[k]));
                        k++;
                        
                    }
        }
    else
        {
            *reqs = malloc((nb_elem_c * nb_elem_r) * sizeof(MPI_Request));

            if (NULL == *reqs)
                {
                    printf("memory allocation failed\n");
                    exit(2);
                }
            
            for (i = 0; i < (nb_elem_c * nb_elem_r) ; i++)
                {
                    MPI_Irecv(&(((double*)Ddesc->mat)[i*(Ddesc->bsiz)]), Ddesc->bsiz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &((*reqs)[i]));
                }
            
        }
    return 0;
}

static void print_block(char * stri, int m, int n, double * block, int blength, int total_size)
{
    int i;
    printf("block size: %d, total size: %d\n", blength, total_size);
    printf("%s (%d,%d)\n", stri, m, n);
    for (i = 0 ; i < min(10, blength) ; i++ )
        printf("%e ", block[i]);
    printf("\n");
    i = total_size - blength;
    for ( ; i < min((total_size - blength) + 10, total_size) ; i++ )
        printf("%e ", block[i]);
    printf("\n\n\n");
}
void data_dist_verif(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc)
{
    int m,n, rank;
    double * buff;
    
    if(Ddesc->mpi_rank == 0)
        {
            m = Pdesc->lmt/2;
            n = Pdesc->lnt/2;
            print_block("orig tile ", 0, 0, plasma_A(Pdesc, 0, 0), Pdesc->nb, Pdesc->bsiz);
            print_block("orig tile ", m, n, plasma_A(Pdesc, m, n), Pdesc->nb, Pdesc->bsiz);
            print_block("orig tile ", Pdesc->lmt - 1, Pdesc->lnt -1, plasma_A(Pdesc, Pdesc->lmt - 1, Pdesc->lnt - 1), Pdesc->nb ,Pdesc->bsiz);
        }
    rank = dplasma_get_rank_for_tile(Ddesc, 0, 0);
    if (Ddesc->mpi_rank == rank)
        {
            buff = (double *) dplasma_get_tile(Ddesc, 0, 0);
            print_block("Dist tile", 0, 0, buff, Ddesc->nb , Ddesc->bsiz);
        }
    m = Ddesc->lmt/2;
    n = Ddesc->lnt/2;
    rank = dplasma_get_rank_for_tile(Ddesc, m, n);
    if (Ddesc->mpi_rank == rank)
        {
            buff = dplasma_get_tile(Ddesc, m, n);
            printf("Check: %d,%d \n", Ddesc->mpi_rank, rank);
            print_block("Dist tile", m, n, buff, Ddesc->nb ,Ddesc->bsiz);
        }
    rank = dplasma_get_rank_for_tile(Ddesc, Ddesc->lmt - 1, Ddesc->lnt - 1);
    if (Ddesc->mpi_rank == rank)
        {
            buff = dplasma_get_tile(Ddesc, Ddesc->lmt - 1, Ddesc->lnt - 1);
            printf("check: %d,%d \n", Ddesc->mpi_rank, rank);
            print_block("dist tile", Ddesc->lmt - 1, Ddesc->lnt - 1, buff , Ddesc->nb ,Ddesc->bsiz);
        }

}

int is_data_distributed(DPLASMA_desc * Ddesc, MPI_Request * reqs)
{
    int j;
    int nb_reqs;
    int nb_elem_r, nb_elem_c;
    MPI_Status * stats;
    /* find the number of tiles this process will handle */
    nb_elem_r = Ddesc->lmt / Ddesc->GRIDrows;
    j = Ddesc->lmt % Ddesc->GRIDrows;
    if (Ddesc->rowRANK < j)
        {
            nb_elem_r++;
        }
    nb_elem_c = Ddesc->lnt / Ddesc->GRIDcols;
    j =  Ddesc->lnt % Ddesc->GRIDcols;
    if (Ddesc->colRANK < j)
        {
            nb_elem_c++;
        }

    if (Ddesc->mpi_rank == 0)
        {
            nb_reqs = ((Ddesc->lmt)*(Ddesc->lnt)) - (nb_elem_c*nb_elem_r);
            //   printf("waiting for completion of %d  Isend\n", nb_reqs);
            stats = malloc(nb_reqs * sizeof(MPI_Status));
            MPI_Waitall(nb_reqs, reqs, stats);
            //     printf("completion of Isend done\n");
        }
    else
        {
            nb_reqs = nb_elem_c*nb_elem_r;
            //            printf("waiting for completion of %d Irecv\n", nb_reqs);
            stats = malloc(nb_reqs * sizeof(MPI_Status));
            MPI_Waitall(nb_reqs, reqs, stats);
            //          printf("completion of Irecv done\n");
        }
    return 1;
}
