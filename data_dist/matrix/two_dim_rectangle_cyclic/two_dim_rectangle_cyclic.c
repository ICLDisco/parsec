/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

#include "two_dim_rectangle_cyclic.h"
#include "data_distribution.h"
#include "matrix.h"


//#define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
/*static inline void * plasma_A(PLASMA_desc * Pdesc, int m, int n)
{
    return &((double*)Pdesc->mat)[Pdesc->bsiz*(m)+Pdesc->bsiz*Pdesc->lmt*(n)];

}
*/

static uint32_t twoDBC_get_rank_for_tile(DAGuE_ddesc_t * desc, ...)
{
    int stc, cr, m, n;
    int str, rr;
    int res;
    va_list ap;
    two_dim_block_cyclic_t * Ddesc;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    str = m / Ddesc->nrst; /* (m,n) is in super-tile (str, stc)*/
    stc = n / Ddesc->ncst;
    
    rr = str % Ddesc->GRIDrows;
    cr = stc % Ddesc->GRIDcols;
    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * Ddesc->GRIDcols + cr;
    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
/*            m, n, res, rr, cr, Ddesc->GRIDrows, Ddesc->GRIDcols); */
    return res;
}


/* #if 0 /\*def USE_MPI*\/ */
/* /\* empty stub for now, should allow for async data transfer from recv side *\/ */
/* void * dplasma_get_tile_async(DPLASMA_desc *Ddesc, int m, int n, MPI_Request *req) */
/* { */
    
/*     return NULL; */
/* } */

/* void * dplasma_get_tile(DPLASMA_desc *Ddesc, int m, int n) */
/* { */
/*     int tile_rank; */
    
/*     tile_rank = dplasma_get_rank_for_tile(Ddesc, m, n); */
/*     if(Ddesc->mpi_rank == tile_rank) */
/*     { */
/*         //        printf("%d get_local_tile (%d, %d)\n", Ddesc->mpi_rank, m, n); */
/*         return dplasma_get_local_tileDdesc, m, n); */
/*     } */
/* #ifdef USE_MPI */
/*     printf("%d get_remote_tile (%d, %d) from %d\n", Ddesc->mpi_rank, m, n, tile_rank); */
/*     MPI_Recv(plasma_A((PLASMA_desc *) Ddesc, m, n), Ddesc->bsiz, MPI_DOUBLE, tile_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); */
/*     return plasma_A((PLASMA_desc *)Ddesc, m, n); */
/* #else */
/*     fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__); */
/*     return NULL; */
/* #endif */
/* } */

/* #else  */
/* /\*#define dplasma_get_local_tile_s dplasma_get_local_tile*\/ */
/* #endif */

static void * twoDBC_get_local_tile(DAGuE_ddesc_t * desc, ...)
{
    int pos, m, n;
    int nb_elem_r, last_c_size;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    assert(desc->myrank == twoDBC_get_rank_for_tile(desc, m, n));

    /**********************************/

    nb_elem_r = Ddesc->nb_elem_r * Ddesc->ncst; /* number of tiles per column of super-tile */

    pos = nb_elem_r * ((n / Ddesc->ncst)/ Ddesc->GRIDcols); /* pos is currently at head of supertile (0xA) */

    if (n >= ((Ddesc->super.lnt/Ddesc->ncst)*Ddesc->ncst )) /* tile is in the last column of super-tile */
        {
            last_c_size = (Ddesc->super.lnt % Ddesc->ncst) * Ddesc->nrst; /* number of tile per super tile in last column */
        }
    else
        {
            last_c_size = Ddesc->ncst * Ddesc->nrst;
        }
    pos += (last_c_size * ((m / Ddesc->nrst) / Ddesc->GRIDrows ) ); /* pos is at head of supertile (BxA) containing (m,n)  */
    
    /* if tile (m,n) is in the last row of super tile and this super tile is smaller than others */
    if (m >= ((Ddesc->super.lmt/Ddesc->nrst)*Ddesc->nrst))
        {           
            last_c_size = Ddesc->super.lmt % Ddesc->nrst;
        }
    else
        {
            last_c_size = Ddesc->nrst;
        }
    pos += ((n % Ddesc->ncst) * last_c_size); /* pos is at (B, n)*/
    pos += (m % Ddesc->nrst); /* pos is at (m,n)*/

    //printf("get tile (%d, %d) is at pos %d\t(ptr %p, base %p)\n", m, n, pos*Ddesc->bsiz,&(((double *) Ddesc->mat)[pos * Ddesc->bsiz]), Ddesc->mat);
    /************************************/
    return &(((char *) Ddesc->mat)[pos * Ddesc->super.bsiz * Ddesc->super.mtype]);
}


void two_dim_block_cyclic_init(two_dim_block_cyclic_t * Ddesc, enum matrix_type mtype, int nodes, int cores, int myrank, int mb, int nb, int ib, int lm, int ln, int i, int j, int m, int n, int nrst, int ncst, int process_GridRows )
{
    int temp;
    int nbstile_r;
    int nbstile_c;

    // Filling matrix description woth user parameter
    Ddesc->super.super.nodes = nodes ;
    Ddesc->super.super.cores = cores ;
    Ddesc->super.super.myrank = myrank ;
    Ddesc->super.mtype = mtype;
    Ddesc->super.mb = mb;
    Ddesc->super.nb = nb;
    Ddesc->super.ib = ib;
    Ddesc->super.lm = lm;
    Ddesc->super.ln = ln;
    Ddesc->super.i = i;
    Ddesc->super.j = j;
    Ddesc->super.m = m;
    Ddesc->super.n = n;
    Ddesc->nrst = nrst;
    Ddesc->ncst = ncst;
    Ddesc->GRIDrows = process_GridRows;

    assert((nodes % process_GridRows) == 0);
    Ddesc->GRIDcols = nodes / process_GridRows;

    // Matrix derived parameters
    Ddesc->super.lmt = ((Ddesc->super.lm)%(Ddesc->super.mb)==0) ? ((Ddesc->super.lm)/(Ddesc->super.mb)) : ((Ddesc->super.lm)/(Ddesc->super.mb) + 1);
    Ddesc->super.lnt = ((Ddesc->super.ln)%(Ddesc->super.nb)==0) ? ((Ddesc->super.ln)/(Ddesc->super.nb)) : ((Ddesc->super.ln)/(Ddesc->super.nb) + 1);
    Ddesc->super.bsiz =  Ddesc->super.mb * Ddesc->super.nb;

    // Submatrix parameters    
    Ddesc->super.mt = ((Ddesc->super.m)%(Ddesc->super.mb)==0) ? ((Ddesc->super.m)/(Ddesc->super.nb)) : ((Ddesc->super.m)/(Ddesc->super.nb) + 1);
    Ddesc->super.nt = ((Ddesc->super.n)%(Ddesc->super.nb)==0) ? ((Ddesc->super.n)/(Ddesc->super.nb)) : ((Ddesc->super.n)/(Ddesc->super.nb) + 1);
    

    /* computing colRANK and rowRANK */
    Ddesc->rowRANK = (Ddesc->super.super.myrank)/(Ddesc->GRIDcols);
    Ddesc->colRANK = (Ddesc->super.super.myrank)%(Ddesc->GRIDcols);


    /* computing the number of rows of super-tile */
    nbstile_r = Ddesc->super.lmt / Ddesc->nrst;
    if((Ddesc->super.lmt % Ddesc->nrst) != 0)
        nbstile_r++;

    /* computing the number of colums of super-tile*/
    nbstile_c = Ddesc->super.lnt / Ddesc->ncst;
    if((Ddesc->super.lnt % Ddesc->ncst) != 0)
        nbstile_c++;

    if ( Ddesc->GRIDrows > nbstile_r || Ddesc->GRIDcols > nbstile_c)
        {
            printf("The process grid chosen is %dx%d, block distribution choosen is %d, %d : cannot generate matrix \n",
                   Ddesc->GRIDrows, Ddesc->GRIDcols, nbstile_r, nbstile_c);
            exit(-1);
        }
    // printf("matrix to be generated distributed by block of %d x %d tiles \n", nbstile_r, nbstile_c);    

    /* find the number of tiles this process will handle */
    Ddesc->nb_elem_r = 0;
    temp = Ddesc->rowRANK * Ddesc->nrst; /* row coordinate of the first tile to handle */
    while ( temp < Ddesc->super.lmt)
        {
            if ( (temp  + (Ddesc->nrst)) < Ddesc->super.lmt)
                {
                    Ddesc->nb_elem_r += (Ddesc->nrst);
                    temp += ((Ddesc->GRIDrows) * (Ddesc->nrst));
                    continue;
                }
            Ddesc->nb_elem_r += ((Ddesc->super.lmt) - temp);
            break;
        }

    Ddesc->nb_elem_c = 0;
    temp = Ddesc->colRANK * Ddesc->ncst;
    while ( temp < Ddesc->super.lnt)
        {
            if ( (temp  + (Ddesc->ncst)) < Ddesc->super.lnt)
                {
                    Ddesc->nb_elem_c += (Ddesc->ncst);
                    temp += (Ddesc->GRIDcols) * (Ddesc->ncst);
                    continue;
                }
            Ddesc->nb_elem_c += ((Ddesc->super.lnt) - temp);
            break;
        }
    /*  printf("process %d(%d,%d) handles %d x %d tiles\n",
        Ddesc->mpi_rank, Ddesc->rowRANK, Ddesc->colRANK, Ddesc->nb_elem_r, Ddesc->nb_elem_c);*/

    /* Allocate memory for matrices in block layout */
    Ddesc->mat = malloc(Ddesc->nb_elem_r * Ddesc->nb_elem_c * Ddesc->super.bsiz * Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
        }
    Ddesc->super.super.rank_of =  twoDBC_get_rank_for_tile;
    Ddesc->super.super.data_of =  twoDBC_get_local_tile;
}

/*
int dplasma_desc_init(const PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc)
{
    Ddesc->dtyp = Pdesc->dtyp;
    Ddesc->mb = Pdesc->mb;
    Ddesc->nb = Pdesc->nb;
    Ddesc->bsiz = Pdesc->bsiz ;
    Ddesc->lm = Pdesc->lm ;
    Ddesc->ln = Pdesc->ln ;
    Ddesc->lmt = Pdesc->lmt ;
    Ddesc->lnt = Pdesc->lnt ;
    Ddesc->i = Pdesc->i ;
    Ddesc->j = Pdesc->j ;
    Ddesc->m = Pdesc->m  ;
    Ddesc->n = Pdesc->n ;
    Ddesc->mt = Pdesc->mt ;
    Ddesc->nt = Pdesc->nt ;
    {
        plasma_context_t *plasma = plasma_context_self();
        Ddesc->ib = PLASMA_IB;
    }
    return ddesc_compute_vals( Ddesc );
}
*/

/* int dplasma_desc_bcast(const PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc) */
/* { */
/* #ifdef USE_MPI */
/*     int tmp_ints[21]; */

/*     if (Ddesc->mpi_rank == 0) /\* send data *\/ */
/*         { */
/*             Ddesc->dtyp= tmp_ints[0] = Pdesc->dtyp; */
/*             Ddesc->mb = tmp_ints[1] = Pdesc->mb; */
/*             Ddesc->nb = tmp_ints[2] = Pdesc->nb; */
/*             Ddesc->bsiz  = tmp_ints[3] = Pdesc->bsiz ; */
/*             Ddesc->lm  = tmp_ints[4] = Pdesc->lm ; */
/*             Ddesc->ln  = tmp_ints[5] = Pdesc->ln ; */
/*             Ddesc->lmt  = tmp_ints[6] = Pdesc->lmt ; */
/*             Ddesc->lnt  = tmp_ints[7] = Pdesc->lnt ; */
/*             Ddesc->i = tmp_ints[8] = Pdesc->i ; */
/*             Ddesc->j = tmp_ints[9] = Pdesc->j ; */
/*             Ddesc->m = tmp_ints[10] = Pdesc->m  ; */
/*             Ddesc->n = tmp_ints[11] = Pdesc->n ; */
/*             Ddesc->mt  = tmp_ints[12] = Pdesc->mt ; */
/*             Ddesc->nt  = tmp_ints[13] = Pdesc->nt ; */
/*             tmp_ints[14] = Ddesc->nrst; */
/*             tmp_ints[15] = Ddesc->ncst;             */
/*             tmp_ints[16] = Ddesc->GRIDrows ; */
/*             tmp_ints[17] = Ddesc->GRIDcols ; */
/*             tmp_ints[18] = Ddesc->cores ; */
/*             tmp_ints[19] = Ddesc->nodes ; */
/*             tmp_ints[20] = Ddesc->ib; */
            
/*             MPI_Bcast(tmp_ints, 21, MPI_INT, 0, MPI_COMM_WORLD); */
/*         } */
/*     else /\* rank != 0, receive data *\/ */
/*         { */
/*             MPI_Bcast(tmp_ints, 21, MPI_INT, 0, MPI_COMM_WORLD); */
        
/*             Ddesc->dtyp= tmp_ints[0]; */
/*             Ddesc->mb = tmp_ints[1]; */
/*             Ddesc->nb = tmp_ints[2]; */
/*             Ddesc->bsiz  = tmp_ints[3]; */
/*             Ddesc->lm  = tmp_ints[4]; */
/*             Ddesc->ln  = tmp_ints[5]; */
/*             Ddesc->lmt  = tmp_ints[6]; */
/*             Ddesc->lnt  = tmp_ints[7]; */
/*             Ddesc->i = tmp_ints[8]; */
/*             Ddesc->j = tmp_ints[9]; */
/*             Ddesc->m = tmp_ints[10]; */
/*             Ddesc->n = tmp_ints[11]; */
/*             Ddesc->mt  = tmp_ints[12]; */
/*             Ddesc->nt  = tmp_ints[13]; */
/*             Ddesc->nrst  = tmp_ints[14]; */
/*             Ddesc->ncst  = tmp_ints[15]; */
/*             Ddesc->GRIDrows  = tmp_ints[16]; */
/*             Ddesc->GRIDcols  = tmp_ints[17]; */
/*             Ddesc->cores  = tmp_ints[18]; */
/*             Ddesc->nodes  = tmp_ints[19]; */
/*             Ddesc->ib     = tmp_ints[20]; */
            
/*             if( -1 == ddesc_compute_vals(Ddesc) ) */
/*                 { */
/*                     MPI_Abort(MPI_COMM_WORLD, 2); */
/*                 } */
/*         } */
    
/*     return dplasma_desc_workspace_allocate(Ddesc); */
/* #else /\* USE_MPI *\/ */
    
/*     fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__); */
/*     return -1; */
/* #endif /\* USE_MPI *\/ */
/* } */


/* int tiling(PLASMA_enum * uplo, int N, double *A, int LDA, int NRHS, PLASMA_desc * descA) */
/* { */
/*     int NB, NT; */
/*     int status; */
/*     double *Abdl; */
/*     plasma_context_t *plasma; */

/*     plasma = plasma_context_self(); */
/*     if (plasma == NULL) { */
/*         plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized"); */
/*         return PLASMA_ERR_NOT_INITIALIZED; */
/*     } */
/*     /\* Check input arguments *\/ */
/*     if (*uplo != PlasmaUpper && *uplo != PlasmaLower) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of uplo"); */
/*         return -1; */
/*     } */
/*     if (N < 0) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of N"); */
/*         return -2; */
/*     } */
/*     if (LDA < max(1, N)) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of LDA"); */
/*         return -4; */
/*     } */
/*     /\* Quick return *\/ */
/*     if (max(N, 0) == 0) */
/*         return PLASMA_SUCCESS; */

/*     /\* Tune NB depending on M, N & NRHS; Set NBNBSIZE *\/ */
/*     status = plasma_tune(PLASMA_FUNC_DGESV, N, N, NRHS); */
/*     if (status != PLASMA_SUCCESS) { */
/*         plasma_error("PLASMA_dpotrf", "plasma_tune() failed"); */
/*         return status; */
/*     } */
/*     /\* Set NT *\/ */
/*     NB = PLASMA_NB; */
/*     NT = (N%NB==0) ? (N/NB) : (N/NB+1); */

/*     /\* Allocate memory for matrices in block layout *\/ */
/*     Abdl = (double *)plasma_shared_alloc(plasma, NT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble); */
/*     if (Abdl == NULL) { */
/*         plasma_error("PLASMA_dpotrf", "plasma_shared_alloc() failed"); */
/*         return PLASMA_ERR_OUT_OF_RESOURCES; */
/*     } */

/*     /\*PLASMA_desc*\/ *descA = plasma_desc_init( */
/*                                              Abdl, PlasmaRealDouble, */
/*                                              PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE, */
/*                                              N, N, 0, 0, N, N); */

/*     plasma_parallel_call_3(plasma_lapack_to_tile, */
/*                            double*, A, */
/*                            int, LDA, */
/*                            PLASMA_desc, *descA); */

/*     //printf("matrix tiled in %dx%d\n", descA->lmt, descA->lnt); */
/*     return 0; */
    
/* } */

/* int untiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA) */
/* { */
/*     plasma_context_t *plasma; */
        
/*     plasma = plasma_context_self(); */
/*     if (plasma == NULL) { */
/*         plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized"); */
/*         return PLASMA_ERR_NOT_INITIALIZED; */
/*     } */
/*     /\* Check input arguments *\/ */
/*     if (*uplo != PlasmaUpper && *uplo != PlasmaLower) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of uplo"); */
/*         return -1; */
/*     } */
/*     if (N < 0) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of N"); */
/*         return -2; */
/*     } */
/*     if (LDA < max(1, N)) { */
/*         plasma_error("PLASMA_dpotrf", "illegal value of LDA"); */
/*         return -4; */
/*     } */
/*     /\* Quick return *\/ */
/*     if (max(N, 0) == 0) */
/*         return PLASMA_SUCCESS; */
 
/*     plasma_parallel_call_3(plasma_tile_to_lapack, */
/*                            PLASMA_desc, *descA, */
/*                            double*, A, */
/*                            int, LDA); */
    
/*     //printf("matrix untiled from %dx%d\n", descA->lmt, descA->lnt); */
/*     return PLASMA_SUCCESS; */
    
/* } */




/* #ifdef HEAVY_DEBUG */
/* static void print_block(char * stri, int m, int n, double * block, int blength, int total_size) */
/* { */
/*     int i; */
/*     printf("%s (%d,%d)\n", stri, m, n); */
/*     for (i = 0 ; i < min(10, blength) ; i++ ) */
/*         printf("%e ", block[i]); */
/*     printf("\n"); */
/*     i = total_size - blength; */
/*     for ( ; i < min((total_size - blength) + 10, total_size) ; i++ ) */
/*         printf("%e ", block[i]); */
/*     printf("\n\n"); */
/* } */
/* #endif */

/* void data_dist_verif(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc) */
/* { */
/* #ifdef USE_MPI */
/*     int m, n, k, rank; */
/*     double * buff; */
/*     double * buff2; */

/*     for (m = 0 ; m < Ddesc->lmt ; m++) */
/*         for ( n = 0 ; n < Ddesc->lnt ; n++) */
/*         { */
/*             MPI_Barrier(MPI_COMM_WORLD); */
            
/*             rank = dplasma_get_rank_for_tile(Ddesc, m, n); */
            
/*             if ((rank == 0) && (Ddesc->mpi_rank == 0))/\* this proc is rank 0 and handles the tile*\/ */
/*             { */
/*                 buff = plasma_A(Pdesc, m, n); */
/*                 buff2 = dplasma_get_local_tile(Ddesc, m, n); */
/*                 for ( k = 0; k < Ddesc->bsiz ; k++) */
/*                 { */
/*                     if(buff[k] != buff2[k]) */
/*                     { */
/*                         printf("WARNING: tile(%d, %d) differ !\n", m, n); */
/*                         break; */
/*                     } */
/*                 } */
/*                 continue; */
/*             } */
/* #ifdef HEAVY_DEBUG */
/*             if(Ddesc->mpi_rank == 0) */
/*                 print_block("orig tile ", m, n, plasma_A(Pdesc, m, n), Pdesc->nb, Pdesc->bsiz); */
            
            
/*             if (Ddesc->mpi_rank == rank) */
/*             { */
/*                 buff = dplasma_get_local_tile(Ddesc, m, n); */
/*                 printf("Check: rank %d has tile %d, %d\n", Ddesc->mpi_rank, m, n); */
/*                 print_block("Dist tile", m, n, buff, Ddesc->nb, Ddesc->bsiz); */
/*             } */
/* #endif */
/*         } */
/*     MPI_Barrier(MPI_COMM_WORLD); */
/*     printf("Data verification ended\n"); */
/* #else */
/*     fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__); */
/* #endif     */
/* } */

/* int data_dump(DPLASMA_desc * Ddesc){ */
/*     FILE * tmpf; */
/*     int i, j, k; */
/*     double * buf; */
/*     tmpf = fopen("tmp_local_data_dump.txt", "w"); */
/*     if(NULL == tmpf) */
/*         { */
/*             perror("opening file: tmp_local_data_dump.txt" ); */
/*             return -1; */
/*         } */
/*     for (i = 0 ; i < Ddesc->lmt ; i++) */
/*         for ( j = 0 ; j< Ddesc->lnt ; j++) */
/*             { */
/*                 if (dplasma_get_rank_for_tile(Ddesc, i, j) == Ddesc->mpi_rank) */
/*                     { */
/*                         buf = (double*)dplasma_get_local_tile(Ddesc, i, j); */
/*                         for (k = 0 ; k < Ddesc->bsiz ; k++) */
/*                             { */
/*                                 fprintf(tmpf, "%e ", buf[k]); */
/*                             } */
/*                         fprintf(tmpf, "\n"); */
/*                     } */
/*             } */
/*     fclose(tmpf); */
/*     return 0; */
/* } */

/* int plasma_dump(PLASMA_desc * Pdesc){ */
/*     FILE * tmpf; */
/*     int i, j, k; */
/*     double * buf; */
/*     tmpf = fopen("tmp_plasma_data_dump.txt", "w"); */
/*     if(NULL == tmpf) */
/*         { */
/*             perror("opening file: tmp_plasma_data_dump.txt" ); */
/*             return -1; */
/*         } */
/*     for (i = 0 ; i < Pdesc->lmt ; i++) */
/*         for ( j = 0 ; j< Pdesc->lnt ; j++) */
/*             { */
/*                 buf = (double*)plasma_A(Pdesc, i, j); */
/*                 for (k = 0 ; k < Pdesc->bsiz ; k++) */
/*                     { */
/*                         fprintf(tmpf, "%e ", buf[k]); */
/*                     } */
/*                 fprintf(tmpf, "\n"); */
/*             } */
/*     fclose(tmpf); */
/*     return 0; */
/* } */

/* int compare_distributed_tiles(DPLASMA_desc * A, DPLASMA_desc * B, int row, int col, double precision) */
/* { */
/*     int i; */
/*     double * a; */
/*     double * b; */
/*     double c; */

/*     /\* check memory locality of the tiles *\/ */
/*     if (   (A->mpi_rank != dplasma_get_rank_for_tile(A, row, col)) */
/*         || (B->mpi_rank != dplasma_get_rank_for_tile(B, row, col)) */
/*         || (A->mpi_rank != B->mpi_rank)) */
/*         { */
/*             printf("Compare tile failed: (%d, %d) is not local to process %d\n", row, col, A->mpi_rank); */
/*             return 0; */
/*         } */
    
/*     /\* assign values *\/ */
/*     a = (double *)dplasma_get_local_tile_s(A, row, col); */
/*     b = (double *)dplasma_get_local_tile_s(B, row, col); */
/*     /\* compare each value*\/ */
/*     for(i = 0 ; i < A->bsiz ; i++) */
/*         { */
/*             c = a[i] - b[i]; */
/*             if(0.0-precision < c && c < precision) */
/*                 continue; */
/*             else */
/*                 { */
/*                     printf("difference discovered in matrix. Tile: (%d, %d), position: %d, difference: %f\n", row, col, i, c); */
/*                     return 0; */
/*                 } */
/*         } */
/*     return 1; */
/* } */

/* int compare_matrices(DPLASMA_desc * A, DPLASMA_desc * B, double precision) */
/* { */
/*     int i, j, mt, nt, rank; */
/*     int res = 1; */
/*     mt = (A->mt < B->mt) ? A->mt : B->mt; */
/*     nt = (A->nt < B->nt) ? A->nt : B->nt; */
/*     for (i = 0 ; i < mt ; i++) */
/*         for( j = 0 ; i < nt ; j++) */
/*             { */
/*                 rank = dplasma_get_rank_for_tile(A, i, j); */
/*                 if (rank == A->mpi_rank) */
/*                     { */
/*                         res &= compare_distributed_tiles(A, B, i, j, precision); */
/*                     } */
/*                 if (res == 0) */
/*                     return 0; */
/*             } */
/*     return res; */
/* } */

/* int compare_plasma_matrices(PLASMA_desc * A, PLASMA_desc * B, double precision) */
/* { */
/*     int i, j, k, mt, nt; */
/*     double * a; */
/*     double * b; */
/*     double c; */

/*     mt = (A->mt < B->mt) ? A->mt : B->mt; */
/*     nt = (A->nt < B->nt) ? A->nt : B->nt; */
/*     printf("compare matrices of size %d x %d\n", mt, nt); */
/*     for (i = 0 ; i < mt ; i++) */
/*         for( j = 0 ; j < nt ; j++) */
/*             { */
/*                 a = (double *)plasma_A(A, i, j); */
/*                 b = (double *)plasma_A(B, i, j); */
/*                 /\* compare each value*\/ */
/*                 for(k = 0 ; k < A->bsiz ; k++) */
/*                     { */
/*                         c = a[k] - b[k]; */
/*                         if(0.0-precision < c && c < precision) */
/*                             continue; */
/*                         else */
/*                             { */
/*                                 printf("difference discovered in matrix. Tile: (%d, %d), position: %d, difference: %f\n", i, j, k, c); */
/*                                 return 0; */
/*                             } */
/*                     } */
/*                 printf("tile (%d, %d) passed  (bsiz == %d)\n", i, j, A->bsiz); */
/*             } */
/*     printf("Matrices almost identical\n"); */
/*     return 1; */
/* } */


