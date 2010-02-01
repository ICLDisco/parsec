/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "cblas.h"
#include <math.h>

#include "plasma.h"
#include "../src/common.h"
#include "../src/lapack.h"
#include "../src/context.h"
#include "../src/allocate.h"
#include "data_management.h"

//#define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
static inline void * plasma_A(PLASMA_desc * Pdesc, int m, int n)
{
    return &((double*)Pdesc->mat)[Pdesc->bsiz*(m)+Pdesc->bsiz*Pdesc->lmt*(n)];

}

static int ddesc_compute_vals( DPLASMA_desc * Ddesc )
{
    int i;
    int nbstile_r;
    int nbstile_c;
    
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
    
    printf("mpi rank %d is map to P(%d,%d)\n", Ddesc->mpi_rank, Ddesc->rowRANK, Ddesc->colRANK);
    
    /* computing the number of rows of super-tile */
    nbstile_r = Ddesc->lmt / Ddesc->nrst;
    if((Ddesc->lmt % Ddesc->nrst) != 0)
        nbstile_r++;
    
    /* computing the number of colums of super-tile*/
    nbstile_c = Ddesc->lnt / Ddesc->ncst;
    if((Ddesc->lnt % Ddesc->ncst) != 0)
        nbstile_c++;
    
    if (Ddesc->mpi_rank == 0) 
        printf("matrix is super-tiled in (%d, %d)", nbstile_r, nbstile_c);
    /* allocate memory for tiles data */
    
    /*  nbt = Ddesc->lmt * Ddesc->lnt;  total number of tiles */ 
    if ( Ddesc->GRIDrows > nbstile_r || Ddesc->GRIDcols > nbstile_c)
    {
        printf("The process grid chosen is %dx%d, supertiling is %d, %d\n", Ddesc->GRIDrows, Ddesc->GRIDcols, nbstile_r, nbstile_c);
        return -1;
    }
    return 0;
}

static int ddesc_allocate( DPLASMA_desc * Ddesc ) 
{
    int nb_elem_r, nb_elem_c, j;
    
    /* find the number of tiles this process will handle */
    nb_elem_r = 0;
    j = Ddesc->rowRANK * Ddesc->nrst;
    while ( j < Ddesc->lmt)
    {
        if ( (j  + (Ddesc->nrst)) < Ddesc->lmt)
        {
            nb_elem_r += (Ddesc->nrst);
            j += (Ddesc->GRIDrows) * (Ddesc->nrst);
            continue;
        }
        nb_elem_r += ((Ddesc->lmt) - j);
        break;
    }
    
    nb_elem_c = 0;
    j = Ddesc->colRANK * Ddesc->ncst;
    while ( j < Ddesc->lnt)
    {
        if ( (j  + (Ddesc->ncst)) < Ddesc->lnt)
        {
            nb_elem_c += (Ddesc->ncst);
            j += (Ddesc->GRIDcols) * (Ddesc->ncst);
            continue;
        }
        nb_elem_c += ((Ddesc->lnt) - j);
        break;
    }
    printf("process %d(%d,%d) handles %d x %d tiles\n", Ddesc->mpi_rank, Ddesc->rowRANK, Ddesc->colRANK, nb_elem_r, nb_elem_c);
    
    Ddesc->mat = malloc(sizeof(double) * nb_elem_c * nb_elem_r * Ddesc->bsiz);
    return 0;
}

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
    return ddesc_compute_vals( Ddesc );
}

int dplasma_desc_bcast(const PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc)
{
#ifdef USE_MPI
    int * tmp_ints;
        
    tmp_ints = malloc(sizeof(int)*18);
    if (tmp_ints == NULL)
        {
            printf("memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
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
            tmp_ints[14] = Ddesc->nrst;
            tmp_ints[15] = Ddesc->ncst;            
            tmp_ints[16] = Ddesc->GRIDrows ;
            tmp_ints[17] = Ddesc->GRIDcols ;

            MPI_Bcast(tmp_ints, 18, MPI_INT, 0, MPI_COMM_WORLD);

        }
    else /* rank != 0, receive data */
        {
            MPI_Bcast(tmp_ints, 18, MPI_INT, 0, MPI_COMM_WORLD);

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
            Ddesc->nrst  = tmp_ints[14];
            Ddesc->ncst  = tmp_ints[15];
            Ddesc->GRIDrows  = tmp_ints[16];
            Ddesc->GRIDcols  = tmp_ints[17];
        }
    free(tmp_ints);

    if( -1 == ddesc_compute_vals(Ddesc) )
    {
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    ddesc_allocate(Ddesc);
    return 0;
#else
    fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__);
    return -1;
#endif
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

int untiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA)
{
    plasma_context_t *plasma;
        
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
 
    plasma_parallel_call_3(plasma_tile_to_lapack,
                           PLASMA_desc, *descA,
                           double*, A,
                           int, LDA);
    
    printf("matrix untiled from %dx%d\n", descA->lmt, descA->lnt);
    return PLASMA_SUCCESS;
    
}


int dplasma_get_rank_for_tile(DPLASMA_desc * Ddesc, int m, int n)
{
    int stc, cr;
    int str, rr;
    int res;

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

#ifdef USE_MPI
/* empty stub for now, should allow for async data transfer from recv side */
void * dplasma_get_tile_async(DPLASMA_desc *Ddesc, int m, int n, MPI_Request *req)
{
    
    return NULL;
}
#endif

void * dplasma_get_tile(DPLASMA_desc *Ddesc, int m, int n)
{
    int tile_rank;
    
    tile_rank = dplasma_get_rank_for_tile(Ddesc, m, n);
    if(Ddesc->mpi_rank == tile_rank)
    {
        //        printf("%d get_local_tile (%d, %d)\n", Ddesc->mpi_rank, m, n);
        return dplasma_get_local_tile(Ddesc, m, n);
    }
#ifdef USE_MPI
    printf("%d get_remote_tile (%d, %d) from %d\n", Ddesc->mpi_rank, m, n, tile_rank);
    MPI_Recv(plasma_A((PLASMA_desc *) Ddesc, m, n), Ddesc->bsiz, MPI_DOUBLE, tile_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return plasma_A((PLASMA_desc *)Ddesc, m, n);
#else
    fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__);
    return NULL;
#endif
}

void * dplasma_get_local_tile(DPLASMA_desc * Ddesc, int m, int n)
{
    int pos;
    int nb_elem_r, last_c_size;
    int j;
    assert(Ddesc->mpi_rank == dplasma_get_rank_for_tile(Ddesc, m, n));

    /**********************************/
    nb_elem_r = 0; /* number of row tiles handled per column*/
    j = Ddesc->rowRANK * Ddesc->nrst;
    while ( j < Ddesc->lmt)
        {
            if ( (j  + (Ddesc->nrst)) < Ddesc->lmt)
                {
                    nb_elem_r += (Ddesc->nrst);
                    j += (Ddesc->GRIDrows * Ddesc->nrst);
                    continue;
                }
            nb_elem_r += ((Ddesc->lmt) - j);
            break;
        }
    nb_elem_r = nb_elem_r * Ddesc->ncst; /* number of tiles per column of super-tile */

    pos = nb_elem_r * ((n / Ddesc->ncst)/ Ddesc->GRIDcols); /* pos is currently at head of supertile (0xA) */

    if (n >= ((Ddesc->lnt/Ddesc->ncst)*Ddesc->ncst )) /* tile is in the last column of super-tile */
        {
            last_c_size = (Ddesc->lnt % Ddesc->ncst) * Ddesc->nrst; /* number of tile per super tile in last column */
        }
    else
        {
            last_c_size = Ddesc->ncst * Ddesc->nrst;
        }
    pos += (last_c_size * ((m / Ddesc->nrst) / Ddesc->GRIDrows ) ); /* pos is at head of supertile (BxA) containing (m,n)  */

    /* if tile (m,n) is in the last super tile in the row and this super tile is smaller than others */
        if (m >= ((Ddesc->lmt/Ddesc->nrst)*Ddesc->nrst))
        {           
            last_c_size = Ddesc->lmt % Ddesc->nrst;
        }
    else
        {
            last_c_size = Ddesc->nrst;
        }
    pos += ((n % Ddesc->ncst) * last_c_size); /* pos is at (B, n)*/
    pos += (m % Ddesc->nrst); /* pos is at (m,n)*/

    //    printf("get tile (%d, %d) is at pos %d\n", m, n,  pos*Ddesc->bsiz);
    /************************************/
    return &(((double *) Ddesc->mat)[pos * Ddesc->bsiz]);
}

int dplasma_set_local_tile(DPLASMA_desc * Ddesc, int m, int n, void * buff)
{
    double * tile;
    tile = dplasma_get_tile(Ddesc, m, n);
    if (tile == NULL)
        {
            return 2;
        }
    memcpy( tile, buff, Ddesc->bsiz*sizeof(double));
    return 0;
}

#ifdef USE_MPI
static int nb_request(DPLASMA_desc * Ddesc, int rank)
{
    int nb_req = 0; //number of request
    int nbr_c;      // number of request per column
    int str;        // number of super tile per column
    int i, j, r;
    int colr, rowr;
    if (rank == 0)
        {
            for( i = 1; i < (Ddesc->GRIDcols * Ddesc->GRIDrows) ; i++)
                {
                    j = nb_request(Ddesc, i);
                    nb_req += j;
                    //                    printf("nb_request adjust for rank 0 to %d (+ %d requests to rank %d)\n", nb_req, j, i);
                }
            return nb_req;
        }
    colr = 0;
    rowr = 0;
    r = rank;
    /* find rowRANK for rank */
    while ( r >= Ddesc->GRIDcols)
        {
            rowr++;
            r = r - Ddesc->GRIDcols;
        }
    /* find colRANK */
    colr = r;

    
    str = Ddesc->lmt / Ddesc->nrst; // number of super tile in a column
    if (Ddesc->lmt % Ddesc->nrst)
        str++;

    str = str - rowr; 
    nbr_c = str / Ddesc->GRIDrows;
    if (str % Ddesc->GRIDrows)
        nbr_c++;

    i = colr * Ddesc->ncst;
    while(i < Ddesc->lnt)
        {
            if (i + Ddesc->ncst > Ddesc->lnt)
                {
                    nb_req = nb_req + ( nbr_c * (Ddesc->lnt - i));
                    return nb_req;
                }
            nb_req = nb_req + (nbr_c * Ddesc->ncst);
            i+=(Ddesc->ncst * Ddesc->GRIDcols); 
        }
    return nb_req;
}
#endif

#ifdef USE_MPI
int distribute_data(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc, MPI_Request ** reqs, int * req_count)
{

    int i, j, k, nb, pos, rank;
    int tile_size, str, stc;
    double * target;
    pos = 0;
    k = 0;
    *req_count = nb_request(Ddesc, Ddesc->mpi_rank);
    printf("number of request for proc %d: %d\n", Ddesc->mpi_rank, *req_count);
    *reqs = (MPI_Request *)malloc((*req_count) * sizeof(MPI_Request));
    if (NULL == *reqs)
        {
            printf("memory allocation failed\n");
            exit(2);
        }
    if (Ddesc->mpi_rank == 0)
        {
            str = Ddesc->lmt / Ddesc->nrst; // number of super tile in a column
            if (Ddesc->lmt % Ddesc->nrst)
                str++;
            stc = Ddesc->lnt / Ddesc->ncst; // number of super tile in a row
            if (Ddesc->lnt % Ddesc->ncst)
                stc++;
            for (i = 0 ; i < stc; i++) /* for each super tile column */
                for (j = 0 ; j < str ; j++) /* for each super tile row in that column */
                    {
                        rank = dplasma_get_rank_for_tile(Ddesc, j*Ddesc->nrst, i*Ddesc->ncst);
                        //  printf("tile (%d,%d) belongs to %d\n", j*Ddesc->nrst, i*Ddesc->ncst, rank);
                        if (rank == 0) /* this tile belongs to me */
                            {
                                tile_size = min(Ddesc->nrst, Ddesc->lmt-(j*Ddesc->nrst));
                                //printf("number of tile to copy at once: %d, ", tile_size);
                                target = (double *) plasma_A(Pdesc, j*Ddesc->nrst, i*Ddesc->ncst);
                                //printf(" -->tile (%d, %d) for self, memcpy at pos %d\n", j*Ddesc->nrst , i*Ddesc->ncst, pos );
                                for (nb = 0 ; nb < min(Ddesc->ncst, Ddesc->lnt - (i*Ddesc->ncst)) ; nb++)
                                    {
                                        //printf("start nb=%d, end %d, ", nb , min(Ddesc->ncst, Ddesc->lnt - (i*Ddesc->ncst)));
                                        //printf("target at %e\n", target[0]);
                                        memcpy(&(((double *)Ddesc->mat)[pos]), target , tile_size * Ddesc->bsiz*(sizeof(double)));
                                        //printf("--->memcpy %d eme tile to %d (%ld bytes)\n", nb +1, pos, tile_size * Ddesc->bsiz*(sizeof(double)));
                                        pos += (Ddesc->bsiz * tile_size);
                                        target += Ddesc->lmt * Ddesc->bsiz;
                                    }
                                
                                continue;
                            }
                        tile_size = min(Ddesc->nrst, Ddesc->lmt-(j*Ddesc->nrst));
                        target = (double *)plasma_A(Pdesc, j*Ddesc->nrst, i*Ddesc->ncst);
                        for (nb = 0 ; nb < min(Ddesc->ncst, Ddesc->lnt - (i*Ddesc->ncst)) ; nb++)
                            {                                        
                                MPI_Isend(target, tile_size * Ddesc->bsiz, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &((*reqs)[k]));
                                k++;
                                target += Ddesc->lmt * Ddesc->bsiz;
                            }
                    }
        }
    else /* mpi_rank != 0*/
    {
        pos = 0;
        k = 0;
        str = Ddesc->lmt / Ddesc->nrst; // number of super tile in a column
        if (Ddesc->lmt % Ddesc->nrst)
            str++;
        stc = Ddesc->lnt / Ddesc->ncst; // number of super tile in a row
        if (Ddesc->lnt % Ddesc->ncst)
            stc++;
        for (i = 0 ; i < stc; i++) /* for each super tile column */
            for (j = 0 ; j < str ; j++) /* for each super tile row in that column */
            {
                rank = dplasma_get_rank_for_tile(Ddesc, j*Ddesc->nrst, i*Ddesc->ncst);
                if (rank == Ddesc->mpi_rank) /* this tile belongs to me */
                {
                    tile_size = min(Ddesc->nrst, Ddesc->lmt-(j*Ddesc->nrst));
                    
                    for (nb = 0 ; nb < min(Ddesc->ncst, Ddesc->lnt - (i*Ddesc->ncst)) ; nb++)
                    {                                        
                        MPI_Irecv(&(((double*)Ddesc->mat)[pos]), tile_size * Ddesc->bsiz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &((*reqs)[k]));
                        pos += tile_size * Ddesc->bsiz;
                        k++;
                    }
                }
            }
    }
    return 0;

}
#endif    

int gather_data(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc) 
{
#ifdef USE_MPI
    int i, j,  rank;
    int req_count;
    MPI_Request * reqs;
    
    if ( Ddesc->mpi_rank == 0)
    {
        reqs = malloc(sizeof(MPI_Request) * Ddesc->lmt * Ddesc->lnt);
        req_count = 0;

        for (i = 0 ; i < Ddesc->lmt ; i++ )
            for(j = 0; j < Ddesc->lnt ; j++)
            {
                rank = dplasma_get_rank_for_tile(Ddesc, i, j);
                if (rank == 0)
                    memcpy(plasma_A(Pdesc, i, j ), dplasma_get_local_tile(Ddesc, i, j), Ddesc->bsiz * sizeof(double)) ;
                else
                    MPI_Irecv( plasma_A(Pdesc, i, j), Ddesc->bsiz, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &reqs[req_count++] );
            }
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        free(reqs);
    }
    else
    {
        for (i = 0 ; i < Ddesc->lmt ; i++ )
            for(j = 0; j < Ddesc->lnt ; j++)
            {
                rank = dplasma_get_rank_for_tile(Ddesc, i, j);
                if (rank == Ddesc->mpi_rank)
                    MPI_Send( dplasma_get_local_tile(Ddesc, i, j), Ddesc->bsiz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD );
            }
        
    }
    return 0;
#else
    fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__);
    return -1;
#endif    
}

#ifdef HEAVY_DEBUG
static void print_block(char * stri, int m, int n, double * block, int blength, int total_size)
{
    int i;
    printf("%s (%d,%d)\n", stri, m, n);
    for (i = 0 ; i < min(10, blength) ; i++ )
        printf("%e ", block[i]);
    printf("\n");
    i = total_size - blength;
    for ( ; i < min((total_size - blength) + 10, total_size) ; i++ )
        printf("%e ", block[i]);
    printf("\n\n");
}
#endif

void data_dist_verif(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc)
{
#ifdef USE_MPI
    int m, n, k, rank;
    double * buff;
    double * buff2;

    for (m = 0 ; m < Ddesc->lmt ; m++)
        for ( n = 0 ; n < Ddesc->lnt ; n++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            
            rank = dplasma_get_rank_for_tile(Ddesc, m, n);
            
            if (rank == 0 && Ddesc->mpi_rank == 0)/* this proc is rank 0 and handles the tile*/
            {
                buff = plasma_A(Pdesc, m, n);
                buff2 = dplasma_get_tile(Ddesc, m, n);
                for ( k = 0; k < Ddesc->bsiz ; k++)
                {
                    if(buff[k] != buff2[k])
                    {
                        printf("WARNING: tile(%d, %d) differ !\n", m, n);
                        break;
                    }
                }
                continue;
            }
#ifdef HEAVY_DEBUG
            if(Ddesc->mpi_rank == 0)
                print_block("orig tile ", m, n, plasma_A(Pdesc, m, n), Pdesc->nb, Pdesc->bsiz);
            
            
            if (Ddesc->mpi_rank == rank)
            {
                buff = dplasma_get_tile(Ddesc, m, n);
                printf("Check: rank %d has tile %d, %d\n", Ddesc->mpi_rank, m, n);
                print_block("Dist tile", m, n, buff, Ddesc->nb, Ddesc->bsiz);
            }
#else 
            if(Ddesc->mpi_rank == rank)
            {
                printf("Check: rank %d has tile %d, %d\n", Ddesc->mpi_rank, m, n);
            }
#endif
        }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Data verification ended\n");
#else
    fprintf(stderr, "MPI disabled, you should not call this function (%s) in this mode\n", __FUNCTION__);
#endif    
}

int data_dump(DPLASMA_desc * Ddesc){
    FILE * tmpf;
    int i, j, k;
    double * buf;
    tmpf = fopen("tmp_local_data_dump.txt", "w");
    if(NULL == tmpf)
        {
            perror("opening file: tmp_local_data_dump.txt" );
            return -1;
        }
    for (i = 0 ; i < Ddesc->lmt ; i++)
        for ( j = 0 ; j< Ddesc->lnt ; j++)
            {
                if (dplasma_get_rank_for_tile(Ddesc, i, j) == Ddesc->mpi_rank)
                    {
                        buf = (double*)dplasma_get_local_tile(Ddesc, i, j);
                        for (k = 0 ; k < Ddesc->bsiz ; k++)
                            {
                                fprintf(tmpf, "%e ", buf[k]);
                            }
                        fprintf(tmpf, "\n");
                    }
            }
    fclose(tmpf);
    return 0;
}

int plasma_dump(PLASMA_desc * Pdesc){
    FILE * tmpf;
    int i, j, k;
    double * buf;
    tmpf = fopen("tmp_plasma_data_dump.txt", "w");
    if(NULL == tmpf)
        {
            perror("opening file: tmp_plasma_data_dump.txt" );
            return -1;
        }
    for (i = 0 ; i < Pdesc->lmt ; i++)
        for ( j = 0 ; j< Pdesc->lnt ; j++)
            {
                buf = (double*)plasma_A(Pdesc, i, j);
                for (k = 0 ; k < Pdesc->bsiz ; k++)
                    {
                        fprintf(tmpf, "%e ", buf[k]);
                    }
                fprintf(tmpf, "\n");
            }
    fclose(tmpf);
    return 0;
}

#ifdef USE_MPI
int is_data_distributed(DPLASMA_desc * Ddesc, MPI_Request * reqs, int req_count)
{

    MPI_Status * stats;
    
    stats = malloc(req_count * sizeof(MPI_Status));
    MPI_Waitall(req_count, reqs, stats);
    return 1;

}
#endif    
