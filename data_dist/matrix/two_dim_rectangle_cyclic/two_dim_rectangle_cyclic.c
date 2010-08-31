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

#ifdef USE_MPI
#include <mpi.h>
#endif /* USE_MPI */

#include "dague_config.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

static uint32_t twoDBC_get_rank_for_tile(dague_ddesc_t * desc, ...)
{
    unsigned int stc, cr, m, n;
    unsigned int str, rr;
    unsigned int res;
    va_list ap;
    two_dim_block_cyclic_t * Ddesc;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
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



static void * twoDBC_get_local_tile(dague_ddesc_t * desc, ...)
{
    unsigned int pos, m, n;
    unsigned int nb_elem_r, last_c_size;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);
#ifdef DISTRIBUTED
    //   if ( desc->myrank != twoDBC_get_rank_for_tile(desc, m, n) )
    //  {
    //      printf("Tile (%d, %d) is looked for on process %d but is not local\n", m, n, desc->myrank);
    assert(desc->myrank == twoDBC_get_rank_for_tile(desc, m, n));
            //  }
#endif /* DISTRIBUTED */

    /**********************************/

    nb_elem_r = Ddesc->nb_elem_r * Ddesc->ncst; /* number of tiles per column of super-tile */

    pos = nb_elem_r * ((n / Ddesc->ncst)/ Ddesc->GRIDcols); /* pos is currently at head of supertile (0xA) */

    if (n >= ((Ddesc->super.lnt/Ddesc->ncst) * Ddesc->ncst )) /* tile is in the last column of super-tile */
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


void two_dim_block_cyclic_init(two_dim_block_cyclic_t * Ddesc, enum matrix_type mtype, unsigned int nodes, unsigned int cores, unsigned int myrank, unsigned int mb, unsigned int nb, unsigned int ib, unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, unsigned int m, unsigned int n, unsigned int nrst, unsigned int ncst, unsigned int process_GridRows )
{
    unsigned int temp;
    unsigned int nbstile_r;
    unsigned int nbstile_c;

#ifdef DAGUE_DEBUG
    printf("two_dim_block_cyclic_init: Ddesc = %p, mtype = %zu, nodes = %u, cores = %u, myrank = %u, mb = %u, nb = %u, ib = %u, lm = %u, ln = %u, i = %u, j = %u, m = %u, n = %u, nrst = %u, ncst = %u, process_GridRows = %u\n", Ddesc, (size_t) mtype, nodes, cores, myrank,  mb,  nb,  ib,  lm,  ln,  i,  j,  m,  n,  nrst,  ncst,  process_GridRows);
#endif


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
    Ddesc->super.mt = ((Ddesc->super.m)%(Ddesc->super.mb)==0) ? ((Ddesc->super.m)/(Ddesc->super.mb)) : ((Ddesc->super.m)/(Ddesc->super.mb) + 1);
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
            printf("The process grid chosen is %ux%u, block distribution choosen is %u, %u : cannot generate matrix \n",
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
    printf("Process %u: Ddesc->nb_elem_r = %u, Ddesc->nb_elem_c = %u, Ddesc->super.bsiz = %u, Ddesc->super.mtype = %zu\n", myrank, Ddesc->nb_elem_r, Ddesc->nb_elem_c, Ddesc->super.bsiz, (size_t) Ddesc->super.mtype);
    Ddesc->mat = dague_allocate_matrix((size_t)Ddesc->nb_elem_r * (size_t)Ddesc->nb_elem_c * (size_t)Ddesc->super.bsiz * (size_t) Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
        }
    Ddesc->super.super.rank_of =  twoDBC_get_rank_for_tile;
    Ddesc->super.super.data_of =  twoDBC_get_local_tile;
}


#ifdef USE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm){
    return MPI_File_open(comm, filename, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, handle);
}

int close_matrix_file(MPI_File * handle){
    return MPI_File_close(handle);
}




#endif /* USE_MPI */
