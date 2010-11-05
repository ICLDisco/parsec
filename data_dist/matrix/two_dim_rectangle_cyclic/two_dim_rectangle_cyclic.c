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

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "dague.h"
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
    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += ((tiled_matrix_desc_t *)desc)->i;
    n += ((tiled_matrix_desc_t *)desc)->j;
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

    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += ((tiled_matrix_desc_t *)desc)->i;
    n += ((tiled_matrix_desc_t *)desc)->j;

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


#ifdef DAGUE_PROF_TRACE
static uint32_t twoDBC_data_key(struct dague_ddesc *desc, ...) /* return a unique key (unique only for the specified dague_ddesc) associated to a data */
{
    unsigned int m, n;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    return ((n * Ddesc->super.lmt) + m);
}
static int  twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size) /* return a string meaningful for profiling about data */
{
    two_dim_block_cyclic_t * Ddesc;
    unsigned int row, column;
    int res;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    column = datakey / Ddesc->super.lmt;
    row = datakey % Ddesc->super.lmt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", row, column);
    if (res < 0)
        {
            printf("error in key_to_string for tile (%u, %u) key: %u\n", row, column, datakey);
        }
    return res;
}
#endif /* DAGUE_PROF_TRACE */

void two_dim_block_cyclic_init(two_dim_block_cyclic_t * Ddesc, enum matrix_type mtype, unsigned int nodes, unsigned int cores, unsigned int myrank, unsigned int mb, unsigned int nb, unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, unsigned int m, unsigned int n, unsigned int nrst, unsigned int ncst, unsigned int process_GridRows )
{
    unsigned int temp;
    unsigned int nbstile_r;
    unsigned int nbstile_c;

    // Filling matrix description woth user parameter
    Ddesc->super.super.nodes = nodes ;
    Ddesc->super.super.cores = cores ;
    Ddesc->super.super.myrank = myrank ;
    Ddesc->super.mtype = mtype;
    Ddesc->super.mb = mb;
    Ddesc->super.nb = nb;
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

    Ddesc->super.nb_local_tiles = Ddesc->nb_elem_r * Ddesc->nb_elem_c;


    //   printf("Process %u: Ddesc->nb_elem_r = %u, Ddesc->nb_elem_c = %u, Ddesc->super.bsiz = %u, Ddesc->super.mtype = %zu\n", myrank, Ddesc->nb_elem_r, Ddesc->nb_elem_c, Ddesc->super.bsiz, (size_t) Ddesc->super.mtype);

    /* Ddesc->mat = dague_data_allocate((size_t)Ddesc->nb_elem_r * (size_t)Ddesc->nb_elem_c * (size_t)Ddesc->super.bsiz * (size_t) Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
            }*/
    Ddesc->super.super.rank_of =  twoDBC_get_rank_for_tile;
    Ddesc->super.super.data_of =  twoDBC_get_local_tile;
#ifdef DAGUE_PROF_TRACE
    Ddesc->super.super.data_key = twoDBC_data_key;
    Ddesc->super.super.key_to_string = twoDBC_key_to_string;
    Ddesc->super.super.key = NULL;
    asprintf(&Ddesc->super.super.key_dim, "(%u, %u)", Ddesc->super.mt, Ddesc->super.nt);
#endif /* DAGUE_PROF_TRACE */
#ifdef DAGUE_DEBUG
     printf("two_dim_block_cyclic_init: Ddesc = %p, mtype = %zu, nodes = %u, cores = %u, myrank = %u, mb = %u, nb = %u, lm = %u, ln = %u, i = %u, j = %u, m = %u, n = %u, nrst = %u, ncst = %u, process_GridRows = %u\n", Ddesc, (size_t) Ddesc->super.mtype, Ddesc->super.super.nodes, Ddesc->super.super.cores, Ddesc->super.super.myrank, Ddesc->super.mb, Ddesc->super.nb, Ddesc->super.lm,  Ddesc->super.ln,  Ddesc->super.i, Ddesc->super.j, Ddesc->super.m, Ddesc->super.n, Ddesc->nrst, Ddesc->ncst, Ddesc->GRIDrows);

#endif /* DAGUE_DEBUG*/
}

int twoDBC_tolapack(two_dim_block_cyclic_t *Mdesc, void* A, int lda)
{
    /* switch(Mdesc->super.mtype) { */
    /* case matrix_RealFloat: */
    /*     //twoDBC_stolapack(Mdesc, (float*)A, lda); */
    /*     break; */
    /* case matrix_RealDouble: */
    /*     twoDBC_to_lapack_double(Mdesc, (double*)A, lda); */
    /*     break; */
    /* case matrix_ComplexFloat: */
    /*     //twoDBC_ctolapack(Mdesc, (Dague_Complex32_t*)A, lda); */
    /*     break; */
    /* case matrix_ComplexDouble: */
    /*     //twoDBC_ztolapack(Mdesc, (Dague_Complex64_t*)A, lda); */
    /*     break; */
    /* default: */
           twoDBC_dtolapack(Mdesc, (double*)A, lda);
    /*     printf("The matrix type is not handle by this function\n"); */
    /*     return -1; */
    /* } */

  return 0;
}

#ifdef HAVE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm){
    return MPI_File_open(comm, filename, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, handle);
}

int close_matrix_file(MPI_File * handle){
    return MPI_File_close(handle);
}




#endif /* HAVE_MPI */
