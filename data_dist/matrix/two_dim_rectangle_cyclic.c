/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
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
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

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
    str = m / Ddesc->grid.strows; /* (m,n) is in super-tile (str, stc)*/
    stc = n / Ddesc->grid.stcols;

    rr = str % Ddesc->grid.rows;
    cr = stc % Ddesc->grid.cols;
    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * Ddesc->grid.cols + cr;
    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
/*            m, n, res, rr, cr, Ddesc->grid.rows, Ddesc->grid.cols); */
    return res;
}


static void * twoDBC_get_local_tile(dague_ddesc_t * desc, ...)
{
    int pos, m, n;
    int nb_elem_r, last_c_size;
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

    nb_elem_r = Ddesc->nb_elem_r * Ddesc->grid.stcols; /* number of tiles per column of super-tile */

    pos = nb_elem_r * ((n / Ddesc->grid.stcols)/ Ddesc->grid.cols); /* pos is currently at head of supertile (0xA) */

    if (n >= ((Ddesc->super.lnt/Ddesc->grid.stcols) * Ddesc->grid.stcols )) /* tile is in the last column of super-tile */
        {
            last_c_size = (Ddesc->super.lnt % Ddesc->grid.stcols) * Ddesc->grid.strows; /* number of tile per super tile in last column */
        }
    else
        {
            last_c_size = Ddesc->grid.stcols * Ddesc->grid.strows;
        }
    pos += (last_c_size * ((m / Ddesc->grid.strows) / Ddesc->grid.rows ) ); /* pos is at head of supertile (BxA) containing (m,n)  */

    /* if tile (m,n) is in the last row of super tile and this super tile is smaller than others */
    if (m >= ((Ddesc->super.lmt/Ddesc->grid.strows)*Ddesc->grid.strows))
        {
            last_c_size = Ddesc->super.lmt % Ddesc->grid.strows;
        }
    else
        {
            last_c_size = Ddesc->grid.strows;
        }
    pos += ((n % Ddesc->grid.stcols) * last_c_size); /* pos is at (B, n)*/
    pos += (m % Ddesc->grid.strows); /* pos is at (m,n)*/

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

void two_dim_block_cyclic_init(two_dim_block_cyclic_t * Ddesc, 
                               enum matrix_type mtype, 
                               int nodes, int cores, int myrank, 
                               int mb,   int nb,   /* Tile size */
                               int lm,   int ln,   /* Global matrix size (what is stored)*/
                               int i,    int j,    /* Staring point in the global matrix */
                               int m,    int n,    /* Submatrix size (the one concerned by the computation */
                               int nrst, int ncst, /* Super-tiling size */
                               int process_GridRows )
{
    int temp;
    int nbstile_r;
    int nbstile_c;

    /* Initialize the dague_ddesc */
    {
        dague_ddesc_t *o = &(Ddesc->super.super);

        o->nodes  = nodes;
        o->cores  = cores;
        o->myrank = myrank;
        
        o->rank_of       = twoDBC_get_rank_for_tile;
        o->data_of       = twoDBC_get_local_tile;
#if defined(DAGUE_PROF_TRACE)
        o->data_key      = twoDBC_data_key;
        o->key_to_string = twoDBC_key_to_string;
        o->key_dim       = NULL;
        o->key           = NULL;
#endif
    }

    /* Initialize the tiled_matrix descriptor */
    tiled_matrix_desc_init( &(Ddesc->super), mtype, matrix_Tile,
                            mb, nb, lm, ln, i, j, m, n);

    assert((nodes % process_GridRows) == 0);
    grid_2Dcyclic_init(&Ddesc->grid, myrank, process_GridRows, nodes/process_GridRows, nrst, ncst);

    /* computing the number of rows of super-tile */
    nbstile_r = Ddesc->super.lmt / Ddesc->grid.strows;
    if((Ddesc->super.lmt % Ddesc->grid.strows) != 0)
        nbstile_r++;

    /* computing the number of colums of super-tile*/
    nbstile_c = Ddesc->super.lnt / Ddesc->grid.stcols;
    if((Ddesc->super.lnt % Ddesc->grid.stcols) != 0)
        nbstile_c++;

#if 0
    if ( Ddesc->grid.rows > nbstile_r || Ddesc->grid.cols > nbstile_c)
        {
            printf("The process grid chosen is %ux%u, block distribution choosen is %u, %u : cannot generate matrix \n",
                   Ddesc->grid.rows, Ddesc->grid.cols, nbstile_r, nbstile_c);
            exit(-1);
        }
    // printf("matrix to be generated distributed by block of %d x %d tiles \n", nbstile_r, nbstile_c);
#endif

    /* find the number of tiles this process will handle */
    Ddesc->nb_elem_r = 0;
    temp = Ddesc->grid.rrank * Ddesc->grid.strows; /* row coordinate of the first tile to handle */
    while ( temp < Ddesc->super.lmt)
        {
            if ( (temp  + (Ddesc->grid.strows)) < Ddesc->super.lmt)
                {
                    Ddesc->nb_elem_r += (Ddesc->grid.strows);
                    temp += ((Ddesc->grid.rows) * (Ddesc->grid.strows));
                    continue;
                }
            Ddesc->nb_elem_r += ((Ddesc->super.lmt) - temp);
            break;
        }

    Ddesc->nb_elem_c = 0;
    temp = Ddesc->grid.crank * Ddesc->grid.stcols;
    while ( temp < Ddesc->super.lnt)
        {
            if ( (temp  + (Ddesc->grid.stcols)) < Ddesc->super.lnt)
                {
                    Ddesc->nb_elem_c += (Ddesc->grid.stcols);
                    temp += (Ddesc->grid.cols) * (Ddesc->grid.stcols);
                    continue;
                }
            Ddesc->nb_elem_c += ((Ddesc->super.lnt) - temp);
            break;
        }
/*    printf("process %d(%d,%d) handles %d x %d tiles\n",
        myrank, Ddesc->grid.rrank, Ddesc->grid.crank, Ddesc->nb_elem_r, Ddesc->nb_elem_c);*/

    Ddesc->super.nb_local_tiles = Ddesc->nb_elem_r * Ddesc->nb_elem_c;


    //   printf("Process %u: Ddesc->nb_elem_r = %u, Ddesc->nb_elem_c = %u, Ddesc->super.bsiz = %u, Ddesc->super.mtype = %zu\n", myrank, Ddesc->nb_elem_r, Ddesc->nb_elem_c, Ddesc->super.bsiz, (size_t) Ddesc->super.mtype);

    /* Ddesc->mat = dague_data_allocate((size_t)Ddesc->nb_elem_r * (size_t)Ddesc->nb_elem_c * (size_t)Ddesc->super.bsiz * (size_t) Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
            }*/

    DEBUG(("two_dim_block_cyclic_init: Ddesc = %p, mtype = %zu, nodes = %u, cores = %u, myrank = %d, mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, nrst = %d, ncst = %d, process_GridRows = %d\n",
           Ddesc, (size_t) Ddesc->super.mtype, Ddesc->super.super.nodes, Ddesc->super.super.cores,
           Ddesc->super.super.myrank, Ddesc->super.mb, Ddesc->super.nb,
           Ddesc->super.lm,  Ddesc->super.ln,
           Ddesc->super.i, Ddesc->super.j,
           Ddesc->super.m, Ddesc->super.n, Ddesc->grid.strows, Ddesc->grid.stcols, Ddesc->grid.rows));
}

#ifdef HAVE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm){
    return MPI_File_open(comm, filename, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, handle);
}

int close_matrix_file(MPI_File * handle){
    return MPI_File_close(handle);
}




#endif /* HAVE_MPI */
