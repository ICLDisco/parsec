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
#include <limits.h>

#include "dague.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

static uint32_t sym_twoDBC_get_rank_for_tile(dague_ddesc_t * desc, ...)
{
    unsigned int rr, cr, m, n;
    unsigned int res;
    va_list ap;
    sym_two_dim_block_cyclic_t * Ddesc;
    Ddesc = (sym_two_dim_block_cyclic_t *) desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);
    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += ((tiled_matrix_desc_t *)desc)->i;
    n += ((tiled_matrix_desc_t *)desc)->j;
    
    if ( ((Ddesc->uplo == MatrixLower) && (m < n)) ||  ((Ddesc->uplo == MatrixUpper) && (m > n)) )
        {
            //        printf("Tried to get rank for tile (%d,%d)\n", m,n);
            return UINT_MAX;
        }
    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    
    rr = m % Ddesc->grid.rows;
    cr = n % Ddesc->grid.cols;
    /* P(rr, cr) has the tile, compute the mpi rank*/
    res = rr * Ddesc->grid.cols + cr;
    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
/*            m, n, res, rr, cr, Ddesc->grid.rows, Ddesc->grid.cols); */
    return res;
}

static void * sym_twoDBC_get_local_tile(dague_ddesc_t * desc, ...)
{
    unsigned int pos, m, n;
    unsigned int nb_elem, nb_elem_col, column;
    sym_two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* asking for tile (m,n) in submatrix, compute which tile it corresponds in full matrix */
    m += ((tiled_matrix_desc_t *)desc)->i;
    n += ((tiled_matrix_desc_t *)desc)->j;
    
    /*if ( desc->myrank != sym_twoDBC_get_rank_for_tile(desc, m, n) )
        {
            printf("Tile (%d, %d) is looked for on process %u but is not local\n", m, n, desc->myrank);*/
    assert(desc->myrank == sym_twoDBC_get_rank_for_tile(desc, m, n));
       /* }*/
    

    /**********************************/
    if(Ddesc->uplo == MatrixLower )
        {
            pos = 0; /* current position (as number of tile) in the buffer */
            column = Ddesc->grid.crank; /* tile column considered */
            nb_elem_col = (Ddesc->super.lmt) / (Ddesc->grid.rows); //nb of tile associated to that proc in a full column
            if( (Ddesc->super.lmt) % (Ddesc->grid.rows) > Ddesc->grid.rrank )
                nb_elem_col++;
            
            while(column != n) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
                {
                    nb_elem = column / (Ddesc->grid.rows);
                    if ( (column % (Ddesc->grid.rows)) > Ddesc->grid.rrank)
                        nb_elem++;
                    
                    pos += (nb_elem_col - nb_elem);
                    column += Ddesc->grid.cols;
                    
                }
            
            pos += ((m - n) / (Ddesc->grid.rows));
        }
    else /* Ddesc->uplo == MatrixUpper */
        {
            pos = 0; /* current position (as number of tile) in the buffer */
            column = Ddesc->grid.crank; /* tile column considered */
                        
            while(column != n) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
                {
                    nb_elem = (column + 1) / (Ddesc->grid.rows);
                    if ( ( (column + 1) % (Ddesc->grid.rows)) > Ddesc->grid.rrank)
                        nb_elem++;
                    
                    pos += nb_elem;
                    column += Ddesc->grid.cols;
                    
                }
            
            pos += (m / (Ddesc->grid.rows));
            
        }
    /**********************************/
    //printf("get tile (%d, %d) is at pos %d\t(ptr %p, base %p)\n", m, n, pos*Ddesc->bsiz,&(((double *) Ddesc->mat)[pos * Ddesc->bsiz]), Ddesc->mat);
    /************************************/
    return &(((char *) Ddesc->mat)[pos * Ddesc->super.bsiz * Ddesc->super.mtype]);
}



#ifdef DAGUE_PROF_TRACE
static uint32_t sym_twoDBC_data_key(struct dague_ddesc *desc, ...) /* return a unique key (unique only for the specified dague_ddesc) associated to a data */
{
    unsigned int m, n;
    sym_two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    return ((n * Ddesc->super.lmt) + m);    
}
static int  sym_twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size) /* return a string meaningful for profiling about data */
{
    sym_two_dim_block_cyclic_t * Ddesc;    
    unsigned int row, column;
    int res;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;
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


void sym_two_dim_block_cyclic_init(sym_two_dim_block_cyclic_t * Ddesc, enum matrix_type mtype, unsigned int nodes, unsigned int cores, unsigned int myrank, unsigned int mb, unsigned int nb, unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, unsigned int m, unsigned int n, unsigned int process_GridRows, int uplo )
{
    unsigned int nb_elem, nb_elem_col, column, total;

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

    assert((nodes % process_GridRows) == 0);
    grid_2Dcyclic_init(&Ddesc->grid, myrank, process_GridRows, nodes/process_GridRows, 1, 1);

    // Matrix derived parameters
    Ddesc->super.lmt = ((Ddesc->super.lm)%(Ddesc->super.mb)==0) ? ((Ddesc->super.lm)/(Ddesc->super.mb)) : ((Ddesc->super.lm)/(Ddesc->super.mb) + 1);
    Ddesc->super.lnt = ((Ddesc->super.ln)%(Ddesc->super.nb)==0) ? ((Ddesc->super.ln)/(Ddesc->super.nb)) : ((Ddesc->super.ln)/(Ddesc->super.nb) + 1);
    Ddesc->super.bsiz =  Ddesc->super.mb * Ddesc->super.nb;

    // Submatrix parameters    
    Ddesc->super.mt = ((Ddesc->super.m)%(Ddesc->super.mb)==0) ? ((Ddesc->super.m)/(Ddesc->super.mb)) : ((Ddesc->super.m)/(Ddesc->super.mb) + 1);
    Ddesc->super.nt = ((Ddesc->super.n)%(Ddesc->super.nb)==0) ? ((Ddesc->super.n)/(Ddesc->super.nb)) : ((Ddesc->super.n)/(Ddesc->super.nb) + 1);
    
    Ddesc->uplo = uplo;
    /* find the number of tiles this process will handle */

    total = 0; /* number of tiles handled by the process */
    column = Ddesc->grid.crank; /* tile column considered */
    nb_elem_col = (Ddesc->super.lmt) / (Ddesc->grid.rows); //nb of tile associated to that proc in a full column
    if( (Ddesc->super.lmt) % (Ddesc->grid.rows) > Ddesc->grid.rrank )
        nb_elem_col++;
    
    while(column < Ddesc->super.lnt) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
        {
            nb_elem = column / (Ddesc->grid.rows);
            if ( (column % (Ddesc->grid.rows)) > Ddesc->grid.rrank)
                nb_elem++;

            total += (nb_elem_col - nb_elem);
            column += Ddesc->grid.cols;
        }

    
    /*  printf("process %d(%d,%d) handles %d x %d tiles\n",
        Ddesc->mpi_rank, Ddesc->grid.rrank, Ddesc->grid.crank, Ddesc->nb_elem_r, Ddesc->nb_elem_c);*/

    /* Allocate memory for matrices in block layout */
    //printf("Process %u allocates %u tiles\n", myrank, total);
    Ddesc->super.nb_local_tiles = total;

    Ddesc->super.super.rank_of =  sym_twoDBC_get_rank_for_tile;
    Ddesc->super.super.data_of =  sym_twoDBC_get_local_tile;
#if defined(DAGUE_PROF_TRACE)
    Ddesc->super.super.data_key = sym_twoDBC_data_key;
    Ddesc->super.super.key_to_string = sym_twoDBC_key_to_string;
    Ddesc->super.super.key = NULL;
    asprintf(&Ddesc->super.super.key_dim, "(%u, %u)", Ddesc->super.mt, Ddesc->super.nt);
#endif /* DAGUE_PROF_TRACE */

}
