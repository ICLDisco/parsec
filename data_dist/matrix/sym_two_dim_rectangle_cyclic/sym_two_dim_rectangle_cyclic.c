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
#include <limits.h>

#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"

static uint32_t sym_twoDBC_get_rank_for_tile(dague_ddesc_t * desc, ...)
{
    int rr, cr, m, n;
    int res;
    va_list ap;
    sym_two_dim_block_cyclic_t * Ddesc;
    Ddesc = (sym_two_dim_block_cyclic_t *) desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    if ( m < n )
        {
    //        printf("Tried to get rank for tile (%d,%d)\n", m,n);
            return UINT_MAX;
        }
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

static void * sym_twoDBC_get_local_tile(dague_ddesc_t * desc, ...)
{
    int pos, m, n;
    int nb_elem, nb_elem_col, column;
    sym_two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (sym_two_dim_block_cyclic_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    /*if ( desc->myrank != sym_twoDBC_get_rank_for_tile(desc, m, n) )
        {
            printf("Tile (%d, %d) is looked for on process %u but is not local\n", m, n, desc->myrank);*/
            assert(desc->myrank == sym_twoDBC_get_rank_for_tile(desc, m, n));
       /* }*/
    

    /**********************************/
    pos = 0; /* current position (as number of tile) in the buffer */
    column = Ddesc->colRANK; /* tile column considered */
    nb_elem_col = (Ddesc->super.lmt) / (Ddesc->GRIDrows); //nb of tile associated to that proc in a full column
    if( (Ddesc->super.lmt) % (Ddesc->GRIDrows) > Ddesc->rowRANK )
        nb_elem_col++;
    
    while(column != n) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
        {
            nb_elem = column / (Ddesc->GRIDrows);
            if ( (column % (Ddesc->GRIDrows)) > Ddesc->rowRANK)
                nb_elem++;

            pos += (nb_elem_col - nb_elem);
            column += Ddesc->GRIDcols;
            
        }

    pos += ((m - n) / (Ddesc->GRIDrows));

    /**********************************/
    //printf("get tile (%d, %d) is at pos %d\t(ptr %p, base %p)\n", m, n, pos*Ddesc->bsiz,&(((double *) Ddesc->mat)[pos * Ddesc->bsiz]), Ddesc->mat);
    /************************************/
    return &(((char *) Ddesc->mat)[pos * Ddesc->super.bsiz * Ddesc->super.mtype]);
}


void sym_two_dim_block_cyclic_init(sym_two_dim_block_cyclic_t * Ddesc, enum matrix_type mtype, int nodes, int cores, int myrank, int mb, int nb, int ib, int lm, int ln, int i, int j, int m, int n, int process_GridRows )
{
    int nb_elem, nb_elem_col, column, total;

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


    /* find the number of tiles this process will handle */

    total = 0; /* number of tiles handled by the process */
    column = Ddesc->colRANK; /* tile column considered */
    nb_elem_col = (Ddesc->super.lmt) / (Ddesc->GRIDrows); //nb of tile associated to that proc in a full column
    if( (Ddesc->super.lmt) % (Ddesc->GRIDrows) > Ddesc->rowRANK )
        nb_elem_col++;
    
    while(column < Ddesc->super.lnt) /* for each column of tiles in memory before searched element, compute the number of tile for displacement */
        {
            nb_elem = column / (Ddesc->GRIDrows);
            if ( (column % (Ddesc->GRIDrows)) > Ddesc->rowRANK)
                nb_elem++;

            total += (nb_elem_col - nb_elem);
            column += Ddesc->GRIDcols;
        }

    
    /*  printf("process %d(%d,%d) handles %d x %d tiles\n",
        Ddesc->mpi_rank, Ddesc->rowRANK, Ddesc->colRANK, Ddesc->nb_elem_r, Ddesc->nb_elem_c);*/

    /* Allocate memory for matrices in block layout */
    Ddesc->mat = dague_allocate_matrix(total * Ddesc->super.bsiz * Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
        }
    Ddesc->super.super.rank_of =  sym_twoDBC_get_rank_for_tile;
    Ddesc->super.super.data_of =  sym_twoDBC_get_local_tile;
}
