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

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    /* for tile (m,n), first find coordinate of process in
       process grid which possess the tile in block cyclic dist */
    /* (m,n) is in super-tile (str, stc)*/
    str = m / Ddesc->grid.strows;
    stc = n / Ddesc->grid.stcols;

    /* P(rr, cr) has the tile, compute the mpi rank*/
    rr = str % Ddesc->grid.rows;
    cr = stc % Ddesc->grid.cols;
    res = rr * Ddesc->grid.cols + cr;

    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
    /*            m, n, res, rr, cr, Ddesc->grid.rows, Ddesc->grid.cols); */
    return res;
}

static size_t twoDBC_compute_mempos_from_global_coordinates(two_dim_block_cyclic_t *Ddesc, int m, int n)
{
    int nb_elem_r, last_c_size;
    size_t pos;

    if ( Ddesc->super.storage == matrix_Tile ) {
        /* number of tiles per column of super-tile */
        nb_elem_r = Ddesc->nb_elem_r * Ddesc->grid.stcols;
        
        /* pos is currently at head of supertile (0xA) */
        pos = nb_elem_r * ((n / Ddesc->grid.stcols)/ Ddesc->grid.cols);
        
        /* tile is in the last column of super-tile */
        if (n >= ((Ddesc->super.lnt/Ddesc->grid.stcols) * Ddesc->grid.stcols )) {
            /* number of tile per super tile in last column */
            last_c_size = (Ddesc->super.lnt % Ddesc->grid.stcols) * Ddesc->grid.strows;
        }
        else {
            last_c_size = Ddesc->grid.stcols * Ddesc->grid.strows;
        }
        
        /* pos is at head of supertile (BxA) containing (m,n)  */
        pos += (last_c_size * ((m / Ddesc->grid.strows) / Ddesc->grid.rows ) );
        
        /* if tile (m,n) is in the last row of super tile and this super tile is smaller than others */
        if (m >= ((Ddesc->super.lmt/Ddesc->grid.strows)*Ddesc->grid.strows)) {
            last_c_size = Ddesc->super.lmt % Ddesc->grid.strows;
        }
        else {
            last_c_size = Ddesc->grid.strows;
        }
        pos += ((n % Ddesc->grid.stcols) * last_c_size); /* pos is at (B, n)*/
        pos += (m % Ddesc->grid.strows); /* pos is at (m,n)*/
    } 
    /* Lapack Storage */
    else {
        int local_m, local_n;

        /* Compute the local row */
        local_m = ( m / (Ddesc->grid.strows * Ddesc->grid.rows) ) * Ddesc->grid.strows;

        m = m % (Ddesc->grid.strows * Ddesc->grid.rows);
        
        assert( m / Ddesc->grid.strows == Ddesc->grid.rrank);
        local_m += m % Ddesc->grid.strows;
        
        /* Compute the local column */
        local_n = ( n / (Ddesc->grid.stcols * Ddesc->grid.cols) ) * Ddesc->grid.stcols;

        n = n % (Ddesc->grid.stcols * Ddesc->grid.cols);
        
        assert( n / Ddesc->grid.stcols == Ddesc->grid.crank);
        local_n += n % Ddesc->grid.stcols;

        pos = local_n * Ddesc->nb_elem_r + local_m;
    }
        
    pos *= (size_t)Ddesc->super.bsiz * dague_datadist_getsizeoftype(Ddesc->super.mtype);

    return pos;
}

static void * twoDBC_get_local_tile(dague_ddesc_t * desc, ...)
{
    size_t pos;
    int m, n;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    
    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == twoDBC_get_rank_for_tile(desc, m, n));
#endif

    pos = twoDBC_compute_mempos_from_global_coordinates(Ddesc, m, n);

    return &(((char *) Ddesc->mat)[pos]);
}

static int32_t twoDBC_get_vpid(dague_ddesc_t *desc, ...)
{
    size_t pos;
    int m, n;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;
    
    /* Get coordinates */
    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

#if defined(DISTRIBUTED)
    assert(desc->myrank == twoDBC_get_rank_for_tile(desc, m, n));
#endif

    pos = twoDBC_compute_mempos_from_global_coordinates(Ddesc, m, n);
    /* pos is expressed in bytes */
    assert( (pos % (Ddesc->super.bsiz * dague_datadist_getsizeoftype(Ddesc->super.mtype))) == 0 );
    return tiled_matrix_get_vpid(&Ddesc->super, pos / (Ddesc->super.bsiz * dague_datadist_getsizeoftype(Ddesc->super.mtype)));
}

#ifdef DAGUE_PROF_TRACE
/* return a unique key (unique only for the specified dague_ddesc) associated to a data */
static uint32_t twoDBC_data_key(struct dague_ddesc *desc, ...)
{
    unsigned int m, n;
    two_dim_block_cyclic_t * Ddesc;
    va_list ap;
    Ddesc = (two_dim_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += Ddesc->super.i / Ddesc->super.mb;
    n += Ddesc->super.j / Ddesc->super.nb;

    return ((n * Ddesc->super.lmt) + m);
}

/* return a string meaningful for profiling about data */
static int  twoDBC_key_to_string(struct dague_ddesc * desc, uint32_t datakey, char * buffer, uint32_t buffer_size)
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
                               int P )
{
    int temp;
    int nbstile_r;
    int nbstile_c;
    int Q;

    /* Initialize the dague_ddesc */
    {
        dague_ddesc_t *o = &(Ddesc->super.super);

        o->nodes  = nodes;
        o->cores  = cores;
        o->myrank = myrank;
        
        o->rank_of       = twoDBC_get_rank_for_tile;
        o->data_of       = twoDBC_get_local_tile;
        o->vpid_of       = twoDBC_get_vpid;
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

    assert((nodes % P) == 0);
    Q = nodes / P;
    grid_2Dcyclic_init(&Ddesc->grid, myrank, P, Q, nrst, ncst);

    /* Compute the number of rows of super-tile */
    nbstile_r = Ddesc->super.lmt / Ddesc->grid.strows;
    if((Ddesc->super.lmt % Ddesc->grid.strows) != 0)
        nbstile_r++;

    /* Compute the number of colums of super-tile */
    nbstile_c = Ddesc->super.lnt / Ddesc->grid.stcols;
    if((Ddesc->super.lnt % Ddesc->grid.stcols) != 0)
        nbstile_c++;

    /* Compute the number of rows handled by the local process */
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

    /* Compute the number of columns handled by the local process */
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

    /* Total number of tiles stored locally */
    Ddesc->super.nb_local_tiles = Ddesc->nb_elem_r * Ddesc->nb_elem_c;

    DEBUG(("two_dim_block_cyclic_init: \n"
           "      Ddesc = %p, mtype = %d, nodes = %u, cores = %u, myrank = %d, \n"
           "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
           "      nrst = %d, ncst = %d, P = %d, Q = %d\n",
           Ddesc, Ddesc->super.mtype, Ddesc->super.super.nodes, Ddesc->super.super.cores,
           Ddesc->super.super.myrank, 
           Ddesc->super.mb, Ddesc->super.nb,
           Ddesc->super.lm, Ddesc->super.ln,
           Ddesc->super.i,  Ddesc->super.j,
           Ddesc->super.m,  Ddesc->super.n, 
           Ddesc->grid.strows, Ddesc->grid.stcols, 
           P, Q));
}

#ifdef HAVE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm){
    return MPI_File_open(comm, filename, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, handle);
}

int close_matrix_file(MPI_File * handle){
    return MPI_File_close(handle);
}

#endif /* HAVE_MPI */
