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

#include "data_dist/matrix/tabular_distribution/tabular_distribution.h"

static uint32_t td_get_rank_for_tile(dague_ddesc_t * desc, ...)
{
    int m, n, res;
    va_list ap;
    tabular_distribution_t * Ddesc;
    Ddesc = (tabular_distribution_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    res = (Ddesc->super.lmt * n) + m;
    return Ddesc->tiles_table[res].rank;
}



static void * td_get_local_tile(dague_ddesc_t * desc, ...)
{
    int res, m, n;
    tabular_distribution_t * Ddesc;
    va_list ap;
    Ddesc = (tabular_distribution_t *)desc;
    va_start(ap, desc);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
#ifdef DISTRIBUTED
    assert(desc->myrank == td_get_rank_for_tile(desc, m, n));    
#endif /* DISTRIBUTED */

    res = (Ddesc->super.lmt * n) + m;
    
    return  Ddesc->tiles_table[res].tile;
}

void tabular_distribution_init(tabular_distribution_t * Ddesc, enum matrix_type mtype, uint32_t nodes, uint32_t cores, uint32_t myrank, uint32_t mb, uint32_t nb, uint32_t ib, uint32_t lm, uint32_t ln, uint32_t i, uint32_t j, uint32_t m, uint32_t n, uint32_t * table )
{
    int res;
    

    // Filling matrix description with user parameter
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

    // Matrix derived parameters
    Ddesc->super.lmt = ((Ddesc->super.lm)%(Ddesc->super.mb)==0) ? ((Ddesc->super.lm)/(Ddesc->super.mb)) : ((Ddesc->super.lm)/(Ddesc->super.mb) + 1);
    Ddesc->super.lnt = ((Ddesc->super.ln)%(Ddesc->super.nb)==0) ? ((Ddesc->super.ln)/(Ddesc->super.nb)) : ((Ddesc->super.ln)/(Ddesc->super.nb) + 1);
    Ddesc->super.bsiz =  Ddesc->super.mb * Ddesc->super.nb;

    // Submatrix parameters    
    Ddesc->super.mt = ((Ddesc->super.m)%(Ddesc->super.mb)==0) ? ((Ddesc->super.m)/(Ddesc->super.nb)) : ((Ddesc->super.m)/(Ddesc->super.nb) + 1);
    Ddesc->super.nt = ((Ddesc->super.n)%(Ddesc->super.nb)==0) ? ((Ddesc->super.n)/(Ddesc->super.nb)) : ((Ddesc->super.n)/(Ddesc->super.nb) + 1);
    

    /* allocate the table*/
    Ddesc->tiles_table = malloc((Ddesc->super.lmt) * (Ddesc->super.lnt) * sizeof(tile_elem_t));

    /* affecting corresponding ranks and allocating tile memory */
    
    for (res = 0 ; res < (Ddesc->super.lmt) * (Ddesc->super.lnt) ; res++)
        {
            Ddesc->tiles_table[res].rank = table[res];
            if(table[res] == myrank) /* this tile belongs to me, allocating memory*/
                {
                    Ddesc->tiles_table[res].tile = dague_allocate_matrix( Ddesc->super.bsiz * Ddesc->super.mtype);
                    if (Ddesc->tiles_table[res].tile == NULL)
                        {
                            perror("matrix memory allocation failed\n");
                            exit(-1);
                        }
                    
                }
            else
                Ddesc->tiles_table[res].tile = NULL;
        }

    /*

    Ddesc->mat = dague_allocate_matrix(Ddesc->nb_elem_r * Ddesc->nb_elem_c * Ddesc->super.bsiz * Ddesc->super.mtype);
    if (Ddesc->mat == NULL)
        {
            perror("matrix memory allocation failed\n");
            exit(-1);
        }

    */
    Ddesc->super.super.rank_of =  td_get_rank_for_tile;
    Ddesc->super.super.data_of =  td_get_local_tile;
    
}


