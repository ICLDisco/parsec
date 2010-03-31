/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "scalapack_convert.h"
#include "data_management.h"

extern int Cblacs_pinfo( int *, int *);
extern int Cblacs_get( int , int, int * );
extern int Cblacs_gridinit( int *, char* , int, int ); 
extern int Cblacs_gridinfo( int, int *, int *, int *, int * );
extern int descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);

/* scalapack/tools/descinit.f */

/* !!! TODO: only consider matrix size as a multiple of tile size (i.e. no padding inside any tile) !!! */
static int blacs_ctxt = -1;
int tiles_to_scalapack(DPLASMA_desc * Ddesc, int * desc, double ** sca_mat)
{

    int nprow, npcol, myrow, mycol, ldd, i, j, k, cc, cr;
    int info,itemp;
    int ZERO=0,ONE=1; /* stupid thing required for fortran call */
    double ** column_tiles;
    double * current_position;
    /************************* scalapack matrix description initialization ************************************/
    
    Cblacs_pinfo( &(Ddesc->mpi_rank), &(Ddesc->nodes));/* inform blacs about nb procs and local rank */
    Cblacs_get( -1, 0, &blacs_ctxt ); /* retrieve blacs context handle */
    printf("value of blacs_ctxt in tile_to_scalapack: %d\n", blacs_ctxt);
    Cblacs_gridinit( &blacs_ctxt, "Row", Ddesc->GRIDrows, Ddesc->GRIDcols ); /* should produce the same process grid as dplasma one */
    Cblacs_gridinfo( blacs_ctxt, &nprow, &npcol, &myrow, &mycol ); /* retrieve the process grid mapped */
    if ( (nprow != Ddesc->GRIDrows) || (npcol != Ddesc->GRIDcols) || (myrow != Ddesc->rowRANK) || (mycol != Ddesc->colRANK))
        {
            printf("process grid mismatch !!! GRID: %d x %d | %d x %d coordinate: %d x %d | %d x %d\n",
                   nprow, npcol, Ddesc->GRIDrows, Ddesc->GRIDcols, myrow, mycol, Ddesc->rowRANK, Ddesc->colRANK);
            return 0;
        }

    cr = Ddesc->nrst * Ddesc->mb;
    cc = Ddesc->ncst * Ddesc->nb;
    ldd = Ddesc->nb_elem_r * Ddesc->mb;
    printf("for proc %d, ldd= %d (%d x %d)\n",Ddesc->mpi_rank, ldd , Ddesc->nb_elem_r ,  Ddesc->mb);

    /*
      DESCINIT(DESCA, M_A, N_A, MB_A, NB_A, RSRC_A, CSRC_A,CONTEXT, LLD_A, IERR)

      * DESCA = the filled-in descriptor vector returned by the routine
      * M_A = number of rows in the global array A
      * N_A = number of columns in the global array A
      * MB_A = number of rows in a block of A
      * NB_A = number of columns in a block of A
      * RSRC_A = processor grid row that has the first block of A (usually 0)
      * CSRC_A = processor grid column that has first block of A (usually 0)
      * CONTEXT = BLACS context
      * LLD_A = number of rows of the local array that stores the blocks of A (local leading dimension).This element is processor-dependent
      * IERR = status value returned to indicate if the routine worked correctly;
      ierr = 0, routine successful.
      ierr = -i, the $i^{th}$ argument had an illegal value

     */
    descinit_(desc, &(Ddesc->lm), &(Ddesc->ln), &cr, &cc, &ZERO, &ZERO, &blacs_ctxt, &ldd, &info);
    if (info !=0)
        {
            printf("proc %d had illegal value given at descinit: %d\n", Ddesc->mpi_rank, info);
            return -1;
        }
    
    /* matrix description completed, now convert matrix */
    
    
    /* allocate matrix */
    *sca_mat = (double *)malloc(sizeof(double) * Ddesc->nb_elem_r * Ddesc->nb_elem_c* Ddesc->bsiz );
    if (*sca_mat == NULL)
        {
            perror("tiles to scalapack memory allocation\n");
            return -1;
        }

    /* copy values from Ddesc->mat to *sca_mat */
    column_tiles = (double**)malloc(sizeof(double*) *Ddesc->nb_elem_r ); /* prepare a pointer to each
                                                                            local tile of the local view of a column */
    current_position = (*sca_mat);

    cr = Ddesc->rowRANK * Ddesc->nrst;
    cc = Ddesc->colRANK * Ddesc->ncst;

    for (i = 0 ; i < Ddesc->nb_elem_c ; i++) /* for each local tile column */
        {
            /* assign pointer to each tile in that column */
            k = 0;
            for ( j = cr; j < Ddesc->lmt ; j++ )
                if (Ddesc->mpi_rank == dplasma_get_rank_for_tile(Ddesc, j, cc))
                    {
                        column_tiles[k]= dplasma_get_local_tile_s(Ddesc, j, cc);
                        k++;
                    }
            
            /* copy values of the tiles column by column */
            for (j = 0 ; j < Ddesc->nb ; j++)
                for ( k = 0 ; k < Ddesc-> nb_elem_r ; k++)
                    {
                        memcpy((void*)current_position , (void*) &((column_tiles[k])[j*Ddesc->mb]),
                               (sizeof(double)*Ddesc->mb) ); /* copy all the column */

                        current_position += Ddesc->mb;
                    }
            /* prepare for next tile column */
            for ( j = (cc + 1) ; j < Ddesc->lnt ; j++)
                if(Ddesc->mpi_rank == dplasma_get_rank_for_tile(Ddesc, cr, j))
                    {
                        cc = j;
                        break;
                    }
        }
    return 0;
}


int scalapack_finalize()
{
    /*
    blacs_gridexit_( &blacs_ctxt );
    blacs_exit_( CONTINUE );
    */
    return 0;
}



int scalapack_to_tiles(DPLASMA_desc * Ddesc, int * desc, double ** sca_mat)
{
    return 1;
}
