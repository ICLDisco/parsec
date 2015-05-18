/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "scalapack_convert.h"
#include "dague/data_distribution.h"
#include "matrix.h"
#include "dague/debug.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#include <stdio.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

/**
 *(as of http://www.netlib.org/scalapack/slug/node77.html#secdesc1)
 *
 * scalapack desc : input integer vector of length 9 which is to contain the array descriptor information.
 *
 * desc(0) = 1     : (global) This is the descriptor type. In this case,  1 means "dense matrix".
 * desc(1) = ICTXT : (global) An input integer which indicates the BLACS context handle.
 * desc(2) = M     : (global) An input integer which indicates the row size of the global array which is being described.
 * desc(3) = N     : (global) An input integer which indicates the column size of the global array which is being described.
 * desc(4) = MB    : (global) An input integer which indicates the blocking factor used to distribute the rows of the matrix being described.
 * desc(5) = NB    : (global) An input integer which indicates the blocking factor used to distribute the columns of the matrix being described.
 * desc(6) = IRSRC : (global) An input integer which indicates the processor grid row over which the first row of the array being described is distributed.
 * desc(7) = ICSRC : (global) An input integer which indicates the processor grid column over which the first column of the array being described is distributed.
 * desc(8) = LLD   : (local)  An input integer indicating the leading dimension of the local array which is to be used for storing the local blocks of the array being described.
 *
 * DAGuE  allocate_scalapack_matrix will set desc (from tiled_matrix_desc_t) to:
 * desc(0) = 1     : 1
 * desc(1) = ICTXT : 0  as BLACS is not initialized yet. // side note: blacs context handles the process grid info
 * desc(2) = M     : m
 * desc(3) = N     : n
 * desc(4) = MB    : mb
 * desc(5) = NB    : nb
 * desc(6) = IRSRC : 0
 * desc(7) = ICSRC : 0
 * desc(8) = LLD   : leading dimension of the local array (rare thing to actually compute)
 */


void * allocate_scalapack_matrix(tiled_matrix_desc_t * Ddesc, int * sca_desc,  int process_grid_rows)
{
    int pgr, pgc, rr, cr, nb_elem_r, nb_elem_c, rlength, clength;
    void * smat;

    if((Ddesc->super.nodes % process_grid_rows) != 0 )
        {
            printf("bad numbers for process grid for scalapack conversion\n");
            return NULL;
        }
    pgr = process_grid_rows;
    pgc = Ddesc->super.nodes / pgr;

    if( (pgr > (Ddesc->mt)) || (pgc > (Ddesc->nt)) )
        {
            printf("process grid incompatible with matrix size\n");
            return NULL;
        }

    rr = Ddesc->super.myrank / pgc;
    cr = Ddesc->super.myrank % pgc;

    /* find the number of blocks this process will handle in 2D block cyclic */
    nb_elem_r = (Ddesc->mt) / pgr;
    if ( rr < ((Ddesc->mt) % pgr))
        nb_elem_r++;

    nb_elem_c = (Ddesc->nt) / pgc;
    if ( cr < ((Ddesc->nt) % pgc))
        nb_elem_c++;
    
    /* find actual length in number of elements in both dimensions */
    rlength = nb_elem_r * Ddesc->mb;
    clength = nb_elem_c * Ddesc->nb;
    
    /* now remove possible padding added from tiling */
    if ( ((Ddesc->mt - 1) % pgr) == rr ) /* this process hold blocks of the last row of blocks */
        {
            rlength = rlength - ((Ddesc->mt * Ddesc->mb) - Ddesc->m);
        }

    if ( ((Ddesc->nt - 1) % pgc) == cr ) /* this process hold blocks of the last column of blocks */
        {
            clength = clength - ((Ddesc->nt * Ddesc->nb) - Ddesc->n);
        }

    DEBUG3(("allocate scalapack matrix: process %u(%d,%d) handles %d x %d blocks, for a total of %d x %d elements (matrix size is %d by %d)\n",
           Ddesc->super.myrank, rr, cr, nb_elem_r, nb_elem_c, rlength, clength, Ddesc->m, Ddesc->n));
    
    smat =  dague_data_allocate(rlength * clength * dague_datadist_getsizeoftype(Ddesc->mtype));

    sca_desc[0] = 1;
    sca_desc[1] = 0;
    sca_desc[2] = Ddesc->m;
    sca_desc[3] = Ddesc->n;
    sca_desc[4] = Ddesc->mb;
    sca_desc[5] = Ddesc->nb;
    sca_desc[6] = 0;
    sca_desc[7] = 0;
    sca_desc[8] = rlength;

    DEBUG3(("allocate scalapack matrix: scalapack descriptor: [(dense == 1) %d, (ICTX) %d, (M) %d, (N) %d, (MB) %d, (NB) %d,(IRSRC) %d, (ICSRC) %d, (LLD) %d ]\n ",
           sca_desc[0], sca_desc[1], sca_desc[2], sca_desc[3], sca_desc[4], sca_desc[5], sca_desc[6], sca_desc[7], sca_desc[8]));

    memset(smat, 0 , rlength * clength * dague_datadist_getsizeoftype(Ddesc->mtype));
    return smat;    
}

int tiles_to_scalapack_info_init(scalapack_info_t * info, tiled_matrix_desc_t * Ddesc, int * sca_desc, void * sca_mat, int process_grid_rows)
{
#ifdef HAVE_MPI
    int length, size;
#endif

    info->Ddesc = Ddesc;
    info->sca_desc = sca_desc;
    info->sca_mat = sca_mat;
    info->process_grid_rows = process_grid_rows;
    
#ifdef HAVE_MPI
    /* mpi type creation*/
    /* type for full blocks */
    MPI_Type_vector(info->Ddesc->nb, info->Ddesc->mb, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_full_block));
    MPI_Type_commit (&(info->MPI_Sca_full_block));

    /* type for last row of tiles */
    length = info->Ddesc->mt*info->Ddesc->mb != info->Ddesc->m ? info->Ddesc->m - ((info->Ddesc->mt - 1)*info->Ddesc->mb ) : info->Ddesc->mb;
    MPI_Type_vector(info->Ddesc->nb, length, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_last_row));
    MPI_Type_commit (&(info->MPI_Sca_last_row));


    /* type for last column of tiles */
    length = info->Ddesc->nt*info->Ddesc->nb != info->Ddesc->n ? info->Ddesc->n - ((info->Ddesc->nt - 1)*info->Ddesc->nb) : info->Ddesc->nb;
    MPI_Type_vector(length, info->Ddesc->mb, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_last_col));
    MPI_Type_commit (&(info->MPI_Sca_last_col));

    /* type for last tile */
    length = info->Ddesc->mt*info->Ddesc->mb != info->Ddesc->m ? info->Ddesc->m - ((info->Ddesc->mt - 1)*info->Ddesc->mb ) : info->Ddesc->mb;
    size = info->Ddesc->nt*info->Ddesc->nb != info->Ddesc->n ? info->Ddesc->n - ((info->Ddesc->nt - 1)*info->Ddesc->nb) : info->Ddesc->nb;
    MPI_Type_vector(size, length, sca_desc[8], MPI_DOUBLE, &(info->MPI_Sca_last_block));
    MPI_Type_commit (&(info->MPI_Sca_last_block));


    /* type for full tiles */
    MPI_Type_contiguous(info->Ddesc->bsiz, MPI_DOUBLE, &(info->MPI_Dague_full_block));
    MPI_Type_commit (&(info->MPI_Dague_full_block));

    /* type for last row of tiles */
    length = info->Ddesc->mt*info->Ddesc->mb != info->Ddesc->m ? info->Ddesc->m - ((info->Ddesc->mt - 1)*info->Ddesc->mb ) : info->Ddesc->mb;

    MPI_Type_vector(info->Ddesc->nb, length, info->Ddesc->mb,
                    MPI_DOUBLE, &(info->MPI_Dague_last_row));
    MPI_Type_commit (&(info->MPI_Dague_last_row));


    /* type for last column of tiles */
    length = info->Ddesc->nt*info->Ddesc->nb != info->Ddesc->n ? info->Ddesc->n - ((info->Ddesc->nt - 1)*info->Ddesc->nb) : info->Ddesc->nb;
    MPI_Type_contiguous(length * info->Ddesc->mb, MPI_DOUBLE, &(info->MPI_Dague_last_col));
    MPI_Type_commit (&(info->MPI_Dague_last_col));

    /* type for last tile */
    length = info->Ddesc->mt*info->Ddesc->mb != info->Ddesc->m ? info->Ddesc->m - ((info->Ddesc->mt - 1)*info->Ddesc->mb ) : info->Ddesc->mb;
    size = info->Ddesc->nt*info->Ddesc->nb != info->Ddesc->n ? info->Ddesc->n - ((info->Ddesc->nt - 1)*info->Ddesc->nb) : info->Ddesc->nb;
    MPI_Type_vector(size, length, info->Ddesc->mb, MPI_DOUBLE, &(info->MPI_Dague_last_block));
    MPI_Type_commit (&(info->MPI_Dague_last_block));

    /* MPI_Type_vector(count, blocklength, stride, MPI_DOUBLE, &(info->MPI_Sca_last_block)); */
#endif /* HAVE_MPI */
    return 0;
}

void tiles_to_scalapack_info_destroy(scalapack_info_t * info)
{
#ifdef HAVE_MPI
    MPI_Type_free(&(info->MPI_Sca_full_block));
    MPI_Type_free(&(info->MPI_Sca_last_row));
    MPI_Type_free(&(info->MPI_Sca_last_col));
    MPI_Type_free(&(info->MPI_Sca_last_block));
    MPI_Type_free(&(info->MPI_Dague_full_block));
    MPI_Type_free(&(info->MPI_Dague_last_row));
    MPI_Type_free(&(info->MPI_Dague_last_col));
    MPI_Type_free(&(info->MPI_Dague_last_block));
#else
    (void)info;
#endif /* HAVE_MPI */
    return;
}


#ifdef HAVE_MPI
/* to compute which process will get this tile as a scalapack block */
static int twoDBC_get_rank(tiled_matrix_desc_t * Ddesc, int process_grid_rows, int row, int col)
{
    int cr, rr, res, GRIDcols, GRIDrows;

    GRIDrows = process_grid_rows;
    GRIDcols = Ddesc->super.nodes / GRIDrows;

    rr = row % GRIDrows;
    cr = col % GRIDcols;

    res = rr * GRIDcols + cr;
    /* printf("tile (%d, %d) belongs to process %d [%d,%d] in a grid of %dx%d\n", */
    /*            m, n, res, rr, cr, GRIDrows, GRIDcols); */
    return res;
}

void tile_to_block_double(scalapack_info_t * info, int row, int col)
{
    int x, y, dec, GRIDcols, GRIDrows;
    uint32_t src, dest;
    double *bdl, *lapack;
    int il, jl, max_mb, max_nb;
    MPI_Status status;
         
    src = info->Ddesc->super.rank_of((dague_ddesc_t *)(info->Ddesc), row, col);
    dest = twoDBC_get_rank( info->Ddesc, info->process_grid_rows, row, col);
    dec = -1;

    if (INT_MAX == src)
        return;
   
    if(src == dest) {  /* local operation */
        if(src == info->Ddesc->super.myrank) {
            GRIDrows = info->process_grid_rows;
            GRIDcols = info->Ddesc->super.nodes / GRIDrows;
            //rr = info->Ddesc->super.myrank / GRIDcols;
            //cr = info->Ddesc->super.myrank % GRIDcols;
                    
            max_mb = ((info->Ddesc->mb * (row + 1)) <=  info->Ddesc->m) ? info->Ddesc->mb :  (info->Ddesc->m - ((info->Ddesc->mb * row)));
            max_nb = ((info->Ddesc->nb * (col + 1)) <=  info->Ddesc->n) ? info->Ddesc->nb :  (info->Ddesc->n - ((info->Ddesc->nb * col)));
                    
            il = row / GRIDrows;
            jl = col / GRIDcols;
            dec = (info->Ddesc->nb * (int)info->sca_desc[8] * jl) + (info->Ddesc->mb * il);
                    
            bdl = (double *)info->Ddesc->super.data_of((dague_ddesc_t *)info->Ddesc, row, col);
            lapack = (double*) &(((double*)(info->sca_mat))[ dec ]);
                    
            for (y = 0; y < max_nb; y++)
                for (x = 0; x < max_mb ; x++)
                    lapack[info->sca_desc[8] * y + x] = bdl[(info->Ddesc->mb)*y + x];
        }
    }
    else if ( src == info->Ddesc->super.myrank ) {  /* process have the tile to send */
        printf("weird\n");
        bdl = (double *)info->Ddesc->super.data_of((dague_ddesc_t *)info->Ddesc, row, col);
        if (row + 1 == info->Ddesc->mt) {
            if( col + 1 == info->Ddesc->nt) {
                MPI_Send(bdl, 1, info->MPI_Dague_last_block, dest, 0, MPI_COMM_WORLD );
            } else {
                MPI_Send(bdl, 1, info->MPI_Dague_last_row, dest, 0, MPI_COMM_WORLD );
            }
        } else if (col + 1 == info->Ddesc->nt) {
            MPI_Send(bdl, 1, info->MPI_Dague_last_col, dest, 0, MPI_COMM_WORLD );
        } else {
            MPI_Send(bdl, 1, info->MPI_Dague_full_block, dest, 0, MPI_COMM_WORLD );
        }
    } else if (dest == info->Ddesc->super.myrank) {  /* process have to receive the block */
        GRIDrows = info->process_grid_rows;
        GRIDcols = info->Ddesc->super.nodes / GRIDrows;
        il = row / GRIDrows;
        jl = col / GRIDcols;
        dec = (info->Ddesc->nb * (int)info->sca_desc[8] * jl) + (info->Ddesc->mb * il);
        lapack = (double*) &(((double*)(info->sca_mat))[ dec ]);
        if (row + 1 == info->Ddesc->mt) {
            if( col + 1 == info->Ddesc->nt) {
                MPI_Recv(lapack, 1, info->MPI_Sca_last_block, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(lapack, 1, info->MPI_Sca_last_row, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
        } else if (col + 1 == info->Ddesc->nt) {
            MPI_Recv(lapack, 1, info->MPI_Sca_last_row, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(lapack, 1, info->MPI_Sca_full_block, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }
    return;
}

int tiles_to_scalapack(scalapack_info_t * info)
{
    int i,j;
    for(i = 0 ; i < info->Ddesc->mt ; i++)
        for(j = 0 ; j < info->Ddesc->nt ; j++)
            tile_to_block_double(info, i, j);    
    return 0;
}

#else /* ! HAVE_MPI */

void tile_to_block_double(scalapack_info_t * info, int row, int col)
{
    int x, y, dec;
    double *bdl, *lapack;

    bdl = (double *)info->Ddesc->super.data_of((dague_ddesc_t *)info->Ddesc, row, col);
    dec = ((info->Ddesc->nb)*(info->Ddesc->m)*col) + ((info->Ddesc->mb)*row);
    lapack = (double*)&(((double*)(info->sca_mat))[ dec ]);
    
    for (y = 0; y < (info->Ddesc->nb); y++)
        for (x = 0; x < (info->Ddesc->mb); x++)
            lapack[(info->Ddesc->m)*y+x] = bdl[(info->Ddesc->mb)*y + x];

    return;
}

//TODO : multi-threading ?
int tiles_to_scalapack(scalapack_info_t * info)
{
    int i,j;
    for(i = 0 ; i < info->Ddesc->mt ; i++)
        for(j = 0 ; j < info->Ddesc->nt ; j++)
            tile_to_block_double(info, i, j);    
    return 0;
}

/*
int tiles_to_scalapack(tiled_matrix_desc_t * Ddesc, int * desc, void * sca_mat, int process_grid_rows)
{
    int i, j, il, jl, x, y;
    double *bdl, *lapack;
    int64_t dec;

    // check which tiles to generate 
    for ( j = 0 ; j < Ddesc->super.lnt ; j++)
        for ( i = 0 ; i < Ddesc->super.lmt ; i++)
        {
	    if( Ddesc->super.super.myrank ==
		Ddesc->super.super.rank_of((dague_ddesc_t *)Ddesc, i, j ) )
                {
                    il = i / ( Ddesc->grid.strows * Ddesc->grid.rows ) +  (i % ( Ddesc->grid.strows * Ddesc->grid.rows )) - ( Ddesc->grid.strows * Ddesc->grid.rrank );
                    jl = j / ( Ddesc->grid.stcols * Ddesc->grid.cols ) +  (j % ( Ddesc->grid.stcols * Ddesc->grid.cols )) - ( Ddesc->grid.stcols * Ddesc->grid.crank );
                    dec = ((int64_t)(Ddesc->super.nb)*(int64_t)(Ddesc->lm)*(int64_t)(jl)) + (int64_t)((Ddesc->super.mb)*(il));
                    bdl = Ddesc->super.super.data_of((dague_ddesc_t *)Ddesc, i, j );
                    lapack = &sca_mat[ dec ];
                    
                    for (y = 0; y < (Ddesc->super.nb); y++)
                        for (x = 0; x < (Ddesc->super.mb); x++)
                            lapack[(Ddesc->lm)*y+x] = bdl[(Ddesc->super.nb)*y + x];
                }
	}
    return 0;
}
*/
/*
int scalapack_to_tiles(DPLASMA_desc * Ddesc, int * desc, double ** sca_mat)
{
    return 1;
}

*/
#endif
