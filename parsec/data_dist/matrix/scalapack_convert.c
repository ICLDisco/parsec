/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "scalapack_convert.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/debug.h"

#ifdef PARSEC_HAVE_STRING_H
#include <string.h>
#endif
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif
#include <stdio.h>
#ifdef PARSEC_HAVE_MPI
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
 * PaRSEC  allocate_scalapack_matrix will set desc (from parsec_tiled_matrix_dc_t) to:
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


void * allocate_scalapack_matrix(parsec_tiled_matrix_dc_t * dc, int * sca_desc,  int process_grid_rows)
{
    int pgr, pgc, rr, cr, nb_elem_r, nb_elem_c, rlength, clength;
    void * smat;

    if((dc->super.nodes % process_grid_rows) != 0 )
        {
            printf("bad numbers for process grid for scalapack conversion\n");
            return NULL;
        }
    pgr = process_grid_rows;
    pgc = dc->super.nodes / pgr;

    if( (pgr > (dc->mt)) || (pgc > (dc->nt)) )
        {
            printf("process grid incompatible with matrix size\n");
            return NULL;
        }

    rr = dc->super.myrank / pgc;
    cr = dc->super.myrank % pgc;

    /* find the number of blocks this process will handle in 2D block cyclic */
    nb_elem_r = (dc->mt) / pgr;
    if ( rr < ((dc->mt) % pgr))
        nb_elem_r++;

    nb_elem_c = (dc->nt) / pgc;
    if ( cr < ((dc->nt) % pgc))
        nb_elem_c++;
    
    /* find actual length in number of elements in both dimensions */
    rlength = nb_elem_r * dc->mb;
    clength = nb_elem_c * dc->nb;
    
    /* now remove possible padding added from tiling */
    if ( ((dc->mt - 1) % pgr) == rr ) /* this process hold blocks of the last row of blocks */
        {
            rlength = rlength - ((dc->mt * dc->mb) - dc->m);
        }

    if ( ((dc->nt - 1) % pgc) == cr ) /* this process hold blocks of the last column of blocks */
        {
            clength = clength - ((dc->nt * dc->nb) - dc->n);
        }

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "allocate scalapack matrix: process %u(%d,%d) handles %d x %d blocks, for a total of %d x %d elements (matrix size is %d by %d)",
           dc->super.myrank, rr, cr, nb_elem_r, nb_elem_c, rlength, clength, dc->m, dc->n);
    
    smat =  parsec_data_allocate(rlength * clength * parsec_datadist_getsizeoftype(dc->mtype));

    sca_desc[0] = 1;
    sca_desc[1] = 0;
    sca_desc[2] = dc->m;
    sca_desc[3] = dc->n;
    sca_desc[4] = dc->mb;
    sca_desc[5] = dc->nb;
    sca_desc[6] = 0;
    sca_desc[7] = 0;
    sca_desc[8] = rlength;

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "allocate scalapack matrix: scalapack descriptor: [(dense == 1) %d, (ICTX) %d, (M) %d, (N) %d, (MB) %d, (NB) %d,(IRSRC) %d, (ICSRC) %d, (LLD) %d ]\n ",
           sca_desc[0], sca_desc[1], sca_desc[2], sca_desc[3], sca_desc[4], sca_desc[5], sca_desc[6], sca_desc[7], sca_desc[8]);

    memset(smat, 0 , rlength * clength * parsec_datadist_getsizeoftype(dc->mtype));
    return smat;    
}

int tiles_to_scalapack_info_init(scalapack_info_t * info, parsec_tiled_matrix_dc_t * dc, int * sca_desc, void * sca_mat, int process_grid_rows)
{
#ifdef PARSEC_HAVE_MPI
    int length, size;
#endif

    info->dc = dc;
    info->sca_desc = sca_desc;
    info->sca_mat = sca_mat;
    info->process_grid_rows = process_grid_rows;
    
#ifdef PARSEC_HAVE_MPI
    /* mpi type creation*/
    /* type for full blocks */
    MPI_Type_vector(info->dc->nb, info->dc->mb, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_full_block));
    MPI_Type_commit (&(info->MPI_Sca_full_block));

    /* type for last row of tiles */
    length = info->dc->mt*info->dc->mb != info->dc->m ? info->dc->m - ((info->dc->mt - 1)*info->dc->mb ) : info->dc->mb;
    MPI_Type_vector(info->dc->nb, length, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_last_row));
    MPI_Type_commit (&(info->MPI_Sca_last_row));


    /* type for last column of tiles */
    length = info->dc->nt*info->dc->nb != info->dc->n ? info->dc->n - ((info->dc->nt - 1)*info->dc->nb) : info->dc->nb;
    MPI_Type_vector(length, info->dc->mb, sca_desc[8],
                    MPI_DOUBLE, &(info->MPI_Sca_last_col));
    MPI_Type_commit (&(info->MPI_Sca_last_col));

    /* type for last tile */
    length = info->dc->mt*info->dc->mb != info->dc->m ? info->dc->m - ((info->dc->mt - 1)*info->dc->mb ) : info->dc->mb;
    size = info->dc->nt*info->dc->nb != info->dc->n ? info->dc->n - ((info->dc->nt - 1)*info->dc->nb) : info->dc->nb;
    MPI_Type_vector(size, length, sca_desc[8], MPI_DOUBLE, &(info->MPI_Sca_last_block));
    MPI_Type_commit (&(info->MPI_Sca_last_block));


    /* type for full tiles */
    MPI_Type_contiguous(info->dc->bsiz, MPI_DOUBLE, &(info->MPI_PaRSEC_full_block));
    MPI_Type_commit (&(info->MPI_PaRSEC_full_block));

    /* type for last row of tiles */
    length = info->dc->mt*info->dc->mb != info->dc->m ? info->dc->m - ((info->dc->mt - 1)*info->dc->mb ) : info->dc->mb;

    MPI_Type_vector(info->dc->nb, length, info->dc->mb,
                    MPI_DOUBLE, &(info->MPI_PaRSEC_last_row));
    MPI_Type_commit (&(info->MPI_PaRSEC_last_row));


    /* type for last column of tiles */
    length = info->dc->nt*info->dc->nb != info->dc->n ? info->dc->n - ((info->dc->nt - 1)*info->dc->nb) : info->dc->nb;
    MPI_Type_contiguous(length * info->dc->mb, MPI_DOUBLE, &(info->MPI_PaRSEC_last_col));
    MPI_Type_commit (&(info->MPI_PaRSEC_last_col));

    /* type for last tile */
    length = info->dc->mt*info->dc->mb != info->dc->m ? info->dc->m - ((info->dc->mt - 1)*info->dc->mb ) : info->dc->mb;
    size = info->dc->nt*info->dc->nb != info->dc->n ? info->dc->n - ((info->dc->nt - 1)*info->dc->nb) : info->dc->nb;
    MPI_Type_vector(size, length, info->dc->mb, MPI_DOUBLE, &(info->MPI_PaRSEC_last_block));
    MPI_Type_commit (&(info->MPI_PaRSEC_last_block));

    /* MPI_Type_vector(count, blocklength, stride, MPI_DOUBLE, &(info->MPI_Sca_last_block)); */
#endif /* PARSEC_HAVE_MPI */
    return 0;
}

void tiles_to_scalapack_info_destroy(scalapack_info_t * info)
{
#ifdef PARSEC_HAVE_MPI
    MPI_Type_free(&(info->MPI_Sca_full_block));
    MPI_Type_free(&(info->MPI_Sca_last_row));
    MPI_Type_free(&(info->MPI_Sca_last_col));
    MPI_Type_free(&(info->MPI_Sca_last_block));
    MPI_Type_free(&(info->MPI_PaRSEC_full_block));
    MPI_Type_free(&(info->MPI_PaRSEC_last_row));
    MPI_Type_free(&(info->MPI_PaRSEC_last_col));
    MPI_Type_free(&(info->MPI_PaRSEC_last_block));
#else
    (void)info;
#endif /* PARSEC_HAVE_MPI */
    return;
}


#ifdef PARSEC_HAVE_MPI
/* to compute which process will get this tile as a scalapack block */
static int twoDBC_get_rank(parsec_tiled_matrix_dc_t * dc, int process_grid_rows, int row, int col)
{
    int cr, rr, res, GRIDcols, GRIDrows;

    GRIDrows = process_grid_rows;
    GRIDcols = dc->super.nodes / GRIDrows;

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
         
    src = info->dc->super.rank_of((parsec_data_collection_t *)(info->dc), row, col);
    dest = twoDBC_get_rank( info->dc, info->process_grid_rows, row, col);
    dec = -1;

    if (INT_MAX == src)
        return;
   
    if(src == dest) {  /* local operation */
        if(src == info->dc->super.myrank) {
            GRIDrows = info->process_grid_rows;
            GRIDcols = info->dc->super.nodes / GRIDrows;
            //rr = info->dc->super.myrank / GRIDcols;
            //cr = info->dc->super.myrank % GRIDcols;
                    
            max_mb = ((info->dc->mb * (row + 1)) <=  info->dc->m) ? info->dc->mb :  (info->dc->m - ((info->dc->mb * row)));
            max_nb = ((info->dc->nb * (col + 1)) <=  info->dc->n) ? info->dc->nb :  (info->dc->n - ((info->dc->nb * col)));
                    
            il = row / GRIDrows;
            jl = col / GRIDcols;
            dec = (info->dc->nb * (int)info->sca_desc[8] * jl) + (info->dc->mb * il);
                    
            bdl = (double *)info->dc->super.data_of((parsec_data_collection_t *)info->dc, row, col);
            lapack = (double*) &(((double*)(info->sca_mat))[ dec ]);
                    
            for (y = 0; y < max_nb; y++)
                for (x = 0; x < max_mb ; x++)
                    lapack[info->sca_desc[8] * y + x] = bdl[(info->dc->mb)*y + x];
        }
    }
    else if ( src == info->dc->super.myrank ) {  /* process have the tile to send */
        printf("weird\n");
        bdl = (double *)info->dc->super.data_of((parsec_data_collection_t *)info->dc, row, col);
        if (row + 1 == info->dc->mt) {
            if( col + 1 == info->dc->nt) {
                MPI_Send(bdl, 1, info->MPI_PaRSEC_last_block, dest, 0, MPI_COMM_WORLD );
            } else {
                MPI_Send(bdl, 1, info->MPI_PaRSEC_last_row, dest, 0, MPI_COMM_WORLD );
            }
        } else if (col + 1 == info->dc->nt) {
            MPI_Send(bdl, 1, info->MPI_PaRSEC_last_col, dest, 0, MPI_COMM_WORLD );
        } else {
            MPI_Send(bdl, 1, info->MPI_PaRSEC_full_block, dest, 0, MPI_COMM_WORLD );
        }
    } else if (dest == info->dc->super.myrank) {  /* process have to receive the block */
        GRIDrows = info->process_grid_rows;
        GRIDcols = info->dc->super.nodes / GRIDrows;
        il = row / GRIDrows;
        jl = col / GRIDcols;
        dec = (info->dc->nb * (int)info->sca_desc[8] * jl) + (info->dc->mb * il);
        lapack = (double*) &(((double*)(info->sca_mat))[ dec ]);
        if (row + 1 == info->dc->mt) {
            if( col + 1 == info->dc->nt) {
                MPI_Recv(lapack, 1, info->MPI_Sca_last_block, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(lapack, 1, info->MPI_Sca_last_row, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
        } else if (col + 1 == info->dc->nt) {
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
    for(i = 0 ; i < info->dc->mt ; i++)
        for(j = 0 ; j < info->dc->nt ; j++)
            tile_to_block_double(info, i, j);    
    return 0;
}

#else /* ! PARSEC_HAVE_MPI */

void tile_to_block_double(scalapack_info_t * info, int row, int col)
{
    int x, y, dec;
    double *bdl, *lapack;

    bdl = (double *)info->dc->super.data_of((parsec_data_collection_t *)info->dc, row, col);
    dec = ((info->dc->nb)*(info->dc->m)*col) + ((info->dc->mb)*row);
    lapack = (double*)&(((double*)(info->sca_mat))[ dec ]);
    
    for (y = 0; y < (info->dc->nb); y++)
        for (x = 0; x < (info->dc->mb); x++)
            lapack[(info->dc->m)*y+x] = bdl[(info->dc->mb)*y + x];

    return;
}

//TODO : multi-threading ?
int tiles_to_scalapack(scalapack_info_t * info)
{
    int i,j;
    for(i = 0 ; i < info->dc->mt ; i++)
        for(j = 0 ; j < info->dc->nt ; j++)
            tile_to_block_double(info, i, j);    
    return 0;
}

/*
int tiles_to_scalapack(parsec_tiled_matrix_dc_t * dc, int * desc, void * sca_mat, int process_grid_rows)
{
    int i, j, il, jl, x, y;
    double *bdl, *lapack;
    int64_t dec;

    // check which tiles to generate 
    for ( j = 0 ; j < dc->super.lnt ; j++)
        for ( i = 0 ; i < dc->super.lmt ; i++)
        {
	    if( dc->super.super.myrank ==
		dc->super.super.rank_of((parsec_data_collection_t *)dc, i, j ) )
                {
                    il = i / ( dc->grid.strows * dc->grid.rows ) +  (i % ( dc->grid.strows * dc->grid.rows )) - ( dc->grid.strows * dc->grid.rrank );
                    jl = j / ( dc->grid.stcols * dc->grid.cols ) +  (j % ( dc->grid.stcols * dc->grid.cols )) - ( dc->grid.stcols * dc->grid.crank );
                    dec = ((int64_t)(dc->super.nb)*(int64_t)(dc->lm)*(int64_t)(jl)) + (int64_t)((dc->super.mb)*(il));
                    bdl = dc->super.super.data_of((parsec_data_collection_t *)dc, i, j );
                    lapack = &sca_mat[ dec ];
                    
                    for (y = 0; y < (dc->super.nb); y++)
                        for (x = 0; x < (dc->super.mb); x++)
                            lapack[(dc->lm)*y+x] = bdl[(dc->super.nb)*y + x];
                }
	}
    return 0;
}
*/
/*
int scalapack_to_tiles(DPLASMA_desc * dc, int * desc, double ** sca_mat)
{
    return 1;
}

*/
#endif
