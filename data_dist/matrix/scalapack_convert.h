/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __SCALAPACK_CONVERT_H__
#define __SCALAPACK_CONVERT_H__

#include "parsec/parsec_config.h"
#include "parsec/data_distribution.h"
#include "matrix.h"

#ifdef PARSEC_HAVE_MPI
#include <mpi.h>
#endif/* PARSEC_HAVE_MPI */

BEGIN_C_DECLS

typedef struct scalapack_info_t {
    parsec_tiled_matrix_dc_t * dc;
    int * sca_desc;
    void * sca_mat;
    int process_grid_rows;
#ifdef PARSEC_HAVE_MPI
    MPI_Datatype MPI_Sca_full_block;
    MPI_Datatype MPI_Sca_last_row;
    MPI_Datatype MPI_Sca_last_col;
    MPI_Datatype MPI_Sca_last_block;

    MPI_Datatype MPI_PaRSEC_full_block;
    MPI_Datatype MPI_PaRSEC_last_row;
    MPI_Datatype MPI_PaRSEC_last_col;
    MPI_Datatype MPI_PaRSEC_last_block;
#endif /* PARSEC_HAVE_MPI */
} scalapack_info_t;

/* allocate buffer size to handle a matrix in scalapack format in 2D block cyclic, given a parsec matrix specification and a process grid
 * @param dc: parsec format description of the matrix to convert (distributed in any fashion)
 * @param process_grid_rows: number of rows in the process grid for 2D block cyclic (number of column computed internally)
 * @return buffer allocated to contain scalapack conversion
 */
void * allocate_scalapack_matrix(parsec_tiled_matrix_dc_t * dc, int * sca_desc, int process_grid_rows);

int tiles_to_scalapack_info_init(scalapack_info_t * info, parsec_tiled_matrix_dc_t * dc, int * sca_desc, void * sca_mat, int process_grid_rows);

void tiles_to_scalapack_info_destroy(scalapack_info_t * info);
    
void tile_to_block_double(scalapack_info_t * info, int row, int col);

/* Convert the local view of a matrix from parsec format to scalapack format.
 * @param dc: parsec format description of the matrix to convert (distributed in any fashion)
 * @param desc:  scalapack format description, should be already allocated with size = 9;
 * @param sca_mat: pointer to the converted matrix location
 */
int tiles_to_scalapack(scalapack_info_t * info);



/* Convert the local view of a matrix from scalapack to dplasma format.
 * @param dc: parsec format description 
 * @param Sdesc: scalapack format description
 * @param sca_mat: pointer to the scalapack matrix location
 */
//int scalapack_to_tiles(scalapack_info_t * info);

END_C_DECLS

#endif /* __SCALAPACK_CONVERT_H__ */
