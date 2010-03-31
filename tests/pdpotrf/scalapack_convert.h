/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __SCALAPACK_CONVERT_H__
#define __SCALAPACK_CONVERT_H__


#include "data_management.h"

/* Convert the local view of a matrix from dplasma format to scalapack format.
 * @param Ddesc: dplasma format description
 * @param Sdesc: scalapack format description, should be already allocated with size = 9;
 * @param sca_mat: pointer to the converted matrix location; will be allocated by tiles_to_scalapack
 */
int tiles_to_scalapack(DPLASMA_desc * Ddesc, int * Sdesc, double ** sca_mat);


/* Convert the local view of a matrix from scalapack to dplasma format.
 * @param Ddesc: dplasma format description ; Ddesc->mat will be allocated by scalapack_to_tiles
 * @param Sdesc: scalapack format description
 * @param sca_mat: pointer to the scalapack matrix location
 */
int scalapack_to_tiles(DPLASMA_desc * Ddesc, int * desc, double ** sca_mat);


int scalapack_finalize();
#endif
