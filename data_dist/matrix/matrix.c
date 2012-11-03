/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 *distributed matrix generation
 ************************************************************/
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#include "dague_config.h"
#include "data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "matrix.h"
#include "data.h"

/***************************************************************************//**
 *  Internal static descriptor initializer (PLASMA code)
 **/
void tiled_matrix_desc_init( tiled_matrix_desc_t *tdesc,
                             enum matrix_type    mtyp,
                             enum matrix_storage storage,
                             int dtype, int nodes, int cores, int myrank,
                             int mb, int nb,
                             int lm, int ln,
                             int i,  int j,
                             int m,  int n)
{
    /* Matrix address */
    /* tdesc->mat = NULL;*/
    /* tdesc->A21 = (lm - lm%mb)*(ln - ln%nb); */
    /* tdesc->A12 = (     lm%mb)*(ln - ln%nb) + tdesc->A21; */
    /* tdesc->A22 = (lm - lm%mb)*(     ln%nb) + tdesc->A12; */

    /* Super setup */
    tdesc->super.nodes = nodes;    
    tdesc->super.cores = cores;
    tdesc->super.myrank = myrank;

    /* Matrix properties */
    tdesc->data_map = NULL;
    tdesc->mtype    = mtyp;
    tdesc->storage  = storage;
    tdesc->dtype    = tiled_matrix_desc_type | dtype;
    tdesc->tileld   = (storage == matrix_Tile) ? mb : lm;
    tdesc->mb       = mb;
    tdesc->nb       = nb;
    tdesc->bsiz     = mb * nb;

    /* Large matrix parameters */
    tdesc->lm = lm;
    tdesc->ln = ln;

    /* Large matrix derived parameters */
    /* tdesc->lm1 = (lm/mb); */
    /* tdesc->ln1 = (ln/nb); */
    tdesc->lmt = (lm%mb==0) ? (lm/mb) : (lm/mb+1);
    tdesc->lnt = (ln%nb==0) ? (ln/nb) : (ln/nb+1);

    /* Update lm and ln to include the padding */
    tdesc->lm = tdesc->lmt * tdesc->mb;
    tdesc->ln = tdesc->lnt * tdesc->nb;

    /* WARNING: This has to be removed when padding will be removed */
#if defined(HAVE_MPI)
    if ( storage == matrix_Lapack ) {
        if ( tdesc->lm %mb != 0 ) {
            fprintf(stderr, "In distributed with Lapack storage, lm has to be a multiple of mb\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if ( tdesc->ln %nb != 0 ) {
            fprintf(stderr, "In distributed with Lapack storage, ln has to be a multiple of nb\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
#endif

    /* Submatrix parameters */
    tdesc->i = i;
    tdesc->j = j;
    tdesc->m = m;
    tdesc->n = n;

    /* Submatrix derived parameters */
    tdesc->mt = (i+m-1)/mb - i/mb + 1;
    tdesc->nt = (j+n-1)/nb - j/nb + 1;

    assert(vpmap_get_nb_vp() > 0);

#if defined(DAGUE_PROF_TRACE)
    asprintf(&(tdesc->super.key_dim), "(%d, %d)", tdesc->lmt, tdesc->lnt);
#endif
    return;
}

tiled_matrix_desc_t *
tiled_matrix_submatrix( tiled_matrix_desc_t *tdesc,
                        int i, int j, int m, int n)
{
    int mb, nb;
    tiled_matrix_desc_t *newdesc;

    if( tdesc->dtype & two_dim_block_cyclic_type ) {
        newdesc = (tiled_matrix_desc_t*) malloc ( sizeof(two_dim_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(two_dim_block_cyclic_t) );
    }
    else if( tdesc->dtype & sym_two_dim_block_cyclic_type ) {
        newdesc = (tiled_matrix_desc_t*) malloc ( sizeof(sym_two_dim_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(sym_two_dim_block_cyclic_t) );
    } else {
        fprintf(stderr, "Type not completely defined\n");
        return NULL;
    }

    mb = tdesc->mb;
    nb = tdesc->nb;
    // Submatrix parameters
    newdesc->i = i;
    newdesc->j = j;
    newdesc->m = m;
    newdesc->n = n;
    // Submatrix derived parameters
    newdesc->mt = (i+m-1)/mb - i/mb + 1;
    newdesc->nt = (j+n-1)/nb - j/nb + 1;
    return newdesc;
}

/*
 * Writes the data into the file filename
 * Sequential function per node
 */
int tiled_matrix_data_write(tiled_matrix_desc_t *tdesc, char *filename)
{
    dague_ddesc_t *ddesc = &(tdesc->super);
    dague_data_t* data;
    FILE *tmpf;
    char *buf;
    int i, j, k;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  dague_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        fprintf(stderr, "ERROR: The file %s cannot be open\n", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    data = ddesc->data_of( ddesc, i, j );
                    buf = DAGUE_DATA_GET_PTR(data);
                    fwrite(buf, eltsize, tdesc->bsiz, tmpf );
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    data = ddesc->data_of( ddesc, i, j );
                    buf = DAGUE_DATA_GET_PTR(data);
                    for (k=0; k<tdesc->nb; k++) {
                        fwrite(buf, eltsize, tdesc->mb, tmpf );
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }


    fclose(tmpf);
    return 0;
}

/*
 * Read the data from the file filename
 * Sequential function per node
 */
int tiled_matrix_data_read(tiled_matrix_desc_t *tdesc, char *filename)
{
    dague_ddesc_t *ddesc = &(tdesc->super);
    dague_data_t* data;
    FILE *tmpf;
    char *buf;
    int i, j, k, ret;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  dague_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        fprintf(stderr, "ERROR: The file %s cannot be open\n", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    data = ddesc->data_of( ddesc, i, j );
                    buf = DAGUE_DATA_GET_PTR(data);
                    ret = fread(buf, eltsize, tdesc->bsiz, tmpf );
                    if ( ret !=  tdesc->bsiz ) {
                        fprintf(stderr, "ERROR: The read on tile(%d, %d) read %d elements instead of %d\n",
                                i, j, ret, tdesc->bsiz);
                        return -1;
                    }
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( ddesc->rank_of( ddesc, i, j ) == myrank ) {
                    data = ddesc->data_of( ddesc, i, j );
                    buf = DAGUE_DATA_GET_PTR(data);
                    for (k=0; k < tdesc->nb; k++) {
                        ret = fread(buf, eltsize, tdesc->mb, tmpf );
                        if ( ret !=  tdesc->mb ) {
                            fprintf(stderr, "ERROR: The read on tile(%d, %d) read %d elements instead of %d\n",
                                    i, j, ret, tdesc->mb);
                            return -1;
                        }
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }

    fclose(tmpf);
    return 0;
}
