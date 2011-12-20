/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "map2.h"

#define BLKLDD(_desc, _k) (_desc)->mb

#ifndef max
#define max(a, b) ((a)>(b)?(a):(b))
#endif

struct lanhe_args_s {
  PLASMA_enum ntype;
  tiled_matrix_desc_t *desc;
};
typedef struct lanhe_args_s lanhe_args_t;

static int
dague_operator_zlanhe_max( struct dague_execution_unit *eu,
                           const void* src,
                           void* dest,
                           void* op_data, ... )
{
    va_list ap;
    lanhe_args_t *args = (lanhe_args_t*)op_data;
    PLASMA_enum uplo;
    int m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;

    (void)eu;
    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    descA = args->desc;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

    if ( uplo == PlasmaUpperLower ) {
      CORE_zlange( args->ntype, tempmm, tempnn,
		   (PLASMA_Complex64_t*)src, ldam, NULL, (double*)dest );
    } else {
      CORE_zlanhe( args->ntype, uplo, tempmm, 
		   (PLASMA_Complex64_t*)src, ldam, NULL, (double*)dest );
    }    
    return 0;
}

/***************************************************************************/
/**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlanhe_New - Sets the elements of the matrix A on the diagonal
 *  to beta and on the off-diagonals to alpha
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          = PlasmaMaxNorm: Max norm
 *          = PlasmaOneNorm: One norm
 *          = PlasmaInfNorm: Infinity norm
 *          = PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in,out] A
 *         On entry, the M-by-N tile A.
 *         On exit, A has been set accordingly.
 *
 **/
#if 0
dague_object_t* dplasma_zlanhe_New( PLASMA_enum ntype,
                                    tiled_matrix_desc_t *A,
                                    double *result )
{
    dague_zlanhe_object_t* object;
    lanhe_args_t args;
    return (dague_object_t*)object;
}
#endif

double dplasma_zlanhe( dague_context_t *dague, 
		       PLASMA_enum ntype,
		       PLASMA_enum uplo,
		       tiled_matrix_desc_t *A) 
{
#if defined(DAGUE_DRY_RUN) || defined(DAGUE_PROF_DRY_BODY) || defined(DAGUE_PROF_DRY_DEP)
    return -1.0;
#else
    dague_operator_t op;
    double *work = NULL;
    two_dim_block_cyclic_t workD, workS;
    lanhe_args_t args;
    double result = -1.0;

    switch( ntype ) {
    case PlasmaFrobeniusNorm:
    case PlasmaInfNorm:
      fprintf(stderr, "zlanhe: Only PlasmaMaxNorm is supported\n");

    case PlasmaMaxNorm:
        PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
            workD, two_dim_block_cyclic, 
            (&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank,
             1, 1, A->mt, A->nt, 0, 0, A->mt, A->nt, 
             ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols, 
             ((two_dim_block_cyclic_t*)A)->grid.rows));

        op = dague_operator_zlanhe_max;
        break;

    /* case PlasmaOneNorm: */
    /*     two_dim_block_cyclic_init(&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank, */
    /*                               1, A->nb, A->mt, A->n, 0, 0, A->mt, A->n, */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.rows); */

    /*     op = dague_operator_zlanhe_one; */
    /*     work = (double *)malloc( max(A->n, A->mt) * sizeof(double) );  */
    /*     break; */

    /* case PlasmaInfNorm: */
    /*     two_dim_block_cyclic_init(&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank, */
    /*                               A->mb, 1, A->m,  A->nt, 0, 0, A->m,  A->nt,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.rows); */

    /*     op = dague_operator_zlanhe_inf; */
    /*     work = (double *)malloc( max(A->nt, A->m) * sizeof(double) );  */
    /*     break; */
    default:
        return -1;
    }

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&workD);

    args.ntype = ntype;
    args.desc = A;
    
    /* First reduction by tile */
    dplasma_map2( dague, uplo, A, (tiled_matrix_desc_t*)&workD, op, (void *)&args );

    /* Second one with on element (one double or one vector )  per tile */
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        workS, two_dim_block_cyclic, 
        (&workS, matrix_RealDouble, 1, workD.super.super.cores, workD.super.super.myrank,
         1, 1, A->mt, A->nt, 0, 0, A->mt, A->nt, 1, 1, 1));
    
    dplasma_zlacpy(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&workD, (tiled_matrix_desc_t*)&workS);

    if ( workS.super.super.myrank == 0 ) {
        CORE_dlansy(
            ntype, uplo, workS.super.m, 
            (double*)workS.mat, workS.super.lm, work, &result);
    }

    dague_data_free(workD.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&workD);
    dague_data_free(workS.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&workS);

    if ( work != NULL )
        free(work);
    
#if defined(HAVE_MPI)
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, dplasma_comm);
#endif

    return result;
#endif
}

#if 0
void
dplasma_zlanhe_Destruct( dague_object_t *o )
{
    dague_zlanhe_object_t *dague_zlanhe = (dague_zlanhe_object_t *)o;
    dague_zlanhe_destroy(dague_zlanhe);
}
#endif
