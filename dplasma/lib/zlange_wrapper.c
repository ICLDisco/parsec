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

struct lange_args_s {
  PLASMA_enum ntype;
  tiled_matrix_desc_t *desc;
};
typedef struct lange_args_s lange_args_t;

static int
dague_operator_zlange_max( struct dague_execution_unit *eu,
                           const void* src,
                           void* dest,
                           void* op_data, ... )
{
    va_list ap;
    lange_args_t *args = (lange_args_t*)op_data;
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

    (void)uplo;

    descA = args->desc;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

    CORE_zlange( args->ntype, tempmm, tempnn, 
                 (PLASMA_Complex64_t*)src, ldam, NULL, (double*)dest );
    
    return 0;
}

#if 0
static int
dague_operator_zlange_one( struct dague_execution_unit *eu,
                           const void* src,
                           void* dest,
                           void* op_data, ... )
{
    va_list ap;
    lange_args_t *args = (lange_args_t*)op_data;
    int m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;

    (void)eu;
    va_start(ap, op_data);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    descA = args->desc;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

    CORE_dzasum(
        PlasmaColumnwise, PlasmaUpperLower,
        tempmm, tempnn, 
        (PLASMA_Complex64_t*)src, ldam, (double *)dest);

    return 0;
}

static int
dague_operator_zlange_inf( struct dague_execution_unit *eu,
                           const void* src,
                           void* dest,
                           void* op_data, ... )
{
    va_list ap;
    lange_args_t *args = (lange_args_t*)op_data;
    int m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;

    (void)eu;
    va_start(ap, op_data);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    descA = args->desc;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

     CORE_dzasum(
         PlasmaRowwise, PlasmaUpperLower,
         tempmm, tempnn, (PLASMA_Complex64_t*)src, ldam, (double*)dest );
    return 0;
}
#endif

/***************************************************************************/
/**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlange_New - Sets the elements of the matrix A on the diagonal
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
dague_object_t* dplasma_zlange_New( PLASMA_enum ntype,
                                    tiled_matrix_desc_t *A,
                                    double *result )
{
    dague_zlange_object_t* object;
    lange_args_t args;
    return (dague_object_t*)object;
}
#endif

double dplasma_zlange( dague_context_t *dague, 
                    PLASMA_enum ntype,
                    tiled_matrix_desc_t *A) 
{
    dague_object_t *dague_zlange = NULL;
    dague_operator_t op;
    double *work = NULL;
    two_dim_block_cyclic_t workD, workS;
    lange_args_t args;
    double result;

    switch( ntype ) {
    case PlasmaFrobeniusNorm:
    case PlasmaInfNorm:
      fprintf(stderr, "zlange: Only PlasmaMaxNorm is supported\n");

    case PlasmaMaxNorm:
        two_dim_block_cyclic_init(&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank,
                                  1, 1, A->mt, A->nt, 0, 0, A->mt, A->nt, 
                                  ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols, 
                                  ((two_dim_block_cyclic_t*)A)->grid.rows);

        op = dague_operator_zlange_max;
        break;

    /* case PlasmaOneNorm: */
    /*     two_dim_block_cyclic_init(&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank, */
    /*                               1, A->nb, A->mt, A->n, 0, 0, A->mt, A->n, */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.rows); */

    /*     op = dague_operator_zlange_one; */
    /*     work = (double *)malloc( max(A->n, A->mt) * sizeof(double) );  */
    /*     break; */

    /* case PlasmaInfNorm: */
    /*     two_dim_block_cyclic_init(&workD, matrix_RealDouble, A->super.nodes, A->super.cores, A->super.myrank, */
    /*                               A->mb, 1, A->m,  A->nt, 0, 0, A->m,  A->nt,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,  */
    /*                               ((two_dim_block_cyclic_t*)A)->grid.rows); */

    /*     op = dague_operator_zlange_inf; */
    /*     work = (double *)malloc( max(A->nt, A->m) * sizeof(double) );  */
    /*     break; */
    default:
        return -1;
    }

    workD.mat = dague_data_allocate((size_t)workD.super.nb_local_tiles * 
                                    (size_t)workD.super.bsiz * 
                                    (size_t)workD.super.mtype);

    args.ntype = ntype;
    args.desc = A;
    
    /* First reduction by tile */
    dague_zlange = (dague_object_t*) dague_map2_new((dague_ddesc_t*)&workD, (dague_ddesc_t*)A, 
                                                    PlasmaUpperLower,
						    *A, workD.super, 
                                                    op, (void *)&args);
    dague_enqueue( dague, (dague_object_t*)dague_zlange);
    dplasma_progress(dague);
    dague_map2_destroy( (dague_map2_object_t*)dague_zlange );

    /* Second one with on element (one double or one vector )  per tile */
    two_dim_block_cyclic_init(&workS, matrix_RealDouble, workD.super.super.nodes, workD.super.super.cores, workD.super.super.myrank,
                              1, 1, workD.super.mt, workD.super.nt, 0, 0, workD.super.mt, workD.super.nt, workD.super.mt, workD.super.nt, 1);
    workS.mat = dague_data_allocate((size_t)workS.super.nb_local_tiles * 
                                    (size_t)workS.super.bsiz * 
                                    (size_t)workS.super.mtype);

    
    dplasma_zlacpy(dague, PlasmaUpperLower, (tiled_matrix_desc_t*)&workD, (tiled_matrix_desc_t*)&workS);

    if ( workS.super.super.myrank == 0 ) {
        CORE_dlange(
            ntype, workS.super.m, workS.super.n, 
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
}

#if 0
void
dplasma_zlange_Destruct( dague_object_t *o )
{
    dague_zlange_object_t *dague_zlange = (dague_zlange_object_t *)o;
    dague_zlange_destroy(dague_zlange);
}
#endif
