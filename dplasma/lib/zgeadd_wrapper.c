/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "dague.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "map2.h"

#define BLKLDD(_desc, _k) (_desc).mb

struct zgeadd_args_s {
  Dague_Complex64_t alpha;
  tiled_matrix_desc_t *descA;
  tiled_matrix_desc_t *descB;
};
typedef struct zgeadd_args_s zgeadd_args_t;

static int
dague_operator_zgeadd( struct dague_execution_unit *eu,
		      const void* _A,
		      void* _B,
		      void* op_data, ... )
{
    va_list ap;
    zgeadd_args_t *args = (zgeadd_args_t*)op_data;
    PLASMA_enum uplo;
    int j, m, n;
    int tempmm, tempnn, ldam, ldbm;
    tiled_matrix_desc_t *descA, *descB;
    Dague_Complex64_t *A = (Dague_Complex64_t*)_A;
    Dague_Complex64_t *B = (Dague_Complex64_t*)_B;
    (void)eu;
    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);

    descA = args->descA;
    descB = args->descB;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );
    ldbm = BLKLDD( *descB, m );

    switch ( uplo ) {
    case PlasmaLower:
      for (j = 0; j < tempnn; j++, tempmm--, A+=ldam+1, B+=ldbm+1) {
	cblas_zaxpy(tempmm, CBLAS_SADDR(args->alpha), A, 1, B, 1);
      }
      break;
    case PlasmaUpper:
      for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
	cblas_zaxpy(j+1, CBLAS_SADDR(args->alpha), A, 1, B, 1);
      }
      break;
    case PlasmaUpperLower:
    default:
      for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
	cblas_zaxpy(tempmm, CBLAS_SADDR(args->alpha), A, 1, B, 1);
      }
/*       CORE_zgeadd( tempmm, tempnn, args->alpha, */
/* 		  A, ldam, B, ldbm); */
    }

    return 0;
}

/***************************************************************************//**
 *
 * @ingroup Dague_Complex64_t
 *
 *  dplasma_zgeadd_New - Compute the operation B = alpha * A + B
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The matrix A of size M-by-N
 *
 * @param[in,out] B
 *          On entry, the matrix B of size equal or greater to M-by-N
 *          On exit, the matrix B with the M-by-N part overwrite by alpha*A+B
 *
 ******************************************************************************/
dague_object_t* dplasma_zgeadd_New( PLASMA_enum uplo, Dague_Complex64_t alpha,
				   tiled_matrix_desc_t *A,
				   tiled_matrix_desc_t *B)
{
    dague_map2_object_t* object;
    zgeadd_args_t *params = (zgeadd_args_t*)malloc(sizeof(zgeadd_args_t));

    params->alpha = alpha;
    params->descA = A;
    params->descB = B;

    object = dague_map2_new((dague_ddesc_t*)B, (dague_ddesc_t*)A, 
			    uplo, *A, *B, 
			    dague_operator_zgeadd, (void *)params);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_map2_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_object_t*)object;
}

int dplasma_zgeadd( dague_context_t *dague,
		   PLASMA_enum uplo,
		   Dague_Complex64_t alpha,
		   tiled_matrix_desc_t *A,
		   tiled_matrix_desc_t *B) 
{
    dague_object_t *dague_zgeadd = NULL;

    dague_zgeadd = dplasma_zgeadd_New(uplo, alpha, A, B);

    dague_enqueue(dague, (dague_object_t*)dague_zgeadd);
    dplasma_progress(dague);

    dplasma_zgeadd_Destruct( dague_zgeadd );
    return 0;
}

void
dplasma_zgeadd_Destruct( dague_object_t *o )
{
    dague_map2_object_t *dague_zgeadd = (dague_map2_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zgeadd->arenas[DAGUE_map2_DEFAULT_ARENA   ]->opaque_dtt) );
    free(dague_zgeadd->op_args);
    dague_map2_destroy(dague_zgeadd);
}

