/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
  
#include "zgebut.h"
#include "zhebut.h"

/* HE for Hermitian */

/*
 * dplasma_zhebut_New() applies the butterfly transformation on the submatrix (it,jt)(nt,nt) of A
 */
dague_object_t* 
dplasma_zhebut_New( tiled_matrix_desc_t *A, int it, int jt, int nt, int *info)
{
    dague_object_t *dague_zhebut = NULL;
    (void)info;
    if( nt%2 ){
        dplasma_error("dplasma_zhebut_New", "illegal number of tiles in matrix");
        return NULL;
    }

    dague_zhebut = (dague_object_t *)dague_zhebut_new(*A, (dague_ddesc_t*)A, it, jt, nt, nt);
    
    dplasma_add2arena_tile(((dague_zhebut_object_t*)dague_zhebut)->arenas[DAGUE_zhebut_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zhebut;
}

void
dplasma_zhebut_Destruct( dague_object_t *o )
{
    dague_zhebut_object_t *obut = (dague_zhebut_object_t *)o;
    
    dplasma_datatype_undefine_type( &(obut->arenas[DAGUE_zhebut_DEFAULT_ARENA]->opaque_dtt) );

    dague_zhebut_destroy(obut);
}

/* GE for General */

/*
 * dplasma_zgebut_New() applies the butterfly transformation on the submatrix (it,jt)(nt,nt) of A
 */
dague_object_t* 
dplasma_zgebut_New( tiled_matrix_desc_t *A, int it, int jt, int nt, int *info)
{
    dague_object_t *dague_zgebut = NULL;
    (void)info;
    if( nt%2 ){
        dplasma_error("dplasma_zgebut_New", "illegal number of tiles in matrix");
        return NULL;
    }

    dague_zgebut = (dague_object_t *)dague_zgebut_new(*A, (dague_ddesc_t*)A, it, jt, nt, nt);
    
    dplasma_add2arena_tile(((dague_zgebut_object_t*)dague_zgebut)->arenas[DAGUE_zgebut_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zgebut;
}

void
dplasma_zgebut_Destruct( dague_object_t *o )
{
    dague_zgebut_object_t *obut = (dague_zgebut_object_t *)o;
    
    dplasma_datatype_undefine_type( &(obut->arenas[DAGUE_zgebut_DEFAULT_ARENA]->opaque_dtt) );

    dague_zgebut_destroy(obut);
}

/*
 * Blocking Interface
 */

static dague_object_t **iterate_ops(tiled_matrix_desc_t *A, int curlevel,
	      				 	int maxlevel, int it, int jt,
					       	int nt, dague_object_t **subop,
					       	dague_context_t *dague, 
						int destroy, int *info)
{
    if(curlevel == maxlevel){
        if( it == jt ){
	    if( destroy ){
	        dplasma_zhebut_Destruct(*subop);
	    }else{
	        *subop = dplasma_zhebut_New(A, it, jt, nt, info);
	    }
	}else{
	    if( destroy ){
	        dplasma_zgebut_Destruct(*subop);
	    }else{
	        *subop = dplasma_zgebut_New(A, it, jt, nt, info);
	    }
	}
	if( !destroy ){
            dague_enqueue(dague, *subop);
	}
        return subop+1;
    }else{
        if( it == jt ){
            subop = iterate_ops(A, curlevel+1, maxlevel, it,      jt,      nt/2, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, it+nt/2, jt,      nt/2, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, it+nt/2, jt+nt/2, nt/2, subop, dague, destroy, info);
	}else{
            subop = iterate_ops(A, curlevel+1, maxlevel, it,      jt,      nt/2, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, it+nt/2, jt,      nt/2, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, it,      jt+nt/2, nt/2, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, it+nt/2, jt+nt/2, nt/2, subop, dague, destroy, info);
	}
        return subop;
    }

}


int dplasma_zhebut(dague_context_t *dague, tiled_matrix_desc_t *A, int level)
{
    dague_object_t **subop;
    int info = 0;
    int nbhe = 1<<level;
    int nbge = (1<<(level-1))*((1<<level)-1);
    int final_nt = A->nt/nbhe;
    if( final_nt == 0 ){
        fprintf(stderr,"Too many butterflies. Death by starvation.\n");
        return -1;
    }     
    if( A->nt%nbhe != 0 ){
        fprintf(stderr,"Please use a matrix size that is divisible by 2^level.\n");
        return -1;
    }     

    subop = (dague_object_t **)malloc((nbhe+nbge) * sizeof(dague_object_t*));
    
    (void)iterate_ops(A, 0, level, 0, 0, A->nt, subop, dague, 0, &info);    
    dplasma_progress(dague);
    (void)iterate_ops(A, 0, level, 0, 0, A->nt, subop, dague, 1, &info);    
    free(subop);
    return info;
}

