/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#ifndef _DPLASMA_CORES_H_
#define _DPLASMA_CORES_H_

#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <plasma.h>
#include "dague.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"

/***************************************************************************//**
 *
 **/
static inline int plasma_element_size(int type)
{
    switch(type) {
    case PlasmaByte:          return          1;
    case PlasmaInteger:       return   sizeof(int);
    case PlasmaRealFloat:     return   sizeof(float);
    case PlasmaRealDouble:    return   sizeof(double);
    case PlasmaComplexFloat:  return 2*sizeof(float);
    case PlasmaComplexDouble: return 2*sizeof(double);
    default: /*plasma_fatal_error("plasma_element_size", "undefined type");*/
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
}
/***************************************************************************//**
 *  Internal function to return adress of block (m,n)
 **/
static inline void *plasma_getaddr(PLASMA_desc A, int m, int n)
{
    size_t mm = m+A.i/A.mb;
    size_t nn = n+A.j/A.nb;
    size_t eltsize = plasma_element_size(A.dtyp);
    size_t offset = 0;

    if (mm < (size_t)A.lm1) {
        if (nn < (size_t)A.ln1)
            offset = A.bsiz*(mm+A.lm1*nn);
        else
            offset = A.A12 + (A.mb*(A.ln%A.nb)*mm);
    }
    else {
        if (nn < (size_t)A.ln1)
            offset = A.A21 + ((A.lm%A.mb)*A.nb*nn);
        else
            offset = A.A22;
    }

    return (void*)((intptr_t)A.mat + (offset*eltsize) );
}

#define PLASMA_BLKADDR(A, type, m, n)  (type *)plasma_getaddr(A, m, n)
#define PLASMA_BLKLDD(A, k) ( ( (k) + (A).i/(A).mb) < (A).lm1 ? (A).mb : (A).lm%(A).mb )

#endif /* _DPLASMA_CORES_H_ */
