/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <precision.h>

#ifndef DAGUE_HAVE_COMPLEX_H

#include <math.h>

float cabsf(float _Complex z)
{
    float *zp, x, y;
    zp = (float *)&z;

    /* first find out the large component */
    if (zp[0] > zp[1]) {
        x = zp[0];
        y = zp[1];
    } else {
        x = zp[1];
        y = zp[0];
    }

    return fabsf(x) * sqrtf(1.0f + y / x);
}

double cabs(double _Complex z)
{
    double *zp, x, y;
    zp = (double *)&z;

    /* first find out the large component */
    if (zp[0] > zp[1]) {
        x = zp[0];
        y = zp[1];
    } else {
        x = zp[1];
        y = zp[0];
    }

    return fabs(x) * sqrt(1.0f + y / x);
}

double cimag(dague_complex64_t z)
{
    return ((double *)&z)[1];
}

double creal(dague_complex64_t z)
{
    return ((double *)&z)[0];
}

dague_complex64_t conj(dague_complex64_t z)
{
    double *zp, *vp;
    dague_complex64_t v;

    zp = (double *)&z;
    vp = (double *)&v;
    vp[0] = zp[0];
    vp[1] = -zp[1];
    return v;
}

#endif /* DAGUE_HAS_COMPLEX */
