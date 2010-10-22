#include <precision.h>

#ifndef DAGUE_HAS_COMPLEX_H

#ifdef __cplusplus
extern "C" {
#endif

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

double cimag(Dague_Complex64_t z)
{
    return ((double *)&z)[1];
}

double creal(Dague_Complex64_t z)
{
    return ((double *)&z)[0];
}

Dague_Complex64_t conj(Dague_Complex64_t z)
{
    double *zp, *vp;
    Dague_Complex64_t v;

    zp = (double *)&z;
    vp = (double *)&v;
    vp[0] = zp[0];
    vp[1] = -zp[1];
    return v;
}

#ifdef __cplusplus
}
#endif

#endif /* DAGUE_HAS_COMPLEX */
