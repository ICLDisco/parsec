/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _PRECISION_H_ 
#define _PRECISION_H_ 

/** ****************************************************************************
 * Dague Complex numbers
 **/
#define DAGUE_HAS_COMPLEX_H 1

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* Windows and non-Intel compiler */
#include <complex>
typedef std::complex<float>  Dague_Complex32_t;
typedef std::complex<double> Dague_Complex64_t;
#undef DAGUE_HAS_COMPLEX_H 1
#else
typedef float  _Complex Dague_Complex32_t;
typedef double _Complex Dague_Complex64_t;
#endif

/* Sun doesn't ship the complex.h header. Sun Studio doesn't have it and older GCC compilers don't have it either. */
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC) || defined(sun) || defined(__sun)
#undef DAGUE_HAS_COMPLEX_H
#endif

#ifdef DAGUE_HAS_COMPLEX_H
#include <complex.h>
#else

#ifdef __cplusplus
extern "C" {
#endif

/* These declarations will not clash with what C++ provides because the names in C++ are name-mangled. */

extern double cabs(double _Complex z);
extern float  cabsf(float _Complex z);
extern double cimag(Dague_Complex64_t z);
extern double creal(Dague_Complex64_t z);
extern Dague_Complex64_t conj(Dague_Complex64_t z);

#ifdef __cplusplus
}
#endif

#endif /* DAGUE_HAS_COMPLEX_H */

#endif /* _PRECISION_H_ */
