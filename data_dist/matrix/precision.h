/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _PRECISION_H_
#define _PRECISION_H_

/******************************************************************************
 * Dague Complex numbers
 **/

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* Windows and non-Intel compiler */
#include <complex>
typedef std::complex<float>  dague_complex32_t;
typedef std::complex<double> dague_complex64_t;
#else
typedef float  _Complex dague_complex32_t;
typedef double _Complex dague_complex64_t;
#endif


#ifdef __cplusplus
extern "C" {
#endif

#if defined(DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED)

#if !defined(__cplusplus) && defined(HAVE_COMPLEX_H)
#include <complex.h>
#else

/* These declarations will not clash with what C++ provides because the names in C++ are name-mangled. */

extern double cabs     (dague_complex64_t z);
extern double carg     (dague_complex64_t z);
extern double creal    (dague_complex64_t z);
extern double cimag    (dague_complex64_t z);

extern float  cabsf    (dague_complex32_t z);
extern float  cargf    (dague_complex32_t z);
extern float  crealf   (dague_complex32_t z);
extern float  cimagf   (dague_complex32_t z);

extern dague_complex64_t conj  (dague_complex64_t z);
extern dague_complex64_t cproj (dague_complex64_t z);
extern dague_complex64_t csqrt (dague_complex64_t z);
extern dague_complex64_t cexp  (dague_complex64_t z);
extern dague_complex64_t clog  (dague_complex64_t z);
extern dague_complex64_t cpow  (dague_complex64_t z, dague_complex64_t w);

extern dague_complex32_t conjf (dague_complex32_t z);
extern dague_complex32_t cprojf(dague_complex32_t z);
extern dague_complex32_t csqrtf(dague_complex32_t z);
extern dague_complex32_t cexpf (dague_complex32_t z);
extern dague_complex32_t clogf (dague_complex32_t z);
extern dague_complex32_t cpowf (dague_complex32_t z, dague_complex32_t w);

#endif /* DAGUE_HAS_COMPLEX_H */

#endif /* defined(DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED) */

#ifdef __cplusplus
}
#endif

#endif /* _PRECISION_H_ */
