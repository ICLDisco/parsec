/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DPLASMA_COMPLEX_H_
#define _DPLASMA_COMPLEX_H_

/******************************************************************************
 * PaRSEC Complex numbers
 **/

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* Windows and non-Intel compiler */
#include <complex>
typedef std::complex<float>  parsec_complex32_t;
typedef std::complex<double> parsec_complex64_t;
#else
typedef float  _Complex parsec_complex32_t;
typedef double _Complex parsec_complex64_t;
#endif


#ifdef __cplusplus
extern "C" {
#endif

#if defined(PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED)

#if !defined(__cplusplus) && defined(PARSEC_HAVE_COMPLEX_H)
#include <complex.h>
#else

/* These declarations will not clash with what C++ provides because the names in C++ are name-mangled. */

extern double cabs     (parsec_complex64_t z);
extern double carg     (parsec_complex64_t z);
extern double creal    (parsec_complex64_t z);
extern double cimag    (parsec_complex64_t z);

extern float  cabsf    (parsec_complex32_t z);
extern float  cargf    (parsec_complex32_t z);
extern float  crealf   (parsec_complex32_t z);
extern float  cimagf   (parsec_complex32_t z);

extern parsec_complex64_t conj  (parsec_complex64_t z);
extern parsec_complex64_t cproj (parsec_complex64_t z);
extern parsec_complex64_t csqrt (parsec_complex64_t z);
extern parsec_complex64_t cexp  (parsec_complex64_t z);
extern parsec_complex64_t clog  (parsec_complex64_t z);
extern parsec_complex64_t cpow  (parsec_complex64_t z, parsec_complex64_t w);

extern parsec_complex32_t conjf (parsec_complex32_t z);
extern parsec_complex32_t cprojf(parsec_complex32_t z);
extern parsec_complex32_t csqrtf(parsec_complex32_t z);
extern parsec_complex32_t cexpf (parsec_complex32_t z);
extern parsec_complex32_t clogf (parsec_complex32_t z);
extern parsec_complex32_t cpowf (parsec_complex32_t z, parsec_complex32_t w);

#endif /* PARSEC_HAS_COMPLEX_H */

#endif /* defined(PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED) */

#ifdef __cplusplus
}
#endif

#endif /* _DPLASMA_COMPLEX_H_ */
