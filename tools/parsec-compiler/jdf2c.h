#ifndef _jdf2c_h
#define _jdf2c_h

#include "jdf.h"

/**
 * The namespace used by the compiler to generate internal variables.
 */
#define JDF2C_NAMESPACE "__jdf2c_"

int jdf_optimize( jdf_t* jdf );

int jdf2c(const char *output_c, const char *output_h, const char *_basename, jdf_t *jdf);

#endif  /* _jdf2c_h */
