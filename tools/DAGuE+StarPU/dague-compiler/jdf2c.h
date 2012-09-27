#ifndef _jdf2c_h
#define _jdf2c_h

#include "jdf.h"

int jdf_optimize( jdf_t* jdf );

int jdf2c(const char *output_c, const char *output_h, const char *_basename, const jdf_t *jdf);

#endif
