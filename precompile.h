#ifndef _precompile_h
#define _precompile_h

#include <stdio.h>

/**
 * Adds a specific preamble for a specific language
 *
 * @param [IN]  language the language in which this preamble is entered
 * @param [IN]  code the code for the preamble in this language
 *
 */
void dplasma_precompiler_add_preamble(const char *language, const char *code);

/**
 * Dump all defined dplasma_t objetcs in a C-like format in the out file
 *
 * @param [INOUT] out the file to dump to
 */
void dplasma_dump_all_c( FILE *out );

#endif
