/*
 * Copyright (c) 2010     The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_CONFIG_H_HAS_BEEN_INCLUDED
#error "dague_config_bottom.h should only be included from dague_config.h"
#endif

/*
 * Flex is trying to include the unistd.h file. As there is no configure
 * option or this, the flex generated files will try to include the file
 * even on platforms without unistd.h (such as Windows). Therefore, if we
 * know this file is not available, we can prevent flex from including it.
 */
#ifndef HAVE_UNISTD_H
#define YY_NO_UNISTD_H
#endif

/*
 * BEGIN_C_DECLS should be used at the beginning of your declarations,
 * so that C++ compilers don't mangle their names.  Use END_C_DECLS at
 * the end of C declarations.
 */
#undef BEGIN_C_DECLS
#undef END_C_DECLS
#if defined(c_plusplus) || defined(__cplusplus)
# define BEGIN_C_DECLS extern "C" {
# define END_C_DECLS }
#else
#define BEGIN_C_DECLS          /* empty */
#define END_C_DECLS            /* empty */
#endif

#if defined(HAVE_MPI)
# define DISTRIBUTED
#else
# undef DISTRIBUTED
#endif

#if defined(DAGUE_DEBUG_DRY_RUN)
# define DAGUE_DEBUG_DRY_BODY
# define DAGUE_DEBUG_DRY_DEP
#endif

