/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
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

#if defined(HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* HAVE_STDDEF_H */
#include <stdint.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#if defined(HAVE_MPI)
# define DISTRIBUTED
#else
# undef DISTRIBUTED
#endif

/*#define DAGUE_HARD_SUPERTILE */

#if defined(DAGUE_PROF_DRY_RUN)
# define DAGUE_PROF_DRY_BODY
# define DAGUE_PROF_DRY_DEP
#endif

#if DAGUE_DIST_EAGER_LIMIT == 0
#define RDEP_MSG_EAGER_LIMIT    0
#else
#define RDEP_MSG_EAGER_LIMIT    ((DAGUE_DIST_EAGER_LIMIT)*1024)
#endif

#if DAGUE_DIST_SHORT_LIMIT == 0
#define RDEP_MSG_SHORT_LIMIT    0
#else
#define RDEP_MSG_SHORT_LIMIT    ((DAGUE_DIST_SHORT_LIMIT)*1024)
#endif

#if defined(DAGUE_SCHED_DEPS_MASK)
typedef uint32_t dague_dependency_t;
#else
/**
 * Should be large enough to support MAX_PARAM_COUNT values.
 */
typedef uint32_t dague_dependency_t;

#endif

/*
 * A set of constants defining the capabilities of the underlying
 * runtime.
 */
#define MAX_LOCAL_COUNT  20
#define MAX_PARAM_COUNT  20

#define MAX_DEP_IN_COUNT  10
#define MAX_DEP_OUT_COUNT 10

#define MAX_TASK_STRLEN 128

#define COMPARISON_VAL(it, off)                 (*((int*)(((uintptr_t)it)+off)))
#define HIGHER_IS_BETTER
#if defined(HIGHER_IS_BETTER)
#define A_LOWER_PRIORITY_THAN_B(a, b, off)      (COMPARISON_VAL((a), (off)) <  COMPARISON_VAL((b), (off)))
#define A_HIGHER_PRIORITY_THAN_B(a, b, off)     (COMPARISON_VAL((a), (off)) >  COMPARISON_VAL((b), (off)))
#define SET_HIGHEST_PRIORITY(task, off)         (*((int*)(((uintptr_t)task)+off))) = 0x7fffffff;
#define SET_LOWEST_PRIORITY(task, off)          (*((int*)(((uintptr_t)task)+off))) = 0xffffffff;
#else
#define A_LOWER_PRIORITY_THAN_B(a, b, off)      (COMPARISON_VAL((a), (off)) >  COMPARISON_VAL((b), (off)))
#define A_HIGHER_PRIORITY_THAN_B(a, b, off)     (COMPARISON_VAL((a), (off)) <  COMPARISON_VAL((b), (off)))
#define SET_HIGHEST_PRIORITY(task, off)         (*((int*)(((uintptr_t)task)+off))) = 0xffffffff;
#define SET_LOWEST_PRIORITY(task, off)          (*((int*)(((uintptr_t)task)+off))) = 0x7fffffff;
#endif

typedef struct dague_remote_deps_s           dague_remote_deps_t;
typedef struct dague_arena_t                 dague_arena_t;
typedef struct dague_arena_chunk_t           dague_arena_chunk_t;
typedef struct dague_data_pair_t             dague_data_pair_t;
typedef struct _moesi_master                 moesi_master_t;
typedef struct _moesi_map                    moesi_map_t;
typedef struct dague_function_s              dague_function_t;
typedef struct dague_dependencies_t          dague_dependencies_t;
/**< The most basic execution flow. Each virtual process includes
 *   multiple execution units (posix threads + local data) */
typedef struct dague_execution_unit          dague_execution_unit_t;
/**< Each distributed process includes multiple virtual processes */
typedef struct dague_vp                      dague_vp_t;
/* The description of the content of each data mouvement/copy */
typedef struct dague_dep_data_description_s  dague_dep_data_description_t;

