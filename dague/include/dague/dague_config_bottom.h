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
#ifndef DAGUE_HAVE_UNISTD_H
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

#if defined(DAGUE_HAVE_ATTRIBUTE_VISIBILITY)
#    define __dague_attribute_visibility__(a) __attribute__((__visibility__(a)))
#else
#    define __dague_attribute_visibility__(a)
#endif

#if defined(DAGUE_HAVE_ATTRIBUTE_ALWAYS_INLINE)
#    define __dague_attribute_always_inline__ __attribute__((__always_inline__))
#else
#    define __dague_attribute_always_inline__
#endif

#if defined(DAGUE_HAVE_BUILTIN_EXPECT)
#define DAGUE_LIKELY(x)       __builtin_expect(!!(x), 1)
#define DAGUE_UNLIKELY(x)     __builtin_expect(!!(x), 0)
#else
#define DAGUE_LIKELY(x)       (x)
#define DAGUE_UNLIKELY(x)     (x)
#endif

#if defined(DAGUE_HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* DAGUE_HAVE_STDDEF_H */
#include <stdint.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#if defined(DAGUE_HAVE_MPI)
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


/***********************************************************************
 *
 * Windows library interface declaration code
 *
 **********************************************************************/
#if !defined(__WINDOWS__)
#  if defined(_WIN32) || defined(WIN32) || defined(WIN64)
#    define __WINDOWS__
#  endif
#endif  /* !defined(__WINDOWS__) */

#if defined(__WINDOWS__)

#  if defined(_USRDLL)    /* building shared libraries (.DLL) */
#    if defined(DAGUE_EXPORTS)
#      define DAGUE_DECLSPEC        __declspec(dllexport)
#      define DAGUE_MODULE_DECLSPEC
#    else
#      if defined(DAGUE_IMPORTS)
#        define DAGUE_DECLSPEC      __declspec(dllimport)
#      else
#        define DAGUE_DECLSPEC
#      endif  /*defined(DAGUE_IMPORTS)*/
#      if defined(DAGUE_MODULE_EXPORTS)
#        define DAGUE_MODULE_DECLSPEC __declspec(dllexport)
#      else
#        define DAGUE_MODULE_DECLSPEC __declspec(dllimport)
#      endif  /* defined(DAGUE_MODULE_EXPORTS) */
#    endif  /* defined(DAGUE_EXPORTS) */
#  else          /* building static library */
#    if defined(DAGUE_IMPORTS)
#      define DAGUE_DECLSPEC        __declspec(dllimport)
#    else
#      define DAGUE_DECLSPEC
#    endif  /* defined(DAGUE_IMPORTS) */
#    define DAGUE_MODULE_DECLSPEC
#  endif  /* defined(_USRDLL) */
#  include "dague/win32/win_compat.h"
#else
#  if defined(DAGUE_C_DAGUE_HAVE_VISIBILITY)
#    define DAGUE_DECLSPEC           __dague_attribute_visibility__("default")
#    define DAGUE_MODULE_DECLSPEC    __dague_attribute_visibility__("default")
#  else
#    define DAGUE_DECLSPEC
#    define DAGUE_MODULE_DECLSPEC
#  endif
#endif  /* defined(__WINDOWS__) */

/*
 * Set the compile-time path-separator on this system and variable separator
 */
#ifdef __WINDOWS__
#define DAGUE_PATH_SEP "\\"
#define DAGUE_ENV_SEP  ';'
#define MAXPATHLEN _MAX_PATH
#else
#define DAGUE_PATH_SEP "/"
#define DAGUE_ENV_SEP  ':'
#endif
