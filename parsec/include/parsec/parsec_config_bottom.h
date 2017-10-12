/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_CONFIG_BOTTOM_H_HAS_BEEN_INCLUDED
#define PARSEC_CONFIG_BOTTOM_H_HAS_BEEN_INCLUDED

#if !defined(PARSEC_CONFIG_H_HAS_BEEN_INCLUDED)
#error "parsec_config_bottom.h should only be included from parsec_config.h"
#endif

/*
 * Flex is trying to include the unistd.h file. As there is no configure
 * option or this, the flex generated files will try to include the file
 * even on platforms without unistd.h (such as Windows). Therefore, if we
 * know this file is not available, we can prevent flex from including it.
 */
#ifndef PARSEC_HAVE_UNISTD_H
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

#if defined(PARSEC_HAVE_ATTRIBUTE_VISIBILITY) && defined(BUILD_PARSEC)
#    define __parsec_attribute_visibility__(a) __attribute__((__visibility__(a)))
#else
#    define __parsec_attribute_visibility__(a)
#endif

#if defined(PARSEC_HAVE_ATTRIBUTE_ALWAYS_INLINE) && defined(BUILD_PARSEC)
#    define __parsec_attribute_always_inline__ __attribute__((__always_inline__))
#else
#    define __parsec_attribute_always_inline__
#endif

#if defined(PARSEC_HAVE_BUILTIN_EXPECT) && defined(BUILD_PARSEC)
#define PARSEC_LIKELY(x)       __builtin_expect(!!(x), 1)
#define PARSEC_UNLIKELY(x)     __builtin_expect(!!(x), 0)
#else
#define PARSEC_LIKELY(x)       (x)
#define PARSEC_UNLIKELY(x)     (x)
#endif

#if defined(PARSEC_HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* PARSEC_HAVE_STDDEF_H */
#include <stdint.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#if defined(PARSEC_HAVE_MPI)
# define DISTRIBUTED
#else
# undef DISTRIBUTED
#endif

/*#define PARSEC_HARD_SUPERTILE */

#if defined(PARSEC_PROF_DRY_RUN)
# define PARSEC_PROF_DRY_BODY
# define PARSEC_PROF_DRY_DEP
#endif

#if PARSEC_DIST_EAGER_LIMIT == 0
#define RDEP_MSG_EAGER_LIMIT    0
#else
#define RDEP_MSG_EAGER_LIMIT    ((PARSEC_DIST_EAGER_LIMIT)*1024)
#endif

#if PARSEC_DIST_SHORT_LIMIT == 0
#define RDEP_MSG_SHORT_LIMIT    0
#else
#define RDEP_MSG_SHORT_LIMIT    ((PARSEC_DIST_SHORT_LIMIT)*1024)
#endif

#if defined(PARSEC_SCHED_DEPS_MASK)
typedef uint32_t parsec_dependency_t;
#else
/**
 * Should be large enough to support MAX_PARAM_COUNT values.
 */
typedef uint32_t parsec_dependency_t;
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

#define PARSEC_MAX_STR_KEY_LEN 64

#define COMPARISON_VAL(it, off)                 (*((int*)(((uintptr_t)(it))+off)))
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

#if defined(PARSEC_HAVE_ATTRIBUTE_FORMAT_PRINTF)
#define PARSEC_ATTRIBUTE_FORMAT_PRINTF(a, b) __attribute__ ((format (printf, a, b)))
#else
#define PARSEC_ATTRIBUTE_FORMAT_PRINTF(a, b)
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
#    if defined(PARSEC_EXPORTS)
#      define PARSEC_DECLSPEC        __declspec(dllexport)
#      define PARSEC_MODULE_DECLSPEC
#    else
#      if defined(PARSEC_IMPORTS)
#        define PARSEC_DECLSPEC      __declspec(dllimport)
#      else
#        define PARSEC_DECLSPEC
#      endif  /*defined(PARSEC_IMPORTS)*/
#      if defined(PARSEC_MODULE_EXPORTS)
#        define PARSEC_MODULE_DECLSPEC __declspec(dllexport)
#      else
#        define PARSEC_MODULE_DECLSPEC __declspec(dllimport)
#      endif  /* defined(PARSEC_MODULE_EXPORTS) */
#    endif  /* defined(PARSEC_EXPORTS) */
#  else          /* building static library */
#    if defined(PARSEC_IMPORTS)
#      define PARSEC_DECLSPEC        __declspec(dllimport)
#    else
#      define PARSEC_DECLSPEC
#    endif  /* defined(PARSEC_IMPORTS) */
#    define PARSEC_MODULE_DECLSPEC
#  endif  /* defined(_USRDLL) */
#  include "parsec/win32/win_compat.h"
#else
#  if defined(PARSEC_C_PARSEC_HAVE_VISIBILITY)
#    define PARSEC_DECLSPEC           __parsec_attribute_visibility__("default")
#    define PARSEC_MODULE_DECLSPEC    __parsec_attribute_visibility__("default")
#  else
#    define PARSEC_DECLSPEC
#    define PARSEC_MODULE_DECLSPEC
#  endif
#endif  /* defined(__WINDOWS__) */

/*
 * Set the compile-time path-separator on this system and variable separator
 */
#ifdef __WINDOWS__
#define PARSEC_PATH_SEP "\\"
#define PARSEC_ENV_SEP  ';'
#define MAXPATHLEN _MAX_PATH
#else
#define PARSEC_PATH_SEP "/"
#define PARSEC_ENV_SEP  ':'
#endif

#endif  /* PARSEC_CONFIG_BOTTOM_H_HAS_BEEN_INCLUDED */

