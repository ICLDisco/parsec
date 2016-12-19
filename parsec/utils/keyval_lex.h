/* -*- C -*-
 *
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#ifndef PARSEC_UTIL_KEYVAL_LEX_H_
#define PARSEC_UTIL_KEYVAL_LEX_H_

#include <parsec_config.h>

#ifdef malloc
#undef malloc
#endif
#ifdef realloc
#undef realloc
#endif
#ifdef free
#undef free
#endif

#include <stdio.h>
#if defined(PARSEC_HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(PARSEC_HAVE_STDBOOL_H) */

int parsec_util_keyval_yylex(void);
int parsec_util_keyval_init_buffer(FILE *file);
int parsec_util_keyval_yylex_destroy(void);

extern FILE *parsec_util_keyval_yyin;
extern bool parsec_util_keyval_parse_done;
extern char *parsec_util_keyval_yytext;
extern int parsec_util_keyval_yynewlines;

/*
 * Make lex-generated files not issue compiler warnings
 */
#define YY_STACK_USED 0
#define YY_ALWAYS_INTERACTIVE 0
#define YY_NEVER_INTERACTIVE 0
#define YY_MAIN 0
#define YY_NO_UNPUT 1
#define YY_SKIP_YYWRAP 1

enum {
    PARSEC_UTIL_KEYVAL_PARSE_DONE,
    PARSEC_UTIL_KEYVAL_PARSE_ERROR,

    PARSEC_UTIL_KEYVAL_PARSE_NEWLINE,
    PARSEC_UTIL_KEYVAL_PARSE_EQUAL,
    PARSEC_UTIL_KEYVAL_PARSE_SINGLE_WORD,
    PARSEC_UTIL_KEYVAL_PARSE_VALUE,

    PARSEC_UTIL_KEYVAL_PARSE_MAX
};

#endif
