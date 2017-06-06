/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#ifndef PARSEC_SHOW_HELP_LEX_H
#define PARSEC_SHOW_HELP_LEX_H

#if defined(PARSEC_HAVE_STDBOOL_H)
#include <stdbool.h>
#endif  /* defined(PARSEC_HAVE_STDBOOL_H) */

#include <stdio.h>
BEGIN_C_DECLS
int parsec_show_help_yylex(void);
int parsec_show_help_init_buffer(FILE *file);
int parsec_show_help_yylex_destroy(void);

extern FILE *parsec_show_help_yyin;
extern bool parsec_show_help_parse_done;
extern char *parsec_show_help_yytext;
extern int parsec_show_help_yynewlines;

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
    PARSEC_SHOW_HELP_PARSE_DONE,
    PARSEC_SHOW_HELP_PARSE_ERROR,

    PARSEC_SHOW_HELP_PARSE_TOPIC,
    PARSEC_SHOW_HELP_PARSE_MESSAGE,

    PARSEC_SHOW_HELP_PARSE_MAX
};
END_C_DECLS
#endif
