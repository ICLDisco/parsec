%option nounput noinput

%{ /* -*- C -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2022 The University of Tennessee and The University
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

/* When FLEX_FOUND is set, normal dependency tracking will generate the
 * .l.c in the build directory from the .l in the source directory (this file).
 * In addition, one shall invoke by hand the rule to update the pregen
 * .l.c files, before committing any changes to the .l files, e.g.,
 * `make parsec_pregen_flex_bison`.
 *
 * When FLEX_FOUND is not set, the .l.c version of this file is copied
 * from the pregenerated .l.c file in `contrib/pregen_flex_bison`, and
 * modifying this file will result in a compilation error.
 */

#include "parsec/parsec_config.h"

#include <stdio.h>
#ifdef PARSEC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "parsec/utils/show_help_lex.h"

BEGIN_C_DECLS

/*
 * public functions
 */
extern int parsec_show_help_finish_parsing(void);

/*
 * local functions
 */
static int parsec_show_help_yywrap(void);

END_C_DECLS

/*
 * global variables
 */
int parsec_show_help_yynewlines = 1;
bool parsec_show_help_parse_done = false;

%}

WHITE       [\f\t\v ]
CHAR        [A-Za-z0-9_\-\.]

%x CHOMP
%option prefix="parsec_show_help_yy"

%%

#.*\n               ; /* comment line */

^\[.+\]/[^\]\n]*\n { BEGIN(CHOMP); return PARSEC_SHOW_HELP_PARSE_TOPIC; }

<CHOMP>.*\n { BEGIN(INITIAL); }

.*/\n { BEGIN(CHOMP); return PARSEC_SHOW_HELP_PARSE_MESSAGE; }

%%

/* Old flex (2.5.4a? and older) does not define a destroy function */
#if !defined(YY_FLEX_SUBMINOR_VERSION)
#define YY_FLEX_SUBMINOR_VERSION 0
#endif

#if (YY_FLEX_MAJOR_VERSION < 2) || (YY_FLEX_MAJOR_VERSION == 2 && (YY_FLEX_MINOR_VERSION < 5 || (YY_FLEX_MINOR_VERSION == 5 && YY_FLEX_SUBMINOR_VERSION < 5)))
int parsec_show_help_yylex_destroy(void)
{
    if (NULL != YY_CURRENT_BUFFER) {
        yy_delete_buffer(YY_CURRENT_BUFFER); 
#if defined(YY_CURRENT_BUFFER_LVALUE)
        YY_CURRENT_BUFFER_LVALUE = NULL;
#else
        YY_CURRENT_BUFFER = NULL;
#endif  /* YY_CURRENT_BUFFER_LVALUE */
    }
    return YY_NULL;
}
#endif

static int parsec_show_help_yywrap(void)
{
    parsec_show_help_parse_done = true;
    return 1;
}


/*
 * Ensure that we have a valid yybuffer to use.  Specifically, if this
 * scanner is invoked a second time, finish_parsing() (above) will
 * have been executed, and the current buffer will have been freed.
 * Flex doesn't recognize this fact because as far as it's concerned,
 * its internal state was already initialized, so it thinks it should
 * have a valid buffer.  Hence, here we ensure to give it a valid
 * buffer.
 */
int parsec_show_help_init_buffer(FILE *file)
{
    YY_BUFFER_STATE buf = yy_create_buffer(file, YY_BUF_SIZE);
    yy_switch_to_buffer(buf);

    return 0;
}
