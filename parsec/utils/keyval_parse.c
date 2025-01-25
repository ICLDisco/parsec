/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
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

#include "parsec/parsec_config.h"

#include "parsec/constants.h"
#include "parsec/utils/keyval_parse.h"
#include "parsec/utils/keyval_lex.h"
#include "parsec/utils/output.h"
#ifdef PARSEC_HAVE_STRING_H
#include <string.h>
#endif  /* PARSEC_HAVE_STRING_H */
#include <pthread.h>

static const char *keyval_filename;
static parsec_keyval_parse_fn_t keyval_callback;
static char *key_buffer = NULL;
static size_t key_buffer_len = 0;
static pthread_mutex_t keyval_mutex = PTHREAD_MUTEX_INITIALIZER;

static int parse_line(void);
static void parse_error(int num);

int parsec_util_keyval_parse_init(void)
{
    return PARSEC_SUCCESS;
}


int
parsec_util_keyval_parse_finalize(void)
{
    if (NULL != key_buffer) free(key_buffer);

    return PARSEC_SUCCESS;
}


int
parsec_util_keyval_parse(const char *filename,
                        parsec_keyval_parse_fn_t callback)
{
    int val;
    int ret = PARSEC_SUCCESS;;

    pthread_mutex_lock(&keyval_mutex);

    keyval_filename = filename;
    keyval_callback = callback;

    parsec_util_keyval_yyin = fopen(keyval_filename, "r");
    if (NULL == parsec_util_keyval_yyin) {
        ret = PARSEC_ERR_NOT_FOUND;
        goto cleanup;
    }

    parsec_util_keyval_parse_done = false;
    parsec_util_keyval_yynewlines = 1;
    parsec_util_keyval_init_buffer(parsec_util_keyval_yyin);
    while (!parsec_util_keyval_parse_done) {
        val = parsec_util_keyval_yylex();
        switch (val) {
        case PARSEC_UTIL_KEYVAL_PARSE_DONE:
            /* This will also set parsec_util_keyval_parse_done to true, so just
               break here */
            break;

        case PARSEC_UTIL_KEYVAL_PARSE_NEWLINE:
            /* blank line!  ignore it */
            break;

        case PARSEC_UTIL_KEYVAL_PARSE_SINGLE_WORD:
            parse_line();
            break;

        default:
            /* anything else is an error */
            parse_error(1);
            ret = val;
        }
    }
    fclose(parsec_util_keyval_yyin);
    parsec_util_keyval_yylex_destroy ();

cleanup:
    pthread_mutex_unlock(&keyval_mutex);
    return ret;
}



static int parse_line(void)
{
    int val;

    /* Save the name name */
    if (key_buffer_len < strlen(parsec_util_keyval_yytext) + 1) {
        char *tmp;
        key_buffer_len = strlen(parsec_util_keyval_yytext) + 1;
        tmp = (char*)realloc(key_buffer, key_buffer_len);
        if (NULL == tmp) {
            free(key_buffer);
            key_buffer_len = 0;
            key_buffer = NULL;
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        key_buffer = tmp;
    }

    strncpy(key_buffer, parsec_util_keyval_yytext, key_buffer_len);

    /* The first thing we have to see is an "=" */

    val = parsec_util_keyval_yylex();
    if (parsec_util_keyval_parse_done || PARSEC_UTIL_KEYVAL_PARSE_EQUAL != val) {
        parse_error(2);
        return PARSEC_ERROR;
    }

    /* Next we get the value */

    val = parsec_util_keyval_yylex();
    if (PARSEC_UTIL_KEYVAL_PARSE_SINGLE_WORD == val ||
        PARSEC_UTIL_KEYVAL_PARSE_VALUE == val) {
        keyval_callback(key_buffer, parsec_util_keyval_yytext);

        /* Now we need to see the newline */

        val = parsec_util_keyval_yylex();
        if (PARSEC_UTIL_KEYVAL_PARSE_NEWLINE == val ||
            PARSEC_UTIL_KEYVAL_PARSE_DONE == val) {
            return PARSEC_SUCCESS;
        }
    }

    /* Did we get an EOL or EOF? */

    else if (PARSEC_UTIL_KEYVAL_PARSE_DONE == val ||
             PARSEC_UTIL_KEYVAL_PARSE_NEWLINE == val) {
        keyval_callback(key_buffer, NULL);
        return PARSEC_SUCCESS;
    }

    /* Nope -- we got something unexpected.  Bonk! */
    parse_error(3);
    return PARSEC_ERROR;
}


static void parse_error(int num)
{
    parsec_output(0, "keyval parser: error %d reading file %s at line %d:\n  %s\n",
                  num, keyval_filename, parsec_util_keyval_yynewlines, parsec_util_keyval_yytext);
}
