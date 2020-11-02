/*
 * Copyright (c) 2020      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef WIN_COMPAT_H_HAS_BEEN_INCLUDED
#define WIN_COMPAT_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include <stdio.h>

#if !defined(PARSEC_HAVE_GETLINE)
ssize_t getdelim(char **buf, size_t *bufsiz, int delimiter, FILE *fp);
ssize_t getline(char **buf, size_t *bufsiz, FILE *fp);
#endif  /* !defined(PARSEC_HAVE_GETLINE) */

#endif  /* WIN_COMPAT_H_HAS_BEEN_INCLUDED */
