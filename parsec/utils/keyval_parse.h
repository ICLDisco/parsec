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

/** @file */

#ifndef PARSEC_UTIL_KEYVAL_PARSE_H
#define PARSEC_UTIL_KEYVAL_PARSE_H

BEGIN_C_DECLS

/**
 * Callback triggered for each key = value pair
 *
 * Callback triggered from parsec_util_keyval_parse for each key = value
 * pair.  Both key and value will be pointers into static buffers.
 * The buffers must not be free()ed and contents may be overwritten
 * immediately after the callback returns.
 */
typedef void (*parsec_keyval_parse_fn_t)(const char *key, const char *value);

/**
 * Parse \c filename, made up of key = value pairs.
 *
 * Parse \c filename, made up of key = value pairs.  For each line
 * that appears to contain a key = value pair, \c callback will be
 * called exactly once.  In a multithreaded context, calls to
 * parsec_util_keyval_parse() will serialize multiple calls.
 */
PARSEC_DECLSPEC int parsec_util_keyval_parse(const char *filename, 
                                           parsec_keyval_parse_fn_t callback);

PARSEC_DECLSPEC int parsec_util_keyval_parse_init(void);

PARSEC_DECLSPEC int parsec_util_keyval_parse_finalize(void);

END_C_DECLS

#endif
