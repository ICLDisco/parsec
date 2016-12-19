/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_CONSTANTS_H_HAS_BEEN_INCLUDED
#define PARSEC_CONSTANTS_H_HAS_BEEN_INCLUDED

#define PARSEC_ERR_BASE  0
#define PARSEC_ERR_MAX   (PARSEC_ERR_BASE - 100)

enum {
    PARSEC_SUCCESS                 = (PARSEC_ERR_BASE -  1),  /* keep going, we're doing good */
    PARSEC_ERROR                   = (PARSEC_ERR_BASE -  2),  /* oops, can't recover */
    PARSEC_ERR_OUT_OF_RESOURCE     = (PARSEC_ERR_BASE -  3),  /* running low on resources */
    PARSEC_ERR_NOT_FOUND           = (PARSEC_ERR_BASE -  4),  /* not found (?) */
    PARSEC_ERR_BAD_PARAM           = (PARSEC_ERR_BASE -  5),  /* bad argument passed down to a function */
    PARSEC_EXISTS                  = (PARSEC_ERR_BASE -  6),  /* file/object exists */
    PARSEC_ERR_NOT_IMPLEMENTED     = (PARSEC_ERR_BASE -  7),  /* functionality not yet supported */
    PARSEC_NOT_SUPPORTED           = (PARSEC_ERR_BASE -  8),  /* concept not supported */
    PARSEC_ERR_VALUE_OUT_OF_BOUNDS = (PARSEC_ERR_BASE  - 9),
};

#endif  /* PARSEC_CONSTANTS_H_HAS_BEEN_INCLUDED */

