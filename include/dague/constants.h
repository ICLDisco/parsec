/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_CONSTANTS_H_HAS_BEEN_INCLUDED
#define DAGUE_CONSTANTS_H_HAS_BEEN_INCLUDED

#define DAGUE_ERR_BASE  0
#define DAGUE_ERR_MAX   (DAGUE_ERR_BASE - 100)

enum {
    DAGUE_SUCCESS                 = (DAGUE_ERR_BASE -  1),  /* keep going, we're doing good */
    DAGUE_ERROR                   = (DAGUE_ERR_BASE -  2),  /* oops, can't recover */
    DAGUE_ERR_OUT_OF_RESOURCE     = (DAGUE_ERR_BASE -  3),  /* running low on resources */
    DAGUE_ERR_NOT_FOUND           = (DAGUE_ERR_BASE -  4),  /* not found (?) */
    DAGUE_ERR_BAD_PARAM           = (DAGUE_ERR_BASE -  5),  /* bad argument passed down to a function */
    DAGUE_EXISTS                  = (DAGUE_ERR_BASE -  6),  /* file/object exists */
    DAGUE_ERR_NOT_IMPLEMENTED     = (DAGUE_ERR_BASE -  7),  /* functionality not yet supported */
    DAGUE_ERR_VALUE_OUT_OF_BOUNDS = (DAGUE_ERR_BASE  - 8),
};

#endif  /* DAGUE_CONSTANTS_H_HAS_BEEN_INCLUDED */

