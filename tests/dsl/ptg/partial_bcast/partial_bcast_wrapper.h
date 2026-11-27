/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef PARTIAL_BCAST_WRAPPER_H
#define PARTIAL_BCAST_WRAPPER_H

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <stddef.h>
#include <stdint.h>

/* The cases, one bit each; the MASK global selects which ones run. */
#define PB_CASE_LOWER     1     /* short int8 payload into a lower-triangular tile */
#define PB_CASE_TRANS     2     /* rendezvous int8 payload into a transposed tile */
#define PB_CASE_DBL       4     /* short int8 payload into NB*NB doubles of an 8-byte arena */
#define PB_CASE_RBYTE     8     /* int8 payload into bytes resized to an 8-byte extent */
#define PB_CASE_TILE      16    /* short int8 payload into the tile */
#define PB_CASE_BIGTILE   32    /* rendezvous int8 payload into the tile */
#define PB_CASE_BIGDBL    64    /* rendezvous doubles into NB*NB doubles: a shorter count */
#define PB_CASE_BIGTYPED  128   /* rendezvous doubles into one tile: not a whole element */
#define PB_CASE_TRANS2    256   /* one whole element into two transposed tiles */
#define PB_CASE_PACKED    512   /* two consumers per rank with different receive types */
#define PB_CASE_OVER      1024  /* short payload longer than the declared receive */
#define PB_CASE_ALL       2047

/* Where the bytes of a payload land once received through a layout. */
#define PB_LAYOUT_CONTIG  0
#define PB_LAYOUT_TRANS   1
#define PB_LAYOUT_RBYTE   2
#define PB_LAYOUT_LOWER   3

extern int32_t partial_bcast_failures;

/* The byte rank 0 stores at offset b of the tile every root sends from. */
static inline unsigned char partial_bcast_pattern(size_t b)
{
    return (unsigned char)((b * 131u + 17u) & 0xffu);
}

parsec_taskpool_t *partial_bcast_new(parsec_matrix_block_cyclic_t *A, int nb, int nc,
                                     int nodd, int nbig, int nrb, int mask);

#endif /* PARTIAL_BCAST_WRAPPER_H */
