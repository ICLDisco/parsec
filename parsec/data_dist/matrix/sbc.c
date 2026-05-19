/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/**
 * This data distribution is based on the one described in
 * "Symmetric Block-Cyclic Distribution: Fewer Communications
 * Leads to Faster Dense Cholesky Factorization".
 * Thus, the naming stands for "Symmetric Block-Cyclic".
 *
 * The distribution is defined on a repeated r x r pattern.  Every
 * off-diagonal pattern position (a, b) is paired with its symmetric position
 * (b, a), and both positions are owned by the same rank.  Ranks for these
 * pairs are numbered in packed upper-triangular order:
 *
 *   rank({a,b}) = max(a,b) * (max(a,b) - 1) / 2 + min(a,b)
 *
 * The diagonal positions are the only irregular part of the pattern.  This
 * implementation supports both variants described in the paper:
 *   - extended SBC, with r * (r - 1) / 2 ranks;
 *   - basic SBC, with r * r / 2 ranks, valid only for even r.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>

#include "parsec/data_dist/matrix/sbc.h"
#include "parsec/mca/device/device.h"
#include "parsec/vpmap.h"

#if !defined(UINT_MAX)
#define UINT_MAX (~0UL)
#endif

static uint32_t
parsec_matrix_sbc_rank_of_global(const parsec_matrix_sbc_t *dc, int m, int n)
{
    int r = dc->r;
    int a, b, min, max;

    if( ((dc->uplo == PARSEC_MATRIX_LOWER) && (m < n)) ||
        ((dc->uplo == PARSEC_MATRIX_UPPER) && (n < m)) ) {
        return UINT_MAX;
    }

    a = m % r;
    b = n % r;

    /*
     * Off-diagonal pattern positions are symmetric pairs: (a,b) and (b,a)
     * have the same owner.  The owner is the packed index of that unordered
     * pair among the r * (r - 1) / 2 off-diagonal ranks.
     */
    if( a != b ) {
        min = a < b ? a : b;
        max = a < b ? b : a;
        return (uint32_t)((max * (max - 1)) / 2 + min);
    }

    /*
     * Basic SBC adds r/2 extra ranks for the diagonal and assigns two
     * diagonal pattern positions to each of them in round-robin order.
     */
    if( !dc->extended ) {
        return (uint32_t)((r * (r - 1)) / 2 + (a % (r / 2)));
    }

    /*
     * Extended SBC uses only the off-diagonal ranks on the diagonal.  Diagonal
     * patterns rotate column-wise over blocks of r tile columns.  Odd r uses
     * (r - 1) / 2 patterns.  Even r uses r - 1 patterns: the first r/2 - 1 are
     * the odd-r construction, and the rest are built by shifting the left and
     * right packs around the bonus pack described in the paper.
     */
    {
        int d = a;
        int pattern = (n / r) % dc->diag_patterns;
        int l;

        if( r % 2 ) {
            l = pattern + 1;
            if( d < (r - l) ) {
                a = d;
                b = d + l;
            } else {
                a = d + l - r;
                b = d;
            }
        } else {
            int half = r / 2;
            int normal_patterns = half - 1;

            if( pattern < normal_patterns ) {
                l = pattern + 1;
                if( d < (r - l) ) {
                    a = d;
                    b = d + l;
                } else {
                    a = d + l - r;
                    b = d;
                }
            } else {
                int shifted = pattern - normal_patterns;

                if( d < half ) {
                    a = d;
                    b = (shifted == 0) ? d + half : d + shifted;
                } else if( shifted == normal_patterns ) {
                    a = d - half;
                    b = d;
                } else {
                    l = shifted + 1;
                    if( d < (r - l) ) {
                        a = d;
                        b = d + l;
                    } else {
                        a = d + l - r;
                        b = d;
                    }
                }
            }
        }
    }

    assert(a != b);
    min = a < b ? a : b;
    max = a < b ? b : a;
    return (uint32_t)((max * (max - 1)) / 2 + min);
}

static int
parsec_matrix_sbc_coordinates_to_position(const parsec_matrix_sbc_t *dc, int m, int n)
{
    int col, row;
    int position = 0;

    assert(((dc->uplo == PARSEC_MATRIX_LOWER) && (m >= n)) ||
           ((dc->uplo == PARSEC_MATRIX_UPPER) && (n >= m)));
    assert((int)dc->super.super.myrank == (int)parsec_matrix_sbc_rank_of_global(dc, m, n));

    /* The rank pattern is intentionally irregular on the diagonal, so keep the
     * first implementation simple: local memory is packed in triangular
     * column-major order over the full tiled matrix.
     */
    for(col = 0; col < dc->super.lnt; col++) {
        int first_row = (dc->uplo == PARSEC_MATRIX_UPPER) ? 0 : col;
        int last_row  = (dc->uplo == PARSEC_MATRIX_UPPER) ? col : dc->super.lmt - 1;

        if( first_row >= dc->super.lmt ) {
            continue;
        }
        if( last_row >= dc->super.lmt ) {
            last_row = dc->super.lmt - 1;
        }

        for(row = first_row; row <= last_row; row++) {
            if( (row == m) && (col == n) ) {
                return position;
            }
            if( (int)parsec_matrix_sbc_rank_of_global(dc, row, col) == (int)dc->super.super.myrank ) {
                position++;
            }
        }
    }

    assert(0);
    return -1;
}

static int
parsec_matrix_sbc_count_local_tiles(const parsec_matrix_sbc_t *dc)
{
    int col, row;
    int total = 0;

    for(col = 0; col < dc->super.lnt; col++) {
        int first_row = (dc->uplo == PARSEC_MATRIX_UPPER) ? 0 : col;
        int last_row  = (dc->uplo == PARSEC_MATRIX_UPPER) ? col : dc->super.lmt - 1;

        if( first_row >= dc->super.lmt ) {
            continue;
        }
        if( last_row >= dc->super.lmt ) {
            last_row = dc->super.lmt - 1;
        }

        for(row = first_row; row <= last_row; row++) {
            if( (int)parsec_matrix_sbc_rank_of_global(dc, row, col) == (int)dc->super.super.myrank ) {
                total++;
            }
        }
    }

    return total;
}

static int parsec_matrix_sbc_memory_register(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_sbc_t *sbc = (parsec_matrix_sbc_t *)desc;
    if( (NULL == sbc->mat) || (0 == sbc->super.nb_local_tiles) ) {
        return PARSEC_SUCCESS;
    }
    return device->memory_register(device, desc,
                                   sbc->mat,
                                   ((size_t)sbc->super.nb_local_tiles * (size_t)sbc->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(sbc->super.mtype)));
}

static int parsec_matrix_sbc_memory_unregister(parsec_data_collection_t* desc, parsec_device_module_t* device)
{
    parsec_matrix_sbc_t *sbc = (parsec_matrix_sbc_t *)desc;
    if( (NULL == sbc->mat) || (0 == sbc->super.nb_local_tiles) ) {
        return PARSEC_SUCCESS;
    }
    return device->memory_unregister(device, desc, sbc->mat);
}

static uint32_t parsec_matrix_sbc_rank_of(parsec_data_collection_t * desc, ...)
{
    int m, n;
    va_list ap;
    parsec_matrix_sbc_t *dc = (parsec_matrix_sbc_t *)desc;

    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    assert(m < dc->super.mt);
    assert(n < dc->super.nt);

    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert(m < dc->super.lmt);
    assert(n < dc->super.lnt);

    return parsec_matrix_sbc_rank_of_global(dc, m, n);
}

static uint32_t parsec_matrix_sbc_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_tiled_matrix_t *dc = (parsec_tiled_matrix_t *)desc;

    m = (key % dc->lmt) - dc->i / dc->mb;
    n = (key / dc->lmt) - dc->j / dc->nb;
    return parsec_matrix_sbc_rank_of(desc, m, n);
}

static parsec_data_t* parsec_matrix_sbc_data_of(parsec_data_collection_t *desc, ...)
{
    int m, n, position;
    size_t pos = 0;
    va_list ap;
    parsec_matrix_sbc_t *dc = (parsec_matrix_sbc_t *)desc;

    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    assert(m < dc->super.mt);
    assert(n < dc->super.nt);

    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert(((dc->uplo == PARSEC_MATRIX_LOWER) && (m >= n)) ||
           ((dc->uplo == PARSEC_MATRIX_UPPER) && (n >= m)));

    position = parsec_matrix_sbc_coordinates_to_position(dc, m, n);

    if( NULL != dc->mat ) {
        pos = (size_t)position * dc->super.bsiz;
    }

    return parsec_tiled_matrix_create_data(&dc->super,
                                           (char*)dc->mat + pos * parsec_datadist_getsizeoftype(dc->super.mtype),
                                           position, (n * dc->super.lmt) + m);
}

static parsec_data_t* parsec_matrix_sbc_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_tiled_matrix_t *dc = (parsec_tiled_matrix_t *)desc;

    m = (key % dc->lmt) - dc->i / dc->mb;
    n = (key / dc->lmt) - dc->j / dc->nb;
    return parsec_matrix_sbc_data_of(desc, m, n);
}

static int32_t parsec_matrix_sbc_vpid_of(parsec_data_collection_t *desc, ...)
{
    int m, n, position;
    int nbvp = vpmap_get_nb_vp();
    va_list ap;
    parsec_matrix_sbc_t *dc = (parsec_matrix_sbc_t *)desc;

    if( nbvp == 1 ) {
        return 0;
    }

    va_start(ap, desc);
    m = (int)va_arg(ap, unsigned int);
    n = (int)va_arg(ap, unsigned int);
    va_end(ap);

    assert(m < dc->super.mt);
    assert(n < dc->super.nt);

    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    position = parsec_matrix_sbc_coordinates_to_position(dc, m, n);
    return position % nbvp;
}

static int32_t parsec_matrix_sbc_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_tiled_matrix_t *dc = (parsec_tiled_matrix_t *)desc;

    m = (key % dc->lmt) - dc->i / dc->mb;
    n = (key / dc->lmt) - dc->j / dc->nb;
    return parsec_matrix_sbc_vpid_of(desc, m, n);
}

int parsec_matrix_sbc_init( parsec_matrix_sbc_t * dc,
                            parsec_matrix_type_t mtype,
                            int myrank,
                            int mb, int nb, int lm, int ln,
                            int i, int j, int m, int n,
                            int nodes, int r,
                            parsec_matrix_uplo_t uplo )
{
    int64_t extended_nodes;
    int64_t basic_nodes;
    parsec_data_collection_t *o = &(dc->super.super);

    if( r < 2 ) {
        parsec_warning("SBC Distribution:\tinvalid pattern size r=%d", r);
        return PARSEC_ERR_BAD_PARAM;
    }
    if( r > UINT16_MAX ) {
        parsec_warning("SBC Distribution:\tpattern size r=%d exceeds %u", r,
                       (unsigned int)UINT16_MAX);
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (uplo != PARSEC_MATRIX_LOWER) && (uplo != PARSEC_MATRIX_UPPER) ) {
        parsec_warning("SBC Distribution:\tonly upper/lower triangular storage is supported");
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (myrank < 0) || (myrank >= nodes) ) {
        parsec_warning("SBC Distribution:\tinvalid rank %d for %d nodes", myrank, nodes);
        return PARSEC_ERR_BAD_PARAM;
    }

    extended_nodes = ((int64_t)r * (int64_t)(r - 1)) / 2;
    basic_nodes = ((int64_t)r * (int64_t)r) / 2;

    if( nodes == extended_nodes ) {
        dc->extended = (uint8_t)1;
        dc->diag_patterns = (uint16_t)(r - 1);
        if( r % 2 ) {
            dc->diag_patterns = (uint16_t)((r - 1) / 2);
        }
    } else if( ((r % 2) == 0) && (nodes == basic_nodes) ) {
        dc->extended = (uint8_t)0;
        dc->diag_patterns = (uint16_t)1;
    } else {
        parsec_warning("SBC Distribution:\tnodes=%d is incompatible with r=%d", nodes, r);
        return PARSEC_ERR_BAD_PARAM;
    }

    parsec_tiled_matrix_init(&(dc->super), mtype, PARSEC_MATRIX_TILE,
                             parsec_matrix_sbc_type, nodes, myrank,
                             mb, nb, lm, ln, i, j, m, n);

    dc->mat = NULL;
    dc->uplo = uplo;
    dc->r = (uint16_t)r;

    o->rank_of     = parsec_matrix_sbc_rank_of;
    o->rank_of_key = parsec_matrix_sbc_rank_of_key;
    o->vpid_of     = parsec_matrix_sbc_vpid_of;
    o->vpid_of_key = parsec_matrix_sbc_vpid_of_key;
    o->data_of     = parsec_matrix_sbc_data_of;
    o->data_of_key = parsec_matrix_sbc_data_of_key;

    o->register_memory   = parsec_matrix_sbc_memory_register;
    o->unregister_memory = parsec_matrix_sbc_memory_unregister;

    dc->super.nb_local_tiles = parsec_matrix_sbc_count_local_tiles(dc);
    dc->super.data_map = (parsec_data_t**)calloc(dc->super.nb_local_tiles, sizeof(parsec_data_t*));
    dc->super.slm = dc->super.llm = dc->super.lm;
    dc->super.sln = dc->super.lln = dc->super.ln;

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "parsec_matrix_sbc_init: \n"
           "      dc = %p, mtype = %d, nodes = %u, myrank = %d, \n"
           "      mb = %d, nb = %d, lm = %d, ln = %d, i = %d, j = %d, m = %d, n = %d, \n"
           "      mt = %d, nt = %d, lmt = %d, lnt = %d, nb_local_tile = %d, \n"
           "      r = %d, variant = %s, diagonal patterns = %d",
           dc, dc->super.mtype, dc->super.super.nodes,
           dc->super.super.myrank,
           dc->super.mb, dc->super.nb,
           dc->super.lm, dc->super.ln,
           dc->super.i, dc->super.j,
           dc->super.m, dc->super.n,
           dc->super.mt, dc->super.nt,
           dc->super.lmt, dc->super.lnt,
           dc->super.nb_local_tiles,
           dc->r, dc->extended ? "extended" : "basic", dc->diag_patterns);

    return PARSEC_SUCCESS;
}
