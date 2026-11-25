/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include "auto_merge_ops.h"

#include <stdio.h>
#include <unistd.h>

int am_check_tile(const int *tile, int count, int m, int n, int nb_ops)
{
    for( int i = 0; i < count; i++ ) {
        if( (uint32_t)tile[i] != am_expected(m, n, i, nb_ops) ) return 1;
    }
    return 0;
}

void am_delay(int us)
{
    if( us > 0 ) usleep((useconds_t)us);
}

void am_cpu_stage(int *tile, int count, int m, int n, int op, int delay, int32_t *nb_viol)
{
    am_delay(delay);
    if( am_check_tile(tile, count, m, n, op) ) {
        parsec_atomic_fetch_inc_int32(nb_viol);
    }
    for( int i = 0; i < count; i++ ) {
        tile[i] = (int)(am_op_a(op) * (uint32_t)tile[i] + am_op_b(op));
    }
}

void am_cpu_check(const int *tile, int count, int m, int n, int nb_ops, int delay, int32_t *nb_viol)
{
    am_delay(delay);
    if( am_check_tile(tile, count, m, n, nb_ops) ) {
        parsec_atomic_fetch_inc_int32(nb_viol);
    }
}
