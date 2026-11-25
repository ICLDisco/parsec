/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef AUTO_MERGE_OPS_H_HAS_BEEN_INCLUDED
#define AUTO_MERGE_OPS_H_HAS_BEEN_INCLUDED

#include <stdint.h>

/*
 * The taskpools of the auto_merge test apply a sequence of operations to every
 * tile of one matrix of integers. Operation number op replaces every element v
 * by am_op_a(op) * v + am_op_b(op) (modulo 2^32). Two such operations do not
 * commute, so a tile that went through them in any other order than the one
 * the taskpools were given to the context ends up with a different value.
 *
 * Every CPU body also checks, before applying its operation, that the tile has
 * been through exactly the operations that precede it: a violation is a task
 * that saw a tile too early (read before an earlier taskpool wrote it) or too
 * late (written by a later taskpool while it was still reading it).
 */

static inline uint32_t am_op_a(int op) { return 2u * (uint32_t)op + 3u; }
static inline uint32_t am_op_b(int op) { return 7u * (uint32_t)op + 1u; }

/* The value element i of tile (m, n) starts with */
static inline uint32_t am_init_value(int m, int n, int i)
{
    return (uint32_t)m * 1000003u + (uint32_t)n * 7919u + (uint32_t)i;
}

/* The value element i of tile (m, n) has after operations 0 .. nb_ops-1 */
static inline uint32_t am_expected(int m, int n, int i, int nb_ops)
{
    uint32_t v = am_init_value(m, n, i);
    for( int op = 0; op < nb_ops; op++ ) v = am_op_a(op) * v + am_op_b(op);
    return v;
}

/* Number of elements of the tile that do not hold their value after nb_ops
 * operations (stops at the first one: only 0 or 1 matters) */
int  am_check_tile(const int *tile, int count, int m, int n, int nb_ops);

/* Sleep for us microseconds, if positive */
void am_delay(int us);

/*
 * Body of a CPU task that applies operation op to tile (m, n): sleep for delay
 * microseconds, check that the tile went through operations 0 .. op-1, count a
 * violation in *nb_viol if it did not, then apply operation op.
 */
void am_cpu_stage(int *tile, int count, int m, int n, int op, int delay, int32_t *nb_viol);

/*
 * Body of a CPU task that reads tile (m, n): sleep for delay microseconds,
 * then check that the tile went through operations 0 .. nb_ops-1.
 */
void am_cpu_check(const int *tile, int count, int m, int n, int nb_ops, int delay, int32_t *nb_viol);

/* 1 when am_cuda_affine() runs a real kernel, 0 when this build has none */
extern const int am_have_cuda_kernels;

/* Apply v = a * v + b to the count elements of the device tile, after
 * spinning for about spin_us microseconds on the device */
void am_cuda_affine(int *dev_tile, int count, uint32_t a, uint32_t b, int spin_us, void *stream);

#endif  /* AUTO_MERGE_OPS_H_HAS_BEEN_INCLUDED */
