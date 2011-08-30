/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _expr_h
#define _expr_h

typedef struct expr expr_t;
struct dague_object;

#include <stdio.h>

#include "assignment.h"

#define EXPR_OP_BINARY_RANGE          24
#define EXPR_OP_INLINE                100

struct dague_object;
typedef int (*expr_op_inline_func_t)(const struct dague_object *__dague_object_parent, const assignment_t *assignments);

struct expr {
    union {
        struct {
            const struct expr *op1;
            const struct expr *op2;
        } binary;
        expr_op_inline_func_t inline_func;
    } u_expr;
    unsigned char op;
};

#define bop1        u_expr.binary.op1
#define bop2        u_expr.binary.op2
#define inline_func u_expr.inline_func

#endif
