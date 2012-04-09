/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED
#define DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

typedef struct assignment assignment_t;
typedef struct expr expr_t;
typedef struct dague_flow dague_flow_t;
typedef struct dep dep_t;
typedef struct symbol symbol_t;
typedef struct dague_datatype dague_datatype_t;

struct dague_object;
struct dague_function;

/**
 * Assignments
 */
struct assignment {
    int value;
};

/**
 * Expressions
 */
#define EXPR_OP_BINARY_RANGE          24
#define EXPR_OP_INLINE                100

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

/**
 * Flows (data or control)
 */
/**< Remark: (sym_type == SYM_INOUT) if (sym_type & SYM_IN) && (sym_type & SYM_OUT) */
#define SYM_IN     0x01
#define SYM_OUT    0x02
#define SYM_INOUT  (SYM_IN | SYM_OUT)

#define ACCESS_NONE     0x00
#define ACCESS_READ     0x01
#define ACCESS_WRITE    0x02
#define ACCESS_RW       (ACCESS_READ | ACCESS_WRITE)

struct dague_flow {
    char               *name;
    unsigned char       sym_type;
    unsigned char       access_type;
    dague_dependency_t  flow_index;
    const dep_t        *dep_in[MAX_DEP_IN_COUNT];
    const dep_t        *dep_out[MAX_DEP_OUT_COUNT];
};

/**
 * Dependencies
 */
#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

struct dague_datatype {
    int index;
    expr_op_inline_func_t index_fct;
    int nb_elt;
    expr_op_inline_func_t nb_elt_fct;
};

struct dep {
    const expr_t                *cond;
    const struct dague_function *dague;
    const expr_t                *call_params[MAX_CALL_PARAM_COUNT];
    const dague_flow_t          *flow;
    dague_datatype_t             datatype;
};

void dep_dump(const dep_t *d, const struct dague_object *dague_object, const char *prefix);

/**
 * Parameters
 */

#define DAGUE_SYMBOL_IS_GLOBAL      0x0001     /**> This symbol is a global one. */
#define DAGUE_SYMBOL_IS_STANDALONE  0x0002     /**> standalone symbol, with dependencies only to global symbols */

struct symbol {
    uint32_t        flags;
    const char     *name;
    const char     *type;
    const expr_t   *min;
    const expr_t   *max;
};

/**
 * Return 1 if the symbol is global.
 */
static inline int dague_symbol_is_global( const symbol_t* symbol )
{
    return (symbol->flags & DAGUE_SYMBOL_IS_GLOBAL ? 1 : 0);
}

#endif  /* DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED */
