/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED
#define DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

typedef struct assignment_s assignment_t;
typedef struct expr_s expr_t;
typedef struct dague_flow_s dague_flow_t;
typedef struct dep_s dep_t;
typedef struct symbol_s symbol_t;

struct dague_handle_s;

BEGIN_C_DECLS

/**
 * Assignments
 */
struct assignment_s {
    int value;
};

/**
 * Expressions
 */
#define EXPR_OP_RANGE_CST_INCREMENT   24
#define EXPR_OP_RANGE_EXPR_INCREMENT  25
#define EXPR_OP_INLINE                100

typedef int32_t (*expr_op_int32_inline_func_t)(const struct dague_handle *__dague_handle_parent, const assignment_t *assignments);
typedef int64_t (*expr_op_int64_inline_func_t)(const struct dague_handle *__dague_handle_parent, const assignment_t *assignments);

struct expr_s {
    union {
        struct {
            const struct expr_s *op1;
            const struct expr_s *op2;
            union {
                int cst;
                const struct expr_s *expr;
            } increment;
        } range;
        expr_op_int32_inline_func_t inline_func_int32;
        expr_op_int64_inline_func_t inline_func_int64;
    } u_expr;
    unsigned char op;
};

#define rop1          u_expr.range.op1
#define rop2          u_expr.range.op2
#define rcstinc       u_expr.range.increment.cst
#define rexprinc      u_expr.range.increment.expr
#define inline_func32 u_expr.inline_func_int32
#define inline_func64 u_expr.inline_func_int64

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

struct dague_flow_s {
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

typedef union dague_cst_or_fct_32_u {
    int32_t                      cst;
    expr_op_int32_inline_func_t  fct;
} dague_cst_or_fct_32_t;

typedef union dague_cst_or_fct_64_u {
    int64_t                      cst;
    expr_op_int64_inline_func_t  fct;
} dague_cst_or_fct_64_t;

struct dague_comm_desc_s {
    dague_cst_or_fct_32_t       type;
    expr_op_int32_inline_func_t layout;
    dague_cst_or_fct_64_t       count;
    dague_cst_or_fct_64_t       displ;
};

struct dep_s {
    const expr_t                *cond;           /**< The runtime-evaluable condition on this dependency */
    const expr_t                *ctl_gather_nb;  /**< In case of control gather, the runtime-evaluable number of controls to expect */
    const int                    function_id;    /**< Index of the target dague function in the object function array */
    const dague_flow_t          *flow;           /**< Pointer to the flow pointed to/from this dependency */
    struct dague_comm_desc_s     datatype;       /**< Datatype associated with this dependency */
    const expr_t                *call_params[MAX_CALL_PARAM_COUNT]; /**< Parameters of the dague function pointed by this dependency */
};

void dep_dump(const dep_t *d, const struct dague_handle_s *dague_handle, const char *prefix);

/**
 * Parameters
 */

#define DAGUE_SYMBOL_IS_GLOBAL      0x0001     /**> This symbol is a global one. */
#define DAGUE_SYMBOL_IS_STANDALONE  0x0002     /**> standalone symbol, with dependencies only to global symbols */

struct symbol_s {
    uint32_t        flags;           /*< mask of GLOBAL and STANDALONE */
    const char     *name;            /*< Name, used for debugging purposes */
    int             context_index;   /*< Location of this symbol's value in the execution_context->locals array */
    const expr_t   *min;             /*< Expression that represents the minimal value of this symbol */
    const expr_t   *max;             /*< Expression that represents the maximal value of this symbol */
    const expr_t   *expr_inc;        /*< Expression that represents the increment of this symbol. NULL if and only if cst_inc is defined */
    int             cst_inc;         /*< If expr_inc is NULL, represents the integer increment of this symbol. */
};

/**
 * Return 1 if the symbol is global.
 */
static inline int dague_symbol_is_global( const symbol_t* symbol )
{
    return (symbol->flags & DAGUE_SYMBOL_IS_GLOBAL ? 1 : 0);
}

END_C_DECLS

#endif  /* DAGUE_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED */
