/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED
#define PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED

typedef struct assignment_s assignment_t;
typedef struct expr_s expr_t;
typedef struct parsec_flow_s parsec_flow_t;
typedef struct dep_s dep_t;
typedef struct symbol_s symbol_t;

struct parsec_handle_s;
#include "parsec/datatype.h"

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

typedef parsec_datatype_t (*expr_op_datatype_inline_func_t)(const struct parsec_handle_s *__parsec_handle_parent, const assignment_t *assignments);
typedef int32_t (*expr_op_int32_inline_func_t)(const struct parsec_handle_s *__parsec_handle_parent, const assignment_t *assignments);
typedef int64_t (*expr_op_int64_inline_func_t)(const struct parsec_handle_s *__parsec_handle_parent, const assignment_t *assignments);
typedef parsec_data_t *(*direct_data_lookup_func_t)(const struct parsec_handle_s *__parsec_handle_parent, const assignment_t *assignments);

struct expr_s {
    union {
        struct {
            struct expr_s const *op1;
            struct expr_s const *op2;
            union {
                int cst;
                struct expr_s const *expr;
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
#define SYM_IN     ((uint8_t)(1 << 0))
#define SYM_OUT    ((uint8_t)(1 << 1))
#define SYM_INOUT  (SYM_IN | SYM_OUT)

#define FLOW_ACCESS_NONE     ((uint8_t)0x00)
#define FLOW_ACCESS_READ     ((uint8_t)(1 << 2))
#define FLOW_ACCESS_WRITE    ((uint8_t)(1 << 3))
#define FLOW_ACCESS_RW       (FLOW_ACCESS_READ | FLOW_ACCESS_WRITE)
#define FLOW_ACCESS_MASK     (FLOW_ACCESS_READ | FLOW_ACCESS_WRITE)
#define FLOW_HAS_IN_DEPS     ((uint8_t)(1 << 4))

struct parsec_flow_s {
    char               *name;
    uint8_t             sym_type;
    uint8_t             flow_flags;
    uint8_t             flow_index; /**< The input index of the flow. This index is used
                                     *   while computing the mask. */
    parsec_dependency_t  flow_datatype_mask;  /**< The bitmask of dep_datatype_index of all deps */
    dep_t const        *dep_in[MAX_DEP_IN_COUNT];
    dep_t const        *dep_out[MAX_DEP_OUT_COUNT];
};

/**
 * Dependencies
 */
#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

typedef union parsec_cst_or_fct_32_u {
    int32_t                      cst;
    expr_op_int32_inline_func_t  fct;
} parsec_cst_or_fct_32_t;

typedef union parsec_cst_or_fct_64_u {
    int64_t                      cst;
    expr_op_int64_inline_func_t  fct;
} parsec_cst_or_fct_64_t;

typedef union parsec_cst_or_fct_datatype_u {
    parsec_datatype_t                cst;
    expr_op_datatype_inline_func_t  fct;
} parsec_cst_or_fct_datatype_t;

struct parsec_comm_desc_s {
    parsec_cst_or_fct_32_t         type;
    parsec_cst_or_fct_datatype_t   layout;
    parsec_cst_or_fct_64_t         count;
    parsec_cst_or_fct_64_t         displ;
};

struct dep_s {
    expr_t const               *cond;           /**< The runtime-evaluable condition on this dependency */
    expr_t const               *ctl_gather_nb;  /**< In case of control gather, the runtime-evaluable number of controls to expect */
    uint8_t                    function_id;     /**< Index of the target parsec function in the object function array */
    uint8_t                    dep_index;      /**< Output index of the dependency. This is used to store the flow
                                                *   before tranfering it to the successors. */
    uint8_t                    dep_datatype_index;  /**< Index of the output datatype. */
    parsec_flow_t const        *flow;           /**< Pointer to the flow pointed to/from this dependency */
    parsec_flow_t const        *belongs_to;     /**< The flow this dependency belongs tp */
    direct_data_lookup_func_t  direct_data;    /**< Lookup the data associated with this dep, if (and only if)
                                                *   this dep is a direct memory access */
};

void dep_dump(const dep_t *d, const struct parsec_handle_s *parsec_handle, const char *prefix);

/**
 * Parameters
 */

#define PARSEC_SYMBOL_IS_GLOBAL      0x0001     /**> This symbol is a global one. */
#define PARSEC_SYMBOL_IS_STANDALONE  0x0002     /**> standalone symbol, with dependencies only to global symbols */

struct symbol_s {
    uint32_t        flags;           /*< mask of GLOBAL and STANDALONE */
    char const     *name;            /*< Name, used for debugging purposes */
    int             context_index;   /*< Location of this symbol's value in the execution_context->locals array */
    expr_t const   *min;             /*< Expression that represents the minimal value of this symbol */
    expr_t const   *max;             /*< Expression that represents the maximal value of this symbol */
    expr_t const   *expr_inc;        /*< Expression that represents the increment of this symbol. NULL if and only if cst_inc is defined */
    int             cst_inc;         /*< If expr_inc is NULL, represents the integer increment of this symbol. */
};

/**
 * Return 1 if the symbol is global.
 */
static inline int parsec_symbol_is_global( const symbol_t* symbol )
{
    return (symbol->flags & PARSEC_SYMBOL_IS_GLOBAL ? 1 : 0);
}

END_C_DECLS

#endif  /* PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED */
