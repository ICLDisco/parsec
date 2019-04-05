/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED
#define PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/datatype.h"

BEGIN_C_DECLS

typedef struct parsec_assignment_s parsec_assignment_t;
typedef struct parsec_expr_s parsec_expr_t;
typedef struct parsec_flow_s parsec_flow_t;
typedef struct parsec_dep_s parsec_dep_t;
typedef struct parsec_symbol_s parsec_symbol_t;

struct parsec_taskpool_s;
#include "parsec/datatype.h"

/**
 * Assignments
 */
struct parsec_assignment_s {
    int value;
};

/**
 * Expressions
 */
#define PARSEC_EXPR_OP_RANGE_CST_INCREMENT   24
#define PARSEC_EXPR_OP_RANGE_EXPR_INCREMENT  25
#define PARSEC_EXPR_OP_INLINE                100


typedef parsec_datatype_t  (*parsec_expr_op_datatype_inline_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);
typedef parsec_data_t     *(*parsec_data_lookup_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);
typedef int32_t            (*parsec_expr_op_int32_inline_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);
typedef int64_t            (*parsec_expr_op_int64_inline_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);
typedef float              (*parsec_expr_op_float_inline_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);
typedef double             (*parsec_expr_op_double_inline_func_t)(const struct parsec_taskpool_s *tp, const parsec_assignment_t *assignments);

typedef enum {
    PARSEC_RETURN_TYPE_INT32  = 0,
    PARSEC_RETURN_TYPE_INT64  = 1,
    PARSEC_RETURN_TYPE_FLOAT  = 2,
    PARSEC_RETURN_TYPE_DOUBLE = 3
} parsec_return_type_t;


/**
 * Flows (data or control)
 */
/**< Remark: (sym_type == PARSEC_SYM_INOUT) if (sym_type & PARSEC_SYM_IN) && (sym_type & PARSEC_SYM_OUT) */
#define PARSEC_SYM_IN     ((uint8_t)(1 << 0))
#define PARSEC_SYM_OUT    ((uint8_t)(1 << 1))
#define PARSEC_SYM_INOUT  (PARSEC_SYM_IN | PARSEC_SYM_OUT)

#define PARSEC_FLOW_ACCESS_NONE     ((uint8_t)0x00)
#define PARSEC_FLOW_ACCESS_READ     ((uint8_t)(1 << 2))
#define PARSEC_FLOW_ACCESS_WRITE    ((uint8_t)(1 << 3))
#define PARSEC_FLOW_ACCESS_RW       (PARSEC_FLOW_ACCESS_READ | PARSEC_FLOW_ACCESS_WRITE)
#define PARSEC_FLOW_ACCESS_MASK     (PARSEC_FLOW_ACCESS_READ | PARSEC_FLOW_ACCESS_WRITE)
#define PARSEC_FLOW_HAS_IN_DEPS     ((uint8_t)(1 << 4))

struct parsec_expr_s {
    union {
        struct {
            struct parsec_expr_s const *op1;
            struct parsec_expr_s const *op2;
            union {
                int cst;
                struct parsec_expr_s const *expr;
            } increment;
        } range;
        struct {
            parsec_return_type_t type;
            union {
                parsec_expr_op_int32_inline_func_t  inline_func_int32;
                parsec_expr_op_int64_inline_func_t  inline_func_int64;
                parsec_expr_op_float_inline_func_t  inline_func_float;
                parsec_expr_op_double_inline_func_t inline_func_double;
            } func;
        } v_func;
    } u_expr;
    unsigned char op;
};

struct parsec_flow_s {
    char               *name;
    uint8_t             sym_type;
    uint8_t             flow_flags;
    uint8_t             flow_index; /**< The input index of the flow. This index is used
                                     *   while computing the mask. */
    parsec_dependency_t flow_datatype_mask;  /**< The bitmask of dep_datatype_index of all deps */
    parsec_dep_t const *dep_in[MAX_DEP_IN_COUNT];
    parsec_dep_t const *dep_out[MAX_DEP_OUT_COUNT];
};

/**
 * Dependencies
 */
#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

typedef union parsec_cst_or_fct_32_u {
    int32_t                      cst;
    parsec_expr_op_int32_inline_func_t  fct;
} parsec_cst_or_fct_32_t;

typedef union parsec_cst_or_fct_64_u {
    int64_t                      cst;
    parsec_expr_op_int64_inline_func_t  fct;
} parsec_cst_or_fct_64_t;

typedef union parsec_cst_or_fct_datatype_u {
    parsec_datatype_t               cst;
    parsec_expr_op_datatype_inline_func_t  fct;
} parsec_cst_or_fct_datatype_t;

struct parsec_comm_desc_s {
    parsec_cst_or_fct_32_t         type;
    parsec_cst_or_fct_datatype_t   layout;
    parsec_cst_or_fct_64_t         count;
    parsec_cst_or_fct_64_t         displ;
};

struct parsec_dep_s {
    parsec_expr_t const        *cond;           /**< The runtime-evaluable condition on this dependency */
    parsec_expr_t const        *ctl_gather_nb;  /**< In case of control gather, the runtime-evaluable number of controls to expect */
    uint8_t                    task_class_id;   /**< Index of the target parsec function in the object function array */
    uint8_t                    dep_index;      /**< Output index of the dependency. This is used to store the flow
                                                *   before transfering it to the successors. */
    uint8_t                    dep_datatype_index;  /**< Index of the output datatype. */
    parsec_flow_t const        *flow;           /**< Pointer to the flow pointed to/from this dependency */
    parsec_flow_t const        *belongs_to;     /**< The flow this dependency belongs tp */
    parsec_data_lookup_func_t  direct_data;    /**< Lookup the data associated with this dep, if (and only if)
                                                *   this dep is a direct memory access */
};

/**
 * Parameters
 */

#define PARSEC_SYMBOL_IS_GLOBAL      0x0001     /**> This symbol is a global one. */
#define PARSEC_SYMBOL_IS_STANDALONE  0x0002     /**> standalone symbol, with dependencies only to global symbols */

struct parsec_symbol_s {
    uint32_t               flags;           /*< mask of GLOBAL and STANDALONE */
    char const            *name;            /*< Name, used for debugging purposes */
    int                    context_index;   /*< Location of this symbol's value in the execution_context->locals array */
    parsec_expr_t const   *min;             /*< Expression that represents the minimal value of this symbol */
    parsec_expr_t const   *max;             /*< Expression that represents the maximal value of this symbol */
    parsec_expr_t const   *expr_inc;        /*< Expression that represents the increment of this symbol. NULL if and only if cst_inc is defined */
    int                    cst_inc;         /*< If expr_inc is NULL, represents the integer increment of this symbol. */
};

/**
 * Return 1 if the symbol is global.
 */
static inline int parsec_symbol_is_global( const parsec_symbol_t* symbol )
{
    return (symbol->flags & PARSEC_SYMBOL_IS_GLOBAL ? 1 : 0);
}

END_C_DECLS

#endif  /* PARSEC_DESCRIPTION_STRUCTURES_H_HAS_BEEN_INCLUDED */
