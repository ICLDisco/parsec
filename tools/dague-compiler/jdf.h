#ifndef jdf_h
#define jdf_h

#include <stdint.h>

/**
 * This file holds all data structures to parse the JDF file format.
 * It should be independent from the internal representation of the JDF,
 * although it mechanically mimics some of its structures.
 *
 * Nothing of this strucutre holds atomic things, or memory efficient representations,
 * since we don't expect to do High Performance Parsing.
 */

void jdf_prepare_parsing(void);
void jdf_warn(int lineno, const char *format, ...);

/**
 * Checks the sanity of the current_jdf.
 *
 * @param [IN] mask defines what level of warnings must be raised
 *
 * @return -1 if a fatal error was encountered
 * @return 0 if no warning was signaled (except for warnings during parsing)
 * @return >0 the number of warnings signaled if a non fatal but probable 
 *            error was encountered
 */
typedef uint64_t jdf_warning_mask_t;
#define JDF_WARN_MASKED_GLOBALS        ((jdf_warning_mask_t)(1 <<  0))
#define JDF_ALL_WARNINGS               ((jdf_warning_mask_t)(0xffffffffffffffff))
int jdf_sanity_checks( jdf_warning_mask_t mask );

/**
 * Toplevel structure: three linked lists: preambles, globals and functions 
 */
typedef struct jdf {
    struct jdf_preamble_entry *preambles;
    struct jdf_global_entry   *globals;
    struct jdf_function_entry *functions;
} jdf_t;

extern jdf_t current_jdf;

/** A Preamble is a c-code that is dumped as-is with a #line directive 
 *  We remember the line number in the JDF file where this preamble was found
 */
typedef struct jdf_preamble_entry {
    struct jdf_preamble_entry *next;
    char                      *preamble;
    int                        lineno;
} jdf_preamble_entry_t;

/** A global is a variable name, optionally an expression to define it,
 *  and a line number associated with it for error printing purposes
 */
typedef struct jdf_global_entry {
    struct jdf_global_entry *next;
    char                    *name;
    struct jdf_expr         *expression;
    int                      lineno;
} jdf_global_entry_t;

/** A JDF function is the complex object described below
 *  It uses a jdf_flags_t type for its flags
 */

typedef unsigned int jdf_flags_t;
#define JDF_FUNCTION_FLAG_HIGH_PRIORITY ((jdf_flags_t)(1 << 0))

typedef struct jdf_function_entry {
    struct jdf_function_entry *next;

    char                      *fname;
    struct jdf_name_list      *parameters;
    jdf_flags_t                flags;
    struct jdf_def_list       *definitions;
    struct jdf_expr_list      *predicates;
    struct jdf_dataflow_list  *dataflow;
    struct jdf_expr           *priority;
    char                      *body;
    int                        lineno;
} jdf_function_entry_t;

/*******************************************************************/
/*          Internal structures of the jdf_function                */
/*******************************************************************/

typedef struct jdf_name_list {
    struct jdf_name_list *next;
    char *name;
} jdf_name_list_t;

typedef struct jdf_def_list {
    struct jdf_def_list *next;
    char                *name;
    struct jdf_expr     *expr;
    int                  lineno;
} jdf_def_list_t;

typedef struct jdf_dataflow_list {
    struct jdf_dataflow_list *next;
    struct jdf_dataflow      *flow;
} jdf_dataflow_list_t;

typedef unsigned int jdf_access_type_t;
#define JDF_VAR_TYPE_READ  ((jdf_access_type_t)(1<<0))
#define JDF_VAR_TYPE_WRITE ((jdf_access_type_t)(1<<1))
typedef struct jdf_dataflow {
    char                     *varname;
    struct jdf_dep_list      *deps;
    jdf_access_type_t         access_type;
    int                       lineno;
} jdf_dataflow_t;

typedef struct jdf_dep_list {
    struct jdf_dep_list *next;
    struct jdf_dep      *dep;
} jdf_dep_list_t;

typedef unsigned int jdf_dep_type_t;
#define JDF_DEP_TYPE_IN  ((jdf_dep_type_t)(1<<0))
#define JDF_DEP_TYPE_OUT ((jdf_dep_type_t)(1<<1))

typedef struct jdf_dep {
    jdf_dep_type_t           type;
    struct jdf_guarded_call *guard;
    char                    *datatype;
    int                      lineno;
} jdf_dep_t;

typedef enum { JDF_GUARD_UNCONDITIONAL,
               JDF_GUARD_BINARY,
               JDF_GUARD_TERNARY } jdf_guard_type_t;

typedef struct jdf_guarded_call {
    jdf_guard_type_t         guard_type;
    struct jdf_expr          *guard;
    struct jdf_call          *calltrue;
    struct jdf_call          *callfalse;
} jdf_guarded_call_t;

typedef struct jdf_call {
    char                     *var;
    char                     *func_or_mem;
    struct jdf_expr_list     *parameters;
} jdf_call_t;

/*******************************************************************/
/*             Expressions (and list of expressions)              */
/*******************************************************************/

typedef struct jdf_expr_list {
    struct jdf_expr_list *next;
    struct jdf_expr      *expr;
} jdf_expr_list_t;

typedef enum { JDF_EQUAL,
               JDF_NOTEQUAL,
               JDF_AND,
               JDF_OR,
               JDF_XOR,
               JDF_LESS,
               JDF_LEQ,
               JDF_MORE,
               JDF_MEQ,
               JDF_NOT,
               JDF_PLUS,
               JDF_MINUS,
               JDF_TIMES,
               JDF_DIV,
               JDF_MODULO,
               JDF_SHL,
               JDF_SHR,
               JDF_RANGE,
               JDF_TERNARY,
               JDF_VAR,
               JDF_CST
} jdf_expr_operand_t;

typedef struct jdf_expr {
    jdf_expr_operand_t op;
    union {
        struct {
            struct jdf_expr *test;
            struct jdf_expr *arg1;
            struct jdf_expr *arg2;
        } ternary;
        struct {
            struct jdf_expr *arg1;
            struct jdf_expr *arg2;
        } binary;
        struct {
            struct jdf_expr *arg;
        } unary;
        char *varname;
        int   cstval;
    } u;
} jdf_expr_t;

#define jdf_ua  u.unary.arg
#define jdf_ba1 u.binary.arg1
#define jdf_ba2 u.binary.arg2
#define jdf_tat u.ternary.test
#define jdf_ta1 u.ternary.arg1
#define jdf_ta2 u.ternary.arg2
#define jdf_var u.varname
#define jdf_cst u.cstval

#endif
