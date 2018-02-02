/**
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef jdf_h
#define jdf_h

#include "parsec/parsec_config.h"

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
void jdf_fatal(int lineno, const char *format, ...);

typedef struct jdf_object_t {
    uint32_t    refcount;
    int         lineno;    /**< line number in the JDF file where the object has been defined */
    char       *filename;  /**< the name of the JDF file source of this object */
    char       *comment;   /**< Additional comments to be dumped through the code generation process */
    char       *oname;     /* name of the object for simplified reference */
} jdf_object_t;

/**
 * Macros to handle the basic information for the jdf_object_t.
 */
#define JDF_OBJECT_RETAIN(OBJ) ((((struct jdf_object_t*)(OBJ))->refcount)++)
#define JDF_OBJECT_RELEASE(OBJ)                                         \
    do {                                                                \
        if( --(((struct jdf_object_t*)(OBJ))->refcount) == 0 ) {        \
            free((OBJ));                                                \
        }                                                               \
    } while(0)
#define JDF_OBJECT_SET( OBJ, FILENAME, LINENO, COMMENT )                \
    do {                                                                \
        ((jdf_object_t*)(OBJ))->refcount = 1;  /* no copy here */       \
        ((jdf_object_t*)(OBJ))->filename = (FILENAME);  /* no copy here */ \
        ((jdf_object_t*)(OBJ))->lineno   = (LINENO);                    \
        ((jdf_object_t*)(OBJ))->comment  = (COMMENT);                   \
        ((jdf_object_t*)(OBJ))->oname    = NULL;                        \
    } while (0)
#define JDF_OBJECT_LINENO( OBJ )   ((OBJ)->super.lineno)
#define JDF_OBJECT_FILENAME( OBJ ) ((OBJ)->super.filename)
#define JDF_OBJECT_COMMENT( OBJ )  ((OBJ)->super.comment)
#define JDF_OBJECT_ONAME( OBJ )    (OBJ)->super.oname

/**
 * Internam name marker for the arena allocation of the WRITE-only
 * dependencies. This name is internally associated with the corresponding
 * variable, and can be safely used as a marker.
 */
#define PARSEC_WRITE_MAGIC_NAME "__parsec_write_type"
#define PARSEC_NULL_MAGIC_NAME "__parsec_null_type"

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
#define JDF_WARN_MASKED_GLOBALS          ((jdf_warning_mask_t)(1 <<  0))
#define JDF_WARN_MUTUAL_EXCLUSIVE_INPUTS ((jdf_warning_mask_t)(1 <<  1))
#define JDF_WARN_REMOTE_MEM_REFERENCE    ((jdf_warning_mask_t)(1 <<  2))

#define JDF_WARNINGS_ARE_ERROR           (jdf_warning_mask_t)(1 <<  3)

#define JDF_WARNINGS_DISABLED_BY_DEFAULT (JDF_WARNINGS_ARE_ERROR)
#define JDF_ALL_WARNINGS                 ((jdf_warning_mask_t)~JDF_WARNINGS_DISABLED_BY_DEFAULT)
int jdf_sanity_checks( jdf_warning_mask_t mask );

#define DEP_MANAGEMENT_DYNAMIC_HASH_TABLE_STRING "dynamic-hash-table"
#define DEP_MANAGEMENT_DYNAMIC_HASH_TABLE 1
#define DEP_MANAGEMENT_INDEX_ARRAY_STRING        "index-array"
#define DEP_MANAGEMENT_INDEX_ARRAY        2

typedef struct jdf_compiler_global_args {
    char *input;
    char *output_c;
    char *output_h;
    char *output_o;
    char *funcid;
    jdf_warning_mask_t wmask;
    int   compile;  /**< Should we generate the .[cho] files directly or should we
                     *   limit the generation to the .[ch] files */
    int   dep_management;
    int   noline;  /**< Don't dump the jdf line number in the generate .c file */
    struct jdf_name_list *ignore_properties; /**< Properties to ignore */
} jdf_compiler_global_args_t;
extern jdf_compiler_global_args_t JDF_COMPILER_GLOBAL_ARGS;

/**
 * Toplevel structure: four linked lists: prologues, epilogues, globals and functions
 */
typedef struct jdf {
    struct jdf_object_t        super;
    struct jdf_external_entry *prologue;
    struct jdf_external_entry *epilogue;
    struct jdf_global_entry   *globals;
    struct jdf_def_list       *global_properties;
    struct jdf_function_entry *functions;
    struct jdf_data_entry     *data;
    struct jdf_name_list      *datatypes;
    struct jdf_expr           *inline_c_functions;
    const char                *nb_local_tasks_fn_name;
} jdf_t;

/**
 * dumps a jdf_t structure into a file. The output should be parsable by the current grammar.
 *
 * @param [IN]    jdf: the jdf to output
 * @param [INOUT] out: the file to output to
 */
int jdf_unparse( const jdf_t *jdf, FILE *out );

extern jdf_t current_jdf;

/** A prologue/epilogue is a c-code that is dumped as-is with a sharp-line directive
 *  We remember the line number in the JDF file where this external code was found
 */
typedef struct jdf_external_entry {
    struct jdf_object_t       super;
    char                      *external_code;
} jdf_external_entry_t;

/** A global is a variable name, optionally an expression to define it,
 *  and a line number associated with it for error printing purposes
 */
typedef struct jdf_global_entry {
    struct jdf_object_t      super;
    struct jdf_global_entry *next;
    char                    *name;
    struct jdf_def_list     *properties;
    struct jdf_expr         *expression;
    struct jdf_data_entry   *data;
} jdf_global_entry_t;

/**
 * The definition of a BODY.
 */
typedef struct jdf_body {
    struct jdf_object_t      super;
    struct jdf_body         *next;
    struct jdf_def_list     *properties;
    char                    *external_code;
} jdf_body_t;

/**
 * A JDF function is the complex object described below
 * It uses a jdf_flags_t type for its flags
 */
typedef unsigned int jdf_flags_t;
#define JDF_FUNCTION_FLAG_HIGH_PRIORITY     ((jdf_flags_t)(1 << 0))
#define JDF_FUNCTION_FLAG_CAN_BE_STARTUP    ((jdf_flags_t)(1 << 1))
#define JDF_FUNCTION_FLAG_NO_SUCCESSORS     ((jdf_flags_t)(1 << 2))
#define JDF_FUNCTION_FLAG_HAS_DISPLACEMENT  ((jdf_flags_t)(1 << 3))
#define JDF_FUNCTION_FLAG_HAS_DATA_INPUT    ((jdf_flags_t)(1 << 4))
#define JDF_FUNCTION_FLAG_HAS_DATA_OUTPUT   ((jdf_flags_t)(1 << 5))
#define JDF_FUNCTION_FLAG_NO_PREDECESSORS   ((jdf_flags_t)(1 << 6))

#define JDF_FUNCTION_HAS_UD_HASH_STRUCT     ((jdf_flags_t)(1 << 0))
#define JDF_PROP_UD_MAKE_KEY_FN_NAME        "make_key_fn"
#define JDF_PROP_UD_HASH_STRUCT_NAME        "hash_struct"

#define JDF_PROP_UD_NB_LOCAL_TASKS_FN_NAME     "nb_local_tasks_fn"

#define JDF_FUNCTION_HAS_UD_STARTUP_TASKS_FUN  ((jdf_flags_t)(1 << 2))
#define JDF_PROP_UD_STARTUP_TASKS_FN_NAME      "startup_fn"

#define JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS  ((jdf_flags_t)(1 << 3))
#define JDF_PROP_UD_FIND_DEPS_FN_NAME          "find_deps_fn"
#define JDF_PROP_UD_ALLOC_DEPS_FN_NAME         "alloc_deps_fn"
#define JDF_PROP_UD_FREE_DEPS_FN_NAME          "free_deps_fn"

typedef struct jdf_function_entry {
    struct jdf_object_t        super;
    struct jdf_function_entry *next;
    char                      *fname;
    struct jdf_name_list      *parameters;
    jdf_flags_t                flags;
    jdf_flags_t                user_defines;
    int32_t                    task_class_id;
    struct jdf_def_list       *locals;
    struct jdf_call           *predicate;
    struct jdf_dataflow       *dataflow;
    struct jdf_expr           *priority;
    struct jdf_expr           *simcost;
    struct jdf_def_list       *properties;
    struct jdf_body           *bodies;
    struct jdf_expr           *inline_c_functions;
} jdf_function_entry_t;

typedef struct jdf_data_entry {
    struct jdf_object_t      super;
    struct jdf_data_entry   *next;
    char                    *dname;
    struct jdf_global_entry *global;
    int                      nbparams;
} jdf_data_entry_t;

/*******************************************************************/
/*          Internal structures of the jdf_function                */
/*******************************************************************/

typedef struct jdf_name_list {
    struct jdf_object_t   super;
    struct jdf_name_list *next;
    char                 *name;
} jdf_name_list_t;

typedef struct jdf_def_list {
    struct jdf_object_t  super;
    struct jdf_def_list *next;
    char                *name;
    struct jdf_expr     *expr;
    struct jdf_def_list *properties;
} jdf_def_list_t;

typedef struct jdf_dataflow jdf_dataflow_t;
typedef struct jdf_dep jdf_dep_t;
typedef uint32_t jdf_flow_flags_t;
#define JDF_FLOW_TYPE_CTL     ((jdf_flow_flags_t)(1 << 0))
#define JDF_FLOW_TYPE_READ    ((jdf_flow_flags_t)(1 << 1))
#define JDF_FLOW_TYPE_WRITE   ((jdf_flow_flags_t)(1 << 2))
#define JDF_FLOW_HAS_DISPL    ((jdf_flow_flags_t)(1 << 3))
#define JDF_FLOW_HAS_IN_DEPS  ((jdf_flow_flags_t)(1 << 4))
#define JDF_FLOW_IS_IN        ((jdf_flow_flags_t)(1 << 5))
#define JDF_FLOW_IS_OUT       ((jdf_flow_flags_t)(1 << 6))

struct jdf_dataflow {
    struct jdf_object_t       super;
    jdf_flow_flags_t          flow_flags;
    jdf_dataflow_t           *next;
    char                     *varname;
    struct jdf_dep           *deps;
    uint8_t                   flow_index;
    uint32_t                  flow_dep_mask_out;
    uint32_t                  flow_dep_mask_in;
};

typedef uint16_t jdf_dep_flags_t;
#define JDF_DEP_FLOW_IN    ((jdf_dep_flags_t)(1 << 0))
#define JDF_DEP_FLOW_OUT   ((jdf_dep_flags_t)(1 << 1))
#define JDF_DEP_HAS_DISPL  ((jdf_dep_flags_t)(1 << 2))
#define JDF_DEP_HAS_IN     ((jdf_dep_flags_t)(1 << 3))

typedef struct jdf_datatransfer_type {
    struct jdf_object_t           super;
    struct jdf_expr              *type;    /**< the internal type of the data associated with the dependency */
    struct jdf_expr              *layout;  /**< the basic memory layout in case it is different from the type.
                                            *< InMPI case this must be an MPI datatype, working together with the
                                            *< displacement and the count. */
    struct jdf_expr              *count;   /**< number of elements of layout type to transfer */
    struct jdf_expr              *displ;   /**< displacement in number of bytes from the pointer associated with
                                            *< the dependency */
} jdf_datatransfer_type_t;

struct jdf_dep {
    struct jdf_object_t      super;
    jdf_dep_t               *next;
    struct jdf_guarded_call *guard;
    jdf_datatransfer_type_t  datatype;
    jdf_dep_flags_t          dep_flags;
    uint8_t                  dep_index;           /**< the index of the dependency in the context of the function */
    uint8_t                  dep_datatype_index;  /**< the smallest index of all dependencies
                                                   *   sharing a common datatype. */
};

typedef enum { JDF_GUARD_UNCONDITIONAL,
               JDF_GUARD_BINARY,
               JDF_GUARD_TERNARY } jdf_guard_type_t;

typedef struct jdf_guarded_call {
    struct jdf_object_t       super;
    jdf_guard_type_t          guard_type;
    struct jdf_expr          *guard;
    struct jdf_def_list      *properties;
    struct jdf_call          *calltrue;
    struct jdf_call          *callfalse;
} jdf_guarded_call_t;

typedef struct jdf_call {
    struct jdf_object_t       super;
    char                     *var;
    char                     *func_or_mem;
    struct jdf_expr          *parameters;
} jdf_call_t;

#define JDF_IS_CALL_WITH_NO_INPUT(CALL)                         \
    ((NULL == (CALL)->var) && (NULL == (CALL)->parameters))

/**
 * Return true if the flow is set only to define the global datatype of WRITE-only flow
 * If it is the case the guard is unconditional with only the NEW keyword and
 * optionnally some properties as follow:
 *   WRITE X <- NEW  [type = DEFAULT]
 */
#define JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(DEP)                           \
    ((NULL == (DEP)->guard->guard) &&                                   \
     (NULL != (DEP)->guard->calltrue) &&                                \
     (NULL == (DEP)->guard->callfalse) &&                               \
     (0 == strcmp(PARSEC_WRITE_MAGIC_NAME, (DEP)->guard->calltrue->func_or_mem)))

/*******************************************************************/
/*             Expressions (and list of expressions)              */
/*******************************************************************/

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
               JDF_STRING,
               JDF_CST,
               JDF_C_CODE
} jdf_expr_operand_t;

#define JDF_OP_IS_UNARY(op)    ( (op) == JDF_NOT )
#define JDF_OP_IS_TERNARY(op)  ( (op) == JDF_TERNARY )
#define JDF_OP_IS_CST(op)      ( (op) == JDF_CST )
#define JDF_OP_IS_STRING(op)   ( (op) == JDF_STRING )
#define JDF_OP_IS_VAR(op)      ( (op) == JDF_VAR )
#define JDF_OP_IS_C_CODE(op)   ( (op) == JDF_C_CODE )
#define JDF_OP_IS_BINARY(op)   ( !( JDF_OP_IS_UNARY(op) ||              \
                                    JDF_OP_IS_TERNARY(op) ||            \
                                    JDF_OP_IS_CST(op) ||                \
                                    JDF_OP_IS_VAR(op) ||                \
                                    JDF_OP_IS_C_CODE(op)) )

typedef struct jdf_expr {
    struct jdf_object_t           super;
    struct jdf_expr              *next;
    struct jdf_expr              *next_inline;
    jdf_expr_operand_t            op;
    union {
        struct {
            struct jdf_expr      *arg1;
            struct jdf_expr      *arg2;
            struct jdf_expr      *arg3;
        } ternary;
        struct {
            struct jdf_expr      *arg1;
            struct jdf_expr      *arg2;
        } binary;
        struct {
            struct jdf_expr      *arg;
        } unary;
        char                     *varname;
        struct {
            char                 *code;
            int                   lineno;
            char                 *fname;
            jdf_function_entry_t *function_context;
        } c_code;
        int                       cstval;
    } u;
} jdf_expr_t;

#define jdf_ua  u.unary.arg
#define jdf_ba1 u.binary.arg1
#define jdf_ba2 u.binary.arg2
#define jdf_ta1 u.ternary.arg1
#define jdf_ta2 u.ternary.arg2
#define jdf_tat u.ternary.arg3
#define jdf_ta3 u.ternary.arg3
#define jdf_var u.varname
#define jdf_cst u.cstval
#define jdf_c_code u.c_code

char *malloc_and_dump_jdf_expr_list( const jdf_expr_t *e );

#define JDF_COUNT_LIST_ENTRIES(LIST, TYPEOF, NEXT, COUNT)    \
    do {                                                     \
        TYPEOF* _item = (LIST);                              \
        (COUNT) = 0;                                         \
        while( NULL != _item) {                              \
            (COUNT)++;                                       \
            _item = _item->NEXT;                             \
        }                                                    \
    } while (0)

/**
 * Parse a list of properties in search for a specific name. The list of expressions is
 * supposed to be of the form VAR = expr. If the returned value is not NULL the property
 * parameter is updated to point to the matched property (if not NULL).
 *
 * @param [IN] the properties list
 * @param [IN] the name of the property to search
 * @param [OUT] if not NULL upon return it will contain the pointer to the matched property
 *
 * @return NULL if the requested property has not been found.
 * @return the expr on the left side of the = otherwise.
 */
jdf_expr_t* jdf_find_property( const jdf_def_list_t* properties, const char* property_name, jdf_def_list_t** property );

/**
 * Accessors for the properties
 */
int jdf_property_get_int( const jdf_def_list_t* properties, const char* prop_name, int ret_if_not_found );
const char *jdf_property_get_string( const jdf_def_list_t* properties, const char* prop_name, const char *ret_if_not_found );

/**
 * Add a new string property
 */
jdf_def_list_t *jdf_add_string_property(jdf_def_list_t **properties, const char *prop_name, const char *prop_value);

/**
 * Function cleanup and management. Available in jdf.c
 */
int jdf_flatten_function(jdf_function_entry_t* function);

#endif
