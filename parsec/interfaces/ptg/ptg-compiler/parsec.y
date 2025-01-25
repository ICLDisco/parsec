%{
/**
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* When BISON_FOUND is set, normal dependency tracking will generate the
 * .y.[ch] in the build directory from the .y in the source directory (this file).
 * In addition, one shall invoke by hand the rule to update the pregen
 * .y.[ch] files, before committing any changes to the .y files, e.g.,
 * `make parsec_pregen_flex_bison`.
 *
 * When BISON_FOUND is not set, the .y.[ch] version of this file is copied
 * from the pregenerated .y.[ch] file in `contrib/pregen_flex_bison`, and
 * modifying this file will result in a compilation error.
 */

#include "parsec/parsec_config.h"

#include <stdio.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <stdlib.h>
#include <assert.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"

#define PARSEC_ERR_NEW_AS_OUTPUT  "Automatic data allocation with NEW only supported in IN dependencies.\n"
#define PARSEC_ERR_NULL_AS_OUTPUT "NULL data only supported in IN dependencies.\n"

#define YYDEBUG_LEXER_TEXT yytext

/**
 * Better error handling
 */
#define YYERROR_VERBOSE 1
struct yyscan_t;
#if defined(YYPURE) && YYPURE
extern int yylex(struct yyscan_t *yycontrol);
#else
extern int yylex(void);
#endif

#include "parsec.y.h"

extern int current_lineno;

static jdf_expr_t *inline_c_functions = NULL;
static jdf_expr_t *current_locally_bound_variables = NULL;
static int current_locally_bound_variables_scope = 0;

/**
 *
 * http://oreilly.com/linux/excerpts/9780596155971/error-reporting-recovery.html
 *
 */
int yylex();

static void yyerror(
#if defined(YYPURE) && YYPURE
                    YYLTYPE *locp,
                    struct yyscan_t* yyscanner,
#endif  /* defined(YYPURE) && YYPURE */
                    char const *msg)
{
#if !defined(YYPURE) || !YYPURE
    YYLTYPE *locp = &yylloc;
#endif
    if(NULL != locp) {
        if(locp->first_line) {
            fprintf(stderr, "parse error at (%d) %d.%d-%d.%d: %s\n",
                    current_lineno, locp->first_line, locp->first_column,
                    locp->last_line, locp->last_column, msg);
        } else {
            fprintf(stderr, "parse error at (%d): %s\n",
                    current_lineno, msg);
        }
    } else {
#if defined(YYPURE) && YYPURE
        fprintf(stderr, "parse error near line %d: %s\n ", yyget_lineno(), msg);
#endif
    }
}

#define new(type) (type*)calloc(1, sizeof(type))

jdf_def_list_t*
jdf_create_properties_list( const char* name,
                            int default_int,
                            const char* default_char,
                            jdf_def_list_t* next )
{
    jdf_def_list_t* property;
    jdf_expr_t *e;

    property                    = new(jdf_def_list_t);
    property->next              = next;
    property->name              = strdup(name);
    JDF_OBJECT_LINENO(property) = current_lineno;

    if( NULL != default_char ) {
        e = new(jdf_expr_t);
        e->op = JDF_STRING;
        e->jdf_var = strdup(default_char);
    } else {
        e = new(jdf_expr_t);
        e->op = JDF_CST;
        e->jdf_cst = default_int;
    }
    property->expr = e;
    return property;
}

static jdf_data_entry_t* jdf_find_or_create_data(jdf_t* jdf, const char* dname)
{
    jdf_data_entry_t* data = jdf->data;
    jdf_global_entry_t* global = jdf->globals;

    while( NULL != data ) {
        if( !strcmp(data->dname, dname) ) {
            return data;
        }
        data = data->next;
    }
    /* not found, create */
    data = new(jdf_data_entry_t);
    data->dname             = strdup(dname);
    JDF_OBJECT_LINENO(data) = current_lineno;
    data->nbparams          = -1;
    data->next              = jdf->data;
    jdf->data               = data;
    /* Check if there is a global with the same name */
    while( NULL != global ) {
        if( !strcmp(global->name, data->dname) ) {
            assert(NULL == global->data);
            global->data = data;
            data->global = global;

            if( jdf_find_property( global->properties, "type", NULL ) == NULL ) {
                global->properties = jdf_create_properties_list( "type", 0, "parsec_data_collection_t*", global->properties);
            }

            return data;
        }
        global = global->next;
    }
    assert(NULL == global);
    global                    = new(jdf_global_entry_t);
    global->name              = strdup(data->dname);
    global->properties        = jdf_create_properties_list( "type", 0, "parsec_data_collection_t*", NULL );
    global->data              = data;
    global->expression        = NULL;
    JDF_OBJECT_LINENO(global) = current_lineno;
    data->global              = global;
    /* Chain it with the other globals */
    global->next = jdf->globals;
    jdf->globals = global;

    return data;
}

static int key_from_type(char *type) {
    if (!strcmp(type, "int64_t")) return 1;
    if (!strcmp(type, "float")) return 2;
    if (!strcmp(type, "double")) return 3;
    return 0;
}

static void named_expr_push_scope(void) {
    current_locally_bound_variables_scope++;
}

static void named_expr_pop_scope(void) {
    assert(current_locally_bound_variables_scope > 0);
    while(NULL != current_locally_bound_variables && current_locally_bound_variables->scope == current_locally_bound_variables_scope) {
        current_locally_bound_variables = current_locally_bound_variables->next;
    }
    current_locally_bound_variables_scope--;
}

static jdf_expr_t *named_expr_push_in_scope(char *var, jdf_expr_t *e) {
    e->next = current_locally_bound_variables;
    e->alias = var;
    e->ldef_index = -1; /* Invalid index, used to capture mistakes and remember if an index was already assigned to this local definition */
    e->scope = current_locally_bound_variables_scope;
    current_locally_bound_variables = e;
    return e;
}

static void
process_datatype(jdf_datatransfer_type_t *datatype,
                 jdf_dep_t *d,
                 const jdf_def_list_t *properties,
                 jdf_def_list_t *property,
                 char *layout,
                 char *count,
                 char *displ)
{
    struct jdf_name_list *g, *e, *prec;
    jdf_expr_t *expr = datatype->type;
    if( NULL != jdf_find_property( properties, "arena_index", &property ) ) {
        jdf_fatal(current_lineno, "Old construct arena_index used. Please update the code to use type instead.\n");
        yyerror("");
    }
    if( NULL != jdf_find_property( properties, "nb_elt", &property ) ) {
        jdf_fatal(current_lineno, "Old construct nb_elt used. Please update the code to use count instead.\n");
        yyerror("");
    }

    if (expr->op == JDF_C_CODE){ /* Correct the type of the JDF_C_CODE function */
        expr->jdf_type = PARSEC_RETURN_TYPE_ARENA_DATATYPE_T;
    }

    if( (JDF_STRING == expr->op) || (JDF_VAR == expr->op) ) {
        /* Special case: [type = SOMETHING] -> define the PARSEC_ARENA_SOMETHING arena index */
        for(prec = NULL, g = current_jdf.datatypes; g != NULL; g = g->next) {
            if( 0 == strcmp(expr->jdf_var, g->name) ) {
                break;
            }
            prec = g;
        }
        if( NULL == g ) {
            e = new(struct jdf_name_list);
            e->name = strdup(expr->jdf_var);
            e->next = NULL;
            if( NULL != prec ) {
                prec->next = e;
            } else {
                current_jdf.datatypes = e;
            }
        }
    }

    /**
     * The memory layout used for the transfer. Works together with the
     * count and the displacement. If not layout is specified it is assumed
     * the default layout of the type (arena) will be used.
     */
    datatype->layout = jdf_find_property( d->guard->properties, layout, &property );

    /**
     * The number of types to transfer.
     */
    expr = jdf_find_property( d->guard->properties, count, &property );
    if( NULL == expr ) {
        expr          = new(jdf_expr_t);
        expr->op      = JDF_CST;
        expr->jdf_cst = 1;
    }
    datatype->count = expr;

    /**
     * The displacement from the begining of the type.
     */
    expr = jdf_find_property( d->guard->properties, displ, &property );
    if( NULL == expr ) {
        expr          = new(jdf_expr_t);
        expr->op      = JDF_CST;
        expr->jdf_cst = 0;
    } else {
        if( !((JDF_CST != expr->op) && (0 == expr->jdf_cst)) )
            d->dep_flags |= JDF_DEP_HAS_DISPL;
    }
    datatype->displ = expr;
}
%}

%union {
    int                   number;
    char*                 string;
    jdf_expr_operand_t    expr_op;
    jdf_external_entry_t *external_code;
    jdf_global_entry_t   *global;
    jdf_function_entry_t *function;
    jdf_param_list_t     *param_list;
    jdf_def_list_t       *property;
    jdf_name_list_t      *name_list;
    jdf_variable_list_t  *variable_list;
    jdf_dataflow_t       *dataflow;
    jdf_expr_t           *named_expr;
    jdf_dep_t            *dep;
    jdf_dep_flags_t       dep_type;
    jdf_guarded_call_t   *guarded_call;
    jdf_call_t           *call;
    jdf_expr_t           *expr;
    jdf_body_t           *body;
};

%glr-parser
%expect 5

%type <function>function
%type <param_list>param_list
%type <variable_list>local_variable
%type <variable_list>local_variables
%type <call>partitioning
%type <dataflow>dataflow_list
%type <dataflow>dataflow
%type <named_expr>named_expr
%type <named_expr>named_expr_list
%type <dep>dependencies
%type <dep>dependency
%type <guarded_call>guarded_call
%type <property>property
%type <property>properties
%type <property>properties_list
%type <call>call
%type <expr>expr_list
%type <expr>expr_list_range
%type <expr>expr_range
%type <expr>expr_simple
%type <expr>priority
%type <expr>simulation_cost
%type <expr>variable
%type <number>optional_flow_flags
%type <external_code>prologue
%type <external_code>epilogue
%type <body>body
%type <body>bodies

%type <string>VAR
%type <string>EXTERN_DECL
%type <string>BODY_END
%type <dep_type>ARROW
%type <number>PROPERTIES_ON
%type <number>PROPERTIES_OFF
%type <string>STRING
%type <number>JDF_INT
%type <number>DEPENDENCY_TYPE
%type <number>DATA_NEW
%type <number>DATA_NULL

%token VAR ASSIGNMENT EXTERN_DECL COMMA OPEN_PAR CLOSE_PAR BODY_START BODY_END STRING SIMCOST
%token COLON SEMICOLON DEPENDENCY_TYPE ARROW QUESTION_MARK PROPERTIES_ON PROPERTIES_OFF
%token DATA_NEW DATA_NULL
%token EQUAL NOTEQUAL LESS LEQ MORE MEQ AND OR XOR NOT JDF_INT
%token PLUS MINUS TIMES DIV MODULO SHL SHR RANGE OPTION

/* C99 operator precedence: http://en.cppreference.com/w/c/language/operator_precedence */
%nonassoc RANGE
%left ARROW
%left COMMA
%right ASSIGNMENT
%right QUESTION_MARK COLON
%left OR
%left AND
%left XOR
%left EQUAL NOTEQUAL
%left LESS LEQ MORE MEQ
%left SHL SHR
%left PLUS MINUS
%left MODULO TIMES DIV
%right NOT
%left OPEN_PAR CLOSE_PAR

%debug
/*%pure-parser*/
%locations
%define parse.error verbose
/*%parse-param {struct yyscan_t *yycontrol}*/
/*%lex-param   {struct yyscan_t *yycontrol}*/
%%
jdf_file:       prologue jdf epilogue
                {
                    assert( NULL == current_jdf.prologue );
                    assert( NULL == current_jdf.epilogue );
                    current_jdf.prologue = $1;
                    current_jdf.epilogue = $3;
                }
        ;
prologue:       EXTERN_DECL
                {
                    $$ = new(jdf_external_entry_t);
                    $$->external_code = $1;
                    JDF_OBJECT_LINENO($$) = current_lineno;
                }
        |
                {
                    $$ = NULL;
                }
        ;
epilogue:       EXTERN_DECL
                {
                    $$ = new(jdf_external_entry_t);
                    $$->external_code = $1;
                    JDF_OBJECT_LINENO($$) = current_lineno;
                }
        |
                {
                    $$ = NULL;
                }
        ;
jdf:            jdf function
                {
                    jdf_expr_t *el;

                    if( NULL == current_jdf.functions ) {
                        $2->task_class_id = 0;
                    } else {
                        $2->task_class_id = current_jdf.functions->task_class_id + 1;
                    }
                    $2->next = current_jdf.functions;
                    current_jdf.functions = $2;
                    if( NULL != inline_c_functions) {
                        /* Every inline functions declared here where within the context of $2 */
                        for(el = inline_c_functions; NULL != el; el = el->next_inline) {
                            el->jdf_c_code.function_context = $2;
                        }
                    }
                    $2->inline_c_functions = inline_c_functions;
                    inline_c_functions = NULL;
                }
        |       jdf VAR ASSIGNMENT expr_simple properties
                {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    jdf_expr_t *el;

                    e->next              = NULL;
                    e->name              = $2;
                    e->properties        = $5;
                    e->expression        = $4;
                    JDF_OBJECT_LINENO(e) = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next_inline; el = el->next_inline) /* nothing */ ;
                        el->next_inline = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
        |       jdf VAR properties
                {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    jdf_expr_t *el;

                    e->next              = NULL;
                    e->name              = $2;
                    e->properties        = $3;
                    e->expression        = NULL;
                    JDF_OBJECT_LINENO(e) = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next_inline; el = el->next_inline) /* nothing */ ;
                        el->next_inline = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
        | jdf OPTION property
                {
                    $3->next = current_jdf.global_properties;
                    current_jdf.global_properties = $3;
                }
        |
                {
                    if( NULL != inline_c_functions ) {
                        jdf_expr_t *el;
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next_inline; el = el->next_inline) /* nothing */ ;
                        el->next_inline = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
        ;

properties:   PROPERTIES_ON properties_list PROPERTIES_OFF
              {
                  $$ = $2;
              }
       |
              {
                  $$ = NULL;
              }
       ;

property:     VAR ASSIGNMENT expr_simple
              {
                  jdf_def_list_t* assign = new(jdf_def_list_t);
                  assign->next              = NULL;
                  assign->name              = strdup($1);
                  assign->expr              = $3;
                  JDF_OBJECT_LINENO(assign) = JDF_OBJECT_LINENO($3);
                  $$ = assign;
              }
       |      VAR VAR ASSIGNMENT expr_simple
              {
                  jdf_def_list_t* assign = new(jdf_def_list_t);
                  assign->next              = NULL;
                  assign->name              = strdup($2);
                  assign->expr              = $4;
                  assign->expr->jdf_type    = key_from_type($1);
                  JDF_OBJECT_LINENO(assign) = JDF_OBJECT_LINENO($4);
                  $$ = assign;
              }
       ;

properties_list: property properties_list
              {
                 $1->next = $2;
                 $$       = $1;
              }
       | property
              { $$ = $1; }
       ;

body:         BODY_START properties BODY_END
             {
                 jdf_body_t* body = new(jdf_body_t);
                 body->properties = $2;
                 /**
                  * Go over the list of properties and tag them with the type of the BODY. This
                  * allows us to protect the generated code to avoid compiler warnings.
                  */
                 jdf_expr_t* type_str = jdf_find_property( body->properties, "type", NULL );
                 if( NULL != type_str ) {
                     char* protected_by;
                     asprintf(&protected_by, "PARSEC_HAVE_DEV_%s_SUPPORT", type_str->jdf_var);
                     jdf_def_list_t* property = body->properties;
                     while( NULL != property ) {
                         assert(NULL == property->expr->protected_by);
                         property->expr->protected_by = strdup(protected_by);
                         property = property->next;
                     }
                     free(protected_by);
                 }
                 body->next = NULL;
                 /* The body of the function is stored as a string in BODY_END */
                 body->external_code = $3;
                 JDF_OBJECT_LINENO(body) = current_lineno;
                 $$ = body;
             }
       ;
bodies: body
             {
                 $$ = $1;
             }
       | body bodies
             {
                 jdf_body_t* body = $1;
                 body->next = $2;
                 $$ = $1;
             }
       ;
function:       VAR OPEN_PAR param_list CLOSE_PAR properties local_variables simulation_cost partitioning dataflow_list priority bodies
                {
                    int rc;
                    jdf_function_entry_t *e = new(jdf_function_entry_t);
                    e->fname             = $1;
                    e->parameters        = $3;
                    e->properties        = $5;
                    e->locals            = $6;
                    e->simcost           = $7;
                    e->predicate         = $8;
                    e->dataflow          = $9;
                    e->priority          = $10;
                    e->bodies            = $11;

                    jdf_link_params_and_locals(e);  /* link params and locals */
                    jdf_assign_ldef_index(e);       /* find the datatype indexes */

                    rc = jdf_flatten_function(e);
                    if( rc < 0 )
                        YYERROR;
                    $$ = e;
                    JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($3);
                }
        ;

param_list:        VAR COMMA param_list
                {
                    jdf_param_list_t *l = new(jdf_param_list_t);
                    l->next = $3;
                    l->name = $1;
                    l->local = NULL;
                    JDF_OBJECT_LINENO(l) = current_lineno;

                    $$ = l;
                }
         |      VAR
                {
                    jdf_param_list_t *l = new(jdf_param_list_t);
                    l->next = NULL;
                    l->name = $1;
                    l->local = NULL;
                    JDF_OBJECT_LINENO(l) = current_lineno;

                    $$ = l;
                }
         ;

local_variable:
                VAR ASSIGNMENT expr_range properties
                {
                    jdf_variable_list_t *l = new(jdf_variable_list_t);
                    l->name               = $1;
                    l->expr               = $3;
                    l->properties         = $4;
                    l->next               = NULL;
                    l->param              = NULL;
                    $$ = l;
                    JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($3);
                }
         ;
local_variables:
                local_variable local_variables
                {
                    $1->next = $2;
                    $$ = $1;
                }
         |      local_variable
                {
                    $$ = $1;
                }
         ;

simulation_cost:
                SIMCOST expr_simple
                {
                    $$ = $2;
                    $$->protected_by = strdup("PARSEC_SIM");
                }
             |  {   $$ = NULL; }
             ;

partitioning:   COLON VAR OPEN_PAR expr_list CLOSE_PAR
              {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = $2;
                  data = jdf_find_or_create_data(&current_jdf, $2);
                  c->parameters = $4;
                  JDF_COUNT_LIST_ENTRIES($4, jdf_expr_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    $2, nbparams, current_lineno, data->nbparams, JDF_OBJECT_LINENO(data));
                          YYERROR;
                      }
                  } else {
                      data->nbparams          = nbparams;
                  }
                  $$ = c;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($4);
              }
         ;

dataflow_list:  dataflow dataflow_list
                {
                    $1->next = $2;
                    $$ = $1;
                }
         |
                {
                    $$ = NULL;
                }
         ;

optional_flow_flags :
                DEPENDENCY_TYPE
                {
                    $$ = $1;
                }
         |      { $$ = JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE; }
         ;

dataflow:       optional_flow_flags VAR dependencies
                {
                    for(jdf_global_entry_t* g = current_jdf.globals; g != NULL; g = g->next) {
                        if( !strcmp(g->name, $2) ) {
                            jdf_fatal(current_lineno, "Flow %s cannot shadow global variable with the same name (defined line %d)\n",
                                      $2, JDF_OBJECT_LINENO(g));
                            YYERROR;
                        }
                    }

                    jdf_dataflow_t *flow  = new(jdf_dataflow_t);
                    flow->flow_flags      = $1;
                    flow->varname         = $2;
                    flow->deps            = $3;

                    $$ = flow;
                    if( NULL == $3) {
                        JDF_OBJECT_LINENO($$) = current_lineno;
                    } else {
                        JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($3);
                    }
                }
        ;

named_expr: PROPERTIES_ON { named_expr_push_scope(); } named_expr_list PROPERTIES_OFF
               {
                   $$ = $3;
               }
        |
               {
                   $$ = NULL;
                   /* We still create a new scope for the (inexisting) named range
                    * as the scope will be popped unconditionally */
                   named_expr_push_scope();
               }
        ;

named_expr_list: VAR ASSIGNMENT expr_range
               {
                   $$ = named_expr_push_in_scope($1, $3);
               }
        |  named_expr_list COMMA VAR ASSIGNMENT expr_range
               {
                   $$ = named_expr_push_in_scope($3, $5);
               }
       ;

dependencies:  dependency dependencies
               {
                   $1->next = $2;
                   $$ = $1;
               }
        |
               {
                   $$ = NULL;
               }
       ;

dependency:   ARROW named_expr guarded_call properties
              {
                  jdf_dep_t *d = new(jdf_dep_t);
                  jdf_expr_t *expr;
                  jdf_expr_t *expr_remote;
                  jdf_expr_t *expr_data;
                  jdf_def_list_t* property = $4;
                  jdf_def_list_t* property_remote = $4;
                  jdf_def_list_t* property_data = $4;

                  d->local_defs = $2;

                  expr = jdf_find_property($4, "type", &property);
                  expr_remote = jdf_find_property($4, "type_remote", &property_remote);
                  expr_data = jdf_find_property($4, "type_data", &property_data);
                  /* Warn when using type_remote & type_data on the same dep.
                   * Here we should check if type|type_data on in/out desc.
                   * And type|type_remote on task dependencies.
                   */
                  if((expr_remote != NULL) && (expr_data != NULL)){
                      jdf_fatal(JDF_OBJECT_LINENO($4),
                          " the simultaneous usage of type_remote and type_data is not supported. \n");
                      YYERROR;
                  }

                  /* If neither is defined, we define the old simple DEFAULT arena */
                  if( NULL == expr ) {
                      property = jdf_create_properties_list( "type", 0, "DEFAULT", NULL );
                      expr = jdf_find_property( property, "type", &property );
                      property->next = $4;
                  }
                  if( NULL == expr_remote ) {
                      property_remote = jdf_create_properties_list( "type_remote", 0, "DEFAULT", NULL );
                      expr_remote = jdf_find_property( property_remote, "type_remote", &property_remote );
                      property_remote->next = $4;
                  }
                  if( NULL == expr_data ) {
                      property_data = jdf_create_properties_list( "type_data", 0, "DEFAULT", NULL );
                      expr_data = jdf_find_property( property_data, "type_data", &property_data );
                      property_data->next = $4;
                  }

                  /* Validate the datatype definition of WRITE only data */
                  if( (JDF_GUARD_UNCONDITIONAL == $3->guard_type) &&
                      (NULL == $3->guard) && (NULL == $3->callfalse) ) {
                      if( 0 == strcmp(PARSEC_WRITE_MAGIC_NAME, $3->calltrue->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($3), PARSEC_ERR_NEW_AS_OUTPUT);
                              YYERROR;
                          }
                      }
                  }

                  /* Validate the WRITE only data allocation, and the unused data */
                  if( JDF_IS_CALL_WITH_NO_INPUT($3->calltrue) ) {
                      if( 0 == strcmp(PARSEC_WRITE_MAGIC_NAME, $3->calltrue->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($3), PARSEC_ERR_NEW_AS_OUTPUT);
                              YYERROR;
                          }
                      }
                      else if( 0 == strcmp(PARSEC_NULL_MAGIC_NAME, $3->calltrue->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($3), PARSEC_ERR_NULL_AS_OUTPUT);
                              YYERROR;
                          }
                      } else {
                          jdf_fatal(JDF_OBJECT_LINENO($3),
                                    "%s is not a supported keyword to describe a data (Only NEW and NULL are supported by themselves)\n",
                                    $3->calltrue->func_or_mem );
                          YYERROR;
                      }
                  }

                  /* Validate the WRITE only data allocation, and the unused data */
                  if( $3->callfalse && JDF_IS_CALL_WITH_NO_INPUT($3->callfalse) ) {
                      if( 0 == strcmp(PARSEC_WRITE_MAGIC_NAME, $3->callfalse->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($3), PARSEC_ERR_NEW_AS_OUTPUT);
                              YYERROR;
                          }
                      }
                      else if( 0 == strcmp(PARSEC_NULL_MAGIC_NAME, $3->callfalse->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($3), PARSEC_ERR_NULL_AS_OUTPUT);
                              YYERROR;
                          }
                      } else {
                          jdf_fatal(JDF_OBJECT_LINENO($3),
                                    "%s is not a supported keyword to describe a data (Only NEW and NULL are supported by themselves)\n",
                                    $3->callfalse->func_or_mem );
                          YYERROR;
                      }
                  }

                  $3->properties   = property;
                  d->dep_flags     = $1;
                  d->guard         = $3;
                  d->datatype_local.type = expr;
                  d->datatype_remote.type = expr_remote;
                  d->datatype_data.type = expr_data;
                  JDF_OBJECT_LINENO(&d->datatype_local) = current_lineno;
                  JDF_OBJECT_LINENO(&d->datatype_remote) = current_lineno;
                  JDF_OBJECT_LINENO(&d->datatype_data) = current_lineno;

                  process_datatype(&d->datatype_local, d, $4, property, "layout", "count", "displ");
                  process_datatype(&d->datatype_remote, d, $4, property_remote, "layout_remote", "count_remote", "displ_remote");
                  process_datatype(&d->datatype_data, d, $4, property_data, "layout_data", "count_data", "displ_data");

                  JDF_OBJECT_LINENO(d) = JDF_OBJECT_LINENO($3);
                  assert( 0 != JDF_OBJECT_LINENO($3) );
                  $$ = d;

                  named_expr_pop_scope();
              }
       ;

guarded_call: call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_UNCONDITIONAL;
                  g->guard = NULL;
                  g->calltrue = $1;
                  g->callfalse = NULL;
                  $$ = g;
                  assert( 0 != JDF_OBJECT_LINENO($1) );
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
              }
       |      expr_simple QUESTION_MARK call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = NULL;
                  $$ = g;
                  assert( 0 != JDF_OBJECT_LINENO($1) );
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
              }
       |      expr_simple QUESTION_MARK call COLON call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = $5;
                  $$ = g;
                  assert( 0 != JDF_OBJECT_LINENO($1) );
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
              }
       ;

call:         named_expr VAR VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = $2;
                  c->local_defs = $1;
                  c->func_or_mem = $3;
                  c->parameters = $5;
                  $$ = c;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($5);
                  assert( 0 != JDF_OBJECT_LINENO($$) );
                  named_expr_pop_scope();
              }
       |      VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = $1;
                  c->parameters = $3;
                  c->local_defs = NULL;
                  JDF_OBJECT_LINENO(c) = JDF_OBJECT_LINENO($3);
                  $$ = c;
                  assert( 0 != JDF_OBJECT_LINENO($$) );
                  data = jdf_find_or_create_data(&current_jdf, $1);
                  JDF_COUNT_LIST_ENTRIES($3, jdf_expr_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    $1, nbparams, current_lineno, data->nbparams, JDF_OBJECT_LINENO(data));
                          YYERROR;
                      }
                  } else {
                      data->nbparams          = nbparams;
                  }
                  JDF_OBJECT_LINENO(data) = JDF_OBJECT_LINENO($3);
              }
       |      DATA_NEW
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = NULL;
                  c->local_defs = NULL;
                  c->func_or_mem = strdup(PARSEC_WRITE_MAGIC_NAME);
                  c->parameters = NULL;
                  JDF_OBJECT_LINENO(c) = current_lineno;
                  $$ = c;
             }
       |     DATA_NULL
             {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = NULL;
                  c->local_defs = NULL;
                  c->func_or_mem = strdup(PARSEC_NULL_MAGIC_NAME);
                  c->parameters = NULL;
                  JDF_OBJECT_LINENO(c) = current_lineno;
                  $$ = c;
             }
       ;

priority:     SEMICOLON expr_simple
              {
                    $$ = $2;
              }
       |      { $$ = NULL; }
       ;

/* And now, the expressions */

expr_list_range: expr_range COMMA expr_list_range
              {
                  $1->next = $3;
                  $$=$1;
                  assert( 0 != JDF_OBJECT_LINENO($$) );
              }
      |       expr_range
              {
                  $1->next = NULL;
                  $$=$1;
                  assert( 0 != JDF_OBJECT_LINENO($$) );
              }
      ;

expr_list:    expr_simple COMMA expr_list
              {
                  $1->next = $3;
                  $$=$1;
              }
      |       expr_simple
              {
                  $1->next = NULL;
                  $$=$1;
              }
      ;

expr_range: expr_simple RANGE expr_simple
            {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ta1 = $1;               /* from */
                  e->jdf_ta2 = $3;               /* to */

                  e->jdf_ta3 = new(jdf_expr_t);  /* step */
                  e->jdf_ta3->op = JDF_CST;
                  e->jdf_ta3->jdf_cst = 1;
                  e->jdf_ta3->local_variables = current_locally_bound_variables;
                  e->jdf_ta3->scope = -1;
                  e->jdf_ta3->alias = NULL;

                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
                  assert( 0 != JDF_OBJECT_LINENO($$) );
            }
          | expr_simple RANGE expr_simple RANGE expr_simple
            {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ta1 = $1;  /* from */
                  e->jdf_ta2 = $3;  /* to */
                  e->jdf_ta3 = $5;  /* step */
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
                  assert( 0 != JDF_OBJECT_LINENO($$) );
            }
          | expr_simple
            {
                  $$ = $1;
                  assert( 0 != JDF_OBJECT_LINENO($$) );
            }
          | PROPERTIES_ON { named_expr_push_scope(); } named_expr_list PROPERTIES_OFF expr_simple
            {
                   /* we cannot simply say it's 'named_expr expr_simple', or this creates a lot of
                    * ambiguous shift-reduce conflicts because named_expr can be empty */
                   $$ = $5;
                   named_expr_pop_scope();
            }
          ;

variable: VAR
            {
                 jdf_expr_t *e = new(jdf_expr_t);
                 e->op = JDF_VAR;
                 e->local_variables = current_locally_bound_variables;
                 e->scope = -1;
                 e->alias = NULL;
                 e->jdf_var = strdup($1);
                 $$ = e;
                 JDF_OBJECT_LINENO($$) = current_lineno;
            }
        | variable ARROW VAR
            {
                 char* tmp = NULL;
                 if( asprintf(&tmp, "%s->%s", $1->jdf_var, $3) <= 0 )
                     tmp = NULL;
                 free($1->jdf_var);
                 $1->jdf_var = tmp;
                 $$ = $1;
            }
        ;

expr_simple:  expr_simple EQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_EQUAL;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple NOTEQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOTEQUAL;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple LESS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LESS;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple LEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LEQ;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MORE expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MORE;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MEQ;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple AND expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_AND;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple OR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_OR;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple XOR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_XOR;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      NOT expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOT;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ua = $2;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      OPEN_PAR expr_simple CLOSE_PAR
              {
                  $$ = $2;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple PLUS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_PLUS;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MINUS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MINUS;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple TIMES expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TIMES;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple DIV expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_DIV;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MODULO expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MODULO;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple SHL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHL;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple SHR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHR;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple QUESTION_MARK expr_simple COLON expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TERNARY;
                  e->local_variables = current_locally_bound_variables;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_tat = $1;
                  e->jdf_ta1 = $3;
                  e->jdf_ta2 = $5;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      variable
              {
                  $$ = $1;
              }
       |      JDF_INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->local_variables = NULL;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_cst = $1;
                  e->jdf_type = 0;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      MINUS JDF_INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->local_variables = NULL;
                  e->scope = -1;
                  e->alias = NULL;
                  e->jdf_cst = -$2;
                  e->jdf_type = 0;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      STRING
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->local_variables = NULL;
                  e->scope = -1;
                  e->alias = NULL;
                  e->op = JDF_STRING;
                  e->jdf_var = strdup($1);
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      EXTERN_DECL
              {
                  $$ = new(jdf_expr_t);
                  $$->op = JDF_C_CODE;
                  $$->jdf_c_code.code = $1;
                  $$->jdf_type = PARSEC_RETURN_TYPE_INT32;
                  $$->local_variables = current_locally_bound_variables;
                  $$->scope = -1;
                  $$->alias = NULL;
                  $$->jdf_c_code.lineno = current_lineno;
                  /* This will be set by the upper level parsing if necessary */
                  $$->jdf_c_code.function_context = NULL;
                  $$->jdf_c_code.fname = NULL;

                  $$->next_inline = inline_c_functions;
                  inline_c_functions = $$;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
        ;

%%

