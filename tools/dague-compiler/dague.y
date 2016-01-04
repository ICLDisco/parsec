%{
/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#include <stdio.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#include <stdlib.h>
#include <assert.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"

#define YYDEBUG_LEXER_TEXT yytext

/**
 * Better error handling
 */
#define YYERROR_VERBOSE 1
struct yyscan_t;

#include "dague.y.h"

extern int current_lineno;

static jdf_expr_t *inline_c_functions = NULL;

/**
 *
 * http://oreilly.com/linux/excerpts/9780596155971/error-reporting-recovery.html
 *
 */
static void yyerror(YYLTYPE *locp,
#if defined(YYPURE) && YYPURE
                    struct yyscan_t* yyscanner,
#endif  /* defined(YYPURE) && YYPURE */
                    char const *msg)
{
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
        fprintf(stderr, "parse error near line %d: %s\n ", yyget_lineno(), msg);
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
                global->properties = jdf_create_properties_list( "type", 0, "dague_ddesc_t*", global->properties);
            }

            return data;
        }
        global = global->next;
    }
    assert(NULL == global);
    global                    = new(jdf_global_entry_t);
    global->name              = strdup(data->dname);
    printf("Create data %s\n", data->dname);
    global->properties        = jdf_create_properties_list( "type", 0, "dague_ddesc_t*", NULL );
    global->data              = data;
    global->expression        = NULL;
    JDF_OBJECT_LINENO(global) = current_lineno;
    data->global              = global;
    /* Chain it with the other globals */
    global->next = jdf->globals;
    jdf->globals = global;

    return data;
}

%}

%union {
    int                   number;
    char*                 string;
    jdf_expr_operand_t    expr_op;
    jdf_external_entry_t *external_code;
    jdf_global_entry_t   *global;
    jdf_function_entry_t *function;
    jdf_def_list_t       *property;
    jdf_name_list_t      *name_list;
    jdf_def_list_t       *def_list;
    jdf_dataflow_t       *dataflow;
    jdf_dep_t            *dep;
    jdf_dep_flags_t       dep_type;
    jdf_guarded_call_t   *guarded_call;
    jdf_call_t           *call;
    jdf_expr_t           *expr;
    jdf_body_t           *body;
};

%type <function>function
%type <name_list>varlist
%type <def_list>execution_space
%type <call>partitioning
%type <dataflow>dataflow_list
%type <dataflow>dataflow
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
%type <number>INT
%type <number>DEPENDENCY_TYPE

%token VAR ASSIGNMENT EXTERN_DECL COMMA OPEN_PAR CLOSE_PAR BODY_START BODY_END STRING SIMCOST
%token COLON SEMICOLON DEPENDENCY_TYPE ARROW QUESTION_MARK PROPERTIES_ON PROPERTIES_OFF
%token EQUAL NOTEQUAL LESS LEQ MORE MEQ AND OR XOR NOT INT
%token PLUS MINUS TIMES DIV MODULO SHL SHR RANGE OPTION

/* C99 operator precedence: http://en.cppreference.com/w/c/language/operator_precedence */
%nonassoc RANGE
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
%error-verbose
%parse-param {struct yyscan_t *yycontrol}
%lex-param   {struct yyscan_t *yycontrol}
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
                        $2->function_id = 0;
                    } else {
                        $2->function_id = current_jdf.functions->function_id + 1;
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
                 body->next = NULL;
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
function:       VAR OPEN_PAR varlist CLOSE_PAR properties execution_space simulation_cost partitioning dataflow_list priority bodies
                {
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

                    JDF_OBJECT_LINENO(e) = current_lineno;

                    jdf_flatten_function(e);
                    $$ = e;
                }
        ;

varlist:        VAR COMMA varlist
                {
                    jdf_name_list_t *l = new(jdf_name_list_t);
                    l->next = $3;
                    l->name = $1;

                    $$ = l;
                }
         |      VAR
                {
                    jdf_name_list_t *l = new(jdf_name_list_t);
                    l->next = NULL;
                    l->name = $1;

                    $$ = l;
                }
         |
                {
                    $$ = NULL;
                }
         ;

execution_space:
                VAR ASSIGNMENT expr_range properties execution_space
                {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name               = $1;
                    l->expr               = $3;
                    l->properties         = $4;
                    l->next               = $5;
                    $$ = l;
                    JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($3);
                }
         |      VAR ASSIGNMENT expr_range properties
                {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name               = $1;
                    l->expr               = $3;
                    l->properties         = $4;
                    l->next               = NULL;
                    $$ = l;
                    JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($3);
                }
         ;

simulation_cost:
                SIMCOST expr_simple
                {
                    $$ = $2;
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

dependency:   ARROW guarded_call properties
              {
                  struct jdf_name_list *g, *e, *prec;
                  jdf_dep_t *d = new(jdf_dep_t);
                  jdf_expr_t *expr;
                  jdf_def_list_t* property = $3;

                  /* If neither is defined, we define the old simple DEFAULT arena */
                  if( NULL == (expr = jdf_find_property($3, "type", &property)) ) {
                      property = jdf_create_properties_list( "type", 0, "DEFAULT", NULL );
                      expr = jdf_find_property( property, "type", &property );
                      property->next = $3;
                  }
                  /* Validate the WRITE only data allocation */
                  if( (JDF_GUARD_UNCONDITIONAL == $2->guard_type) && (NULL == $2->guard) && (NULL == $2->callfalse) ) {
                      if( 0 == strcmp(PARSEC_WRITE_MAGIC_NAME, $2->calltrue->func_or_mem) ) {
                          if($1 != JDF_DEP_FLOW_IN) {
                              jdf_fatal(JDF_OBJECT_LINENO($2),
                                        "Automatic data allocation only supported in IN dependencies.\n");
                              YYERROR;
                          }
                      }
                  }

                  $2->properties   = property;
                  d->dep_flags     = $1;
                  d->guard         = $2;
                  d->datatype.type = expr;
                  JDF_OBJECT_LINENO(&d->datatype) = current_lineno;

                  if( NULL != jdf_find_property( $3, "arena_index", &property ) ) {
                      jdf_fatal(current_lineno, "Old construct arena_index used. Please update the code to use type instead.\n");
                      YYERROR;
                  }
                  if( NULL != jdf_find_property( $3, "nb_elt", &property ) ) {
                      jdf_fatal(current_lineno, "Old construct nb_elt used. Please update the code to use count instead.\n");
                      YYERROR;
                  }
                  if( (JDF_STRING == expr->op) || (JDF_VAR == expr->op) ) {
                      /* Special case: [type = SOMETHING] -> define the DAGUE_ARENA_SOMETHING arena index */
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
                   * count and the displacement.
                   */
                  expr = jdf_find_property( d->guard->properties, "layout", &property );
                  if( NULL == expr ) {
                      expr = jdf_find_property( d->guard->properties, "type", &property );
                  }
                  d->datatype.layout = expr;

                  /**
                   * The number of types to transfer.
                   */
                  expr = jdf_find_property( d->guard->properties, "count", &property );
                  if( NULL == expr ) {
                      expr          = new(jdf_expr_t);
                      expr->op      = JDF_CST;
                      expr->jdf_cst = 1;
                  }
                  d->datatype.count = expr;

                  /**
                   * The displacement from the begining of the type.
                   */
                  expr = jdf_find_property( d->guard->properties, "displ", &property );
                  if( NULL == expr ) {
                      expr          = new(jdf_expr_t);
                      expr->op      = JDF_CST;
                      expr->jdf_cst = 0;
                  } else {
                      if( !((JDF_CST != expr->op) && (0 == expr->jdf_cst)) )
                          d->dep_flags |= JDF_DEP_HAS_DISPL;
                  }
                  d->datatype.displ = expr;

                  JDF_OBJECT_LINENO(d) = JDF_OBJECT_LINENO($2);
                  assert( 0 != JDF_OBJECT_LINENO($2) );
                  $$ = d;
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
       |
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = NULL;
                  c->func_or_mem = strdup(PARSEC_WRITE_MAGIC_NAME);
                  c->parameters = NULL;
                  JDF_OBJECT_LINENO(c) = current_lineno;

                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_UNCONDITIONAL;
                  g->guard = NULL;
                  g->calltrue = c;
                  g->callfalse = NULL;
                  $$ = g;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       ;

call:         VAR VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = $1;
                  c->func_or_mem = $2;
                  c->parameters = $4;
                  $$ = c;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($4);
                  assert( 0 != JDF_OBJECT_LINENO($$) );
              }
       |      VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = $1;
                  c->parameters = $3;
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
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = JDF_OBJECT_LINENO($1);
                  assert( 0 != JDF_OBJECT_LINENO($$) );
            }
          | expr_simple
            {
                  $$ = $1;
                  assert( 0 != JDF_OBJECT_LINENO($$) );
            }
          ;

expr_simple:  expr_simple EQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_EQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple NOTEQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOTEQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple LESS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LESS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple LEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MORE expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MORE;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple AND expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_AND;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple OR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_OR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple XOR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_XOR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      NOT expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOT;
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
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MINUS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MINUS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple TIMES expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TIMES;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple DIV expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_DIV;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple MODULO expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MODULO;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple SHL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple SHR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      expr_simple QUESTION_MARK expr_simple COLON expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TERNARY;
                  e->jdf_tat = $1;
                  e->jdf_ta1 = $3;
                  e->jdf_ta2 = $5;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      VAR
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_VAR;
                  e->jdf_var = strdup($1);
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->jdf_cst = $1;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      MINUS INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->jdf_cst = -$2;
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      STRING
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_STRING;
                  e->jdf_var = strdup($1);
                  $$ = e;
                  JDF_OBJECT_LINENO($$) = current_lineno;
              }
       |      EXTERN_DECL
              {
                  jdf_expr_t *ne;
                  $$ = new(jdf_expr_t);
                  $$->op = JDF_C_CODE;
                  $$->jdf_c_code.code = $1;
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

