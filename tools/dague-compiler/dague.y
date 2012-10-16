%{
/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "jdf.h"

/**
 * Better error handling
 */
#define YYERROR_VERBOSE 1
struct yyscan_t;

#include "dague.y.h"

extern int current_lineno;

static jdf_expr_t *inline_c_functions = NULL;

static void yyerror(YYLTYPE *locp,
                    struct yyscan_t* yyscanner,
                    char const *msg)
{
    fprintf(stderr, "parse error at %d.%d-%d.%d: %s\n",
            locp->first_line, locp->first_column,
            locp->last_line, locp->last_column,
            msg);
}

#define new(type)  (type*)calloc(1, sizeof(type))

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
    jdf_dep_type_t        dep_type;
    jdf_guarded_call_t   *guarded_call;
    jdf_call_t           *call;
    jdf_expr_t           *expr;
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
%type <property>properties
%type <property>properties_list
%type <call>call
%type <expr>expr_list
%type <expr>expr_list_range
%type <expr>expr_complete
%type <expr>expr_range
%type <expr>expr_simple
%type <expr>priority
%type <expr>simulation_cost
%type <number>optional_access_type
%type <external_code>prologue
%type <external_code>epilogue

%type <string>VAR
%type <string>EXTERN_DECL
%type <string>BODY
%type <dep_type>ARROW
%type <number>PROPERTIES_ON
%type <number>PROPERTIES_OFF
%type <string>STRING
%type <number>INT
%type <number>DEPENDENCY_TYPE

%token VAR ASSIGNMENT EXTERN_DECL COMMA OPEN_PAR CLOSE_PAR BODY STRING SIMCOST
%token COLON SEMICOLON DEPENDENCY_TYPE ARROW QUESTION_MARK PROPERTIES_ON PROPERTIES_OFF
%token EQUAL NOTEQUAL LESS LEQ MORE MEQ AND OR XOR NOT INT
%token PLUS MINUS TIMES DIV MODULO SHL SHR RANGE

%nonassoc EQUAL NOTEQUAL RANGE QUESTION_MARK COLON
%nonassoc LESS LEQ MORE MEQ
%right NOT
%left AND OR XOR
%left MODULO SHL SHR
%left PLUS MINUS
%left TIMES DIV
%left COMMA

%debug
%defines
%locations
%pure-parser
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
                    jdf_expr_t *el, *pl;

                    if( NULL == current_jdf.functions ) {
                        $2->function_id = 0;
                    } else {
                        $2->function_id = current_jdf.functions->function_id + 1;
                    }
                    $2->next = current_jdf.functions;
                    current_jdf.functions = $2;
                    if( NULL != inline_c_functions) {
                        /* Every inline functions declared here where within the context of $2 */
                        for(el = inline_c_functions; NULL != el; el = el->next) {
                            pl = el;
                            el->jdf_c_code.function_context = $2;
                        }
                        pl->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
        |       jdf VAR ASSIGNMENT expr_complete properties
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
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
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
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
        | jdf properties
                {
                    jdf_def_list_t *p;
                    if( NULL != $2 ) {
                        for(p = $2; p->next != NULL; p = p->next) /*nothing*/ ;
                        p->next = current_jdf.global_properties;
                        current_jdf.global_properties = $2;
                    }
                }
        |
                {
                    jdf_expr_t *el;
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
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

properties_list: VAR ASSIGNMENT expr_complete properties_list
              {
                 jdf_def_list_t* assign = new(jdf_def_list_t);
                 assign->next              = $4;
                 assign->name              = strdup($1);
                 assign->expr              = $3;
                 JDF_OBJECT_LINENO(assign) = current_lineno;
                 $$ = assign;
              }
       | VAR ASSIGNMENT expr_complete
             {
                 jdf_def_list_t* assign = new(jdf_def_list_t);
                 assign->next              = NULL;
                 assign->name              = strdup($1);
                 assign->expr              = $3;
                 JDF_OBJECT_LINENO(assign) = current_lineno;
                 $$ = assign;
             }
       ;

function:       VAR OPEN_PAR varlist CLOSE_PAR properties execution_space simulation_cost partitioning dataflow_list priority BODY
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
                    e->body              = $11;

                    JDF_OBJECT_LINENO(e) = current_lineno;

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
                    l->name              = $1;
                    l->expr              = $3;
                    JDF_OBJECT_LINENO(l) = current_lineno;
                    l->properties        = $4;
                    l->next              = $5;
                    $$ = l;
                }
         |      VAR ASSIGNMENT expr_range properties
                {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name              = $1;
                    l->expr              = $3;
                    JDF_OBJECT_LINENO(l) = current_lineno;
                    l->properties        = $4;
                    l->next              = NULL;
                    $$ = l;
                }
         ;

simulation_cost:
                SIMCOST expr_complete
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
                      JDF_OBJECT_LINENO(data) = current_lineno;
                  }
                  $$ = c;
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

optional_access_type :
                DEPENDENCY_TYPE
                {
                    $$ = $1;
                }
         |      { $$ = JDF_VAR_TYPE_READ | JDF_VAR_TYPE_WRITE; }
         ;

dataflow:       optional_access_type VAR dependencies
                {
                    jdf_dataflow_t *flow = new(jdf_dataflow_t);
                    flow->access_type       = $1;
                    flow->varname           = $2;
                    flow->deps              = $3;
                    JDF_OBJECT_LINENO(flow) = current_lineno;

                    $$ = flow;
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
                  jdf_expr_t *expr_simple, *expr_complex;
                  jdf_def_list_t* property;

                  d->type = $1;
                  d->guard = $2;

                  expr_simple = jdf_find_property( $3, "type", &property );
                  expr_complex = jdf_find_property( $3, "arena_index", &property );

                  /* If neither is defined, we define the old simple DEFAULT arena */
                  property = NULL;
                  if( NULL == expr_simple && NULL == expr_complex ) {
                      property = jdf_create_properties_list( "type", 0, "DEFAULT", NULL );
                      expr_simple = jdf_find_property( property, "type", &property );
                  }
                  if( NULL == property )
                      property = $3;
                  else
                      property->next = $3;
                  $2->properties = property;

                  if( NULL != expr_simple ) {
                      if( NULL != expr_complex ) {
                          jdf_fatal(current_lineno, "Dependency defined with both properties type and arena_index\n");
                          YYERROR;
                      }
                      /* Simple old type */
                      if( JDF_STRING == expr_simple->op ||
                          JDF_VAR == expr_simple->op ) {
                          /* Old way: [type = SOMETHING] -> define the DAGUE_ARENA_SOMETHING arena index */
                          d->datatype.simple = 1;
                          for(prec = NULL, g = current_jdf.datatypes; g != NULL; g = g->next) {
                              if( 0 == strcmp(expr_simple->jdf_var, g->name) ) {
                                  break;
                              }
                              prec = g;
                          }
                          if( NULL == g ) {
                              e = new(struct jdf_name_list);
                              e->name = strdup(expr_simple->jdf_var);
                              e->next = NULL;
                              if( NULL != prec ) {
                                  prec->next = e;
                              } else {
                                  current_jdf.datatypes = e;
                              }
                          }
                          d->datatype.u.simple_name = strdup(expr_simple->jdf_var);
                      } else {
                          /* Let's allow [type = inline_c %{ ... %}], even if it should really be [arena_index = inline_c %{ ... %} ] */
                          expr_complex = expr_simple;
                          expr_simple = NULL;
                      }
                  }
                  if( NULL != expr_complex ) {
                      assert(NULL == expr_simple);
                      d->datatype.simple = 0;
                      d->datatype.u.complex_expr = expr_complex;
                  }

                  expr_simple = jdf_find_property( d->guard->properties, "nb_elt", &property );
                  if( NULL == expr_simple ) {
                      d->datatype.nb_elt = new(jdf_expr_t);
                      d->datatype.nb_elt->op = JDF_CST;
                      d->datatype.nb_elt->jdf_cst = 1;
                  } else {
                      d->datatype.nb_elt = expr_simple;
                  }
                  JDF_OBJECT_LINENO(d) = current_lineno;

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
              }
       |      expr_simple QUESTION_MARK call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = NULL;
                  $$ = g;
              }
       |      expr_simple QUESTION_MARK call COLON call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = $5;
                  $$ = g;
              }
       |      expr_complete QUESTION_MARK call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = NULL;
                  $$ = g;
              }
       |      expr_complete QUESTION_MARK call COLON call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = $5;
                  $$ = g;
              }
       ;

call:         VAR VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = $1;
                  c->func_or_mem = $2;
                  c->parameters = $4;
                  $$ = c;
              }
       |      VAR OPEN_PAR expr_list_range CLOSE_PAR
              {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = $1;
                  c->parameters = $3;
                  $$ = c;
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
                      JDF_OBJECT_LINENO(data) = current_lineno;
                  }
              }
       ;

priority:     SEMICOLON expr_complete
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
              }
      |       expr_range
              {
                  $1->next = NULL;
                  $$=$1;
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

expr_range: expr_complete RANGE expr_complete
            {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ta1 = $1;               /* from */
                  e->jdf_ta2 = $3;               /* to */
                  e->jdf_ta3 = new(jdf_expr_t);  /* step */
                  e->jdf_ta3->op = JDF_CST;
                  e->jdf_ta3->jdf_cst = 1;
                  $$ = e;
            }
          | expr_complete RANGE expr_complete RANGE expr_complete
            {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ta1 = $1;  /* from */
                  e->jdf_ta2 = $3;  /* to */
                  e->jdf_ta3 = $5;  /* step */
                  $$ = e;
            }
          | expr_complete
            {
                  $$ = $1;
            }
          ;

expr_complete: expr_simple
               {
                   $$ = $1;
               }
      |        EXTERN_DECL
               {
                   jdf_expr_t *ne;
                   $$ = new(jdf_expr_t);
                   $$->op = JDF_C_CODE;
                   $$->jdf_c_code.code = $1;
                   $$->jdf_c_code.lineno = current_lineno;
                   /* This will  be set by the upper level parsing if necessary */
                   $$->jdf_c_code.function_context = NULL;
                   $$->jdf_c_code.fname = NULL;

                   $$->next = inline_c_functions;
                   inline_c_functions = $$;
               }
     ;

expr_simple:  expr_simple EQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_EQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple NOTEQUAL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOTEQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple LESS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LESS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple LEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple MORE expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MORE;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple MEQ expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple AND expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_AND;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple OR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_OR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple XOR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_XOR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      NOT expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOT;
                  e->jdf_ua = $2;
                  $$ = e;
              }
       |      OPEN_PAR expr_simple CLOSE_PAR
              {
                  $$ = $2;
              }
       |      expr_simple PLUS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_PLUS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple MINUS expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MINUS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple TIMES expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TIMES;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple DIV expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_DIV;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple MODULO expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MODULO;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple SHL expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple SHR expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr_simple QUESTION_MARK expr_simple COLON expr_simple
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TERNARY;
                  e->jdf_tat = $1;
                  e->jdf_ta1 = $3;
                  e->jdf_ta2 = $5;
                  $$ = e;
              }
       |      VAR
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_VAR;
                  e->jdf_var = strdup($1);
                  $$ = e;
              }
       |      INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->jdf_cst = $1;
                  $$ = e;
              }
       |      STRING
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_STRING;
                  e->jdf_var = strdup($1);
                  $$ = e;
              }
       ;

%%

