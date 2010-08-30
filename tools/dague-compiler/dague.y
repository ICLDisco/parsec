%{
/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "jdf.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/*
static expr_t *global_expr_stack[MAX_EXPR_STACK_COUNT];
static unsigned int global_expr_stack_size = 0;
*/

extern int yyparse(void);
extern int yylex(void);

extern int current_lineno;

static void yyerror(const char *str)
{
    fprintf(stderr, "parse error at line %d: %s\n", current_lineno, str);
}

int yywrap(void); 

int yywrap(void)
{
    return 1;
}

#define new(type)  (type*)calloc(1, sizeof(type))

static jdf_data_entry_t* jdf_find_or_create_data(jdf_t* jdf, const char* dname)
{
    jdf_data_entry_t* data = jdf->data;

    while( NULL != data ) {
        if( !strcmp(data->dname, dname) ) {
            return data;
        }
        data = data->next;
    }
    /* not found, create */
    data = new(jdf_data_entry_t);
    data->dname    = strdup(dname);
    data->lineno   = -1;
    data->nbparams = -1;
    data->next     = jdf->data;
    jdf->data = data;
    return data;
}

%}

%union {
  int                   number;
  char*                 string;
  jdf_expr_operand_t    expr_op;
  jdf_preamble_entry_t *preamble;
  jdf_global_entry_t   *global;
  jdf_function_entry_t *function;
  jdf_name_list_t      *name_list;
  jdf_flags_t           flags;
  jdf_def_list_t       *def_list;
  jdf_dataflow_list_t  *dataflow_list;
  jdf_dataflow_t       *dataflow;
  jdf_dep_list_t       *dep_list;
  jdf_dep_t            *dep;
  jdf_dep_type_t        dep_type;
  jdf_guarded_call_t   *guarded_call;
  jdf_call_t           *call;
  jdf_expr_list_t      *expr_list;
  jdf_expr_t           *expr;
};

%type <flags>flags_list
%type <flags>flags
%type <function>function
%type <name_list>varlist
%type <def_list>execution_space
%type <call>partitioning
%type <dataflow_list>dataflow_list
%type <dataflow>dataflow
%type <dep_list>dependencies
%type <dep>dependency
%type <guarded_call>guarded_call
%type <string>optional_type
%type <call>call
%type <expr_list>expr_list
%type <expr>expr
%type <expr>priority
%type <number>optional_access_type

%type <string>VAR
%type <string>EXTERN_DECL
%type <string>BODY
%type <dep_type>ARROW
%type <string>OPTIONAL_INFO
%type <number>INT
%type <number>DEPENDENCY_TYPE

%token VAR ASSIGNMENT EXTERN_DECL COMMA OPEN_PAR CLOSE_PAR BODY
%token COLON SEMICOLON DEPENDENCY_TYPE ARROW QUESTION_MARK OPTIONAL_INFO 
%token EQUAL NOTEQUAL LESS LEQ MORE MEQ AND OR XOR NOT INT
%token PLUS MINUS TIMES DIV MODULO SHL SHR RANGE 

%left NOT
%left AND OR XOR
%nonassoc EQUAL NOTEQUAL RANGE QUESTION_MARK COLON
%nonassoc LESS LEQ MORE MEQ
%left MODULO SHL SHR
%left PLUS MINUS
%left DIV
%left TIMES
%left COMMA

%%

jdf:            jdf function
                {
                    $2->next = current_jdf.functions;
                    current_jdf.functions = $2;
                }
        |       jdf VAR optional_type ASSIGNMENT expr 
                {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    e->next = NULL;
                    e->name = $2;
                    e->type = $3;
                    e->expression = $5;
                    e->lineno = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }
                } 
        |       jdf VAR optional_type
                {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    e->next = NULL;
                    e->name = $2;
                    e->type = $3;
                    e->expression = NULL;
                    e->lineno = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }                
                }
        |       jdf EXTERN_DECL 
                {
                    jdf_preamble_entry_t *p, *e = new(jdf_preamble_entry_t);
                    e->next = NULL;
                    e->preamble = $2;
                    e->lineno = current_lineno;
                    if( NULL == current_jdf.preambles ) {
                        current_jdf.preambles = e;
                    } else {
                        for(p = current_jdf.preambles; NULL != p->next; p = p->next)
                            /* nothing */ ;
                        p->next = e;
                    }
                }
        |
        ;

optional_type: 
              OPTIONAL_INFO 
              {
                  $$ = $1;
              }
       |
              {
                  $$ = NULL;
              }
       ;

flags_list:     VAR COMMA flags_list
                {
                    if( 0 == strcasecmp("high_priority", $1) ) {
                        $$ = ($3 | JDF_FUNCTION_FLAG_HIGH_PRIORITY);
                    } else {
                        printf( "Ignore unknown DAGuE flag %s.\n", $1 );
                        $$ = $3;
                    }
                }
        |       VAR
                {
                    if( 0 == strcasecmp("high_priority", $1) ) {
                        $$ = JDF_FUNCTION_FLAG_HIGH_PRIORITY;
                    } else {
                        printf( "Ignore unknown DAGuE flag %s.\n", $1 );
                        $$ = 0;
                    }
                }
        ;

flags:          OPEN_PAR flags_list CLOSE_PAR
                {
                    $$ = $2;
                }
        |
                {
                    $$ = 0;
                }
        ;

function:       VAR OPEN_PAR varlist CLOSE_PAR flags execution_space partitioning dataflow_list priority BODY
                {
                    jdf_function_entry_t *e = new(jdf_function_entry_t);
                    e->fname = $1;
                    e->parameters = $3;
                    e->flags = $5;
                    e->definitions = $6;
                    e->predicate = $7;
                    e->dataflow = $8;
                    e->priority = $9;
                    e->body = $10;

                    e->lineno  = current_lineno;

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
                VAR ASSIGNMENT expr execution_space
                {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name   = $1;
                    l->expr   = $3;
                    l->lineno = current_lineno;
                    l->next   = $4;

                    $$ = l;
                }
         |      VAR ASSIGNMENT expr 
                {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name   = $1;
                    l->expr   = $3;
                    l->lineno = current_lineno;
                    l->next   = NULL;

                    $$ = l;
                }
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
                  JDF_COUNT_LIST_ENTRIES($4, jdf_expr_list_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    $2, nbparams, current_lineno, data->nbparams, data->lineno);
                          YYERROR;
                      }
                  } else {
                      data->nbparams = nbparams;
                      data->lineno = current_lineno;
                  }
                  $$ = c;                  
              }
         ;

dataflow_list:  dataflow dataflow_list 
                {
                    jdf_dataflow_list_t *l = new(jdf_dataflow_list_t);
                    l->flow = $1;
                    l->next = $2;
                    $$ = l;
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
                    flow->access_type = $1;
                    flow->varname     = $2;
                    flow->deps        = $3;
                    flow->lineno      = current_lineno;

                    $$ = flow;
                }
        ;

dependencies:  dependency dependencies
               {
                   jdf_dep_list_t *l = new(jdf_dep_list_t);
                   l->next = $2;
                   l->dep = $1;
                   $$ = l;
               }
        | 
               {
                   $$ = NULL;
               }
       ;

dependency:   ARROW guarded_call optional_type 
              {
                  jdf_dep_t *d = new(jdf_dep_t);
                  d->type = $1;
                  d->guard = $2;
                  d->datatype = $3;
                  d->lineno = current_lineno;
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
       |      expr QUESTION_MARK call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = NULL;
                  $$ = g;
              }
       |      expr QUESTION_MARK call COLON call
              {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = $1;
                  g->calltrue = $3;
                  g->callfalse = $5;
                  $$ = g;
              }
       ;

call:         VAR VAR OPEN_PAR expr_list CLOSE_PAR
              {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = $1;
                  c->func_or_mem = $2;
                  c->parameters = $4;
                  $$ = c;
              }
       |      VAR OPEN_PAR expr_list CLOSE_PAR
              {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = $1;
                  c->parameters = $3;
                  $$ = c;                  
                  data = jdf_find_or_create_data(&current_jdf, $1);
                  JDF_COUNT_LIST_ENTRIES($3, jdf_expr_list_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    $1, nbparams, current_lineno, data->nbparams, data->lineno);
                          YYERROR;
                      }
                  } else {
                      data->nbparams = nbparams;
                      data->lineno = current_lineno;
                  }
              }
       ;

priority:     SEMICOLON expr
              {
                    $$ = $2;
              }
       |      { $$ = NULL; }
       ;

/* And now, the expressions */

expr_list:    expr COMMA expr_list
              {
                  jdf_expr_list_t *l = new(jdf_expr_list_t);
                  l->next = $3;
                  l->expr = $1;
                  $$=l;
              }
      |       expr
              {
                  jdf_expr_list_t *l = new(jdf_expr_list_t);
                  l->next = NULL;
                  l->expr = $1;
                  $$=l;
              }
      ;

expr:         expr EQUAL expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_EQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr NOTEQUAL expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOTEQUAL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr LESS expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LESS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr LEQ expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr MORE expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MORE;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr MEQ expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MEQ;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr AND expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_AND;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr OR expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_OR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr XOR expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_XOR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      NOT expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOT;
                  e->jdf_ua = $2;
                  $$ = e;
              }      
       |      OPEN_PAR expr CLOSE_PAR
              {
                  $$ = $2;
              }
       |      expr PLUS expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_PLUS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr MINUS expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MINUS;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr TIMES expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TIMES;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr DIV expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_DIV;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr MODULO expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MODULO;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr SHL expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHL;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr SHR expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHR;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr RANGE expr
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ba1 = $1;
                  e->jdf_ba2 = $3;
                  $$ = e;
              }
       |      expr QUESTION_MARK expr COLON expr
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
                  e->jdf_var = $1;
                  $$ = e;
              }
       |      INT
              {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->jdf_cst = $1;
                  $$ = e;
              }
       ;

%%

