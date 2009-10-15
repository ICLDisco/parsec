%{
/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dplasma.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct symbol_stack_elt {
    symbol_t                *sym;
    struct symbol_stack_elt *next;
} symbol_stack_elt_t;

static symbol_stack_elt_t *dplasma_symbol_stack = NULL;

static dplasma_t* global_dplasma = NULL;
static int global_varlist_index = 0;
extern int dplasma_lineno;

extern int yyparse(void);
extern int yylex(void);

static void yyerror(const char *str)
{
    fprintf(stderr, "parse error at line %d: %s\n", dplasma_lineno, str);
}

int yywrap()
{
	return 1;
}

int main(int argc, char *argv[])
{
    dplasma_lineno = 1;
	yyparse();

	return 0;
}
%}

%union
{
    int        number;
    char*      string;
    char       operand;
    expr_t*    expr;
    dplasma_t* dplasma;
}

%token DPLASMA_COMMA DPLASMA_OPEN_PAR DPLASMA_CLOSE_PAR DPLASMA_RANGE
%token DPLASMA_EQUAL  DPLASMA_ASSIGNMENT DPLASMA_QUESTION DPLASMA_COLON
%token <number>  DPLASMA_INT
%token <string>  DPLASMA_VAR
%token <string>  DPLASMA_BODY
%token <operand> DPLASMA_OP
%token <operand> DPLASMA_DEPENDENCY_TYPE
%token <operand> DPLASMA_ARROW

%type  <expr>    expr

%nonassoc DPLASMA_ASSIGNMENT
%nonassoc DPLASMA_RANGE
%left DPLASMA_EQUAL
%left DPLASMA_OP

%%

prog:
    dplasma {
               dplasma_dump(global_dplasma, "");
            } prog
    |
;

dplasma:
     DPLASMA_VAR {
                     global_dplasma = (dplasma_t*)calloc(1, sizeof(dplasma_t));
                     global_dplasma->name = $1;
                     global_varlist_index = 0;
                 }
     DPLASMA_OPEN_PAR varlist DPLASMA_CLOSE_PAR
     execution_space
     partitioning
     params
     DPLASMA_BODY
                {
                    global_dplasma->body = $9;
                }
;

varlist:   DPLASMA_VAR DPLASMA_COMMA {
                                        symbol_stack_elt_t *s;
                                        if( global_varlist_index == MAX_LOCAL_COUNT ) {
                                            fprintf(stderr,
                                                    "Internal Error while parsing at line %d:\n"
                                                    "  Maximal variable list count reached: %d (I told you guys this will happen)\n",
                                                    dplasma_lineno,
                                                    global_varlist_index);
                                            YYERROR;
                                        } else {
                                            global_dplasma->locals[global_varlist_index] = (symbol_t*)calloc(1, sizeof(symbol_t));
                                            global_dplasma->locals[global_varlist_index]->name = $1;
                                            
                                            s = (symbol_stack_elt_t*)calloc(1, sizeof(symbol_stack_elt_t));
                                            s->sym = global_dplasma->locals[global_varlist_index];
                                            s->next = dplasma_symbol_stack;
                                            dplasma_symbol_stack = s;

                                            global_varlist_index++;
                                        }
                                     } varlist
         | DPLASMA_VAR {
                          symbol_stack_elt_t *s;
                          if( global_varlist_index == MAX_LOCAL_COUNT ) {
                               fprintf(stderr,
                                       "Internal Error while parsing at line %d:\n"
                                       "  Maximal variable list count reached: %d (I told you guys this will happen)\n",
                                       dplasma_lineno,
                                       global_varlist_index);
                               YYERROR;
                          } else {
                              global_dplasma->locals[global_varlist_index] = (symbol_t*)calloc(1, sizeof(symbol_t));
                              global_dplasma->locals[global_varlist_index]->name = $1;

                              s = (symbol_stack_elt_t*)calloc(1, sizeof(symbol_stack_elt_t));
                              s->sym = global_dplasma->locals[global_varlist_index];
                              s->next = dplasma_symbol_stack;
                              dplasma_symbol_stack = s;

                              global_varlist_index++;
                          }
                       }
         |
;

execution_space: assignment execution_space
         | 
;

assignment: DPLASMA_VAR DPLASMA_ASSIGNMENT expr {
                                                    int i;
                                                    for(i = 0; NULL != global_dplasma->locals[i] && i < MAX_LOCAL_COUNT; i++) {
                                                        if( strcmp(global_dplasma->locals[i]->name, $1) ) {
                                                            continue;
                                                        }
                                                        break;
                                                    }
                                                    if( i == MAX_LOCAL_COUNT ) {
                                                        fprintf(stderr,
                                                                "Parse Error at line %d:\n"
                                                                "  '%s' is an unbound variable\n",
                                                                dplasma_lineno,
                                                                $1);
                                                        YYERROR;
                                                    }
                                                    if( EXPR_OP_BINARY_RANGE == $3->op ) {
                                                        global_dplasma->locals[i]->min = $3->bop1;
                                                        global_dplasma->locals[i]->max = $3->bop2;
                                                        free($3);
                                                    } else {
                                                        global_dplasma->locals[i]->min = $3;
                                                        global_dplasma->locals[i]->max = $3;
                                                    }
                                                }
;

partitioning: DPLASMA_COLON expr partitioning
         | 
;

params: param params
         |
;

param: DPLASMA_DEPENDENCY_TYPE DPLASMA_VAR dependencies
;

dependencies: DPLASMA_ARROW dependency dependencies
        | 
;

dependency: call
        | expr DPLASMA_QUESTION dependency DPLASMA_COLON dependency
;

call: DPLASMA_VAR DPLASMA_VAR DPLASMA_OPEN_PAR expr_list DPLASMA_CLOSE_PAR
        | DPLASMA_DEPENDENCY_TYPE DPLASMA_OPEN_PAR expr_list DPLASMA_CLOSE_PAR
;

expr_list: expr DPLASMA_COMMA expr_list
        | expr
;

expr:     DPLASMA_VAR                                { 
                                                         symbol_stack_elt_t *s = dplasma_symbol_stack;
                                                         symbol_t           *unknown;
                                                         
                                                         while( NULL != s && strcmp(s->sym->name, $1) ) {
                                                             s = s->next;
                                                         }
                                                         if( NULL == s ) {
                                                             unknown = (symbol_t*)calloc(1, sizeof(symbol_t));
                                                             unknown->name = strdup($1);
                                                             s = (symbol_stack_elt_t*)calloc(1, sizeof(symbol_stack_elt_t));
                                                             s->sym = unknown;
                                                             s->next = dplasma_symbol_stack;
                                                             dplasma_symbol_stack = s;
                                                         }
                                                         $$ = expr_new_var(s->sym);
                                                         free($1);
                                                     }
        | DPLASMA_INT                                { $$ = expr_new_int($1); }
        | expr DPLASMA_OP expr                       { $$ = expr_new_binary($1, $2, $3); }
        | DPLASMA_OPEN_PAR expr DPLASMA_CLOSE_PAR    { $$ = $2; }
        | expr DPLASMA_EQUAL expr                    { $$ = expr_new_binary($1, '=', $3); }
        | expr DPLASMA_RANGE expr                    { $$ = expr_new_binary($1, '.', $3);; }
;

%%

