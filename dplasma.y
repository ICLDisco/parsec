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
                printf( "Parse %s\n%s\n", global_dplasma->name,
                        global_dplasma->body );
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
                                            global_varlist_index++;
                                        }
                                     } varlist
         | DPLASMA_VAR {
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
                              global_varlist_index++;
                          }
                       }
         |
;

execution_space: assignment execution_space
         | assignment
;

assignment: DPLASMA_VAR DPLASMA_ASSIGNMENT expr
;

partitioning: DPLASMA_COLON expr partitioning
         | DPLASMA_COLON expr
;

params: param params
         | param
;

param: DPLASMA_DEPENDENCY_TYPE DPLASMA_VAR dependencies
;

dependencies: DPLASMA_ARROW dependency dependencies
        | DPLASMA_ARROW dependency
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

expr: DPLASMA_VAR                                    { $$ = NULL; }
        | DPLASMA_INT                                { $$ = NULL; }
        | expr DPLASMA_OP expr                       { $$ = NULL; }
        | DPLASMA_OPEN_PAR expr DPLASMA_CLOSE_PAR    { $$ = NULL; }
        | expr DPLASMA_EQUAL expr                    { $$ = NULL; }
        | expr DPLASMA_RANGE expr                    { $$ = NULL; }
;

%%

