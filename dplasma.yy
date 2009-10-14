%{
#include "dplasma.h"
#include <stdio.h>
#include <string.h>

void yyerror(const char *str)
{
	fprintf(stderr, "error: %s\n", str);
}

int yywrap()
{
	return 1;
}

static expr_t *e = NULL;

int main(int argc, char *argv[])
{
	yyparse();

	expr_dump(e);
	return 0;
}
%}

%union
{
	int     number;
	char   *string;
	expr_t *e;
}

%token PLUS EQUAL MODULO
%token <number> INTEGER
%token <string> IDENTIFIER
%type  <e> expr

prog: expr
{
	e = $1;
}
;

expr:
	IDENTIFIER
{
	$$=(expr_t*)malloc(sizeof(expr_t));
	$$->op = EXPR_OP_SYMB;
	$$->var = $1;
}
|	NUMBER
{
	$$=(expr_t*)malloc(sizeof(expr_t));
	$$->op = EXPR_OP_CONST_INT;
	$$->const_int = $1;
}
|	expr PLUS expr
{
	$$=(expr_t*)malloc(sizeof(expr_t));
	$$->op = EXPR_OP_BINARY_PLUS;
	$$->bop1 = $1;
	$$->bop2 = $3;
}
|	expr EQUAL expr
{
	$$=(expr_t*)malloc(sizeof(expr_t));
	$$->op = EXPR_OP_BINARY_EQUAL;
	$$->bop1 = $1;
	$$->bop2 = $3;
}
|	expr MODULO expr
{
	$$=(expr_t*)malloc(sizeof(expr_t));
	$$->op = EXPR_OP_BINARY_MODULO;
	$$->bop1 = $1;
	$$->bop2 = $3;
}
;
