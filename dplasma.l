%{
/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <string.h>
#include "dplasma.h"
#include "y.tab.h"

int dplasma_lineno = 1;

%}

WHITE         [\f\t\v ]

%x comment
%x body

%%

[bB][oO][dD][yY]([^;]|\n)*;\n*[eE][nN][dD] {
                 int i;
                 yylval.string = malloc(strlen(yytext)-6);
                 strncpy( yylval.string, yytext + 4, strlen(yytext) - 7);
                 for( i = 0; yylval.string[i] != '\0'; i++ ) {
                     if( yylval.string[i] == '\n' )
                         dplasma_lineno++;
                 }
                 return DPLASMA_BODY; }
{WHITE}*\n     { dplasma_lineno++; }
"//".*\n       { dplasma_lineno++; }
"/*"           { BEGIN(comment); }
<comment>[^*\n]*  ;  /* Eat up non '*'s */
<comment>"*"+[^*/\n]* ;  /* Eat '*'s not followed by a '/' */
<comment>\n    { dplasma_lineno++; }
<comment>"*"+"/" { BEGIN(INITIAL);  /* Done with the BLOCK comment */ }
{WHITE}+       ;  /* Eat multiple white-spaces */
[0-9]+           { yylval.number = atol(yytext);
                   return DPLASMA_INT; }

INOUT          { yylval.operand = (char)SYM_INOUT;
                 return DPLASMA_DEPENDENCY_TYPE; }
IN             { yylval.operand = (char)SYM_IN;
                 return DPLASMA_DEPENDENCY_TYPE; }
OUT            { yylval.operand = (char)SYM_OUT;
                 return DPLASMA_DEPENDENCY_TYPE; }

"->"           { yylval.operand = '>';
                 return DPLASMA_ARROW; }
"<-"           { yylval.operand = '<';
                 return DPLASMA_ARROW; }
[a-zA-Z]+[a-zA-Z0-9]* { yylval.string = strdup(yytext);
                        return DPLASMA_VAR; }
"("            { return DPLASMA_OPEN_PAR; }
")"            { return DPLASMA_CLOSE_PAR; }
"=="           { return DPLASMA_EQUAL; }
"="            { return DPLASMA_ASSIGNMENT; }
[\+\-\*/%]        { yylval.operand = yytext[0];
                 return DPLASMA_OP; }
"?"            { return DPLASMA_QUESTION; }
":"            { return DPLASMA_COLON; }
","            { return DPLASMA_COMMA; }
".."           { return DPLASMA_RANGE; }
%%

