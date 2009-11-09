%option nounput

%{
/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <string.h>
#include <stdio.h>

extern char *strdup(const char *);
int fileno(FILE *stream);

#include "dplasma.h"
#include "dplasma.tab.h"

int dplasma_lineno = 1;

%}

WHITE         [\f\t\v ]

%x comment
%x body

%%

[bB][oO][dD][yY]([^E]|([eE][^N])|([eE][nN][^D]))+[eE][nN][dD] {
                 int i;
                 yylval.string = malloc(strlen(yytext)-6);
                 strncpy( yylval.string, yytext + 4, strlen(yytext) - 7);
                 for( i = 0; yylval.string[i] != '\0'; i++ ) {
                     if( yylval.string[i] == '\n' )
                         dplasma_lineno++;
                 }
                 return DPLASMA_BODY; }
extern{WHITE}+\"[^\"]+\"{WHITE}+\%\{([^\%]|(\%[^\}]))+\%\}    {
                   int i, j;
                   yylval.string = (char *)malloc(strlen(yytext)-13);
                   /* Put the extern language, then a ", then the body */
                   /* This assumes that no extern language rightfully has a " in its name, which is also an assumption of the lexer anyway */
                   for(i = 0; yytext[i] != '"'; i++) /* nothing */;
                   j = 0;
                   for(i = i+1; yytext[i] != '"'; i++)
                       yylval.string[j++] = yytext[i];
                   yylval.string[j++] = '"';
                   for(i = i+1; yytext[i] != '{'; i++) /* nothing */;
                   memcpy(yylval.string+j, yytext + i + 1, strlen(yytext+i+1)-2);
                   yylval.string[j+strlen(yytext+i+1)-1] = '\0';
                   return DPLASMA_EXTERN_DECL;
                 }
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
"\!="          { return DPLASMA_NOT_EQUAL; }
"="            { return DPLASMA_ASSIGNMENT; }
[\+\-\*/%\\\|\&\^]  { yylval.operand = yytext[0];
                      return DPLASMA_OP; }
"?"            { return DPLASMA_QUESTION; }
":"            { return DPLASMA_COLON; }
","            { return DPLASMA_COMMA; }
".."           { return DPLASMA_RANGE; }
%%

