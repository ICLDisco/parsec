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

extern char *yyfilename;

#include "dplasma.h"
#include "dplasma.tab.h"

int dplasma_lineno = 1;

char *extern_code(char *base, int len)
{
   char *res;
   int nblines;
   int p, rp, sp, rlen;
   
   nblines = 1;
   for(p = 0; p < len; p++) {
      if( base[p] == '\n' )
         nblines++;
   }
   rlen = len + nblines * (16 + strlen(yyfilename));
   res = (char*)calloc(1, rlen);

   rp = 0;
   sp = 0;
   for(p = 0; p < len; p++) {
      if( base[p] == '\n' ) {
         rp += snprintf(res + rp, rlen - rp, "#line %d \"%s\"\n", dplasma_lineno, yyfilename);
	 dplasma_lineno++;
         memcpy(res + rp, base + sp, p-sp);
	 rp += p-sp;
	 res[rp++] = '\n';
	 sp = p+1;
      }
   }
   res[rp] = '\0';
   return res;
}

%}

WHITE         [\f\t\v ]

%x comment
%x body

%%

[bB][oO][dD][yY]([^E]|([eE][^N])|([eE][nN][^D]))+[eE][nN][dD] {
                 yylval.string = extern_code( yytext+4, strlen(yytext) - 7 );
                 return DPLASMA_BODY; }
extern{WHITE}+\"[^\"]+\"{WHITE}+\%\{([^\%]|(\%[^\}]))+\%\}    {
                   int lstart, lend, bstart;
                   /* This assumes that no extern language rightfully has a " in its name, which is also an assumption of the lexer anyway */

                   /* find the beginning of the language */
                   for(lstart = 0; yytext[lstart] != '"'; lstart++) /* nothing */;
                   lstart++;

                   /* find the end of the language */
                   for(lend = lstart + 1; yytext[lend] != '"'; lend++) /* nothing */;
                   yylval.two_strings.language = (char*)calloc(1, lend-lstart+1);
                   memcpy(yylval.two_strings.language, yytext + lstart, lend-lstart);

                   /* Search for the start of the body */
                   for(bstart = lend + 1; yytext[bstart] != '{'; bstart++ ) /* nothing */;
                   bstart++;
                   yylval.two_strings.code = extern_code( yytext + bstart, strlen(yytext)-bstart-2 );
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

