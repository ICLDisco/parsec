%{
#include <std.h>
#include <string.h>
#include "y.tab.h"
%}
%%
\+			return PLUS;
==			return EQUAL;
\%			return MODULO;
[0-9]+			yylval.number=strtol(yytext, NULL, 0); return NUMBER;
[a-zA-Z_][a-ZA-Z0-9_]*	yylval.string=strdup(yytext);          return IDENTIFIER;
\n			/* ignore end of line */;
[ \t]+			/* ignore whitespace */;
%%
