%option nounput

%{
/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "jdf.h"
#include "dague.tab.h"

extern char *yyfilename;
extern int current_lineno;

char *extern_code(char *base, int len);
char *extern_code(char *base, int len)
{
   char *res;
   int rp, rlen;
   
   rlen = len + 16 + strlen(yyfilename);
   res = (char*)calloc(1, rlen);

   rp = 0;
   if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
     /**
      * As each BODY in contiguous we only have to set the #line once.
      */
     rp += snprintf(res + rp, rlen - rp, "#line %d \"%s\"\n", current_lineno, yyfilename);
  }
   memcpy(res + rp, base, len);
   rp += len;
   res[rp] = '\0';
   for(rp = 0; rp < len; rp++) {
     if( base[rp] == '\n' ) {
       current_lineno++;
     }
   }
   return res;
}

%}

WHITE         [\f\t\v ]

%x comment
%x body

%%

[bB][oO][dD][yY]{WHITE}*"\n"([^E]|([eE][^N])|([eE][nN][^D]))+[eE][nN][dD]                       {
                          char *b = yytext + 4;
                          while( *b != '\n' ) b++;
                          yylval.string = extern_code( b+1, strlen(b+1) - 3 );
                          current_lineno++;
                          return BODY; 
                      }
extern{WHITE}+\"[^\"]+\"{WHITE}+\%\{{WHITE}*"\n"([^\%]|(\%[^\}]))+\%\}                          {
                          int lstart, lend, bstart, skip_line = 0;
                          /* This assumes that no extern language rightfully has a " 
                           * in its name, which is also an assumption of the lexer anyway */
                          
                          /* find the beginning of the language */
                          for(lstart = 0; yytext[lstart] != '"'; lstart++) /* nothing */;
                          lstart++;
                          
                          /* find the end of the language */
                          for(lend = lstart + 1; yytext[lend] != '"'; lend++) /* nothing */;
                          assert( (yytext[lstart] == 'C') && ((lend-lstart) == 1) );
                          
                          /* Search for the start of the body */
                          for(bstart = lend + 1; yytext[bstart] != '{'; bstart++ ) /* nothing */;
                          bstart++;
                          while( yytext[bstart] != '\n' ) bstart++;
                          bstart++;
                          for( lend = 0; lend <= bstart; lend++ )
                              if( yytext[lend] == '\n' ) skip_line++;
                          current_lineno += skip_line;
                          yylval.string = extern_code( yytext + bstart, strlen(yytext)-bstart-2 );
                          current_lineno++;
                          return EXTERN_DECL;
                      }
{WHITE}*\n            { current_lineno++;                                  }
"//".*\n              { current_lineno++;                                  }
"/*"                  { BEGIN(comment);                                    }
<comment>[^*\n]*      {  /* Eat up non '*'s */                             } 
<comment>"*"+[^*/\n]* {  /* Eat '*'s not followed by a '/' */              }
<comment>\n           { current_lineno++;                                  }
<comment>"*"+"/"      { BEGIN(INITIAL);  /* Done with the BLOCK comment */ }
{WHITE}+              {  /* Eat multiple white-spaces */                   }
[0-9]+                { yylval.number = atol(yytext);
                        return INT;                                        }
RW                    { yylval.number = JDF_VAR_TYPE_READ | JDF_VAR_TYPE_WRITE;
                        return DEPENDENCY_TYPE;                            }
READ                  { yylval.number = JDF_VAR_TYPE_READ;
                        return DEPENDENCY_TYPE;                            }
WRITE                 { yylval.number = JDF_VAR_TYPE_WRITE;
                        return DEPENDENCY_TYPE;                            }
"->"                  { yylval.dep_type = JDF_DEP_TYPE_OUT;
                        return ARROW;                                      }
"<-"                  { yylval.dep_type = JDF_DEP_TYPE_IN;
                        return ARROW;                                      }
[a-zA-Z_]+[a-zA-Z0-9_]*                                                                         {
                        yylval.string = strdup(yytext);
                        return VAR;                                        }
"("                   { return OPEN_PAR;                                   }
")"                   { return CLOSE_PAR;                                  }
\[.*\]                { 
                        yylval.string = strdup(yytext + 1);
                        yylval.string[ strlen(yylval.string) - 1 ] = '\0';
                        return OPTIONAL_INFO;                              }
"=="                  { return EQUAL;                                      }
"\!="                 { return NOTEQUAL;                                   }
"&"                   { return AND;                                        }
"|"                   { return OR;                                         }
"^"                   { return XOR;                                        }
"<"                   { return LESS;                                       }
"<="                  { return LEQ;                                        }
">"                   { return MORE;                                       }
">="                  { return MEQ;                                        }
"\!"                  { return NOT;                                        }
"\+"                  { return PLUS;                                       }
"-"                   { return MINUS;                                      }
"\*"                  { return TIMES;                                      }
"/"                   { return DIV;                                        }
"%"                   { return MODULO;                                     }
"<<"                  { return SHL;                                        }
">>"                  { return SHR;                                        }
".."                  { return RANGE;                                      }
"="                   { return ASSIGNMENT;                                 }
"?"                   { return QUESTION_MARK;                              }
":"                   { return COLON;                                      }
";"                   { return SEMICOLON;                                  }
","                   { return COMMA;                                      }

%%
