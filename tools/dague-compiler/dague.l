%{
/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "jdf.h"
#include "dague.y.h"

extern char *yyfilename;
extern int current_lineno;


static char *extern_code(char *base, int len)
{
   char *res;
   int rp, rlen;

   rlen = len + 16 + strlen(yyfilename);
   res = (char*)calloc(1, rlen);

   rp = 0;
   if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
     /**
      * As each BODY is contiguous we only have to set the #line once.
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

#define MAX_STR_CONST  1024

char string_buf[MAX_STR_CONST];
char *string_buf_ptr;
extern YYSTYPE yylval;

#if defined(HAVE_RECENT_LEX)
#define YY_USER_ACTION yylloc->first_line = yylloc->last_line = yylineno; \
    yylloc->first_column = yycolumn; yylloc->last_column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;
#define YYLVAL(field)         yylval->field
#define YY_PUSH_STATE(STATE)  yy_push_state(STATE, yyscanner)
#define YY_POP_STATE()        yy_pop_state(yyscanner)
#else
#define YYLVAL(field)         yylval.field
#define YY_PUSH_STATE(STATE)  yy_push_state(STATE)
#define YY_POP_STATE()        yy_pop_state()
typedef void* yyscan_t;
#endif  /* defined(HAVE_RECENT_LEX) */

%}
/*%option debug*/
/*%option bison-bridge bison-locations*/
%option noyywrap yylineno
/*%option reentrant*/
%option stack

WHITE         [\f\t\v ]

%x comment
%x properties
%x dstr

%%

[bB][oO][dD][yY]{WHITE}*"\n"([^E]|([eE][^N])|([eE][nN][^D]))+[eE][nN][dD]                       {
                          char *b = yytext + 4;
                          while( *b != '\n' ) b++;
                          YYLVAL(string) = extern_code( b+1, strlen(b+1) - 3 );
                          current_lineno++;
                          return BODY;
                      }
extern{WHITE}+\"[^\"]+\"{WHITE}+\%\{{WHITE}*"\n"([^\%]|(\%[^\}]))*\%\}                          {
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
                          YYLVAL(string) = extern_code( yytext + bstart, strlen(yytext)-bstart-2 );
                          return EXTERN_DECL;
                      }
<INITIAL,properties>inline_c{WHITE}*\%\{([^\%]|(\%[^\}]))+\%\}            {
                          int start;
                          for(start = 8; yytext[start] != '{'; start++) /* nothing */ ;
			  start++;
                          YYLVAL(string) = extern_code( yytext + start, strlen(yytext)-start-2 );
			  return EXTERN_DECL;
                      }
SIMCOST               { return SIMCOST; }
<*>"/*"               { YY_PUSH_STATE(comment);                            }
<comment>[^*\n]*      {  /* Eat up non '*'s */                             }
<comment>"*"+[^*/\n]* {  /* Eat '*'s not followed by a '/' */              }
<comment>"*"+"/"      { YY_POP_STATE();  /* Done with the comment BLOCK */ }
<*>{WHITE}+           {  /* Eat multiple white-spaces */                   }
<*>[0-9]+             { YYLVAL(number) = atol(yytext);
                        return INT;                                        }
CTL                   { YYLVAL(number) = JDF_VAR_TYPE_CTL;
                        return DEPENDENCY_TYPE;                            }
RW                    { YYLVAL(number) = JDF_VAR_TYPE_READ | JDF_VAR_TYPE_WRITE;
                        return DEPENDENCY_TYPE;                            }
READ                  { YYLVAL(number) = JDF_VAR_TYPE_READ;
                        return DEPENDENCY_TYPE;                            }
RO                    { YYLVAL(number) = JDF_VAR_TYPE_READ;
                        return DEPENDENCY_TYPE;                            }
WRITE                 { YYLVAL(number) = JDF_VAR_TYPE_WRITE;
                        return DEPENDENCY_TYPE;                            }
WO                    { YYLVAL(number) = JDF_VAR_TYPE_WRITE;
                        return DEPENDENCY_TYPE;                            }
"->"                  { YYLVAL(dep_type) = JDF_DEP_TYPE_OUT;
                        return ARROW;                                      }
"<-"                  { YYLVAL(dep_type) = JDF_DEP_TYPE_IN;
                        return ARROW;                                      }
"\["                  { YYLVAL(property) = NULL;
                        YY_PUSH_STATE(properties);
                        return PROPERTIES_ON;                              }
<properties>"\]"      { YY_POP_STATE();
                        return PROPERTIES_OFF;                             }
<properties>[oO][nN]  { YYLVAL(number) = 1;
                        return INT;                                        }
<properties>[tT][rR][uU][eE] {
                        YYLVAL(number) = 1;
                        return INT;                                        }
<properties>[yY][eE][sS] {
                        YYLVAL(number) = 1;
                        return INT;                                        }
<properties>[oO][fF][fF] {
                        YYLVAL(number) = 0;
                        return INT;                                        }
<properties>[fF][aA][lL][sS][eE] {
                        YYLVAL(number) = 0;
                        return INT;                                        }
<properties>[nN][oO] {  YYLVAL(number) = 0;
                        return INT;                                        }
<*>"("                { return OPEN_PAR;                                   }
<*>")"                { return CLOSE_PAR;                                  }
<*>"=="               { return EQUAL;                                      }
<*>"\!="              { return NOTEQUAL;                                   }
<*>\&\&?              { return AND;                                        }
<*>\|\|?              { return OR;                                         }
<*>"^"                { return XOR;                                        }
<*>"<"                { return LESS;                                       }
<*>"<="               { return LEQ;                                        }
<*>">"                { return MORE;                                       }
<*>">="               { return MEQ;                                        }
<*>"\!"               { return NOT;                                        }
<*>"\+"               { return PLUS;                                       }
<*>"-"                { return MINUS;                                      }
<*>"\*"               { return TIMES;                                      }
<*>"/"                { return DIV;                                        }
<*>"%"                { return MODULO;                                     }
<*>"<<"               { return SHL;                                        }
<*>">>"               { return SHR;                                        }
<*>".."               { return RANGE;                                      }
<*>"="                { return ASSIGNMENT;                                 }
<*>"?"                { return QUESTION_MARK;                              }
<*>":"                { return COLON;                                      }
<*>";"                { return SEMICOLON;                                  }
<*>","                { return COMMA;                                      }

<dstr>\"              { /* saw closing quote - all done */
                        YY_POP_STATE();
                        *string_buf_ptr = '\0';
                        /* return string constant token type and
                        * value to parser
                        */
                        YYLVAL(string) = string_buf;
                        return STRING;                                     }
<*>\"                 { string_buf_ptr = string_buf;
                        YY_PUSH_STATE(dstr);                               }

<dstr>\n              {
                        /* error - unterminated string constant */
                        printf("Unterminated string constant at line %d\n", current_lineno);
                        exit(-1);                                          }

<dstr>\\[0-7]{1,3}    { /* octal escape sequence */
                        int result;

                        (void) sscanf( yytext + 1, "%o", &result );

                        if ( result > 0xff ) {
                            /* error, constant is out-of-bounds */
                            printf("Constant out-of-bound in octal escape sequence at line %d\n", current_lineno);
                            exit(-1);
                        }
                        *string_buf_ptr++ = (char)result;                  }

<dstr>\\[0-9]+        { /* generate error - bad escape sequence; something
                         * like '\48' or '\0777777'
                         */
                         printf("Bad escape sequence at line %d\n", current_lineno);
                         exit(-1);                                         }

<dstr>\\n             { *string_buf_ptr++ = '\n'; current_lineno++;        }
<dstr>\\t             { *string_buf_ptr++ = '\t';                          }
<dstr>\\r             { *string_buf_ptr++ = '\r';                          }
<dstr>\\b             { *string_buf_ptr++ = '\b';                          }
<dstr>\\f             { *string_buf_ptr++ = '\f';                          }

<dstr>\\(.|\n)        { *string_buf_ptr++ = yytext[1];                     }

<dstr>[^\\\n\"]+      {
                        char *yptr = yytext;
                        while ( *yptr )
                            *string_buf_ptr++ = *yptr++;                   }

<*>{WHITE}*\n         { current_lineno++;                                  }
<*>"//".*\n           { current_lineno++;                                  }
<*>[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)* {
                        YYLVAL(string) = strdup(yytext);
                        return VAR;                                        }
%%

#include "main.c"

