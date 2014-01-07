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

#include <dague_config.h>
#include "jdf.h"
#include "dague.y.h"

extern char *yyfilename;
extern int current_lineno;

#if 0
#define ECHO {fwrite("[", 1, 1, yyout);fwrite( yytext, yyleng, 1, yyout );fwrite("]", 1, 1, yyout);}
#endif
#define ECHO

static char *extern_code(char *base, int len, int line)
{
     char *res;
     int rp, rlen;

     rlen = len + 16 + strlen(yyfilename);
     res = (char*)calloc(1, rlen);

     rp = 0;
     if( !JDF_COMPILER_GLOBAL_ARGS.noline && line) {
         /**
          * As each extern_code is contiguous we only have to set the #line once.
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
/*%option reentrant*/
%option noyywrap yylineno
%option stack
/*%option debug*/

WHITE         [\f\t\v ]

%x comment
%x properties
%x dstr
%x body
%x body_init

%%
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
                          YYLVAL(string) = extern_code( yytext + bstart, strlen(yytext)-bstart-2, 1 );
                          ECHO; return EXTERN_DECL;
                      }
<INITIAL,properties>inline_c{WHITE}*\%\{([^\%]|(\%[^\}]))+\%\}            {
                          int start;
                          for(start = 8; yytext[start] != '{'; start++) /* nothing */ ;
                          start++;
                          YYLVAL(string) = extern_code( yytext + start, strlen(yytext)-start-2, 0 );
                          ECHO; return EXTERN_DECL;
                      }
SIMCOST               { ECHO; return SIMCOST; }
<*>"/*"               { YY_PUSH_STATE(comment);                            }
<comment>[^*\n]*      {  /* Eat up non '*'s */                             }
<comment>"*"+[^*/\n]* {  /* Eat '*'s not followed by a '/' */              }
<comment>"*"+"/"      { YY_POP_STATE();  /* Done with the comment BLOCK */ }
<*>{WHITE}+           {  /* Eat multiple white-spaces */                   }
<*>[0-9]+             { YYLVAL(number) = atol(yytext);
                        ECHO; return INT;                                  }
CTL                   { YYLVAL(number) = JDF_FLOW_TYPE_CTL;
                        ECHO; return DEPENDENCY_TYPE;                      }
RW                    { YYLVAL(number) = JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE;
                        ECHO; return DEPENDENCY_TYPE;                      }
READ                  { YYLVAL(number) = JDF_FLOW_TYPE_READ;
                        ECHO; return DEPENDENCY_TYPE;                      }
RO                    { YYLVAL(number) = JDF_FLOW_TYPE_READ;
                        ECHO; return DEPENDENCY_TYPE;                      }
WRITE                 { YYLVAL(number) = JDF_FLOW_TYPE_WRITE;
                        ECHO; return DEPENDENCY_TYPE;                      }
WO                    { YYLVAL(number) = JDF_FLOW_TYPE_WRITE;
                        ECHO; return DEPENDENCY_TYPE;                      }
"->"                  { YYLVAL(dep_type) = JDF_DEP_FLOW_OUT;
                        ECHO; return ARROW;                                }
"<-"                  { YYLVAL(dep_type) = JDF_DEP_FLOW_IN;
                        ECHO; return ARROW;                                }
"\["                  { YYLVAL(property) = NULL;
                        YY_PUSH_STATE(properties);
                        ECHO; return PROPERTIES_ON;                        }
<properties>"\]"      { YY_POP_STATE();
                        ECHO; return PROPERTIES_OFF;                       }
<properties>[oO][nN]  { YYLVAL(number) = 1;
                        ECHO; return INT;                                  }
<properties>[tT][rR][uU][eE] {
                        YYLVAL(number) = 1;
                        ECHO; return INT;                                  }
<properties>[yY][eE][sS] {
                        YYLVAL(number) = 1;
                        ECHO; return INT;                                  }
<properties>[oO][fF][fF] {
                        YYLVAL(number) = 0;
                        ECHO; return INT;                                  }
<properties>[fF][aA][lL][sS][eE] {
                        YYLVAL(number) = 0;
                        ECHO; return INT;                                  }
<properties>[nN][oO] {  YYLVAL(number) = 0;
                        ECHO; return INT;                                  }

<INITIAL>^{WHITE}*(?i:body) {
                           YY_PUSH_STATE(body_init);
                           return BODY_START;                              }
<body_init>{WHITE}*[\[] {
                           YY_PUSH_STATE(properties);
                           return PROPERTIES_ON;                           }
<body_init>{WHITE}*\n {    current_lineno++;
                           YY_POP_STATE();
                           YY_PUSH_STATE(body);                            }
^{WHITE}*(?i:body)    { YY_PUSH_STATE(body); }
<body>^{WHITE}*(?i:end) { YY_POP_STATE();
                          YYLVAL(string) = extern_code(yytext, strlen(yytext) - 3, 1);
                          ECHO; return BODY_END;                           }
<body>[^\n]*          { yymore();}
<body>{WHITE}*\n      { current_lineno++; yymore();}
^%(?i:option)         { ECHO; return OPTION;                               }
<*>"("                { ECHO; return OPEN_PAR;                             }
<*>")"                { ECHO; return CLOSE_PAR;                            }
<*>"=="               { ECHO; return EQUAL;                                }
<*>"\!="              { ECHO; return NOTEQUAL;                             }
<*>\&\&?              { ECHO; return AND;                                  }
<*>\|\|?              { ECHO; return OR;                                   }
<*>"^"                { ECHO; return XOR;                                  }
<*>"<"                { ECHO; return LESS;                                 }
<*>"<="               { ECHO; return LEQ;                                  }
<*>">"                { ECHO; return MORE;                                 }
<*>">="               { ECHO; return MEQ;                                  }
<*>"\!"               { ECHO; return NOT;                                  }
<*>"\+"               { ECHO; return PLUS;                                 }
<*>"-"                { ECHO; return MINUS;                                }
<*>"\*"               { ECHO; return TIMES;                                }
<*>"/"                { ECHO; return DIV;                                  }
<*>"%"                { ECHO; return MODULO;                               }
<*>"<<"               { ECHO; return SHL;                                  }
<*>">>"               { ECHO; return SHR;                                  }
<*>".."               { ECHO; return RANGE;                                }
<*>"="                { ECHO; return ASSIGNMENT;                           }
<*>"?"                { ECHO; return QUESTION_MARK;                        }
<*>":"                { ECHO; return COLON;                                }
<*>";"                { ECHO; return SEMICOLON;                            }
<*>","                { ECHO; return COMMA;                                }

<dstr>\"              { /* saw closing quote - all done */
                        YY_POP_STATE();
                        *string_buf_ptr = '\0';
                        /* return string constant token type and
                        * value to parser
                        */
                        YYLVAL(string) = string_buf;
                        ECHO; return STRING;                               }
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

<*>\n                 { current_lineno++;                                  }
<*>"//".*\n           { current_lineno++;                                  }
<*>[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)* {
                        YYLVAL(string) = strdup(yytext);
                        ECHO; return VAR;                                  }
%%

#include "main.c"

