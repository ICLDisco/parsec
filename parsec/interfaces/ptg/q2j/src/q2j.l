D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "jdf.h"
#include "node_struct.h"
#include "parse_utility.h"
#include "symtab.h"
#include "q2j.y.h"

int check_type();
void count();
void comment();
void cpp_comment();

%}

%%
"/*"			{ comment(); }
"//"			{ cpp_comment(); }

"PARSEC_INVARIANT"	{ count(); return(DIR_PARSEC_INVARIANT); }
"PARSEC_DATA_COLOCATED"	{ count(); return(DIR_PARSEC_DATA_COLOCATED); }
"PARSEC_TASK_BLACKBOX" { count(); return(DIR_PARSEC_TASK_BLACKBOX); }
"PARSEC_TASK_START"	{ count(); return(DIR_PARSEC_TASK_START); }
"DEF"          { count(); return(DEF); }
"USE"          { count(); return(USE); }
"pragma"		{ count(); return(PRAGMA); }
"break"			{ count(); return(BREAK); }
"case"			{ count(); return(CASE); }
"continue"		{ count(); return(CONTINUE); }
"default"		{ count(); return(DEFAULT); }
"do"			{ count(); return(DO); }
"else"			{ count(); return(ELSE); }
"for"			{ count(); return(FOR); }
"goto"			{ count(); return(GOTO); }
"if"			{ count(); return(IF); }
"return"		{ count(); return(RETURN); }
"sizeof"		{ count(); return(SIZEOF); }
"switch"		{ count(); return(SWITCH); }
"typedef"		{ count(); return(TYPEDEF); }
"while"			{ count(); return(WHILE); }

"auto"			{ count();
                          yylval.string = strdup(yytext);
                          return(AUTO);
                        }

"char"			{ count();
                          yylval.string = strdup(yytext);
                          return(CHAR);
                        }

"const"			{ count();
                          yylval.string = strdup(yytext);
                          return(CONST);
                        }

"double"		{ count();
                          yylval.string = strdup(yytext);
                          return(DOUBLE);
                        }

"enum"			{ count();
                          yylval.string = strdup(yytext);
                          return(ENUM);
                        }

"extern"		{ count();
                          yylval.string = strdup(yytext);
                          return(EXTERN);
                        }

"float"			{ count();
                          yylval.string = strdup(yytext);
                          return(FLOAT);
                        }

"int"			{ count();
                          yylval.string = strdup(yytext);
                          return(INT);
                        }

"long"			{ count();
                          yylval.string = strdup(yytext);
                          return(LONG);
                        }

"int8_t"                { count();
                          yylval.string = strdup(yytext);
                          return(INT8);
                        }

"int16_t"               { count();
                          yylval.string = strdup(yytext);
                          return(INT16);
                        }

"int32_t"               { count();
                          yylval.string = strdup(yytext);
                          return(INT32);
                        }

"int64_t"               { count();
                          yylval.string = strdup(yytext);
                          return(INT64);
                        }

"uint8_t"               { count();
                          yylval.string = strdup(yytext);
                          return(UINT8);
                        }

"uint16_t"              { count();
                          yylval.string = strdup(yytext);
                          return(UINT16);
                        }

"uint32_t"              { count();
                          yylval.string = strdup(yytext);
                          return(UINT32);
                        }

"uint64_t"              { count();
                          yylval.string = strdup(yytext);
                          return(UINT64);
                        }

"intptr_t"              { count();
                          yylval.string = strdup(yytext);
                          return(INTPTR);
                        }

"uintptr_t"             { count();
                          yylval.string = strdup(yytext);
                          return(UINTPTR);
                        }

"intmax_t"              { count();
                          yylval.string = strdup(yytext);
                          return(INTMAX);
                        }

"uintmax_t"             { count();
                          yylval.string = strdup(yytext);
                          return(UINTMAX);
                        }


"register"		{ count();
                          yylval.string = strdup(yytext);
                          return(REGISTER);
                        }

"short"			{ count();
                          yylval.string = strdup(yytext);
                          return(SHORT);
                        }

"signed"		{ count();
                          yylval.string = strdup(yytext);
                          return(SIGNED);
                        }

"static"		{ count();
                          yylval.string = strdup(yytext);
                          return(STATIC);
                        }

"struct"		{ count();
                          yylval.string = strdup(yytext);
                          return(STRUCT);
                        }

"union"			{ count();
                          yylval.string = strdup(yytext);
                          return(UNION);
                        }

"unsigned"		{ count();
                          yylval.string = strdup(yytext);
                          return(UNSIGNED);
                        }

"void"			{ count();
                          yylval.string = strdup(yytext);
                          return(VOID);
                        }

"volatile"		{ count();
                          yylval.string = strdup(yytext);
                          return(VOLATILE);
                        }

"PLASMA_Complex32_t"	{ count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_COMPLEX32_T);
			}

"PLASMA_Complex64_t"	{ count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_COMPLEX64_T);
			}

"PLASMA_enum"		{ count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_ENUM);
			}

"PLASMA_sequence"       { count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_SEQUENCE);
			}

"PLASMA_desc"           { count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_DESC);
			}

"PLASMA_request"        { count();
                          yylval.string = strdup(yytext);
                          return(PLASMA_REQUEST);
			}

{L}({L}|{D})*		{ count(); 
                          if( IDENTIFIER == check_type()){
                              memset(&(yylval.node),0,sizeof(node_t));
                              yylval.node.u.var_name = strdup(yytext);
                              yylval.node.type = IDENTIFIER;
                              yylval.node.u.kids.kid_count = 0;
                              yylval.node.symtab = st_get_current_st();
                              yylval.node.lineno = yyget_lineno();
                              return(IDENTIFIER);
                          }else{
                              yylval.string = strdup(yytext);
                              return(TYPE_NAME);
                          }
                        }

0[bB](0|1)+		{ 
                          int i,len;
                          char *num = yytext+2;
                          count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.i64_value = 0;
                          len = strlen(num)-1;
                          for(i=len; i>=0; i--){
                              yylval.node.const_val.i64_value += ((*(num+i))-'0')<<(len-i);
                          }
                          yylval.node.type = BIN_MASK;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(BIN_MASK);
                        }

0[xX]{H}+{IS}?		{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.i64_value = strtoll(yytext, NULL, 16);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(INTCONSTANT);
                        }

0{D}+{IS}?		{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.i64_value = strtoll(yytext, NULL, 8);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(INTCONSTANT);
                        }
{D}+{IS}?		{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(INTCONSTANT);
                        }
{D}+{E}{FS}?		{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(INTCONSTANT);
                        }
L?'(\\.|[^\\'])+'	{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(FLOATCONSTANT);
                        }

{D}*"."{D}+({E})?{FS}?	{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(FLOATCONSTANT);
                        }
{D}+"."{D}*({E})?{FS}?	{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(FLOATCONSTANT);
                        }

L?\"(\\.|[^\\"])*\"	{ count();
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.const_val.str = strdup(yytext);
                          yylval.node.u.kids.kid_count = 0;
                          yylval.node.lineno = yyget_lineno();
                          return(STRING_LITERAL);
                        }

"..."			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = ELLIPSIS;
                          yylval.node.lineno = yyget_lineno();
                          return(ELLIPSIS);
                        }

">>="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = RIGHT_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(RIGHT_ASSIGN);
                        }

"<<="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = LEFT_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(LEFT_ASSIGN);
                        }
".."           { count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = DDOT;
                          yylval.node.lineno = yyget_lineno();
                          return(DDOT);
                        }
"+="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = ADD_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(ADD_ASSIGN);
                        }

"-="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = SUB_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(SUB_ASSIGN);
                        }

"*="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = MUL_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(MUL_ASSIGN);
                        }

"/="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = DIV_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(DIV_ASSIGN);
                        }

"%="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = MOD_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(MOD_ASSIGN);
                        }

"&="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = AND_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(AND_ASSIGN);
                        }

"^="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = XOR_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(XOR_ASSIGN);
                        }

"|="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = OR_ASSIGN;
                          yylval.node.lineno = yyget_lineno();
                          return(OR_ASSIGN);
                        }

">>"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = RIGHT_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(RIGHT_OP);
                        }

"<<"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = LEFT_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(LEFT_OP);
                        }

"++"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = INC_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(INC_OP);
                        }

"--"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = DEC_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(DEC_OP);
                        }

"->"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = PTR_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(PTR_OP);
                        }

"<-"           { count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = TASK_USE_DEP;
                          yylval.node.lineno = yyget_lineno();
                          return(TASK_USE_DEP);
                        }

"&&"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = L_AND;
                          yylval.node.lineno = yyget_lineno();
                          return(L_AND);
                        }

"||"			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = L_OR;
                          yylval.node.lineno = yyget_lineno();
                          return(L_OR);
                        }

"<="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = LE_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(LE_OP);
                        }

">="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = GE_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(GE_OP);
                        }

"=="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = EQ_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(EQ_OP);
                        }

"!="			{ count(); 
                          memset(&(yylval.node),0,sizeof(node_t));
                          yylval.node.type = NE_OP;
                          yylval.node.lineno = yyget_lineno();
                          return(NE_OP);
                        }


";"			{ count(); return(';'); }
("{"|"<%")		{ count();
                          (void)st_enter_new_scope();
                          return('{');
                        }
("}"|"%>")		{ count();
                          (void)st_exit_scope();
                          return('}');
                        }
","			{ count(); return(','); }
":"			{ count(); return(':'); }
"="			{ count(); return('='); }
"("			{ count(); return('('); }
")"			{ count(); return(')'); }
("["|"<:")		{ count(); return('['); }
("]"|":>")		{ count(); return(']'); }
"."			{ count(); return('.'); }
"&"			{ count(); return('&'); }
"!"			{ count(); return('!'); }
"~"			{ count(); return('~'); }
"-"			{ count(); return('-'); }
"+"			{ count(); return('+'); }
"*"			{ count(); return('*'); }
"/"			{ count(); return('/'); }
"%"			{ count(); return('%'); }
"<"			{ count(); return('<'); }
">"			{ count(); return('>'); }
"^"			{ count(); return('^'); }
"|"			{ count(); return('|'); }
"?"			{ count(); return('?'); }

[ \t\v\n\f]		{ count(); }
.			{ /* ignore bad characters */ }

%%

int yywrap(){
	return(1);
}


void cpp_comment(){
    int c;
#ifdef __cplusplus
    while ((c = yyinput()) != '\n' && c != 0);
#else
    while ((c = input()) != '\n' && c != 0);
#endif
    if( c == '\n' )
        yyset_lineno( yyget_lineno() + 1 );
}

void comment(){
	int c, c1;

loop:
#ifdef __cplusplus
	while ((c = yyinput()) != '*' && c != 0){
#else
	while ((c = input()) != '*' && c != 0){
#endif
            if( c == '\n' )
                yyset_lineno( yyget_lineno() + 1 );
        }

#ifdef __cplusplus
	if ((c1 = yyinput()) != '/' && c != 0)
#else
	if ((c1 = input()) != '/' && c != 0)
#endif
	{
		unput(c1);
		goto loop;
	}

}


int column = 0;

void count()
{
	int i;

	for (i = 0; yytext[i] != '\0'; i++)
		if (yytext[i] == '\n'){
			column = 0;
			yyset_lineno( yyget_lineno() + 1 );
                }
		else if (yytext[i] == '\t')
			column += 8 - (column % 8);
		else
			column++;

}


int check_type() {

    if( NULL != lookup_type(yytext) ){
        return(TYPE_NAME);
    }

    return(IDENTIFIER);
}
