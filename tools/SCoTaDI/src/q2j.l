D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
#include <stdio.h>
#include <stdlib.h>
#include "node_struct.h"
#include "q2j.tab.h"

void count();
void comment();
%}

%%
"/*"			{ comment(); }

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
                          yylval.node.u.var_name = strdup(yytext);
                          yylval.node.type = IDENTIFIER;
                          yylval.node.u.kids.kid_count = 0;
                          return(IDENTIFIER);
                          //return(check_type());
                        }

0[xX]{H}+{IS}?		{ count(); 
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(INTCONSTANT);
                        }
0{D}+{IS}?		{ count();
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(INTCONSTANT);
                        }
{D}+{IS}?		{ count();
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(INTCONSTANT);
                        }
{D}+{E}{FS}?		{ count();
                          yylval.node.const_val.i64_value = atoll(yytext);
                          yylval.node.type = INTCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(INTCONSTANT);
                        }
L?'(\\.|[^\\'])+'	{ count();
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(FLOATCONSTANT);
                        }

{D}*"."{D}+({E})?{FS}?	{ count();
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(FLOATCONSTANT);
                        }
{D}+"."{D}*({E})?{FS}?	{ count();
                          yylval.node.const_val.f64_value = atof(yytext);
                          yylval.node.type = FLOATCONSTANT;
                          yylval.node.u.kids.kid_count = 0;
                          return(FLOATCONSTANT);
                        }

L?\"(\\.|[^\\"])*\"	{ count();
                          yylval.node.const_val.str = strdup(yytext);
                          yylval.node.u.kids.kid_count = 0;
                          return(STRING_LITERAL);
                        }

"..."			{ count(); 
                          yylval.node.type = ELLIPSIS;
                          return(ELLIPSIS);
                        }

">>="			{ count(); 
                          yylval.node.type = RIGHT_ASSIGN;
                          return(RIGHT_ASSIGN);
                        }

"<<="			{ count(); 
                          yylval.node.type = LEFT_ASSIGN;
                          return(LEFT_ASSIGN);
                        }

"+="			{ count(); 
                          yylval.node.type = ADD_ASSIGN;
                          return(ADD_ASSIGN);
                        }

"-="			{ count(); 
                          yylval.node.type = SUB_ASSIGN;
                          return(SUB_ASSIGN);
                        }

"*="			{ count(); 
                          yylval.node.type = MUL_ASSIGN;
                          return(MUL_ASSIGN);
                        }

"/="			{ count(); 
                          yylval.node.type = DIV_ASSIGN;
                          return(DIV_ASSIGN);
                        }

"%="			{ count(); 
                          yylval.node.type = MOD_ASSIGN;
                          return(MOD_ASSIGN);
                        }

"&="			{ count(); 
                          yylval.node.type = AND_ASSIGN;
                          return(AND_ASSIGN);
                        }

"^="			{ count(); 
                          yylval.node.type = XOR_ASSIGN;
                          return(XOR_ASSIGN);
                        }

"|="			{ count(); 
                          yylval.node.type = OR_ASSIGN;
                          return(OR_ASSIGN);
                        }

">>"			{ count(); 
                          yylval.node.type = RIGHT_OP;
                          return(RIGHT_OP);
                        }

"<<"			{ count(); 
                          yylval.node.type = LEFT_OP;
                          return(LEFT_OP);
                        }

"++"			{ count(); 
                          yylval.node.type = INC_OP;
                          return(INC_OP);
                        }

"--"			{ count(); 
                          yylval.node.type = DEC_OP;
                          return(DEC_OP);
                        }

"->"			{ count(); 
                          yylval.node.type = PTR_OP;
                          return(PTR_OP);
                        }

"&&"			{ count(); 
                          yylval.node.type = L_AND;
                          return(L_AND);
                        }

"||"			{ count(); 
                          yylval.node.type = L_OR;
                          return(L_OR);
                        }

"<="			{ count(); 
                          yylval.node.type = LE_OP;
                          return(LE_OP);
                        }

">="			{ count(); 
                          yylval.node.type = GE_OP;
                          return(GE_OP);
                        }

"=="			{ count(); 
                          yylval.node.type = EQ_OP;
                          return(EQ_OP);
                        }

"!="			{ count(); 
                          yylval.node.type = NE_OP;
                          return(NE_OP);
                        }


";"			{ count(); return(';'); }
("{"|"<%")		{ count(); return('{'); }
("}"|"%>")		{ count(); return('}'); }
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


void comment(){
	char c, c1;

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

//	if (c != 0)
//		putchar(c1);
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

	/*ECHO;*/
}


int check_type()
{
/*
* pseudo code --- this is what it should check
*
*	if (yytext == type_name)
*		return(TYPE_NAME);
*
*	return(IDENTIFIER);
*/

/*
*	it actually will only return IDENTIFIER
*/

	return(IDENTIFIER);
}
