/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     DPLASMA_COMMA = 258,
     DPLASMA_OPEN_PAR = 259,
     DPLASMA_CLOSE_PAR = 260,
     DPLASMA_RANGE = 261,
     DPLASMA_EQUAL = 262,
     DPLASMA_NOT_EQUAL = 263,
     DPLASMA_ASSIGNMENT = 264,
     DPLASMA_QUESTION = 265,
     DPLASMA_LESS = 266,
     DPLASMA_MORE = 267,
     DPLASMA_LESS_OR_EQUAL = 268,
     DPLASMA_MORE_OR_EQUAL = 269,
     DPLASMA_COLON = 270,
     DPLASMA_SEMICOLON = 271,
     DPLASMA_INT = 272,
     DPLASMA_VAR = 273,
     DPLASMA_BODY = 274,
     DPLASMA_OPTIONAL_INFO = 275,
     DPLASMA_OP = 276,
     DPLASMA_DEPENDENCY_TYPE = 277,
     DPLASMA_ARROW = 278,
     DPLASMA_EXTERN_DECL = 279
   };
#endif
/* Tokens.  */
#define DPLASMA_COMMA 258
#define DPLASMA_OPEN_PAR 259
#define DPLASMA_CLOSE_PAR 260
#define DPLASMA_RANGE 261
#define DPLASMA_EQUAL 262
#define DPLASMA_NOT_EQUAL 263
#define DPLASMA_ASSIGNMENT 264
#define DPLASMA_QUESTION 265
#define DPLASMA_LESS 266
#define DPLASMA_MORE 267
#define DPLASMA_LESS_OR_EQUAL 268
#define DPLASMA_MORE_OR_EQUAL 269
#define DPLASMA_COLON 270
#define DPLASMA_SEMICOLON 271
#define DPLASMA_INT 272
#define DPLASMA_VAR 273
#define DPLASMA_BODY 274
#define DPLASMA_OPTIONAL_INFO 275
#define DPLASMA_OP 276
#define DPLASMA_DEPENDENCY_TYPE 277
#define DPLASMA_ARROW 278
#define DPLASMA_EXTERN_DECL 279




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 46 "DAGuE.y"
{
    int        number;
    char*      string;
    char       operand;
    expr_t*    expr;
    DAGuE_t* DAGuE;
    struct {
        char  *code;
        char  *language;
    }          two_strings;
}
/* Line 1529 of yacc.c.  */
#line 109 "/Users/herault/Documents/Recherche/DAGuE/DAGuE.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

