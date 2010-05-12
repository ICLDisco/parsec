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
     VAR = 258,
     ASSIGNMENT = 259,
     EXTERN_DECL = 260,
     COMMA = 261,
     OPEN_PAR = 262,
     CLOSE_PAR = 263,
     BODY = 264,
     COLON = 265,
     SEMICOLON = 266,
     DEPENDENCY_TYPE = 267,
     ARROW = 268,
     QUESTION_MARK = 269,
     OPTIONAL_INFO = 270,
     EQUAL = 271,
     NOTEQUAL = 272,
     LESS = 273,
     LEQ = 274,
     MORE = 275,
     MEQ = 276,
     AND = 277,
     OR = 278,
     XOR = 279,
     NOT = 280,
     INT = 281,
     PLUS = 282,
     MINUS = 283,
     TIMES = 284,
     DIV = 285,
     MODULO = 286,
     SHL = 287,
     SHR = 288,
     RANGE = 289
   };
#endif
/* Tokens.  */
#define VAR 258
#define ASSIGNMENT 259
#define EXTERN_DECL 260
#define COMMA 261
#define OPEN_PAR 262
#define CLOSE_PAR 263
#define BODY 264
#define COLON 265
#define SEMICOLON 266
#define DEPENDENCY_TYPE 267
#define ARROW 268
#define QUESTION_MARK 269
#define OPTIONAL_INFO 270
#define EQUAL 271
#define NOTEQUAL 272
#define LESS 273
#define LEQ 274
#define MORE 275
#define MEQ 276
#define AND 277
#define OR 278
#define XOR 279
#define NOT 280
#define INT 281
#define PLUS 282
#define MINUS 283
#define TIMES 284
#define DIV 285
#define MODULO 286
#define SHL 287
#define SHR 288
#define RANGE 289




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 57 "dague.y"
{
  int                   number;
  char*                 string;
  jdf_expr_operand_t    expr_op;
  jdf_preamble_entry_t *preamble;
  jdf_global_entry_t   *global;
  jdf_function_entry_t *function;
  jdf_name_list_t      *name_list;
  jdf_flags_t           flags;
  jdf_def_list_t       *def_list;
  jdf_dataflow_list_t  *dataflow_list;
  jdf_dataflow_t       *dataflow;
  jdf_dep_list_t       *dep_list;
  jdf_dep_t            *dep;
  jdf_dep_type_t        dep_type;
  jdf_guarded_call_t   *guarded_call;
  jdf_call_t           *call;
  jdf_expr_list_t      *expr_list;
  jdf_expr_t           *expr;
}
/* Line 1489 of yacc.c.  */
#line 138 "/Users/bouteill/dague/dplasma/tools/dague-compiler/dague.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

