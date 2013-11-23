/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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
     GPU = 265,
     MODEL = 266,
     STRING = 267,
     SIMCOST = 268,
     COLON = 269,
     SEMICOLON = 270,
     DEPENDENCY_TYPE = 271,
     ARROW = 272,
     QUESTION_MARK = 273,
     PROPERTIES_ON = 274,
     PROPERTIES_OFF = 275,
     EQUAL = 276,
     NOTEQUAL = 277,
     LESS = 278,
     LEQ = 279,
     MORE = 280,
     MEQ = 281,
     AND = 282,
     OR = 283,
     XOR = 284,
     NOT = 285,
     INT = 286,
     PLUS = 287,
     MINUS = 288,
     TIMES = 289,
     DIV = 290,
     MODULO = 291,
     SHL = 292,
     SHR = 293,
     RANGE = 294
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 115 "dague.y"

    int                   number;
    char*                 string;
    jdf_expr_operand_t    expr_op;
    jdf_external_entry_t *external_code;
    jdf_global_entry_t   *global;
    jdf_function_entry_t *function;
    jdf_def_list_t       *property;
    jdf_name_list_t      *name_list;
    jdf_def_list_t       *def_list;
    jdf_dataflow_t       *dataflow;
    jdf_dep_t            *dep;
    jdf_dep_flags_t        dep_type;
    jdf_guarded_call_t   *guarded_call;
    jdf_call_t           *call;
    jdf_expr_t           *expr;



/* Line 2068 of yacc.c  */
#line 109 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/dague-compiler/dague.y.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


