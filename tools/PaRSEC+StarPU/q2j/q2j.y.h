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
     IDENTIFIER = 258,
     INTCONSTANT = 259,
     BIN_MASK = 260,
     FLOATCONSTANT = 261,
     STRING_LITERAL = 262,
     SIZEOF = 263,
     PTR_OP = 264,
     INC_OP = 265,
     DEC_OP = 266,
     LEFT_OP = 267,
     RIGHT_OP = 268,
     LE_OP = 269,
     GE_OP = 270,
     EQ_OP = 271,
     NE_OP = 272,
     L_AND = 273,
     L_OR = 274,
     MUL_ASSIGN = 275,
     DIV_ASSIGN = 276,
     MOD_ASSIGN = 277,
     ADD_ASSIGN = 278,
     SUB_ASSIGN = 279,
     LEFT_ASSIGN = 280,
     RIGHT_ASSIGN = 281,
     AND_ASSIGN = 282,
     XOR_ASSIGN = 283,
     OR_ASSIGN = 284,
     TYPE_NAME = 285,
     TYPEDEF = 286,
     PRAGMA = 287,
     EXTERN = 288,
     STATIC = 289,
     AUTO = 290,
     REGISTER = 291,
     STARPU_CODELET = 292,
     STARPU_FUNC = 293,
     DIR_PARSEC_DATA_COLOCATED = 294,
     DIR_PARSEC_INVARIANT = 295,
     DIR_PARSEC_TASK_START = 296,
     CHAR = 297,
     SHORT = 298,
     INT = 299,
     LONG = 300,
     SIGNED = 301,
     UNSIGNED = 302,
     FLOAT = 303,
     DOUBLE = 304,
     CONST = 305,
     VOLATILE = 306,
     VOID = 307,
     INT8 = 308,
     INT16 = 309,
     INT32 = 310,
     INT64 = 311,
     UINT8 = 312,
     UINT16 = 313,
     UINT32 = 314,
     UINT64 = 315,
     INTPTR = 316,
     UINTPTR = 317,
     INTMAX = 318,
     UINTMAX = 319,
     PLASMA_COMPLEX32_T = 320,
     PLASMA_COMPLEX64_T = 321,
     PLASMA_ENUM = 322,
     PLASMA_REQUEST = 323,
     PLASMA_DESC = 324,
     PLASMA_SEQUENCE = 325,
     STRUCT = 326,
     UNION = 327,
     ENUM = 328,
     ELLIPSIS = 329,
     CASE = 330,
     DEFAULT = 331,
     IF = 332,
     ELSE = 333,
     SWITCH = 334,
     WHILE = 335,
     DO = 336,
     FOR = 337,
     GOTO = 338,
     CONTINUE = 339,
     BREAK = 340,
     RETURN = 341
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 31 "src/q2j.y"

    node_t node;
    type_node_t type_node;
    char *string;
    StarPU_param         *param;
    StarPU_param_list    *param_list;
    StarPU_function_list *function_list;
    StarPU_fun_decl      *starpu_fun_d;



/* Line 2068 of yacc.c  */
#line 148 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/q2j/q2j.y.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


