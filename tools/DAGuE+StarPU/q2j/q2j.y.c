/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 1 "src/q2j.y"

/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "node_struct.h"
#include "utility.h"
#include "omega_interface.h"
#include "symtab.h"
#include "parse_utility.h"
#include "starpu_struct.h"

extern int yyget_lineno();
void yyerror(const char *s);
extern int yylex (void);
extern int _q2j_add_phony_tasks;
extern StarPU_codelet_list *codelet_list;

type_list_t *type_hash[HASH_TAB_SIZE] = {0};



/* Line 268 of yacc.c  */
#line 102 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/q2j/q2j.y.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


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
     DIR_DAGUE_DATA_COLOCATED = 294,
     DIR_DAGUE_INVARIANT = 295,
     DIR_DAGUE_TASK_START = 296,
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

/* Line 293 of yacc.c  */
#line 31 "src/q2j.y"

    node_t node;
    type_node_t type_node;
    char *string;
    StarPU_param         *param;
    StarPU_param_list    *param_list;
    StarPU_function_list *function_list;
    StarPU_fun_decl      *starpu_fun_d;



/* Line 293 of yacc.c  */
#line 236 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/q2j/q2j.y.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 248 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/q2j/q2j.y.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  74
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1666

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  74
/* YYNRULES -- Number of rules.  */
#define YYNRULES  253
/* YYNRULES -- Number of states.  */
#define YYNSTATES  416

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   341

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    98,     2,     2,     2,   100,    93,     2,
      87,    88,    94,    95,    92,    96,    91,    99,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   106,   108,
     101,   107,   102,   105,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    89,     2,    90,   103,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   109,   104,   110,    97,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    11,    13,    17,    19,
      24,    28,    33,    37,    41,    44,    47,    49,    53,    55,
      58,    61,    64,    67,    72,    74,    76,    78,    80,    82,
      84,    86,    91,    93,    97,   101,   105,   107,   111,   115,
     117,   121,   125,   127,   131,   135,   139,   143,   145,   149,
     153,   155,   159,   161,   165,   167,   171,   173,   177,   179,
     183,   185,   191,   193,   197,   199,   201,   203,   205,   207,
     209,   211,   213,   215,   217,   219,   221,   225,   227,   230,
     234,   239,   241,   245,   247,   251,   253,   255,   258,   262,
     266,   270,   275,   277,   280,   282,   285,   287,   290,   292,
     296,   298,   302,   304,   306,   308,   310,   312,   314,   316,
     318,   320,   322,   324,   326,   328,   330,   332,   334,   336,
     338,   340,   342,   344,   346,   348,   350,   352,   354,   356,
     358,   360,   362,   364,   366,   368,   370,   376,   381,   384,
     386,   388,   390,   393,   397,   399,   403,   409,   414,   419,
     421,   425,   434,   437,   439,   442,   444,   446,   450,   452,
     457,   463,   466,   468,   472,   474,   478,   480,   482,   485,
     487,   489,   493,   498,   502,   507,   512,   516,   518,   521,
     524,   528,   530,   533,   535,   539,   541,   545,   548,   551,
     553,   555,   559,   561,   564,   566,   568,   571,   575,   578,
     582,   586,   591,   594,   598,   602,   607,   609,   613,   618,
     620,   624,   626,   628,   630,   632,   634,   636,   638,   642,
     647,   651,   654,   658,   662,   667,   669,   672,   674,   677,
     679,   682,   688,   696,   702,   708,   716,   723,   731,   735,
     738,   741,   744,   748,   750,   753,   755,   757,   759,   761,
     763,   768,   772,   776
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     112,     0,    -1,   182,    -1,     3,    -1,     4,    -1,     6,
      -1,     7,    -1,    87,   132,    88,    -1,   113,    -1,   114,
      89,   132,    90,    -1,   114,    87,    88,    -1,   114,    87,
     115,    88,    -1,   114,    91,     3,    -1,   114,     9,     3,
      -1,   114,    10,    -1,   114,    11,    -1,   130,    -1,   115,
      92,   130,    -1,   114,    -1,    10,   116,    -1,    11,   116,
      -1,   117,   118,    -1,     8,   116,    -1,     8,    87,   168,
      88,    -1,    93,    -1,    94,    -1,    95,    -1,    96,    -1,
      97,    -1,    98,    -1,   116,    -1,    87,   168,    88,   118,
      -1,   118,    -1,   119,    94,   118,    -1,   119,    99,   118,
      -1,   119,   100,   118,    -1,   119,    -1,   120,    95,   119,
      -1,   120,    96,   119,    -1,   120,    -1,   121,    12,   120,
      -1,   121,    13,   120,    -1,   121,    -1,   122,   101,   121,
      -1,   122,   102,   121,    -1,   122,    14,   121,    -1,   122,
      15,   121,    -1,   122,    -1,   123,    16,   122,    -1,   123,
      17,   122,    -1,   123,    -1,   124,    93,   123,    -1,   124,
      -1,   125,   103,   124,    -1,   125,    -1,   126,   104,   125,
      -1,   126,    -1,   127,    18,   126,    -1,   127,    -1,   128,
      19,   127,    -1,   128,    -1,   128,   105,   132,   106,   129,
      -1,   129,    -1,   116,   131,   130,    -1,   107,    -1,    20,
      -1,    21,    -1,    22,    -1,    23,    -1,    24,    -1,    25,
      -1,    26,    -1,    27,    -1,    28,    -1,    29,    -1,   130,
      -1,   132,    92,   130,    -1,   129,    -1,   140,   108,    -1,
     140,   141,   108,    -1,    31,   140,     3,   108,    -1,     3,
      -1,     3,   106,   136,    -1,   136,    -1,   136,    92,   137,
      -1,     3,    -1,     5,    -1,     3,   138,    -1,    32,     3,
     138,    -1,    32,    40,   132,    -1,    32,    39,   138,    -1,
      32,    41,     3,   137,    -1,   143,    -1,   143,   140,    -1,
     144,    -1,   144,   140,    -1,   159,    -1,   159,   140,    -1,
     142,    -1,   141,    92,   142,    -1,   160,    -1,   160,   107,
     171,    -1,    33,    -1,    34,    -1,    35,    -1,    36,    -1,
      52,    -1,    42,    -1,    43,    -1,    44,    -1,    45,    -1,
      53,    -1,    54,    -1,    55,    -1,    56,    -1,    57,    -1,
      58,    -1,    59,    -1,    60,    -1,    61,    -1,    62,    -1,
      63,    -1,    64,    -1,    48,    -1,    49,    -1,    46,    -1,
      47,    -1,   145,    -1,   156,    -1,    65,    -1,    66,    -1,
      67,    -1,    68,    -1,    69,    -1,    70,    -1,    30,    -1,
     146,     3,   109,   147,   110,    -1,   146,   109,   147,   110,
      -1,   146,     3,    -1,    71,    -1,    72,    -1,   148,    -1,
     147,   148,    -1,   153,   154,   108,    -1,     3,    -1,     3,
      92,   149,    -1,    38,   107,   109,   149,   110,    -1,    91,
       3,   107,     3,    -1,    91,     3,   107,     4,    -1,   150,
      -1,   150,    92,   151,    -1,    71,    37,     3,   107,   109,
     151,   110,   108,    -1,   144,   153,    -1,   144,    -1,   159,
     153,    -1,   159,    -1,   155,    -1,   154,    92,   155,    -1,
     160,    -1,    73,   109,   157,   110,    -1,    73,     3,   109,
     157,   110,    -1,    73,     3,    -1,   158,    -1,   157,    92,
     158,    -1,     3,    -1,     3,   107,   133,    -1,    50,    -1,
      51,    -1,   162,   161,    -1,   161,    -1,     3,    -1,    87,
     160,    88,    -1,   161,    89,   133,    90,    -1,   161,    89,
      90,    -1,   161,    87,   164,    88,    -1,   161,    87,   167,
      88,    -1,   161,    87,    88,    -1,    94,    -1,    94,   163,
      -1,    94,   162,    -1,    94,   163,   162,    -1,   159,    -1,
     163,   159,    -1,   165,    -1,   165,    92,    74,    -1,   166,
      -1,   165,    92,   166,    -1,   140,   160,    -1,   140,   169,
      -1,   140,    -1,     3,    -1,   167,    92,     3,    -1,   153,
      -1,   153,   169,    -1,   162,    -1,   170,    -1,   162,   170,
      -1,    87,   169,    88,    -1,    89,    90,    -1,    89,   133,
      90,    -1,   170,    89,    90,    -1,   170,    89,   133,    90,
      -1,    87,    88,    -1,    87,   164,    88,    -1,   170,    87,
      88,    -1,   170,    87,   164,    88,    -1,   130,    -1,   109,
     172,   110,    -1,   109,   172,    92,   110,    -1,   171,    -1,
     172,    92,   171,    -1,   174,    -1,   175,    -1,   178,    -1,
     179,    -1,   180,    -1,   181,    -1,   139,    -1,     3,   106,
     173,    -1,    75,   133,   106,   173,    -1,    76,   106,   173,
      -1,   109,   110,    -1,   109,   177,   110,    -1,   109,   176,
     110,    -1,   109,   176,   177,   110,    -1,   134,    -1,   176,
     134,    -1,   173,    -1,   177,   173,    -1,   108,    -1,   132,
     108,    -1,    77,    87,   132,    88,   173,    -1,    77,    87,
     132,    88,   173,    78,   173,    -1,    79,    87,   132,    88,
     173,    -1,    80,    87,   132,    88,   173,    -1,    81,   173,
      80,    87,   132,    88,   108,    -1,    82,    87,   178,   178,
      88,   173,    -1,    82,    87,   178,   178,   132,    88,   173,
      -1,    83,     3,   108,    -1,    84,   108,    -1,    85,   108,
      -1,    86,   108,    -1,    86,   132,   108,    -1,   183,    -1,
     183,   182,    -1,   184,    -1,   134,    -1,   135,    -1,   139,
      -1,   152,    -1,   140,   160,   176,   175,    -1,   140,   160,
     175,    -1,   160,   176,   175,    -1,   160,   175,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   227,   227,   238,   248,   249,   250,   251,   255,   259,
     278,   287,   310,   318,   326,   334,   350,   358,   371,   372,
     380,   388,   396,   403,   412,   413,   414,   415,   416,   417,
     421,   422,   431,   432,   440,   448,   459,   460,   468,   479,
     480,   488,   499,   500,   508,   516,   524,   535,   536,   544,
     555,   556,   567,   568,   579,   580,   591,   592,   603,   604,
     615,   616,   628,   632,   643,   644,   645,   646,   647,   648,
     649,   650,   651,   652,   653,   657,   658,   669,   676,   680,
     740,   747,   750,   756,   759,   765,   773,   781,   793,   796,
     800,   822,   829,   830,   835,   839,   844,   848,   856,   861,
     872,   873,   884,   885,   886,   887,   891,   892,   893,   894,
     895,   896,   897,   898,   899,   900,   901,   902,   903,   904,
     905,   906,   907,   908,   909,   910,   911,   912,   913,   914,
     915,   916,   917,   918,   919,   920,   925,   926,   927,   931,
     932,   936,   937,   941,   945,   961,   974,   990,  1024,  1038,
    1045,  1055,  1082,  1087,  1088,  1093,  1097,  1098,  1102,  1108,
    1109,  1110,  1114,  1115,  1119,  1120,  1124,  1125,  1129,  1135,
    1143,  1144,  1148,  1165,  1187,  1193,  1194,  1198,  1199,  1200,
    1201,  1209,  1210,  1215,  1216,  1224,  1229,  1236,  1253,  1259,
    1263,  1264,  1273,  1274,  1278,  1279,  1280,  1284,  1285,  1286,
    1287,  1288,  1289,  1290,  1291,  1292,  1296,  1297,  1298,  1302,
    1303,  1307,  1308,  1309,  1310,  1311,  1312,  1313,  1317,  1318,
    1319,  1323,  1329,  1341,  1355,  1388,  1395,  1410,  1419,  1431,
    1432,  1436,  1444,  1453,  1464,  1474,  1484,  1502,  1520,  1521,
    1522,  1523,  1524,  1528,  1538,  1542,  1564,  1571,  1575,  1579,
    1586,  1595,  1608,  1616
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "IDENTIFIER", "INTCONSTANT", "BIN_MASK",
  "FLOATCONSTANT", "STRING_LITERAL", "SIZEOF", "PTR_OP", "INC_OP",
  "DEC_OP", "LEFT_OP", "RIGHT_OP", "LE_OP", "GE_OP", "EQ_OP", "NE_OP",
  "L_AND", "L_OR", "MUL_ASSIGN", "DIV_ASSIGN", "MOD_ASSIGN", "ADD_ASSIGN",
  "SUB_ASSIGN", "LEFT_ASSIGN", "RIGHT_ASSIGN", "AND_ASSIGN", "XOR_ASSIGN",
  "OR_ASSIGN", "TYPE_NAME", "TYPEDEF", "PRAGMA", "EXTERN", "STATIC",
  "AUTO", "REGISTER", "STARPU_CODELET", "STARPU_FUNC",
  "DIR_DAGUE_DATA_COLOCATED", "DIR_DAGUE_INVARIANT",
  "DIR_DAGUE_TASK_START", "CHAR", "SHORT", "INT", "LONG", "SIGNED",
  "UNSIGNED", "FLOAT", "DOUBLE", "CONST", "VOLATILE", "VOID", "INT8",
  "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64",
  "INTPTR", "UINTPTR", "INTMAX", "UINTMAX", "PLASMA_COMPLEX32_T",
  "PLASMA_COMPLEX64_T", "PLASMA_ENUM", "PLASMA_REQUEST", "PLASMA_DESC",
  "PLASMA_SEQUENCE", "STRUCT", "UNION", "ENUM", "ELLIPSIS", "CASE",
  "DEFAULT", "IF", "ELSE", "SWITCH", "WHILE", "DO", "FOR", "GOTO",
  "CONTINUE", "BREAK", "RETURN", "'('", "')'", "'['", "']'", "'.'", "','",
  "'&'", "'*'", "'+'", "'-'", "'~'", "'!'", "'/'", "'%'", "'<'", "'>'",
  "'^'", "'|'", "'?'", "':'", "'='", "';'", "'{'", "'}'", "$accept",
  "starting_decl", "primary_expression", "postfix_expression",
  "argument_expression_list", "unary_expression", "unary_operator",
  "cast_expression", "multiplicative_expression", "additive_expression",
  "shift_expression", "relational_expression", "equality_expression",
  "and_expression", "exclusive_or_expression", "inclusive_or_expression",
  "logical_and_expression", "logical_or_expression",
  "conditional_expression", "assignment_expression", "assignment_operator",
  "expression", "constant_expression", "declaration", "typedef_specifier",
  "pragma_options", "task_arguments", "pragma_parameters",
  "pragma_specifier", "declaration_specifiers", "init_declarator_list",
  "init_declarator", "storage_class_specifier", "type_specifier",
  "struct_or_union_specifier", "struct_or_union",
  "struct_declaration_list", "struct_declaration", "starpu_function_list",
  "starpu_codelet_params", "starpu_codelet_params_list", "starpu_codelet",
  "specifier_qualifier_list", "struct_declarator_list",
  "struct_declarator", "enum_specifier", "enumerator_list", "enumerator",
  "type_qualifier", "declarator", "direct_declarator", "pointer",
  "type_qualifier_list", "parameter_type_list", "parameter_list",
  "parameter_declaration", "identifier_list", "type_name",
  "abstract_declarator", "direct_abstract_declarator", "initializer",
  "initializer_list", "statement", "labeled_statement",
  "compound_statement", "declaration_list", "statement_list",
  "expression_statement", "selection_statement", "iteration_statement",
  "jump_statement", "translation_unit", "external_declaration",
  "function_definition", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,    40,    41,    91,
      93,    46,    44,    38,    42,    43,    45,   126,    33,    47,
      37,    60,    62,    94,   124,    63,    58,    61,    59,   123,
     125
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   111,   112,   113,   113,   113,   113,   113,   114,   114,
     114,   114,   114,   114,   114,   114,   115,   115,   116,   116,
     116,   116,   116,   116,   117,   117,   117,   117,   117,   117,
     118,   118,   119,   119,   119,   119,   120,   120,   120,   121,
     121,   121,   122,   122,   122,   122,   122,   123,   123,   123,
     124,   124,   125,   125,   126,   126,   127,   127,   128,   128,
     129,   129,   130,   130,   131,   131,   131,   131,   131,   131,
     131,   131,   131,   131,   131,   132,   132,   133,   134,   134,
     135,   136,   136,   137,   137,   138,   138,   138,   139,   139,
     139,   139,   140,   140,   140,   140,   140,   140,   141,   141,
     142,   142,   143,   143,   143,   143,   144,   144,   144,   144,
     144,   144,   144,   144,   144,   144,   144,   144,   144,   144,
     144,   144,   144,   144,   144,   144,   144,   144,   144,   144,
     144,   144,   144,   144,   144,   144,   145,   145,   145,   146,
     146,   147,   147,   148,   149,   149,   150,   150,   150,   151,
     151,   152,   153,   153,   153,   153,   154,   154,   155,   156,
     156,   156,   157,   157,   158,   158,   159,   159,   160,   160,
     161,   161,   161,   161,   161,   161,   161,   162,   162,   162,
     162,   163,   163,   164,   164,   165,   165,   166,   166,   166,
     167,   167,   168,   168,   169,   169,   169,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   171,   171,   171,   172,
     172,   173,   173,   173,   173,   173,   173,   173,   174,   174,
     174,   175,   175,   175,   175,   176,   176,   177,   177,   178,
     178,   179,   179,   179,   180,   180,   180,   180,   181,   181,
     181,   181,   181,   182,   182,   183,   183,   183,   183,   183,
     184,   184,   184,   184
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     3,     1,     4,
       3,     4,     3,     3,     2,     2,     1,     3,     1,     2,
       2,     2,     2,     4,     1,     1,     1,     1,     1,     1,
       1,     4,     1,     3,     3,     3,     1,     3,     3,     1,
       3,     3,     1,     3,     3,     3,     3,     1,     3,     3,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     5,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     1,     2,     3,
       4,     1,     3,     1,     3,     1,     1,     2,     3,     3,
       3,     4,     1,     2,     1,     2,     1,     2,     1,     3,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     5,     4,     2,     1,
       1,     1,     2,     3,     1,     3,     5,     4,     4,     1,
       3,     8,     2,     1,     2,     1,     1,     3,     1,     4,
       5,     2,     1,     3,     1,     3,     1,     1,     2,     1,
       1,     3,     4,     3,     4,     4,     3,     1,     2,     2,
       3,     1,     2,     1,     3,     1,     3,     2,     2,     1,
       1,     3,     1,     2,     1,     1,     2,     3,     2,     3,
       3,     4,     2,     3,     3,     4,     1,     3,     4,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     3,     4,
       3,     2,     3,     3,     4,     1,     2,     1,     2,     1,
       2,     5,     7,     5,     5,     7,     6,     7,     3,     2,
       2,     2,     3,     1,     2,     1,     1,     1,     1,     1,
       4,     3,     3,     2
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,   170,   135,     0,     0,   102,   103,   104,   105,   107,
     108,   109,   110,   125,   126,   123,   124,   166,   167,   106,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   129,   130,   131,   132,   133,   134,   139,   140,
       0,     0,   177,     0,   246,   247,   248,     0,    92,    94,
     127,     0,   249,   128,    96,     0,   169,     0,     2,   243,
     245,   139,     0,     0,     0,     0,     0,     0,   161,     0,
       0,   181,   179,   178,     1,    78,     0,    98,   100,    93,
      95,   138,     0,    97,     0,   225,     0,   253,     0,     0,
       0,   168,   244,     0,    85,    86,    88,    90,     3,     4,
       5,     6,     0,     0,     0,     0,    24,    25,    26,    27,
      28,    29,     8,    18,    30,     0,    32,    36,    39,    42,
      47,    50,    52,    54,    56,    58,    60,    62,    75,    89,
       0,     0,     0,   164,     0,   162,   171,   182,   180,     0,
      79,     0,   251,     0,     0,   153,     0,   141,     0,   155,
       3,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   229,   221,     0,   217,   227,   211,   212,     0,
       0,   213,   214,   215,   216,   100,   226,   252,   190,   176,
     189,     0,   183,   185,     0,   173,    30,    77,     0,    80,
      87,     0,    22,     0,    19,    20,     0,   192,     0,     0,
      14,    15,     0,     0,     0,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    64,     0,    21,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    81,    83,
      91,     0,     0,     0,     0,   159,    99,     0,   206,   101,
     250,     0,   152,   137,   142,     0,   156,   158,   154,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   239,   240,
     241,     0,   230,   223,     0,   222,   228,     0,     0,   187,
     194,   188,   195,   174,     0,   175,     0,   172,     0,     7,
       0,   194,   193,     0,    13,    10,     0,    16,     0,    12,
      63,    33,    34,    35,    37,    38,    40,    41,    45,    46,
      43,    44,    48,    49,    51,    53,    55,    57,    59,     0,
      76,     0,     0,     0,   160,   165,   163,   209,     0,   136,
       0,   143,   218,     0,   220,     0,     0,     0,     0,     0,
     238,   242,   224,   202,     0,     0,   198,     0,   196,     0,
       0,   184,   186,   191,    23,    31,    11,     0,     9,     0,
      82,    84,     0,     0,   149,     0,     0,   207,   157,   219,
       0,     0,     0,     0,     0,   203,   197,   199,   204,     0,
     200,     0,    17,    61,     0,     0,     0,     0,   208,   210,
     231,   233,   234,     0,     0,     0,   205,   201,     0,     0,
     150,   151,     0,     0,   236,     0,   144,     0,   147,   148,
     232,   235,   237,     0,   146,   145
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    43,   112,   113,   296,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     216,   164,   188,    85,    45,   239,   240,    96,   165,    86,
      76,    77,    48,    49,    50,    51,   146,   147,   407,   364,
     365,    52,   148,   255,   256,    53,   134,   135,    54,    55,
      56,    57,    73,   344,   182,   183,   184,   198,   345,   282,
     249,   328,   166,   167,   168,    88,   170,   171,   172,   173,
     174,    58,    59,    60
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -244
static const yytype_int16 yypact[] =
{
     932,  -244,  -244,  1549,   174,  -244,  -244,  -244,  -244,  -244,
    -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,
    -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,
    -244,  -244,  -244,  -244,  -244,  -244,  -244,  -244,   -19,  -244,
       9,    34,    -6,    25,  -244,  -244,  -244,    17,  1549,  1549,
    -244,    13,  -244,  -244,  1549,  1327,     3,    19,  -244,   932,
    -244,  -244,    47,    48,    48,   824,    70,   134,    64,   175,
      10,  -244,  -244,    -6,  -244,  -244,   -36,  -244,  1259,  -244,
    -244,    88,  1593,  -244,   308,  -244,    17,  -244,  1327,  1074,
     704,     3,  -244,   108,    48,  -244,  -244,  -244,  -244,  -244,
    -244,  -244,   836,   859,   859,   692,  -244,  -244,  -244,  -244,
    -244,  -244,  -244,   182,   851,   824,  -244,   124,   132,   243,
      15,   241,   154,   165,   149,   252,    14,  -244,  -244,   183,
     277,   178,   175,   179,   -52,  -244,  -244,  -244,  -244,    34,
    -244,   422,  -244,  1327,  1593,  1593,  1121,  -244,    34,  1593,
     176,   824,   186,   194,   201,   202,   596,   212,   297,   193,
     195,   200,  -244,  -244,   -22,  -244,  -244,  -244,  -244,   404,
     500,  -244,  -244,  -244,  -244,   197,  -244,  -244,  -244,  -244,
      77,   214,   213,  -244,    44,  -244,  -244,  -244,   216,  -244,
    -244,   692,  -244,   824,  -244,  -244,    73,    87,   219,   310,
    -244,  -244,   632,   824,   314,  -244,  -244,  -244,  -244,  -244,
    -244,  -244,  -244,  -244,  -244,  -244,   824,  -244,   824,   824,
     824,   824,   824,   824,   824,   824,   824,   824,   824,   824,
     824,   824,   824,   824,   824,   824,   824,   824,   215,   228,
    -244,   217,   -50,   824,   175,  -244,  -244,   422,  -244,  -244,
    -244,  1190,  -244,  -244,  -244,    -9,  -244,  -244,  -244,   596,
     218,   596,   824,   824,   824,   242,   554,   220,  -244,  -244,
    -244,    -7,  -244,  -244,   536,  -244,  -244,  1003,   763,  -244,
      81,  -244,   107,  -244,  1504,  -244,   320,  -244,   239,  -244,
    1395,   150,  -244,   824,  -244,  -244,    95,  -244,   160,  -244,
    -244,  -244,  -244,  -244,   124,   124,   132,   132,   243,   243,
     243,   243,    15,    15,   241,   154,   165,   149,   252,     1,
    -244,   277,   277,    -4,  -244,  -244,  -244,  -244,   -46,  -244,
      34,  -244,  -244,   596,  -244,    97,    98,   117,   250,   554,
    -244,  -244,  -244,  -244,   251,   258,  -244,   240,   107,  1457,
     800,  -244,  -244,  -244,  -244,  -244,  -244,   824,  -244,   824,
    -244,  -244,   222,   342,   290,   276,   238,  -244,  -244,  -244,
     596,   596,   596,   824,   812,  -244,  -244,  -244,  -244,   309,
    -244,   306,  -244,  -244,   289,   292,    -4,   301,  -244,  -244,
     322,  -244,  -244,   137,   596,   138,  -244,  -244,   410,   257,
    -244,  -244,   596,   311,  -244,   596,   328,   312,  -244,  -244,
    -244,  -244,  -244,   410,  -244,  -244
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -244,  -244,  -244,  -244,  -244,   -76,  -244,  -105,    41,    42,
       8,    49,   190,   191,   198,   207,   189,  -244,   -83,  -135,
    -244,   -64,  -138,    32,  -244,   106,   113,   -29,    36,     0,
    -244,   303,  -244,   -67,  -244,  -244,   299,  -125,    31,  -244,
      59,  -244,   -88,  -244,   148,  -244,   350,   248,   -11,   -39,
     -48,   -18,  -244,   -85,  -244,   209,  -244,   304,  -128,  -217,
    -236,  -244,  -151,  -244,   -12,    85,   325,  -243,  -244,  -244,
    -244,   437,  -244,  -244
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      47,   129,    70,    62,   181,   265,   248,   187,    78,    91,
     217,   327,    68,   260,   186,   145,    81,   197,    67,   276,
       1,   254,     1,   339,    72,    74,   192,   194,   195,   225,
     226,    71,    44,   235,   362,    97,    46,     1,   145,   186,
     244,   196,   244,    87,    17,    18,   366,   175,    79,    80,
      93,    94,   281,    95,    83,   138,   139,   252,   245,    47,
     324,   258,   137,   348,   367,   190,   142,   297,   187,   292,
     237,   149,   140,   130,   348,   186,   177,   145,   145,   145,
       1,   300,   145,   330,     1,   237,   272,   363,    42,   180,
      89,    44,    90,   237,   149,    46,   374,   271,   136,   331,
     175,   341,   320,   197,    41,   325,    41,   359,   332,   257,
     334,    42,   248,   301,   302,   303,   227,   228,    69,   236,
     176,    41,    82,   276,   145,    75,   254,   196,    42,   196,
     389,   250,   285,   149,   149,   149,   286,   131,   149,   298,
     347,   279,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     187,   289,   280,   143,   277,   237,   278,   186,   277,   169,
     278,    42,   319,   132,   290,   176,   278,    63,   133,   291,
     149,    42,   369,   356,   145,   370,   371,   357,   355,   237,
     237,   199,   200,   201,   349,   187,   350,   144,   335,   336,
     337,   176,   186,    98,    99,   372,   100,   101,   102,   237,
     103,   104,   381,    64,    65,    66,   189,   186,   218,   390,
     391,   392,   382,   219,   220,   403,   405,   221,   222,   237,
     237,   248,    91,   308,   309,   310,   311,   290,    70,   278,
     149,    98,    99,   404,   100,   101,   102,   231,   103,   104,
     358,   410,   237,   233,   412,   223,   224,   229,   230,   280,
     408,   409,   304,   305,   379,   306,   307,   187,   232,   202,
     234,   203,   291,   204,   186,   237,   383,   180,   312,   313,
     238,   262,   259,   186,   180,   241,   243,   105,   263,   264,
     180,   257,   261,   106,   107,   108,   109,   110,   111,   266,
     267,   268,   283,   269,   141,   284,   287,   293,   270,   393,
     395,   150,    99,   294,   100,   101,   102,   299,   103,   104,
     322,   321,   338,   353,   333,   105,   323,   354,   340,   384,
     377,   106,   107,   108,   109,   110,   111,   373,     2,   375,
       4,     5,     6,     7,     8,   385,   376,   247,   388,   180,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    61,
      39,    40,   386,   151,   152,   153,   387,   154,   155,   156,
     157,   158,   159,   160,   161,   105,   397,   396,   398,   399,
     402,   106,   107,   108,   109,   110,   111,   150,    99,   401,
     100,   101,   102,   406,   103,   104,   162,    84,   163,   411,
     413,   314,   414,   315,   318,    98,    99,   360,   100,   101,
     102,   316,   103,   104,     2,   361,     4,     5,     6,     7,
       8,   317,   246,   251,   415,   400,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    61,    39,    40,   368,   151,
     152,   153,   242,   154,   155,   156,   157,   158,   159,   160,
     161,   105,   326,   352,   274,   288,    92,   106,   107,   108,
     109,   110,   111,   150,    99,     0,   100,   101,   102,   105,
     103,   104,   162,    84,   273,   106,   107,   108,   109,   110,
     111,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   247,     4,     0,     0,     0,     0,     0,     0,   150,
      99,     0,   100,   101,   102,     0,   103,   104,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    98,    99,     0,
     100,   101,   102,     0,   103,   104,     0,     0,     4,     0,
       0,     0,     0,     0,     0,   151,   152,   153,     0,   154,
     155,   156,   157,   158,   159,   160,   161,   105,     0,     0,
       0,     0,     0,   106,   107,   108,   109,   110,   111,   150,
      99,     0,   100,   101,   102,     0,   103,   104,   162,    84,
     275,   151,   152,   153,     0,   154,   155,   156,   157,   158,
     159,   160,   161,   105,     0,     0,     0,     0,     4,   106,
     107,   108,   109,   110,   111,    98,    99,     0,   100,   101,
     102,   105,   103,   104,   162,    84,   342,   106,   107,   108,
     109,   110,   111,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   162,     0,     0,     0,     0,     0,     0,     0,
       0,   151,   152,   153,     0,   154,   155,   156,   157,   158,
     159,   160,   161,   105,     0,     0,     0,     0,     0,   106,
     107,   108,   109,   110,   111,    98,    99,     0,   100,   101,
     102,     0,   103,   104,   162,    84,     0,    98,    99,     0,
     100,   101,   102,     0,   103,   104,     0,     0,     0,   105,
     295,     0,     2,     0,     0,   106,   107,   108,   109,   110,
     111,     0,     0,     0,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    61,    39,    40,    98,    99,     0,   100,
     101,   102,     0,   103,   104,     0,     0,     0,     0,   105,
       0,     0,     0,     0,     0,   106,   107,   108,   109,   110,
     111,   105,     0,     0,   185,     0,     0,   106,   107,   108,
     109,   110,   111,    98,    99,     0,   100,   101,   102,     0,
     103,   104,     0,     0,     0,    98,    99,     0,   100,   101,
     102,     0,   103,   104,     0,     0,     0,    98,    99,     0,
     100,   101,   102,     0,   103,   104,     0,     0,     0,    98,
      99,     0,   100,   101,   102,     0,   103,   104,     0,     0,
     105,     0,     0,   346,     0,     0,   106,   107,   108,   109,
     110,   111,    98,    99,     0,   100,   101,   102,     0,   103,
     104,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,     0,     0,     0,     0,     0,     0,   105,     0,     0,
     380,     0,     0,   106,   107,   108,   109,   110,   111,   105,
     394,     0,     0,     0,     0,   106,   107,   108,   109,   110,
     111,   105,     0,     0,     0,     0,     0,   106,   107,   108,
     109,   110,   111,   191,     0,     0,     0,     0,     0,   106,
     107,   108,   109,   110,   111,     1,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   193,     0,     0,     0,
       0,     0,   106,   107,   108,   109,   110,   111,   215,     0,
       0,     0,     2,     3,     4,     5,     6,     7,     8,     0,
       0,     0,     0,     0,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,     1,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    41,
       0,     0,     0,     0,     0,     0,    42,     0,     0,     0,
       0,     0,     0,     2,     0,     0,     5,     6,     7,     8,
       0,     0,     0,     0,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    61,    39,    40,   178,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     277,   343,   278,     0,     0,     0,     0,    42,     0,     0,
       0,     0,     0,     0,     2,     0,     0,     5,     6,     7,
       8,     0,     0,     0,     0,     0,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    61,    39,    40,     0,     0,
       0,     2,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   179,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    61,    39,    40,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       2,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   253,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    61,    39,    40,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     2,
       0,     0,     5,     6,     7,     8,     0,     0,     0,     0,
     329,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      61,    39,    40,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     2,     0,     0,
       5,     6,     7,     8,     0,     0,   141,     0,    84,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    61,    39,
      40,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     2,     0,     0,     5,     6,
       7,     8,     0,     0,     0,     0,    84,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    61,    39,    40,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   290,   343,   278,     0,     0,     2,     0,    42,
       5,     6,     7,     8,     0,     0,     0,     0,     0,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    61,    39,
      40,     0,     0,     0,     2,     0,     0,     5,     6,     7,
       8,     0,     0,     0,     0,   378,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    61,    39,    40,   351,     2,
       0,     0,     5,     6,     7,     8,     0,     0,     0,     0,
       0,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      61,    39,    40,     2,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    61,    39,    40
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-244))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       0,    65,    41,     3,    89,   156,   141,    90,    47,    57,
     115,   247,     3,   151,    90,    82,     3,   105,    37,   170,
       3,   146,     3,   266,    42,     0,   102,   103,   104,    14,
      15,    42,     0,    19,    38,    64,     0,     3,   105,   115,
      92,   105,    92,    55,    50,    51,    92,    86,    48,    49,
       3,     3,   180,     5,    54,    73,    92,   145,   110,    59,
     110,   149,    73,   280,   110,    94,    78,   202,   151,   197,
      92,    82,   108,     3,   291,   151,    88,   144,   145,   146,
       3,   216,   149,    92,     3,    92,   108,    91,    94,    89,
      87,    59,    89,    92,   105,    59,   339,   161,    88,   108,
     139,   108,   237,   191,    87,   243,    87,   106,   259,   148,
     261,    94,   247,   218,   219,   220,   101,   102,   109,   105,
      88,    87,   109,   274,   191,   108,   251,   191,    94,   193,
     366,   143,    88,   144,   145,   146,    92,     3,   149,   203,
     278,   180,   218,   219,   220,   221,   222,   223,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   234,   235,
     243,    88,   180,    78,    87,    92,    89,   243,    87,    84,
      89,    94,   236,   109,    87,   143,    89,     3,     3,   197,
     191,    94,   333,    88,   251,    88,    88,    92,   293,    92,
      92,     9,    10,    11,    87,   278,    89,   109,   262,   263,
     264,   169,   278,     3,     4,    88,     6,     7,     8,    92,
      10,    11,   350,    39,    40,    41,   108,   293,    94,   370,
     371,   372,   357,    99,   100,    88,    88,    95,    96,    92,
      92,   366,   280,   225,   226,   227,   228,    87,   277,    89,
     251,     3,     4,   394,     6,     7,     8,    93,    10,    11,
      90,   402,    92,   104,   405,    12,    13,    16,    17,   277,
       3,     4,   221,   222,   349,   223,   224,   350,   103,    87,
      18,    89,   290,    91,   350,    92,   359,   277,   229,   230,
       3,    87,   106,   359,   284,   107,   107,    87,    87,    87,
     290,   330,   106,    93,    94,    95,    96,    97,    98,    87,
       3,   108,    88,   108,   107,    92,    90,    88,   108,   373,
     374,     3,     4,     3,     6,     7,     8,     3,    10,    11,
      92,   106,    80,     3,   106,    87,   109,    88,   108,   107,
      90,    93,    94,    95,    96,    97,    98,    87,    30,    88,
      32,    33,    34,    35,    36,     3,    88,   109,   110,   349,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    92,    75,    76,    77,   110,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    90,    88,   109,   107,
      78,    93,    94,    95,    96,    97,    98,     3,     4,   108,
       6,     7,     8,     3,    10,    11,   108,   109,   110,   108,
      92,   231,   110,   232,   235,     3,     4,   321,     6,     7,
       8,   233,    10,    11,    30,   322,    32,    33,    34,    35,
      36,   234,   139,   144,   413,   386,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,   330,    75,
      76,    77,   132,    79,    80,    81,    82,    83,    84,    85,
      86,    87,   244,   284,   169,   191,    59,    93,    94,    95,
      96,    97,    98,     3,     4,    -1,     6,     7,     8,    87,
      10,    11,   108,   109,   110,    93,    94,    95,    96,    97,
      98,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   109,    32,    -1,    -1,    -1,    -1,    -1,    -1,     3,
       4,    -1,     6,     7,     8,    -1,    10,    11,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,    -1,
       6,     7,     8,    -1,    10,    11,    -1,    -1,    32,    -1,
      -1,    -1,    -1,    -1,    -1,    75,    76,    77,    -1,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      -1,    -1,    -1,    93,    94,    95,    96,    97,    98,     3,
       4,    -1,     6,     7,     8,    -1,    10,    11,   108,   109,
     110,    75,    76,    77,    -1,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    -1,    -1,    32,    93,
      94,    95,    96,    97,    98,     3,     4,    -1,     6,     7,
       8,    87,    10,    11,   108,   109,   110,    93,    94,    95,
      96,    97,    98,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   108,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    75,    76,    77,    -1,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    95,    96,    97,    98,     3,     4,    -1,     6,     7,
       8,    -1,    10,    11,   108,   109,    -1,     3,     4,    -1,
       6,     7,     8,    -1,    10,    11,    -1,    -1,    -1,    87,
      88,    -1,    30,    -1,    -1,    93,    94,    95,    96,    97,
      98,    -1,    -1,    -1,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,     3,     4,    -1,     6,
       7,     8,    -1,    10,    11,    -1,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    93,    94,    95,    96,    97,
      98,    87,    -1,    -1,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,     3,     4,    -1,     6,     7,     8,    -1,
      10,    11,    -1,    -1,    -1,     3,     4,    -1,     6,     7,
       8,    -1,    10,    11,    -1,    -1,    -1,     3,     4,    -1,
       6,     7,     8,    -1,    10,    11,    -1,    -1,    -1,     3,
       4,    -1,     6,     7,     8,    -1,    10,    11,    -1,    -1,
      87,    -1,    -1,    90,    -1,    -1,    93,    94,    95,    96,
      97,    98,     3,     4,    -1,     6,     7,     8,    -1,    10,
      11,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    -1,    -1,    -1,    -1,    -1,    -1,    87,    -1,    -1,
      90,    -1,    -1,    93,    94,    95,    96,    97,    98,    87,
      88,    -1,    -1,    -1,    -1,    93,    94,    95,    96,    97,
      98,    87,    -1,    -1,    -1,    -1,    -1,    93,    94,    95,
      96,    97,    98,    87,    -1,    -1,    -1,    -1,    -1,    93,
      94,    95,    96,    97,    98,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    87,    -1,    -1,    -1,
      -1,    -1,    93,    94,    95,    96,    97,    98,   107,    -1,
      -1,    -1,    30,    31,    32,    33,    34,    35,    36,    -1,
      -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      87,    88,    89,    -1,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    30,    -1,    -1,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    -1,    -1,
      -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    88,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   110,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    30,
      -1,    -1,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
     110,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    30,    -1,    -1,
      33,    34,    35,    36,    -1,    -1,   107,    -1,   109,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    30,    -1,    -1,    33,    34,
      35,    36,    -1,    -1,    -1,    -1,   109,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    87,    88,    89,    -1,    -1,    30,    -1,    94,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    -1,    -1,    -1,    30,    -1,    -1,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    88,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    30,
      -1,    -1,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    30,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,    30,    31,    32,    33,    34,    35,    36,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    87,    94,   112,   134,   135,   139,   140,   143,   144,
     145,   146,   152,   156,   159,   160,   161,   162,   182,   183,
     184,    71,   140,     3,    39,    40,    41,    37,     3,   109,
     160,   159,   162,   163,     0,   108,   141,   142,   160,   140,
     140,     3,   109,   140,   109,   134,   140,   175,   176,    87,
      89,   161,   182,     3,     3,     5,   138,   138,     3,     4,
       6,     7,     8,    10,    11,    87,    93,    94,    95,    96,
      97,    98,   113,   114,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   132,
       3,     3,   109,     3,   157,   158,    88,   159,   162,    92,
     108,   107,   175,   176,   109,   144,   147,   148,   153,   159,
       3,    75,    76,    77,    79,    80,    81,    82,    83,    84,
      85,    86,   108,   110,   132,   139,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   160,   134,   175,     3,    88,
     140,   164,   165,   166,   167,    90,   116,   129,   133,   108,
     138,    87,   116,    87,   116,   116,   132,   153,   168,     9,
      10,    11,    87,    89,    91,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,   107,   131,   118,    94,    99,
     100,    95,    96,    12,    13,    14,    15,   101,   102,    16,
      17,    93,   103,   104,    18,    19,   105,    92,     3,   136,
     137,   107,   157,   107,    92,   110,   142,   109,   130,   171,
     175,   147,   153,   110,   148,   154,   155,   160,   153,   106,
     133,   106,    87,    87,    87,   173,    87,     3,   108,   108,
     108,   132,   108,   110,   177,   110,   173,    87,    89,   160,
     162,   169,   170,    88,    92,    88,    92,    90,   168,    88,
      87,   162,   169,    88,     3,    88,   115,   130,   132,     3,
     130,   118,   118,   118,   119,   119,   120,   120,   121,   121,
     121,   121,   122,   122,   123,   124,   125,   126,   127,   132,
     130,   106,    92,   109,   110,   133,   158,   171,   172,   110,
      92,   108,   173,   106,   173,   132,   132,   132,    80,   178,
     108,   108,   110,    88,   164,   169,    90,   133,   170,    87,
      89,    74,   166,     3,    88,   118,    88,    92,    90,   106,
     136,   137,    38,    91,   150,   151,    92,   110,   155,   173,
      88,    88,    88,    87,   178,    88,    88,    90,    88,   164,
      90,   133,   130,   129,   107,     3,    92,   110,   110,   171,
     173,   173,   173,   132,    88,   132,    88,    90,   109,   107,
     151,   108,    78,    88,   173,    88,     3,   149,     3,     4,
     173,   108,   173,    92,   110,   149
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

/* Line 1806 of yacc.c  */
#line 228 "src/q2j.y"
    {	      
	      rename_induction_variables(trans_list->tr->node);
	      convert_OUTPUT_to_INOUT(trans_list->tr->node);
	      if( _q2j_add_phony_tasks )
		  add_entry_and_exit_task_loops(trans_list->tr->node);

	      analyze_deps(trans_list->tr->node);
	  }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 239 "src/q2j.y"
    { 
//              char *name=$1.u.var_name;
//              char *var_type = st_type_of_variable(name, $1.symtab);
//              if( NULL == var_type ){
//                  printf("No entry for \"%s\" in symbol table\n",name);
//              }else{
//                  printf("\"%s\" is of type \"%s\"\n",name, var_type);
//              }
          }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 251 "src/q2j.y"
    {(yyval.node) = (yyvsp[(2) - (3)].node);}
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 256 "src/q2j.y"
    { 
          (yyval.node) = (yyvsp[(1) - (1)].node);
          }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 260 "src/q2j.y"
    {

              if( ARRAY == (yyvsp[(1) - (4)].node).type ){
                  int count;
                  (yyval.node) = (yyvsp[(1) - (4)].node);
                  count = ++((yyval.node).u.kids.kid_count);
                  (yyval.node).u.kids.kids = (node_t **)realloc( (yyval.node).u.kids.kids, count*sizeof(node_t *) );
                  (yyval.node).u.kids.kids[count-1] = node_to_ptr((yyvsp[(3) - (4)].node));
              }else{
                  (yyval.node).type = ARRAY;
                  (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  (yyval.node).u.kids.kid_count = 2;
                  (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (4)].node));
                  (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (4)].node));
//		  st_insert_new_variable($1.u.var_name, "int");
              }

          }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 279 "src/q2j.y"
    {
              (yyval.node).type = FCALL;
              (yyval.node).lineno = yyget_lineno();

              (yyval.node).u.kids.kids = (node_t **)calloc(1, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 1;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
          }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 288 "src/q2j.y"
    {
              node_t *tmp, *flwr;
              int i, count = 0;

              (yyval.node).type = FCALL;
              (yyval.node).lineno = yyget_lineno();

              for(tmp=(yyvsp[(3) - (4)].node).next; NULL != tmp ; flwr=tmp, tmp=tmp->prev){
                  count++;
              }
              (yyval.node).u.kids.kids = (node_t **)calloc(count+1, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = count+1;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (4)].node));

              /* Unchain the temporary list of arguments and make them the */
              /* kids of this FCALL */
              for(i=1; i<count+1; ++i){
                  assert(flwr != NULL);
                  (yyval.node).u.kids.kids[i] = flwr;
                  flwr = flwr->next;
              }
          }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 311 "src/q2j.y"
    {
              (yyval.node).type = S_U_MEMBER;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 319 "src/q2j.y"
    {
              (yyval.node).type = PTR_OP;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 327 "src/q2j.y"
    {
              (yyval.node).type = EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (2)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 335 "src/q2j.y"
    {
              (yyval.node).type = EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (2)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 351 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (1)].node));
              tmp->prev = NULL;
              tmp->next = NULL;
              (yyval.node).next = tmp;
          }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 359 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(3) - (3)].node));
              tmp->next = NULL;
              tmp->prev = (yyvsp[(1) - (3)].node).next;
              tmp->prev->next = tmp;
              (yyval.node).next = tmp;
          }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 371 "src/q2j.y"
    { (yyval.node) = (yyvsp[(1) - (1)].node); }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 373 "src/q2j.y"
    {
              (yyval.node).type = EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (2)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 381 "src/q2j.y"
    {
              (yyval.node).type = EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (2)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 389 "src/q2j.y"
    {
              (yyval.node).type = EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (2)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 397 "src/q2j.y"
    {
              (yyval.node).type = SIZEOF;
              (yyval.node).u.kids.kid_count = 1;
              (yyval.node).u.kids.kids = (node_t **)calloc(1,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(2) - (2)].node));
          }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 404 "src/q2j.y"
    {
              (yyval.node).type = SIZEOF;
              (yyval.node).u.kids.kid_count = 0;
              (yyval.node).u.var_name = strdup((yyvsp[(3) - (4)].string));
          }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 412 "src/q2j.y"
    {(yyval.node).type = ADDR_OF;}
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 413 "src/q2j.y"
    {(yyval.node).type = STAR;}
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 414 "src/q2j.y"
    {(yyval.node).type = PLUS;}
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 415 "src/q2j.y"
    {(yyval.node).type = MINUS;}
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 416 "src/q2j.y"
    {(yyval.node).type = TILDA;}
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 417 "src/q2j.y"
    {(yyval.node).type = BANG;}
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 423 "src/q2j.y"
    {
            (yyval.node) = (yyvsp[(4) - (4)].node);
            (yyval.node).var_type = strdup((yyvsp[(2) - (4)].string));
	    (yyval.node).cast = 1;
	  }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 433 "src/q2j.y"
    {
              (yyval.node).type = MUL;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 441 "src/q2j.y"
    {
              (yyval.node).type = DIV;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 449 "src/q2j.y"
    {
              (yyval.node).type = MOD;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 461 "src/q2j.y"
    {
              (yyval.node).type = ADD;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 469 "src/q2j.y"
    {
              (yyval.node).type = SUB;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 481 "src/q2j.y"
    {
              (yyval.node).type = LSHIFT;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 489 "src/q2j.y"
    {
              (yyval.node).type = RSHIFT;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 501 "src/q2j.y"
    {
              (yyval.node).type = LT;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 509 "src/q2j.y"
    {
              (yyval.node).type = GT;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 517 "src/q2j.y"
    {
              (yyval.node).type = LE;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 525 "src/q2j.y"
    {
              (yyval.node).type = GE;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 537 "src/q2j.y"
    {
              (yyval.node).type = EQ_OP;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 545 "src/q2j.y"
    {
              (yyval.node).type = NE_OP;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 557 "src/q2j.y"
    {
              (yyval.node).type = B_AND;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 569 "src/q2j.y"
    {
              (yyval.node).type = B_XOR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 581 "src/q2j.y"
    {
              (yyval.node).type = B_OR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 593 "src/q2j.y"
    {
              (yyval.node).type = (yyvsp[(2) - (3)].node).type;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 605 "src/q2j.y"
    {
              (yyval.node).type = (yyvsp[(2) - (3)].node).type;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 615 "src/q2j.y"
    { (yyval.node) = (yyvsp[(1) - (1)].node); }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 617 "src/q2j.y"
    {
            (yyval.node).type = COND;
            (yyval.node).u.kids.kid_count = 3;
            (yyval.node).u.kids.kids = (node_t **)calloc(3, sizeof(node_t *));
            (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (5)].node));
            (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (5)].node));
            (yyval.node).u.kids.kids[2] = node_to_ptr((yyvsp[(5) - (5)].node));
          }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 629 "src/q2j.y"
    { 
            (yyval.node) = (yyvsp[(1) - (1)].node);
          }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 633 "src/q2j.y"
    {
	      (yyval.node).type = (yyvsp[(2) - (3)].node).type;
	      (yyval.node).u.kids.kid_count = 2;
	      (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
	      (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
	      (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 643 "src/q2j.y"
    {(yyval.node).type = ASSIGN; }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 644 "src/q2j.y"
    {(yyval.node).type = MUL_ASSIGN;}
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 645 "src/q2j.y"
    {(yyval.node).type = DIV_ASSIGN;}
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 646 "src/q2j.y"
    {(yyval.node).type = MOD_ASSIGN;}
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 647 "src/q2j.y"
    {(yyval.node).type = ADD_ASSIGN;}
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 648 "src/q2j.y"
    {(yyval.node).type = SUB_ASSIGN;}
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 649 "src/q2j.y"
    {(yyval.node).type = LEFT_ASSIGN;}
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 650 "src/q2j.y"
    {(yyval.node).type = RIGHT_ASSIGN;}
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 651 "src/q2j.y"
    {(yyval.node).type = AND_ASSIGN;}
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 652 "src/q2j.y"
    {(yyval.node).type = XOR_ASSIGN;}
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 653 "src/q2j.y"
    {(yyval.node).type = OR_ASSIGN;}
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 659 "src/q2j.y"
    {
              (yyval.node).type = COMMA_EXPR;
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 670 "src/q2j.y"
    {
	      (yyval.node) = (yyvsp[(1) - (1)].node);
	  }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 677 "src/q2j.y"
    {
//              fprintf(stderr,"DEBUG: Is this correct C?: \"%s;\"\n",(char *)$1);
          }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 681 "src/q2j.y"
    {
              node_t *tmp;
	      node_t *tmp2 = node_to_ptr((yyvsp[(2) - (3)].node));
              // rewind the pointer to the beginning of the list
/*	      for(tmp = tmp2; tmp != NULL; tmp = tmp->next)
	      {

		  fprintf(stdout, "::: %s :::", tmp->type == ASSIGN ? (tmp->u.kids.kids[0])->u.var_name : tmp->u.var_name);
	      }
	      fprintf(stdout, "ENNNND\n");
/*
              for(tmp=$2.next; NULL != tmp->prev ; tmp=tmp->prev);
	      {
	
	      }
*/
		  // traverse the list
	      
	      for(tmp=tmp2->next; NULL != tmp ; tmp=tmp->next){
                  node_t *variable = tmp;
                  if( ASSIGN == tmp->type )
                      variable = tmp->u.kids.kids[0];

                  if( IDENTIFIER == variable->type ){
		      char *str = strdup((yyvsp[(1) - (3)].string));
		      if(variable->var_type != NULL)
			  str = append_to_string(str, variable->var_type, "%s", strlen(variable->var_type));
                      st_insert_new_variable(variable->u.var_name, str);
#if 0 // debug
                      printf("st_insert(%s, %s)\n",variable->u.var_name, str);
#endif
                  }
		  if(ARRAY == variable->type)
		  {
		      char *str = strdup((yyvsp[(1) - (3)].string));
		      if(variable->var_type != NULL)
			  str = append_to_string(str, variable->var_type, "%s", strlen(variable->var_type));			  
		      st_insert_new_variable((variable->u.kids.kids[0])->u.var_name, str);
		  }
		  
              }
#if 0 // debug
              printf("%s ",(char *)(yyvsp[(1) - (3)].string));
              // rewind the pointer to the beginning of the list
              for(tmp=(yyvsp[(2) - (3)].node).next; NULL != tmp->prev; tmp=tmp->prev);
              // traverse the list
              for(; NULL != tmp; tmp=tmp->next){
                  if(NULL != tmp->prev){
                      printf(", ");
                  }
                  printf("%s",tree_to_str(tmp));
              }
              printf("\n");
#endif // debug
              (yyval.node) = (yyvsp[(2) - (3)].node);
          }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 741 "src/q2j.y"
    {
              add_type(tree_to_str(&((yyvsp[(3) - (4)].node))), (yyvsp[(2) - (4)].string));
	  }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 748 "src/q2j.y"
    {
          }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 751 "src/q2j.y"
    {
          }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 757 "src/q2j.y"
    {
	  }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 760 "src/q2j.y"
    {
	  }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 766 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (1)].node));
              tmp->prev = NULL;
              tmp->next = NULL;
              (yyval.node).next = tmp;
          }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 774 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (1)].node));
              tmp->prev = NULL;
              tmp->next = NULL;
              (yyval.node).next = tmp;
          }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 782 "src/q2j.y"
    {
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (2)].node));
              tmp->next = NULL;
              tmp->prev = (yyvsp[(2) - (2)].node).next;
              tmp->prev->next = tmp;
              (yyval.node).next = tmp;
	  }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 794 "src/q2j.y"
    {
	  }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 797 "src/q2j.y"
    {
	      store_global_invariant(node_to_ptr((yyvsp[(3) - (3)].node)));
	  }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 801 "src/q2j.y"
    {
              //#pragma DAGUE_DATA_COLOCATED T A
              //int i=0;
              node_t *tmp, *reference;

              // find the reference matrix in the pragma
              for(reference=(yyvsp[(3) - (3)].node).next; NULL != reference->prev; reference=reference->prev)
                  /* nothing */ ;

              // traverse the list backwards
              //printf("(");
              for(tmp=(yyvsp[(3) - (3)].node).next; NULL != tmp->prev; tmp=tmp->prev){
                  //if(i++)
                  //    printf(" and ");
                  //printf("%s",tmp->u.var_name);
                  add_colocated_data_info(tmp->u.var_name, reference->u.var_name);
              }
              // add a tautologic relation from the reference element to itself
              add_colocated_data_info(reference->u.var_name, reference->u.var_name);
              //printf(") is co-located with %s\n",tmp->u.var_name);
	  }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 823 "src/q2j.y"
    {
              //#pragma DAGUE_TASK_START  TASK_NAME  PARAM[:PSEUDONAME]:(IN|OUT|INOUT|SCRATCH)[:TYPE_NAME] [, ...]
	  }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 831 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
              (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
          }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 836 "src/q2j.y"
    {
              (yyval.string) = (yyvsp[(1) - (1)].string);
          }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 840 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
              (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
          }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 845 "src/q2j.y"
    {
              (yyval.string) = (yyvsp[(1) - (1)].string);
          }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 849 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
	      (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
          }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 857 "src/q2j.y"
    { 
	      (yyval.node).next = node_to_ptr((yyvsp[(1) - (1)].node));
	      (yyval.node).prev = NULL;
          }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 862 "src/q2j.y"
    {
	      node_t *tmp  = node_to_ptr((yyvsp[(1) - (3)].node));
	      node_t *tmp3 = node_to_ptr((yyvsp[(3) - (3)].node));
	      tmp3->next       = tmp->next;
	      tmp->next->prev  = tmp3;
	      (yyval.node).next = tmp3;
	  }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 874 "src/q2j.y"
    {
            (yyval.node).type = ASSIGN;
            (yyval.node).u.kids.kid_count = 2;
            (yyval.node).u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
            (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
            (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (3)].node));
          }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 925 "src/q2j.y"
    {}
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 926 "src/q2j.y"
    {}
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 927 "src/q2j.y"
    {}
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 946 "src/q2j.y"
    {
	      (yyval.function_list) = NULL;
/*	      StarPU_function_list *l;
	      char *tmp = node_to_ptr($1);
	      if(strcmp(tmp, "NULL") == 0)
	      {
		  $$ = NULL;
	      } else {
		  l       = (StarPU_function_list*) calloc(1, sizeof(struct StarPU_function_list_t));
		  l->name = strdup(tmp->u.var_name);
		  l->next = NULL;
		  $$      = l;
	      }
*/
          }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 962 "src/q2j.y"
    {

	      StarPU_function_list *l;
	      node_t *tmp = node_to_ptr((yyvsp[(1) - (3)].node));
	      l       = (StarPU_function_list*) calloc(1, sizeof(struct StarPU_function_list_t));
	      l->name = strdup(tmp->u.var_name);
	      l->next = (yyvsp[(3) - (3)].function_list);	      
	      (yyval.function_list)      = l;
	  }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 975 "src/q2j.y"
    {
	      char *tmp           = strdup((yyvsp[(1) - (5)].string));
	      StarPU_param *param = (StarPU_param*) calloc(1, sizeof(struct StarPU_param_t));
	      if(strcmp(tmp, ".cpu_funcs") == 0)
	      {
		  param->type = CODELET_CPU;
		  param->p.l  = (yyvsp[(4) - (5)].function_list);
	      } else if (strcmp(tmp, ".cuda_funcs") == 0)
	      {
		  param->type = CODELET_CUDA;
		  param->p.l  = (yyvsp[(4) - (5)].function_list);
	      }
	      free(tmp);
	      (yyval.param) = param;
	  }
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 991 "src/q2j.y"
    {  
	      StarPU_param *param = (StarPU_param*) calloc(1, sizeof(struct StarPU_param_t));
	      node_t       *tmp   = node_to_ptr((yyvsp[(2) - (4)].node));
	      char         *type  = tmp->u.var_name;
	      tmp                 = node_to_ptr((yyvsp[(4) - (4)].node));
	      if(strcmp(type, "modes")      == 0)
	      {
		  param->type    = CODELET_MODE;
		  param->p.modes = strdup(tmp->u.var_name);
	      }
	      if(strcmp(type, "where")      == 0)
	      {
		  param->type    = CODELET_WHERE;
		  param->p.where = strdup(tmp->u.var_name);		  
	      }
	      if(strcmp(type, "cpu_funcs")  == 0)
	      {
		  param->type    = CODELET_CPU;
		  param->p.l   = strdup(tmp->u.var_name);
	      }
	      if(strcmp(type, "cuda_funcs") == 0)
	      {
		  param->type    = CODELET_CUDA;
		  param->p.l  = strdup(tmp->u.var_name);
	      }
	      if(strcmp(type, "nbuffers")   == 0)
	      {
		  param->type       = CODELET_NBUFF;
		  param->p.nbuffers = atoi(tmp->u.var_name);
	      }
	      free(type);
	      (yyval.param) = param;
	  }
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 1025 "src/q2j.y"
    {
	      StarPU_param *param = (StarPU_param*) calloc(1, sizeof(struct StarPU_param_t));
              node_t       *tmp   = node_to_ptr((yyvsp[(2) - (4)].node));
              char         *type  = tmp->u.var_name;
	      param->type         = CODELET_NBUFF;
	      tmp                 = node_to_ptr((yyvsp[(4) - (4)].node));
	      param->p.nbuffers   = tmp->const_val.i64_value;
	      free(type);
	      (yyval.param) = param;
	  }
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 1039 "src/q2j.y"
    {
	      StarPU_param_list *pl = (StarPU_param_list*) calloc(1, sizeof(struct StarPU_param_list_t));
	      pl->p                 = (yyvsp[(1) - (1)].param);
	      pl->next              = NULL;	      
	      (yyval.param_list)                    = pl;
	  }
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 1046 "src/q2j.y"
    {
	      StarPU_param_list *pl = (StarPU_param_list*) calloc(1, sizeof(struct StarPU_param_list_t));
	      pl->p                 = (yyvsp[(1) - (3)].param);
	      pl->next              = (yyvsp[(3) - (3)].param_list);
	      (yyval.param_list)                    = pl;
	  }
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 1056 "src/q2j.y"
    {
	      StarPU_codelet_list *cl_list = NULL;
	      StarPU_codelet *cl           = (StarPU_codelet*) calloc(1, sizeof(struct StarPU_codelet_t));
	      node_t *tmp                  = node_to_ptr((yyvsp[(3) - (8)].node));
	      cl->name                     = strdup(tmp->u.var_name);
	      cl->l                        = (yyvsp[(6) - (8)].param_list);
	      if(codelet_list == NULL)
	      {
		  codelet_list       = (StarPU_codelet_list*) calloc(1, sizeof(struct StarPU_codelet_list_t));
		  codelet_list->prev = NULL;
		  codelet_list->next = NULL;
		  codelet_list->cl   = cl;
	      }
	      else
	      {
		  cl_list            = (StarPU_codelet_list*) calloc(1, sizeof(struct StarPU_codelet_list_t));
		  cl_list->cl        = cl;
		  cl_list->prev      = codelet_list;
		  codelet_list->next = cl_list;
		  cl_list->next      = NULL;
		  codelet_list       = cl_list;
	      }
	  }
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 1083 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
              (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
          }
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 1089 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
              (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
          }
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 1102 "src/q2j.y"
    {}
    break;

  case 168:

/* Line 1806 of yacc.c  */
#line 1130 "src/q2j.y"
    {
	    (yyval.node) = (yyvsp[(2) - (2)].node);
	    (yyval.node).var_type = strdup((yyvsp[(1) - (2)].string));
	    (yyval.node).pointer = 1;
	  }
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 1136 "src/q2j.y"
    {
	      (yyval.node).pointer = 0;
	      (yyval.node) = (yyvsp[(1) - (1)].node);
	  }
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 1145 "src/q2j.y"
    {
              (yyval.node) = (yyvsp[(2) - (3)].node);
          }
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 1149 "src/q2j.y"
    {
              if( ARRAY == (yyvsp[(1) - (4)].node).type ){
                  int count;
                  (yyval.node) = (yyvsp[(1) - (4)].node);
                  count = ++((yyval.node).u.kids.kid_count);
                  (yyval.node).u.kids.kids = (node_t **)realloc( (yyval.node).u.kids.kids, count*sizeof(node_t *) );
                  (yyval.node).u.kids.kids[count-1] = node_to_ptr((yyvsp[(3) - (4)].node));
              }else{
                  (yyval.node).type = ARRAY;
                  (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  (yyval.node).u.kids.kid_count = 2;
                  (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (4)].node));
                  (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (4)].node));
//		  st_insert_new_variable($1.u.var_name, "int");
              }
          }
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 1166 "src/q2j.y"
    {
              if( ARRAY == (yyvsp[(1) - (3)].node).type ){
                  int count;
                  (yyval.node) = (yyvsp[(1) - (3)].node);
                  count = ++((yyval.node).u.kids.kid_count);
                  (yyval.node).u.kids.kids = (node_t **)realloc( (yyval.node).u.kids.kids, count*sizeof(node_t *) );
                  node_t tmp;
                  tmp.type=EMPTY;
          
        (yyval.node).u.kids.kids[count-1] = node_to_ptr(tmp);
              }else{
                  (yyval.node).type = ARRAY;
                  (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
                  (yyval.node).u.kids.kid_count = 2;
                  (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(1) - (3)].node));
                  node_t tmp;
                  tmp.type=EMPTY;
                  (yyval.node).u.kids.kids[1] = node_to_ptr(tmp);
//		  st_insert_new_variable($1.u.var_name, "int");
              }
          }
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 1188 "src/q2j.y"
    {
	      symbol_t *sym;
              (yyvsp[(1) - (4)].node).symtab = st_get_current_st();
	      (yyval.node) = (yyvsp[(1) - (4)].node);
	  }
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 1198 "src/q2j.y"
    {	    (yyval.string) = strdup("*");}
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 1199 "src/q2j.y"
    { char *str = strdup("*"); (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), "%s", strlen((yyvsp[(2) - (2)].string)));}
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 1200 "src/q2j.y"
    {char *str = strdup("*"); (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), "%s", 2+strlen((yyvsp[(2) - (2)].string)));}
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 1202 "src/q2j.y"
    { 
	    char *str = strdup("*"); str = append_to_string(str, (yyvsp[(2) - (3)].string), "%s", strlen((yyvsp[(2) - (3)].string))); 
	    (yyval.string) = append_to_string(str, (yyvsp[(3) - (3)].string), "%s", strlen((yyvsp[(3) - (3)].string)));
	}
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 1217 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (3)].string));
              (yyval.string) = append_to_string(str, "", ", ...", 5);
          }
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 1225 "src/q2j.y"
    {
              (void)st_enter_new_scope();
              st_insert_new_variable((yyvsp[(1) - (1)].type_node).var, (yyvsp[(1) - (1)].type_node).type);
          }
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 1230 "src/q2j.y"
    {
              st_insert_new_variable((yyvsp[(3) - (3)].type_node).var, (yyvsp[(3) - (3)].type_node).type);
          }
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 1237 "src/q2j.y"
    {
	      char *str;
	      if((yyvsp[(2) - (2)].node).var_type != NULL)
	      {
		  str = (yyvsp[(2) - (2)].node).var_type;
		  (yyvsp[(2) - (2)].node).var_type = NULL;
		  (yyval.type_node).type = append_to_string((yyvsp[(1) - (2)].string), str, "%s", strlen(str));
		  (yyval.type_node).var  = tree_to_str(&((yyvsp[(2) - (2)].node)));
		  (yyvsp[(2) - (2)].node).var_type = (yyval.type_node).type;
	      }
	      else
	      {
		  (yyval.type_node).type = strdup((yyvsp[(1) - (2)].string));
		  (yyval.type_node).var  = tree_to_str(&((yyvsp[(2) - (2)].node)));
	      }
          }
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 1254 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (2)].string));
              str = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 1+strlen((yyvsp[(2) - (2)].string)) );
              printf("WARNING: the following parameter declaration is not inserted into the symbol table:\n%s\n",str);
          }
    break;

  case 189:

/* Line 1806 of yacc.c  */
#line 1259 "src/q2j.y"
    { }
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 1263 "src/q2j.y"
    { (yyval.string) = (yyvsp[(1) - (1)].node).u.var_name; }
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 1265 "src/q2j.y"
    {
              char *str = strdup((yyvsp[(1) - (3)].string));
              str = append_to_string(str, ", ", NULL, 0 );
              (yyval.string) = append_to_string(str, (yyvsp[(3) - (3)].node).u.var_name, NULL, 0 );
          }
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 1274 "src/q2j.y"
    { char *str = strdup((yyvsp[(1) - (2)].string)); (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), "%s", 2+strlen((yyvsp[(2) - (2)].string)));}
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 1278 "src/q2j.y"
    { (yyval.string) = strdup((yyvsp[(1) - (1)].string));}
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 1280 "src/q2j.y"
    { char *str = strdup((yyvsp[(1) - (2)].string)); (yyval.string) = append_to_string(str, (yyvsp[(2) - (2)].string), " %s", 2+strlen((yyvsp[(2) - (2)].string))+1);}
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 1284 "src/q2j.y"
    {}
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 1285 "src/q2j.y"
    {}
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 1286 "src/q2j.y"
    {}
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 1287 "src/q2j.y"
    {}
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 1288 "src/q2j.y"
    {}
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 1289 "src/q2j.y"
    {}
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 1290 "src/q2j.y"
    {}
    break;

  case 204:

/* Line 1806 of yacc.c  */
#line 1291 "src/q2j.y"
    {}
    break;

  case 205:

/* Line 1806 of yacc.c  */
#line 1292 "src/q2j.y"
    {}
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 1297 "src/q2j.y"
    { }
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 1298 "src/q2j.y"
    {}
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 1302 "src/q2j.y"
    {}
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 1303 "src/q2j.y"
    {}
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 1324 "src/q2j.y"
    { 
	      (yyval.node).type = BLOCK;
              (yyval.node).u.block.first = NULL;
              (yyval.node).u.block.last = NULL;
          }
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 1330 "src/q2j.y"
    {
              node_t *tmp;

              (yyval.node).type = BLOCK;
              (yyval.node).u.block.last = (yyvsp[(2) - (3)].node).next;
              for(tmp=(yyvsp[(2) - (3)].node).next->prev; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;
              if( NULL == tmp )
                  (yyval.node).u.block.first = (yyval.node).u.block.last;
              else
                  (yyval.node).u.block.first = tmp;
          }
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 1342 "src/q2j.y"
    {
              node_t *tmp;
	      node_t *doll2 = node_to_ptr((yyvsp[(2) - (3)].node));
              (yyval.node).type = BLOCK;
//              $$.u.block.last = $2;
              for(tmp=(yyvsp[(2) - (3)].node).next; NULL != tmp && NULL != tmp->next; tmp=tmp->next) ;
	      (yyval.node).u.block.first = ((yyvsp[(2) - (3)].node).next->next);
	      (yyval.node).u.block.last  = tmp;
	      /* if( NULL == tmp )
                  $$.u.block.first = $$.u.block.last;
              else
	      $$.u.block.first = tmp;*/
          }
    break;

  case 224:

/* Line 1806 of yacc.c  */
#line 1356 "src/q2j.y"
    {
              // We've got two separate lists and we need to chain them together.
              node_t *tmp, *tmp2;

              (yyval.node).type = BLOCK;
              // take as last the last of the second list
              (yyval.node).u.block.last  = (yyvsp[(3) - (4)].node).next;
	      (yyval.node).u.block.first = ((yyvsp[(2) - (4)].node).next)->next;

              // then walk back the first list to find the beginning.
//              for(tmp=$2.next->prev; NULL != tmp && NULL != tmp->prev; tmp=tmp->prev) ;

//              if( NULL == tmp )
//                  $$.u.block.first = $2.next;
//              else
//                  $$.u.block.first = tmp;

              // then walk back the second list to find its beginning.
              for(tmp=(yyvsp[(3) - (4)].node).next;  NULL != tmp  && NULL != tmp->prev;  tmp=tmp->prev) ;
              for(tmp2=(yyvsp[(2) - (4)].node).next; NULL != tmp2 && NULL != tmp2->next; tmp2=tmp2->next) ;

              // Now connect the end of the first list to the beginning of the second.
              tmp2->next = tmp;
              if( NULL != tmp )
                  tmp->prev = tmp2;

//for(; NULL != tmp; tmp=tmp->next)
//    dump_tree(*tmp,0);
          }
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 1389 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (1)].node));
              tmp->prev = NULL;
              (yyval.node).next = tmp;
          }
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 1396 "src/q2j.y"
    { 
              node_t *t;
	      node_t *tmp1, *tmp2;
	      tmp2 = node_to_ptr((yyvsp[(2) - (2)].node));
	      tmp1 = node_to_ptr((yyvsp[(1) - (2)].node));
	      for(t = tmp1; t!=NULL && t->next != NULL; t = t->next);
	      t->next = tmp2->next; 
	      tmp2->next->prev = t;
	      tmp1->next->prev = &(yyval.node);
              (yyval.node).next = tmp1->next;
          }
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 1411 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(1) - (1)].node));
              tmp->prev = NULL;
              tmp->next = NULL;
              (yyval.node).next = tmp;
          }
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 1420 "src/q2j.y"
    { 
              node_t *tmp;
              tmp = node_to_ptr((yyvsp[(2) - (2)].node));
              tmp->next = NULL;
              tmp->prev = (yyvsp[(1) - (2)].node).next;
              tmp->prev->next = tmp;
              (yyval.node).next = tmp;
          }
    break;

  case 229:

/* Line 1806 of yacc.c  */
#line 1431 "src/q2j.y"
    { (yyval.node).type = EMPTY; }
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 1432 "src/q2j.y"
    {(yyval.node)=(yyvsp[(1) - (2)].node);}
    break;

  case 231:

/* Line 1806 of yacc.c  */
#line 1437 "src/q2j.y"
    {
              (yyval.node).type = IF;
              (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(3) - (5)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(5) - (5)].node));
          }
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 1445 "src/q2j.y"
    {
              (yyval.node).type = IF;
              (yyval.node).u.kids.kids = (node_t **)calloc(3, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 3;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(3) - (7)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(5) - (7)].node));
              (yyval.node).u.kids.kids[2] = node_to_ptr((yyvsp[(7) - (7)].node));
          }
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 1454 "src/q2j.y"
    {
	      (yyval.node).type = SWITCH;
	      (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t*));
	      (yyval.node).u.kids.kid_count = 2;
	      (yyval.node).u.kids.kids[0]   = node_to_ptr((yyvsp[(3) - (5)].node));
	      (yyval.node).u.kids.kids[1]   = node_to_ptr((yyvsp[(5) - (5)].node));
	  }
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 1465 "src/q2j.y"
    {
              (yyval.node).type = WHILE;
              (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(3) - (5)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(5) - (5)].node));
              (yyval.node).trip_count = -1;
              (yyval.node).loop_depth = -1;
          }
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 1475 "src/q2j.y"
    {
              (yyval.node).type = DO;
              (yyval.node).u.kids.kids = (node_t **)calloc(2, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 2;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(5) - (7)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(3) - (7)].node));
              (yyval.node).trip_count = -1;
              (yyval.node).loop_depth = -1;
          }
    break;

  case 236:

/* Line 1806 of yacc.c  */
#line 1485 "src/q2j.y"
    {
              (yyval.node).type = FOR;
              (yyval.node).u.kids.kids = (node_t **)calloc(4, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 4;

              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(3) - (6)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(4) - (6)].node));
              node_t tmp;
              tmp.type=EMPTY;
              (yyval.node).u.kids.kids[2]  = node_to_ptr(tmp);
              (yyval.node).u.kids.kids[3]  = node_to_ptr((yyvsp[(6) - (6)].node));
              DA_canonicalize_for(&((yyval.node)));

              (yyval.node).trip_count = -1;
              (yyval.node).loop_depth = -1;
          }
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 1503 "src/q2j.y"
    {
              (yyval.node).type = FOR;

              (yyval.node).u.kids.kids = (node_t **)calloc(4, sizeof(node_t *));
              (yyval.node).u.kids.kid_count = 4;
              (yyval.node).u.kids.kids[0] = node_to_ptr((yyvsp[(3) - (7)].node));
              (yyval.node).u.kids.kids[1] = node_to_ptr((yyvsp[(4) - (7)].node));
              (yyval.node).u.kids.kids[2] = node_to_ptr((yyvsp[(5) - (7)].node));
              (yyval.node).u.kids.kids[3] = node_to_ptr((yyvsp[(7) - (7)].node));
              DA_canonicalize_for(&((yyval.node)));

              (yyval.node).trip_count = -1;
              (yyval.node).loop_depth = -1;
          }
    break;

  case 243:

/* Line 1806 of yacc.c  */
#line 1529 "src/q2j.y"
    {
          
		      /*
	      rename_induction_variables(trans_list->tr);
	      convert_OUTPUT_to_INOUT(trans_list->tr);
	      if( _q2j_add_phony_tasks )
		  add_entry_and_exit_task_loops(trans_list->tr);
		  analyze_deps(trans_list->tr);              */       	      
	  }
    break;

  case 244:

/* Line 1806 of yacc.c  */
#line 1538 "src/q2j.y"
    {}
    break;

  case 245:

/* Line 1806 of yacc.c  */
#line 1543 "src/q2j.y"
    {
	      StarPU_translation_list *l = (StarPU_translation_list*) calloc(1, sizeof(struct StarPU_translation_list_t));
	      if(trans_list == NULL)
	      {
		  l->next    = NULL;
		  l->prev    = NULL;
		  l->tr      = (yyvsp[(1) - (1)].starpu_fun_d);
		  trans_list = l;
	      } else
	      {
		  l->next    = NULL;
		  l->prev    = trans_list;
		  l->tr      = (yyvsp[(1) - (1)].starpu_fun_d);
		  trans_list = l;
	      }
	      //             rename_induction_variables(&($1));
	      //       convert_OUTPUT_to_INOUT(&($1));
	      //     if( _q2j_add_phony_tasks )
	      //     add_entry_and_exit_task_loops(&($1));
	      //   analyze_deps(&($1));
          }
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 1565 "src/q2j.y"
    {
              // Here is where the global scope variables were declared
              static node_t tmp;
              tmp.type=EMPTY;
              (yyval.node)=tmp;
          }
    break;

  case 247:

/* Line 1806 of yacc.c  */
#line 1572 "src/q2j.y"
    {
              /* do nothing */
          }
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 1576 "src/q2j.y"
    {
              /* do nothing */
          }
    break;

  case 249:

/* Line 1806 of yacc.c  */
#line 1580 "src/q2j.y"
    {
	      /* do nothing */
          }
    break;

  case 250:

/* Line 1806 of yacc.c  */
#line 1587 "src/q2j.y"
    {
	      StarPU_fun_decl *f = (StarPU_fun_decl*) calloc(1, sizeof(struct StarPU_fun_decl_t));
              f->node = node_to_ptr((yyvsp[(4) - (4)].node));
	      f->name = strdup((yyvsp[(2) - (4)].node).u.var_name);
	      (yyval.starpu_fun_d) = f;

	      DA_parentize(node_to_ptr((yyvsp[(4) - (4)].node)));
          }
    break;

  case 251:

/* Line 1806 of yacc.c  */
#line 1596 "src/q2j.y"
    {

	      StarPU_fun_decl *f = (StarPU_fun_decl*) calloc(1, sizeof(struct StarPU_fun_decl_t));
	      symbol_t *sym;
              f->node = node_to_ptr((yyvsp[(3) - (3)].node));

              f->name = strdup((yyvsp[(2) - (3)].node).u.var_name);
	      (yyval.starpu_fun_d) = f;

	      DA_parentize(node_to_ptr((yyvsp[(3) - (3)].node)));

          }
    break;

  case 252:

/* Line 1806 of yacc.c  */
#line 1609 "src/q2j.y"
    {
	      StarPU_fun_decl *f = (StarPU_fun_decl*) calloc(1, sizeof(struct StarPU_fun_decl_t));
              f->node = node_to_ptr((yyvsp[(3) - (3)].node));
              f->name = strdup((yyvsp[(1) - (3)].node).u.var_name);
              (yyval.starpu_fun_d) = f;
	      DA_parentize(node_to_ptr((yyvsp[(3) - (3)].node)));
          }
    break;

  case 253:

/* Line 1806 of yacc.c  */
#line 1617 "src/q2j.y"
    {
	      StarPU_fun_decl *f = (StarPU_fun_decl*) calloc(1, sizeof(struct StarPU_fun_decl_t));
              f->node = node_to_ptr((yyvsp[(2) - (2)].node));
              f->name = strdup((yyvsp[(1) - (2)].node).u.var_name);
	      (yyval.starpu_fun_d) = f;

	      DA_parentize(node_to_ptr((yyvsp[(2) - (2)].node)));
          }
    break;



/* Line 1806 of yacc.c  */
#line 4085 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/q2j/q2j.y.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 1627 "src/q2j.y"


extern char yytext[];
extern int column;

void yyerror(const char *s){
	fflush(stdout);
	fprintf(stderr,"Syntax error near line %d. %s\n", yyget_lineno(), s);
	exit(-1);
}



void add_type(char *new_type, char *old_type){
    unsigned long int h;
    type_list_t *t, *tmp;

    if( NULL != lookup_type(new_type) ){
        fprintf(stderr,"type defined aready\n");
        exit(-1);
    }

    tmp = (type_list_t *)calloc(1, sizeof(type_list_t));
    tmp->new_type = strdup(new_type);
    tmp->old_type = strdup(old_type);

    h = hash(new_type);
    t=type_hash[h];
    if( NULL == t ){
        type_hash[h] = tmp;
        return;
    }

    for(; NULL != t->next; t=t->next);
    t->next = tmp;
 
    return;
}



char *lookup_type(char *new_type){
    unsigned long int h;
    type_list_t *t;

    h = hash(new_type);
    for(t=type_hash[h]; NULL != t; t=t->next){
        if( !strcmp(t->new_type, new_type) )
            return t->old_type;
    }

    return NULL;
}


/* Modified djb2 hash from: http://www.cse.yorku.ca/~oz/hash.html */
unsigned long hash(char *str) {
    unsigned long result = 5381;
    int c;

    while( 0 != (c = *str++) ){
        result = ((result << 5) + result) + c; /* result * 33 + c */
    }

    unsigned long tmp = result;
    while( tmp > (HASH_TAB_SIZE/4) ){
        tmp /= 2;
    }
    result = tmp + result%(HASH_TAB_SIZE-(HASH_TAB_SIZE/4));

    return result;
}


