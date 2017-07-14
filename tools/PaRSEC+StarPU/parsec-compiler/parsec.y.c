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
#line 1 "parsec.y"

/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "jdf.h"

/**
 * Better error handling
 */
#define YYERROR_VERBOSE 1

#include "parsec.y.h"

extern int yyparse(void);
extern int yylex(void);

extern int current_lineno;

static jdf_expr_t *inline_c_functions = NULL;

static void yyerror(const char *str)
{
    fprintf(stderr, "parse error at line %d: %s\n", current_lineno, str);
}

int yywrap(void); 

int yywrap(void)
{
    return 1;
}

#define new(type)  (type*)calloc(1, sizeof(type))

static jdf_def_list_t* jdf_create_properties_list( const char* name, int default_int, const char* default_char, jdf_def_list_t* next )
{
    jdf_def_list_t* property;
    jdf_expr_t *e;

    property         = new(jdf_def_list_t);
    property->next   = next;
    property->name   = strdup(name);
    property->lineno = current_lineno;

    if( NULL != default_char ) {
        e = new(jdf_expr_t);
        e->op = JDF_STRING;
        e->jdf_var = strdup(default_char);
    } else {
        e = new(jdf_expr_t);
        e->op = JDF_CST;
        e->jdf_cst = default_int;
    }
    property->expr = e;
    return property;
}

static jdf_data_entry_t* jdf_find_or_create_data(jdf_t* jdf, const char* dname)
{
    jdf_data_entry_t* data = jdf->data;
    jdf_global_entry_t* global = jdf->globals;

    while( NULL != data ) {
        if( !strcmp(data->dname, dname) ) {
            return data;
        }
        data = data->next;
    }
    /* not found, create */
    data = new(jdf_data_entry_t);
    data->dname    = strdup(dname);
    data->lineno   = current_lineno;
    data->nbparams = -1;
    data->next     = jdf->data;
    jdf->data = data;
    /* Check if there is a global with the same name */
    while( NULL != global ) {
        if( !strcmp(global->name, data->dname) ) {
            assert(NULL == global->data);
            global->data = data;
            data->global = global;

            if( jdf_find_property( global->properties, "type", NULL ) == NULL ) {
                global->properties = jdf_create_properties_list( "type", 0, "parsec_data_collection_t*", global->properties);
            }

            return data;
        }
        global = global->next;
    }
    assert(NULL == global);
    global             = new(jdf_global_entry_t);
    global->name       = strdup(data->dname);
    global->properties = jdf_create_properties_list( "type", 0, "parsec_data_collection_t*", NULL );
    global->data       = data;
    global->expression = NULL;
    global->lineno     = current_lineno;
    data->global       = global;
    /* Chain it with the other globals */
    global->next = jdf->globals;
    jdf->globals = global;
    
    return data;
}



/* Line 268 of yacc.c  */
#line 186 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/parsec-compiler/parsec.y.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
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

/* Line 293 of yacc.c  */
#line 115 "parsec.y"

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



/* Line 293 of yacc.c  */
#line 281 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/parsec-compiler/parsec.y.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 293 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/parsec-compiler/parsec.y.c"

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
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   261

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  40
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  25
/* YYNRULES -- Number of rules.  */
#define YYNRULES  72
/* YYNRULES -- Number of states.  */
#define YYNSTATES  135

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   294

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
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
      35,    36,    37,    38,    39
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     7,     9,    11,    12,    15,    21,    25,
      26,    30,    31,    36,    40,    52,    65,    78,    92,    96,
      98,    99,   104,   108,   111,   112,   118,   121,   122,   124,
     125,   129,   132,   133,   137,   139,   143,   149,   153,   159,
     165,   170,   173,   174,   178,   180,   184,   186,   190,   192,
     194,   196,   200,   204,   208,   212,   216,   220,   224,   228,
     232,   235,   239,   243,   247,   251,   255,   259,   263,   267,
     273,   275,   277
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      41,     0,    -1,    42,    44,    43,    -1,     5,    -1,     5,
      -1,    -1,    44,    47,    -1,    44,     3,    45,     4,    63,
      -1,    44,     3,    45,    -1,    -1,    19,    46,    20,    -1,
      -1,     3,     4,    63,    46,    -1,     3,     4,    63,    -1,
       3,     7,    48,     8,    45,    49,    50,    51,    52,    59,
       9,    -1,     3,     7,    48,     8,    45,    49,    50,    51,
      52,    59,     9,    11,    -1,     3,     7,    48,     8,    45,
      49,    50,    51,    52,    59,     9,    10,    -1,     3,     7,
      48,     8,    45,    49,    50,    51,    52,    59,     9,    10,
      11,    -1,     3,     6,    48,    -1,     3,    -1,    -1,     3,
       4,    62,    49,    -1,     3,     4,    62,    -1,    13,    63,
      -1,    -1,    14,     3,     7,    61,     8,    -1,    54,    52,
      -1,    -1,    16,    -1,    -1,    53,     3,    55,    -1,    56,
      55,    -1,    -1,    17,    57,    45,    -1,    58,    -1,    64,
      18,    58,    -1,    64,    18,    58,    14,    58,    -1,    63,
      18,    58,    -1,    63,    18,    58,    14,    58,    -1,     3,
       3,     7,    60,     8,    -1,     3,     7,    60,     8,    -1,
      15,    63,    -1,    -1,    62,     6,    60,    -1,    62,    -1,
      64,     6,    61,    -1,    64,    -1,    63,    39,    63,    -1,
      63,    -1,    64,    -1,     5,    -1,    64,    21,    64,    -1,
      64,    22,    64,    -1,    64,    23,    64,    -1,    64,    24,
      64,    -1,    64,    25,    64,    -1,    64,    26,    64,    -1,
      64,    27,    64,    -1,    64,    28,    64,    -1,    64,    29,
      64,    -1,    30,    64,    -1,     7,    64,     8,    -1,    64,
      32,    64,    -1,    64,    33,    64,    -1,    64,    34,    64,
      -1,    64,    35,    64,    -1,    64,    36,    64,    -1,    64,
      37,    64,    -1,    64,    38,    64,    -1,    64,    18,    64,
      14,    64,    -1,     3,    -1,    31,    -1,    12,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   185,   185,   193,   200,   207,   211,   228,   253,   279,
     291,   296,   301,   310,   321,   340,   359,   378,   404,   412,
     421,   427,   437,   450,   454,   457,   482,   488,   494,   498,
     501,   513,   519,   524,   566,   575,   584,   593,   602,   613,
     621,   646,   650,   655,   660,   667,   672,   679,   687,   693,
     697,   713,   721,   729,   737,   745,   753,   761,   769,   777,
     785,   792,   796,   804,   812,   820,   828,   836,   844,   852,
     861,   868,   875
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "VAR", "ASSIGNMENT", "EXTERN_DECL",
  "COMMA", "OPEN_PAR", "CLOSE_PAR", "BODY", "GPU", "MODEL", "STRING",
  "SIMCOST", "COLON", "SEMICOLON", "DEPENDENCY_TYPE", "ARROW",
  "QUESTION_MARK", "PROPERTIES_ON", "PROPERTIES_OFF", "EQUAL", "NOTEQUAL",
  "LESS", "LEQ", "MORE", "MEQ", "AND", "OR", "XOR", "NOT", "INT", "PLUS",
  "MINUS", "TIMES", "DIV", "MODULO", "SHL", "SHR", "RANGE", "$accept",
  "jdf_file", "prologue", "epilogue", "jdf", "properties",
  "properties_list", "function", "varlist", "execution_space",
  "simulation_cost", "partitioning", "dataflow_list",
  "optional_flow_flags", "dataflow", "dependencies", "dependency",
  "guarded_call", "call", "priority", "expr_list_range", "expr_list",
  "expr_range", "expr_complete", "expr_simple", 0
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
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    40,    41,    42,    43,    43,    44,    44,    44,    44,
      45,    45,    46,    46,    47,    47,    47,    47,    48,    48,
      48,    49,    49,    50,    50,    51,    52,    52,    53,    53,
      54,    55,    55,    56,    57,    57,    57,    57,    57,    58,
      58,    59,    59,    60,    60,    61,    61,    62,    62,    63,
      63,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     3,     1,     1,     0,     2,     5,     3,     0,
       3,     0,     4,     3,    11,    12,    12,    13,     3,     1,
       0,     4,     3,     2,     0,     5,     2,     0,     1,     0,
       3,     2,     0,     3,     1,     3,     5,     3,     5,     5,
       4,     2,     0,     3,     1,     3,     1,     3,     1,     1,
       1,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       2,     3,     3,     3,     3,     3,     3,     3,     3,     5,
       1,     1,     1
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     3,     0,     9,     1,     5,    11,     4,     2,     6,
      20,     0,     8,    19,     0,     0,     0,     0,    20,    11,
       0,    10,    70,    50,     0,    72,     0,    71,     7,    49,
      18,     0,    13,     0,    60,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    24,    12,    61,     0,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    62,    63,    64,    65,
      66,    67,    68,     0,     0,     0,     0,    22,    48,    23,
       0,    27,    69,    21,     0,     0,    28,    42,     0,    27,
      47,     0,     0,     0,    32,    26,     0,    46,    41,    14,
       0,    30,    32,    25,     0,    16,    15,    70,    11,    34,
       0,     0,    31,    45,    17,     0,     0,    33,     0,     0,
       0,     0,    44,     0,    37,    35,     0,    40,     0,     0,
       0,    39,    43,    38,    36
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     8,     5,    12,    16,     9,    14,    53,
      75,    81,    87,    88,    89,   101,   102,   108,   109,    93,
     121,    96,   122,    78,    29
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -91
static const yytype_int16 yypact[] =
{
      35,   -91,    65,   -91,   -91,     2,    44,   -91,   -91,   -91,
      66,    71,    73,    69,    76,    81,    59,    61,    66,    68,
      61,   -91,   -91,   -91,   114,   -91,   114,   -91,   -91,   153,
     -91,    85,    71,    21,   223,   114,   114,   114,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   114,   114,
     114,   114,    90,    92,   -91,   -91,   132,   174,   174,   211,
     211,   211,   211,    -1,    -1,    -1,   -25,   -25,   -91,   -91,
      98,    98,    98,    61,    61,   101,   114,    85,    50,   -91,
     103,    25,   174,   -91,    61,   112,   -91,   107,   122,    25,
     -91,   114,    61,   118,   119,   -91,   120,    75,   -91,    51,
     111,   -91,   119,   -91,   114,   123,   -91,     1,    68,   -91,
     121,   195,   -91,   -91,   -91,   128,    61,   -91,   134,   117,
      61,   130,   137,     1,   126,   135,   143,   -91,    61,   134,
     134,   -91,   -91,   -91,   -91
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -91,   -91,   -91,   -91,   -91,   -18,   131,   -91,   144,    95,
     -91,   -91,    63,   -91,   -91,    82,   -91,   -91,   -47,   -91,
     -90,    79,   100,   -14,   -24
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -30
static const yytype_int16 yytable[] =
{
      33,    31,    34,    28,   115,     6,    32,     7,   116,    47,
      48,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,   -29,    55,
     126,    45,    46,    47,    48,    49,    50,    51,   132,    35,
       1,    86,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    10,    82,    45,    46,    47,    48,    49,    50,    51,
      79,   105,   106,    11,    22,     4,    23,    97,    24,    13,
      90,   124,   125,    25,    15,    18,   111,    17,    98,    21,
      97,   104,   133,   134,    19,    20,   110,    11,    52,    84,
     117,    26,    27,    35,    73,    56,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    74,    85,    45,    46,    47,
      48,    49,    50,    51,   107,    80,    23,    22,    24,    91,
     107,    24,    92,    25,    24,    94,    25,    99,   103,    25,
      45,    46,    47,    48,   114,   120,   100,   123,   127,   118,
     129,    26,    27,   128,    26,    27,    76,    26,    27,   130,
      35,   131,    95,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    30,    54,    45,    46,    47,    48,    49,    50,
      51,    35,    83,    77,    36,    37,    38,    39,    40,    41,
      42,    43,    44,   113,   112,    45,    46,    47,    48,    49,
      50,    51,   -30,     0,     0,   -30,   -30,    38,    39,    40,
      41,    42,    43,    44,     0,     0,    45,    46,    47,    48,
      49,    50,    51,   119,     0,     0,    36,    37,    38,    39,
      40,    41,    42,    43,    44,     0,     0,    45,    46,    47,
      48,    49,    50,    51,   -30,   -30,   -30,   -30,    42,    43,
      44,     0,     0,    45,    46,    47,    48,    49,    50,    51,
      42,    43,    44,     0,     0,    45,    46,    47,    48,    49,
      50,    51
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-91))

#define yytable_value_is_error(yytable_value) \
  ((yytable_value) == (-30))

static const yytype_int16 yycheck[] =
{
      24,    19,    26,    17,     3,     3,    20,     5,     7,    34,
      35,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,     3,     8,
     120,    32,    33,    34,    35,    36,    37,    38,   128,    18,
       5,    16,    21,    22,    23,    24,    25,    26,    27,    28,
      29,     7,    76,    32,    33,    34,    35,    36,    37,    38,
      74,    10,    11,    19,     3,     0,     5,    91,     7,     3,
      84,   118,   119,    12,     3,     6,   100,     4,    92,    20,
     104,     6,   129,   130,     8,     4,   100,    19,     3,    39,
     108,    30,    31,    18,     4,   119,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    13,     3,    32,    33,    34,
      35,    36,    37,    38,     3,    14,     5,     3,     7,     7,
       3,     7,    15,    12,     7,     3,    12,     9,     8,    12,
      32,    33,    34,    35,    11,     7,    17,     3,     8,    18,
      14,    30,    31,     6,    30,    31,    14,    30,    31,    14,
      18,     8,    89,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    18,    32,    32,    33,    34,    35,    36,    37,
      38,    18,    77,    73,    21,    22,    23,    24,    25,    26,
      27,    28,    29,   104,   102,    32,    33,    34,    35,    36,
      37,    38,    18,    -1,    -1,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    -1,    32,    33,    34,    35,
      36,    37,    38,    18,    -1,    -1,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    -1,    32,    33,    34,
      35,    36,    37,    38,    23,    24,    25,    26,    27,    28,
      29,    -1,    -1,    32,    33,    34,    35,    36,    37,    38,
      27,    28,    29,    -1,    -1,    32,    33,    34,    35,    36,
      37,    38
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     5,    41,    42,     0,    44,     3,     5,    43,    47,
       7,    19,    45,     3,    48,     3,    46,     4,     6,     8,
       4,    20,     3,     5,     7,    12,    30,    31,    63,    64,
      48,    45,    63,    64,    64,    18,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    32,    33,    34,    35,    36,
      37,    38,     3,    49,    46,     8,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,     4,    13,    50,    14,    62,    63,    63,
      14,    51,    64,    49,    39,     3,    16,    52,    53,    54,
      63,     7,    15,    59,     3,    52,    61,    64,    63,     9,
      17,    55,    56,     8,     6,    10,    11,     3,    57,    58,
      63,    64,    55,    61,    11,     3,     7,    45,    18,    18,
       7,    60,    62,     3,    58,    58,    60,     8,     6,    14,
      14,     8,    60,    58,    58
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
#line 186 "parsec.y"
    {
                    assert( NULL == current_jdf.prologue );
                    assert( NULL == current_jdf.epilogue );
                    current_jdf.prologue = (yyvsp[(1) - (3)].external_code);
                    current_jdf.epilogue = (yyvsp[(3) - (3)].external_code);
                }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 194 "parsec.y"
    {
                    (yyval.external_code) = new(jdf_external_entry_t);
                    (yyval.external_code)->external_code = (yyvsp[(1) - (1)].string);
                    (yyval.external_code)->lineno = current_lineno;
                }
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 201 "parsec.y"
    {
                    (yyval.external_code) = new(jdf_external_entry_t);
                    (yyval.external_code)->external_code = (yyvsp[(1) - (1)].string);
                    (yyval.external_code)->lineno = current_lineno;
                }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 207 "parsec.y"
    {
                    (yyval.external_code) = NULL;
                }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 212 "parsec.y"
    {
                    jdf_expr_t *el, *pl;

                    (yyvsp[(2) - (2)].function)->next = current_jdf.functions;
                    current_jdf.functions = (yyvsp[(2) - (2)].function);
                    if( NULL != inline_c_functions) {
                        /* Every inline functions declared here where within the context of $2 */
                        for(el = inline_c_functions; NULL != el; el = el->next) {
                            pl = el;
                            el->jdf_c_code.function_context = (yyvsp[(2) - (2)].function);
                        }
                        pl->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 229 "parsec.y"
    {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    jdf_expr_t *el;

                    e->next       = NULL;
                    e->name       = (yyvsp[(2) - (5)].string);
                    e->properties = (yyvsp[(3) - (5)].property);
                    e->expression = (yyvsp[(5) - (5)].expr);
                    e->lineno     = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 254 "parsec.y"
    {
                    jdf_global_entry_t *g, *e = new(jdf_global_entry_t);
                    jdf_expr_t *el;

                    e->next       = NULL;
                    e->name       = (yyvsp[(2) - (3)].string);
                    e->properties = (yyvsp[(3) - (3)].property);
                    e->expression = NULL;
                    e->lineno     = current_lineno;
                    if( current_jdf.globals == NULL ) {
                        current_jdf.globals = e;
                    } else {
                        for(g = current_jdf.globals; g->next != NULL; g = g->next)
                            /* nothing */ ;
                        g->next = e;
                    }                
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 279 "parsec.y"
    {
                    jdf_expr_t *el;
                    if( NULL != inline_c_functions ) {
                        /* Every inline functions declared here where within the context of globals only (no assignment) */
                        for(el = inline_c_functions; NULL != el->next; el = el->next) /* nothing */ ;
                        el->next = current_jdf.inline_c_functions;
                        current_jdf.inline_c_functions = inline_c_functions;
                        inline_c_functions = NULL;
                    }
                }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 292 "parsec.y"
    {
                  (yyval.property) = (yyvsp[(2) - (3)].property);
              }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 296 "parsec.y"
    {
                  (yyval.property) = NULL;
              }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 302 "parsec.y"
    {
                 jdf_def_list_t* assign = new(jdf_def_list_t);
                 assign->next = (yyvsp[(4) - (4)].property);
                 assign->name = strdup((yyvsp[(1) - (4)].string));
                 assign->expr = (yyvsp[(3) - (4)].expr);
                 assign->lineno = current_lineno;
                 (yyval.property) = assign;
              }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 311 "parsec.y"
    {
                 jdf_def_list_t* assign = new(jdf_def_list_t);
                 assign->next = NULL;
                 assign->name = strdup((yyvsp[(1) - (3)].string));
                 assign->expr = (yyvsp[(3) - (3)].expr);
                 assign->lineno = current_lineno;
                 (yyval.property) = assign;
             }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 322 "parsec.y"
    {
                    jdf_function_entry_t *e = new(jdf_function_entry_t);
                    e->fname = (yyvsp[(1) - (11)].string);
                    e->parameters = (yyvsp[(3) - (11)].name_list);
                    e->properties = (yyvsp[(5) - (11)].property);
                    e->definitions = (yyvsp[(6) - (11)].def_list);
                    e->simcost = (yyvsp[(7) - (11)].expr);
                    e->predicate = (yyvsp[(8) - (11)].call);
                    e->dataflow = (yyvsp[(9) - (11)].dataflow);
                    e->priority = (yyvsp[(10) - (11)].expr);
                    e->body = (yyvsp[(11) - (11)].string);
		    e->body_gpu = NULL;
		    e->model = NULL;

                    e->lineno  = current_lineno;

                    (yyval.function) = e;
                }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 341 "parsec.y"
    {
                    jdf_function_entry_t *e = new(jdf_function_entry_t);
                    e->fname = (yyvsp[(1) - (12)].string);
                    e->parameters = (yyvsp[(3) - (12)].name_list);
                    e->properties = (yyvsp[(5) - (12)].property);
                    e->definitions = (yyvsp[(6) - (12)].def_list);
                    e->simcost = (yyvsp[(7) - (12)].expr);
                    e->predicate = (yyvsp[(8) - (12)].call);
                    e->dataflow = (yyvsp[(9) - (12)].dataflow);
                    e->priority = (yyvsp[(10) - (12)].expr);
                    e->body = (yyvsp[(11) - (12)].string);
		    e->model = (yyvsp[(12) - (12)].string);

                    e->lineno  = current_lineno;

                    (yyval.function) = e;
                }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 360 "parsec.y"
    {
		  jdf_function_entry_t *e = new(jdf_function_entry_t);
		  e->fname = (yyvsp[(1) - (12)].string);
		  e->parameters = (yyvsp[(3) - (12)].name_list);
		  e->properties = (yyvsp[(5) - (12)].property);
		  e->definitions = (yyvsp[(6) - (12)].def_list);
		  e->simcost = (yyvsp[(7) - (12)].expr);
		  e->predicate = (yyvsp[(8) - (12)].call);
		  e->dataflow = (yyvsp[(9) - (12)].dataflow);
		  e->priority = (yyvsp[(10) - (12)].expr);
		  e->body = (yyvsp[(11) - (12)].string);
		  e->body_gpu = (yyvsp[(12) - (12)].string);
		  e->model = NULL;
		  
		  e->lineno  = current_lineno;
		    
		  (yyval.function) = e;
	      }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 379 "parsec.y"
    {
		  jdf_function_entry_t *e = new(jdf_function_entry_t);
		  e->fname = (yyvsp[(1) - (13)].string);
		  e->parameters = (yyvsp[(3) - (13)].name_list);
		  e->properties = (yyvsp[(5) - (13)].property);
		  e->definitions = (yyvsp[(6) - (13)].def_list);
		  e->simcost = (yyvsp[(7) - (13)].expr);
		  e->predicate = (yyvsp[(8) - (13)].call);
		  e->dataflow = (yyvsp[(9) - (13)].dataflow);
		  e->priority = (yyvsp[(10) - (13)].expr);
		  e->body = (yyvsp[(11) - (13)].string);
		  e->body_gpu = (yyvsp[(12) - (13)].string);
		  e->model = (yyvsp[(13) - (13)].string);

		  e->lineno  = current_lineno;
		    
		  (yyval.function) = e;
	      }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 405 "parsec.y"
    {
                    jdf_name_list_t *l = new(jdf_name_list_t);
                    l->next = (yyvsp[(3) - (3)].name_list);
                    l->name = (yyvsp[(1) - (3)].string);

                    (yyval.name_list) = l;
                }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 413 "parsec.y"
    {
                    jdf_name_list_t *l = new(jdf_name_list_t);
                    l->next = NULL;
                    l->name = (yyvsp[(1) - (1)].string);

                    (yyval.name_list) = l;
                }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 421 "parsec.y"
    {
                    (yyval.name_list) = NULL;
                }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 428 "parsec.y"
    {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name   = (yyvsp[(1) - (4)].string);
                    l->expr   = (yyvsp[(3) - (4)].expr);
                    l->lineno = current_lineno;
                    l->next   = (yyvsp[(4) - (4)].def_list);

                    (yyval.def_list) = l;
                }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 438 "parsec.y"
    {
                    jdf_def_list_t *l = new(jdf_def_list_t);
                    l->name   = (yyvsp[(1) - (3)].string);
                    l->expr   = (yyvsp[(3) - (3)].expr);
                    l->lineno = current_lineno;
                    l->next   = NULL;

                    (yyval.def_list) = l;
                }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 451 "parsec.y"
    {
                    (yyval.expr) = (yyvsp[(2) - (2)].expr);
                }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 454 "parsec.y"
    {   (yyval.expr) = NULL; }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 458 "parsec.y"
    {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = (yyvsp[(2) - (5)].string);
                  data = jdf_find_or_create_data(&current_jdf, (yyvsp[(2) - (5)].string));
                  c->parameters = (yyvsp[(4) - (5)].expr);
                  JDF_COUNT_LIST_ENTRIES((yyvsp[(4) - (5)].expr), jdf_expr_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    (yyvsp[(2) - (5)].string), nbparams, current_lineno, data->nbparams, data->lineno);
                          YYERROR;
                      }
                  } else {
                      data->nbparams = nbparams;
                      data->lineno = current_lineno;
                  }
                  (yyval.call) = c;                  
              }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 483 "parsec.y"
    {
                    (yyvsp[(1) - (2)].dataflow)->next = (yyvsp[(2) - (2)].dataflow);
                    (yyval.dataflow) = (yyvsp[(1) - (2)].dataflow);
                }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 488 "parsec.y"
    {
                    (yyval.dataflow) = NULL;
                }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 495 "parsec.y"
    {
                    (yyval.number) = (yyvsp[(1) - (1)].number);
                }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 498 "parsec.y"
    { (yyval.number) = JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE; }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 502 "parsec.y"
    {
                    jdf_dataflow_t *flow = new(jdf_dataflow_t);
                    flow->flow_flags = (yyvsp[(1) - (3)].number);
                    flow->varname     = (yyvsp[(2) - (3)].string);
                    flow->deps        = (yyvsp[(3) - (3)].dep);
                    flow->lineno      = current_lineno;

                    (yyval.dataflow) = flow;
                }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 514 "parsec.y"
    {
                   (yyvsp[(1) - (2)].dep)->next = (yyvsp[(2) - (2)].dep);
                   (yyval.dep) = (yyvsp[(1) - (2)].dep);
               }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 519 "parsec.y"
    {
                   (yyval.dep) = NULL;
               }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 525 "parsec.y"
    {
                  struct jdf_name_list *g, *e, *prec;
                  jdf_dep_t *d = new(jdf_dep_t);
                  jdf_expr_t* expr;
                  jdf_def_list_t* property;

                  d->type = (yyvsp[(1) - (3)].dep_type);
                  d->guard = (yyvsp[(2) - (3)].guarded_call);
                  if( NULL == (yyvsp[(3) - (3)].property) ) {
                      (yyvsp[(3) - (3)].property) = jdf_create_properties_list( "type", 0, "DEFAULT", NULL );
                  }
                  (yyvsp[(2) - (3)].guarded_call)->properties = (yyvsp[(3) - (3)].property);

                  expr = jdf_find_property( (yyvsp[(3) - (3)].property), "type", &property );
                  assert( NULL != expr );
                  if( (JDF_VAR != expr->op) && (JDF_STRING != expr->op) ) {
                      printf("Warning: Incorrect value for the \"type\" property defined at line %d\n", property->lineno );
                  } else {
                      for(prec = NULL, g = current_jdf.datatypes; g != NULL; g = g->next) {
                          if( 0 == strcmp(expr->jdf_var, g->name) ) {
                              break;
                          }
                          prec = g;
                      }
                  }
                  if( NULL == g ) {
                      e = new(struct jdf_name_list);
                      e->name = strdup(expr->jdf_var);
                      e->next = NULL;
                      if( NULL != prec ) {
                          prec->next = e;
                      } else {
                          current_jdf.datatypes = e;
                      }
                  }
                  d->datatype_name = strdup(expr->jdf_var);
                  d->lineno = current_lineno;
                  (yyval.dep) = d;
              }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 567 "parsec.y"
    {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_UNCONDITIONAL;
                  g->guard = NULL;
                  g->calltrue = (yyvsp[(1) - (1)].call);
                  g->callfalse = NULL;
                  (yyval.guarded_call) = g;
              }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 576 "parsec.y"
    {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = (yyvsp[(1) - (3)].expr);
                  g->calltrue = (yyvsp[(3) - (3)].call);
                  g->callfalse = NULL;
                  (yyval.guarded_call) = g;
              }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 585 "parsec.y"
    {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = (yyvsp[(1) - (5)].expr);
                  g->calltrue = (yyvsp[(3) - (5)].call);
                  g->callfalse = (yyvsp[(5) - (5)].call);
                  (yyval.guarded_call) = g;
              }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 594 "parsec.y"
    {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_BINARY;
                  g->guard = (yyvsp[(1) - (3)].expr);
                  g->calltrue = (yyvsp[(3) - (3)].call);
                  g->callfalse = NULL;
                  (yyval.guarded_call) = g;
              }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 603 "parsec.y"
    {
                  jdf_guarded_call_t *g = new(jdf_guarded_call_t);
                  g->guard_type = JDF_GUARD_TERNARY;
                  g->guard = (yyvsp[(1) - (5)].expr);
                  g->calltrue = (yyvsp[(3) - (5)].call);
                  g->callfalse = (yyvsp[(5) - (5)].call);
                  (yyval.guarded_call) = g;
              }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 614 "parsec.y"
    {
                  jdf_call_t *c = new(jdf_call_t);
                  c->var = (yyvsp[(1) - (5)].string);
                  c->func_or_mem = (yyvsp[(2) - (5)].string);
                  c->parameters = (yyvsp[(4) - (5)].expr);
                  (yyval.call) = c;
              }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 622 "parsec.y"
    {
                  jdf_data_entry_t* data;
                  jdf_call_t *c = new(jdf_call_t);
                  int nbparams;

                  c->var = NULL;
                  c->func_or_mem = (yyvsp[(1) - (4)].string);
                  c->parameters = (yyvsp[(3) - (4)].expr);
                  (yyval.call) = c;                  
                  data = jdf_find_or_create_data(&current_jdf, (yyvsp[(1) - (4)].string));
                  JDF_COUNT_LIST_ENTRIES((yyvsp[(3) - (4)].expr), jdf_expr_t, next, nbparams);
                  if( data->nbparams != -1 ) {
                      if( data->nbparams != nbparams ) {
                          jdf_fatal(current_lineno, "Data %s used with %d parameters at line %d while used with %d parameters line %d\n",
                                    (yyvsp[(1) - (4)].string), nbparams, current_lineno, data->nbparams, data->lineno);
                          YYERROR;
                      }
                  } else {
                      data->nbparams = nbparams;
                      data->lineno = current_lineno;
                  }
              }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 647 "parsec.y"
    {
                    (yyval.expr) = (yyvsp[(2) - (2)].expr);
              }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 650 "parsec.y"
    { (yyval.expr) = NULL; }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 656 "parsec.y"
    {
                  (yyvsp[(1) - (3)].expr)->next = (yyvsp[(3) - (3)].expr);
                  (yyval.expr)=(yyvsp[(1) - (3)].expr);
              }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 661 "parsec.y"
    {
                  (yyvsp[(1) - (1)].expr)->next = NULL;
                  (yyval.expr)=(yyvsp[(1) - (1)].expr);
              }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 668 "parsec.y"
    {
                  (yyvsp[(1) - (3)].expr)->next = (yyvsp[(3) - (3)].expr);
                  (yyval.expr)=(yyvsp[(1) - (3)].expr);
              }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 673 "parsec.y"
    {
                  (yyvsp[(1) - (1)].expr)->next = NULL;
                  (yyval.expr)=(yyvsp[(1) - (1)].expr);
              }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 680 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_RANGE;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
            }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 688 "parsec.y"
    {
                  (yyval.expr) = (yyvsp[(1) - (1)].expr);
            }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 694 "parsec.y"
    {
                   (yyval.expr) = (yyvsp[(1) - (1)].expr);
               }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 698 "parsec.y"
    {
                   jdf_expr_t *ne;
                   (yyval.expr) = new(jdf_expr_t);
                   (yyval.expr)->op = JDF_C_CODE;
                   (yyval.expr)->jdf_c_code.code = (yyvsp[(1) - (1)].string);
                   (yyval.expr)->jdf_c_code.lineno = current_lineno;
                   /* This will  be set by the upper level parsing if necessary */
                   (yyval.expr)->jdf_c_code.function_context = NULL;
                   (yyval.expr)->jdf_c_code.fname = NULL;

                   (yyval.expr)->next = inline_c_functions;
                   inline_c_functions = (yyval.expr);
               }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 714 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_EQUAL;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 722 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOTEQUAL;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 730 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LESS;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 738 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_LEQ;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 746 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MORE;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 754 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MEQ;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 762 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_AND;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 770 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_OR;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 778 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_XOR;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 786 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_NOT;
                  e->jdf_ua = (yyvsp[(2) - (2)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 793 "parsec.y"
    {
                  (yyval.expr) = (yyvsp[(2) - (3)].expr);
              }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 797 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_PLUS;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 805 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MINUS;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 813 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TIMES;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 821 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_DIV;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 829 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_MODULO;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 837 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHL;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 845 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_SHR;
                  e->jdf_ba1 = (yyvsp[(1) - (3)].expr);
                  e->jdf_ba2 = (yyvsp[(3) - (3)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 853 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_TERNARY;
                  e->jdf_tat = (yyvsp[(1) - (5)].expr);
                  e->jdf_ta1 = (yyvsp[(3) - (5)].expr);
                  e->jdf_ta2 = (yyvsp[(5) - (5)].expr);
                  (yyval.expr) = e;
              }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 862 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_VAR;
                  e->jdf_var = strdup((yyvsp[(1) - (1)].string));
                  (yyval.expr) = e;
              }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 869 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_CST;
                  e->jdf_cst = (yyvsp[(1) - (1)].number);
                  (yyval.expr) = e;
              }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 876 "parsec.y"
    {
                  jdf_expr_t *e = new(jdf_expr_t);
                  e->op = JDF_STRING;
                  e->jdf_var = strdup((yyvsp[(1) - (1)].string));
                  (yyval.expr) = e;
              }
    break;



/* Line 1806 of yacc.c  */
#line 2670 "/home/vcohen/bosilca-dplasma-1c0372a47a55/tools/parsec-compiler/parsec.y.c"
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
#line 884 "parsec.y"



