/**
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include "jdf.h"
#include "jdf2c.h"
#include "parsec/utils/argv.h"

#include "parsec.y.h"

extern int current_lineno;
extern int yydebug;
char *yyfilename;
char** extra_argv;

static jdf_compiler_global_args_t DEFAULTS = {
    .input = "-",
    .output_c = "a.c",
    .output_h = "a.h",
    .output_o = "a.o",
    .funcid = "a",
    .wmask = JDF_ALL_WARNINGS,
    .compile = 1,  /* by default the file must be compiled */
    .dep_management = DEP_MANAGEMENT_DYNAMIC_HASH_TABLE,
#if defined(PARSEC_HAVE_INDENT) && !defined(PARSEC_HAVE_AWK)
    .noline = 1 /*< By default, don't print the #line per default if can't fix the line numbers with awk */
#else
    .noline = 0 /*< Otherwise, go for it (without INDENT or with INDENT but without AWK, lines will be ok) */
#endif
};
jdf_compiler_global_args_t JDF_COMPILER_GLOBAL_ARGS = { /* .input = */ NULL, /* .output_c = */ NULL, /* .output_h = */ NULL, /* .output_o = */ NULL,
                                                        /* .funcid = */ NULL, /* .wmask = */ 0x0, /* .compile = */ 1, /* .noline = */ 0 };

static void usage(void)
{
    fprintf(stderr,
            "Usage: parsec_ptgpp [OPTIONS] [-- COMPILER_OPTIONS]\n"
            "  Compile a PTG representation, JDF file, into a PaRSEC representation (.h and .c files)\n"
            "  and compile that .c file into a .o file (unless -E is specified)\n"
            "  unrecognized options and COMPILER_OPTIONS are all added to the options\n"
            "  passed to the final compiler.\n"
            "  recognized OPTIONS are the following:\n"
            "  --debug|-d         Enable bison debug output\n"
            "  --input|-i         Input File (JDF) (default '%s')\n"
            "  --output|-o        Set the BASE name for .c, .h, .o and function name (no default).\n"
            "                     Changing this value has precendence over the defaults of\n"
            "                     --output-c, --output-h, and --function-name\n"
            "  --output-c|-C      Set the name of the .c output file (default '%s' or BASE.c)\n"
            "  --output-h|-H      Set the name of the .h output file (default '%s' or BASE.h)\n"
            "  --function-name|-f Set the unique identifier of the generated function\n"
            "                     The generated function will be called PaRSEC_<ID>_new\n"
            "                     (default %s)\n"
            "\n"
            "  --dep-management|-M Select how dependencies tracking is managed. Possible choices\n"
            "                      are '"DEP_MANAGEMENT_INDEX_ARRAY_STRING"' or '"DEP_MANAGEMENT_DYNAMIC_HASH_TABLE_STRING"'\n"
            "                      (default '%s')\n"
            "\n"
            "  --noline           Do not dump the JDF line number in the .c output file\n"
            "  --line             Force dumping the JDF line number in the .c output file\n"
            "                     Default: %s\n"
            "  --preproc|-E       Stop after the preprocessing stage. The output is generated\n"
            "                     in the form of preprocessed source code, but they are not compiled.\n"
            "  --showme           Print the flags used to compile for preprocessed files\n"
            "\n"
            " Warning Options: Default is to print ALL warnings. You can disable the following:\n"
            "  --Werror           Exit with non zero value if at least one warning is encountered\n"
            "  --Wmasked          Do NOT print warnings for masked variables\n"
            "  --Wmutexin         Do NOT print warnings for non-obvious mutual exclusion of\n"
            "                     input flows\n"
            "  --Wremoteref       Do NOT print warnings for potential remote memory references\n"
            "\n",
            DEFAULTS.input,
            DEFAULTS.output_c,
            DEFAULTS.output_h,
            DEFAULTS.funcid,
            (DEFAULTS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ? DEP_MANAGEMENT_INDEX_ARRAY_STRING :
             (DEFAULTS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ? DEP_MANAGEMENT_DYNAMIC_HASH_TABLE_STRING :
              ("Unknown dep management string"))),
            DEFAULTS.noline?"--noline":"--line");
}

static char** prepare_execv_arguments(void)
{
    /* Count the number of tokens in the CMAKE_PARSEC_C_FLAGS. This version
     * doesn't take \ in account, but should cover the most basic needs. */
    char** flags_argv = parsec_argv_split(CMAKE_PARSEC_C_FLAGS, ' ');
    char** include_argv = parsec_argv_split(CMAKE_PARSEC_C_INCLUDES, ';');

    /* Let's prepare the include_argv by prepending -I to each one */
    int i, token_count = parsec_argv_count(include_argv);
    for( i = 0; i < token_count; i++ ) {
        char* temp = include_argv[i];
        asprintf(&include_argv[i], "-I%s", temp);
        free(temp);
    }

    token_count = 5 + (parsec_argv_count(flags_argv) +
                       parsec_argv_count(include_argv) +
                       parsec_argv_count(extra_argv));
    char** exec_argv = NULL;

    /* Now let's join all arguments together */
    token_count = 0;
    parsec_argv_append(&token_count, &exec_argv, CMAKE_PARSEC_C_COMPILER);
    for( i = 0; i < parsec_argv_count(flags_argv); ++i ) {
        parsec_argv_append(&token_count, &exec_argv, flags_argv[i]);
    }
    free(flags_argv);

    for( i = 0; i < parsec_argv_count(include_argv); ++i ) {
        parsec_argv_append(&token_count, &exec_argv, include_argv[i]);
    }
    free(include_argv);

    for( i = 0; i < parsec_argv_count(extra_argv); ++i ) {
        parsec_argv_append(&token_count, &exec_argv, extra_argv[i]);
    }

    parsec_argv_append(&token_count, &exec_argv, "-c");
    parsec_argv_append(&token_count, &exec_argv, JDF_COMPILER_GLOBAL_ARGS.output_c);
    parsec_argv_append(&token_count, &exec_argv, "-o");
    parsec_argv_append(&token_count, &exec_argv, JDF_COMPILER_GLOBAL_ARGS.output_o);
    return exec_argv;
}

static void parse_args(int argc, char *argv[])
{
    int ch, i;
    int wmasked = 0;
    int wmutexinput = 0;
    int wremoteref = 0;
    int print_jdf_line;
    int werror = 0;
    int token_count = 0;
    char *c = NULL;
    char *O = NULL;
    char *h = NULL;
    char *o = NULL;
    char *f = NULL;

    struct option longopts[] = {
        { "debug",         no_argument,         &yydebug,   0  },
        { "input",         required_argument,       NULL,  'i' },
        { "output-c",      required_argument,       NULL,  'C' },
        { "output-h",      required_argument,       NULL,  'H' },
        { "output-o",      required_argument,       NULL,  'O' },
        { "output",        required_argument,       NULL,  'o' },
        { "function-name", required_argument,       NULL,  'f' },
        { "Wmasked",       no_argument,         &wmasked,   1  },
        { "Wmutexin",      no_argument,     &wmutexinput,   1  },
        { "Wremoteref",    no_argument,      &wremoteref,   1  },
        { "Werror",        no_argument,          &werror,   1  },
        { "noline",        no_argument,  &print_jdf_line,   0  },
        { "line",          no_argument,  &print_jdf_line,   1  },
        { "help",          no_argument,             NULL,  'h' },
        { "preproc",       no_argument,             NULL,  'E' },
        { "showme",        no_argument,             NULL,  's' },
        { "include",       required_argument,       NULL,  'I' },
        { "dep-management",required_argument,       NULL,  'M' },
        { NULL,            0,                       NULL,   0  }
    };

    JDF_COMPILER_GLOBAL_ARGS.wmask = JDF_ALL_WARNINGS;
    JDF_COMPILER_GLOBAL_ARGS.dep_management = DEFAULTS.dep_management;
    
    print_jdf_line = !DEFAULTS.noline;

    while( (ch = getopt_long(argc, argv, "d:i:C:H:o:f:hEsIO:M:", longopts, NULL)) != -1) {
        switch(ch) {
        case 'd':
            yydebug = 1;
            break;
        case 'i':
            if( NULL != JDF_COMPILER_GLOBAL_ARGS.input )
                free(JDF_COMPILER_GLOBAL_ARGS.input);
            JDF_COMPILER_GLOBAL_ARGS.input = strdup(optarg);
            break;
        case 'C':
            if( NULL != c)
                free( c );
            c = strdup(optarg);
            break;
        case 'O':
            if( NULL != O)
                free( O );
            O = strdup(optarg);
            break;
        case 'H':
            if( NULL != h)
                free( h );
            h = strdup(optarg);
            break;
        case 'o':
            if( NULL != o)
                free( o );
            o = strdup(optarg);
            break;
        case 'f':
            if( NULL != f )
                free( f );
            f = strdup(optarg);
            break;
        case 0:
            if( wmasked ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_MASKED_GLOBALS;
            }
            if( wmutexinput ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_MUTUAL_EXCLUSIVE_INPUTS;
            }
            if( wremoteref ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_REMOTE_MEM_REFERENCE;
            }
            if( werror ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask |= JDF_WARNINGS_ARE_ERROR;
            }
            break;
        case 'E':
            /* Don't compile the preprocessed file, instead stop after the preprocessing stage */
            JDF_COMPILER_GLOBAL_ARGS.compile = 0;
            break;
        case 's': {
            /* print the compilation options used to compile the preprocessed output */
            char** exec_argv = prepare_execv_arguments();
            for( int i = 0; i < parsec_argv_count(exec_argv); ++i )
                fprintf(stderr, "%s ", exec_argv[i]);
            fprintf(stderr, "\n");
            free(exec_argv);
            exit(0);
        }
        case 'M':
            if( strcmp(optarg, DEP_MANAGEMENT_DYNAMIC_HASH_TABLE_STRING) == 0 )
                JDF_COMPILER_GLOBAL_ARGS.dep_management = DEP_MANAGEMENT_DYNAMIC_HASH_TABLE;
            else if( strcmp(optarg, DEP_MANAGEMENT_INDEX_ARRAY_STRING) == 0 )
                JDF_COMPILER_GLOBAL_ARGS.dep_management = DEP_MANAGEMENT_INDEX_ARRAY;
            else {
                fprintf(stderr, "Unknown dependencies management method: '%s'\n", optarg);
                usage();
                exit(1);
            }
            break;
        case 'h':
            usage();
            exit(0);
        default:
            /* save the option for later */
            parsec_argv_append(&token_count, &extra_argv, optarg);
        }
    }

    for (i = optind; i < argc; i++) {
        parsec_argv_append(&token_count, &extra_argv, argv[i]);
    }

    JDF_COMPILER_GLOBAL_ARGS.noline = !print_jdf_line;

    if( NULL == JDF_COMPILER_GLOBAL_ARGS.input ) {
        JDF_COMPILER_GLOBAL_ARGS.input = DEFAULTS.input;
    }

    /**
     * If we have the compiled file name just use it. Otherwise use the provided
     * generic name (-o). If none of these succeed, use the default instead.
     */
    if( NULL != O ) {
        JDF_COMPILER_GLOBAL_ARGS.output_o = O;
    } else {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.output_o = (char*)malloc(strlen(o) + 3);
            sprintf(JDF_COMPILER_GLOBAL_ARGS.output_o, "%s.o", o);
        } else
            JDF_COMPILER_GLOBAL_ARGS.output_o = DEFAULTS.output_o;
    }

    if( NULL == c) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.output_c = (char*)malloc(strlen(o) + 3);
            sprintf(JDF_COMPILER_GLOBAL_ARGS.output_c, "%s.c", o);
        } else {
            JDF_COMPILER_GLOBAL_ARGS.output_c = DEFAULTS.output_c;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.output_c = c;
        c = NULL;
    }

    if( NULL == h ) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.output_h = (char*)malloc(strlen(o) + 3);
            sprintf(JDF_COMPILER_GLOBAL_ARGS.output_h, "%s.h", o);
        } else {
            JDF_COMPILER_GLOBAL_ARGS.output_h = DEFAULTS.output_h;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.output_h = h;
        h = NULL;
    }

    if( NULL == f ) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.funcid = o;
            o = NULL;
        } else {
            JDF_COMPILER_GLOBAL_ARGS.funcid = DEFAULTS.funcid;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.funcid = f;
        f = NULL;
    }

    if( NULL != c )
        free(c);
    if( NULL != h )
        free(h);
    if( NULL != o )
        free(o);
}

int main(int argc, char *argv[])
{
    int rc;
    yyscan_t scanner = NULL;

    parse_args(argc, argv);
#if defined(PARSEC_HAVE_RECENT_LEX)
    yylex_init( &scanner );
    yyset_debug( 1, scanner );
#endif  /* defined(PARSEC_HAVE_RECENT_LEX) */
    if( strcmp(JDF_COMPILER_GLOBAL_ARGS.input, DEFAULTS.input) ) {
        FILE* my_file = fopen(JDF_COMPILER_GLOBAL_ARGS.input, "r");
        if( my_file == NULL ) {
            fprintf(stderr, "unable to open input file %s: %s\n", JDF_COMPILER_GLOBAL_ARGS.input, strerror(errno));
            exit(1);
        }
#if defined(PARSEC_HAVE_RECENT_LEX)
        yyset_in( my_file, scanner );
#else
        yyin = my_file;
#endif  /* defined(PARSEC_HAVE_RECENT_LEX) */
        yyfilename = strdup(JDF_COMPILER_GLOBAL_ARGS.input);
    } else {
        yyfilename = strdup("(stdin)");
    }

    jdf_prepare_parsing();

    /*yydebug = 5;*/
    if( yyparse(scanner) > 0 ) {
        exit(1);
    }
#if defined(PARSEC_HAVE_RECENT_LEX)
    yylex_destroy( scanner );
#endif  /* defined(PARSEC_HAVE_RECENT_LEX) */

    rc = jdf_sanity_checks( JDF_COMPILER_GLOBAL_ARGS.wmask );
    if(rc < 0)
        return 1;

    if( (JDF_COMPILER_GLOBAL_ARGS.wmask & JDF_WARNINGS_ARE_ERROR) &&
        (rc != 0) ) {
        return 1;
    }

    /* Lets try to optimize the jdf */
    jdf_optimize( &current_jdf );

    if( jdf2c(JDF_COMPILER_GLOBAL_ARGS.output_c,
              JDF_COMPILER_GLOBAL_ARGS.output_h,
              JDF_COMPILER_GLOBAL_ARGS.funcid,
              &current_jdf) < 0 ) {
        return 1;
    }

    /* Compile the file */
    if(JDF_COMPILER_GLOBAL_ARGS.compile) {

        char** exec_argv = prepare_execv_arguments();
        for( int i = 0; i < parsec_argv_count(exec_argv); ++i )
            fprintf(stderr, "%s ", exec_argv[i]);
        fprintf(stderr, "\n");
        execv(exec_argv[0], exec_argv);
        fprintf(stderr, "Compilation failed with error %d (%s)\n", errno, strerror(errno));
        free(exec_argv);
        return -1;
    }

    return 0;
}
