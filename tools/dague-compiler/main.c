#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include "jdf.h"
#include "jdf2c.h"

extern int yyparse();
extern int current_lineno;
extern FILE *yyin;
char *yyfilename;

typedef struct args {
    char *input;
    char *output_c;
    char *output_h;
    char *funcid;
    jdf_warning_mask_t wmask;   
} args_t;
static args_t DEFAULTS = {
    .input = "-",
    .output_c = "a.c",
    .output_h = "a.h",
    .funcid = "a",
    .wmask = JDF_ALL_WARNINGS
};
static args_t ARGS = { NULL, };

static void usage(void)
{
    fprintf(stderr, 
            "Usage:\n"
            "  Compile a JDF into a DAGuE representation (.h and .c files)\n"
            "  --input|-i         Input File (JDF) (default '%s')\n"
            "  --output|-o        Set the BASE name for .c, .h and function name (no default).\n"
            "                     Changing this value has precendence over the defaults of\n"
            "                     --output-c, --output-h, and --function-name\n"
            "  --output-c|-C      Set the name of the .c output file (default '%s' or BASE.c)\n"
            "  --output-h|-H      Set the name of the .h output file (default '%s' or BASE.h)\n"
            "  --function-name|-f Set the unique identifier of the generated function\n"
            "                     The generated function will be called DAGuE_<ID>_new\n"
            "                     (default %s)\n"
            "\n"
            " Warning Options: Default is to print ALL warnings. You can disable the following:\n"
            "  --Wmasked          Do NOT print warnings for masked variables\n"
            "\n",
            DEFAULTS.input,
            DEFAULTS.output_c,
            DEFAULTS.output_h,
            DEFAULTS.funcid);            
}

static void parse_args(int argc, char *argv[])
{
    int ch;
    int wmasked;
    char *c = NULL;
    char *h = NULL;
    char *o = NULL;
    char *f = NULL;

    struct option longopts[] = {
        { "input",         required_argument,       NULL,  'i' },
        { "output-c",      required_argument,       NULL,  'C' },
        { "output-h",      required_argument,       NULL,  'H' },
        { "output",        required_argument,       NULL,  'o' },
        { "function-name", required_argument,       NULL,  'f' },
        { "Wmasked",       no_argument,         &wmasked,   1  },
        { "help",          no_argument,             NULL,  'h' },
        { NULL,            0,                       NULL,   0  }
    };

    ARGS.wmask = JDF_ALL_WARNINGS;

    while( (ch = getopt_long(argc, argv, "i:C:H:o:f:h", longopts, NULL)) != -1) {
        switch(ch) {
        case 'i':
            if( NULL != ARGS.input )
                free(ARGS.input);
            ARGS.input = strdup(optarg);
            break;
        case 'C':
            if( NULL != c) 
                free( c );
            c = strdup(optarg);
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
                ARGS.wmask ^= ~JDF_WARN_MASKED_GLOBALS;
            }
        case 'h':
        default:
            usage();
            exit( (ch == 'h') );
        }
    }

    if( NULL == ARGS.input ) {
        ARGS.input = DEFAULTS.input;
    }

    if( NULL == c) {
        if( NULL != o ) {
            ARGS.output_c = (char*)malloc(strlen(o) + 3);
            sprintf(ARGS.output_c, "%s.c", o);
        } else {
            ARGS.output_c = DEFAULTS.output_c;
        }
    } else {
        ARGS.output_c = c;
        c = NULL;
    }

    if( NULL == h ) {
        if( NULL != o ) {
            ARGS.output_h = (char*)malloc(strlen(o) + 3);
            sprintf(ARGS.output_h, "%s.h", o);
        } else {
            ARGS.output_h = DEFAULTS.output_h;
        }
    } else {
        ARGS.output_h = h;
        h = NULL;
    }

    if( NULL == f ) {
        if( NULL != o ) {
            ARGS.funcid = o;
            o = NULL;
        } else {
            ARGS.funcid = DEFAULTS.funcid;
        }
    } else {
        ARGS.funcid = f;
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

    parse_args(argc, argv);
    if( strcmp(ARGS.input, DEFAULTS.input) ) {
        yyin = fopen(ARGS.input, "r");
        if( yyin == NULL ) {
            fprintf(stderr, "unable to open input file %s: %s\n", ARGS.input, strerror(errno));
            exit(1);
        }
        yyfilename = strdup(ARGS.input);
    } else {
        yyfilename = strdup("(stdin)");
    }

    jdf_prepare_parsing();

	if( yyparse() > 0 ) {
        exit(1);
    }

    rc = jdf_sanity_checks( ARGS.wmask );
    if(rc < 0)
        return -1;
    
    if( jdf2c(ARGS.output_c, ARGS.output_h, ARGS.funcid, &current_jdf) < 0 ) {
        return -1;
    }

	return 0;
}
