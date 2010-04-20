#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include "jdf.h"

extern int yyparse();
extern int current_lineno;
extern FILE *yyin;
char *yyfilename;

int main(int argc, char *argv[])
{
    if( argc != 1 ) {
        yyin = fopen(argv[1], "r");
        if( yyin == NULL ) {
            fprintf(stderr, "unable to open input file %s: %s\n", argv[1], strerror(errno));
            exit(1);
        }
        yyfilename = strdup(argv[1]);
    } else {
        yyfilename = strdup("(stdin)");
    }

    jdf_prepare_parsing();

	if( yyparse() > 0 ) {
        exit(1);
    }

	return 0;
}
