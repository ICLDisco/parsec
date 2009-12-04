#include <dplasma.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern int yyparse();
extern int dplasma_lineno;

int main(int argc, char *argv[])
{
    int d;

    dplasma_lineno = 1;
	if( (d = yyparse()) > 0 ) {
        exit(1);
    }

    dplasma_dump_all_c(stdout);
    exit(0);

	return 0;
}
