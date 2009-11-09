#include <dplasma.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern int yyparse();
extern int dplasma_lineno;

static int generic_hook(const dplasma_execution_context_t* exec_context)
{
    char tmp[128];
    char* color;

    if(0 == strcmp(exec_context->function->name, "DGEQRT") ) {
        color = "#4488AA";
    } else if(0 == strcmp(exec_context->function->name, "DTSQRT") ) {
        color = "#CC99EE";
    } else if(0 == strcmp(exec_context->function->name, "DLARFB") ) {
        color = "#99CCFF";
    } else if(0 == strcmp(exec_context->function->name, "DSSRFB") ) {
        color = "#CCFF00";
    } else if(0 == strcmp(exec_context->function->name, "POTRF") ) {
        color = "#4488AA";
    } else if(0 == strcmp(exec_context->function->name, "SYRK") ) {
        color = "#CC99EE";
    } else if(0 == strcmp(exec_context->function->name, "TRSM") ) {
        color = "#99CCFF";
    } else if(0 == strcmp(exec_context->function->name, "GEMM") ) {
        color = "#CCFF00";
    } else {
        color = "#FFFFFF";
    }
    dplasma_service_to_string(exec_context, tmp, 128);
    printf("%s [style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\"];\n",
           tmp, color, tmp);
    /*printf("Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128));*/
    return 0;
}

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
