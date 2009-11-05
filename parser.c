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
    } else {
        printf( "CONNARD!!!\n" );
    }
    dplasma_service_to_string(exec_context, tmp, 128);
    printf("%s [style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\"];\n",
           tmp, color, tmp);
    /*printf("Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128));*/
    return 0;
}

int main(int argc, char *argv[])
{
    dplasma_lineno = 1;
	yyparse();

    /*
      Test Thomas
      dplasma_dump_all_c(stdout);
      exit(0);
     */

    /*symbol_dump_all("");*/
    /*dplasma_dump_all();*/

    {
        /* Setup generic hook for all services */
        dplasma_t* object;
        int i;
        for( i = 0; NULL != (object = (dplasma_t*)dplasma_element_at(i)); i++ ) {
            object->hook = generic_hook;
        }
    }

    printf("digraph G {\n");
    {
        dplasma_execution_context_t exec_context;
        /* I know what I'm doing ;) */
        /*exec_context.function = (dplasma_t*)dplasma_find("POTRF");*/
        exec_context.function = (dplasma_t*)dplasma_find("DGEQRT");
        if( NULL == exec_context.function ) {
            printf("Unable to find the expected function. Giving up.\n");
            exit(-1);
        }
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_execute(&exec_context);
    }
    printf("}\n");
	return 0;
}
