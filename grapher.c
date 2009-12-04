#include "dplasma.h"
#include "scheduling.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern int yyparse();
extern int dplasma_lineno;

static int generic_hook(const dplasma_execution_context_t* exec_context)
{
    char tmp[128];
#ifdef DPLASMA_GENERATE_DOT
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
#else
    printf("Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128));
#endif  /* DPLASMA_GENERATE_DOT */
    return 0;
}

int main(int argc, char *argv[])
{
    dplasma_lineno = 1;
	yyparse();

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
        int i = 0, rc;

        for( i = 0; ; i++ ) {
            memset(&exec_context, 0, sizeof(dplasma_execution_context_t));
            exec_context.function = (dplasma_t*)dplasma_element_at(i);
            if( NULL == exec_context.function ) {
                break;
            }
            dplasma_set_initial_execution_context(&exec_context);
            rc = dplasma_service_can_be_startup( &exec_context );
            if( rc == 0 ) {
                dplasma_schedule(&exec_context);
                dplasma_progress();
            }
        }
#if 0
        /* I know what I'm doing ;) */
        exec_context.function = (dplasma_t*)dplasma_find("POTRF");
        if( NULL == exec_context.function ) {
            exec_context.function = (dplasma_t*)dplasma_find("DGEQRT");
            if( NULL == exec_context.function ) {
                printf("Unable to find the expected function. Giving up.\n");
                exit(-1);
            }
        }
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(&exec_context);
        dplasma_progress();
#endif
    }
    printf("}\n");
	return 0;
}
