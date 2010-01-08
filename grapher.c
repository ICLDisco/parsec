#include "dplasma.h"
#include "scheduling.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char *yyfilename;

extern int yyparse();
extern int dplasma_lineno;

static int generate_dots = 1;

static int generic_hook( dplasma_execution_unit_t* eu_context,
                         const dplasma_execution_context_t* exec_context )
{
    char tmp[128];
    if( generate_dots ) {
        char* color;
        
        if(0 == strcmp(exec_context->function->name, "DGEQRT") ) {
            color = "#4488AA";
        } else if(0 == strcmp(exec_context->function->name, "DTSQRT") ) {
            color = "#CC99EE";
        } else if(0 == strcmp(exec_context->function->name, "DORMQR") ) {
            color = "#99CCFF";
        } else if(0 == strcmp(exec_context->function->name, "DSSMQR") ) {
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
    } else {
        printf("Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128));
    }
    return 0;
}

int main(int argc, char *argv[])
{
    dplasma_context_t* dplasma;
    int total_nb_tasks = 0;

    yyfilename = strdup("(stdin)");
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
            total_nb_tasks += dplasma_compute_nb_tasks( object, 1 );
        }
        dplasma_register_nb_tasks(total_nb_tasks);
    }

    dplasma = dplasma_init(1, NULL, NULL);

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
                dplasma_schedule(dplasma, &exec_context);
                dplasma_progress(dplasma);
                break;
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
        dplasma_schedule(dplasma, &exec_context);
        dplasma_progress(dplasma);
#endif
    }
    dplasma_fini(&dplasma);
	return 0;
}
