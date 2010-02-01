#include "dplasma.h"
#include "scheduling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *yyfilename;
extern FILE* __dplasma_graph_file;

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
        fprintf( __dplasma_graph_file, "%s [style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\"];\n",
                 tmp, color, tmp );
    } else {
        printf("Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128));
    }
    return dplasma_trigger_dependencies( eu_context, exec_context, 1 );
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

    /* The dot output file should always be initialized before calling dplasma_init */
    __dplasma_graph_file = fopen("dplasma.dot", "w");

    dplasma = dplasma_init(1, NULL, NULL);

    /* If arguments are provided then they are supposed to initialize some of the
     * global symbols. Try to do so ...
     */
    if( argc > 0 ) {
        int i, j, nb_syms = dplasma_symbol_get_count();
        const symbol_t* symbol;

        for( i = 1; i < argc; i += 2 ) {
            for( j = 0; j < nb_syms; j++ ) {
                symbol = dplasma_symbol_get_element_at(j);
                if( 0 == strcmp(argv[i], symbol->name) ) {
                    expr_t* constant;

                    constant = expr_new_int( atoi(argv[i+1]) );
                    dplasma_assign_global_symbol( symbol->name, constant );
                }
            }
        }
    }

    /* Make sure all symbols are correctly initialized */
    {
        const symbol_t* symbol;
        int i = dplasma_symbol_get_count();
        int uninitialized_symbols = 0;

        for( --i; i >= 0; i-- ) {
            symbol = dplasma_symbol_get_element_at(i);
            if( (NULL == symbol->min) || (NULL == symbol->max) ) {
                printf( "Symbol %s is not initialized\n", symbol->name );
                uninitialized_symbols++;
            }
        }
        if( uninitialized_symbols ) {
            printf( "We cannot generate the dependencies graph if there are uninitialized symbols\n" );
            exit(-1);
        }
    }

    /* Setup generic hook for all services */
    {
        dplasma_t* object;
        int i;
        for( i = 0; NULL != (object = (dplasma_t*)dplasma_element_at(i)); i++ ) {
            object->hook = generic_hook;
            total_nb_tasks += dplasma_compute_nb_tasks( object, 1 );
        }
        dplasma_register_nb_tasks(dplasma, total_nb_tasks);
    }

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
    }
    dplasma_fini(&dplasma);

    fclose(__dplasma_graph_file);

	return 0;
}
