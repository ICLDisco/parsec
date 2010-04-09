/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dplasma_config.h"
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
                         dplasma_execution_context_t* exec_context )
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

static int generic_release_dependencies(dplasma_execution_unit_t *eu_context,
                                        const dplasma_execution_context_t *exec_context,
                                        int action_mask,
                                        struct dplasma_remote_deps_t* upstream_remote_deps,
                                        gc_data_t **data)
{
    char tmp[128];
    dplasma_service_to_string(exec_context, tmp, 128);

    fprintf( __dplasma_graph_file, "RELEASE deps for %s\n", tmp );
    return dplasma_trigger_dependencies( eu_context, exec_context, 1 );
}

/**
 * This function generate all possible execution context for a given function with
 * respect to the predicates.
 */
int dplasma_find_start_values( dplasma_execution_context_t* exec_context, int use_predicates )
{
    const expr_t** predicates = (const expr_t**)exec_context->function->preds;
    const dplasma_t* object = exec_context->function;
    int rc, actual_loop;

    DEBUG(( "Function %s (loops %d)\n", object->name, object->nb_locals ));
    if( 0 == object->nb_locals ) {
        /* special case for the IN/OUT obejcts */
        return 0;
    }

    /* Clear the predicates if not needed */
    if( !use_predicates ) predicates = NULL;

    actual_loop = object->nb_locals - 1;
    while(1) {
        int value;

        /* If this can be the starting point task return */
        rc = dplasma_service_can_be_startup( exec_context );
        if( rc == 0 )
            return 0;

        /* Go to the next valid value for this loop context */
        rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                            exec_context->locals, &value );

        /* If no more valid values, go to the previous loop,
         * compute the next valid value and redo and reinitialize all other loops.
         */
        if( rc != EXPR_SUCCESS ) {
            int current_loop = actual_loop;
        one_loop_up:
            DEBUG(("Loop index %d based on %s failed to get next value. Going up ...\n",
                   actual_loop, object->locals[actual_loop]->name));
            if( 0 == actual_loop ) {  /* we're done */
                goto end_of_all_loops;
            }
            actual_loop--;  /* one level up */
            rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                                exec_context->locals, &value );
            if( rc != EXPR_SUCCESS ) {
                goto one_loop_up;
            }
            DEBUG(("Keep going on the loop level %d (symbol %s value %d)\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            for( actual_loop++; actual_loop <= current_loop; actual_loop++ ) {
                rc = dplasma_symbol_get_first_value(object->locals[actual_loop], predicates,
                                                    exec_context->locals, &value );
                if( rc != EXPR_SUCCESS ) {  /* no values for this symbol in this context */
                    goto one_loop_up;
                }
                DEBUG(("Loop index %d based on %s get first value %d\n", actual_loop,
                       object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            }
            actual_loop = current_loop;  /* go back to the original loop */
        } else {
            DEBUG(("Loop index %d based on %s get next value %d\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
        }
    }
 end_of_all_loops:

    return -1;
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

    dplasma = dplasma_init(1, &argc, &argv, 0);

    __dplasma_graph_file = stdout;
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
                    break;
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
            object->release_deps = generic_release_dependencies;
            total_nb_tasks += dplasma_compute_nb_tasks( object, 1 );
        }
        // dplasma_register_nb_tasks is waiting an unsigned int.
        // problem arise with -1...
        if ( total_nb_tasks < 0 ) {
            fprintf( stderr, "Error during task generation, aborting" );
            exit( 1 );
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
            if( 0 == exec_context.function->nb_locals ) {
                continue;
            }
            dplasma_set_initial_execution_context(&exec_context);
            rc = dplasma_find_start_values( &exec_context, 0 );
            if( rc == 0 ) {
                dplasma_schedule(dplasma, &exec_context);
                dplasma_progress(dplasma);
                break;
            }
        }
    }
    dplasma_fini(&dplasma);

    return 0;
}
