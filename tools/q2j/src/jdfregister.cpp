/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/wait.h>
#include <assert.h>
#include <map>
#include <set>
#include <list>
#include <sstream>
#include "jdf.h"
#include "string_arena.h"
#include "node_struct.h"
#include "q2j.y.h"
#include "omega.h"
#include "utility.h"
#include "omega_interface.h"
#include "jdfregister.h"

extern char *_q2j_data_prefix;
extern jdf_t _q2j_jdf;

void jdf_register_pools( jdf_t *jdf );

jdf_def_list_t*
jdf_create_properties_list( const char* name,
                            jdf_expr_operand_t op,
                            const char* default_char,
                            jdf_def_list_t* next )
{
    jdf_def_list_t* property;
    jdf_expr_t *e;

    property         = q2jmalloc(jdf_def_list_t, 1);
    property->next   = next;
    property->name   = strdup(name);
    JDF_OBJECT_SET(property, NULL, 0, NULL);

    e = q2jmalloc(jdf_expr_t, 1);
    e->op = op;
    e->jdf_var = strdup(default_char);

    property->expr = e;
    return property;
}

jdf_global_entry_t*
jdf_create_global_entry_list( const char         *name,
                              jdf_def_list_t     *properties,
                              jdf_expr_t         *expression,
                              jdf_data_entry_t   *data,
                              jdf_global_entry_t *next )
{
    jdf_global_entry_t *global_entry;

    global_entry         = q2jmalloc(jdf_global_entry_t, 1);
    global_entry->next   = next;
    global_entry->properties = properties;
    global_entry->expression = expression;
    global_entry->name   = strdup(name);
    JDF_OBJECT_SET(global_entry, NULL, 0, NULL);

    return global_entry;
}

jdf_call_t*
jdf_generate_call_for_data(node_t *data,
                           str_pair_t *subs)
{
    string_arena_t *sa;
    jdf_call_t *call;
    char *str = NULL;
    int i;

    assert( (data != NULL) && (data->type == ARRAY) );

    sa = string_arena_new(16);
    string_arena_add_string( sa, "%s%s",
                             _q2j_data_prefix,
                             tree_to_str_with_substitutions(data->u.kids.kids[0], subs) );

    // Add the predicate for locality
    call = q2jmalloc(jdf_call_t, 1);
    call->var         = NULL;
    call->func_or_mem = strdup(string_arena_get_string(sa));
    call->parameters  = q2jmalloc( jdf_expr_t, 1);

    // TODO: need to be replaced by correct expression
    for(i=1; i<data->u.kids.kid_count; ++i){
        if( i > 1 )
            str = append_to_string( str, ",", NULL, 0);
        str = append_to_string( str, tree_to_str_with_substitutions(data->u.kids.kids[i], subs), NULL, 0 );
    }

    call->parameters->next = NULL;
    call->parameters->op   = JDF_VAR;
    call->parameters->jdf_var = strdup(str);

    string_arena_free(sa);
    return call;
}

////////////////////////////////////////////////////////////////////////////////
//
void jdf_register_prologue(jdf_t *jdf)
{
    if ( jdf->prologue == NULL ){
        jdf->prologue = q2jmalloc(jdf_external_entry_t, 1);
        jdf->prologue->external_code = strdup(
            "/*\n"
            " *  Copyright (c) 2010\n"
            " *\n"
            " *  The University of Tennessee and The University\n"
            " *  of Tennessee Research Foundation.  All rights\n"
            " *  reserved.\n"
            " *\n"
            " * @precisions normal z -> s d c\n"
            " *\n"
            " */\n"
            "#define PRECISION_z\n"
            "\n"
            "#include <plasma.h>\n"
            "#include <core_blas.h>\n"
            "\n"
            "#include \"dague.h\"\n"
            "#include \"data_distribution.h\"\n"
            "#include \"data_dist/matrix/precision.h\"\n"
            "#include \"data_dist/matrix/matrix.h\"\n"
            "#include \"dplasma/lib/memory_pool.h\"\n"
            "#include \"dplasma/lib/dplasmajdf.h\"\n");
        JDF_OBJECT_SET(jdf->prologue, NULL, 0, NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
//
void jdf_register_globals(jdf_t *jdf, node_t *root)
{
    symtab_t *scope;
    q2j_symbol_t *sym;
    jdf_global_entry_t *prev, *e, *e2;
    string_arena_t *sa;

    sa = string_arena_new(64);
    assert(jdf->globals == NULL);

    scope = root->symtab;
    do{
        for(sym=scope->symbols; NULL!=sym; sym=sym->next){
            if( !strcmp(sym->var_type, "PLASMA_desc") ){

                e = q2jmalloc(jdf_global_entry_t, 2);
                e2 = e+1;

                /* Data */
                string_arena_init(sa);
                string_arena_add_string(sa, "%s%s", _q2j_data_prefix, sym->var_name);

                e->next       = e2;
                e->name       = strdup(string_arena_get_string(sa));
                e->properties = jdf_create_properties_list( "type", JDF_STRING, "dague_ddesc_t *", e->properties);
                e->expression = NULL;
                JDF_OBJECT_SET(e, NULL, 0, NULL);

                /* Descriptor */
                string_arena_init(sa);
                string_arena_add_string(sa, "desc%s", sym->var_name);
                e2->next       = NULL;
                e2->name       = strdup(string_arena_get_string(sa));
                e2->properties = NULL;
                e2->expression = NULL;
                JDF_OBJECT_SET(e2, NULL, 0, NULL);

                string_arena_init(sa);
                string_arena_add_string(sa, "*((tiled_matrix_desc_t*)%s%s)",
                                        _q2j_data_prefix, sym->var_name);

                // Inverse order
                e2->properties = jdf_create_properties_list( "default", JDF_STRING,
                                                             string_arena_get_string(sa),
                                                             e2->properties);
                e2->properties = jdf_create_properties_list( "hidden", JDF_VAR,    "on",                  e2->properties);
                e2->properties = jdf_create_properties_list( "type",   JDF_STRING, "tiled_matrix_desc_t", e2->properties);

            } else {

                e = q2jmalloc(jdf_global_entry_t, 1);
                e2 = e;

                /* Data */
                string_arena_init(sa);
                string_arena_add_string(sa, "%s%s", _q2j_data_prefix, sym->var_name);

                e->next       = NULL;
                e->name       = strdup(sym->var_name);
                e->properties = jdf_create_properties_list( "type", JDF_STRING, sym->var_type, e->properties);
                e->expression = NULL;
                JDF_OBJECT_SET(e, NULL, 0, NULL);
            }

            if (jdf->globals == NULL) {
                jdf->globals = e;
            } else {
                prev->next = e;
            }
            prev = e2;
        }
        scope = scope->parent;
    } while(NULL != scope);

    /*
     * Create pool declarations
     */
    jdf_register_pools( jdf );

    string_arena_free(sa);
    return;
}

/**
 * Create and initialize a default datatype. This is a datatype from a
 * specific ARENA, with a specified count and displ.
 */
static int
jdf_set_default_datatype(jdf_datatransfer_type_t* datatype,
                         char* default_ddt,
                         int count,
                         int displ)
{
    datatype->type = q2jmalloc(jdf_expr_t, 1);
    if( NULL == datatype->type ) return -1;
    datatype->type->next    = NULL;
    datatype->type->op      = JDF_STRING;
    datatype->type->jdf_var = default_ddt;
    datatype->count = q2jmalloc(jdf_expr_t, 1);
    if( NULL == datatype->count ) return -1;
    datatype->count->next    = NULL;
    datatype->count->op      = JDF_CST;
    datatype->count->jdf_cst = count;
    datatype->displ = q2jmalloc(jdf_expr_t, 1);
    if( NULL == datatype->displ ) return -1;
    datatype->displ->next    = NULL;
    datatype->displ->op      = JDF_CST;
    datatype->displ->jdf_cst = displ;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//
jdf_call_t *jdf_register_pseudotask(jdf_t *jdf,
                                    jdf_function_entry_t *parent_task,
                                    Relation S_es, Relation cond,
                                    node_t *data_element,
                                    char *var_pseudoname,
                                    int ptask_count, const char *inout )
{
    int var_count;
    char *data_str;

    Relation newS_es;
    set <const char *> prev_vars;
    str_pair_t *solved_vars;
    string_arena_t *sa1, *sa2;

    jdf_function_entry_t *pseudotask;
    jdf_name_list_t *last_param;
    jdf_def_list_t  *last_definition;
    jdf_expr_t *parent_parameters = NULL;
    jdf_expr_t *last_parent_parameters;

    jdf_dep_t *input, *output;
    jdf_call_t *parent_call;

    int is_input = !strcmp(inout,"in");

    pseudotask = q2jmalloc(jdf_function_entry_t, 1);
    pseudotask->next        = parent_task->next;
    parent_task->next       = pseudotask;
    pseudotask->fname       = NULL;
    pseudotask->parameters  = NULL;
    pseudotask->dep_flags       = is_input ? JDF_FUNCTION_FLAG_CAN_BE_STARTUP : 0;
    pseudotask->properties  = NULL;
    pseudotask->locals      = NULL;
    pseudotask->simcost     = NULL;
    pseudotask->predicate   = NULL;
    pseudotask->dataflow    = NULL;
    pseudotask->priority    = NULL;
    pseudotask->body        = strdup(
        "    /* nothing */" );

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    // Create pseudotask name
    string_arena_add_string( sa1, "%s_%s_data_%s%d",
                             parent_task->fname, inout,
                             tree_to_str(DA_array_base(data_element)),
                             ptask_count );
    pseudotask->fname = strdup( string_arena_get_string(sa1) );

    // Add the profile property
    pseudotask->properties = q2jmalloc(jdf_def_list_t, 1);
    JDF_OBJECT_SET(pseudotask->properties, NULL, 0, NULL);
    pseudotask->properties->next       = NULL;
    pseudotask->properties->name       = strdup("profile");
    pseudotask->properties->expr       = q2jmalloc(jdf_expr_t, 1);
    pseudotask->properties->properties = NULL;
    pseudotask->properties->expr->next    = NULL;
    pseudotask->properties->expr->op      = JDF_VAR;
    pseudotask->properties->expr->jdf_var = strdup("off");

    // Find the maximum number of variable substitutions we might need and add one for the termination flag.
    JDF_COUNT_LIST_ENTRIES( parent_task->parameters, jdf_name_list_t, next, var_count);
    solved_vars = (str_pair_t *)calloc(var_count+1, sizeof(str_pair_t));
    var_count = 0;

    // There is at least one parameter
    assert( parent_task->parameters != NULL );

    if( !cond.is_null() ){
        newS_es = Intersection(copy(S_es), Domain(copy(cond)));
    }else{
        newS_es = copy(S_es);
    }

    for(jdf_name_list_t *var = parent_task->parameters;
        var != NULL; var = var->next ) {
        const char *var_name = var->name;
        expr_t *expr = relation_to_tree(newS_es);
        expr_t *solution = solve_expression_tree_for_var(expr, var_name, copy(newS_es));
        // If there is a solution it means that this parameter has a fixed value and not a range.
        // That means that there is no point in including it as a parameter of the pseudo-task.
        if( NULL != solution ){
            const char *solution_str = expr_tree_to_str(solution);
            solved_vars[var_count].str1 = var_name;
            solved_vars[var_count].str2 = solution_str;
            var_count++;

            /* Add the expression to the parent call */
            {
                jdf_expr_t *e = q2jmalloc(jdf_expr_t, 1);

                e->next    = NULL;
                e->op      = JDF_VAR;
                e->jdf_var = strdup(solution_str);

                if (parent_parameters == NULL) {
                    parent_parameters = e;
                } else {
                    last_parent_parameters->next = e;
                }
                last_parent_parameters = e;
            }
        } else {

            /* Update the execution space */
            {
                jdf_def_list_t *d = q2jmalloc(jdf_def_list_t, 1);
                d->next       = NULL;
                d->name       = strdup(var_name);
                d->expr       = q2jmalloc(jdf_expr_t, 1);
                d->properties = NULL;
                JDF_OBJECT_SET(d, NULL, 0, NULL);
                d->expr->next    = NULL;
                d->expr->op      = JDF_VAR;
                d->expr->jdf_var = strdup( find_bounds_of_var(expr, var_name,
                                                              prev_vars, copy(newS_es)) );

                if (pseudotask->locals == NULL) {
                    pseudotask->locals = d;
                } else {
                    last_definition->next = d;
                }
                last_definition = d;
            }
            /* Add the parameter */
            {
                jdf_name_list_t *n = q2jmalloc(jdf_name_list_t, 1);
                n->name = strdup(var_name);
                if (pseudotask->parameters == NULL) {
                    pseudotask->parameters = n;
                } else {
                    last_param->next = n;
                }
                last_param = n;
            }
            /* Add the expression to the parent call */
            {
                jdf_expr_t *e = q2jmalloc(jdf_expr_t, 1);

                e->next    = NULL;
                e->op      = JDF_VAR;
                e->jdf_var = strdup(var_name);

                if (parent_parameters == NULL) {
                    parent_parameters = e;
                } else {
                    last_parent_parameters->next = e;
                }
                last_parent_parameters = e;
            }
            prev_vars.insert(var_name);
        }
        clean_tree(expr);
        clean_tree(solution);
    }

    // Delete the "previous variables" set, to clean up some memory
    while(!prev_vars.empty()){
        prev_vars.erase(prev_vars.begin());
    }

    // Now that we know which parameters define the execution space of
    // this pseudo-task, we can finish to generate the pseudo-task
    //

    // Data string
    string_arena_init(sa2);
    string_arena_add_string( sa2, "%s%s", _q2j_data_prefix,
                             tree_to_str_with_substitutions(data_element, solved_vars) );
    data_str = string_arena_get_string(sa2);

    // Add the predicate for locality
    pseudotask->predicate = jdf_generate_call_for_data( data_element, solved_vars );

    // Add the 2 dataflows
    pseudotask->dataflow = q2jmalloc(jdf_dataflow_t, 1);
    pseudotask->dataflow->next = NULL;
    pseudotask->dataflow->varname     = strdup(var_pseudoname);
    pseudotask->dataflow->deps        = q2jmalloc(jdf_dep_t, 2);
    pseudotask->dataflow->flow_flags = JDF_FLOW_TYPE_READ;
    JDF_OBJECT_SET(pseudotask->dataflow, NULL, 0, NULL);

    input  =  pseudotask->dataflow->deps;
    output = (pseudotask->dataflow->deps)+1;

    // Input
    input->next   = output;
    input->type   = JDF_DEP_FLOW_IN;
    input->guard  = q2jmalloc(jdf_guarded_call_t, 1);
    jdf_set_default_datatype(&input->datatype, "DEFAULT", 1, 0);
    JDF_OBJECT_SET(input, NULL, 0, NULL);

    input->guard->guard_type = JDF_GUARD_UNCONDITIONAL;
    input->guard->guard      = NULL;
    input->guard->properties = NULL;
    input->guard->calltrue   = NULL;
    input->guard->callfalse  = NULL;

    if (is_input) {
        input->guard->calltrue = pseudotask->predicate;
    } else {
        input->guard->calltrue = q2jmalloc(jdf_call_t, 1);
        input->guard->calltrue->var         = strdup(var_pseudoname);
        input->guard->calltrue->func_or_mem = parent_task->fname;
        input->guard->calltrue->parameters  = parent_parameters;
    }

    // Output
    output->next   = NULL;
    output->type   = JDF_DEP_FLOW_OUT;
    output->guard  = q2jmalloc(jdf_guarded_call_t, 1);
    jdf_set_default_datatype(&output->datatype, "DEFAULT", 1, 0);
    JDF_OBJECT_SET(output, NULL, 0, NULL);

    output->guard->guard_type = JDF_GUARD_UNCONDITIONAL;
    output->guard->guard      = NULL;
    output->guard->properties = NULL;
    output->guard->calltrue   = NULL;
    output->guard->callfalse  = NULL;

    if (!is_input) {
        output->guard->calltrue = pseudotask->predicate;
    } else {
        output->guard->calltrue = q2jmalloc(jdf_call_t, 1);
        output->guard->calltrue->var         = strdup(var_pseudoname);
        output->guard->calltrue->func_or_mem = parent_task->fname;
        output->guard->calltrue->parameters  = parent_parameters;
    }

    // Create the dependency to the pseudo task for the parent_function
    parent_call = q2jmalloc(jdf_call_t, 1);
    parent_call->var         = strdup(var_pseudoname);
    parent_call->func_or_mem = pseudotask->fname;
    parent_call->parameters = q2jmalloc(jdf_expr_t, 1);

    last_param = pseudotask->parameters;
    last_parent_parameters = parent_call->parameters;
    while ( last_param != NULL ) {
        last_parent_parameters->op      = JDF_VAR;
        last_parent_parameters->jdf_var = last_param->name;

        if ( last_param->next != NULL )
            last_parent_parameters->next = q2jmalloc(jdf_expr_t, 1);
        last_param = last_param->next;
        last_parent_parameters = last_parent_parameters->next;
    }

    string_arena_free(sa1);
    string_arena_free(sa2);

    return parent_call;
}

void jdf_register_locals( jdf_function_entry_t *this_function,
                          Relation S_es)
{
    jdf_def_list_t *locals;
    set<const char *> prev_vars;
    int i;

    // Malloc and chain the locals
    locals = q2jmalloc( jdf_def_list_t, S_es.n_set() );
    for (i=0; i< (int)(S_es.n_set()-1); i++) {
        locals[i].next = &(locals[i+1]);
    }
    this_function->locals = locals;

    // Print the execution space based on the bounds that exist in the relation.
    for(i=1; i<=S_es.n_set(); i++){
        const char *var_name = strdup(S_es.set_var(i)->char_name());
        expr_t *e = relation_to_tree(S_es);
        expr_t *solution = solve_expression_tree_for_var(e, var_name, S_es);

        locals[i-1].name = strdup(var_name);
        locals[i-1].expr = q2jmalloc(jdf_expr_t, 1);
        locals[i-1].properties = NULL;
        JDF_OBJECT_SET(&locals[i-1], NULL, 0, NULL);

        if( NULL != solution ) {
            locals[i-1].expr->next    = NULL;
            locals[i-1].expr->op      = JDF_VAR;
            locals[i-1].expr->jdf_var = strdup( expr_tree_to_str(solution) );
        } else {
            locals[i-1].expr->next    = NULL;
            locals[i-1].expr->op      = JDF_VAR;
            locals[i-1].expr->jdf_var = strdup( find_bounds_of_var(e, var_name, prev_vars, S_es) );
        }

        clean_tree(e);
        clean_tree(solution);
        prev_vars.insert(var_name);
    }

    // Do some memory clean-up
    while(!prev_vars.empty()){
        free((void *)*prev_vars.begin());
        prev_vars.erase(prev_vars.begin());
    }

    return;
}


jdf_expr_t *jdf_generate_condition_str( expr_t *cond_exp)
{
    jdf_expr_t *expr = NULL;
    string cond = expr_tree_to_str(cond_exp);
    if( !cond.empty() ){
        expr = q2jmalloc(jdf_expr_t, 1);
        expr->next = NULL;
        expr->op = JDF_VAR;
        expr->jdf_var = strdup(cond.c_str());
    }
    return expr;
}

////////////////////////////////////////////////////////////////////////////////
//
jdf_expr_t *jdf_generate_call_parameters( dep_t *dep, expr_t *rel_exp )
{
    string_arena_t *sa;
    jdf_expr_t *parameters = NULL;
    jdf_expr_t *param;
    int i, dst_count;

    Relation R = copy(*(dep->rel));
    set <const char *> vars_in_bounds;

    sa = string_arena_new(16);

    dst_count = R.n_inp();
    for(i=0; i<dst_count; i++){
        const char *var_name = strdup(R.input_var(i+1)->char_name());
        vars_in_bounds.insert(var_name);
    }

    dst_count = R.n_out();

    if (dst_count > 0)
        parameters = q2jmalloc(jdf_expr_t, dst_count);
    param = parameters;

    for(i=0; i<dst_count; i++, param++){
        const char *var_name = strdup(R.output_var(i+1)->char_name());
        expr_t *solution = solve_expression_tree_for_var(copy_tree(rel_exp), var_name, R);

        string_arena_init(sa);
        string_arena_add_string( sa, "%s",
                                ( NULL != solution ) ? expr_tree_to_str(solution)
                                : find_bounds_of_var(copy_tree(rel_exp), var_name, vars_in_bounds, R));
        free((void *)var_name);

        param->next = i < (dst_count-1) ? param+1 : NULL;
        param->op = JDF_VAR;
        param->jdf_var = strdup( string_arena_get_string(sa) );
    }

    string_arena_free(sa);
    return parameters;
}

void jdf_register_input_deps( set<dep_t*> ideps,
                              Relation    S_es,
                              node_t *reference_data_element,
                              jdf_function_entry_t *this_function,
                              jdf_dataflow_t *dataflow,
                              int *pseudotask_count)
{
    set<dep_t *>::iterator dep_it;
    jdf_dep_t *dep = dataflow->deps;
    string_arena_t *sa;
    sa  = string_arena_new(16);

    // print the incoming edges
    for (dep_it=ideps.begin(); dep_it!=ideps.end(); dep_it++ ){
        node_t   *src = (*dep_it)->src;
        node_t   *dst = (*dep_it)->dst;
        Relation *rel = (*dep_it)->rel;

        list< pair<expr_t *, Relation> > cond_list;
        list< pair<expr_t *, Relation> >::iterator cond_it;

        // Needed by Omega
        (void)(*rel).print_with_subs_to_string(false);

        // check that the input dependency has a destination
        Q2J_ASSERT( NULL != dst);

        // If the condition has disjunctions (logical OR operators) then split them so that each one
        // is treated independently.
        cond_list = simplify_conditions_and_split_disjunctions(*rel, S_es);
        for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
            jdf_expr_t *dep_expr;
            jdf_call_t *dep_call;

            if ( dep == NULL ) {
                dataflow->deps = q2jmalloc(jdf_dep_t, 1);
                dep = dataflow->deps;
            } else {
                dep->next = q2jmalloc(jdf_dep_t, 1);
                dep = dep->next;
            }

            dep->next = NULL;
            dep->dep_flags = JDF_DEP_FLOW_IN;
            dep->guard = q2jmalloc( jdf_guarded_call_t, 1);
            jdf_set_default_datatype(&dep->datatype, "DEFAULT", 1, 0);
            JDF_OBJECT_SET(dep, NULL, 0, NULL);

            // Generate the dep_expr
            dep_expr = jdf_generate_condition_str( cond_it->first );

            // Generate the dep_call
            if( NULL != src->function ){
                expr_t *e = relation_to_tree(cond_it->second);
                dep_call = q2jmalloc(jdf_call_t, 1);
                dep_call->var         = strdup(src->var_symname);
                dep_call->func_or_mem = src->function->fname;
                dep_call->parameters  = jdf_generate_call_parameters(*dep_it, e );
                clean_tree(e);
            } else { // ENTRY
                if( need_pseudotask(dst, reference_data_element) ){
                    dep_call = jdf_register_pseudotask( &_q2j_jdf, this_function,
                                                        S_es, cond_it->second,
                                                        dst, dataflow->varname,
                                                        (*pseudotask_count)++, "in" );
                } else {
                    dep_call = jdf_generate_call_for_data( dst, NULL );
                }
            }

            // guarded call
            dep->guard->guard_type = dep_expr == NULL ? JDF_GUARD_UNCONDITIONAL : JDF_GUARD_BINARY;
            dep->guard->guard      = dep_expr;
            dep->guard->properties = NULL; /*TODO: get datatype */
            dep->guard->calltrue   = dep_call;
            dep->guard->callfalse  = NULL;
        }

        for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
            clean_tree( (*cond_it).first );
        }
    }

    string_arena_free(sa);
}

void jdf_register_fake_read( Relation S_es,
                             node_t *reference_data_element,
                             jdf_function_entry_t *this_function,
                             jdf_dataflow_t *dataflow,
                             node_t *dst,
                             int *pseudotask_count)
{
    Relation emptyR;
    jdf_dep_t  *dep = dataflow->deps;
    jdf_call_t *dep_call;
    string_arena_t *sa;
    sa  = string_arena_new(16);

    if ( dataflow->deps != NULL ) {
        while( dep->next != NULL ) {
            dep = dep->next;
        }
        dep->next = q2jmalloc(jdf_dep_t, 1);
        dep = dep->next;
    } else {
        dataflow->deps = q2jmalloc(jdf_dep_t, 1);
        dep = dataflow->deps;
    }

    dep->next = NULL;
    dep->dep_flags = JDF_DEP_FLOW_IN;
    dep->guard = q2jmalloc( jdf_guarded_call_t, 1);
    dep->datatype.type = q2jmalloc(jdf_expr_t, 1);
    dep->datatype.type->next    = NULL;
    dep->datatype.type->op      = JDF_STRING;
    dep->datatype.type->jdf_var = "DEFAULT";
    dep->datatype.count = q2jmalloc(jdf_expr_t, 1);
    dep->datatype.count->next    = NULL;
    dep->datatype.count->op      = JDF_CST;
    dep->datatype.count->jdf_cst = 1;
    dep->datatype.displ = q2jmalloc(jdf_expr_t, 1);
    dep->datatype.displ->next    = NULL;
    dep->datatype.displ->op      = JDF_CST;
    dep->datatype.displ->jdf_cst = 0;
    JDF_OBJECT_SET(dep, NULL, 0, NULL);

    if( need_pseudotask(dst, reference_data_element) ){
        dep_call = jdf_register_pseudotask( &_q2j_jdf, this_function,
                                            S_es, emptyR,
                                            dst, dataflow->varname,
                                            (*pseudotask_count)++, "in" );
    } else {
        dep_call = jdf_generate_call_for_data( dst, NULL );
    }

    // guarded call
    dep->guard->guard_type = JDF_GUARD_UNCONDITIONAL;
    dep->guard->guard      = NULL;
    dep->guard->properties = NULL; /*TODO: get datatype */
    dep->guard->calltrue   = dep_call;
    dep->guard->callfalse  = NULL;

    string_arena_free(sa);
}

void jdf_register_output_deps( set<dep_t*> odeps,
                               Relation    S_es,
                               node_t *reference_data_element,
                               jdf_function_entry_t *this_function,
                               jdf_dataflow_t *dataflow,
                               int *pseudotask_count)
{
    set<dep_t *>::iterator dep_it;
    jdf_dep_t *dep = dataflow->deps;
    string_arena_t *sa;
    sa  = string_arena_new(16);

    if ( dataflow->deps != NULL ) {
        while( dep->next != NULL ) {
            dep = dep->next;
        }
    }

    // print the incoming edges
    for (dep_it=odeps.begin(); dep_it!=odeps.end(); dep_it++ ){
        node_t   *src = (*dep_it)->src;
        node_t   *dst = (*dep_it)->dst;
        Relation *rel = (*dep_it)->rel;

        list< pair<expr_t *, Relation> > cond_list;
        list< pair<expr_t *, Relation> >::iterator cond_it;

        // Needed by Omega
        (void)(*rel).print_with_subs_to_string(false);

        // check that the input dependency has a destination
        Q2J_ASSERT( NULL != src );

        // If the condition has disjunctions (logical OR operators) then split them so that each one
        // is treated independently.
        cond_list = simplify_conditions_and_split_disjunctions(*rel, S_es);
        for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
            jdf_expr_t *dep_expr;
            jdf_call_t *dep_call;

            if ( dataflow->deps == NULL ) {
                dataflow->deps = q2jmalloc(jdf_dep_t, 1);
                dep = dataflow->deps;
            } else {
                dep->next = q2jmalloc(jdf_dep_t, 1);
                dep = dep->next;
            }

            dep->next = NULL;
            dep->dep_flags = JDF_DEP_FLOW_OUT;
            dep->guard = q2jmalloc( jdf_guarded_call_t, 1);
            jdf_set_default_datatype(&dep->datatype, "DEFAULT", 1, 0);
            JDF_OBJECT_SET(dep, NULL, 0, NULL);

            // Generate the dep_expr
            dep_expr = jdf_generate_condition_str( cond_it->first );

            // Generate the dep_call
            if( NULL != dst ){
                expr_t *e = relation_to_tree(cond_it->second);
                dep_call = q2jmalloc(jdf_call_t, 1);
                dep_call->var         = strdup(dst->var_symname);
                dep_call->func_or_mem = dst->function->fname;
                dep_call->parameters  = jdf_generate_call_parameters(*dep_it, e);
                clean_tree(e);
            } else { // EXIT
                if( need_pseudotask(src, reference_data_element) ){
                    dep_call = jdf_register_pseudotask( &_q2j_jdf, this_function,
                                                        S_es, cond_it->second,
                                                        src, dataflow->varname,
                                                        (*pseudotask_count)++, "out" );
                } else {
                    dep_call = jdf_generate_call_for_data( src, NULL );
                }
            }

            // guarded call
            dep->guard->guard_type = dep_expr == NULL ? JDF_GUARD_UNCONDITIONAL : JDF_GUARD_BINARY;
            dep->guard->guard      = dep_expr;
            dep->guard->properties = NULL; /*TODO: get datatype */
            dep->guard->calltrue   = dep_call;
            dep->guard->callfalse  = NULL;
        }

        for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
            clean_tree( (*cond_it).first );
        }
    }

    string_arena_free(sa);
}

////////////////////////////////////////////////////////////////////////////////
//
void jdf_register_dependencies_and_pseudotasks(jdf_function_entry_t       *this_function,
                                               node_t                     *reference_data_element,
                                               Relation                    S_es,
                                               set<char *>                &vars,
                                               map<char *, set<dep_t *> > &outg_map,
                                               map<char *, set<dep_t *> > &incm_map)
{
    int i, pseudotask_count = 0;
    jdf_dataflow_t *dataflows, *dataflow;

    if( outg_map.empty() && incm_map.empty() ){
        return;
    }

    // Malloc and chain the dataflow
    dataflows = q2jmalloc( jdf_dataflow_t, vars.size() );
    for (i=0; i< (int)(vars.size()-1); i++) {
        dataflows[i].next = &(dataflows[i+1]);
    }

    this_function->dataflow = dataflows;
    dataflow = dataflows;

    // For each variable print all the incoming and the outgoing edges
    set<char *>::iterator var_it;
    for (var_it=vars.begin(); var_it!=vars.end(); var_it++, dataflow = dataflow->next ){
        bool insert_fake_read = false;
        char *var_pseudoname = *var_it;
        set<dep_t *>ideps = incm_map[var_pseudoname];
        set<dep_t *>odeps = outg_map[var_pseudoname];
        int nb_ideps, nb_odeps, nb_deps;

        nb_ideps = ideps.size();
        nb_odeps = odeps.size();
        nb_deps = nb_ideps + nb_odeps;

        assert( dataflow != NULL );
        dataflow->varname = strdup( var_pseudoname );
        JDF_OBJECT_SET(dataflow, NULL, 0, NULL);
        dataflow->deps = NULL;

        if( nb_ideps > 0 && nb_odeps > 0 ){
            dataflow->flow_flags = JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE;
        }else if( nb_ideps > 0 ){
            dataflow->flow_flags = JDF_FLOW_TYPE_READ;
        }else if( nb_odeps > 0 ){
            /*
             * DAGuE does not like write-only variables, so make it RW and make
             * it read from the data matrix tile that corresponds to this variable.
             */
            dataflow->flow_flags = JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE;
            insert_fake_read = true;
        }else{
            Q2J_ASSERT(0);
        }

        if( nb_ideps > MAX_DEP_IN_COUNT )
            fprintf(stderr,"WARNING: Number of incoming edges (%d) for variable \"%s\" exceeds %d\n",
                    nb_ideps, var_pseudoname, MAX_DEP_IN_COUNT);
        if( nb_odeps > MAX_DEP_OUT_COUNT )
            fprintf(stderr,"WARNING: Number of outgoing edges (%d) for variable \"%s\" exceeds %d\n",
                    nb_odeps, var_pseudoname, MAX_DEP_OUT_COUNT);

        jdf_register_input_deps( ideps, S_es,
                                 reference_data_element,
                                 this_function,
                                 dataflow,
                                 &pseudotask_count );

        if(insert_fake_read){
            jdf_register_fake_read( S_es,
                                    reference_data_element,
                                    this_function,
                                    dataflow, (*(odeps.begin()))->dst,
                                    &pseudotask_count );
        }

        jdf_register_output_deps( odeps, S_es,
                                  reference_data_element,
                                  this_function,
                                  dataflow,
                                  &pseudotask_count );
    }

    return;
}

void jdf_register_anti_dependency( dep_t *dep )
{
    static int nb_ctl_dep = 0;
    jdf_dataflow_t *dataflow;
    string_arena_t *sa;

    jdf_function_entry_t *src = dep->src->function;
    jdf_function_entry_t *dst = dep->dst->function;
    Relation             *rel = dep->rel;
    dep_t dep2;
    expr_t *expr;

    sa = string_arena_new(8);
    string_arena_add_string( sa, "ctl%d", nb_ctl_dep );
    nb_ctl_dep++;

    // Simple CTL
    dataflow = q2jmalloc(jdf_dataflow_t, 1);
    dataflow->next = NULL;
    dataflow->varname     = strdup(string_arena_get_string(sa));
    dataflow->deps        = q2jmalloc(jdf_dep_t, 1);
    dataflow->flow_flags = JDF_FLOW_TYPE_CTL;
    JDF_OBJECT_SET(dataflow, NULL, 0, NULL);

    dataflow->deps->next   = NULL;
    dataflow->deps->type   = JDF_DEP_FLOW_OUT;
    dataflow->deps->guard  = q2jmalloc(jdf_guarded_call_t, 1);
    jdf_set_default_datatype(&dataflow->deps->datatype, "DEFAULT", 1, 0);
    JDF_OBJECT_SET(dataflow->deps, NULL, 0, NULL);

    (void)(*dep->rel).print_with_subs_to_string(false);
    expr = relation_to_tree(*rel);
    dataflow->deps->guard->guard_type = JDF_GUARD_UNCONDITIONAL;
    dataflow->deps->guard->guard      = NULL;
    dataflow->deps->guard->properties = NULL;
    dataflow->deps->guard->callfalse  = NULL;
    dataflow->deps->guard->calltrue = q2jmalloc(jdf_call_t, 1);
    dataflow->deps->guard->calltrue->var         = strdup(string_arena_get_string(sa));
    dataflow->deps->guard->calltrue->func_or_mem = dst->fname;
    dataflow->deps->guard->calltrue->parameters  = jdf_generate_call_parameters( dep, expr );
    clean_tree(expr);

    dataflow->next = src->dataflow;
    src->dataflow  = dataflow;

    // Gather
    dataflow = q2jmalloc(jdf_dataflow_t, 1);
    dataflow->next = NULL;
    dataflow->varname     = strdup(string_arena_get_string(sa));
    dataflow->deps        = q2jmalloc(jdf_dep_t, 1);
    dataflow->flow_flags = JDF_FLOW_TYPE_CTL;
    JDF_OBJECT_SET(dataflow, NULL, 0, NULL);

    dataflow->deps->next   = NULL;
    dataflow->deps->type   = JDF_DEP_FLOW_IN;
    dataflow->deps->guard  = q2jmalloc(jdf_guarded_call_t, 1);
    jdf_set_default_datatype(&dataflow->deps->datatype, "DEFAULT", 1, 0);
    JDF_OBJECT_SET(dataflow->deps, NULL, 0, NULL);

    dataflow->deps->guard->guard_type = JDF_GUARD_UNCONDITIONAL;
    dataflow->deps->guard->guard      = NULL;
    dataflow->deps->guard->properties = NULL;
    dataflow->deps->guard->callfalse  = NULL;
    dataflow->deps->guard->calltrue = q2jmalloc(jdf_call_t, 1);
    dataflow->deps->guard->calltrue->var         = strdup(string_arena_get_string(sa));
    dataflow->deps->guard->calltrue->func_or_mem = src->fname;

    // Reverse the relation
    Relation inv = *dep->rel;
    dep2.src = dep->src;
    dep2.dst = dep->dst;
    dep2.rel = new Relation( Inverse(inv) );

    (void)(*dep2.rel).print_with_subs_to_string(false);
    expr = relation_to_tree( *dep2.rel );
    dataflow->deps->guard->calltrue->parameters  = jdf_generate_call_parameters( &dep2, expr );
    clean_tree(expr);

#ifdef DEBUG
    {
        std::cerr << "Anti-dependency: " << src->fname << " => " << dst->fname << " " << (*dep->rel).print_with_subs_to_string();
        std::cerr << "                 " << dst->fname << " => " << src->fname << " " << (*dep2.rel).print_with_subs_to_string();
    }
#endif

    dataflow->next = dst->dataflow;
    dst->dataflow  = dataflow;
}

void jdf_register_anti_dependencies( jdf_function_entry_t *this_function,
                                     map<char *, set<dep_t *> > synch_edges )
{
    map<char *, set<dep_t *> >::iterator synch_edge_it;

    for( synch_edge_it = synch_edges.begin(); synch_edge_it!= synch_edges.end(); ++synch_edge_it){
        if( strcmp( synch_edge_it->first, this_function->fname ) )
            continue;
        set<dep_t *> synch_dep_set = synch_edge_it->second;
        set<dep_t *>::iterator synch_dep_it;

        // Traverse all the entries of the set stored in synch_edges[ this task's name ] and print them
        for(synch_dep_it=synch_dep_set.begin(); synch_dep_it!=synch_dep_set.end(); ++synch_dep_it){
            assert(((*synch_dep_it)->src)->function == this_function);

            jdf_register_anti_dependency( (*synch_dep_it) );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
void jdf_register_body(jdf_function_entry_t *this_function,
                       node_t *task_node)
{
    this_function->body = strdup( tree_to_body(task_node) );
}


void jdf_register_function(jdf_function_entry_t       *this_function,
                           node_t                     *this_node,
                           node_t                     *reference_data_element,
                           Relation                    S_es,
                           set<char *>                &vars,
                           map<char *, set<dep_t *> > &outg_map,
                           map<char *, set<dep_t *> > &incm_map,
                           map<char *, set<dep_t *> > &synch_edges)
{
    jdf_register_locals( this_function, S_es );
    this_function->predicate = jdf_generate_call_for_data(reference_data_element, NULL);

    jdf_register_dependencies_and_pseudotasks(this_function,
                                              reference_data_element, S_es,
                                              vars, outg_map, incm_map);

    if( NULL != this_function->fname )
        jdf_register_anti_dependencies( this_function, synch_edges );

    jdf_register_body(this_function, this_node);
}
