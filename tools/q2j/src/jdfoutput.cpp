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
#include "jdfoutput.h"
#include "omega_interface.h"

extern char *_q2j_data_prefix;
extern FILE *_q2j_output;
extern int _q2j_dump_mapping;

string_arena_t *create_pool_declarations();

////////////////////////////////////////////////////////////////////////////////
//
void print_header(void)
{
    jdfoutput("extern \"C\" %%{\n"
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
              "#include \"dplasma/lib/dplasmajdf.h\"\n"
              "\n"
              "%%}\n\n");
}

////////////////////////////////////////////////////////////////////////////////
//
void print_types_of_formal_parameters(node_t *root){
    symtab_t *scope;
    q2j_symbol_t *sym;

    scope = root->symtab;
    do{
        for(sym=scope->symbols; NULL!=sym; sym=sym->next){
            if( !strcmp(sym->var_type, "PLASMA_desc") ){
                jdfoutput("%s%-5s [type = \"dague_ddesc_t *\"]\n"
                          "desc%-5s [type = \"tiled_matrix_desc_t\" hidden=on default=\"*((tiled_matrix_desc_t*)%s%s)\" ]\n",
                          _q2j_data_prefix, sym->var_name,
                          sym->var_name,
                          _q2j_data_prefix, sym->var_name);
            } else {
                jdfoutput("%-9s [type = \"%s\"]\n",
                          sym->var_name, sym->var_type);
            }
        }
        scope = scope->parent;
    } while(NULL != scope);

    /*
     * Create pool declarations
     */
    string_arena_t *sa = create_pool_declarations();
    jdfoutput( "%s", string_arena_get_string( sa ) );
    string_arena_free( sa );
    return;
}

////////////////////////////////////////////////////////////////////////////////
//
void print_execution_space(Relation S)
{
    set<const char *> prev_vars;
    int i;

    // Print the execution space based on the bounds that exist in the relation.
    jdfoutput("  /* Execution space */\n");
    for(i=1; i<=S.n_set(); i++){
        const char *var_name = strdup(S.set_var(i)->char_name());
        expr_t *e = relation_to_tree(S);
        expr_t *solution = solve_expression_tree_for_var(e, var_name, S);

        jdfoutput("  %s = ", var_name);
        if( NULL != solution )
            jdfoutput("%s\n", expr_tree_to_str(solution));
        else
            jdfoutput("%s\n", find_bounds_of_var(e, var_name, prev_vars, S));
        prev_vars.insert(var_name);

        clean_tree(e);
        clean_tree(solution);
    }
    jdfoutput("\n");

    // Do some memory clean-up
    while(!prev_vars.empty()){
        free((void *)*prev_vars.begin());
        prev_vars.erase(prev_vars.begin());
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//
void print_default_task_placement(node_t *task_node)
{
    if(NULL != task_node){
        string_arena_t *sa = string_arena_new(16);
        jdfoutput("  : %s\n\n", dump_data(sa, task_node));
        string_arena_free(sa);
    }else{
        jdfoutput("  : BOGUS_AFFINITY()\n\n");
    }
}

////////////////////////////////////////////////////////////////////////////////
//
void print_pseudo_variables(set<dep_t *>out_deps, set<dep_t *>in_deps){
    set<dep_t *>::iterator it;
    map<string, string> pseudo_vars;

    // Create a mapping between pseudo_variables and actual variables

    // OUTGOING
    for (it=out_deps.begin(); it!=out_deps.end(); it++){
        dep_t *dep = *it;
        // WARNING: This is needed by Omega. If you remove it you get strange
        // assert() calls being triggered inside the Omega library.
        (void)(*dep->rel).print_with_subs_to_string(false);

        if( NULL != dep->src->task ){
            pseudo_vars[dep->src->var_symname] = tree_to_str(dep->src);
        }
        if( NULL != dep->dst ){
            pseudo_vars[dep->dst->var_symname] = tree_to_str(dep->dst);
        }
    }

    // INCOMING
    for (it=in_deps.begin(); it!=in_deps.end(); it++){
        dep_t *dep = *it;
        // WARNING: This is needed by Omega. If you remove it you get strange
        // assert() calls being triggered inside the Omega library.
        (void)(*dep->rel).print_with_subs_to_string(false);

        Q2J_ASSERT( NULL != dep->dst);
        pseudo_vars[dep->dst->var_symname] = tree_to_str(dep->dst);

        if( NULL != dep->src->task ){
            pseudo_vars[dep->src->var_symname] = tree_to_str(dep->src);
        }
    }

    // Dump the map.
    string_arena_t *sa = string_arena_new(16);
    map<string, string>::iterator pvit;
    for (pvit=pseudo_vars.begin(); pvit!=pseudo_vars.end(); pvit++){
        jdfoutput("  /* %s == %s */\n",
                  (pvit->first).c_str(),
                  (pvit->second).c_str() );
    }
    string_arena_free(sa);
}

////////////////////////////////////////////////////////////////////////////////
//
char *create_pseudotask(node_t *parent_task,
                               Relation S_es, Relation cond,
                               node_t *data_element,
                               char *var_pseudoname,
                               int ptask_count, const char *inout,
                               string_arena_t *sa_pseudotask_name,
                               string_arena_t *sa_pseudotask )
{
    int var_count, firstfp, firstpfp;
    char *mtrx_name;
    char *data_str;

    Relation newS_es;
    set <const char *> prev_vars;
    str_pair_t *solved_vars;
    string_arena_t *sa1, *sa2;
    string_arena_t *sa_exec_space;
    string_arena_t *sa_formal_param;
    string_arena_t *sa_parent_formal_param;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa_exec_space   = string_arena_new(64);
    sa_formal_param = string_arena_new(64);
    sa_parent_formal_param = string_arena_new(64);

    mtrx_name = tree_to_str(DA_array_base(data_element));

    if( !cond.is_null() ){
        newS_es = Intersection(copy(S_es), Domain(copy(cond)));
    }else{
        newS_es = copy(S_es);
    }

    // Find the maximum number of variable substitutions we might need and add one for the termination flag.
    JDF_COUNT_LIST_ENTRIES( parent_task->function->parameters, jdf_name_list_t, next, var_count);
    solved_vars = (str_pair_t *)calloc(var_count+1, sizeof(str_pair_t));
    var_count = 0;

    // We start with the parameters in order to discover which ones should be included.
    firstpfp = 1; firstfp = 1;
    for(int i=0; NULL != parent_task->task->ind_vars[i]; ++i){
        const char *var_name = parent_task->task->ind_vars[i];
        expr_t *e = relation_to_tree(newS_es);
        expr_t *solution = solve_expression_tree_for_var(e, var_name, copy(newS_es));
        // If there is a solution it means that this parameter has a fixed value and not a range.
        // That means that there is no point in including it as a parameter of the pseudo-task.
        if( NULL != solution ){
            const char *solution_str = expr_tree_to_str(solution);
            solved_vars[var_count].str1 = var_name;
            solved_vars[var_count].str2 = solution_str;
            var_count++;

            if( !firstpfp ) {
                string_arena_add_string(sa_parent_formal_param, ", %s",
                                        solution_str );
            } else {
                string_arena_add_string(sa_parent_formal_param, "%s",
                                        solution_str );
            }
        } else {
            string_arena_add_string( sa_exec_space, "  %s = %s\n",
                                     var_name, find_bounds_of_var(e, var_name,
                                                                  prev_vars, copy(newS_es)) );

            // the following code is for generating the string for the caller (the real task)
            if( !firstfp ) {
                string_arena_add_string(sa_formal_param, ", %s",
                                        var_name );
            } else {
                string_arena_add_string(sa_formal_param, "%s",
                                        var_name );
                firstfp = 0;
            }

            if( !firstpfp ) {
                string_arena_add_string(sa_parent_formal_param, ", %s",
                                        var_name );
            } else {
                string_arena_add_string(sa_parent_formal_param, "%s",
                                        var_name );
            }

            clean_tree(e);
            clean_tree(solution);
            prev_vars.insert(var_name);
        }
        firstpfp = 0;
    }

    // Delete the "previous variables" set, to clean up some memory
    while(!prev_vars.empty()){
        prev_vars.erase(prev_vars.begin());
    }

    // Now that we know which parameters define the execution space of this pseudo-task, we can print the pseudo-task
    // in the body of the real task, and complete the "header" of the pseudo-task string.
    //
    // create_pseudotask() was called by print_edges_and_create_pseudotasks(), so if we print here
    // the string will end up at the right place in the body of the real task.

    // Create pseudotask name
    string_arena_init(sa_pseudotask_name);
    string_arena_add_string( sa_pseudotask_name, "%s %s_%s_data_%s%d(%s)",
                             var_pseudoname,
                             parent_task->function->fname, inout, mtrx_name,
                             ptask_count,
                             string_arena_get_string( sa_formal_param ) );

    // Parent task name with variable name
    string_arena_init(sa1);
    string_arena_add_string( sa1, "%s %s(%s)",
                             var_pseudoname,
                             parent_task->function->fname,
                             string_arena_get_string( sa_parent_formal_param ) );

    // Data string
    string_arena_init(sa2);
    string_arena_add_string( sa2, "%s%s", _q2j_data_prefix,
                             tree_to_str_with_substitutions(data_element, solved_vars) );
    data_str = string_arena_get_string(sa2);

    // Pseudo Task
    int is_input = !strcmp(inout,"in");
    string_arena_init(sa_pseudotask);
    string_arena_add_string(sa_pseudotask,
                            "\n/*\n * Pseudo-task\n */"
                            "\n%s [profile = off]"
                            "\n  /* Execution Space */"
                            "\n%s"
                            "\n  /* Locality */"
                            "\n  :%s\n"
                            "\n  RW %s <- %s"
                            "\n     %s -> %s\n"
                            "\nBODY"
                            "\n{"
                            "\n    /* nothing */"
                            "\n}"
                            "\nEND\n",
                            string_arena_get_string(sa_pseudotask_name) + strlen(var_pseudoname)+1,
                            string_arena_get_string(sa_exec_space),
                            data_str,
                            var_pseudoname,
                            is_input ? data_str : string_arena_get_string(sa1),
                            indent(strlen(var_pseudoname), 1),
                            is_input ? string_arena_get_string(sa1) : data_str );

    free(mtrx_name);

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa_exec_space);
    string_arena_free(sa_formal_param);
    string_arena_free(sa_parent_formal_param);

    return string_arena_get_string(sa_pseudotask);
}

char *generate_condition_str(expr_t *exp){
    string cond;
    string_arena_t *sa;

    sa  = string_arena_new(64);
    string_arena_init(sa);

    cond = expr_tree_to_str(exp);
    if( !cond.empty() ){
        string_arena_add_string(sa, "%s ? ",cond.c_str());
    }

    return string_arena_get_string(sa);
}

////////////////////////////////////////////////////////////////////////////////
//
list<char *> print_edges_and_create_pseudotasks(node_t *this_node,
                                                node_t *reference_data_element,
                                                Relation S_es,
                                                set<char *>                &vars,
                                                map<char *, set<dep_t *> > &outg_map,
                                                map<char *, set<dep_t *> > &incm_map)
{
    int pseudotask_count = 0;
    list<char *> ptask_list;
    int nbspaces = 0;
    string_arena_t *sa, *sa2;
    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);

    if( outg_map.empty() && incm_map.empty() ){
        return ptask_list;
    }

    // For each variable print all the incoming and the outgoing edges
    set<char *>::iterator var_it;
    for (var_it=vars.begin(); var_it!=vars.end(); var_it++ ){
        set<dep_t *>::iterator dep_it;
        bool insert_fake_read = false;
        char *var_pseudoname = *var_it;
        set<dep_t *>ideps = incm_map[var_pseudoname];
        set<dep_t *>odeps = outg_map[var_pseudoname];
        int nb_ideps, nb_odeps, nb_deps;
        const char *access;

        nb_ideps = ideps.size();
        nb_odeps = odeps.size();
        nb_deps = nb_ideps + nb_odeps;

        if( nb_ideps > 0 && nb_odeps > 0 ){
            access = "RW";
        }else if( nb_ideps > 0 ){
            access = "READ";
        }else if( nb_odeps > 0 ){
            /*
             * DAGuE does not like write-only variables, so make it RW and make
             * it read from the data matrix tile that corresponds to this variable.
             */
            access = "RW";
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

        // Print the pseudoname
        jdfoutput("  %-5s %-4s ", access, var_pseudoname);
        nbspaces = 13;

        // print the incoming edges
        for (dep_it=ideps.begin(); dep_it!=ideps.end(); dep_it++){
             dep_t *dep = *dep_it;
             list< pair<expr_t *,Relation> > cond_list;
             list< pair<expr_t *, Relation> >::iterator cond_it;
             string cond;

             // Needed by Omega
             (void)(*dep->rel).print_with_subs_to_string(false);

             Q2J_ASSERT( NULL != dep->dst);

             // If the condition has disjunctions (logical OR operators) then split them so that each one
             // is treated independently.
             cond_list = simplify_conditions_and_split_disjunctions(*dep->rel, S_es);
             for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
                 node_t *src_node = dep->src;

                 if ( (dep_it!=ideps.begin()) || (cond_it != cond_list.begin()) )
                     jdfoutput("%s", indent(nbspaces, 1));

                 // Conditions for this input
                 jdfoutput("<- %s", generate_condition_str(cond_it->first) );

                 // Source of the input
                 string_arena_init(sa);
                 if( NULL != src_node->function ){
                     expr_t *e = relation_to_tree(cond_it->second);
                     string_arena_add_string(sa, "%s %s(%s)",
                                             src_node->var_symname,
                                             src_node->function->fname,
                                             dump_actual_parameters(sa2, dep, e) );
                     clean_tree(e);
                 }else{ // ENTRY
                     if( need_pseudotask(dep->dst, reference_data_element) ){
                         create_pseudotask(this_node,
                                           S_es, cond_it->second,
                                           dep->dst, var_pseudoname, pseudotask_count++,
                                           "in", sa, sa2 );
                         ptask_list.push_back( strdup(string_arena_get_string(sa2)) );
                     }else{
                         dump_data(sa, dep->src);
                     }
                 }
                 jdfoutput("%s\n", string_arena_get_string(sa) );
#ifdef DEBUG_2
                 jdfoutput_dbg("%s// %s -> %s ", indent(nbspaces, 1),
                               (NULL != src_node->function) ? src_node->function->fname, "ENTRY" );
                 (*dep->rel).print_with_subs(stdout);
#endif
             }

             for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
                 clean_tree( (*cond_it).first );
             }
        }

        if(insert_fake_read){
            Relation emptyR;
            dep_t *dep = *(odeps.begin());
            if( need_pseudotask(dep->src, reference_data_element) ){
                create_pseudotask(this_node,
                                  S_es, emptyR,
                                  dep->src, var_pseudoname, pseudotask_count++,
                                  "in", sa, sa2 );
                ptask_list.push_back( strdup(string_arena_get_string(sa2)) );
            }else{
                dump_data(sa, dep->dst);
            }
            jdfoutput("%s<- %s\n", indent(nbspaces, 1), string_arena_get_string(sa) );
        }

        // print the outgoing edges
        for (dep_it=odeps.begin(); dep_it!=odeps.end(); dep_it++){
             dep_t *dep = *dep_it;
             list< pair<expr_t *,Relation> > cond_list;
             list< pair<expr_t *,Relation> >::iterator cond_it;
             string cond;

             // Needed by Omega
             (void)(*dep->rel).print_with_subs_to_string(false);

             Q2J_ASSERT( NULL != dep->src->task );

             // If the condition has disjunctions (logical OR operators) then split them so that each one
             // is treated independently.
             cond_list = simplify_conditions_and_split_disjunctions(*dep->rel, S_es);
             for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){

                 // Conditions for this output
                 jdfoutput("%s-> %s", indent(nbspaces, 1), generate_condition_str(cond_it->first) );

                 // Destination of the output
                 node_t *sink = dep->dst;
                 string_arena_init(sa);
                 if( NULL != sink ){
                     expr_t *e = relation_to_tree(cond_it->second);
                     string_arena_add_string(sa, "%s %s(%s)",
                                             sink->var_symname, sink->function->fname,
                                             dump_actual_parameters(sa2, dep, e));
                     clean_tree(e);
                 } else { // EXIT
                     if( need_pseudotask(dep->src, reference_data_element) ){
                         create_pseudotask(this_node,
                                           S_es, cond_it->second,
                                           dep->src, var_pseudoname, pseudotask_count++,
                                           "out", sa, sa2 );
                         ptask_list.push_back( strdup(string_arena_get_string(sa2)) );
                     }else{
                         dump_data(sa, dep->src);
                     }
                 }
                 jdfoutput("%s\n", string_arena_get_string(sa));
#ifdef DEBUG_2
                 jdfoutput_dbg("%s// %s -> %s ", indent(nbspaces, 1),
                               tree_to_str(dep->src),
                               (NULL != sink ) ? sink->task->task_name : "EXIT" );
                 (*dep->rel).print_with_subs(stdout);
#endif
             }

             for(cond_it = cond_list.begin(); cond_it != cond_list.end(); cond_it++){
                 clean_tree( (*cond_it).first );
             }
        }
    }

    string_arena_free(sa);
    string_arena_free(sa2);
    return ptask_list;
}

void print_antidependencies( jdf_function_entry_t *this_function,
                             map<char *, set<dep_t *> > synch_edges )
{
    map<char *, set<dep_t *> >::iterator synch_edge_it;
    bool has_synch_edges = false;

    for( synch_edge_it = synch_edges.begin(); synch_edge_it!= synch_edges.end(); ++synch_edge_it){
        if( strcmp( synch_edge_it->first, this_function->fname ) )
            continue;

        set<dep_t *> synch_dep_set = synch_edge_it->second;
        set<dep_t *>::iterator synch_dep_it;

        if ( !has_synch_edges) {
            jdfoutput("  /*\n  Anti-dependencies:\n");
            has_synch_edges = true;
        }

        // Traverse all the entries of the set stored in synch_edges[ this task's name ] and print them
        for(synch_dep_it=synch_dep_set.begin(); synch_dep_it!=synch_dep_set.end(); ++synch_dep_it){
            string relation;
            node_t *use  = (*synch_dep_it)->src;
            assert(use->function == this_function);
            node_t *sink  = (*synch_dep_it)->dst;
            Relation ad_r = *((*synch_dep_it)->rel);
            char *n1 = use->function->fname;
            char *n2 = sink->function->fname;
            jdfoutput("  ANTI edge from %s:%s to %s:%s ", n1, tree_to_str(use), n2, tree_to_str(sink));
            relation = ad_r.print_with_subs_to_string();
            jdfoutput("%s", relation.c_str());
        }
    }

    if( has_synch_edges )
        jdfoutput("  */\n");
}

////////////////////////////////////////////////////////////////////////////////
//
void print_body(node_t *task_node)
{
    jdfoutput( "BODY\n"
               "{\n"
               "%s\n"
               "}\n"
               "END\n",
               tree_to_body(task_node));
}

////////////////////////////////////////////////////////////////////////////////
//
void print_function(jdf_function_entry_t       *this_function,
                    task_t                     *this_task,
                    node_t                     *reference_data_element,
                    Relation                    S_es,
                    set<char *>                &vars,
                    map<char *, set<dep_t *> > &outg_map,
                    map<char *, set<dep_t *> > &incm_map,
                    map<char *, set<dep_t *> > &synch_edges)
{
    list <char *>::iterator ptask_it;
    list <char *>ptask_list;
    int i;
    char *task_name;

    task_name = this_function->fname;

    jdfoutput("\n\n%s( ",task_name);
    for(i=0; NULL != this_task->ind_vars[i]; ++i){
        if( i ) jdfoutput(", ");
        jdfoutput("%s", this_task->ind_vars[i]);
    }
    jdfoutput(")\n");

    print_execution_space(S_es);
    print_default_task_placement(reference_data_element);

//     if( _q2j_dump_mapping ) {
//         print_pseudo_variables(outg_deps, incm_deps);
//     }

    ptask_list = print_edges_and_create_pseudotasks( this_task->task_node,
                                                     reference_data_element,
                                                     S_es,
                                                     vars, outg_map, incm_map);

    if( NULL != this_function->fname )
        print_antidependencies( this_function, synch_edges );

    print_body(this_task->task_node);

    // Print all the pseudo-tasks that were created by print_edges_and_create_pseudotasks()
    for(ptask_it=ptask_list.begin(); ptask_it!=ptask_list.end(); ++ptask_it){
        jdfoutput("%s\n",*ptask_it);
    }
}
