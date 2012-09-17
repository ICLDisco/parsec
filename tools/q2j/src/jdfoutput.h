/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _JDFOUTPUT_H_
#define _JDFOUTPUT_H_

void jdfoutput(const char *format, ...);    

void print_header();
void print_types_of_formal_parameters(node_t *root);
void print_default_task_placement(node_t *task_node);
void print_execution_space( Relation S_es );
void print_pseudo_variables(set<dep_t *>out_deps, set<dep_t *>in_deps);
list<char *> print_edges_and_create_pseudotasks(node_t *this_node,
                                                node_t *reference_data_element,
                                                Relation S_es,
                                                set<char *>                &vars,
                                                map<char *, set<dep_t *> > &outg_map,
                                                map<char *, set<dep_t *> > &incm_map);
void print_antidependencies( jdf_function_entry_t *this_function,
                             map<char *, set<dep_t *> > synch_edges );
void print_body(node_t *task_node);

void print_function(jdf_function_entry_t       *this_function,
                    task_t                     *this_task,
                    node_t                     *reference_data_element,
                    Relation                    S_es,
                    set<char *>                &vars,
                    map<char *, set<dep_t *> > &outg_map,
                    map<char *, set<dep_t *> > &incm_map,
                    map<char *, set<dep_t *> > &synch_edges);

#endif
