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
list<char *> print_edges_and_create_pseudotasks(set<dep_t *>outg_deps, set<dep_t *>incm_edges, Relation S, node_t *reference_data_element);
void print_antidependencies( jdf_function_entry_t *this_function,
                             map<char *, set<dep_t *> > synch_edges );
void print_body(node_t *task_node);

#endif
