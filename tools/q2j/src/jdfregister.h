/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _JDFREGISTER_H_
#define _JDFREGISTER_H_

jdf_call_t*
jdf_generate_call_for_data(node_t *data,
                           str_pair_t *subs);

void jdf_register_prologue(jdf_t *jdf);
void jdf_register_globals( jdf_t *jdf, node_t *root);
void jdf_register_epilogue(jdf_t *jdf);

void jdf_register_definitions(jdf_function_entry_t *function,
                              Relation S_es);

jdf_call_t *jdf_register_pseudotask(jdf_t *jdf,
                                    jdf_function_entry_t *parent_task,
                                    Relation S_es, Relation cond,
                                    node_t *data_element,
                                    char *var_pseudoname, 
                                    int ptask_count, const char *inout );

void jdf_register_dependencies_and_pseudotasks(jdf_function_entry_t       *this_function,
                                               node_t                     *reference_data_element,
                                               Relation                    S_es,
                                               set<char *>                &vars,
                                               map<char *, set<dep_t *> > &outg_map,
                                               map<char *, set<dep_t *> > &incm_map);

void jdf_register_anti_dependency( dep_t *dep );

void jdf_register_anti_dependencies( jdf_function_entry_t *this_function,
                                     map<char *, set<dep_t *> > synch_edges );

void jdf_register_body(jdf_function_entry_t *this_function,
                       node_t *task_node);

void jdf_register_function(jdf_function_entry_t       *this_function,
                           node_t                     *this_node,
                           node_t                     *reference_data_element,
                           Relation                    S_es,
                           set<char *>                &vars,
                           map<char *, set<dep_t *> > &outg_map,
                           map<char *, set<dep_t *> > &incm_map,
                           map<char *, set<dep_t *> > &synch_edges);

#endif /* _JDFREGISTER_H_ */
