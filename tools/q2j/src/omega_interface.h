/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _OMEGA_INTERFACE_
#define _OMEGA_INTERFACE_

#include "dague_config.h"

#if defined(c_plusplus) || defined(__cplusplus)
list< pair<expr_t *, Relation> > simplify_conditions_and_split_disjunctions(Relation R, Relation S_es);
#endif  /* defined(c_plusplus) || defined(__cplusplus) */

BEGIN_C_DECLS

#if defined(Already_Included_Omega)

struct _dep_t{
    node_t *src;
    node_t *dst;
    Relation *rel;
};

expr_t *copy_tree(expr_t *root);
const char *expr_tree_to_str(expr_t *exp);
expr_t *relation_to_tree( Relation R );
void clean_tree(expr_t *root);
expr_t *solveExpressionTreeForVar(expr_t *exp, const char *var_name, Relation R);
const char *find_bounds_of_var(expr_t *exp, const char *var_name, set<const char *> vars_in_bounds, Relation R);
bool need_pseudotask(node_t *ref1, node_t *ref2);

char *dump_data(string_arena_t *sa, node_t *n);
char *dump_actual_parameters(string_arena_t *sa, dep_t *dep, expr_t *rel_exp);
char *dump_conditions(string_arena_t *sa,
                      list< pair<expr_t *,Relation> > *cond_list,
                      list< pair<expr_t *, Relation> >::iterator *cond_it);
#endif

void interrogate_omega(node_t *node, var_t *head);
void add_colocated_data_info(char *a, char *b);
void store_global_invariant(node_t *node);

END_C_DECLS

#endif
