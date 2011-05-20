/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dague_config.h"
#include "node_struct.h"
#include "utility.h"
#include "q2j.y.h"
#include "omega_interface.h"
#include "omega.h"
#include <map>
#include <set>
#include <list>
#include <sstream>

map<string, string> q2j_colocated_map;

struct _dep_t{
    node_t *src;
    node_t *dst;
    Relation *rel;
};

typedef struct synch_edge_graph_t synch_edge_graph_t;
typedef struct seg_node_t{
    bool visited;
    list<synch_edge_graph_t *> edges;
    list<Relation *> cycles;
} seg_node_t;

struct synch_edge_graph_t{
    seg_node_t *destination;
    Relation *rel;
    int rel_type;
};

#define DEP_FLOW  0x1
#define DEP_OUT   0x2
#define DEP_ANTI  0x4

#define EDGE_INCOMING 0x0
#define EDGE_OUTGOING 0x1

#define LBOUND  0x0
#define UBOUND  0x1

#define SOURCE  0x0
#define SINK    0x1

#if 0
extern void dump_und(und_t *und);
static void dump_full_und(und_t *und);
#endif

static void process_end_condition(node_t *node, F_And *&R_root, map<string, Variable_ID> ivars, Relation &R);
static void print_edges(set<dep_t *>outg_deps, set<dep_t *>incm_edges, Relation S, node_t *reference_data_element);
static void print_pseudo_variables(set<dep_t *>out_deps, set<dep_t *>in_deps);
static Relation process_and_print_execution_space(node_t *node);
static inline set<expr_t *> findAllEQsWithVar(const char *var_name, expr_t *exp);
static inline set<expr_t *> findAllGEsWithVar(const char *var_name, expr_t *exp);
static set<expr_t *> findAllConstraintsWithVar(const char *var_name, expr_t *exp, int constr_type);
static const char *expr_tree_to_str(const expr_t *exp);
static string _expr_tree_to_str(const expr_t *exp);
static int treeContainsVar(expr_t *root, const char *var_name);
static expr_t *solveDirectlySolvableEQ(expr_t *exp, const char *var_name, Relation R);
static void substituteExprForVar(expr_t *exp, const char *var_name, expr_t *root);
static map<string, Free_Var_Decl *> global_vars;

#if 0
void dump_all_uses(und_t *def, var_t *head){
    int after_def = 0;
    var_t *var;
    und_t *und;

    for(var=head; NULL != var; var=var->next){
        for(und=var->und; NULL != und ; und=und->next){
            char *var_name = DA_var_name(DA_array_base(und->node));
            char *def_name = DA_var_name(DA_array_base(def->node));
            if( und == def ){
                after_def = 1;
            }
            if( is_und_read(und) && (0==strcmp(var_name, def_name)) ){
                printf("   ");
                if( after_def )
                    printf(" +%d ",und->task_num);
                else
                    printf(" -%d ",und->task_num);
                dump_full_und(und);
            }
        }
    }
}

void dump_full_und(und_t *und){
    node_t *tmp;

    printf("%d ",und->task_num);
    dump_und(und);
    printf(" ");
    for(tmp=und->node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        printf("%s:", DA_var_name(DA_loop_induction_variable(tmp)) );
        printf("{ %s, ", tree_to_str(DA_loop_lb(tmp)) );
        printf(" %s }", tree_to_str(DA_loop_ub(tmp)) );
        if( NULL != tmp->enclosing_loop )
            printf(",  ");
    }
    printf("\n");
}

void dump_UorD(node_t *node){
    node_t *tmp;
    list<char *> ind;
    printf("%s {",tree_to_str(node));


    for(tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        ind.push_front( DA_var_name(DA_loop_induction_variable(tmp)) );
    }
    for (list<char *>::iterator it=ind.begin(); it!=ind.end(); ++it){
        if( it != ind.begin() )
            printf(",");
        printf("%s",*it);
    }

    printf("}");
}
#endif

void declare_globals_in_tree(node_t *node, set <char *> ind_names){
    char *var_name = NULL;

    switch( node->type ){
        case S_U_MEMBER:
            var_name = tree_to_str(node);
        case IDENTIFIER:
            if( NULL == var_name ){
                var_name = DA_var_name(node);
            }
            // If the var is one of the induction variables, do nothing
            set <char *>::iterator ind_it = ind_names.begin();
            for(; ind_it != ind_names.end(); ++ind_it){
                char *tmp_var = *ind_it;
                if( 0==strcmp(var_name, tmp_var) ){
                    return;
                }
            }
       
            // If the var is not already declared as a global symbol, declare it
            map<string, Free_Var_Decl *>::iterator it;
            it = global_vars.find(var_name);
            if( it == global_vars.end() ){
                global_vars[var_name] = new Free_Var_Decl(var_name);
            }

            return;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            declare_globals_in_tree(tmp, ind_names);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            declare_globals_in_tree(node->u.kids.kids[i], ind_names);
        }
    }

    return;
}



// the parameter "R" (of type "Relation") needs to be passed by reference, otherwise a copy
// is passed every time the recursion goes deeper and the "handle" does not correspond to R.
void expr_to_Omega_coef(node_t *node, Constraint_Handle &handle, int sign, map<string, Variable_ID> vars, Relation &R){
    map<string, Variable_ID>::iterator v_it;
    map<string, Free_Var_Decl *>::iterator g_it;
    char *var_name=NULL;

    switch(node->type){
        case INTCONSTANT:
            handle.update_const(sign*DA_int_val(node));
            break;
        case S_U_MEMBER:
            var_name = tree_to_str(node);
        case IDENTIFIER:
            // look for the ID in the induction variables
            if( NULL==var_name ){
                var_name = DA_var_name(node);
            }
            v_it = vars.find(var_name);
            if( v_it != vars.end() ){
                Variable_ID v = v_it->second;
                handle.update_coef(v,sign);
                break;
            }
            // If not found yet, look for the ID in the global variables
            g_it = global_vars.find(var_name);
            if( g_it != global_vars.end() ){
                Variable_ID v = R.get_local(g_it->second);
                handle.update_coef(v,sign);
                break;
            }
            fprintf(stderr,"expr_to_Omega_coef(): Can't find \"%s\" in either induction, or global variables.\n", DA_var_name(node) );
            exit(-1);
        case ADD:
            expr_to_Omega_coef(node->u.kids.kids[0], handle, sign, vars, R);
            expr_to_Omega_coef(node->u.kids.kids[1], handle, sign, vars, R);
            break;
        case SUB:
            expr_to_Omega_coef(node->u.kids.kids[0], handle, sign, vars, R);
            expr_to_Omega_coef(node->u.kids.kids[1], handle, -sign, vars, R);
            break;
        default:
            fprintf(stderr,"expr_to_Omega_coef(): Can't turn type \"%d (%x)\" into Omega expression.\n",node->type, node->type);
            exit(-1);
    }
    return;
}


#if 0
/*
 * A path can exist from src to dst iff both following conditions are true:
 * 1) src & dst have at least one common enclosing block
 * 2) src & dst are either:
 *    2a) not in different branches of an if-then-else condition
 *    2b) are in different branches of an if-then-else such that both following conditions hold:
 *        i)  the if-then-else is enclosed by at least one loop
 *        ii) the if-then-else condition is not loop invariant w.r.t. the enclosing loop
 */

bool path_can_exist_from_src_to_dst(node_t *src, node_t *dst){
    bool common_parent_found = false;
    node_t *prnt = NULL;
    printf("src: %s\n",tree_to_str(src));
    printf("dst: %s\n",tree_to_str(dst));

    for(prnt=src; prnt!=NULL; prnt=prnt->parent){
        node_t *tmp;
        for(tmp=dst; tmp!=NULL; tmp=tmp->parent){
            if( tmp == prnt ){
                common_parent_found = true;
                break;
            }
        }
        if( common_parent_found )
            break;
    }
    if( !common_parent_found ){
        printf("Source and Destination (%s and %s) do not share a common parent.\n",tree_to_str(src), tree_to_str(dst));
        return false;
    }

    printf("Common parent found: %s\n", DA_type_name(prnt) );
        
    // Check to see if "src" is inside an if-then-else
    for(node_t *tmp=src; tmp!=NULL; tmp=tmp->parent){
        if( IF == tmp->parent ){
            // Since "src" is in an if-then-else, see whether "dst" is inside the same if-then-else
            for(node_t *tmp2=dst; tmp2!=NULL; tmp2=tmp2->parent){
                if( tmp->parent == tmp2->parent ){
                    // Since "dst" is in the same if-then-else as "src" see if they are in the same branch
                    if( tmp2 == tmp ){
                        continue;
                    }else{
                        // Since "src" and "dst" are in different branches of an if-then-else, for a path to exist:
                        // a) the if-then-else must be inside a loop AND
                        // b) it must be loop dependent
#error "HERE"
                    }
                }
            }
        }
        printf("%s ", DA_type_name(tmp) );
    }
    printf("\n");

    printf("dst (%s) ancestory: ",tree_to_str(dst));
    for(node_t *tmp=dst; tmp!=NULL; tmp=tmp->parent){
        printf("%s ", DA_type_name(tmp) );
    }
    printf("\n");

    return true;
}
#endif

/*
 * Find the first loop that encloses both arguments
 */
node_t *find_closest_enclosing_loop(node_t *n1, node_t *n2){
    node_t *tmp1, *tmp2;

    for(tmp1=n1->enclosing_loop; NULL != tmp1; tmp1=tmp1->enclosing_loop ){
        for(tmp2=n2->enclosing_loop; NULL != tmp2; tmp2=tmp2->enclosing_loop ){
            if(tmp1 == tmp2)
                return tmp1;
        }
    }
    return NULL;
}

void add_array_subscript_equalities(Relation &R, F_And *R_root, map<string, Variable_ID> ivars, map<string, Variable_ID> ovars, node_t *def, node_t *use){
    int i,count;

    count = DA_array_dim_count(def);
    if( DA_array_dim_count(use) != count ){
        fprintf(stderr,"add_array_subscript_equalities(): ERROR: Arrays in USE and DEF do not have the same number of subscripts.");
    }

    for(i=0; i<count; i++){
        node_t *iv = DA_array_index(def, i);
        node_t *ov = DA_array_index(use, i);
        EQ_Handle hndl = R_root->add_EQ();
        expr_to_Omega_coef(iv, hndl, 1, ivars, R);
        expr_to_Omega_coef(ov, hndl, -1, ovars, R);
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
// This function assumes that the end condition of the for() loop conforms to the
// regular expression:
// k (<|<=) E0 ( && k (<|<=) Ei )*
// where "k" is the induction variable and "Ei" are expressions of induction
// variables, global variables and constants.
//
void process_end_condition(node_t *node, F_And *&R_root, map<string, Variable_ID> ivars, Relation &R){
    Variable_ID ivar;
    GEQ_Handle imax;
    F_And *new_and;

    switch( node->type ){
        case L_AND:
            new_and = R_root->add_and();
            process_end_condition(node->u.kids.kids[0], new_and, ivars, R);
            process_end_condition(node->u.kids.kids[1], new_and, ivars, R);
            break;
// TODO: handle logical or (L_OR) as well.
//F_Or *or1 = R_root->add_or();
//F_And *and1 = or1->add_and();
        case LT:
            ivar = ivars[DA_var_name(DA_rel_lhs(node))];
            imax = R_root->add_GEQ();
            expr_to_Omega_coef(DA_rel_rhs(node), imax, 1, ivars, R);
            imax.update_coef(ivar,-1);
            imax.update_const(-1);
            break;
        case LE:
            ivar = ivars[DA_var_name(DA_rel_lhs(node))];
            imax = R_root->add_GEQ();
            expr_to_Omega_coef(DA_rel_rhs(node), imax, 1, ivars, R);
            imax.update_coef(ivar,-1);
            break;
        default:
            fprintf(stderr,"ERROR: process_end_condition() cannot deal with node of type: %s\n", DA_type_name(node) );
            exit(-1);
    }

}

Relation create_exit_relation(node_t *exit, node_t *def){
    int i, src_var_count, dst_var_count;
    node_t *tmp, *use;
    char **def_ind_names;
    map<string, Variable_ID> ivars;

    use = exit;

    src_var_count = def->loop_depth;
    dst_var_count = DA_array_dim_count(def);

    Relation R(src_var_count, dst_var_count);

    // Store the names of the induction variables of the loops that enclose the use
    // and make the last item NULL so we know where the array terminates.
    def_ind_names = (char **)calloc(src_var_count+1, sizeof(char *));
    def_ind_names[src_var_count] = NULL;
    i=src_var_count-1;
    for(tmp=def->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        def_ind_names[i] = DA_var_name(DA_loop_induction_variable(tmp));
        --i;
    }

    // Name the input variables using the induction variables of the loops that enclose the def.
    for(i=0; i<src_var_count; ++i){
        R.name_input_var(i+1, def_ind_names[i]);
        ivars[def_ind_names[i]] = R.input_var(i+1);
    }

    // Name the output variables using temporary names "Var_i"
    for(i=0; i<dst_var_count; ++i){
        stringstream ss;
        ss << "Var_" << i;

        R.name_output_var(i+1, ss.str().c_str() );
    }

    F_And *R_root = R.add_and();

    // Bound all induction variables of the loops enclosing the DEF
    // using the loop bounds
    for(tmp=def->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        // Form the Omega expression for the lower bound
        Variable_ID ivar = ivars[DA_var_name(DA_loop_induction_variable(tmp))];

        GEQ_Handle imin = R_root->add_GEQ();
        imin.update_coef(ivar,1);
        expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ivars, R);

        // Form the Omega expression for the upper bound
        process_end_condition(DA_for_econd(tmp), R_root, ivars, R);
    }

    // Add equalities between corresponding input and output array indexes
    int count = DA_array_dim_count(def);
    for(i=0; i<count; i++){
        node_t *iv = DA_array_index(def, i);
        EQ_Handle hndl = R_root->add_EQ();

        hndl.update_coef(R.output_var(i+1), 1);
        expr_to_Omega_coef(iv, hndl, -1, ivars, R);
    }

    R.simplify();
    return R;
}

map<node_t *, Relation> create_entry_relations(node_t *entry, var_t *var, int dep_type){
    int i, src_var_count, dst_var_count;
    und_t *und;
    node_t *tmp, *def, *use;
//    char **def_ind_names;
    char **use_ind_names;
    map<string, Variable_ID> ivars, ovars;
    map<node_t *, Relation> dep_edges;

    def = entry;
    for(und=var->und; NULL != und ; und=und->next){

        if( ((DEP_FLOW==dep_type) && (!is_und_read(und))) || ((DEP_OUT==dep_type) && (!is_und_write(und))) ){
            continue;
        }

        use = und->node;

        dst_var_count = use->loop_depth;
        // we'll make the source match the destination, whatever that is
        src_var_count = DA_array_dim_count(use);

        Relation R(src_var_count, dst_var_count);

        // Store the names of the induction variables of the loops that enclose the use
        // and make the last item NULL so we know where the array terminates.
        use_ind_names = (char **)calloc(dst_var_count+1, sizeof(char *));
        use_ind_names[dst_var_count] = NULL;
        i=dst_var_count-1;
        for(tmp=use->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            use_ind_names[i] = DA_var_name(DA_loop_induction_variable(tmp));
            --i;
        }

        // Name the input variables using temporary names "Var_i"
        // The input variables will remain unnamed, but they should disappear because of the equalities
        for(i=0; i<src_var_count; ++i){
            stringstream ss;
            ss << "Var_" << i;

            R.name_input_var(i+1, ss.str().c_str() );
        }

        // Name the output variables using the induction variables of the loops that enclose the use.
        for(i=0; i<dst_var_count; ++i){
            R.name_output_var(i+1, use_ind_names[i]);
            ovars[use_ind_names[i]] = R.output_var(i+1);
        }

        F_And *R_root = R.add_and();

        // Bound all induction variables of the loops enclosing the USE
        // using the loop bounds
        for(tmp=use->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            // Form the Omega expression for the lower bound
            Variable_ID ovar = ovars[DA_var_name(DA_loop_induction_variable(tmp))];

            GEQ_Handle imin = R_root->add_GEQ();
            imin.update_coef(ovar,1);
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ovars, R);

            // Form the Omega expression for the upper bound
//printf("Form UB: %s\n",tree_to_str(DA_for_econd(tmp)) );
            process_end_condition(DA_for_econd(tmp), R_root, ovars, R);
        }

        // Add equalities between corresponding input and output array indexes
        int count = DA_array_dim_count(use);
        for(i=0; i<count; i++){
            node_t *ov = DA_array_index(use, i);
            EQ_Handle hndl = R_root->add_EQ();

            hndl.update_coef(R.input_var(i+1), 1);
            expr_to_Omega_coef(ov, hndl, -1, ovars, R);
        }

        R.simplify();
        if( R.is_upper_bound_satisfiable() || R.is_lower_bound_satisfiable() ){
            dep_edges[use] = R;
        }

    }

    return dep_edges;
}




////////////////////////////////////////////////////////////////////////////////
//
// The variable names and comments in this function abuse the terms "def" and "use".
// It would be more acurate to use the terms "source" and "destination" for the edges.
map<node_t *, Relation> create_dep_relations(und_t *def_und, var_t *var, int dep_type, node_t *exit_node){
    int i, after_def = 0;
    int src_var_count, dst_var_count;
    und_t *und;
    node_t *tmp, *def, *use;
    char **def_ind_names;
    char **use_ind_names;
    map<string, Variable_ID> ivars, ovars;
    map<node_t *, Relation> dep_edges;

    // In the case of anti-dependencies (write after read) "def" is really a USE.
    def = def_und->node;
    src_var_count = def->loop_depth;
    after_def = 0;
    for(und=var->und; NULL != und ; und=und->next){
        char *var_name, *def_name;

        var_name = DA_var_name(DA_array_base(und->node));
        def_name = DA_var_name(DA_array_base(def));
        assert( !strcmp(var_name, def_name) );

        if( ((DEP_FLOW==dep_type) && (!is_und_read(und))) || ((DEP_OUT==dep_type) && (!is_und_write(und))) || ((DEP_ANTI==dep_type) && (!is_und_write(und))) ){
            // Since we'll bail, let's first check if this is the definition.
            if( und == def_und ){
                after_def = 1;
            }
            continue;
        }

/*
// IF THEN ELSE HERE Here here
        // Experimental code to handle if-then-else
        if( !path_can_exist_from_src_to_dst(def, und->node) )
            continue;
*/

        // In the case of output dependencies (write after write) "use" is really a DEF.
        use = und->node;

        dst_var_count = use->loop_depth;
        Relation R(src_var_count, dst_var_count);

        // Store the names of the induction variables of the loops that enclose the definition
        // and make the last item NULL so we know where the array terminates.
        def_ind_names = (char **)calloc(src_var_count+1, sizeof(char *));
        def_ind_names[src_var_count] = NULL;
        i=src_var_count-1;
        for(tmp=def->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            def_ind_names[i] = DA_var_name(DA_loop_induction_variable(tmp));
            --i;
        }

        // Store the names of the induction variables of the loops that enclose the use
        // and make the last item NULL so we know where the array terminates.
        use_ind_names = (char **)calloc(dst_var_count+1, sizeof(char *));
        use_ind_names[dst_var_count] = NULL;
        i=dst_var_count-1;
        for(tmp=use->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            use_ind_names[i] = DA_var_name(DA_loop_induction_variable(tmp));
            --i;
        }

        // Name the input variables using the induction variables of the loops
        // that enclose the definition
        for(i=0; i<src_var_count; ++i){
            R.name_input_var(i+1, def_ind_names[i]);
             // put it in a map so we can find it later
            ivars[def_ind_names[i]] = R.input_var(i+1);
        }

        // Name the output variables using the induction variables of the loops
        // that enclose the use
        for(i=0; i<dst_var_count; ++i){
            R.name_output_var(i+1, use_ind_names[i]);
            // put it in a map so we can find it later
            ovars[use_ind_names[i]] = R.output_var(i+1);
        }

        F_And *R_root = R.add_and();

        // Bound all induction variables of the loops enclosing the DEF
        // using the loop bounds
        for(tmp=def->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            // Form the Omega expression for the lower bound
            Variable_ID ivar = ivars[DA_var_name(DA_loop_induction_variable(tmp))];

            GEQ_Handle imin = R_root->add_GEQ();
            imin.update_coef(ivar,1);
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ivars, R);

            // Form the Omega expression for the upper bound
            process_end_condition(DA_for_econd(tmp), R_root, ivars, R);
        }

        // Bound all induction variables of the loops enclosing the USE
        // using the loop bounds
        for(tmp=use->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            // Form the Omega expression for the lower bound
            Variable_ID ovar = ovars[DA_var_name(DA_loop_induction_variable(tmp))];

            GEQ_Handle imin = R_root->add_GEQ();
            imin.update_coef(ovar,1);
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ovars, R);

            // Form the Omega expression for the upper bound
            process_end_condition(DA_for_econd(tmp), R_root, ovars, R);
        }


        // Add inequalities of the form (m'>=m || n'>=n || ...) or the form (m'>m || n'>n || ...) if the DU chain is
        // "normal", or loop carried, respectively. The outermost enclosing loop HAS to be k'>=k, it is not
        // part of the "or" conditions.  In the loop carried deps, the outer most loop is ALSO in the "or" conditions,
        // so we have (m'>m || n'>n || k'>k) && k'>=k
        // In addition, if a flow edge is going from a task to itself (because the def and use seemed to be in different
        // lines) it also needs to have greater-than instead of greater-or-equal relationships for the induction variables.

        node_t *encl_loop = find_closest_enclosing_loop(use, def);

        // If USE and DEF are in the same task and that is not in a loop, there is no data flow, go to the next USE.
        // However, if we are looking at anti-dependencies and "def == use" (not the tasks, the actual nodes) then
        // we should keep this contrived edge because we will need to subtract it from all other anti-dep edges.
        if( (NULL == encl_loop) && (def->task == use->task) && ((DEP_ANTI!=dep_type) || (def != use)) ){
            continue;
        }
        
#warning "The following conjunctions are suboptimal. Look at the SC11 submission for better ones."

        // If we are recording anti-dependencies then we have to record an INOUT array as an antidepepdency. We will later
        // subtract the relation of that "self" antidependency from the relations of all other anti-dependencies.
        if( ( after_def && (def->task != use->task) ) || ( (DEP_ANTI==dep_type) && (after_def||(def==use)) ) ){
            // Create Relation due to "normal" DU chain.  In this case, not having a common enclosing loop it acceptable.
            if( NULL != encl_loop ){

                for(tmp=encl_loop; NULL != tmp->enclosing_loop; tmp=tmp->enclosing_loop );
                char *var_name = DA_var_name(DA_loop_induction_variable(tmp));
                Variable_ID ovar = ovars[var_name];
                Variable_ID ivar = ivars[var_name];
                GEQ_Handle ge = R_root->add_GEQ();
                ge.update_coef(ovar,1);
                ge.update_coef(ivar,-1);

                if( tmp != encl_loop ){
                    F_Or *or1 = R_root->add_or();
                    for(tmp=encl_loop; NULL != tmp->enclosing_loop; tmp=tmp->enclosing_loop ){
                        F_And *and1 = or1->add_and();

                        // Force at least one of the input variables (that correspond to the DEF)
                        // to be less or equal to the output variables.
                        char *var_name = DA_var_name(DA_loop_induction_variable(tmp));
                        Variable_ID ovar = ovars[var_name];
                        Variable_ID ivar = ivars[var_name];
                        GEQ_Handle ge = and1->add_GEQ();
                        ge.update_coef(ovar,1);
                        ge.update_coef(ivar,-1);
                    }
                }
            }
        }else{
            // Create Relation due to loop carried DU chain

            if( NULL == encl_loop ){
                fprintf(stderr,"create_dep_relations(): USE is before DEF and they do not have a common enclosing loop\n");
                fprintf(stderr,"USE:%s %s\n", tree_to_str(use), use->task->task_name );
                fprintf(stderr,"DEF:%s %s\n", tree_to_str(def), def->task->task_name);
                exit(-1);
            }

            // Get the ind. var. of the outer most enclosing loop (say k) and create a k'>=k condition.
            for(tmp=encl_loop; NULL != tmp->enclosing_loop; tmp=tmp->enclosing_loop );
            char *var_name = DA_var_name(DA_loop_induction_variable(tmp));
            Variable_ID ovar = ovars[var_name];
            Variable_ID ivar = ivars[var_name];
            GEQ_Handle ge = R_root->add_GEQ();
            ge.update_coef(ovar,1);
            ge.update_coef(ivar,-1);

            F_Or *or1 = R_root->add_or();
            // Get the ind. var. of all enclosing loops (say k,m,n) and create a (k'>k || m'>m || n'>n) condition.
            for(tmp=encl_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
                F_And *and1 = or1->add_and();

                // Force at least one of the input variables (i.e., induction variables for DEF) to be less than
                // the corresponding output variable, so that the USE happens at least one iteration after the DEF
                char *var_name = DA_var_name(DA_loop_induction_variable(tmp));
                Variable_ID ovar = ovars[var_name];
                Variable_ID ivar = ivars[var_name];
                GEQ_Handle ge = and1->add_GEQ();
                ge.update_coef(ovar,1);
                ge.update_coef(ivar,-1);
                ge.update_const(-1);
            }
        }
// DEBUG
//        R.print();
// END DEBUG

        // Add equalities demanded by the array subscripts. For example is the DEF is A[k][k] and the
        // USE is A[m][n] then add (k=m && k=n).
        add_array_subscript_equalities(R, R_root, ivars, ovars, def, use);

        R.simplify();
        if( R.is_upper_bound_satisfiable() || R.is_lower_bound_satisfiable() ){
            dep_edges[use] = R;
        }

        // if this is the definition, all the uses that will follow will be "normal" DU chains.
        if( und == def_und ){
            after_def = 1;
        }
    }

    if( DEP_FLOW==dep_type ){
        dep_edges[exit_node] = create_exit_relation(exit_node, def);
    }

    return dep_edges;
}


void declare_global_vars(node_t *node){
    map<string, Free_Var_Decl *> tmp_map;
    node_t *tmp;


    if( FOR == node->type ){
        set <char *> ind_names;

        // Store the names of the induction variables of all the loops that enclose this loop
        for(tmp=node; NULL != tmp; tmp=tmp->enclosing_loop ){
            ind_names.insert( DA_var_name(DA_loop_induction_variable(tmp)) );
        }

        // Find all the variables in the lower bound that are not induction variables and
        // declare them as global variables (symbolic)
        declare_globals_in_tree(DA_loop_lb(node), ind_names);

        // Find all the variables in the upper bound that are not induction variables and
        // declare them as global variables (symbolic)
        declare_globals_in_tree(DA_for_econd(node), ind_names);
    }


    if( BLOCK == node->type ){
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            declare_global_vars(tmp);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            declare_global_vars(node->u.kids.kids[i]);
        }
    }

    return;
}

long int getVarCoeff(expr_t *root, const char * var_name){
    switch( root->type ){
        case MUL:
            if( INTCONSTANT == root->l->type ){
                assert( (IDENTIFIER == root->r->type) && !strcmp(var_name, root->r->value.name) );
                return root->l->value.int_const;
            }else if( INTCONSTANT == root->r->type ){
                assert( (IDENTIFIER == root->l->type) && !strcmp(var_name, root->l->value.name) );
                return root->r->value.int_const;
            }else{
                fprintf(stderr,"ERROR: getVarCoeff(): malformed expression: \"%s\"\n",expr_tree_to_str(root));
                exit(-1);
            }
            break; // although control can never reach this point

        default:
            if( treeContainsVar(root->l, var_name) ){
                return getVarCoeff(root->l, var_name);
            }else if( treeContainsVar(root->r, var_name) ){
                return getVarCoeff(root->r, var_name);
            }else{
                fprintf(stderr,"ERROR: getVarCoeff(): tree: \"%s\" does not contain variable: \"%s\"\n",expr_tree_to_str(root), var_name);
                exit(-1);
            }
            break; // although control can never reach this point
    }

    // control should never reach this point
    assert(0);
    return 0;
}


void free_tree( expr_t *root ){
//FIXME: do something
    return;
}

// WARNING: this function it destructive.  It actually removes nodes from the tree and deletes them altogether.
// In many cases you will need to pass a copy of the tree to this function.
expr_t *removeVarFromTree(expr_t *root, const char *var_name){

    // If we are given a MUL directly it means the constraint is of the form a*X = b and
    // we were given the "a*X" part.  In that case, we return NULL and the caller will
    // know what to do.
    if( MUL == root->type ){
        if( treeContainsVar(root->l, var_name) || treeContainsVar(root->r, var_name) ){
            return NULL;
        }
    }

    if( treeContainsVar(root->l, var_name) ){
        if( MUL == root->l->type ){
            free_tree( root->l );
            return root->r;
        }else{
            root->l = removeVarFromTree(root->l, var_name);
            return root;
        }
    }else if( treeContainsVar(root->r, var_name) ){
        if( MUL == root->r->type ){
            free_tree( root->r );
            return root->l;
        }else{
            root->r = removeVarFromTree(root->r, var_name);
            return root;
        }
    }else{
        fprintf(stderr,"ERROR: removeVarFromTree(): tree: \"%s\" does not contain variable: \"%s\"\n",expr_tree_to_str(root), var_name);
        exit(-1);
    }

}

expr_t *copy_tree(expr_t *root){
    expr_t *e;

    if( NULL == root )
        return NULL;

    e = (expr_t *)calloc( 1, sizeof(expr_t) );
    e->type = root->type;
    if( INTCONSTANT == root->type )
        e->value.int_const = root->value.int_const;
    else if( IDENTIFIER == root->type )
        e->value.name = strdup(root->value.name);
    e->l = copy_tree(root->l);
    e->r = copy_tree(root->r);

    return e;
}


// WARNING: This function is destructive in that it actually changes
// the nodes of the tree.  If you need your original tree intact,
// you should pass it a copy of the tree.
void negateTree(expr_t *root){
    if( NULL == root )
        return;

    switch( root->type){
        case INTCONSTANT:
            root->value.int_const *= -1;
            break;

        case MUL:
            if( INTCONSTANT == root->l->type ){
                root->l->value.int_const *= -1;
            }else if( INTCONSTANT == root->r->type ){
                root->r->value.int_const *= -1;
            }else{
                fprintf(stderr,"ERROR: negateTree(): malformed expression: \"%s\"\n",expr_tree_to_str(root));
            }
            break;

        default:
            negateTree(root->l);
            negateTree(root->r);
    }

    return;
}


// Move the first argument "e_src" to the second argument "e_dst".
// That means that we negate each element of e_src and add it to e_dst.
//
// WARNING: This function is destructive in that it actually changes
// the nodes of the tree e_src.  If you need your original tree intact,
// you should pass it a copy of the tree.
expr_t *moveToOtherSideOfEQ(expr_t *e_src, expr_t *e_dst){
    expr_t *e;

    if( (INTCONSTANT == e_src->type) && (0 == e_src->value.int_const) )
        return e_dst;

    e = (expr_t *)calloc( 1, sizeof(expr_t) );
    e->type = ADD;
    e->l = e_dst;
    negateTree(e_src);
    e->r = e_src;

    return e;
}

expr_t *solveConstraintForVar(expr_t *constr_exp, const char *var_name){
    expr_t *e, *e_other;
    long int c;

    assert( (EQ_OP == constr_exp->type) || (GE == constr_exp->type) );

    if( treeContainsVar(constr_exp->l, var_name) ){
        e = copy_tree(constr_exp->l);
        e_other = copy_tree(constr_exp->r);
    }else if( treeContainsVar(constr_exp->r, var_name) ){
        e = copy_tree(constr_exp->r);
        e_other = copy_tree(constr_exp->l);
    }else{
        assert(0);
    }

    c = getVarCoeff(e, var_name);
    assert( 0 != c );
    e = removeVarFromTree(e, var_name);

    if( NULL == e ){
        // We are here because the constraint is of the form a*X = b and removeVarFromTree() has
        // returned NULL. Therefore we only keep the RHS, since the coefficient has been saved earlier.
        e = e_other;
    }else{
        // Else removeVarFromTree() returned an expression with the variable removed and we need
        // to move it to the other side.
        if( c < 0 )
            e = moveToOtherSideOfEQ(e_other, e);
        else
            e = moveToOtherSideOfEQ(e, e_other);
    }

    c = labs(c);
    if( c != 1 ){
        fprintf(stderr,"WARNING: solveConstraintForVar() resulted in the generation of a DIV. This has not been tested thoroughly\n");
        expr_t *e_cns = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_cns->type = INTCONSTANT;
        e_cns->value.int_const = c;

        expr_t *e_div = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_div->type = DIV;
        e_div->l = e;
        e_div->r = e_cns;
        e = e_div;
    }
    return e;

}

int treeContainsVar(expr_t *root, const char *var_name){

    if( NULL == root )
        return 0;

    switch( root->type ){
        case IDENTIFIER:
            if( !strcmp(var_name, root->value.name) )
                return 1;
            return 0;
        default:
            if( treeContainsVar(root->l, var_name) )
                return 1;
            if( treeContainsVar(root->r, var_name) )
                return 1;
            return 0;
    }
    return 0;
}

const char *findBoundsOfVar(expr_t *exp, const char *var_name, Relation R){
    char *lb = NULL, *ub = NULL;
    stringstream ss;
    set<expr_t *> ges = findAllGEsWithVar(var_name, exp);

    set<expr_t *>::iterator e_it;
    for(e_it=ges.begin(); e_it!=ges.end(); e_it++){
        int exp_has_output_vars = 0;

        expr_t *ge_exp = *e_it;
        int c = getVarCoeff(ge_exp, var_name);
        assert(c);

        expr_t *rslt_exp = solveConstraintForVar(ge_exp, var_name);

        // If the expression has output variables we need to ignore it
   
        if( !R.is_set() ){
            for(int i=0; i<R.n_out(); i++){
                const char *ovar = R.output_var(i+1)->char_name();
                if( treeContainsVar(rslt_exp, ovar) ){
                    exp_has_output_vars = 1;
                    break;
                }
            }
        }

        if( !exp_has_output_vars ){
            if( c > 0 ){ // then lower bound
                lb = strdup( expr_tree_to_str(rslt_exp) );
            }else{ // else upper bound
                ub = strdup( expr_tree_to_str(rslt_exp) );
            }
        }

    }

    if( NULL != lb ){
        ss << "(" << lb << ")";
        free(lb);
    }else{
        ss << "??";
    }

    ss << "..";

    if( NULL != ub ){
        ss << "(" << ub << ")";
        free(ub);
    }else{
        ss << "??";
    }

    return strdup(ss.str().c_str());
}

////////////////////////////////////////////////////////////////////////////////
//
expr_t *removeNodeFromAND(expr_t *node, expr_t *root){
    if( NULL == root )
        return NULL;

    if( node == root->l ){
        assert( L_AND == root->type );
        return root->r;
    }

    if( node == root->r ){
        assert( L_AND == root->type );
        return root->l;
    }

    root->r = removeNodeFromAND(node, root->r);
    root->l = removeNodeFromAND(node, root->l);
    return root;

}

expr_t *createGEZero(expr_t *exp){
    expr_t *zero = (expr_t *)calloc( 1, sizeof(expr_t) );
    zero->type = INTCONSTANT;
    zero->value.int_const = 0;

    expr_t *e = (expr_t *)calloc( 1, sizeof(expr_t) );
    e->type = GE;
    e->l = exp;
    e->r = zero;

    return e;
}

////////////////////////////////////////////////////////////////////////////////
//
// This function uses the transitivity of the ">=" operator to eliminate variables from
// the GEs.  As an example, if we are trying to eliminate X from: X-a>=0 && b-X-1>=0 we
// solve for "X" and get: b-1>=X && X>=a which due to transitivity gives us: b-1>=a
expr_t *eliminateVarUsingTransitivity(expr_t *exp, const char *var_name, Relation R){
    set<expr_t *> ges;
    set<expr_t *> ubs, lbs;
    set<expr_t *>::iterator it;

    ges = findAllGEsWithVar(var_name, exp);

    // solve all GEs that involve "var_name" and get all the lower/upper bounds.
    for(it=ges.begin(); it!=ges.end(); it++){
        expr_t *ge_exp = *it;
        int c = getVarCoeff(ge_exp, var_name);
        assert(c);

        expr_t *rslt_exp = solveConstraintForVar(ge_exp, var_name);

        if( c > 0 ){ // then lower bound
            lbs.insert( copy_tree(rslt_exp) );
        }else{ // else upper bound
            ubs.insert( copy_tree(rslt_exp) );
        }

        // Remove each GE that involves "var_name" from "exp"
        exp = removeNodeFromAND(ge_exp, exp);
    }

    // Add all the combinations of LBs and UBs in the expression
    for(it=ubs.begin(); it!=ubs.end(); it++){
        set<expr_t *>::iterator lb_it;
        for(lb_it=lbs.begin(); lb_it!=lbs.end(); lb_it++){
            expr_t *new_e = moveToOtherSideOfEQ(*lb_it, *it);
            new_e = createGEZero(new_e);

            expr_t *e = (expr_t *)calloc( 1, sizeof(expr_t) );
            e->type = L_AND;
            e->l = exp;
            e->r = new_e;
            exp = e;

        }
    }
  
    return exp;
}

expr_t *solveExpressionTreeForVar(expr_t *exp, const char *var_name, Relation R){
    set<expr_t *> eqs;
    expr_t *rslt_exp;

    eqs = findAllEQsWithVar(var_name, exp);
    if( eqs.empty() ){
        return NULL;
    }

    // we have to solve for all the ovars that are immediately solvable
    // and propagate the solutions to the other ovars, until everything is
    // solvable (assuming that this is always feasible).
    int i=0;
    while( true ){
        // If we tried to solve for 1000 output variables and still haven't
        // found a solution, probably something went wrong and we've entered
        // an infinite loop
        assert( i++ < 1000 );

        // If one of the equations includes this variable and no other output
        // variables, we solve that equation for the variable and return the solution.
        rslt_exp = solveDirectlySolvableEQ(exp, var_name, R);
        if( NULL != rslt_exp ){
            return rslt_exp;
        }

        // If control reached this point it means that all the equations
        // we are trying to solve contain output vars, so try to solve all
        // the other equations first that are solvable and replace these
        // output variables with the corresponding solutions.
        for(int i=0; i<R.n_out(); i++){
            const char *ovar = strdup(R.output_var(i+1)->char_name());

            // skip the variable we are trying to solve for.
            if( !strcmp(ovar, var_name) )
                continue;

            expr_t *tmp_exp = solveDirectlySolvableEQ(exp, ovar, R);
            if( NULL != tmp_exp ){
                substituteExprForVar(tmp_exp, ovar, exp);
            }
            free((void *)ovar);
        }
    }

    // control should never reach here
    assert(0);
    return NULL;
}


expr_t *findParent(expr_t *root, expr_t *node){
    expr_t *tmp;

    if( NULL == root )
        return NULL;

    switch( root->type ){
        case INTCONSTANT:
        case IDENTIFIER:
            return NULL;

        default:
            if( ((NULL != root->l) && (root->l == node)) || ((NULL != root->r) && (root->r == node)) )
                return root;
            // If this node is not the parent, it might still be an ancestor, so go deeper.
            tmp = findParent(root->r, node);
            if( NULL != tmp ) return tmp;
            tmp = findParent(root->l, node);
            return tmp;
    }

    return NULL;
}

// This function assumes that all variables will be kids of a MUL.
// This assumption is true for all contraint equations that were
// generated by omega.
expr_t *findParentOfVar(expr_t *root, const char *var_name){
    expr_t *tmp;

    switch( root->type ){
        case INTCONSTANT:
            return NULL;

        case MUL:
            if( ((IDENTIFIER == root->l->type) && !strcmp(var_name, root->l->value.name)) ||
                ((IDENTIFIER == root->r->type) && !strcmp(var_name, root->r->value.name))   ){
                return root;
            }
            return NULL;

        default:
            tmp = findParentOfVar(root->r, var_name);
            if( NULL != tmp ) return tmp;
            tmp = findParentOfVar(root->l, var_name);
            return tmp;
    }
    return NULL;
}

void multiplyTreeByConstant(expr_t *exp, int c){

    if( NULL == exp ){
        return;
    }

    switch( exp->type ){
        case IDENTIFIER:
            return;

        case INTCONSTANT:
            exp->value.int_const *= c;
            break;

        case ADD:
            multiplyTreeByConstant(exp->l, c);
            multiplyTreeByConstant(exp->r, c);
            break;

        case MUL:
        case DIV:
            multiplyTreeByConstant(exp->l, c);
            break;
        default:
            fprintf(stderr,"ERROR: multiplyTreeByConstant() Unknown node type: \"%d\"\n",exp->type);
            assert(0);
    }
    return;
}


static void substituteExprForVar(expr_t *exp, const char *var_name, expr_t *root){
    expr_t *eq_exp, *new_exp, *mul, *parent;
    set<expr_t *> cnstr, ges;

    cnstr = findAllEQsWithVar(var_name, root);
    ges = findAllGEsWithVar(var_name, root);
    cnstr.insert(ges.begin(), ges.end());
    
    set<expr_t *>::iterator e_it;
    for(e_it=cnstr.begin(); e_it!=cnstr.end(); e_it++){
        int c;
        eq_exp = *e_it;

        // Get the coefficient of the variable and multiply the
        // expression with it so we replace the whole MUL instead
        // of hanging the expression off the MUL.
        c = getVarCoeff(eq_exp, var_name);
        new_exp = copy_tree(exp);
        multiplyTreeByConstant(new_exp, c);
        
        // Find the MUL and its parent and do the replacement.
        mul = findParentOfVar(eq_exp, var_name);
        parent = findParent(eq_exp, mul);
        assert( NULL != parent );
        if( parent->l == mul ){
            parent->l = new_exp;
        }else if( parent->r == mul ){
            parent->r = new_exp;
        }else{
            assert(0);
        }

    }
    return;
}


static expr_t *solveDirectlySolvableEQ(expr_t *exp, const char *var_name, Relation R){
    set<expr_t *> eqs;
    expr_t *eq_exp, *rslt_exp;

    eqs = findAllEQsWithVar(var_name, exp);
    set<expr_t *>::iterator e_it;
    for(e_it=eqs.begin(); e_it!=eqs.end(); e_it++){
        int exp_has_output_vars = 0;

        eq_exp = *e_it;
        rslt_exp = solveConstraintForVar(eq_exp, var_name);

        if( R.is_set() )
            return rslt_exp;
  
        for(int i=0; i<R.n_out(); i++){
            const char *ovar = R.output_var(i+1)->char_name();
            if( treeContainsVar(rslt_exp, ovar) ){
                exp_has_output_vars = 1;
                break;
            }
        }

        if( !exp_has_output_vars ){
            return rslt_exp;
        }
    }
    return NULL;
}


////////////////////////////////////////////////////////////////////////////////
//
expr_t *findEQwithVar(const char *var_name, expr_t *exp){
    expr_t *tmp_r, *tmp_l;

    switch( exp->type ){
        case IDENTIFIER:
            if( !strcmp(var_name, exp->value.name) )
                return exp; // return something non-NULL
            return NULL;

        case INTCONSTANT:
            return NULL;

        case L_OR:
            // If you find it in both legs, panic
            tmp_l = findEQwithVar(var_name, exp->l);
            tmp_r = findEQwithVar(var_name, exp->r);
            if( (NULL != tmp_l) && (NULL != tmp_r) ){
                fprintf(stderr,"ERROR: findEQwithVar(): variable \"%s\" is not supposed to be in more than one conjuncts.\n",var_name);
                exit( -1 );
            }
            // otherwise proceed normaly
            if( NULL != tmp_l ) return tmp_l;
            if( NULL != tmp_r ) return tmp_r;
            return NULL;

        case L_AND:
        case ADD:
        case MUL:
            // If you find it in either leg, return a non-NULL pointer
            tmp_l = findEQwithVar(var_name, exp->l);
            if( NULL != tmp_l ) return tmp_l;

            tmp_r = findEQwithVar(var_name, exp->r);
            if( NULL != tmp_r ) return tmp_r;

            return NULL;

        case EQ_OP:
            // If you find it in either leg, return this EQ
            if( NULL != findEQwithVar(var_name, exp->l) )
                return exp;
            if( NULL != findEQwithVar(var_name, exp->r) )
                return exp;
            return NULL;

        case GE: // We are not interested on occurances of the variable in GE relations.
        default:
            return NULL;
    }
    return NULL;
}


////////////////////////////////////////////////////////////////////////////////
//
static inline set<expr_t *> findAllEQsWithVar(const char *var_name, expr_t *exp){
    return findAllConstraintsWithVar(var_name, exp, EQ_OP);
}

////////////////////////////////////////////////////////////////////////////////
//
static inline set<expr_t *> findAllGEsWithVar(const char *var_name, expr_t *exp){
    return findAllConstraintsWithVar(var_name, exp, GE);
}

////////////////////////////////////////////////////////////////////////////////
//
static set<expr_t *> findAllConstraintsWithVar(const char *var_name, expr_t *exp, int constr_type){
    set<expr_t *> eq_set;
    set<expr_t *> tmp_r, tmp_l;

    if( NULL == exp )
        return eq_set;

    switch( exp->type ){
        case IDENTIFIER:
            if( !strcmp(var_name, exp->value.name) ){
                eq_set.insert(exp); // just so we return a non-empty set
            }
            return eq_set;

        case INTCONSTANT:
            return eq_set;

        case L_OR:
            // If you find it in both legs, panic
            tmp_l = findAllConstraintsWithVar(var_name, exp->l, constr_type);
            tmp_r = findAllConstraintsWithVar(var_name, exp->r, constr_type);
            if( !tmp_l.empty() && !tmp_r.empty() ){
                fprintf(stderr,"ERROR: findAllConstraintsWithVar(): variable \"%s\" is not supposed to be in more than one conjuncts.\n", var_name);
                exit( -1 );
            }
            // otherwise proceed normaly
            if( !tmp_l.empty() ) return tmp_l;
            if( !tmp_r.empty() ) return tmp_r;
            return eq_set;

        case L_AND:
            // Merge the sets of whatever you find in both legs
            eq_set = findAllConstraintsWithVar(var_name, exp->l, constr_type);
            tmp_r  = findAllConstraintsWithVar(var_name, exp->r, constr_type);
            eq_set.insert(tmp_r.begin(), tmp_r.end());

            return eq_set;

        case ADD:
        case MUL:
            // If you find it in either leg, return a non-empty set
            tmp_l = findAllConstraintsWithVar(var_name, exp->l, constr_type);
            if( !tmp_l.empty() ) return tmp_l;

            tmp_r = findAllConstraintsWithVar(var_name, exp->r, constr_type);
            if( !tmp_r.empty() ) return tmp_r;

            return eq_set;

        case EQ_OP:
            // Only look deeper if the caller wants EQs
            if( EQ_OP == constr_type ){
                // If you find it in either leg, return this EQ
                tmp_l = findAllConstraintsWithVar(var_name, exp->l, constr_type);
                tmp_r = findAllConstraintsWithVar(var_name, exp->r, constr_type);
                if(  !tmp_l.empty() || !tmp_r.empty() ){
                    eq_set.insert(exp);
                }
            }
            return eq_set;

        case GE:
            // Only look deeper if the caller wants GEQs
            if( GE == constr_type ){
                // If you find it in either leg, return this EQ
                tmp_l = findAllConstraintsWithVar(var_name, exp->l, constr_type);
                tmp_r = findAllConstraintsWithVar(var_name, exp->r, constr_type);
                if(  !tmp_l.empty() || !tmp_r.empty() ){
                    eq_set.insert(exp);
                }
            }
            return eq_set;

        default:
            return eq_set;
    }
    return eq_set;
}


////////////////////////////////////////////////////////////////////////////////
//
void printActualParameters(dep_t *dep, expr_t *rel_exp, int type){
    Relation R = *dep->rel;

    int dst_count = R.n_out();
    for(int i=0; i<dst_count; i++){
        if( i ) printf(", ");
        const char *var_name = strdup(R.output_var(i+1)->char_name());

        expr_t *solution = solveExpressionTreeForVar(copy_tree(rel_exp), var_name, R);
        if( NULL != solution )
            printf("%s", expr_tree_to_str(solution));
        else
            printf("%s", findBoundsOfVar(copy_tree(rel_exp), var_name, R));
        free((void *)var_name);
    }
    return;
}

////////////////////////////////////////////////////////////////////////////////
//
expr_t *cnstr_to_tree(Constraint_Handle cnstr, int cnstr_type){
    expr_t *e_add, *root = NULL;
    expr_t *e_cns;
    
    for(Constr_Vars_Iter cvi(cnstr); cvi; cvi++){
        expr_t *e_cns = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_cns->type = INTCONSTANT;
        e_cns->value.int_const = (*cvi).coef;

        expr_t *e_var = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_var->type = IDENTIFIER;
        e_var->value.name = strdup( (*cvi).var->char_name() );

        expr_t *e_mul = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_mul->type = MUL;
        e_mul->l = e_cns;
        e_mul->r = e_var;

        // In the first iteration set the multiplication node to be the root
        if( NULL == root ){
            root = e_mul;
        }else{
            expr_t *e_add = (expr_t *)calloc( 1, sizeof(expr_t) );
            e_add->type = ADD;
            e_add->l = root;
            e_add->r = e_mul;
            root = e_add;
        }

    }
    // Add the constant
    if( cnstr.get_const() ){
        e_cns = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_cns->type = INTCONSTANT;
        e_cns->value.int_const = cnstr.get_const();

        e_add = (expr_t *)calloc( 1, sizeof(expr_t) );
        e_add->type = ADD;
        e_add->l = root;
        e_add->r = e_cns;
        root = e_add;
    }

    // Add a zero on the other size of the constraint (EQ, GEQ)
    e_cns = (expr_t *)calloc( 1, sizeof(expr_t) );
    e_cns->type = INTCONSTANT;
    e_cns->value.int_const = 0;

    e_add = (expr_t *)calloc( 1, sizeof(expr_t) );
    e_add->type = cnstr_type;
    e_add->l = root;
    e_add->r = e_cns;
    root = e_add;

    return root;
}


inline expr_t *eq_to_tree(EQ_Handle e){
    return cnstr_to_tree(e, EQ_OP);
}

inline expr_t *geq_to_tree(GEQ_Handle ge){
    return cnstr_to_tree(ge, GE);
}


expr_t *conj_to_tree( Conjunct *conj ){
    expr_t *root = NULL;

    for(EQ_Iterator ei = conj->EQs(); ei; ei++) {
        expr_t *eq_root = eq_to_tree(*ei);

        if( NULL == root ){
            root = eq_root;
        }else{
            expr_t *e_and = (expr_t *)calloc( 1, sizeof(expr_t) );
            e_and->type = L_AND;
            e_and->l = root;
            e_and->r = eq_root;
            root = e_and;
        }

    }

    for(GEQ_Iterator gi = conj->GEQs(); gi; gi++) {
        expr_t *geq_root = geq_to_tree(*gi);

        if( NULL == root ){
            root = geq_root;
        }else{
            expr_t *e_and = (expr_t *)calloc( 1, sizeof(expr_t) );
            e_and->type = L_AND;
            e_and->l = root;
            e_and->r = geq_root;
            root = e_and;
        }
    }

    return root;
}


expr_t *relation_to_tree( Relation R ){
    expr_t *root = NULL;

    if( R.is_null() )
        return NULL;

    for(DNF_Iterator di(R.query_DNF()); di; di++) {
        expr_t *conj_root = conj_to_tree( *di );

        if( NULL == root ){
            root = conj_root;
        }else{
            expr_t *e_or = (expr_t *)calloc( 1, sizeof(expr_t) );
            e_or->type = L_OR;
            e_or->l = root;
            e_or->r = conj_root;
            root = e_or;
        }
    }
    return root;
}


static set<expr_t *> findAllConjunctions(expr_t *exp){
    set<expr_t *> eq_set;
    set<expr_t *> tmp_r, tmp_l;

    if( NULL == exp )
        return eq_set;

    switch( exp->type ){
        case L_OR:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            tmp_l = findAllConjunctions(exp->l);
            if( !tmp_l.empty() ){
                eq_set.insert(tmp_l.begin(), tmp_l.end());
            }else{
                eq_set.insert(exp->l);
            }

            tmp_r = findAllConjunctions(exp->r);
            if( !tmp_r.empty() ){
                eq_set.insert(tmp_r.begin(), tmp_r.end());
            }else{
                eq_set.insert(exp->r);
            }

            return eq_set;

        default:
            tmp_l = findAllConjunctions(exp->l);
            eq_set.insert(tmp_l.begin(), tmp_l.end());
            tmp_r = findAllConjunctions(exp->r);
            eq_set.insert(tmp_r.begin(), tmp_r.end());
            return eq_set;
        
    }

}


void findAllConstraints(expr_t *tree, set<expr_t *> &eq_set){

    if( NULL == tree )
        return;

    switch( tree->type ){
        case EQ_OP:
        case GE:
            eq_set.insert(tree);
            break;

        default:
            findAllConstraints(tree->l, eq_set);
            findAllConstraints(tree->r, eq_set);
            break;
    }
    return;
}

////////////////////////////////////////////////////////////////////////////////
//
void findAllVars(expr_t *e, set<string> &var_set){

    if( NULL == e )
        return;

    switch( e->type ){
        case IDENTIFIER:
            var_set.insert(e->value.name);
            break;

        default:
            findAllVars(e->l, var_set);
            findAllVars(e->r, var_set);
            break;
    }
    return ;
}


        
    
void tree_to_omega_set(expr_t *tree, Constraint_Handle &handle, map<string, Variable_ID> all_vars, int sign){
    int coef;
    Variable_ID v = NULL;
    string var_name;
    map<string, Variable_ID>::iterator v_it;

    if( NULL == tree ){
        return;
    }

    switch( tree->type ){
        case INTCONSTANT:
            coef = tree->value.int_const;
            handle.update_const(sign*coef);
            break;

        case MUL:
            assert( (tree->l->type == INTCONSTANT && tree->r->type == IDENTIFIER) || (tree->r->type == INTCONSTANT && tree->l->type == IDENTIFIER) );
            if( tree->l->type == INTCONSTANT ){
                coef = tree->l->value.int_const;
                var_name = tree->r->value.name;
                for(v_it=all_vars.begin(); v_it!=all_vars.end(); v_it++){
                    if( !v_it->first.compare(var_name) ){
                        v = v_it->second;
                        break;
                    }
                }
            }else{
                coef = tree->r->value.int_const;
                var_name = tree->l->value.name;
                for(v_it=all_vars.begin(); v_it!=all_vars.end(); v_it++){
                    if( !v_it->first.compare(var_name) ){
                        v = v_it->second;
                        break;
                    }
                }
            }
            assert( NULL != v );
            handle.update_coef(v,sign*coef);
            break;

        default:
            tree_to_omega_set(tree->r, handle, all_vars, sign);
            tree_to_omega_set(tree->l, handle, all_vars, sign);
            break;
    }
    return;

}

expr_t *simplify_constraint_based_on_execution_space(expr_t *tree, Relation S_es, Relation R){
    Relation S_rslt;
    set<expr_t *> e_set;
    set<expr_t *>::iterator e_it;

    findAllConstraints(tree, e_set);

    // For every constraint in the tree
    for(e_it=e_set.begin(); e_it!=e_set.end(); e_it++){
        map<string, Variable_ID> all_vars;
        Relation S_tmp;
        expr_t *e = *e_it;

        // Create a new Set
        set<string>vars;
        findAllVars(e, vars);
        S_tmp = Relation(S_es.n_set());

        for(int i=1; i<=S_es.n_set(); i++){
            string var_name = S_es.set_var(i)->char_name();
            S_tmp.name_set_var( i, strdup(var_name.c_str()) );
            all_vars[var_name] = S_tmp.set_var( i );
        }

        // Deal with the remaining variables in the expression tree,
        // which should all be global.
        set<string>::iterator v_it;
        for(v_it=vars.begin(); v_it!=vars.end(); v_it++){
            string var_name = *v_it;

            // If it's not one of the vars we've already dealt with
            if( all_vars.find(var_name) == all_vars.end() ){
                // Make sure this variable is global
                map<string, Free_Var_Decl *>::iterator g_it;
                g_it = global_vars.find(var_name);
                assert( g_it != global_vars.end() );
                // And get a reference to the local version of the variable in S_tmp.
                all_vars[var_name] = S_tmp.get_local(g_it->second);
            }
        }

        F_And *S_root = S_tmp.add_and();
        Constraint_Handle handle;
        if( EQ_OP == e->type ){
            handle = S_root->add_EQ();
        }else if( GE == e->type ){
            handle = S_root->add_GEQ();
        }else{
            assert(0);
        }

        // Add the two sides of the constraint to the Omega set
        tree_to_omega_set(e->l, handle, all_vars, 1);
        tree_to_omega_set(e->r, handle, all_vars, -1);
        S_tmp.simplify();

        // Calculate S_exec_space - ( S_exec_space ^ S_tmp )
        Relation S_intrs = Intersection(copy(S_es), copy(S_tmp));
        Relation S_diff = Difference(copy(S_es), S_intrs);
        S_diff.simplify();

        // If it's not FALSE, then throw S_tmp in S_rslt
        if( S_diff.is_upper_bound_satisfiable() || S_diff.is_lower_bound_satisfiable() ){
            if( S_rslt.is_null() ){
                S_rslt = copy(S_tmp);
            }else{
                S_rslt = Intersection(S_rslt, S_tmp);
            }
        }

        S_tmp.Null();
        S_diff.Null();
    }

    return relation_to_tree( S_rslt );
}


////////////////////////////////////////////////////////////////////////////////
//
// WARNING: this function is destructive.  It actually removes nodes from the tree
// and deletes them altogether. In many cases you will need to pass a copy of the
// tree to this function.
string simplifyConditions(Relation R, expr_t *exp, Relation S_es){
    stringstream ss;
    set<expr_t *> conj, simpl_conj;
    set<expr_t *>::iterator cj_it;

    conj = findAllConjunctions(exp);

    // If we didn't find any it's because there are no unions, so the whole
    // expression "exp" is one conjunction
    if( conj.empty() ){
        conj.insert(exp);
    }

    // Eliminate the conjunctions that are covered by the execution space
    // and simplify the remaining ones
    for(cj_it = conj.begin(); cj_it != conj.end(); cj_it++){
        expr_t *cur_exp = *cj_it;

        int dst_count = R.n_out();
        for(int i=0; i<dst_count; i++){
            const char *ovar = strdup(R.output_var(i+1)->char_name());
            // If we find the variable in an EQ then we solve for the variable and
            // substitute the solution for the variable everywhere in the conjunction.
            expr_t *solution = solveExpressionTreeForVar(cur_exp, ovar, R);
            if( NULL != solution ){
                substituteExprForVar(solution, ovar, cur_exp);
            }else{
                // If the variable is in no EQs but it's in GEs, we have to use transitivity
                // to eliminate it.  For example: X-a>=0 && b-X-1>=0 => b-1>=X && X>=a => b-1>=a
                cur_exp = eliminateVarUsingTransitivity(cur_exp, ovar, R);
            }
            free((void *)ovar);
        }
        cur_exp = simplify_constraint_based_on_execution_space(cur_exp, S_es, R);
        if(cur_exp){
            simpl_conj.insert(cur_exp);
        }
    }

    if( simpl_conj.size() > 1 )
        ss << "( (";
    for(cj_it = simpl_conj.begin(); cj_it != simpl_conj.end(); cj_it++){
        expr_t *cur_exp = *cj_it;
        if( cj_it != simpl_conj.begin()  )
            ss << ") | ("; /* The symbol for Logical OR in the JDF parser is a single "|" */
        ss << expr_tree_to_str(cur_exp);
    }
    if( simpl_conj.size() > 1 )
        ss << ") )";

    return ss.str();
}


////////////////////////////////////////////////////////////////////////////////
//
void print_body(node_t *task_node){
    printf("BODY\n\n");
    printf("%s\n", quark_tree_to_body(task_node));
    printf("\nEND\n");
}


void print_header(){
    printf("extern \"C\" %%{\n"
           "  /**\n"                 /* The following lines are PLASMA/DPLASMA specific */
           "   * PLASMA include for defined and constants.\n"
           "   *\n"
           "   * @precisions normal z -> s d c\n"
           "   *\n"
           "   */\n"
           "#include <plasma.h>\n"    
           "#include <core_blas.h>\n" /* The previous lines are PLASMA/DPLASMA specific */
           "\n"
           "#include \"dague.h\"\n"
           "#include \"data_distribution.h\"\n"
           "#include \"memory_pool.h\"\n"
           "%%}\n"
           "\n");
}


#if 0

////////////////////////////////////////////////////////////////////////////////
//
set<dep_t *> edge_map_to_dep_set(map<char *, set<dep_t *> > edges){
    map<char *, set<dep_t *> >::iterator edge_it;
    set<dep_t *> deps;

    for(edge_it = edges.begin(); edge_it != edges.end(); ++edge_it ){
        set<dep_t *> tmp;

        tmp = edge_it->second;
        deps.insert(tmp.begin(), tmp.end());
    }

    return deps;
}

////////////////////////////////////////////////////////////////////////////////
//
// Grow the edge in all directions (except toward tasks we've already visited) until you hit the destination.
// When/if you do return the transitive Relation we've created.
// If the recursion returns a succe, put it
Relation *grow_edge(Relation *trns_rel, task_t *rel_dst, task_t *final_dst, set <dep_t *>all_deps, set <task_t *>visited_tasks){
    set <Relation *> transitive_edges;
    Relation R_u;

    set<dep_t *>::iterator it;
    for (it=all_deps.begin(); it!=all_deps.end(); it++){
        dep_t *dep = *it;

        // if this dependency takes us to the EXIT, or back to the same task, ignore it.
        if( NULL == dep->dst || dep->dst->task == rel_dst )
            continue;

        // if this dependency takes us back to a task we've already visited ignore it, _except_ if it's the destination.
        if( dep->dst->task != final_dst && visited_tasks.find(dep->dst->task) != visited_tasks.end() )
            continue;

        // if this dependency picks up where our current transitive edge stoped, process it.
        if( dep->src->task == rel_dst ){
            
// debug starts
//        printf("Composing with edge: ");
//        dep->rel->print_with_subs(stdout);
// debug ends
            Relation rel = Composition(copy(*dep->rel), copy(*trns_rel));
// debug starts
//        printf("Resulting edge: ");
//        rel.print_with_subs(stdout);
// debug ends

            // If we reached our final destination we should store the edge, otherwise try to grow it further
            if( dep->dst->task == final_dst ){
                transitive_edges.insert(new Relation(rel));
            }else{
                visited_tasks.insert(dep->dst->task);
                Relation *new_rel = grow_edge(&rel, dep->dst->task, final_dst, all_deps, visited_tasks);
                if( NULL != new_rel )
                    transitive_edges.insert(new_rel);
                visited_tasks.erase(dep->dst->task);
            }

        }
    }

    if( transitive_edges.empty() )
        return NULL;

    set <Relation *>::iterator te_it;
    for(te_it = transitive_edges.begin(); te_it != transitive_edges.end(); te_it++){
        Relation *rel = *te_it;
        if( R_u.is_null() )
            R_u = *rel;
        R_u = Union(R_u, copy(*rel));
    }

    return new Relation(R_u);
}


////////////////////////////////////////////////////////////////////////////////
//
Relation *union_all_transitive_edges(node_t *src, node_t *dst, set<dep_t *> c_deps, set<dep_t *> f_deps){
    Relation R_u;
    set <task_t *> visited_tasks;
    set <dep_t *> all_deps;
    set <Relation *> transitive_edges;

    all_deps.clear();
    all_deps.insert(c_deps.begin(), c_deps.end());
    all_deps.insert(f_deps.begin(), f_deps.end());

    visited_tasks.insert(src->task);

    // Process every _control_ edge that starts from the same task as "src" except if it is the
    // exact same relation we are trying to reduce in the caller.
    // Ignore edges to EXIT.
    set<dep_t *>::iterator it;
    for (it=c_deps.begin(); it!=c_deps.end(); it++){
        dep_t *dep = *it;
        if( dep->src->task == src->task && NULL != dep->dst && (dep->src != src || dep->dst != dst) ){
            // If we find an anti edge that goes directly to the destination, don't try to grow it
            if( dep->dst->task == dst->task ){
// debug starts
//        printf("Inserting s->e anti edge: ");
//        dep->rel->print_with_subs(stdout);
// debug ends
                transitive_edges.insert(new Relation(*dep->rel));
            }else{
// debug starts
//        printf("growing anti edge: ");
//        dep->rel->print_with_subs(stdout);
// debug ends
                visited_tasks.insert(dep->dst->task);
                Relation *rel = grow_edge(dep->rel, dep->dst->task, dst->task, all_deps, visited_tasks);
                if( NULL != rel )
                    transitive_edges.insert(rel);
                visited_tasks.erase(dep->dst->task);
            }
        }
    }

    // Process every _flow_ edge that starts from the same task as "src". 
    // Ignore edges to EXIT.
    for (it=f_deps.begin(); it!=f_deps.end(); it++){
        dep_t *dep = *it;
        if( dep->src->task == src->task && NULL != dep->dst ){
            // If we find a flow edge that goes directly to the destination, don't try to grow it
            if( dep->dst->task == dst->task ){
// debug starts
//        printf("Inserting s->e flow edge: ");
//        dep->rel->print_with_subs(stdout);
// debug ends
                transitive_edges.insert(new Relation(*dep->rel));
            }else{
                visited_tasks.insert(dep->dst->task);
// debug starts
//        printf("growing flow edge: ");
//        dep->rel->print_with_subs(stdout);
// debug ends
                Relation *rel = grow_edge(dep->rel, dep->dst->task, dst->task, all_deps, visited_tasks);
                if( NULL != rel )
                    transitive_edges.insert(rel);
                visited_tasks.erase(dep->dst->task);
            }
        }
    }

    set <Relation *>::iterator te_it;
    for(te_it = transitive_edges.begin(); te_it != transitive_edges.end(); te_it++){
        Relation *rel = *te_it;
        if( R_u.is_null() )
            R_u = *rel;
// debug starts
//        printf("Unioning: ");
//        (*rel).print_with_subs(stdout);
// debug ends
        // TODO: the following copy() is probably unnessecary and is causing a memory leak. All the
        // pointers to Relations in the set are already generated with a "new", so 
        // destroying them should be a good thing.
        R_u = Union(R_u, copy(*rel));
    }

    return new Relation(R_u);
}


/*
typedef struct tg_node_ tg_node_t;
typedef struct tg_edge_{
    Relation *R;
    tg_node_t *dst;
} tg_edge_t;

struct tg_node_{
    map<task_t *, tg_edge_t *> edges;
};

////////////////////////////////////////////////////////////////////////////////
//
list<task_t *> detect_loop(set<list<task_t *> > &loops, task_t *src_task, set <dep_t *> all_deps, list <task_t *> visited_tasks){

    visited_tasks.push_back(src_task);
    for (it=all_deps.begin(); it!=all_deps.end(); it++){
        dep_t *dep = *it;

        // if this is an edge that starts from the task the recursion ended at, follow it.
        if( dep->src->task == src_task ){
            task_t *dst_task = dep->dst->task;
            list<task_t *> loop = detect_loop(dst_task, all_deps, visited_tasks);
        }
    }
    visited_tasks.pop_back();
    
}
*/
////////////////////////////////////////////////////////////////////////////////
//


/*
void build_graph(tg_node_t *tg_node, task_t *task, list <task_t *> &visited_tasks){
    list<task_t *>::iterator vt_it;
    vt_it = visited_tasks.find(task);
    // If "task" is a task we haven't put in the graph yet, create a new node
    if( edge_it == visited_tasks.end() ){
        Relation R;
        tg_node_t *new_tg_node = (tg_node_t *)calloc(1, sizeof(tg_node_t));
        tg_edge_t *new_tg_edge = (tg_edge_t *)calloc(1, sizeof(tg_edge_t));
        new_tg_edge->dst = new_tg_node;
        new_tg_edge->R = 

        tg_node[task] = new_tg_edge;
    }
}
*/


int task_already_visited(task_t *task, list <task_t *>vt){
    list<task_t *>::iterator it;

    for(it=vt.begin(); it!=vt.end(); it++){
        if( *it == task )
            return 1;
    }
    return 0;
}


void grow_edge_until_loop(task_t *task, Relation *rel, list <task_t *>visited_tasks, list <Relation *>visited_relations){

    // If "task" is a task we have already visited, compute the loop effects
    if( task_already_visited(task, visited_tasks) ){
#error "here"
    }
}


map<task_t *, set<Relation *> > detect_loops_and_compute_transitive_closures(dep_t *antidep, set<dep_t *> c_deps, set<dep_t *> f_deps){
    map<task_t *, set<Relation *> > loop_effects;
    list <task_t *>visited_tasks;
    map<task_t *, Relation *> merged_Rels;
    set <dep_t *> all_deps;
    set< list<task_t *> > loops;

    // Merge all the edges into a single set, except for the edge we are trying to reduce.
    all_deps.clear();
    all_deps.insert(f_deps.begin(), f_deps.end());
    set<dep_t *>::iterator it;
    for (it=c_deps.begin(); it!=c_deps.end(); it++){
        dep_t *dep = *it;
        if (dep->src == antidep->src && dep->dst == antidep->dst && dep->rel == dep->rel)
            continue;
        all_deps.insert(dep);
    }

    for (it=all_deps.begin(); it!=all_deps.end(); it++){
        dep_t *dep = *it;

        // Ignore dependencies that do not start from the correct task
        if (dep->src->task != antidep->src->task)
            continue;

        // For each destination task that can be reached from task "antidep->src->task"
        //   union all relations between these two tasks and store the result into the "merged_Rels" map
        Relation *tmpR = merged_Rels[dep->dst->task];
        if( (NULL != tmpR) && !(tmpR->is_null()) ){
            tmpR = new Relation( Union(copy(dep->rel), tmpR) );
        }else{
            tmpR = new Relation(dep->rel);
        }
        merged_Rels[dep->dst->task] = tmpR;


    }

//    tg_node_t *root_task = (tg_node_t *)calloc(1, sizeof(tg_node_t));

    visited_tasks.push_back(dep->src->task);
    map<task_t *, Relation *>::iterator mr_it;
    for(mr_it=merged_Rels.begin(); mr_it!=merged_Rels.end(); mr_it+){
//        build_graph(root_task, dep->dst->task, visited_tasks);
        task *dst_task = mr_it->first;
        Relation *rel  = mr_it->second;
        grow_edge_until_loop(dst_task, rel, visited_tasks);
#error "coding in grow_edge_until_loop and then I should pick up here"
    }
    visited_tasks.pop_back();





/*
        // if it's a loop to self add it to the result set of loops
        if (dep->dst->task == antidep->src->task){ 
            list<task_t *> loop;
            loop.push_back(dep->dst->task);
            loops.insert(loop);
            continue;
        }
*/

/*
    // 
    visited_tasks.push_back(src_task);
    for (it=all_deps.begin(); it!=all_deps.end(); it++){
        dep_t *dep = *it;

        // if this is an edge that starts from the task that the anti-dependence starts from, follow it.
        if( dep->src->task == src_task ){
            task_t *dst_task = dep->dst->task;
            loops = detect_loop(dst_task, all_deps, visited_tasks);
        }
    }
    visited_tasks.pop_back();
*/
}


////////////////////////////////////////////////////////////////////////////////
//
map<char *, set<dep_t *> > restrict_synch_edges_due_to_transitive_edges(set<dep_t *> c_deps, set<dep_t *> f_deps){
    map<char *, set<dep_t *> > synch_edges;
    map<task_t *, set<Relation *> > loop_effects;

// foreach anti-dependency edge Ai from "src" to "dst"
//     foreach transitive edge Ei from "src" to "dst" through anti and flow deps
//         Eu = Eu union Ei
//     Ai = Ai - Eu

    set<dep_t *>::iterator it;
    for (it=c_deps.begin(); it!=c_deps.end(); it++){
        dep_t *dep = *it;

        node_t *src = dep->src;
        node_t *dst = dep->dst;
        Relation R_ai = *dep->rel;

        loop_effects = detect_loops_and_compute_transitive_closures(dep, c_deps, f_deps);

        Relation *R_eu = union_all_transitive_edges(src, dst, c_deps, f_deps);
        if( !(*R_eu).is_null() ){
// debug starts
            printf("Subtracting: ");
            R_eu->simplify();
//            R_eu->print();
            R_eu->print_with_subs(stdout);
            printf("from: ");
            R_ai.simplify();
//            R_ai.print();
            R_ai.print_with_subs(stdout);
// debug ends

            R_ai = Difference(R_ai, *R_eu);
// debug starts
            printf("Result: ");
            R_ai.simplify();
//            R_ai.print();
            R_ai.print_with_subs(stdout);
// debug ends
        }else{
            printf("Nothing to subtract, returning: ");
            R_ai.print_with_subs(stdout);
        }
        printf("\n");
        // Todo: do we need to release the memory taken by dep->rel ?
        dep->rel = new Relation(R_ai);
    }

    // regroup the deps into a map based on their starting task
    for (it=c_deps.begin(); it!=c_deps.end(); it++){
        dep_t *dep = *it;
        set <dep_t *>dep_set;

        char *task_name = dep->src->task->task_name;

        dep_set.insert(dep);

        // if the map has a set for this task already, merge the two sets
        map<char *, set<dep_t *> >::iterator edge_it;
        edge_it = synch_edges.find(task_name);
        if( edge_it != synch_edges.end() ){
            set<dep_t *>tmp_set;
            tmp_set = synch_edges[task_name];
            dep_set.insert(tmp_set.begin(), tmp_set.end());
        }

        // insert the (merged) set into the map.
        synch_edges[task_name] = dep_set;
    }

    return synch_edges;
}

#endif


void print_types_of_formal_parameters(node_t *root){
    symtab_t *scope;
    symbol_t *sym;

    scope = root->symtab;
    do{
        for(sym=scope->symbols; NULL!=sym; sym=sym->next){
            if( !strcmp(sym->var_type, "PLASMA_desc") ){
                printf("desc_%s [type = \"PLASMA_desc\"]\n",sym->var_name);
                printf("data_%s [type = \"dague_ddesc_t *\"]\n",sym->var_name);
            }else{
                printf("%s [type = \"%s\"]\n",sym->var_name, sym->var_type);
            }
        }
        scope = scope->parent;
    }while(NULL != scope);

    printf("%s",create_pool_declarations());
}

////////////////////////////////////////////////////////////////////////////////
//
void interrogate_omega(node_t *root, var_t *head){
    var_t *var;
    und_t *und;
    map<node_t *, map<node_t *, Relation> > flow_sources, output_sources, anti_sources;
    map<char *, set<dep_t *> > outgoing_edges;
    map<char *, set<dep_t *> > incoming_edges;
    map<char *, set<dep_t *> > synch_edges;

    print_header();
    print_types_of_formal_parameters(root);

    declare_global_vars(root);

    node_t *entry = DA_create_Entry();
    node_t *exit_node = DA_create_Exit();

    ////////////////////////////////////////////////////////////////////////////////
    // Find the correct set of outgoing edges using the following steps:
    // For each variable V:
    // a) create Omega relations corresponding to all FLOW dependency edges associated with V.
    // b) create Omega relations corresponding to all OUTPUT dependency edges associated with V.
    // c) based on the OUTPUT edges, kill FLOW edges that need to be killed.

    for(var=head; NULL != var; var=var->next){
        flow_sources.clear();
        output_sources.clear();
        anti_sources.clear();
        // Create flow edges starting from the ENTRY
        flow_sources[entry]   = create_entry_relations(entry, var, DEP_FLOW);
        output_sources[entry] = create_entry_relations(entry, var, DEP_OUT);

        // For each DEF create all flow and output edges and for each USE create all anti edges.
        for(und=var->und; NULL != und ; und=und->next){
            if(is_und_write(und)){
                node_t *def = und->node;
                flow_sources[def] = create_dep_relations(und, var, DEP_FLOW, exit_node);
                output_sources[def] = create_dep_relations(und, var, DEP_OUT, exit_node);
            }
            if(is_und_read(und)){
                node_t *use = und->node;
                anti_sources[use] = create_dep_relations(und, var, DEP_ANTI, exit_node);
            }
        }


        // Minimize the flow dependencies (also known as true dependencies, or read-after-write)
        // by factoring in the output dependencies (also known as write-after-write).
        //

        // For every DEF that is the source of flow dependencies
        map<node_t *, map<node_t *, Relation> >::iterator flow_src_it;
        for(flow_src_it=flow_sources.begin(); flow_src_it!=flow_sources.end(); ++flow_src_it){
            char *task_name;
            set<dep_t *> dep_set;
            // Extract from the map the actual DEF from which all the deps in this map start from
            node_t *def = flow_src_it->first;
            map<node_t *, Relation> flow_deps = flow_src_it->second;

            // Keep the task name, we'll use it to insert all the edges of the task in a map later on.
            if( ENTRY == def->type ){
#ifdef DEBUG_2
                printf("\n[[ ENTRY ]] =>\n");
#endif
                task_name = strdup("ENTRY");
            }else{
                task_name = def->task->task_name;
#ifdef DEBUG_2
                printf("\n[[ %s(",def->task->task_name);
                for(int i=0; NULL != def->task->ind_vars[i]; ++i){
                    if( i ) printf(",");
                    printf("%s", def->task->ind_vars[i]);
                }
                printf(") %s ]] =>\n", tree_to_str(def));
#endif
            }

            map<node_t *, map<node_t *, Relation> >::iterator out_src_it;
            out_src_it = output_sources.find(def);
            map<node_t *, Relation> output_deps = out_src_it->second;

            // For every flow dependency that has "def" as its source
            map<node_t *, Relation>::iterator fd_it;
            for(fd_it=flow_deps.begin(); fd_it!=flow_deps.end(); ++fd_it){
                dep_t *dep = (dep_t *)calloc(1, sizeof(dep_t));

                dep->src = def;

                // Get the sink of the edge and the actual omega Relation
                node_t *sink = fd_it->first;
                Relation fd1_r = fd_it->second;

#ifdef DEBUG_2
                if( EXIT != sink->type){
                    printf("    => [[ %s(",sink->task->task_name);
                    for(int i=0; NULL != sink->task->ind_vars[i]; ++i){
                    if( i ) printf(",");
                    printf("%s", sink->task->ind_vars[i]);
                    }
                    printf(") %s ]] ", tree_to_str(sink));
                }
                fd1_r.print();
#endif

                Relation rAllKill;
                // For every output dependency that has "def" as its source
                map<node_t *, Relation>::iterator od_it;
                for(od_it=output_deps.begin(); od_it!=output_deps.end(); ++od_it){
                    Relation rKill;

                    // Check if sink of this output edge is
                    // a) the same as the source (loop carried output edge on myself)
                    // b) the source of a new flow dep that has the same sink as the original flow dep.
                    node_t *od_sink = od_it->first;
                    Relation od_r = od_it->second;

                    if( od_sink == def ){
                        // If we made it to here it means that I need to ask omega to compute:
                        // rKill := fd1_r compose od_r;  (Yes, this is the original fd)

#ifdef DEBUG_3
                        printf("Killer output dep:\n");
                        od_r.print();
#endif

                        rKill = Composition(copy(fd1_r), copy(od_r));
                        rKill.simplify();
#ifdef DEBUG_3
                        printf("Killer composed:\n");
                        rKill.print();
#endif
                    }else{

                        // See if there is a flow dep with source equal to od_sink
                        // and sink equal to "sink"
                        map<node_t *, map<node_t *, Relation> >::iterator fd2_src_it;
                        fd2_src_it = flow_sources.find(od_sink);
                        if( fd2_src_it == flow_sources.end() ){
                            continue;
                        }
                        map<node_t *, Relation> fd2_deps = fd2_src_it->second;
                        map<node_t *, Relation>::iterator fd2_it;
                        fd2_it = fd2_deps.find(sink);
                        if( fd2_it == fd2_deps.end() ){
                            continue;
                        }
                        Relation fd2_r = fd2_it->second;
                        // If we made it to here it means that I nedd to ask omega to compute:

#ifdef DEBUG_3
                        printf("Killer flow2 dep:\n");
                        fd2_r.print();
                        printf("Killer output dep:\n");
                        od_r.print();
#endif

                        rKill = Composition(copy(fd2_r), copy(od_r));
                        rKill.simplify();
#ifdef DEBUG_3
                        printf("Killer composed:\n");
                        rKill.print();
#endif
                    }
                    if( rAllKill.is_null() || rAllKill.is_set() ){ // shouldn't this be !rAllKill.is_set() ?
                        rAllKill = rKill;
                    }else{
                        rAllKill = Union(rAllKill, rKill);
                    }
                }
                Relation rReal;
                if( rAllKill.is_null() || rAllKill.is_set() ){
                    rReal = fd1_r;
                }
                else{
#ifdef DEBUG_3
                    printf("Final Killer:\n");
                    rAllKill.print();
#endif
                    rReal = Difference(fd1_r, rAllKill);
                }

                rReal.simplify();
#ifdef DEBUG_3
                printf("Final Edge:\n");
                rReal.print();
                printf("==============================\n");
#endif
                if( rReal.is_upper_bound_satisfiable() || rReal.is_lower_bound_satisfiable() ){

                    if( EXIT == sink->type){
#ifdef DEBUG_2
                        printf("    => [[ EXIT ]] ");
#endif
                        dep->dst = NULL;
                    }else{
                        dep->dst = sink;
#ifdef DEBUG_2
                        printf("    => [[ %s(",sink->task->task_name);
                        for(int i=0; NULL != sink->task->ind_vars[i]; ++i){
                            if( i ) printf(",");
                            printf("%s", sink->task->ind_vars[i]);
                        }
                        printf(") %s ]] ", tree_to_str(sink));
#endif
                    }
#ifdef DEBUG_2
                    rReal.print_with_subs(stdout);
#endif
                    dep->rel = new Relation(rReal);
                    dep_set.insert(dep);

                }
            }

            map<char *, set<dep_t *> >::iterator edge_it;
            edge_it = outgoing_edges.find(task_name);
            if( edge_it != outgoing_edges.end() ){
                set<dep_t *>tmp_set;
                tmp_set = outgoing_edges[task_name];
                dep_set.insert(tmp_set.begin(), tmp_set.end());
            }
            outgoing_edges[task_name] = dep_set;
        }
    }

#ifdef DEBUG_2
printf("================================================================================\n");
#endif

        // Minimize the anti dependencies (also known as write-after-read).
        // by factoring in the flow dependencies (also known as read-after-write).
        //

        // For every USE that is the source of anti dependencies
        map<node_t *, map<node_t *, Relation> >::iterator anti_src_it;
        for(anti_src_it=anti_sources.begin(); anti_src_it!=anti_sources.end(); ++anti_src_it){
            Relation Ra0, Roi, Rai;
            set<dep_t *> dep_set;
            // Extract from the map the actual USE from which all the deps in this map start from
            node_t *use = anti_src_it->first;
            map<node_t *, Relation> anti_deps = anti_src_it->second;

            Ra0.Null(); // just in case.

            // Iterate over all anti-deps that start from this USE looking for one that has the same
            // node as its sink (in other words we are looking for an array passed as INOUT to a kernel).
            map<node_t *, Relation>::iterator ad_it;
            for(ad_it=anti_deps.begin(); ad_it!=anti_deps.end(); ++ad_it){
                // Get the sink of the edge.
                node_t *sink = ad_it->first;
                if( sink == use ){
                    Ra0 = ad_it->second;
                    break;
                }
            }

            // Use this "self edge" (composed with the appropriate output-edge) to reduce all other anti-edges.
            // But if there is no self edge, add all the anti-deps into the set unchanged.

            // Iterate over all anti-dependency edges (but skip the self-edge).
            for(ad_it=anti_deps.begin(); ad_it!=anti_deps.end(); ++ad_it){
                // Get the sink of the edge.
                node_t *sink = ad_it->first;
                // skip the self-edge
                if( sink == use ){
                    continue;
                }
                Rai = ad_it->second;
    
                if( Ra0.is_null() ){
                    dep_t *dep = (dep_t *)calloc(1, sizeof(dep_t));
                    dep->src = use;
                    dep->dst = sink;
                    dep->rel = new Relation(Rai);
                    dep_set.insert(dep);
                }else{
                    // find an output edge that start where the self-edge starts and ends at "sink".
                    Roi.Null();
                    map<node_t *, map<node_t *, Relation> >::iterator out_src_it;
                    out_src_it = output_sources.find(use);
                    if(out_src_it == output_sources.end()){
                       printf("anti w/o out: %s:%s -> %s:%s\n",use->task->task_name, tree_to_str(use), sink->task->task_name, tree_to_str(sink) ); 
                       exit(-1);
                    }
                    map<node_t *, Relation> output_deps = out_src_it->second;
    
                    map<node_t *, Relation>::iterator od_it;
                    for(od_it=output_deps.begin(); od_it!=output_deps.end(); ++od_it){
                        // If the sink of this output edge is the same as the sink of the anti-flow we are good.
                        node_t *od_sink = od_it->first;
                        if( od_sink == sink ){
                            Roi = od_it->second;
                            break;
                        }
                    }
    
                    // Now we have all the necessary edges, so we perform the operation.
                    if( !Roi.is_null() ){
                        Relation rKill = Composition(copy(Roi), copy(Ra0));
                        Rai = Difference(Rai, rKill);
    
                        dep_t *dep = (dep_t *)calloc(1, sizeof(dep_t));
                        dep->src = use;
                        dep->dst = sink;
                        dep->rel = new Relation(Rai);
                        dep_set.insert(dep);
                    }
                }
            }
            // see if the current task already has some synch edges and if so merge them with the new ones.
            map<char *, set<dep_t *> >::iterator edge_it;
            char *task_name = use->task->task_name;
            edge_it = synch_edges.find(task_name);
            if( edge_it != synch_edges.end() ){
                set<dep_t *>tmp_set;
                tmp_set = synch_edges[task_name];
                dep_set.insert(tmp_set.begin(), tmp_set.end());
            }
            synch_edges[task_name] = dep_set;
        }

#if 0
        set<dep_t *>c_deps = edge_map_to_dep_set(synch_edges);
        set<dep_t *>f_deps = edge_map_to_dep_set(outgoing_edges);
        synch_edges = restrict_synch_edges_due_to_transitive_edges(c_deps, f_deps);
#endif

#ifdef DEBUG_2
printf("================================================================================\n");
#endif

    //////////////////////////////////////////////////////////////////////////////////////////
    // inverse all outgoing edges (unless they are going to the EXIT) to generate the incoming
    // edges in the JDF.
    map<char *, set<dep_t *> >::iterator edge_it;
    edge_it = outgoing_edges.begin();
    for( ;edge_it != outgoing_edges.end(); ++edge_it ){

        char *task_name;
        set<dep_t *> outgoing_deps;

        outgoing_deps = edge_it->second;

        if( outgoing_deps.empty() ){
            continue;
        }

#ifdef DEBUG_2
        task_t *src_task = (*outgoing_deps.begin())->src->task;

        if( NULL == src_task )
            printf("ENTRY \n");
        else
            printf("%s \n",src_task->task_name);
#endif

        set<dep_t *>::iterator it;
        for (it=outgoing_deps.begin(); it!=outgoing_deps.end(); it++){
            set<dep_t *>incoming_deps;
            dep_t *dep = *it;

            // If the destination is the EXIT, we do not inverse the edge
            if( NULL == dep->dst ){
                continue;
            }

            node_t *sink = dep->dst;
            task_name = sink->task->task_name;

            // Create the incoming edge by inverting the outgoing one.
            dep_t *new_dep = (dep_t *)calloc(1, sizeof(dep_t));
            Relation inv = *dep->rel;
            new_dep->rel = new Relation(Inverse(copy(inv)));
            new_dep->src = dep->src;
            new_dep->dst = dep->dst;

            // If there are some deps already associated with this task, retrieve them
            map<char *, set<dep_t *> >::iterator edge_it;
            edge_it = incoming_edges.find(task_name);
            if( edge_it != incoming_edges.end() ){
                incoming_deps = incoming_edges[task_name];
            }
            // Add the new dep to the list of deps we will associate with this task
            incoming_deps.insert(new_dep);
            // Associate the deps with the task
            incoming_edges[task_name] = incoming_deps;
        }

    }

#ifdef DEBUG_2
printf("================================================================================\n");
#endif

    //////////////////////////////////////////////////////////////////////////////////////////
    // Print all edges.

    edge_it = outgoing_edges.begin();
    for( ;edge_it != outgoing_edges.end(); ++edge_it ){
        task_t *src_task;
        char *task_name;
        set<dep_t *> deps;

        task_name = edge_it->first;
        deps = edge_it->second;

        // Get the source task from the dependencies
        if( !deps.empty() ){
            src_task = (*deps.begin())->src->task;
        }else{
            // If there are no outgoing deps, get the source task from the incoming dependencies
            set<dep_t *> in_deps = incoming_edges[task_name];
            if( !in_deps.empty() ){
                src_task = (*in_deps.begin())->src->task;
            }else{
                // If there are no incoming and no outgoing deps, skip this task
                continue;
            }
        }

        // If the source task is NOT the ENTRY, then dump all the info
        if( NULL != src_task ){

            printf("\n\n%s(",task_name);
            for(int i=0; NULL != src_task->ind_vars[i]; ++i){
                if( i ) printf(",");
                printf("%s", src_task->ind_vars[i]);
            }
            printf(")\n");

            Relation S_es = process_and_print_execution_space(src_task->task_node);
            printf("\n");
            node_t *reference_data_element = print_default_task_placement(src_task->task_node);
            printf("\n");
            print_pseudo_variables(deps, incoming_edges[task_name]);
            printf("\n");
            print_edges(deps, incoming_edges[task_name], S_es, reference_data_element);
            S_es.Null();
            printf("\n");


// DEBUG start
            printf("  /*\n  The following is a superset of the necessary anti-dependencies:\n");
            map<node_t *, map<node_t *, Relation> >::iterator anti_src_it;
            for(anti_src_it=anti_sources.begin(); anti_src_it!=anti_sources.end(); ++anti_src_it){
                // Extract from the map the actual USE from which all the deps in this map start from
                node_t *use = anti_src_it->first;
                if( use->task == src_task ){
                    map<node_t *, Relation> anti_deps = anti_src_it->second;

                    map<node_t *, Relation>::iterator ad_it;
                    for(ad_it=anti_deps.begin(); ad_it!=anti_deps.end(); ++ad_it){
                        node_t *sink = ad_it->first;
                        Relation ad_r = ad_it->second;
                        char *n1 = use->task->task_name;
                        char *n2 = sink->task->task_name;
                        printf("  ANTI edge from %s:%s to %s:%s ",n1, tree_to_str(use), n2, tree_to_str(sink));
                        ad_r.print_with_subs();
                    }
                }
            }
            printf("\n");
// DEBUG end

#if 0
            map<char *, set<dep_t *> >::iterator edge_it;
            edge_it = control_edges.find(task_name);
            if( edge_it != control_edges.end() ){
                set<dep_t *>tmp_set;
                tmp_set = control_edges[task_name];
                set<dep_t *>::iterator dep_it;
                for (dep_it=tmp_set.begin(); dep_it!=tmp_set.end(); dep_it++){
                        node_t *src = (*dep_it)->src;
                        node_t *dst = (*dep_it)->dst;
                        Relation r = *(*dep_it)->rel;
                        char *n1 = src->task->task_name;
                        char *n2 = dst->task->task_name;
                        printf("  CONTROL %s:%s -> %s:%s ",n1, tree_to_str(src), n2, tree_to_str(dst));
                        r.print_with_subs();
                }
            }
#endif
            printf("  */\n\n");

            print_body(src_task->task_node);
        }
    }
}


static const char *econd_tree_to_ub(node_t *econd){
    stringstream ss;
    char *a, *b;
        
    switch( econd->type ){
        case L_AND:
            a = strdup( econd_tree_to_ub(econd->u.kids.kids[0]) );
            b = strdup( econd_tree_to_ub(econd->u.kids.kids[1]) );
            ss << "( (" << a << " < " << b << ")? " << a << " : " << b << " )";
            free(a);
            free(b);
            return strdup(ss.str().c_str());
// TODO: handle logical or (L_OR) as well.

        case LE:
            ss << tree_to_str(DA_rel_rhs(econd));
            return strdup(ss.str().c_str());

        case LT:
            ss << tree_to_str(DA_rel_rhs(econd)) << "-1";
            return strdup(ss.str().c_str());

        default:
            fprintf(stderr,"ERROR: econd_tree_to_ub() cannot deal with node of type: %d\n",econd->type);
            exit(-1);
    }
}

static Relation process_and_print_execution_space(node_t *node){
    int i;
    node_t *tmp;
    list<node_t *> params;
    map<string, Variable_ID> vars;
    Relation S;

    printf("  /* Execution space */\n");
    for(tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        params.push_front(tmp);
    }
    assert( !params.empty() );

    S = Relation(params.size());
    F_And *S_root = S.add_and();

    for(i=1; !params.empty(); i++ ) {
        char *var_name;
        tmp = params.front();

        var_name = DA_var_name(DA_loop_induction_variable(tmp));
        S.name_set_var( i, var_name );
        vars[var_name] = S.set_var( i );

        printf("  %s = ", var_name);
        printf("%s..", tree_to_str(DA_loop_lb(tmp)));
        printf("%s\n", econd_tree_to_ub(DA_for_econd(tmp)));
        params.pop_front();
    }


    // Bound all induction variables of the loops enclosing the USE
    // using the loop bounds
    i=1;
    for(tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        char *var_name = DA_var_name(DA_loop_induction_variable(tmp));

        // Form the Omega expression for the lower bound
        Variable_ID var = vars[var_name];

        GEQ_Handle imin = S_root->add_GEQ();
        imin.update_coef(var,1);
        expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, vars, S);

        // Form the Omega expression for the upper bound
        process_end_condition(DA_for_econd(tmp), S_root, vars, S);
    }

    S.simplify();
    return S;
}

////////////////////////////////////////////////////////////////////////////////
//
static void print_pseudo_variables(set<dep_t *>out_deps, set<dep_t *>in_deps){
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

       assert( NULL != dep->dst);
       pseudo_vars[dep->dst->var_symname] = tree_to_str(dep->dst);

       if( NULL != dep->src->task ){
           pseudo_vars[dep->src->var_symname] = tree_to_str(dep->src);
       }
   }

   // Dump the map.
   map<string, string>::iterator pvit;
   for (pvit=pseudo_vars.begin(); pvit!=pseudo_vars.end(); pvit++){
       /*
        * JDF & QUARK specific optimization:
        * Add the keyword "data_" infront of the matrix to
        * differentiate the matrix from the struct.
        */
       printf("  /* %s == data_%s */\n",(pvit->first).c_str(), (pvit->second).c_str());
   }

}

// We are assuming that all leaves will be kids of a MUL or a DIV, or they will be an INTCONSTANT
// Conversely we are assuming that all MUL and DIV nodes will have ONLY leaves as kids.
// Leaves BTW are the types INTCONSTANT and IDENTIFIER.
static void groupExpressionBasedOnSign(expr_t *exp, set<expr_t *> &pos, set<expr_t *> &neg){

    if( NULL == exp )
        return;

    switch( exp->type ){
        case INTCONSTANT:
            if( exp->value.int_const < 0 ){
                neg.insert(exp);
            }else{
                pos.insert(exp);
            }
            break;

        case IDENTIFIER:
            pos.insert(exp);
            break;

        case MUL:
        case DIV:
            if( ( (INTCONSTANT != exp->l->type) && (INTCONSTANT != exp->r->type) ) ||
                ( (IDENTIFIER != exp->l->type) && (IDENTIFIER != exp->r->type) )      ){
               fprintf(stderr,"ERROR: groupExpressionBasedOnSign() cannot handle MUL and/or DIV nodes with kids that are not leaves: \"%s\"\n",expr_tree_to_str(exp));
                exit(-1);
            }

            if( INTCONSTANT == exp->l->type ){
                if( exp->l->value.int_const < 0 ){
                    neg.insert(exp);
                }else{
                    pos.insert(exp);
                }
            }else{
                if( exp->r->value.int_const < 0 ){
                    neg.insert(exp);
                }else{
                    pos.insert(exp);
                }
            }
            break;

        default:
            groupExpressionBasedOnSign(exp->l, pos, neg);
            groupExpressionBasedOnSign(exp->r, pos, neg);
            break;
    }
    return;
}

const char *type_to_str(int type){

    switch(type){
        case EMPTY: return "EMPTY";
        case INTCONSTANT: return "INTCONSTANT";
        case IDENTIFIER: return "IDENTIFIER";
        case ADDR_OF: return "ADDR_OF";
        case STAR: return "STAR";
        case PLUS: return "PLUS";
        case MINUS: return "MINUS";
        case TILDA: return "TILDA";
        case BANG: return "BANG";
        case ASSIGN: return "ASSIGN";
        case COND: return "COND";
        case ARRAY: return "ARRAY";
        case FCALL: return "FCALL";
        case ENTRY: return "ENTRY";
        case EXIT: return "EXIT";
        case EXPR: return "EXPR";
        case ADD: return "ADD";
        case SUB: return "SUB";
        case MUL: return "MUL";
        case DIV: return "DIV";
        case MOD: return "MOD";
        case B_AND: return "B_AND";
        case B_XOR: return "B_XOR";
        case B_OR: return "B_OR";
        case LSHIFT: return "LSHIFT";
        case RSHIFT: return "RSHIFT";
        case LT: return "LT";
        case GT: return "GT";
        case LE: return "LE";
        case GE: return "GE";
        case DEREF: return "DEREF";
        case S_U_MEMBER: return "S_U_MEMBER";
        case COMMA_EXPR: return "COMMA_EXPR";
        case BLOCK: return "BLOCK";
        default: return "???";
    }
}


const char *expr_tree_to_str(const expr_t *exp){
    string str;
    str = _expr_tree_to_str(exp);
    return strdup(str.c_str());
}

static string _expr_tree_to_str(const expr_t *exp){
    stringstream ss, ssL, ssR;
    unsigned int skipSymbol=0, first=1;
    unsigned int r_needs_paren = 0, l_needs_paren = 0;
    size_t j;
    set<expr_t *> pos, neg;
    set<expr_t *>::iterator it;
    string str;

    if( NULL == exp )
        return "";

    switch( exp->type ){
        case IDENTIFIER:
            str = string(exp->value.name);
            // If the JDF parser does not like structures,
            // convert all structure members (A.m) into variables Am
#if 0
            while ( (j=str.find('.',0)) != str.npos) { 
                str.replace(j, 1, "" ); 
            } 
#endif
            return str;

        case INTCONSTANT:
            if( exp->value.int_const < 0 )
                ss << "(" << exp->value.int_const << ")";
            else
                ss << exp->value.int_const;
            return ss.str();

        case EQ_OP:
        case GE:

            assert( (NULL != exp->l) && (NULL != exp->r) );

            groupExpressionBasedOnSign(exp->l, pos, neg);
            groupExpressionBasedOnSign(exp->r, neg, pos);

            // print all the positive members on the left
            first = 1;
            for(it=pos.begin(); it != pos.end(); it++){
                expr_t *e = *it;

                if( INTCONSTANT == e->type ){
                    // Only print the constant if it's non-zero, or if it's the only thing on this side
                    if( (0 != e->value.int_const) || (1 == pos.size()) ){
                        if( !first ){
                            ssL << "+";
                            l_needs_paren = 1;
                        }
                        first = 0;
                        ssL << labs(e->value.int_const);
                    }
                }else{
                    if( !first ){
                        ssL << "+";
                        l_needs_paren = 1;
                    }
                    first = 0;
                    if( INTCONSTANT == e->l->type ){
                        // skip the "1*"
                        if( MUL != e->type || labs(e->l->value.int_const) != 1 ){
                            ssL << labs(e->l->value.int_const);
                            ssL << type_to_symbol(e->type);
                        }
                        ssL << _expr_tree_to_str(e->r);
                    }else{
                        // in this case we can skip the "one" even for divisions
                        if( labs(e->r->value.int_const) != 1 ){
                            ssL << labs(e->r->value.int_const);
                            ssL << type_to_symbol(e->type);
                        }
                        ssL << _expr_tree_to_str(e->l);
                    }
                }
            }
            if( 0 == pos.size() ){
                ssL << "0";
            }

            // print all the negative members on the right
            first = 1;
            for(it=neg.begin(); it != neg.end(); it++){
                expr_t *e = *it;
                if( INTCONSTANT == e->type ){
                    // Only print the constant if it's non-zero, or if it's the only thing on this side
                    if( (0 != e->value.int_const) || (1 == neg.size()) ){
                        if( !first ){
                            ssR << "+";
                            r_needs_paren = 1;
                        }
                        first = 0;
                        ssR << labs(e->value.int_const);
                    }
                }else{
                    if( !first ){
                        ssR << "+";
                        r_needs_paren = 1;
                    }
                    first = 0;
                    if( INTCONSTANT == e->l->type ){
                        if( MUL != e->type || labs(e->l->value.int_const) != 1 ){
                            ssR << labs(e->l->value.int_const);
                            ssR << type_to_symbol(e->type);
                        }
                        ssR << _expr_tree_to_str(e->r);
                    }else{
                        if( labs(e->r->value.int_const) != 1 ){
                            ssR << labs(e->r->value.int_const);
                            ssR << type_to_symbol(e->type);
                        }
                        ssR << _expr_tree_to_str(e->l);
                    }
                }
            }
            if( 0 == neg.size() ){
                ssR << "0";
            }

            // Add some parentheses to make it easier for parser that will read in the JDF.
            ss << "(";
            if( l_needs_paren )
                ss << "(" << ssL.str() << ")";
            else
                ss << ssL.str();

            ss << type_to_symbol(exp->type);

            if( r_needs_paren )
                ss << "(" << ssR.str() << "))";
            else
                ss << ssR.str() << ")";

            return ss.str();

        case L_AND:
        case L_OR:

            assert( (NULL != exp->l) && (NULL != exp->r) );

            // If one of the two sides is a tautology, we will get a NULL string
            // and we have no reason to print the "&&" or "||" sign.
            if( _expr_tree_to_str(exp->l).empty() )
                return _expr_tree_to_str(exp->r);
            if( _expr_tree_to_str(exp->r).empty() )
                return _expr_tree_to_str(exp->l);

            if( L_OR == exp->type )
                ss << "(";
            ss << _expr_tree_to_str(exp->l);
            ss << " " << type_to_symbol(exp->type) << " ";
            ss << _expr_tree_to_str(exp->r);
            if( L_OR == exp->type )
                ss << ")";
            return ss.str();

        case ADD:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            if( (NULL != exp->l) && (INTCONSTANT == exp->l->type) ){
                if( exp->l->value.int_const != 0 ){
                    ss << _expr_tree_to_str(exp->l);
                }else{
                    skipSymbol = 1;
                }
            }else{
                ss << _expr_tree_to_str(exp->l);
            }


            // If the thing we add is a "-c" (where c is a constant)
            // detect it so we print "-c" instead of "+(-c)"
            if( (NULL != exp->r) && (INTCONSTANT == exp->r->type) && (exp->r->value.int_const < 0) ){
                ss << "-" << labs(exp->r->value.int_const);
                return ss.str();
            }

            if( !skipSymbol )
                ss << type_to_symbol(ADD);

            ss << _expr_tree_to_str(exp->r);

            return ss.str();

        case MUL:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            if( INTCONSTANT == exp->l->type ){
                if( exp->l->value.int_const != 1 ){
                    ss << "(";
                    ss << _expr_tree_to_str(exp->l);
                }else{
                    skipSymbol = 1;
                }
            }else{
                ss << _expr_tree_to_str(exp->l);
            }

            if( !skipSymbol )
                ss << type_to_symbol(MUL);

            ss << _expr_tree_to_str(exp->r);

            if( !skipSymbol )
                ss << ")";

            return ss.str();

        case DIV:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            ss << "(";
            if( INTCONSTANT == exp->l->type ){
                if( exp->l->value.int_const < 0 ){
                    ss << "(" << _expr_tree_to_str(exp->l) << ")";
                }else{
                    ss << _expr_tree_to_str(exp->l);
                }
            }else{
                ss << _expr_tree_to_str(exp->l);
            }

            ss << type_to_symbol(DIV);

            if( (INTCONSTANT == exp->r->type) && (exp->r->value.int_const > 0) )
                ss << _expr_tree_to_str(exp->r);
            else
                ss << "(" << _expr_tree_to_str(exp->r) << ")";

            ss << ")";
            return ss.str();

        default:
            ss << "{" << exp->type << "}";
            return ss.str();
    }
    return string();
}

////////////////////////////////////////////////////////////////////////////////
//
char *create_pseudotask(task_t *parent_task, Relation S_es, Relation cond, node_t *data_element, char *var_pseudoname, int ptask_count, const char *inout){

// ztsqrt_in_data_A1(k,m1)
//   /* Execution space */
//   /* solution of these
//    *k = 0..(desc_A.nt)-1-1
//    *m1 = k+2..desc_A.mt-1
//    * & (0==k) from the edge
//    */
// 
//   : data_A(m1,k)
// 
//   RW A <- data_A(m1,k)
//        -> K ztsqrt(k,m1)
// 
// BODY
// /* nothing */
// END
    char *parent_task_name, *pseudotask_name;
    char *formal_parameters = NULL;
    char *mtrx_name         = tree_to_str(DA_array_base(data_element));
    char *number;

    for(int i=0; NULL != parent_task->ind_vars[i]; ++i){
        if( i )
            formal_parameters = append_to_string(formal_parameters, ",", NULL, 0);
        formal_parameters = append_to_string(formal_parameters, parent_task->ind_vars[i], NULL, 0);
    }

    asprintf( &parent_task_name , "%s(%s)",parent_task->task_name, formal_parameters);
    asprintf( &pseudotask_name , "%s_%s_data_%s%d(%s)",parent_task->task_name, inout, mtrx_name, ptask_count, formal_parameters);

    Relation newS_es = Intersection(copy(S_es), Domain(copy(cond)));
    newS_es.simplify();

printf("\n################################\n");
printf("%s\n",pseudotask_name);
//S_es.print();
//cond.print();
//newS_es.print();

    for(int i=0; NULL != parent_task->ind_vars[i]; ++i){
        char *var_name = parent_task->ind_vars[i];
        printf("  %s = ", var_name); 
        expr_t *solution = solveExpressionTreeForVar(relation_to_tree(newS_es), var_name, copy(newS_es));
        if( NULL != solution )
            printf("%s\n", expr_tree_to_str(solution));
        else
            printf("%s\n", findBoundsOfVar(relation_to_tree(newS_es), var_name, copy(newS_es)) );
    }
             
char *data_str = tree_to_str(data_element);
printf("  : data_%s\n",data_str);
if( !strcmp(inout,"in") ){
    printf("  RW %s <- data_%s\n",var_pseudoname, data_str);
    printf("       -> %s %s\n",var_pseudoname, parent_task_name);
}else{
    printf("  RW %s <- %s %s\n",var_pseudoname, var_pseudoname, parent_task_name);
    printf("        -> data_%s\n",data_str);
}
printf("BODY\n/* nothing */\nEND\n");
printf("################################\n");


     free(parent_task_name);
     free(pseudotask_name);
     free(mtrx_name);
}



////////////////////////////////////////////////////////////////////////////////
//
void print_edges(set<dep_t *>outg_deps, set<dep_t *>incm_deps, Relation S_es, node_t *reference_data_element){
    task_t *this_task;
    set<dep_t *>::iterator dep_it;
    set<char *> vars;
    map<char *, set<dep_t *> > incm_map, outg_map;

    if( outg_deps.empty() && incm_deps.empty() ){
        return;
    }

    this_task = (*outg_deps.begin())->src->task;


    // Group the edges based on the variable they flow into or from
    for (dep_it=incm_deps.begin(); dep_it!=incm_deps.end(); dep_it++){
           char *symname = (*dep_it)->dst->var_symname;
           incm_map[symname].insert(*dep_it);
           vars.insert(symname);
    }
    for (dep_it=outg_deps.begin(); dep_it!=outg_deps.end(); dep_it++){
           char *symname = (*dep_it)->src->var_symname;
           outg_map[symname].insert(*dep_it);
           vars.insert(symname);
    }

    if( vars.size() > 6 ){
        fprintf(stderr,"WARNING: Number of variables (%lu) exceeds 6",vars.size());
    }


    // For each variable print all the incoming and the outgoing edges
    set<char *>::iterator var_it;
    for (var_it=vars.begin(); var_it!=vars.end(); var_it++){
        int pseudotask_count = 0;
        bool insert_fake_read = false;
        char *var_pseudoname = *var_it;
        set<dep_t *>ideps = incm_map[var_pseudoname];
        set<dep_t *>odeps = outg_map[var_pseudoname];

        if( !ideps.empty() && !odeps.empty() ){
            printf("  RW    ");
        }else if( !ideps.empty() ){
            printf("  READ  ");
        }else if( !odeps.empty() ){
            // printf("  WRITE ");
            /* 
             * DAGuE does not like write-only variables, so make it RW and make
             * it read from the data matrix tile that corresponds to this variable.
             */
            printf("  RW    ");
            insert_fake_read = true;
        }else{
            assert(0);
        }

        if( ideps.size() > 6 )
            fprintf(stderr,"WARNING: Number of incoming edges (%lu) for variable \"%s\" exceeds 6",ideps.size(), var_pseudoname);
        if( odeps.size() > 6 )
            fprintf(stderr,"WARNING: Number of outgoing edges (%lu) for variable \"%s\" exceeds 6",odeps.size(), var_pseudoname);

        // Print the pseudoname
        printf("%s",var_pseudoname);

        // print the incoming edges
        for (dep_it=ideps.begin(); dep_it!=ideps.end(); dep_it++){
             dep_t *dep = *dep_it;
             expr_t *rel_exp;
             string cond;

             // Needed by Omega
             (void)(*dep->rel).print_with_subs_to_string(false);

             rel_exp = relation_to_tree( *dep->rel );
             assert( NULL != dep->dst);
             if ( dep_it!=ideps.begin() )
                 printf("         ");
             printf(" <- ");

             task_t *src_task = dep->src->task;

             cond = simplifyConditions(*dep->rel, copy_tree(rel_exp), S_es);
             if( !cond.empty() )
                 printf("%s ? ",cond.c_str());

             if( NULL != src_task ){

                 printf("%s ", dep->src->var_symname);
                 printf("%s(",src_task->task_name);
                 printActualParameters(dep, rel_exp, SOURCE);
                 printf(") ");
             }else{
                 // ENTRY
                 bool need_pseudotask = false;
                 char *comm_mtrx, *refr_mtrx;

                 comm_mtrx = tree_to_str(DA_array_base(dep->dst));
                 refr_mtrx = tree_to_str(DA_array_base(reference_data_element));
                 // If the matrices are different and not co-located, we need a pseudo-task.
                 if( strcmp(comm_mtrx,refr_mtrx) && q2j_colocated_map[comm_mtrx].compare(q2j_colocated_map[refr_mtrx]) ){
                     need_pseudotask = true;
                 }else{
                     // If the element we are communicating is not the same as the reference element, we also need a pseudo-task.
                     int count = DA_array_dim_count(dep->dst);
                     if( DA_array_dim_count(reference_data_element) != count ){
                         fprintf(stderr,"Matrices with different dimension counts detected \"%s\" and \"%s\"."
                                        " This should never happen in dplasma\n",
                                        tree_to_str(dep->dst), tree_to_str(reference_data_element));
                         need_pseudotask = true;
                     }

                     for(int i=0; i<count && !need_pseudotask; i++){
                         char *a = tree_to_str(DA_array_index(dep->dst, i));
                         char *b = tree_to_str(DA_array_index(reference_data_element, i));
                         if( strcmp(a,b) )
                             need_pseudotask = true;
                         free(a);
                         free(b);
                     }
                 }
                 free(comm_mtrx);
                 free(refr_mtrx);

                 if( need_pseudotask ){
                     printf("[[ data_%s ]]", tree_to_str(dep->dst));
                     char *pseudotask = create_pseudotask(this_task, S_es, *dep->rel, dep->dst, var_pseudoname, pseudotask_count++, "in");
                 }else{
                     /*
                      * JDF & QUARK specific optimization:
                      * Add the keyword "data_" infront of the matrix to
                      * differentiate the matrix from the struct.
                      */
                     printf("data_%s", tree_to_str(dep->dst));
                 }
             }
             printf("\n");
#ifdef DEBUG_2
             if( NULL != src_task ){
                 printf("          // %s -> %s ",src_task->task_name, tree_to_str(dep->dst));
             }else{
                 printf("          // ENTRY -> %s ", tree_to_str(dep->dst));
             }
             (*dep->rel).print_with_subs(stdout);
#endif
        }

        if(insert_fake_read){
            dep_t *dep = *(odeps.begin());
            printf(" <- ");
             /*
              * JDF & QUARK specific optimization:
              * Add the keyword "data_" infront of the matrix to
              * differentiate the matrix from the struct.
              */
            printf("data_%s\n",tree_to_str(dep->src));
        }

        // print the outgoing edges
        for (dep_it=odeps.begin(); dep_it!=odeps.end(); dep_it++){
             dep_t *dep = *dep_it;
             expr_t *rel_exp;
             string cond;

             // Needed by Omega
             (void)(*dep->rel).print_with_subs_to_string(false);


             rel_exp = relation_to_tree( *dep->rel );
             assert( NULL != dep->src->task );
             printf("         ");
             printf(" -> ");

             cond = simplifyConditions(*dep->rel, copy_tree(rel_exp), S_es);
             if( !cond.empty() )
                 printf("%s ? ", cond.c_str());

             node_t *sink = dep->dst;
             if( NULL == sink ){
                 // EXIT
                 bool need_pseudotask = false;
                 char *comm_mtrx, *refr_mtrx;

                 comm_mtrx = tree_to_str(DA_array_base(dep->src));
                 refr_mtrx = tree_to_str(DA_array_base(reference_data_element));
                 // If the matrices are different and not co-located, we need a pseudo-task.
                 if( strcmp(comm_mtrx,refr_mtrx) && q2j_colocated_map[comm_mtrx].compare(q2j_colocated_map[refr_mtrx]) ){
                     need_pseudotask = true;
                 }else{
                     // If the element we are communicating is not the same as the reference element, we also need a pseudo-task.
                     int count = DA_array_dim_count(dep->src);
                     if( DA_array_dim_count(reference_data_element) != count ){
                         fprintf(stderr,"Matrices with different dimension counts detected \"%s\" and \"%s\"."
                                        " This should never happen in dplasma\n",
                                        tree_to_str(dep->src), tree_to_str(reference_data_element));
                         need_pseudotask = true;
                     }

                     for(int i=0; i<count && !need_pseudotask; i++){
                         char *a = tree_to_str(DA_array_index(dep->src, i));
                         char *b = tree_to_str(DA_array_index(reference_data_element, i));
                         if( strcmp(a,b) )
                             need_pseudotask = true;
                         free(a);
                         free(b);
                     }
                 }
                 free(comm_mtrx);
                 free(refr_mtrx);

                 if( need_pseudotask ){
                     printf("[[ data_%s ]]", tree_to_str(dep->src));
                     char *pseudotask = create_pseudotask(this_task, S_es, *dep->rel, dep->src, var_pseudoname, pseudotask_count++, "out");
                 }else{
                     /*
                      * JDF & QUARK specific optimization:
                      * Add the keyword "data_" infront of the matrix to
                      * differentiate the matrix from the struct.
                      */
                     printf("data_%s",tree_to_str(dep->src));
                 }
             }else{
                 printf("%s %s(",sink->var_symname, sink->task->task_name);
                 printActualParameters(dep, rel_exp, SINK);
                 printf(") ");
             }
             printf("\n");
#ifdef DEBUG_2
             printf("       // ");
             (*dep->rel).print_with_subs(stdout);
#endif
        }

    }

}

