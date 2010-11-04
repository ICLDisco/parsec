#include "node_struct.h"
#include "utility.h"
#include "q2j.tab.h"
#include "omega_interface.h"
#include <map>
#include <set>
#include <list>
#include <sstream>

#define DEP_FLOW  0x1
#define DEP_OUT   0x2

#define EDGE_INCOMING 0x0
#define EDGE_OUTGOING 0x1

#define LBOUND  0x0
#define UBOUND  0x1

#define SOURCE  0x0
#define SINK    0x1

extern void dump_und(und_t *und);
static void dump_full_und(und_t *und);

static void process_end_condition(node_t *node, F_And *&R_root, map<string, Variable_ID> ivars, map<string, Free_Var_Decl *> globals, Relation &R);
static void print_edges(set<dep_t *>deps, int edge_type);
static void print_pseudo_variables(set<dep_t *>out_deps, set<dep_t *>in_deps);
static void print_execution_space(node_t *node);
static inline set<expr_t *> findAllEQsWithVar(const char *var_name, expr_t *exp);
static inline set<expr_t *> findAllGEsWithVar(const char *var_name, expr_t *exp);
static set<expr_t *> findAllConstraintsWithVar(const char *var_name, expr_t *exp, int constr_type);
static const char *expr_tree_to_str(const expr_t *exp);
static int treeContainsVar(expr_t *root, const char *var_name);
static expr_t *solveDirectlySolvableEQ(expr_t *exp, const char *var_name, Relation R);
static void substituteExprForVar(expr_t *exp, const char *var_name, expr_t *root);

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

void declare_globals_in_tree(node_t *node, set <char *> ind_names, map<string, Free_Var_Decl *> &globals){
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
            it = globals.find(var_name);
            if( it == globals.end() ){
                globals[var_name] = new Free_Var_Decl(var_name);
            }

            return;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            declare_globals_in_tree(tmp, ind_names, globals);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            declare_globals_in_tree(node->u.kids.kids[i], ind_names, globals);
        }
    }

    return;
}



void expr_to_Omega_coef(node_t *node, Constraint_Handle &handle, int sign, map<string, Variable_ID> vars, map<string, Free_Var_Decl *> globals, Relation &R){
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
            g_it = globals.find(var_name);
            if( g_it != globals.end() ){
                Variable_ID v = R.get_local(g_it->second);
                handle.update_coef(v,sign);
                break;
            }
            fprintf(stderr,"expr_to_Omega_coef(): Can't find \"%s\" in either induction, or global variables.\n", DA_var_name(node) );
            exit(-1);
        case ADD:
            expr_to_Omega_coef(node->u.kids.kids[0], handle, sign, vars, globals, R);
            expr_to_Omega_coef(node->u.kids.kids[1], handle, sign, vars, globals, R);
            break;
        case SUB:
            expr_to_Omega_coef(node->u.kids.kids[0], handle, sign, vars, globals, R);
            expr_to_Omega_coef(node->u.kids.kids[1], handle, -sign, vars, globals, R);
            break;
        default:
            fprintf(stderr,"expr_to_Omega_coef(): Can't turn type \"%d (%x)\" into Omega expression.\n",node->type, node->type);
            exit(-1);
    }
    return;
}


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

void add_array_subscript_equalities(Relation &R, F_And *R_root, map<string, Variable_ID> ivars, map<string, Variable_ID> ovars, map<string, Free_Var_Decl *> globals, node_t *def, node_t *use){
    int i,count;

    count = DA_array_dim_count(def);
    if( DA_array_dim_count(use) != count ){
        fprintf(stderr,"add_array_subscript_equalities(): ERROR: Arrays in USE and DEF do not have the same number of subscripts.");
    }

    for(i=0; i<count; i++){
        node_t *iv = DA_array_index(def, i);
        node_t *ov = DA_array_index(use, i);
        EQ_Handle hndl = R_root->add_EQ();
        expr_to_Omega_coef(iv, hndl, 1, ivars, globals, R);
        expr_to_Omega_coef(ov, hndl, -1, ovars, globals, R);
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
void process_end_condition(node_t *node, F_And *&R_root, map<string, Variable_ID> ivars, map<string, Free_Var_Decl *> globals, Relation &R){
    Variable_ID ivar;
    GEQ_Handle imax;
    F_And *new_and;

    switch( node->type ){
        case L_AND:
            new_and = R_root->add_and();
            process_end_condition(node->u.kids.kids[0], new_and, ivars, globals, R);
            process_end_condition(node->u.kids.kids[1], new_and, ivars, globals, R);
            break;
// TODO: handle logical or (L_OR) as well.
//F_Or *or1 = R_root->add_or();
//F_And *and1 = or1->add_and();
        case LT:
            ivar = ivars[DA_var_name(DA_rel_lhs(node))];
            imax = R_root->add_GEQ();
            expr_to_Omega_coef(DA_rel_rhs(node), imax, 1, ivars, globals, R);
            imax.update_coef(ivar,-1);
            imax.update_const(-1);
            break;
        case LE:
            ivar = ivars[DA_var_name(DA_rel_lhs(node))];
            imax = R_root->add_GEQ();
            expr_to_Omega_coef(DA_rel_rhs(node), imax, 1, ivars, globals, R);
            imax.update_coef(ivar,-1);
            break;
        default:
            fprintf(stderr,"ERROR: process_end_condition() cannot deal with node of type: %d\n",node->type);
            exit(-1);
    }

}

Relation create_exit_relation(node_t *exit, node_t *def, map<string, Free_Var_Decl *> globals){
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
        expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ivars, globals, R);

        // Form the Omega expression for the upper bound
        process_end_condition(DA_for_econd(tmp), R_root, ivars, globals, R);
    }

    // Add equalities between corresponding input and output array indexes
    int count = DA_array_dim_count(def);
    for(i=0; i<count; i++){
        node_t *iv = DA_array_index(def, i);
        EQ_Handle hndl = R_root->add_EQ();

        hndl.update_coef(R.output_var(i+1), 1);
        expr_to_Omega_coef(iv, hndl, -1, ivars, globals, R);
    }

    R.simplify();
    return R;
}

map<node_t *, Relation> create_entry_relations(node_t *entry, var_t *var, int dep_type, map<string, Free_Var_Decl *> globals){
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
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ovars, globals, R);

            // Form the Omega expression for the upper bound
            process_end_condition(DA_for_econd(tmp), R_root, ovars, globals, R);
        }

        // Add equalities between corresponding input and output array indexes
        int count = DA_array_dim_count(use);
        for(i=0; i<count; i++){
            node_t *ov = DA_array_index(use, i);
            EQ_Handle hndl = R_root->add_EQ();

            hndl.update_coef(R.input_var(i+1), 1);
            expr_to_Omega_coef(ov, hndl, -1, ovars, globals, R);
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
map<node_t *, Relation> create_dep_relations(und_t *def_und, var_t *var, int dep_type, map<string, Free_Var_Decl *> globals, node_t *exit_node){
    int i, after_def = 0;
    int src_var_count, dst_var_count;
    und_t *und;
    node_t *tmp, *def, *use;
    char **def_ind_names;
    char **use_ind_names;
    map<string, Variable_ID> ivars, ovars;
    map<node_t *, Relation> dep_edges;

    def = def_und->node;
    src_var_count = def->loop_depth;
    after_def = 0;
    for(und=var->und; NULL != und ; und=und->next){
        char *var_name, *def_name;

        var_name = DA_var_name(DA_array_base(und->node));
        def_name = DA_var_name(DA_array_base(def));
        assert( !strcmp(var_name, def_name) );

        if( ((DEP_FLOW==dep_type) && (!is_und_read(und))) || ((DEP_OUT==dep_type) && (!is_und_write(und))) ){
            // Since we'll bail, let's first check if this is the definition.
            if( und == def_und ){
                after_def = 1;
            }
            continue;
        }
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
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ivars, globals, R);

            // Form the Omega expression for the upper bound
            process_end_condition(DA_for_econd(tmp), R_root, ivars, globals, R);
        }

        // Bound all induction variables of the loops enclosing the USE
        // using the loop bounds
        for(tmp=use->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
            // Form the Omega expression for the lower bound
            Variable_ID ovar = ovars[DA_var_name(DA_loop_induction_variable(tmp))];

            GEQ_Handle imin = R_root->add_GEQ();
            imin.update_coef(ovar,1);
            expr_to_Omega_coef(DA_loop_lb(tmp), imin, -1, ovars, globals, R);

            // Form the Omega expression for the upper bound
            process_end_condition(DA_for_econd(tmp), R_root, ovars, globals, R);
        }


        // Add inequalities of the form (m'>=m || n'>=n || ...) or the form (m'>m || n'>n || ...) if the DU chain is
        // "normal", or loop carried, respectively. The outermost enclosing loop HAS to be k'>=k, it is not
        // part of the "or" conditions.  In the loop carried deps, the outer most loop is ALSO in the "or" conditions,
        // so we have (m'>m || n'>n || k'>k) && k'>=k
        // In addition, if a flow edge is going from a task to itself (because the def and use seemed to be in different
        // lines) it also needs to have greater-than instead of greater-or-equal relationships for the induction variables.

        node_t *encl_loop = find_closest_enclosing_loop(use, def);

        // If USE and DEF are in the same task and that is not in a loop, there is no data flow, go to the next USE.
        if( (NULL == encl_loop) && (def->task == use->task) ){
            continue;
        }

        if( after_def && (def->task != use->task) ){
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

        // Add equalities demanded by the array subscripts. For example is the DEF is A[k][k] and the
        // USE is A[m][n] then add (k=m && k=n).
        add_array_subscript_equalities(R, R_root, ivars, ovars, globals, def, use);

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
        dep_edges[exit_node] = create_exit_relation(exit_node, def, globals);
    }

    return dep_edges;
}


map<string, Free_Var_Decl *> declare_globals(node_t *node){
    map<string, Free_Var_Decl *> globals, tmp_map;
    node_t *tmp;


    if( FOR == node->type ){
        set <char *> ind_names;

        // Store the names of the induction variables of all the loops that enclose this loop
        for(tmp=node; NULL != tmp; tmp=tmp->enclosing_loop ){
            ind_names.insert( DA_var_name(DA_loop_induction_variable(tmp)) );
        }

        // Find all the variables in the lower bound that are not induction variables and
        // declare them as global variables (symbolic)
        declare_globals_in_tree(DA_loop_lb(node), ind_names, globals);

        // Find all the variables in the upper bound that are not induction variables and
        // declare them as global variables (symbolic)
        declare_globals_in_tree(DA_for_econd(node), ind_names, globals);
    }


    if( BLOCK == node->type ){
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            tmp_map = declare_globals(tmp);
            globals.insert(tmp_map.begin(), tmp_map.end());
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            tmp_map = declare_globals(node->u.kids.kids[i]);
            globals.insert(tmp_map.begin(), tmp_map.end());
        }
    }

    return globals;
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
        for(int i=0; i<R.n_out(); i++){
            const char *ovar = R.output_var(i+1)->char_name();
            if( treeContainsVar(rslt_exp, ovar) ){
                exp_has_output_vars = 1;
                break;
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
        ss << lb;
        free(lb);
    }else{
        ss << "??";
    }

    ss << "..";

    if( NULL != ub ){
        ss << ub;
        free(ub);
    }else{
        ss << "??";
    }

    return ss.str().c_str();
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

    // Add all the combinations off LBs and UBs in the expression
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
    while( 1 ){
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
        // the other equations first that are solvable and replace these.
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

////////////////////////////////////////////////////////////////////////////////
//
// WARNING: this function it destructive.  It actually removes nodes from the tree
// and deletes them altogether. In many cases you will need to pass a copy of the
// tree to this function.
void printConditions(Relation R, expr_t *exp){
    set<expr_t *> conj;

    conj = findAllConjunctions(exp);

    // If we didn't find any it's because there are no unions, so the whole
    // expression "exp" is one conjunction
    if( conj.empty() )
        conj.insert(exp);

    printf("(");
    if( conj.size() > 1 )
        printf(" (");
    set<expr_t *>::iterator cj_it;
    for(cj_it = conj.begin(); cj_it != conj.end(); cj_it++){
        expr_t *cur_exp = *cj_it;

        if( cj_it != conj.begin() )
            printf(") || (");
    
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
        printf("%s",expr_tree_to_str(cur_exp));
    }
    printf(")");

    return;
}


////////////////////////////////////////////////////////////////////////////////
//
void interrogate_omega(node_t *root, var_t *head){
    var_t *var;
    und_t *und;
    map<node_t *, map<node_t *, Relation> > flow_sources, output_sources;
    map<string, Free_Var_Decl *> globals;
    map<char *, set<dep_t *> > outgoing_edges;
    map<char *, set<dep_t *> > incoming_edges;

    globals = declare_globals(root);

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
        // Create flow edges starting from the ENTRY
        flow_sources[entry]   = create_entry_relations(entry, var, DEP_FLOW, globals);
        output_sources[entry] = create_entry_relations(entry, var, DEP_OUT, globals);

        // For each DEF create all flow and output edges
        for(und=var->und; NULL != und ; und=und->next){
            if(is_und_write(und)){
                node_t *def = und->node;
                flow_sources[def] = create_dep_relations(und, var, DEP_FLOW, globals, exit_node);
                output_sources[def] = create_dep_relations(und, var, DEP_OUT, globals, exit_node);
            }
        }

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

                        rKill = Composition(copy(fd1_r), copy(od_r));
                        rKill.simplify();
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

                        rKill = Composition(copy(fd2_r), copy(od_r));
                        rKill.simplify();
                    }
                    if( rAllKill.is_null() || rAllKill.is_set() ){
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
                    rReal = Difference(fd1_r, rAllKill);
                }

                rReal.simplify();
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

            print_execution_space(src_task->task_node);
            printf("\n");
            print_pseudo_variables(deps, incoming_edges[task_name]);
            printf("\n");
            print_edges(deps, EDGE_OUTGOING);
            deps = incoming_edges[task_name];
            print_edges(deps, EDGE_INCOMING);
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
            return ss.str().c_str();
// TODO: handle logical or (L_OR) as well.

        case LE:
            ss << tree_to_str(DA_rel_rhs(econd));
            return ss.str().c_str();

        case LT:
            ss << tree_to_str(DA_rel_rhs(econd)) << "-1";
            return ss.str().c_str();

        default:
            fprintf(stderr,"ERROR: econd_tree_to_ub() cannot deal with node of type: %d\n",econd->type);
            exit(-1);
    }
}

static void print_execution_space(node_t *node){
    node_t *tmp;
    list<node_t *> params;

    printf("  /* Execution space */\n");
    for(tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        params.push_front(tmp);
    }

    while( !params.empty() ) {
        tmp = params.front();
        printf("  %s = ", DA_var_name(DA_loop_induction_variable(tmp)) );
        printf("%s..", tree_to_str(DA_loop_lb(tmp)));
        printf("%s\n", econd_tree_to_ub(DA_for_econd(tmp)));
        params.pop_front();
    }
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
       printf("  /* %s == %s */\n",(pvit->first).c_str(), (pvit->second).c_str());
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

const char *expr_tree_to_str(const expr_t *exp){
    stringstream ss, ssL, ssR;
    unsigned int j, skipSymbol=0, first=1;
    unsigned int r_needs_paren = 0, l_needs_paren = 0;
    set<expr_t *> pos, neg;
    set<expr_t *>::iterator it;
    string str;

    if( NULL == exp )
        return "";

    switch( exp->type ){
        case IDENTIFIER:
            // convert all structure members (A.m) into variables Am
            str = string(exp->value.name);
            while ( (j=str.find(".")) != std::string::npos) { 
                str.replace( j, 1, "" ); 
            } 
            return strdup(str.c_str());

        case INTCONSTANT:
            if( exp->value.int_const < 0 )
                ss << "(" << exp->value.int_const << ")";
            else
                ss << exp->value.int_const;
            return ss.str().c_str();

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
                        ssL << expr_tree_to_str(e->r);
                    }else{
                        // in this case we can skip the "one" even for divisions
                        if( labs(e->r->value.int_const) != 1 ){
                            ssL << labs(e->r->value.int_const);
                            ssL << type_to_symbol(e->type);
                        }
                        ssL << expr_tree_to_str(e->l);
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
                        ssR << expr_tree_to_str(e->r);
                    }else{
                        if( labs(e->r->value.int_const) != 1 ){
                            ssR << labs(e->r->value.int_const);
                            ssR << type_to_symbol(e->type);
                        }
                        ssR << expr_tree_to_str(e->l);
                    }
                }
            }
            if( 0 == neg.size() ){
                ssR << "0";
            }

            if( ssL.str().compare( ssR.str() ) ){
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

                // The following strdup() seems to be necessary. If ommited, some c++ compilers
                // generate code that corrupts the returned string. However, it creates a memory leak.
                return strdup(ss.str().c_str());
            }else{
                return NULL;
            }

        case L_AND:
        case L_OR:

            assert( (NULL != exp->l) && (NULL != exp->r) );

            // If one of the two sides is a tautology, we will get a NULL string
            // and we have no reason to print the "&&" or "||" sign.
            if( NULL == expr_tree_to_str(exp->l) )
                return expr_tree_to_str(exp->r);
            if( NULL == expr_tree_to_str(exp->r) )
                return expr_tree_to_str(exp->l);

            if( L_OR == exp->type )
                ss << "(";
            ss << expr_tree_to_str(exp->l);
            ss << " " << type_to_symbol(exp->type) << " ";
            ss << expr_tree_to_str(exp->r);
            if( L_OR == exp->type )
                ss << ")";
            return ss.str().c_str();

        case ADD:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            if( (NULL != exp->l) && (INTCONSTANT == exp->l->type) ){
                if( exp->l->value.int_const != 0 ){
                    ss << "(";
                    ss << expr_tree_to_str(exp->l);
                }else{
                    skipSymbol = 1;
                }
            }else{
                ss << expr_tree_to_str(exp->l);
            }

            if( !skipSymbol )
                ss << type_to_symbol(ADD);

            ss << expr_tree_to_str(exp->r);

            if( !skipSymbol )
                ss << ")";

            return ss.str().c_str();

        case MUL:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            if( INTCONSTANT == exp->l->type ){
                if( exp->l->value.int_const != 1 ){
                    ss << "(";
                    ss << expr_tree_to_str(exp->l);
                }else{
                    skipSymbol = 1;
                }
            }else{
                ss << expr_tree_to_str(exp->l);
            }

            if( !skipSymbol )
                ss << type_to_symbol(MUL);

            ss << expr_tree_to_str(exp->r);

            if( !skipSymbol )
                ss << ")";

            return ss.str().c_str();

        case DIV:
            assert( (NULL != exp->l) && (NULL != exp->r) );

            ss << "(";
            if( INTCONSTANT == exp->l->type ){
                if( exp->l->value.int_const < 0 ){
                    ss << "(" << expr_tree_to_str(exp->l) << ")";
                }else{
                    ss << expr_tree_to_str(exp->l);
                }
            }else{
                ss << expr_tree_to_str(exp->l);
            }

            ss << type_to_symbol(DIV);

            if( (INTCONSTANT == exp->r->type) && (exp->r->value.int_const > 0) )
                ss << expr_tree_to_str(exp->r);
            else
                ss << "(" << expr_tree_to_str(exp->r) << ")";

            ss << ")";
            return ss.str().c_str();

        default:
            ss << "{" << exp->type << "}";
            return ss.str().c_str();
    }
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
//
void print_edges(set<dep_t *>deps, int edge_type){
    task_t *src_task;

    if( deps.empty() ){
        return;
    }

    src_task = (*deps.begin())->src->task;

    set<dep_t *>::iterator it;

/*
    // Group the edges
    for (it=deps.begin(); it!=deps.end(); it++){
           dep->src->var_symname;
    }
*/

    for (it=deps.begin(); it!=deps.end(); it++){
       dep_t *dep = *it;
       expr_t *rel_exp;

       // WARNING: The following call to print_with_subs_to_string(false) is
       // necessary for Omega to build its internal structures and name arrays
       // and whatnot.  If you remove it you get strange assert() calls being
       // triggered.
       (void)(*dep->rel).print_with_subs_to_string(false);

       rel_exp = relation_to_tree( *dep->rel );
//       printf("  ## %s\n",expr_tree_to_str(rel_exp));

       if( EDGE_OUTGOING == edge_type ){
           assert( NULL != dep->src->task );
           printf("  %s -> ", dep->src->var_symname);

           printConditions(*dep->rel, copy_tree(rel_exp));
           printf(" ? ");

           node_t *sink = dep->dst;
           if( NULL == sink ){
//               printf("EXIT  ");
               printf("%s",tree_to_str(dep->src));
           }else{
               printf("%s %s(",sink->var_symname, sink->task->task_name);
               printActualParameters(dep, rel_exp, SINK);
               printf(") ");
           }
       }else if( EDGE_INCOMING == edge_type ){
           assert( NULL != dep->dst);
           printf("  %s <- ",dep->dst->var_symname);

           task_t *src_task = dep->src->task;


           printConditions(*dep->rel, copy_tree(rel_exp));
           printf(" ? ");

           if( NULL != src_task ){

               printf("%s ", dep->src->var_symname);
               printf("%s(",src_task->task_name);
               printActualParameters(dep, rel_exp, SOURCE);
               printf(") ");
           }else{
//               printf("ENTRY ");
               printf("%s", tree_to_str(dep->dst));
           }
       }else{
           fprintf(stderr,"ERROR: print_edges() unknown edge type: %d\n",edge_type);
           exit(-1);
       }
       printf("\n");

//#ifdef DEBUG_2
//       printf("       ");
//       (*dep->rel).print_with_subs(stdout);
//       (*dep->rel).print();
//       printf("\n");
//#endif
    }
}

