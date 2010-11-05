#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "node_struct.h"
#include "q2j.tab.h"
#include "utility.h"
#include "omega_interface.h"

static var_t *var_head=NULL;
static int _ind_depth=0;
static int _task_count=0;
// For the JDF generation we need to emmit the arrays in FORTRAN notation
// and this "variable" will never need to be changed.  However if we need
// to use the code to generate "C" we might want to set it to false.
int FORTRAN_ARRAY_NOTATION = 1;

static void do_parentize(node_t *node);
static void do_loop_parentize(node_t *node, node_t *enclosing_loop);
static int DA_quark_INOUT(node_t *node);
static node_t *_DA_canonicalize_for_econd(node_t *node, node_t *ivar);
static int is_var_repeating(char *iv_str, char **iv_names);

void dump_und(und_t *und){
    char *name;

    name = DA_var_name( DA_array_base(und->node));
    if( NULL == name )
        return;

    name = tree_to_str(und->node);
    switch( und->rw ){
        case UND_READ:
            printf("%s R",name);
            break;
        case UND_WRITE:
            printf("%s W",name);
            break;
        case UND_RW:
            printf("%s RW",name);
            break;
   }
    
}

void add_variable_use_or_def(node_t *node, int rw, int task_count){
    var_t *var=NULL, *prev=NULL;
    und_t *und;
    node_t *base;
    char *var_name=NULL;

    base = DA_array_base(node);
    if( NULL == base ) return;

    var_name = DA_var_name(base);
    if( NULL == var_name ) return;

    // Look for an existing entry for the array "node"
    prev=var_head;
    for(var=var_head; var != NULL; prev=var, var=var->next){
        if( strcmp(var->var_name, var_name) == 0 ){
            // If we found the array, we look for the Use/Def
            for(und=var->und; NULL!=und->next; und=und->next){
                if( und->node == node ){
                    return; 
                }
            }
            if( und->node == node ){
                 return; 
            }

            // If we didn't find the Use/Def, we create a new
            und->next = (und_t *)calloc(1, sizeof(und_t));
            und = und->next;
            und->rw = rw;
            und->task_num = task_count;
            und->node = node;
            und->next = NULL;
            return;
        }
    }
    // If we didn't find the array, we create a new "var" and a new "und"
    und = (und_t *)calloc(1, sizeof(und_t));
    und->rw = rw;
    und->node = node;
    und->next = NULL;

    var = (var_t *)calloc(1, sizeof(var_t));
    var->und = und;
    var->var_name = var_name;
    var->next = NULL;

    if( NULL == prev ){
        var_head = var;
    }else{
        prev->next = var;
    }
}

und_t **get_variable_uses_and_defs(node_t *node){
    var_t *var=NULL;
    und_t *und;
    und_t **rslt;
    char *var_name=NULL;

    node_t *base = DA_array_base(node);
    if( NULL == base ) return NULL;

    var_name = DA_var_name(base);
    if( NULL == var_name ) return NULL;

    // Look for the entry for the array "node"
    for(var=var_head; NULL != var; var=var->next){
        if( strcmp(var->var_name, var_name) == 0 ){
            int count = 0;
            // If we found the array, we count the Use/Def entries and put them in the return array
            for(und=var->und; NULL != und ; und=und->next) ++count;
            rslt = (und_t **)calloc(count, sizeof(und_t *));
            count = 0;
            for(und=var->und; NULL != und ; und=und->next){
                rslt[count++] = und;
            }
        }
    }
    return rslt;
}

void dump_all_unds(void){
    var_t *var;
    und_t *und;
    node_t *tmp;

    for(var=var_head; NULL != var; var=var->next){
        for(und=var->und; NULL != und ; und=und->next){
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
    }
}

/*********************************************************************************/

static void do_loop_parentize(node_t *node, node_t *enclosing_loop){
    node_t *tmp;
    int depth=0;
    if( (NULL == node) || (EMPTY == node->type) )
        return;

    node->enclosing_loop = enclosing_loop;

    for(tmp=node; tmp->enclosing_loop != NULL; tmp = tmp->enclosing_loop){
        ++depth;
    }
    node->loop_depth = depth;

    if( DA_is_loop(node) ){
        enclosing_loop = node;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            do_loop_parentize(tmp, enclosing_loop);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            do_loop_parentize(node->u.kids.kids[i], enclosing_loop);
        }
    }
}

static void do_parentize(node_t *node){
    if( (NULL == node) || (EMPTY == node->type) )
        return;

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            tmp->parent = node;
            do_parentize(tmp);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            node->u.kids.kids[i]->parent = node;
            do_parentize( node->u.kids.kids[i] );
        }
    }
}

void DA_parentize(node_t node){
    do_parentize(&node);
    do_loop_parentize(&node, NULL);
}

void dump_tree(node_t node, int off){
     _ind_depth = off;
    char *str = tree_to_str(&node);
    printf("%s", str);
    free(str);
    return;
}


static char *numToSymName(int num){
    char str[4] = {0,0,0,0};

    assert(num<2600);

    if( num < 26 ){
        str[0] = 'A'+num;
        str[1] = '\0';
    }else{
        int cnt;
        str[0] = 'A'+num%26;
        cnt = num/26;
        snprintf(&str[1], 3, "%d", cnt);
    }
    return strdup(str);
}


// Turn "CORE_taskname_quark" into "taskname" wasting some memory in the process
char *quark_call_to_task_name( char *call_name ){
    char *task_name, *end;

    if( NULL != strstr(call_name, "CORE_") )
        call_name += 5;

    task_name = strdup(call_name);

    end = strstr(task_name, "_quark");
    if( NULL != end ){
        *end = '\0';
    }

    return task_name;
}


static void record_use_and_defs(node_t *node){
    static int symbolic_name_count = 0;
    int i;


    if( FCALL == node->type ){
        int kid_count;
        task_t *task;

        if( strcmp("QUARK_Insert_Task", DA_kid(node,0)->u.var_name) ){
            return;
        }

        kid_count = node->u.kids.kid_count;

        // QUARK specific code. The task is the second parameter.
        if( (kid_count > 2) && (IDENTIFIER == DA_kid(node,2)->type) ){
            task = (task_t *)calloc(1, sizeof(task_t));
            task->task_name = quark_call_to_task_name( DA_var_name(DA_kid(node,2)) );
            task->task_node = node;
            task->ind_vars = (char **)calloc(1+node->loop_depth, sizeof(char *));
            int i=node->loop_depth-1;
            for(node_t *tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
                task->ind_vars[i] = DA_var_name(DA_loop_induction_variable(tmp));
                --i;
            }
            node->task = task;
        }else{
#if DEBUG
            printf("WARNING: probably there is something wrong with this QUARK_Insert_Task().\n");
#endif
            return;
        }

        for(i=1; i<kid_count; ++i){
            node_t *tmp = node->u.kids.kids[i];

            if( ARRAY == tmp->type ){
                tmp->task = task;
                tmp->var_symname = numToSymName(symbolic_name_count++);
                node_t *qual = node->u.kids.kids[i+1];
                add_variable_use_or_def( tmp, DA_quark_INOUT(qual), _task_count );
            }
        }
        _task_count++;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            record_use_and_defs(tmp);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            record_use_and_defs(node->u.kids.kids[i]);
        }
    }

}

void analyze_deps(node_t *node){
    record_use_and_defs(node);
    interrogate_omega(node, var_head);
}

static int is_var_repeating(char *iv_str, char **iv_names){
    int i;
    for(i=0; NULL != iv_names[i]; ++i){
        if( strcmp(iv_str, iv_names[i]) == 0 )
            return 1;
    }
    return 0;
}

static int is_matching_var(char *iv_str, char *old_var){
    int i;
    int iv_len = strlen(iv_str);
    int old_len = strlen(old_var);

    // if iv_str is not the first part of old_var, they don't match
    if( old_var != strstr(old_var, iv_str) )
        return 0;
    
    // if iv_str and old_len have the same size, they are identical
    if( old_len == iv_len )
        return 1;

    // if any of the remaining letters of old_var are anything but digits, it's not a match
    for( i=iv_len; i<old_len; ++i){
        if( !isdigit(old_var[i]) )
            return 0;
    }

    // If not test failed, it's a match
    return 1;
}

static int var_name_to_num(char *name, int prfx_len){
   int len; 
   len = strlen(name);

   if( prfx_len > len )
       return -1;
   if( prfx_len == len )
       return 0;

   return atoi( (const char *)name[prfx_len] );

}

static void do_rename_ivar(char *iv_str, char *new_name, node_t *node){

    if( IDENTIFIER == node->type ){
       if( strcmp(node->u.var_name, iv_str) == 0 ){
           node->u.var_name = new_name;
       }
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            do_rename_ivar(iv_str, new_name, tmp);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            do_rename_ivar(iv_str, new_name, node->u.kids.kids[i]);
        }
    }
}

static char *rename_ivar(char *iv_str, char **iv_names, node_t *node){
    int i, len, count, num, lg;
    char *new_name;
    len = strlen(iv_str);

    // Count how many variables there are in iv_names
    for(count=0; NULL != iv_names[count]; count++);

    // Look for the last variable of the form "strX" where str==iv_str and X is a number
    for(i=count-1; i>=0; --i){
        if( is_matching_var(iv_str, iv_names[i]) )
            break;
    }

    num = var_name_to_num(iv_names[i], strlen(iv_str));
    // The new var will need to be one higher than the higest existing one
    num += 1;
    // Find the number of digits of the number without paying the cost of a log()
    i = 1;
    for(lg=1; lg<num; lg*=10){
        i++;
    }

    // Create the new variable name
    len = 1+i+len;
    new_name = (char *)calloc(len, sizeof(char));
    snprintf(new_name, len, "%s%d", iv_str, num);

    // Perform the renaming everywhere in the tree
    do_rename_ivar(iv_str, new_name, node);

    // Delete the old name
    free(iv_str);

    return new_name;
}

void rename_induction_variables(node_t *node){
    static int len=0, pos=0;
    static char **iv_names;
    node_t *iv_node;
    char *iv_str;

    if( !len ){
        len = 8;
        iv_names = (char **)calloc(len, sizeof(char *));
    }

    switch( node->type ){
        case FOR: // TODO: we should also deal "while" and "do-while"
            iv_node = DA_loop_induction_variable(node);
            iv_str = DA_var_name(iv_node);

            if( 0 == pos ){
                iv_names[pos] = iv_str;
                pos++;
                break;
            }

            if( is_var_repeating(iv_str, iv_names) ){
                iv_str = rename_ivar(iv_str, iv_names, node);
            }
            // Add the new variable into the list (iv_names)
            if( pos >= len-1 ){
                // The array that holds the list needs to be resized
                len*=2;
                iv_names = (char **)realloc(iv_names, len*sizeof(char *));
            }
            iv_names[pos] = iv_str;
            pos++;
            break;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            rename_induction_variables(tmp);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            rename_induction_variables(node->u.kids.kids[i]);
        }
    }
}


char *append_to_string(char *str, const char *app, const char *fmt, size_t add_length){
    size_t len_str;

    if( NULL == app )
        return str;

    if( NULL == str ){
        len_str = 0;
    }else{
        len_str = strlen(str);
    }

    if( NULL == fmt || 0 == add_length )
        add_length = strlen(app);

    str = (char *)realloc(str, (len_str+add_length+1)*sizeof(char));
    if( NULL == fmt ){
        strcpy( &(str[len_str]), app);
    }else{
        snprintf( &(str[len_str]), add_length+1, fmt, app);
    }

    return str;
}



static node_t *_DA_canonicalize_for_econd(node_t *node, node_t *ivar){
    node_t *tmp;

    if( (IDENTIFIER != DA_rel_lhs(node)->type) && (IDENTIFIER != DA_rel_rhs(node)->type) ){
        printf("Cannot canonicalize for's end condition: ");
        dump_tree(*node, 0);
        printf("\n");
        return NULL;
    }

    // If the variable is on the left hand side
    if( (IDENTIFIER == DA_rel_lhs(node)->type) && !strcmp(ivar->u.var_name, (DA_rel_lhs(node)->u.var_name)) ){
        switch( node->type ){
            case LT:  // since the var is in the left, do nothing, that's the canonical form.
                return node;

            case LE:  // add one to the RHS and convert LE to LT
                tmp = DA_create_B_expr(ADD, DA_rel_rhs(node), DA_create_int_const(1));
                tmp = DA_create_relation(LT, DA_rel_lhs(node), tmp);
                return tmp;
        }
    }else if( (IDENTIFIER == DA_rel_rhs(node)->type) && !strcmp(ivar->u.var_name, (DA_rel_rhs(node)->u.var_name)) ){
        // If the variable is on the RHS, flip the relation operator, exchange LHS and RHS and call myself again.
        tmp = DA_create_relation(DA_flip_rel_op(node->type), DA_rel_rhs(node), DA_rel_lhs(node));
        tmp = _DA_canonicalize_for_econd(tmp, ivar);
        return tmp;
    }

    return NULL;
}


/* TODO: This needs a lot more work to become a general canonicalization function */
int DA_canonicalize_for(node_t *node){
    node_t *ivar, *econd, *tmp;
    if( FOR != node->type){
        return 0;
    }

    // Check the first statement to make sure it's a simple assignment
    tmp = DA_for_scond(node);
    if( ASSIGN != tmp->type ){
        return 0;
    }

    // Extract the lhs of the assignment and make sure it's a simple variable
    ivar = DA_assgn_lhs(tmp);
    if( IDENTIFIER != ivar->type ){
        return 0;
    }

    // Extract the end condition and make sure it's a relation.
    econd = DA_for_econd(node);
    if( !DA_is_rel(econd) ){
        return 0;
    }

    // Canonicalize the end condition (the middle statement of the for)
    tmp = _DA_canonicalize_for_econd(econd, ivar);
    if( NULL == tmp ){
        return 0;
    }
    DA_for_econd(node) = tmp;

    return 1;
}

//#define DA_for_body(_N_) DA_kid((_N_), 3)
//#define DA_for_scond(_N_) DA_kid((_N_), 0)
//#define DA_for_econd(_N_) DA_kid((_N_), 1)
//#define DA_for_incrm(_N_) DA_kid((_N_), 2)

node_t *DA_loop_lb(node_t *node){
   node_t *scond = DA_for_scond(node);
   if( (NULL == scond) || (ASSIGN != scond->type) ){
       return NULL;
   }
   return DA_assgn_rhs(scond);
}

node_t *DA_loop_ub(node_t *node){
   node_t *econd = DA_for_econd(node);
   if( (NULL == econd) || (LT != econd->type) ){
       return NULL;
   }
   return DA_rel_rhs(econd);
}

node_t *DA_create_B_expr(int type, node_t *kid0, node_t *kid1){
    node_t rslt;
    rslt.type = type;
    rslt.u.kids.kid_count = 2;
    rslt.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
    rslt.u.kids.kids[0] = kid0;
    rslt.u.kids.kids[1] = kid1;
    return node_to_ptr(rslt);
}

node_t *DA_create_int_const(int64_t val){
    node_t rslt;
    rslt.type = INTCONSTANT;
    rslt.u.kids.kid_count = 0;
    rslt.u.const_val.i64_value = val;
    return node_to_ptr(rslt);
}

node_t *DA_create_Entry(){
    node_t rslt = {0};
    rslt.type = ENTRY;
//    rslt.task = NULL;
    rslt.u.kids.kid_count = 0;
    return node_to_ptr(rslt);
}

node_t *DA_create_Exit(){
    node_t rslt;
    rslt.type = EXIT;
    rslt.u.kids.kid_count = 0;
    return node_to_ptr(rslt);
}

int DA_flip_rel_op(int type){
    switch(type){
        case LT:
            return GT;
        case LE:
            return GE;
        case GT:
            return LT;
        case GE:
            return LT;
        case EQ_OP:
            return EQ_OP;
        case NE_OP:
            return NE_OP;
    }
    return 0;
}

/*
 * Is the node a Relation node.
 */
int DA_is_rel(node_t *node){
    switch(node->type){
        case LT:
        case LE:
        case GT:
        case GE:
        case EQ_OP:
        case NE_OP:
            return 1;
    }
    return 0;
}

/*
 * Is the node a Structured Control Flow (SCF) node.
 */
int DA_is_scf(node_t *node){
    switch(node->type){
        case BLOCK:
        case SWITCH:
        case WHILE:
        case FOR:
        case DO:
        case IF:
            return 1;
    }
    return 0;
}

/*
 * Is the node a Loop node.
 */
int DA_is_loop(node_t *node){
    switch(node->type){
        case WHILE:
        case FOR:
        case DO:
            return 1;
    }
    return 0;
}

/*
 * If the node is a variable, return it's name.
 */
char *DA_var_name(node_t *node){
    if( NULL == node )
        return NULL;

    switch(node->type){
        case IDENTIFIER:
            return node->u.var_name;
    }
    return NULL;
}

static int DA_quark_INOUT(node_t *node){
    int rslt1, rslt2;
    if( NULL == node )
        return -1;

    switch(node->type){
        case IDENTIFIER:
            if( !strcmp(node->u.var_name, "INPUT") ){
                return UND_READ;
            }
            if( !strcmp(node->u.var_name, "OUTPUT") ){
                return UND_WRITE;
            }
            if( !strcmp(node->u.var_name, "INOUT") ){
                return UND_RW;
            }
            return UND_IGNORE;

        case B_OR:
            rslt1 = DA_quark_INOUT(node->u.kids.kids[0]);
            if( rslt1 < 0 ) return -1;
            rslt2 = DA_quark_INOUT(node->u.kids.kids[1]);
            if( rslt2 < 0 ) return -1;

            return rslt1 | rslt2;

        default:
            fprintf(stderr,"DA_quark_INOUT(): unsupported flag type for dep\n");
            exit(-1);

    }
    return -1;
}

node_t *DA_array_base(node_t *node){
    if( NULL == node || ARRAY != node->type )
        return NULL;
    return node->u.kids.kids[0];
}

int DA_array_dim_count(node_t *node){
    if( NULL == node || ARRAY != node->type )
        return NULL;
    return node->u.kids.kid_count-1;
}

node_t *DA_array_index(node_t *node, int i){
    if( NULL == node || ARRAY != node->type || i<0 || i+1 >= node->u.kids.kid_count )
        return NULL;
    return node->u.kids.kids[i+1];
}


char *DA_type_name(node_t *node){
    char *str=NULL;

    switch(node->type){

        case IDENTIFIER:
            str = strdup("IDENTIFIER");
            break;
        case INTCONSTANT:
            str = strdup("INTCONSTANT");
            break;
        case FLOATCONSTANT:
            str = strdup("FLOATCONSTANT");
            break;
        case STRING_LITERAL:
            str = strdup("STRING_LITERAL");
            break;
        case SIZEOF:
            str = strdup("SIZEOF");
            break;
        case PTR_OP:
            str = strdup("PTR_OP");
            break;
        case INC_OP:
            str = strdup("INC_OP");
            break;
        case DEC_OP:
            str = strdup("DEC_OP");
            break;
        case LEFT_OP:
            str = strdup("LEFT_OP");
            break;
        case RIGHT_OP:
            str = strdup("RIGHT_OP");
            break;
        case LE_OP:
            str = strdup("LE_OP");
            break;
        case GE_OP:
            str = strdup("GE_OP");
            break;
        case EQ_OP:
            str = strdup("EQ_OP");
            break;
        case NE_OP:
            str = strdup("NE_OP");
            break;
        case L_AND:
            str = strdup("L_AND");
            break;
        case L_OR:
            str = strdup("L_OR");
            break;
        case MUL_ASSIGN:
            str = strdup("MUL_ASSIGN");
            break;
        case DIV_ASSIGN:
            str = strdup("DIV_ASSIGN");
            break;
        case MOD_ASSIGN:
            str = strdup("MOD_ASSIGN");
            break;
        case ADD_ASSIGN:
            str = strdup("ADD_ASSIGN");
            break;
        case SUB_ASSIGN:
            str = strdup("SUB_ASSIGN");
            break;
        case LEFT_ASSIGN:
            str = strdup("LEFT_ASSIGN");
            break;
        case RIGHT_ASSIGN:
            str = strdup("RIGHT_ASSIGN");
            break;
        case AND_ASSIGN:
            str = strdup("AND_ASSIGN");
            break;
        case XOR_ASSIGN:
            str = strdup("XOR_ASSIGN");
            break;
        case OR_ASSIGN:
            str = strdup("OR_ASSIGN");
            break;
        case TYPE_NAME:
            str = strdup("TYPE_NAME");
            break;
        case TYPEDEF:
            str = strdup("TYPEDEF");
            break;
        case EXTERN:
            str = strdup("EXTERN");
            break;
        case STATIC:
            str = strdup("STATIC");
            break;
        case AUTO:
            str = strdup("AUTO");
            break;
        case REGISTER:
            str = strdup("REGISTER");
            break;
        case CHAR:
            str = strdup("CHAR");
            break;
        case SHORT:
            str = strdup("SHORT");
            break;
        case INT:
            str = strdup("INT");
            break;
        case LONG:
            str = strdup("LONG");
            break;
        case SIGNED:
            str = strdup("SIGNED");
            break;
        case UNSIGNED:
            str = strdup("UNSIGNED");
            break;
        case FLOAT:
            str = strdup("FLOAT");
            break;
        case DOUBLE:
            str = strdup("DOUBLE");
            break;
        case CONST:
            str = strdup("CONST");
            break;
        case VOLATILE:
            str = strdup("VOLATILE");
            break;
        case VOID:
            str = strdup("VOID");
            break;
        case STRUCT:
            str = strdup("STRUCT");
            break;
        case UNION:
            str = strdup("UNION");
            break;
        case ENUM:
            str = strdup("ENUM");
            break;
        case ELLIPSIS:
            str = strdup("ELLIPSIS");
            break;
        case CASE:
            str = strdup("CASE");
            break;
        case DEFAULT:
            str = strdup("DEFAULT");
            break;
        case IF:
            str = strdup("IF");
            break;
        case ELSE:
            str = strdup("ELSE");
            break;
        case SWITCH:
            str = strdup("SWITCH");
            break;
        case WHILE:
            str = strdup("WHILE");
            break;
        case DO:
            str = strdup("DO");
            break;
        case FOR:
            str = strdup("FOR");
            break;
        case GOTO:
            str = strdup("GOTO");
            break;
        case CONTINUE:
            str = strdup("CONTINUE");
            break;
        case BREAK:
            str = strdup("BREAK");
            break;
        case RETURN:
            str = strdup("RETURN");
            break;

        case EMPTY:
            str = strdup("EMPTY");
            break;
        case ADDR_OF:
            str = strdup("ADDR_OF");
            break;
        case STAR:
            str = strdup("STAR");
            break;
        case PLUS:
            str = strdup("PLUS");
            break;
        case MINUS:
            str = strdup("MINUS");
            break;
        case TILDA:
            str = strdup("TILDA");
            break;
        case BANG:
            str = strdup("BANG");
            break;
        case ASSIGN:
            str = strdup("ASSIGN");
            break;
        case COND:
            str = strdup("COND");
            break;
        case ARRAY:
            str = strdup("ARRAY");
            break;
        case FCALL:
            str = strdup("FCALL");
            break;
        case EXPR:
            str = strdup("EXPR");
            break;
        case ADD:
            str = strdup("ADD");
            break;
        case SUB:
            str = strdup("SUB");
            break;
        case MUL:
            str = strdup("MUL");
            break;
        case DIV:
            str = strdup("DIV");
            break;
        case MOD:
            str = strdup("MOD");
            break;
        case B_AND:
            str = strdup("B_AND");
            break;
        case B_XOR:
            str = strdup("B_XOR");
            break;
        case B_OR:
            str = strdup("B_OR");
            break;
        case LSHIFT:
            str = strdup("LSHIFT");
            break;
        case RSHIFT:
            str = strdup("RSHIFT");
            break;
        case LT:
            str = strdup("LT");
            break;
        case GT:
            str = strdup("GT");
            break;
        case LE:
            str = strdup("LE");
            break;
        case GE:
            str = strdup("GE");
            break;
        case DEREF:
            str = strdup("DEREF");
            break;
        case S_U_MEMBER:
            str = strdup("S_U_MEMBER");
            break;
        case COMMA_EXPR:
            str = strdup("COMMA_EXPR");
            break;
        case BLOCK:
            str = strdup("BLOCK");
            break;
    }

    return str;
}


node_t *DA_loop_induction_variable(node_t *loop){
    node_t *n0, *n1, *n2, *tmp;

    switch(loop->type){
        case WHILE:
        case DO:
            return NULL;
        case FOR:
            n0 = DA_for_scond(loop);
            n1 = DA_for_econd(loop);
            n2 = DA_for_incrm(loop);
            assert( (NULL != n0) && (NULL != n1) && (NULL != n2) );
            if( ASSIGN != n0->type ){
                fprintf(stderr,"Don't know how to extract induction variable from type: %s\n",DA_type_name(n0));
                return NULL;
            }
            tmp = DA_assgn_lhs(n0);
            if( IDENTIFIER != tmp->type ){
                fprintf(stderr,"Don't know how to deal with LHS of type: %s\n",DA_type_name(tmp));
                return NULL;
            }

            return tmp;
            
        default:
            return NULL;
    }
}

void dump_for(node_t *node){
    int i;
    int kid_count = node->u.kids.kid_count;

    if( FOR == node->type ){
        char *str = strdup("for( ");
        for(i=0; i<kid_count-1; ++i){
            if(i>0)
                str = append_to_string( str, "; ", NULL, 0);
            str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0 );
        }
        printf("%s\n", append_to_string( str, ")", NULL, 0) );
    }
}


const char *type_to_symbol(int type){
    switch(type){
        case ADD:
            return "+";
        case SUB:
            return "-";
        case MUL:
            return "*";
        case DIV:
            return "/";
        case MOD:
            return "%";
        case B_AND:
            return "&";
        case B_XOR:
            return "^";
        case B_OR:
            return "|";
        case L_AND:
            return "&&";
        case L_OR:
            return "||";
        case LSHIFT:
            return "<<";
        case RSHIFT:
            return ">>";
        case LT:
            return "<";
        case GT:
            return ">";
        case LE:
            return "<=";
        case GE:
            return ">=";
        case EQ_OP:
            return "==";
        case NE_OP:
            return "!=";
        case COMMA_EXPR:
            return ",";
    }
    return "???";
}

int isSimpleVar(char *name){
    int i, len;

    if( NULL == name )
        return 0;

    len = strlen(name);
    for(i=0; i<len; ++i){
        if( !isalnum(name[i]) && ('_' != name[i]) )
            return 0;
    }

    // If we found no operators, then it's a simple variable
    return 1;
}

char *tree_to_str(node_t *node){
    int i, kid_count;
    char prfx[16], *str=NULL;
//    node_t *node_tmp;

    if( NULL == node )
        return strdup("nil");
    if( EMPTY == node->type )
        return NULL;

    kid_count = node->u.kids.kid_count;

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            char *tmp_str;
            char *ws = (char *)calloc(_ind_depth+1, sizeof(char));
            sprintf(ws, "%*s", _ind_depth, " ");
            str = append_to_string(str, ws, NULL, 0);
            free(ws);

            tmp_str = tree_to_str(tmp);
            if( DA_is_scf(tmp ) )
                str = append_to_string(str, tmp_str, "%s\n", 1+strlen(tmp_str) );
            else
                str = append_to_string(str, tmp_str, "%s;\n", 2+strlen(tmp_str) );
        }
        return str;
    }else{
        char *tmp, *lhs, *rhs;
        int j, max_arg_len[4];

        switch( node->type ){
            case IDENTIFIER:
                return strdup(node->u.var_name);

            case INTCONSTANT:
                tmp = (char *)calloc(24,sizeof(char));
                snprintf(tmp, 24, "%"PRIu64, node->u.const_val.i64_value);
                return tmp;

            case FLOATCONSTANT: 
                tmp = (char *)calloc(32,sizeof(char));
                snprintf(tmp, 32, "%lf", node->u.const_val.f64_value);
                return tmp;

            case STRING_LITERAL:
                return strdup(node->u.const_val.str);

            case INC_OP:
                return strdup("++");

            case SIZEOF:
                str = strdup("sizeof(");
                if(node->u.kids.kid_count ){
                    str = append_to_string( str, tree_to_str(node->u.kids.kids[0]), NULL, 0 );
                }else{
                    str = append_to_string( str, node->u.var_name, NULL, 0 );
                }
                str = append_to_string( str, ")", NULL, 0 );
                return str;

            case EXPR:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

	    case ADDR_OF:
                return strdup("&");
	    case STAR:
                return strdup("*");
	    case PLUS:
                return strdup("+");
	    case MINUS:
                return strdup("-");
	    case TILDA:
                return strdup("~");
	    case BANG:
                return strdup("!");

            case ADD:
            case SUB:
            case MUL:
            case DIV:
            case MOD:
            case B_AND:
            case B_XOR:
            case B_OR:
            case L_AND:
            case L_OR:
            case LSHIFT:
            case RSHIFT:
            case LT:
            case GT:
            case LE:
            case GE:
            case COMMA_EXPR:
                lhs = tree_to_str(node->u.kids.kids[0]);
                rhs = tree_to_str(node->u.kids.kids[1]);

                if( isSimpleVar(lhs) ){
                    str = lhs;
                }else{
                    str = strdup("(");
                    str = append_to_string( str, lhs, NULL, 0 );
                    str = append_to_string( str, ")", NULL, 0 );
                }

                str = append_to_string( str, type_to_symbol(node->type), NULL, 0 );

                if( isSimpleVar(rhs) ){
                    str = append_to_string( str, rhs, NULL, 0 );
                }else{
                    str = append_to_string( str, "(", NULL, 0 );
                    str = append_to_string( str, rhs, NULL, 0 );
                    str = append_to_string( str, ")", NULL, 0 );
                }

                return str;


            case ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " = ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case MUL_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " *= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case DIV_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " /= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case MOD_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " %= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case ADD_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " += ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case SUB_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " -= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case LEFT_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " <<= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case RIGHT_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " >>= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case AND_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " &= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case XOR_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " ^= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case OR_ASSIGN:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, " |= ", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case S_U_MEMBER:
                str = tree_to_str(node->u.kids.kids[0]);
                // We ommit the "." to make the struct+member look like a single variable for
                // the JDF->release_dep parser to be able to deal with it.
                // str = append_to_string( str, ".", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case PTR_OP:
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, "->", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                return str;

            case COND:
                str = strdup("(");
                str = append_to_string( str, tree_to_str(node->u.kids.kids[0]), NULL, 0 );
                str = append_to_string( str, ") ? (", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                str = append_to_string( str, ") : (", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[2]), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0 );
                return str;

            case EQ_OP:
                str = strdup("(");
                str = append_to_string( str, tree_to_str(node->u.kids.kids[0]), NULL, 0 );
                str = append_to_string( str, ")==(", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0);
                return str;

            case NE_OP:
                str = strdup("(");
                str = append_to_string( str, tree_to_str(node->u.kids.kids[0]), NULL, 0 );
                str = append_to_string( str, ")!=(", NULL, 0 );
                str = append_to_string( str, tree_to_str(node->u.kids.kids[1]), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0);
                return str;

            case FOR:
                str = strdup("for( ");
                for(i=0; i<kid_count-1; ++i){
                    if(i>0)
                        str = append_to_string( str, "; ", NULL, 0);
                    str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0 );
                }
                str = append_to_string( str, ") {\n", NULL, 0);
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str(DA_for_body(node)), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}\n", NULL, 0);
                return str;

            case WHILE:
                str = strdup("while( ");
                str = append_to_string( str, tree_to_str(DA_while_cond(node)), NULL, 0 );
                str = append_to_string( str, " ) {\n", NULL, 0);
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str(DA_while_body(node)), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}\n", NULL, 0);
                return str;

            case DO:
                str = strdup("do{\n");
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str(DA_do_body(node)), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}while( ", NULL, 0);
                str = append_to_string( str, tree_to_str(DA_do_cond(node)), NULL, 0 );
                str = append_to_string( str, " );\n", NULL, 0);
                return str;

            case FCALL:
                for(j=1; j<=3; j++){
                    max_arg_len[j] = -1;
                    for(i=j; i<node->u.kids.kid_count; i+=3){
                        int tmp;
                        char *arg = tree_to_str(node->u.kids.kids[i]);
                    
                        tmp = strlen(arg);
                        free(arg);
                        if( tmp > max_arg_len[j] )
                            max_arg_len[j] = tmp;
                    }
                }
                str = tree_to_str(node->u.kids.kids[0]);
                str = append_to_string( str, "( ", NULL, 0);
                for(i=1; i<node->u.kids.kid_count; ++i){
                    char fmt[32];
                    if( i > 1 )
                        str = append_to_string( str, ", ", NULL, 0);
                    if( (i>1) && ((i-1)%3 == 0) ){
                        char *ws = (char *)calloc(_ind_depth+4+1, sizeof(char));
                        sprintf(ws, "\n%*s", _ind_depth+4, " ");
                        str = append_to_string(str, ws, NULL, 0);
                        free(ws);
                    }
                    if( i > 3 ){
                        int len = max_arg_len[1+((i-1)%3)];
                        memset(fmt,0,32*sizeof(char));
                        sprintf(fmt,"%%-%ds",len);
                        str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), fmt, len+1 );
                    }else{
                        str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0);
                    }
                }
                str = append_to_string( str, " )", NULL, 0);
                return str;

            case ARRAY:
                str = tree_to_str(node->u.kids.kids[0]);
                if( FORTRAN_ARRAY_NOTATION ){
                    str = append_to_string( str, "(", NULL, 0);
                    for(i=1; i<node->u.kids.kid_count; ++i){
                        if( i > 1 ) 
                            str = append_to_string( str, ",", NULL, 0);
                        str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0 );
                    }
                    str = append_to_string( str, ")", NULL, 0);
                }else{
                    for(i=1; i<node->u.kids.kid_count; ++i){
                        str = append_to_string( str, "[", NULL, 0);
                        str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0 );
                        str = append_to_string( str, "]", NULL, 0);
                    }

                }
                return str;


            default:
                snprintf(prfx, 12, "|>%d<| ", node->type);
                str = append_to_string(NULL, prfx, NULL, 0);
                snprintf(prfx, 15, "kid_count: %d {{", kid_count);
                str = append_to_string(str, prfx, NULL, 0);
                _ind_depth += 4;
                for(i=0; i<kid_count; ++i){
                    if( i > 0 )
                        str = append_to_string( str, " ## ", NULL, 0 );
                    str = append_to_string( str, tree_to_str(node->u.kids.kids[i]), NULL, 0 );
                }
                _ind_depth -= 4;
                str = append_to_string( str, "}}", NULL, 0 );
                return str;
        }
    }

    return NULL;
}


node_t *node_to_ptr(node_t node){
    node_t *tmp = (node_t *)calloc(1, sizeof(node_t));
    *tmp = node;
    return tmp;
}

