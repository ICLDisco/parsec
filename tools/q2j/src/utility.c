/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdarg.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "jdf.h"
#include "string_arena.h"

#include "node_struct.h"
#include "q2j.y.h"
#include "utility.h"
#include "omega_interface.h"

#define QUARK_FIRST_VAR 5
#define QUARK_ELEMS_PER_LINE 3

#define TASK_IN  0
#define TASK_OUT 1

#define IVAR_NOT_FOUND 0
#define IVAR_IS_LEFT   1
#define IVAR_IS_RIGHT -1

extern char *q2j_input_file_name;
extern char *_q2j_data_prefix;
extern int _q2j_generate_line_numbers;
extern int _q2j_direct_output;
extern FILE *_q2j_output;
extern jdf_t _q2j_jdf;

static dague_list_t _dague_pool_list;
static var_t *var_head=NULL;
static int _ind_depth=0;
static int _task_count=0;
// For the JDF generation we need to emmit some things in special ways,
// (i.e. arrays in FORTRAN notation) and this "variable" will never need
// to be changed.  However if we need to use the code to generate proper "C"
// we might want to set it to false.
int JDF_NOTATION = 1;

typedef struct matrix_variable matrix_variable_t;

struct matrix_variable{
    char *matrix_name;
    int matrix_rank;
    matrix_variable_t *next;
};

typedef struct var_def_item {
    dague_list_item_t super;
    char *var;
    char *def;
} var_def_item_t;

static void do_parentize(node_t *node, int off);
static void do_loop_parentize(node_t *node, node_t *enclosing_loop);
static void do_if_parentize(node_t *node, node_t *enclosing_if);
static int DA_quark_INOUT(node_t *node);
static int DA_quark_TYPE(node_t *node);
static node_t *_DA_canonicalize_for_econd(node_t *node, node_t *ivar);
static int is_var_repeating(char *iv_str, char **iv_names);
static char *size_to_pool_name(char *size_str);
static int isArrayLocal(node_t *task_node, int index);
static int isArrayOut(node_t *task_node, int index);
static int isArrayIn(node_t *task_node, int index);
static void add_phony_INOUT_task_loops(matrix_variable_t *list, node_t *node, int task_type);
static void add_entry_task_loops(matrix_variable_t *list, node_t *node);
static void add_exit_task_loops(matrix_variable_t *list, node_t *node);
static matrix_variable_t *quark_find_all_matrices(node_t *node);
static int is_definition_seen(dague_list_t *var_def_list, char *param);
static void mark_definition_as_seen(dague_list_t *var_def_list, char *param);
static int is_acceptable_econd(node_t *node, char *ivar);
static int is_id_or_mul(node_t *node, char *ivar);
static int is_decrementing(node_t *node);
void convert_loop_from_decr_to_incr(node_t *node);
int replace_induction_variable_in_body(node_t *node, node_t *ivar, node_t *replacement);

/**
 * This function is not thread-safe, not reentrant, and not pure. As such it
 * cannot be used twice on the same call to any oter function (including
 * printf's and friends). However, as a side effect, when it is called with
 * the same value for n, it is safe to be used in any of the previously
 * mentioned scenarios.
 */
char *indent(int n, int size)
{
    static char *istr    = NULL;
    static int   istrlen = 0;
    int i;

    if( n * size + 1 > istrlen ) {
        istrlen = n * size + 1;
        istr = (char*)realloc(istr, istrlen);
    }

    for(i = 0; i < n * size; i++)
        istr[i] = ' ';
    istr[i] = '\0';
    return istr;
}

#if defined(__GNUC__)
void jdfoutput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
void jdfoutput(const char *format, ...)
{
    va_list ap;
    int len;

    va_start(ap, format);
    len = vfprintf(_q2j_output, format, ap);
    va_end(ap);

    if( len == -1 ) {
        fprintf(stderr, "Unable to ouptut a string\n");
    }
}

//#if 0
void dump_und(und_t *und){
    char *name;

    name = DA_var_name( DA_array_base(und->node));
    if( NULL == name )
        return;

    name = tree_to_str(und->node);

    switch( und->rw ){
    case UND_READ:
        printf("%s R type:%d", name, und->type);
        break;
    case UND_WRITE:
        printf("%s W type:%d", name, und->type);
        break;
    case UND_RW:
        printf("%s RW type:%d", name, und->type);
        break;
    }

}

void dump_all_unds(void){
    var_t *var;
    und_t *und;
    node_t *tmp;

    printf("###############\n");
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
    printf("###############\n");
}
//#endif

void add_variable_use_or_def(node_t *node, int rw, int type, int task_count){
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
            und->type = type;
            und->task_num = task_count;
            und->node = node;
            und->next = NULL;
            return;
        }
    }
    // If we didn't find the array, we create a new "var" and a new "und"
    und = (und_t *)calloc(1, sizeof(und_t));
    und->rw = rw;
    und->type = type;
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
    und_t **rslt=NULL;
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

/*********************************************************************************/
int DA_tree_contains_only_known_vars(node_t *node, char **known_vars){
    node_t *tmp;
    char *var_name;

    if( (NULL == node) || (EMPTY == node->type) || (NULL == known_vars) )
        return 1;

    switch( node->type ){
        case IDENTIFIER:
        case S_U_MEMBER:
            var_name = tree_to_str(node);
            for (int i=0; NULL != known_vars[i]; i++){
                if( !strcmp(known_vars[i], var_name) )
                    return 1;
            }
            free(var_name);
            return 0;
    }

    if( BLOCK == node->type ){
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            if( !DA_tree_contains_only_known_vars(tmp, known_vars) )
                return 0;
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            if( !DA_tree_contains_only_known_vars(node->u.kids.kids[i], known_vars) )
                return 0;
        }
    }
    return 1;
}

static void do_if_parentize(node_t *node, node_t *enclosing_if){
    node_t *tmp;

    if( (NULL == node) || (EMPTY == node->type) )
        return;

    node->enclosing_if = enclosing_if;

    if( DA_is_if(node) ){
        enclosing_if = node;
    }

    if( BLOCK == node->type ){
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            do_if_parentize(tmp, enclosing_if);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            do_if_parentize(node->u.kids.kids[i], enclosing_if);
        }
    }
}

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

static void do_parentize(node_t *node, int off){
    if( (NULL == node) || (EMPTY == node->type) )
        return;

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            tmp->parent = node;
#ifdef DEBUG
            printf("%*s", off, " ");
            printf("Next in block: %s\n",DA_type_name(tmp));
            fflush(stdout);
#endif
            do_parentize(tmp, off+4);
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            node->u.kids.kids[i]->parent = node;
#ifdef DEBUG
            printf("%*s", off, " ");
            printf("Next kid: %s\n",DA_type_name(node->u.kids.kids[i]));
            fflush(stdout);
#endif
            do_parentize( node->u.kids.kids[i], off+4 );
        }
    }
}

void DA_parentize(node_t *node){
    do_parentize(node, 0);
    do_loop_parentize(node, NULL);
    do_if_parentize(node, NULL);
    node->parent = NULL;
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

    // capital i ("I") has a special meaning in some contexts (I^2==-1), so skip it.
    if(num>=8)
        num++;

    if( num < 26 ){
        str[0] = (char)('A'+num);
        str[1] = '\0';
    }else{
        int cnt;
        str[0] = (char)('A'+num%26);
        cnt = num/26;
        snprintf(&str[1], 3, "%d", cnt);
    }
    return strdup(str);
}


/*
 * Turn "CORE_taskname_quark" into "taskname".  If lineno is non-negative
 * the result is "taskname_lineno" (where lineno is the number, not the string).
 */
static char *quark_call_to_task_name( char *call_name, int32_t lineno ){
    char *task_name, *end;
    ptrdiff_t len;
    uint32_t i, digits=1;

    if( NULL != strstr(call_name, "CORE_") )
        call_name += 5;

    end = strstr(call_name, "_quark");
    if( NULL != end ){
        len = (uintptr_t)end-(uintptr_t)call_name;
    }else{
        len = strlen(call_name);
    }

    for(i=lineno; i>0; i/=10){
        digits++;
    }

    task_name = (char *)calloc(len+digits+2, sizeof(char));
    snprintf(task_name, len+1, "%s", call_name);
    if( lineno >= 0 ){
        snprintf(task_name+len, digits+2, "_%d",lineno);
    }

    return task_name;
}

static jdf_function_entry_t *jdf_register_addfunction( jdf_t        *jdf,
                                                       const char   *fname,
                                                       const node_t *node )
{
    jdf_function_entry_t *f;
    node_t *tmp;

#ifdef DEBUG
    if ( jdf->functions != NULL ) {
        jdf_function_entry_t *f2 = jdf->functions;
        do {
            assert( strcmp(fname, f2->fname ) != 0 );
            f2 = f2->next;
        } while( f2 != NULL );
    }
#endif

    f = q2jmalloc(jdf_function_entry_t, 1);
    f->fname = strdup(fname);
    f->parameters  = NULL;
    f->properties  = NULL;
    f->locals      = NULL;
    f->simcost     = NULL;
    f->predicate   = NULL;
    f->dataflow    = NULL;
    f->priority    = NULL;
    f->body        = NULL;

    for(tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
        jdf_name_list_t *n = q2jmalloc(jdf_name_list_t, 1);
        n->next = f->parameters;
        n->name = DA_var_name(DA_loop_induction_variable(tmp));
        f->parameters = n;
    }

#if 0
    f->next = jdf->functions;
    jdf->functions = f;
#else
    {
        jdf_function_entry_t *n = jdf->functions;
        if (jdf->functions == NULL )
            jdf->functions = f;
        else {
            while (n->next != NULL)
                n = n->next;
            n->next = f;
        }
    }
#endif
    return f;
}

static void quark_record_uses_defs_and_pools(node_t *node, int mult_kernel_occ){
    static int symbolic_name_count = 0;
    int i;
    static int pool_initialized = 0;

    if ( !pool_initialized ) {
        dague_list_construct(&_dague_pool_list);
        pool_initialized++;
    }

    if( FCALL == node->type ){
        char *fname;
        int kid_count;
        task_t *task;
        jdf_function_entry_t *f;

        if( strcmp("QUARK_Insert_Task", DA_kid(node,0)->u.var_name) ){
            return;
        }

        kid_count = node->u.kids.kid_count;

        // QUARK specific code. The task is the second parameter.
        if( (kid_count > 2) && (IDENTIFIER == DA_kid(node,2)->type) ){
            fname = quark_call_to_task_name( DA_var_name(DA_kid(node,2)), 
                                             mult_kernel_occ ? (int32_t)node->lineno : -1 );
            
            f = jdf_register_addfunction( &_q2j_jdf, fname, node );
            task = (task_t *)calloc(1, sizeof(task_t));
            task->task_node = node;
            task->ind_vars = (char **)calloc(1+node->loop_depth, sizeof(char *));
            i=node->loop_depth-1;
            for(node_t *tmp=node->enclosing_loop; NULL != tmp; tmp=tmp->enclosing_loop ){
                task->ind_vars[i] = DA_var_name(DA_loop_induction_variable(tmp));
                --i;
            }

            node->task = task;
            node->function = f;

        } else {
#if defined(DEBUG)
            printf("WARNING: probably there is something wrong with this QUARK_Insert_Task().\n");
#endif
            return;
        }

        assert(f != NULL);
        for(i=1; i<kid_count; ++i){
            node_t *tmp = node->u.kids.kids[i];

            // Record USE of DEF
            if( ARRAY == tmp->type ){
                tmp->task = task;
                tmp->function = f;
                tmp->var_symname = numToSymName(symbolic_name_count++);
                node_t *qual = node->u.kids.kids[i+1];
                add_variable_use_or_def( tmp, DA_quark_INOUT(qual), DA_quark_TYPE(qual), _task_count );
            }

            // Record a pool (size_to_pool_name() will create an entry for each new pool)
            if( (i+1<node->u.kids.kid_count) && (i>1) && !strcmp(tree_to_str(node->u.kids.kids[i+1]), "SCRATCH") ){
                (void)size_to_pool_name( tree_to_str(node->u.kids.kids[i-1]) );
            }
        }
        _task_count++;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            quark_record_uses_defs_and_pools(tmp, mult_kernel_occ);
        }
    }else{
        for(i=0; i<node->u.kids.kid_count; ++i){
            quark_record_uses_defs_and_pools(node->u.kids.kids[i], mult_kernel_occ);
        }
    }
}


static matrix_variable_t *quark_find_all_matrices(node_t *node){
    static matrix_variable_t *matrix_variable_list_head = NULL;
    int i;

    if( FCALL == node->type ){
        if( strcmp("QUARK_Insert_Task", DA_kid(node,0)->u.var_name) ){
            return NULL;
        }

        for(i=1; i<DA_kid_count(node); ++i){
            int already_in = 0;
            matrix_variable_t *curr = NULL;
            node_t *tmp = DA_kid(node,i);
            char *tmp_name = DA_var_name( DA_array_base(tmp) );

            if( ARRAY == tmp->type ){
                if(NULL == matrix_variable_list_head){
                    matrix_variable_list_head = (matrix_variable_t *)calloc(1, sizeof(matrix_variable_t));
                    curr = matrix_variable_list_head;
                }else{
                    // go to the end of the list and check that the matrix does not exist already in the list.
                    for(curr = matrix_variable_list_head; NULL != curr->next; curr = curr->next){
                        if( !strcmp(curr->matrix_name, tmp_name) ){
                            already_in = 1;
                            break;
                        }
                    }
                    if( !strcmp(curr->matrix_name, tmp_name) ){
                        already_in = 1;
                    }
                    if( !already_in ){
                        curr->next = (matrix_variable_t *)calloc(1, sizeof(matrix_variable_t));
                        curr = curr->next;
                    }
                }
                if( !already_in ){
                    curr->matrix_name = tmp_name;
                    curr->matrix_rank = DA_array_dim_count(tmp);
                    curr->next = NULL; /* just being paranoid */
                }
            }
        }
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            (void)quark_find_all_matrices(tmp);
        }
    }else{
        for(i=0; i< DA_kid_count(node); ++i){
            (void)quark_find_all_matrices( DA_kid(node,i) );
        }
    }

    return matrix_variable_list_head;
}

/* 
 * Take the first OUT or INOUT array variable and make it the data element that
 * this task should have affinity to.
 * It would be much better if we found which tile this task writes most times into,
 * instead of the first write, to reduce unnecessary communication.
 */
node_t *quark_get_locality(node_t *task_node){
    int i;

    /*
     * First loop to search LOCALITY flag
     */
    for(i=QUARK_FIRST_VAR; i<task_node->u.kids.kid_count; i+=QUARK_ELEMS_PER_LINE){
        if( isArrayOut(task_node, i) && isArrayLocal(task_node, i) ){
            node_t *data_element = task_node->u.kids.kids[i];
            return data_element;
        }
    }

    /*
     * If no LOCALITY flag, the first output data is used for locality
     */
    for(i=QUARK_FIRST_VAR; i<task_node->u.kids.kid_count; i+=QUARK_ELEMS_PER_LINE){
        if( isArrayOut(task_node, i) ){
            node_t *data_element = task_node->u.kids.kids[i];
            return data_element;
        }
    }

    fprintf(stderr,"WARNING: task: \"%s\" does not alter any memory regions!", task_node->function->fname);
    return NULL;
}

/* 
 * kernel_exists() uses the functions is_definition_seen() and mark_definition_as_seen()
 * not because this code does anything with uses and definitions but as a 
 * set::find() and set::insert() in C++ stl terminology.
 */
static inline int kernel_exists(char *task_name){
    static int kernel_count_initialized = 0;
    static dague_list_t kernel_name_list;

    if ( !kernel_count_initialized ) {
        dague_list_construct(&kernel_name_list);
        kernel_count_initialized = 1;
    }

    if( is_definition_seen(&kernel_name_list, task_name) ){
        return 1;
    }
    mark_definition_as_seen(&kernel_name_list, task_name);

    return 0;
}

static inline int check_for_multiple_kernel_occurances(node_t *node){
    int i;

    if( FCALL == node->type ){
        int kid_count;

        if( strcmp("QUARK_Insert_Task", DA_kid(node,0)->u.var_name) ){
            return 0;
        }

        kid_count = node->u.kids.kid_count;

        // QUARK specific code. The task is the second parameter.
        if( (kid_count > 2) && (IDENTIFIER == DA_kid(node,2)->type) ){
            char *task_name = quark_call_to_task_name( DA_var_name(DA_kid(node,2)), -1 );
            if( kernel_exists(task_name) ){
                return 1;
            }
        }
        return 0;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            if( check_for_multiple_kernel_occurances(tmp) ){
                return 1;
            }
        }
    }else{
        for(i=0; i< DA_kid_count(node); ++i){
            if( check_for_multiple_kernel_occurances( DA_kid(node,i) ) ){
                return 1;
            }
        }
    }

    return 0;
}

void analyze_deps(node_t *node){
    int mult = check_for_multiple_kernel_occurances(node);
    quark_record_uses_defs_and_pools(node, mult);
    //dump_all_unds();
    interrogate_omega(node, var_head);
    if (!_q2j_direct_output){
        jdf_unparse( &_q2j_jdf, stdout );
    }
}


static void add_entry_task_loops(matrix_variable_t *list, node_t *node){
    add_phony_INOUT_task_loops(list, node, TASK_IN);
}

static void add_exit_task_loops(matrix_variable_t *list, node_t *node){
    add_phony_INOUT_task_loops(list, node, TASK_OUT);
}

static void add_phony_INOUT_task_loops(matrix_variable_t *list, node_t *node, int task_type){
    int i, dim;
    // FIXME: This will create variables with names like A.nt, but in the "real" code, these will be structure members. Is that ok?
    node_t *container_block, *ind_vars[2];
    matrix_variable_t *curr;

    assert( NULL != list );

    container_block = NULL;
    if( BLOCK == node->type ){
        container_block = node;
    }else{
        for(i=0; i< DA_kid_count(node); ++i){
            if( BLOCK == DA_kid(node,i)->type ){
                container_block = DA_kid(node,i);
                break;
            }
        }
    }
    assert( NULL != container_block);

    for(curr = list; NULL != curr; curr = curr->next){
        char *curr_matrix = curr->matrix_name;
        int  matrix_rank  = curr->matrix_rank;
        char *tmp_str;
        node_t *new_block, *tmp_block, *enclosing_loop = NULL;

        // Create a block to contain the loop nest.
        new_block = DA_create_Block();
        tmp_block = new_block;

        for(dim=0; dim<matrix_rank; dim++){
            char *ind_var_name;
            node_t *new_node, *scond, *econd, *incr, *body;

            // Create the induction variable.
            asprintf(&ind_var_name,"%c",'i'+dim); //
            ind_vars[dim] = DA_create_ID(ind_var_name);
            free(ind_var_name); // DA_create_ID() will copy the string anyway.

            // Create the assignment of the lower bound to the induction variable (start condition, scond).
            scond = DA_create_B_expr( ASSIGN, ind_vars[dim], DA_create_Int_const(0) );

            // Build a string that matches the name of the upper bound for PLASMA matrices.
            switch(dim){
                case 0:  asprintf(&(tmp_str), "desc%s.mt", curr_matrix);
                         break;
                case 1:  asprintf(&(tmp_str), "desc%s.nt", curr_matrix);
                         break;
                default: fprintf(stderr,"FATAL ERROR in add_phony_INOUT_task_loops(): Currently only 2D matrices are supported\n");
                         abort();
            }

            // Create the comparison of the induction variable against the upper bound (end condition, econd).
            econd = DA_create_B_expr( LT, ind_vars[dim], DA_create_ID(tmp_str) );

            // Reclaim some memory.
            free(tmp_str);

            // Create the incement (i++).
            incr = DA_create_B_expr( EXPR, ind_vars[dim], DA_create_Unary(INC_OP) );

            // Create an empty body.
            body = DA_create_Block();

            // Create the FOR.
            new_node = DA_create_For(scond, econd, incr, body);

            // Put the newly created FOR into the parent BLOCK,
            // and make the body of the for by the parent BLOCK for the next iteration.
            DA_insert_first(tmp_block, new_node);
            tmp_block = body;
        }

        // Create a call like this:
        // QUARK_Insert_Task( phony, CORE_TaskName_quark, phony,
        //                    phony, A(k,k), INOUT, 0 )

        // Create a phony variable.
        node_t *phony_var = DA_create_ID("phony");

        // Create a variable to hold the task name in QUARK specific format.
        // WARNING: The string prefices DAGUE_IN_ and DAGUE_OUT_ are also used in 
        // omega_interface.c:is_phony_Entry_task() and 
        // omega_interface.c:is_phony_Exit_task()
        // so don't change them without changing them there as well.
        if( TASK_IN == task_type ){
            asprintf(&(tmp_str), "CORE_DAGUE_IN_%s_quark", curr_matrix);
        }else if( TASK_OUT == task_type ){
            asprintf(&(tmp_str), "CORE_DAGUE_OUT_%s_quark", curr_matrix);
        }else{
            assert(0);
        }
        node_t *task_name_var = DA_create_ID(tmp_str);
        free(tmp_str);

        // Create the access to the matrix element.
        node_t *matrix_element = DA_create_ArrayAccess(curr_matrix, ind_vars[0], ind_vars[1], NULL);

        // Create the function-call.
        node_t *f_call = DA_create_Fcall("QUARK_Insert_Task", phony_var, task_name_var, phony_var,
                                         phony_var, matrix_element, DA_create_ID("INOUT"),
                                         DA_create_Int_const(0), NULL);
        f_call->enclosing_loop = enclosing_loop;

        // Put the newly created FCALL into the BLOCK of the inner-most loop.
        DA_insert_first(tmp_block, f_call);

        // Put the block with the loop nest at the beginning (or the end) of the container block
        if( TASK_IN == task_type ){
            DA_insert_first(container_block, new_block);
        }else if( TASK_OUT == task_type ){
            DA_insert_last(container_block, new_block);
        }else{
            assert(0);
        }
    }

    return;
}

void add_entry_and_exit_task_loops(node_t *node){
    matrix_variable_t *list;

    list = quark_find_all_matrices(node);
    add_entry_task_loops(list, node);
    add_exit_task_loops(list, node);
    DA_parentize(node);
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

    // If no test failed, it's a match
    return 1;
}

static int var_name_to_num(char *name, int prfx_len){
   int len; 
   len = strlen(name);

   if( prfx_len > len )
       return -1;
   if( prfx_len == len )
       return 0;

   return atoi( (const char *)&name[prfx_len] );

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
        if( S_U_MEMBER == node->type )
            return;
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
    // Find the number of digits in the number
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
        case FOR:
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
            if( pos >= len-1 ){
                // The array that holds the list needs to be resized
                uintptr_t old_size;
                char **tmp_ptr;
                old_size = len*sizeof(char *);
                len*=2;
                tmp_ptr = (char **)calloc(len, sizeof(char *));
                memcpy(tmp_ptr, iv_names, old_size);
                iv_names = tmp_ptr;
            }
            // Add the new variable into the list (iv_names)
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

/*
 * The following is a necessary optimization only because DAGUE does not (currently)
 * support tiles that do not originite in user's memory, and therefore there cannot
 * be a tile that is only OUTPUT without being IN (and therefore without being read
 * from a preallocated memory region).
 */
void convert_OUTPUT_to_INOUT(node_t *node){
    int i;

    if( FCALL == node->type ){
        for(i=QUARK_FIRST_VAR; i<node->u.kids.kid_count; i+=QUARK_ELEMS_PER_LINE){
            if( isArrayOut(node, i) && !isArrayIn(node, i) ){
                node_t *flag = node->u.kids.kids[i+1];
                flag = DA_create_B_expr(B_OR, flag, DA_create_ID("INPUT"));
                node->u.kids.kids[i+1] = flag;
            }
        }
        return;
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            convert_OUTPUT_to_INOUT(tmp);
        }
    }else{
        for(i=0; i<node->u.kids.kid_count; ++i){
            convert_OUTPUT_to_INOUT(node->u.kids.kids[i]);
        }
    }
}


// poor man's asprintf().
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


static int is_id_or_mul(node_t *node, char *ivar){
    /* If it's the induction variable, we are good */
    if( IDENTIFIER == node->type ){
        char *name = DA_var_name(node);
        if( (NULL != name) && !strcmp(ivar, name) ){
            return 1;
        }
    }
    /* If it's a multiplication, check if it includes the induction variable. */
    /* If so, we are good */
    if( (MUL == node->type) ){
        if( is_id_or_mul( DA_rel_lhs(node), ivar) || is_id_or_mul( DA_rel_rhs(node), ivar) ){
            return 1;
        }
    }
    return 0;
}

static int is_acceptable_econd(node_t *node, char *ivar){
    node_t *kid;

    assert( NULL != ivar );

    /* Make sure this is a comparison expression */
    if( !DA_is_rel(node) ){
        return IVAR_NOT_FOUND;
    }

    /* Examine the left hand side of the expression */
    kid = DA_rel_lhs(node);
    if( is_id_or_mul(kid, ivar) ){
        return IVAR_IS_LEFT;
    }
    /* Examine the right hand side of the expression */
    kid = DA_rel_rhs(node);
    if( is_id_or_mul(kid, ivar) ){
        return IVAR_IS_RIGHT;
    }

    return IVAR_NOT_FOUND;
}


static node_t *_DA_canonicalize_for_econd(node_t *node, node_t *ivar){
    node_t *tmp=NULL;
    int ivar_side;
    node_t *lhs, *rhs;

    ivar_side = is_acceptable_econd(node, DA_var_name(ivar));
    if( IVAR_NOT_FOUND == ivar_side ){
        switch( node->type ){
            case L_AND:
                lhs = _DA_canonicalize_for_econd(DA_kid(node,0),ivar);
                rhs = _DA_canonicalize_for_econd(DA_kid(node,1),ivar);
                if( NULL == lhs || NULL == rhs ){
                    break;
                }
                tmp = DA_create_B_expr(L_AND, lhs, rhs);
                return tmp;
            case L_OR:
                lhs = _DA_canonicalize_for_econd(DA_kid(node,0),ivar);
                rhs = _DA_canonicalize_for_econd(DA_kid(node,1),ivar);
                if( NULL == lhs || NULL == rhs ){
                    break;
                }
                tmp = DA_create_B_expr(L_OR, lhs, rhs);
                return tmp;
        }
        printf("Cannot canonicalize end condition of for() loop: ");
        dump_tree(*node, 0);
        printf("\n");
        assert(0);
    }

    // If the variable is on the left hand side
    if( IVAR_IS_LEFT == ivar_side ){
        switch( node->type ){
            case LT:  // since the var is in the left, do nothing, that's the canonical form.
                return node;

            case LE:  // add one to the RHS and convert LE to LT
                tmp = DA_create_B_expr(ADD, DA_rel_rhs(node), DA_create_Int_const(1));
                tmp = DA_create_relation(LT, DA_rel_lhs(node), tmp);
                return tmp;

            // If the variable is on the left and we have a GE or GT,
            // then we are in a loop that uses a decrementing modifier.
            case GE:  // subtract one from the RHS to convert GE to GT
                tmp = DA_create_B_expr(SUB, DA_rel_rhs(node), DA_create_Int_const(1));
                return tmp;
            case GT:  // There is nothing I can do here, convert_loop_from_decr_to_incr() will take care of this.
                return node;

            default: 
                printf("Cannot canonicalize end condition of for() loop: ");
                dump_tree(*node, 0);
                printf("\n");
                break;
        }
    }else if( IVAR_IS_RIGHT == ivar_side ){
        // If the variable is on the RHS, flip the relation operator, exchange LHS and RHS and call myself again.
        tmp = DA_create_relation(DA_flip_rel_op(node->type), DA_rel_rhs(node), DA_rel_lhs(node));
        tmp = _DA_canonicalize_for_econd(tmp, ivar);
        return tmp;
    }

    return NULL;
}


int replace_induction_variable_in_body(node_t *node, node_t *ivar, node_t *replacement){
    int ret;

    if( IDENTIFIER == node->type ){
        char *str1 = DA_var_name(ivar);
        char *str2 = DA_var_name(node);
        if( (NULL != str1) && (NULL != str2) && !strcmp(str1, str2) ){
            return 1;
        }
    }

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            ret = replace_induction_variable_in_body(tmp, ivar, replacement);
            // The induction variable should never appear as a top level
            // statement inside the body of the loop.  Only as a leaf of an subtree.
            assert( !ret );
        }
    }else{
        int i;
        for(i=0; i<node->u.kids.kid_count; ++i){
            ret = replace_induction_variable_in_body(node->u.kids.kids[i], ivar, replacement);
            // If this kid is the induction variable, but not as a member of a struct, or union, then let's replace it
            if( ret && (S_U_MEMBER != node->type) ){
                node->u.kids.kids[i] = replacement;
            }
        }
    }

    return 0;
}

static int is_decrementing(node_t *node){
    int rslt = 0;
    node_t *modifier = DA_for_modifier(node);

    switch(modifier->type){
        case EXPR: // ++ or --
            if( (INC_OP == DA_exp_lhs(modifier)->type) || (INC_OP == DA_exp_rhs(modifier)->type) ){
                rslt = 0;
            }else if( (DEC_OP == DA_exp_lhs(modifier)->type) || (DEC_OP == DA_exp_rhs(modifier)->type) ){
                rslt = 1;
            }
            break;
        case ADD_ASSIGN: // +=
            rslt = 0;
            break;
        case SUB_ASSIGN: // -=
            rslt = 1;
            break;
        default:
            printf("Cannot analyze modifier type \"%s\" of for() loop: ", DA_type_name(modifier));
            dump_tree(*node, 0);
            printf("\n");
            assert(0);
    }
    return rslt;
}

/*
 * This function assumes that the loop end-condition has already
 * been canonicalized to "ivar < B" or "ivar > B" form.
 */
void convert_loop_from_decr_to_incr(node_t *loop){
    node_t *n0, *n1;
    node_t *ivar, *lb, *ub, *new_ub;
    node_t *scond, *econd, *incr, *expr;

    // extract the three parts of the current for() loop
    n0 = DA_for_scond(loop);
    n1 = DA_for_econd(loop);

    // get the induction variable of the loop
    ivar = DA_assgn_lhs(n0);
    // get the upper bound of the loop, from the starting condition
    ub = DA_assgn_rhs(n0);
    // get the lower bound of the loop, from the ending condition
    lb = DA_assgn_rhs(n1);

    // compute the new upper bound (ub-lb), given that the
    // canonicalized loop will have zero as its lower bound.
    new_ub = DA_create_B_expr( SUB, ub, lb );

    scond = DA_create_B_expr( ASSIGN, ivar, DA_create_Int_const(0) );
    econd = DA_create_B_expr( LT,     ivar, new_ub );
    incr  = DA_create_B_expr( EXPR,   ivar, DA_create_Unary(INC_OP) );

    // These are macros that expand to struct fields, so we can assign to them
    DA_for_scond(loop)    = scond;
    DA_for_econd(loop)    = econd;
    DA_for_modifier(loop) = incr;

    // We need to replace all occurances of the induction variable (ivar)
    // in the body of this loop with ub-ivar, where ub is the
    // starting condition of the original loop, not the new_ub.
    expr = DA_create_B_expr( SUB, ub, ivar );
    replace_induction_variable_in_body(DA_for_body(loop), ivar, expr);

    return;
}

/* TODO: This needs a lot more work to become a general canonicalization function */
int DA_canonicalize_for(node_t *node){
    node_t *ivar, *tmp;

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

    // Canonicalize the end condition (the middle statement of the for)
    tmp = _DA_canonicalize_for_econd(DA_for_econd(node), ivar);
    if( NULL == tmp ){
        return 0;
    }
    DA_for_econd(node) = tmp;

    // Check if the loop is incrementing or decrementing the induction variable
    // for(i=LB; i<UB; i++) or for(i=UB; i>LB; i--)
    // If it is the latter, convert it into the former
    if( is_decrementing(node) ){
        convert_loop_from_decr_to_incr(node);
    }

    return 1;
}

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

node_t *DA_create_ID(char *name){
    node_t *node = (node_t *)calloc(1,sizeof(node_t));
    node->type = IDENTIFIER;
    node->u.var_name = strdup(name);
    // TODO: should this variable be in the symbol table?
    return node;
}

node_t *DA_create_B_expr(int type, node_t *kid0, node_t *kid1){
    node_t rslt;
    memset(&rslt, 0, sizeof(node_t));
    rslt.type = type;
    rslt.u.kids.kid_count = 2;
    rslt.u.kids.kids = (node_t **)calloc(2,sizeof(node_t *));
    rslt.u.kids.kids[0] = kid0;
    rslt.u.kids.kids[1] = kid1;
    return node_to_ptr(rslt);
}

node_t *DA_create_Int_const(int64_t val){
    node_t rslt;
    memset(&rslt, 0, sizeof(node_t));
    rslt.type = INTCONSTANT;
    rslt.u.kids.kid_count = 0;
    rslt.const_val.i64_value = val;
    return node_to_ptr(rslt);
}

node_t *DA_create_Unary(uint32_t type){
    node_t *rslt = (node_t *)calloc(1,sizeof(node_t));
    rslt->type = type;
    return rslt;
}

node_t *DA_create_Block(void){
    node_t *node = (node_t *)calloc(1,sizeof(node_t));
    node->type = BLOCK;
    return node;
}

node_t *DA_create_For(node_t *scond, node_t *econd, node_t *incr, node_t *body){
    node_t *node = (node_t *)calloc(1,sizeof(node_t));
    node->type = FOR;
    node->u.kids.kids = (node_t **)calloc(4, sizeof(node_t *));
    node->u.kids.kid_count = 4;
    node->u.kids.kids[0] = scond;
    node->u.kids.kids[1] = econd;
    node->u.kids.kids[2] = incr;
    node->u.kids.kids[3] = body;
    return node;
}

node_t *DA_create_Complex(uint32_t type, char *name, ...){
    va_list argp;
    node_t *node, *indx;
    int i, count=0;

    // Count the number of arguments passed (except for the terminating NULL).
    va_start(argp, name);
    while( NULL != va_arg(argp, node_t *) ){
        count++;
    }
    va_end(argp);

    node = (node_t *)calloc(1,sizeof(node_t));
    node->type = type;
    node->u.kids.kid_count = count+1;
    node->u.kids.kids = (node_t **)calloc(count+1, sizeof(node_t *));
    node->u.kids.kids[0] = DA_create_ID(name);

    va_start(argp, name);
    for(i=1; NULL != (indx = va_arg(argp, node_t *)); i++){
        node->u.kids.kids[i] = indx;
    }
    va_end(argp);

    return node;
}


void DA_insert_first(node_t *block, node_t *new_node){
    node_t *tmp = block->u.block.first;
    if( NULL == tmp ){
        assert( NULL == block->u.block.last );
        block->u.block.first = new_node;
        block->u.block.last = new_node;
        return;
    }
    block->u.block.first = new_node;
    new_node->next = tmp;
    tmp->prev = new_node;
    assert( NULL != block->u.block.last );
    return;
}

void DA_insert_last(node_t *block, node_t *new_node){
    node_t *tmp = block->u.block.last;
    if( NULL == tmp ){
        assert( NULL == block->u.block.first );
        block->u.block.last = new_node;
        block->u.block.first = new_node;
        return;
    }

    block->u.block.last = new_node;
    new_node->prev = tmp;
    tmp->next = new_node;

    assert( NULL != block->u.block.first );
    return;
}

node_t *DA_create_Entry(){
    node_t rslt;
    memset(&rslt, 0, sizeof(node_t));
    rslt.type = ENTRY;
    rslt.u.kids.kid_count = 0;
    return node_to_ptr(rslt);
}

node_t *DA_create_Exit(){
    node_t rslt;
    memset(&rslt, 0, sizeof(node_t));
    rslt.type = EXIT;
    rslt.u.kids.kid_count = 0;
    return node_to_ptr(rslt);
}

node_t *DA_exp_to_ind(node_t *node){
    node_t *lhs, *rhs;
    switch(node->type){
        case MUL:
            lhs = node->u.kids.kids[0];
            rhs = node->u.kids.kids[1];

            assert( INTCONSTANT == lhs->type || INTCONSTANT == rhs->type );
            if( INTCONSTANT == lhs->type ){
                return rhs;
            }else{
                return lhs;
            }
    }

    return node;
}

/* 
 * This function expects to find something like
 * "3*x", or "x*3" in which case it returns "3".
 * If it finds anything else, it returns 1
 */
int DA_exp_to_const(node_t *node){
    node_t *lhs, *rhs;
    switch(node->type){
        case MUL:
            lhs = node->u.kids.kids[0];
            rhs = node->u.kids.kids[1];

            assert( INTCONSTANT == lhs->type || INTCONSTANT == rhs->type );
            if( INTCONSTANT == lhs->type ){
                return DA_int_val(lhs);
            }else if( INTCONSTANT == rhs->type ){
                return DA_int_val(rhs);
            }
    }

    return 1;
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
 * Is the node an if() node.
 */
int DA_is_if(node_t *node){
    if( IF == node->type ){
        return 1;
    }else{
        return 0;
    }
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


static int DA_quark_LOCALITY_FLAG(node_t *node){
    int rslt1, rslt2;
    if( NULL == node )
        return 0;

    switch(node->type){
        case IDENTIFIER:
            if( !strcmp(node->u.var_name, "LOCALITY") ){
                return 1;
            }
            return 0;

        case B_OR:
            rslt1 = DA_quark_LOCALITY_FLAG(node->u.kids.kids[0]);
            if( rslt1 > 0 ) return 1;
            rslt2 = DA_quark_LOCALITY_FLAG(node->u.kids.kids[1]);
            if( rslt2 > 0 ) return 1;

            return 0;

        default:
            fprintf(stderr,"DA_quark_LOCALITY_FLAG(): unsupported flag type for dep\n");
            exit(-1);

    }
    return 0;
}

static int DA_quark_TYPE(node_t *node){
    int rslt1, rslt2;
    if( NULL == node )
        return -1;

    switch(node->type){
        case IDENTIFIER:
            if( !strcmp(node->u.var_name, "QUARK_REGION_U") ){
                return 0x2;
            }
            if( !strcmp(node->u.var_name, "QUARK_REGION_L") ){
                return 0x4;
            }
            if( !strcmp(node->u.var_name, "QUARK_REGION_D") ){
                return 0x1;
            }
            return UND_IGNORE;

        case B_OR:
            rslt1 = DA_quark_TYPE(node->u.kids.kids[0]);
            if( rslt1 < 0 ) return -1;
            rslt2 = DA_quark_TYPE(node->u.kids.kids[1]);
            if( rslt2 < 0 ) return -1;

            return rslt1 | rslt2;

        default:
            fprintf(stderr,"DA_quark_TYPE(): unsupported flag type for dep\n");
            exit(-1);

    }
    return -1;
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
        return 0;
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
            n2 = DA_for_modifier(loop);
            assert( (NULL != n0) && (NULL != n1) && (NULL != n2) );
            if( ASSIGN != n0->type ){
                fprintf(stderr,"Don't know how to extract induction variable from type: %s\n",DA_type_name(n0));
                assert(0);
            }
            tmp = DA_assgn_lhs(n0);
            if( IDENTIFIER != tmp->type ){
                fprintf(stderr,"Don't know how to deal with LHS of type: %s\n",DA_type_name(tmp));
                assert(0);
            }

            return tmp;
            
        default:
            return NULL;
    }
}

node_t *DA_if_condition(node_t *node){
    if( IF == node->type )
        return node->u.kids.kids[0];
    return NULL;
}

node_t *DA_if_then_body(node_t *node){
    if( IF == node->type && node->u.kids.kid_count >= 2)
        return node->u.kids.kids[1];
    return NULL;
}

node_t *DA_if_else_body(node_t *node){
    if( IF == node->type && node->u.kids.kid_count >= 3)
        return node->u.kids.kids[2];
    return NULL;
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
            if( JDF_NOTATION )
                return "&";
            else
                return "&&";
        case L_OR:
            if( JDF_NOTATION )
                return "|";
            else
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
        case S_U_MEMBER:
            return "STRUCT_or_UNION";
    }
    return "???";
}

static int isSimpleVar(char *name){
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

static char *find_definition(char *var_name, node_t *node){
    node_t *curr, *tmp;
    do{
        for(curr=node->prev; NULL!=curr; curr=curr->prev){
            if( ASSIGN == curr->type ){
                tmp = DA_assgn_lhs(curr);
                if( IDENTIFIER == tmp->type ){
                    char *name = DA_var_name(tmp);
                    if( (NULL != name) && !strcmp(name, var_name) ){
                        return tree_to_str( curr );
                    }
                }
            }
        }
        node = node->parent;
    }while(NULL != node);
    return var_name;
}

static int isArrayLocal(node_t *task_node, int index){
    if( index+1 < task_node->u.kids.kid_count ){
        node_t *flag = task_node->u.kids.kids[index+1];
        return DA_quark_LOCALITY_FLAG(flag);
    }
    return 0;
}

static int isArrayOut(node_t *task_node, int index){
    if( index+1 < task_node->u.kids.kid_count ){
        node_t *flag = task_node->u.kids.kids[index+1];
        if( (UND_WRITE & DA_quark_INOUT(flag)) != 0 ){
            return 1;
        }
    }
    return 0;
}

static int isArrayIn(node_t *task_node, int index){
    if( index+1 < task_node->u.kids.kid_count ){
        node_t *flag = task_node->u.kids.kids[index+1];
        if( (UND_READ & DA_quark_INOUT(flag)) != 0 ){
            return 1;
        }
    }
    return 0;
}

static char *int_to_str(int num){
    int lg, i = 1;
    char *str;
    // Find the number of digits of the number;
    for(lg=1; lg<num; lg*=10){
        i++;
    }

    // Create the new variable name
    str = (char *)calloc(1+i, sizeof(char));
    snprintf(str, 1+i, "%d", num);
    return str;
}

/*
 * size_to_pool_name() maintains a map between buffer sizes and memory pools.
 * Since DAGuE does not have a predefind map, or hash-table, we use a linked list where we
 * store the different sizes and the pools they map to and traverse it every time we do
 * a lookup. Given that in reallity the size of this list is not expected to exceed 4 or 5
 * elements, it doesn't really matter.  Also, we use the "var_def_item_t" structure, just to
 * reuse code. "var" will be a size and "def" will be a pool name.
 */
static char *size_to_pool_name(char *size_str){
    static int pool_count = 0;
    char *pool_name = NULL;

    /* See if a pool of this size exists already, and if so return it. */
    DAGUE_ULIST_ITERATOR(&_dague_pool_list, list_item,
    {
        var_def_item_t *true_item = (var_def_item_t *)list_item;
        assert(NULL != true_item->var);
        assert(NULL != true_item->def);
        if( !strcmp(true_item->var, size_str) ){
            return true_item->def;
        }
    });

    /* If control reached here, it means that we didn't find a pool of the given size. */
    pool_name = append_to_string( strdup("pool_"), int_to_str(pool_count), NULL, 0);
    pool_count++;

    /* add then new pool to the list, so we find it next time we look. */
    var_def_item_t *new_item = (var_def_item_t *)calloc(1, sizeof(var_def_item_t));
    new_item->var = size_str;
    new_item->def = pool_name;
    dague_ulist_lifo_push( &_dague_pool_list, (dague_list_item_t *)new_item );

    return pool_name;
}

string_arena_t *create_pool_declarations(){
    string_arena_t *sa = NULL;
    
    sa = string_arena_new(64);
    DAGUE_ULIST_ITERATOR(&_dague_pool_list, list_item,
                         {
                             var_def_item_t *true_item = (var_def_item_t *)list_item;
                             string_arena_add_string(sa, "%s [type = \"dague_memory_pool_t *\" size = \"%s\"]\n",
                                                     true_item->def,
                                                     true_item->var);
                         });
    return sa;
}

void jdf_register_pools( jdf_t *jdf )
{
    jdf_global_entry_t *prev = jdf->globals;

    if ( jdf->globals != NULL ) {
        prev = jdf->globals;
        while( prev->next != NULL )
            prev = prev->next;
    }

    DAGUE_ULIST_ITERATOR(&_dague_pool_list, list_item,
                         {
                             var_def_item_t *true_item = (var_def_item_t *)list_item;
                             jdf_global_entry_t *e = q2jmalloc(jdf_global_entry_t, 1);
                             e->next       = NULL;
                             e->name       = true_item->def;
                             e->properties = q2jmalloc(jdf_def_list_t, 2);
                             e->expression = NULL;
                             JDF_OBJECT_SET(e, NULL, 0, NULL);

                             e->properties[0].next       = (e->properties)+1;
                             e->properties[0].name       = strdup("type");
                             e->properties[0].expr       = q2jmalloc(jdf_expr_t, 1);
                             e->properties[0].properties = NULL;
                             JDF_OBJECT_SET(&(e->properties[0]), NULL, 0, NULL);
                             e->properties[0].expr->op      = JDF_STRING;
                             e->properties[0].expr->jdf_var = strdup("dague_memory_pool_t *");

                             e->properties[1].next       = NULL;
                             e->properties[1].name       = strdup("size");
                             e->properties[1].expr       = q2jmalloc(jdf_expr_t, 1);
                             e->properties[1].properties = NULL;
                             JDF_OBJECT_SET(&(e->properties[1]), NULL, 0, NULL);
                             e->properties[1].expr->op      = JDF_STRING;
                             e->properties[1].expr->jdf_var = strdup(true_item->var);

                             if ( jdf->globals == NULL) {
                                  jdf->globals = e;
                             } else {
                                 prev->next = e;
                             }
                             prev = e;
                         });
    return;
}

/*
 * Traverse the list of variable definitions to see if we have stored a definition for a given variable.
 * Return the value one if "param" is in the list and the value zero if it is not.
 */
static int is_definition_seen(dague_list_t *var_def_list, char *param){
    int i = 0;
    DAGUE_ULIST_ITERATOR(var_def_list, item,
    {
        i++;
        var_def_item_t *true_item = (var_def_item_t *)item;
        assert( NULL != true_item->var );
        if( !strcmp(true_item->var, param) ) {
            return i;
        }
    });
    return 0;
}


/*
 * Add in the list of variable definitions an entry for the given parameter (the definition
 * itself is unnecessary, as we are using this list as a bitmask, in is_definition_seen().)
 */
static void mark_definition_as_seen(dague_list_t *var_def_list, char *param){
    var_def_item_t *new_list_item;

    new_list_item = (var_def_item_t *)calloc(1, sizeof(var_def_item_t));
    new_list_item->var = param;
    new_list_item->def = NULL; // we are not using the actual definition, just marking it as seen
    dague_ulist_lifo_push( var_def_list, (dague_list_item_t *)new_list_item );

    return;
}

/*
 * Traverse the tree containing the QUARK specific code and generate up to five strings.
 * prefix     : The variable declarations (and maybe initializations)
 * pool_pop   : The calls to dague_private_memory_pop() for SCRATCH parameters
 * kernel_call: The actual call to the kernel
 * printStr   : The call to printlog()
 * pool_push  : The calls to dague_private_memory_push() for SCRATCH parameters
 * result     : The concatenation of all the above strings that will be returned
 *
 * The function returns one string containing these five strings concatenated.
 */
char *quark_tree_to_body(node_t *node){
    char *result=NULL, *kernel_call, *prefix=NULL, *tmp;
    char *printStr, *printSuffix;
    char *pool_pop = NULL;
    char *pool_push = NULL;
    int i, j;
    int pool_buf_count = 0;

    dague_list_t var_def_list;
    dague_list_construct(&var_def_list);

    assert( FCALL == node->type );

    //dump_st(node->symtab);

    // Get the name of the function called from the tree.
    kernel_call = tree_to_str(node->u.kids.kids[2]);

    // Remove the suffix
    tmp = strstr(kernel_call, "_quark");
    if( NULL != tmp ){
        *tmp = '\0';
    }

    // Form the printlog string first, because it needs to use the function name in "kernel_call", and only
    // then change "kernel_call" to add the "#line" directive.

    // Form the string for the "printlog"
    printStr = strdup("  printlog(\"");
    printStr = append_to_string( printStr, kernel_call, "%s(", 1+strlen(kernel_call));
    for(i=0; NULL != node->task->ind_vars[i]; i++ ){
        if( i > 0 )
            printStr = append_to_string( printStr, ", ", NULL, 0);
        printStr = append_to_string( printStr, "%d", NULL, 0);
    }
    printStr = append_to_string( printStr, ")\\n\"\n           \"\\t(", NULL, 0);

    // If asked by the user, create the "#line lineno" directive and append a newline at the end.
    if(_q2j_generate_line_numbers){
        tmp = int_to_str(node->lineno);
        tmp = append_to_string(strdup("#line "), tmp, NULL, 0);
        tmp = append_to_string(tmp, q2j_input_file_name, " \"%s\"\n", 4+strlen(q2j_input_file_name));
    }else{
        tmp = NULL;
    }
    // Append the call to the kernel after the directive.
    kernel_call = append_to_string(tmp, kernel_call, "  %s(", 3+strlen(kernel_call));


    // Form the string for the suffix of the "printlog". That is whatever follows the format string, or in
    // other words the variables whose value we are interested in instead of the name.

    printSuffix = strdup(")\\n\",\n  ");

    for(i=0; NULL != node->task->ind_vars[i]; i++ ){
        char *iv = node->task->ind_vars[i];
        if (i == 0)
            printSuffix = append_to_string( printSuffix, iv, "%s", 2+strlen(iv));
        else
            printSuffix = append_to_string( printSuffix, iv, ", %s", 2+strlen(iv));
    }

    // Form the string for the actual function-call as well as the prefix, which is all
    // the definitions of the variables found in the call. Also generate declarations for
    // the variables based on their types.
    j=0;
    for(i=QUARK_FIRST_VAR; i<node->u.kids.kid_count; i+=QUARK_ELEMS_PER_LINE){
        char *param;
        node_t *var_node;
        if( j > 0 ){
            kernel_call = append_to_string( kernel_call, ", ", NULL, 0);
            printStr = append_to_string( printStr, ", ", NULL, 0);
        }
        if( j && !(j%3) )
            kernel_call = append_to_string( kernel_call, "\n\t", NULL, 0);

        // Get the next useful parameter and see if it's pass by VALUE (in which case we need to ignore the "&")
        param = NULL;
        var_node = NULL;
        if( (i+1<node->u.kids.kid_count) && !strcmp(tree_to_str(node->u.kids.kids[i+1]), "VALUE") ){
            if( EXPR == node->u.kids.kids[i]->type ){
                node_t *exp_node = node->u.kids.kids[i];
                // if the expression starts with "&", then take the remaining part of the expression (so, ignore the "&").
                if( ADDR_OF == exp_node->u.kids.kids[0]->type ){
                    var_node = exp_node->u.kids.kids[1];
                    if( NULL != var_node ){
                        param = tree_to_str(var_node);
                    }else{
                        fprintf(stderr,"WARNING: In quark_tree_to_body(), ADDR_OF node does not have an expression as kid\n");
                        fprintf(stderr,"WARNING: dumping node:\n%s\n",tree_to_str(exp_node));
                    }
                }
            }

            if( NULL != var_node && NULL != param ){
                // Find the type of the variable, so we can emmit a proper declaration (e.g. int x=3;).
                char *type_name = NULL;
                if( IDENTIFIER == var_node->type && NULL != var_node->u.var_name && NULL != var_node->symtab){
                    type_name = st_type_of_variable(var_node->u.var_name, var_node->symtab);
#ifdef EMMIT_WARNINGS
                    if( NULL == type_name ){
                        printf("WARNING: %s has an ST but no type!\n", var_node->u.var_name);
#  ifdef DEBUG_3
                    }else{
                        printf("%s is of type \"%s\"\n", var_node->u.var_name, type_name);
#  endif
                    }
#endif
                }

                // If we haven't seen this parameter before, see if it's defined and copy the definition into the body
                if( 0 == is_definition_seen(&var_def_list, param) ){
                    tmp = find_definition(param, node);
                    if( tmp != param ){
                        prefix = append_to_string( prefix, "  ", NULL, 0);
                        if( NULL != type_name )
                            prefix = append_to_string( prefix, type_name, "%s ", 1+strlen(tmp));
                        prefix = append_to_string( prefix, tmp, "%s;\n", 2+strlen(tmp));

                        // Add the definition into the list, so we don't emmit it again.
                        mark_definition_as_seen(&var_def_list, param);
                    }
                }
                kernel_call = append_to_string( kernel_call, param, NULL, 0);
            }
        }else if( (i+1<node->u.kids.kid_count) && !strcmp(tree_to_str(node->u.kids.kids[i+1]), "SCRATCH") ){
            char *pool_name = size_to_pool_name( tree_to_str(node->u.kids.kids[i-1]) );
            char *id = numToSymName(pool_buf_count);
            param = append_to_string( param, id, "p_elem_%s", 7+strlen(id));
            pool_pop = append_to_string( pool_pop, param, "  void *%s = ", 16+strlen(param));
            pool_pop = append_to_string( pool_pop, pool_name, "dague_private_memory_pop( %s );\n", 31+strlen(pool_name));
            pool_push = append_to_string( pool_push, pool_name, "  dague_private_memory_push( %s", 35+strlen(pool_name));
            pool_push = append_to_string( pool_push, param, ", %s );\n", 6+strlen(param));

            kernel_call = append_to_string( kernel_call, param, NULL, 0);

            // Every SCRATCH parameter will need a different buffer from the pool,
            // regardles of how many pools the buffers will belong to.
            pool_buf_count++;

            free(id);
        }else{
            char *symname = node->u.kids.kids[i]->var_symname;
            assert(NULL != symname);
            param = tree_to_str(node->u.kids.kids[i]);
            kernel_call = append_to_string( kernel_call, symname, NULL, 0);
            /*
             * JDF & QUARK specific optimization:
             * Add the keyword _q2j_data_prefix infront of the matrix to
             * differentiate the matrix from the struct.
             */
            kernel_call = append_to_string( kernel_call, _q2j_data_prefix, " /* %s", 4+strlen(_q2j_data_prefix));
            kernel_call = append_to_string( kernel_call, param, "%s */", 3+strlen(param));
        }

        // Add the parameter to the string of the printlog.  If the parameter is an array, we need to
        // do a little more work to print the pointer and the value of the indices instead of their names.
        if( ARRAY == node->u.kids.kids[i]->type ){
            node_t *arr = node->u.kids.kids[i];
            char *base_name = tree_to_str(arr->u.kids.kids[0]);
            printStr = append_to_string( printStr, base_name, "%s(%%d,%%d)[%%p]", 11+strlen(base_name));
            for(int ii=1; ii<arr->u.kids.kid_count; ii++){
                char *var_str = tree_to_str(arr->u.kids.kids[ii]);
                printSuffix = append_to_string( printSuffix, var_str, ", %s", 2+strlen(var_str));
            }
            char *alias = arr->var_symname;
            printSuffix = append_to_string( printSuffix, alias, ", %s", 2+strlen(base_name));
        }else{
            printStr = append_to_string( printStr, param, NULL, 0);
        }

        j++;
    }
    kernel_call = append_to_string( kernel_call, " );", NULL, 0);

    // Finalize printStr by append the suffix to it.
    printStr = append_to_string( printStr, printSuffix, NULL, 0);
    printStr = append_to_string( printStr, ");", NULL, 0);

    // Form the result by concatenating the strings we created in the right order.
    result = append_to_string(result, prefix, NULL, 0);
    result = append_to_string(result, printStr, "\n%s", 1+strlen(printStr));
    result = append_to_string(result, "\n  DRYRUN(\n", NULL, 0);
    if( NULL != pool_pop )
        result = append_to_string(result, pool_pop, "  %s", 2+strlen(pool_pop) );
    result = append_to_string(result, kernel_call, "\n  %s", 3+strlen(kernel_call) );
    if( NULL != pool_push )
        result = append_to_string(result, pool_push, "\n\n  %s", 4+strlen(pool_push) );
    result = append_to_string(result, "\n  );\n", NULL, 0); // close the DRYRUN

    // clean up the list of variables and their definitions
    var_def_item_t *item;
    while( NULL != (item = (var_def_item_t *)dague_ulist_lifo_pop(&var_def_list)) ) {
        free(item);
    }

    return result;
}

/*
 * The second parameter, "subs", holds an array of str_pair_t whose last element contains NULL values.
 */
static inline const char *return_string_or_substitute(char *str, str_pair_t *subs){
    str_pair_t *curr;

    if( NULL == subs ){
        return str;
    }

    for(curr=subs; NULL != curr->str1; curr++){
        if( !strcmp(curr->str1, str) )
            return curr->str2;
    }

    return str;
}

char *tree_to_str(node_t *node){
    return tree_to_str_with_substitutions(node, NULL);
}

char *tree_to_str_with_substitutions(node_t *node, str_pair_t *subs){
    int i, kid_count;
    char prfx[16], *str=NULL;

    if( NULL == node )
        return strdup("nil");
    if( EMPTY == node->type )
        return NULL;

    kid_count = node->u.kids.kid_count;

    if( BLOCK == node->type ){
        node_t *tmp;
        for(tmp=node->u.block.first; NULL != tmp; tmp = tmp->next){
            char *tmp_str, *ws;

            if( tmp->type == EMPTY )
                continue;

            ws = (char *)calloc(_ind_depth+1, sizeof(char));
            sprintf(ws, "%*s", _ind_depth, " ");
            str = append_to_string(str, ws, NULL, 0);
            free(ws);

            tmp_str = tree_to_str_with_substitutions(tmp, subs);
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
                if( NULL != node->var_type ){
                    // I don't think this code does anything any more.
                    // It was an early hack due to lack of symbol table.
                    str = append_to_string(strdup("("), node->var_type, NULL, 0);
                    str = append_to_string(str, ")", NULL, 0);
                }
                /*
                 * JDF & QUARK specific optimization:
                 * Add the keyword "desc" infront of the variable to
                 * differentiate the matrix from the struct.
                 */
                if( (NULL == node->parent) || (ARRAY != node->parent->type) ){
                    char *type = st_type_of_variable(node->u.var_name, node->symtab);
                    if( (NULL != type) && !strcmp("PLASMA_desc", type) ){
                        str = strdup("desc");
                    }
                }

                tmp = (char *)return_string_or_substitute(node->u.var_name, subs);

                return append_to_string(str, strdup(tmp), NULL, 0);

            case INTCONSTANT:
                if( NULL != node->var_type ){
                    int len = 24+strlen(node->var_type)+2;
                    tmp = (char *)calloc(len, sizeof(char));
                    snprintf(tmp, len, "(%s)%"PRIu64, node->var_type, node->const_val.i64_value);
                }else{
                    tmp = (char *)calloc(24,sizeof(char));
                    snprintf(tmp, 24, "%"PRIu64, node->const_val.i64_value);
                }
                return tmp;

            case FLOATCONSTANT: 
                if( NULL != node->var_type ){
                    int len = 32+strlen(node->var_type)+2;
                    tmp = (char *)calloc(len, sizeof(char));
                    snprintf(tmp, len, "(%s)%lf", node->var_type,node->const_val.f64_value);
                }else{
                    tmp = (char *)calloc(32,sizeof(char));
                    snprintf(tmp, 32, "%lf", node->const_val.f64_value);
                }
                return tmp;

            case STRING_LITERAL:
                return strdup(node->const_val.str);

            case INC_OP:
                return strdup("++");

            case DEC_OP:
                return strdup("--");

            case SIZEOF:
                str = strdup("sizeof(");
                if(node->u.kids.kid_count ){
                    str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0 );
                }else{
                    str = append_to_string( str, node->u.var_name, NULL, 0 );
                }
                str = append_to_string( str, ")", NULL, 0 );
                return str;

            case EXPR:
                if( NULL != node->var_type ){
                    str = append_to_string(strdup("("), node->var_type, NULL, 0);
                    str = append_to_string(str, ")", NULL, 0);
                }
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0);
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
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
                lhs = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                rhs = tree_to_str_with_substitutions(node->u.kids.kids[1], subs);

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
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " = ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case MUL_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " *= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case DIV_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " /= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case MOD_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " %= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case ADD_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " += ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case SUB_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " -= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case LEFT_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " <<= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case RIGHT_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " >>= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case AND_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " &= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case XOR_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " ^= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case OR_ASSIGN:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, " |= ", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case S_U_MEMBER:
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0 );
                str = append_to_string( str, ".", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case PTR_OP:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                str = append_to_string( str, "->", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                return str;

            case COND:
                str = strdup("(");
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0 );
                str = append_to_string( str, ") ? (", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                str = append_to_string( str, ") : (", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[2], subs), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0 );
                return str;

            case EQ_OP:
                str = strdup("(");
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0 );
                str = append_to_string( str, ")==(", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0);
                return str;

            case NE_OP:
                str = strdup("(");
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[0], subs), NULL, 0 );
                str = append_to_string( str, ")!=(", NULL, 0 );
                str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[1], subs), NULL, 0 );
                str = append_to_string( str, ")", NULL, 0);
                return str;

            case FOR:
                str = strdup("for( ");
                for(i=0; i<kid_count-1; ++i){
                    if(i>0)
                        str = append_to_string( str, "; ", NULL, 0);
                    str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), NULL, 0 );
                }
                str = append_to_string( str, ") {\n", NULL, 0);
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str_with_substitutions(DA_for_body(node), subs), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}\n", NULL, 0);
                return str;

            case WHILE:
                str = strdup("while( ");
                str = append_to_string( str, tree_to_str_with_substitutions(DA_while_cond(node), subs), NULL, 0 );
                str = append_to_string( str, " ) {\n", NULL, 0);
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str_with_substitutions(DA_while_body(node), subs), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}\n", NULL, 0);
                return str;

            case DO:
                str = strdup("do{\n");
                _ind_depth += 4;
                str = append_to_string( str, tree_to_str_with_substitutions(DA_do_body(node), subs), NULL, 0 );
                _ind_depth -= 4;
                for(i=0; i<_ind_depth; i+=4){
                    str = append_to_string(str, "    ", NULL, 0);
                }
                str = append_to_string( str, "}while( ", NULL, 0);
                str = append_to_string( str, tree_to_str_with_substitutions(DA_do_cond(node), subs), NULL, 0 );
                str = append_to_string( str, " );\n", NULL, 0);
                return str;

            case FCALL:
                for(j=1; j<=3; j++){
                    max_arg_len[j] = -1;
                    for(i=j; i<node->u.kids.kid_count; i+=3){
                        int tmp2;
                        char *arg = tree_to_str_with_substitutions(node->u.kids.kids[i], subs);
                    
                        tmp2 = strlen(arg);
                        free(arg);
                        if( tmp2 > max_arg_len[j] )
                            max_arg_len[j] = tmp2;
                    }
                }
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
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
                        str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), fmt, len+1 );
                    }else{
                        str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), NULL, 0);
                    }
                }
                str = append_to_string( str, " )", NULL, 0);
                return str;

            case ARRAY:
                str = tree_to_str_with_substitutions(node->u.kids.kids[0], subs);
                if( JDF_NOTATION ){
                    str = append_to_string( str, "(", NULL, 0);
                    for(i=1; i<node->u.kids.kid_count; ++i){
                        if( i > 1 ) 
                            str = append_to_string( str, ",", NULL, 0);
                        str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), NULL, 0 );
                    }
                    str = append_to_string( str, ")", NULL, 0);
                }else{
                    for(i=1; i<node->u.kids.kid_count; ++i){
                        str = append_to_string( str, "[", NULL, 0);
                        str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), NULL, 0 );
                        str = append_to_string( str, "]", NULL, 0);
                    }

                }
                return str;


            default:
                snprintf(prfx, 12, "|>%u<| ", node->type);
                str = append_to_string(NULL, prfx, NULL, 0);
                snprintf(prfx, 15, "kid_count: %d {{", kid_count);
                str = append_to_string(str, prfx, NULL, 0);
                _ind_depth += 4;
                for(i=0; i<kid_count; ++i){
                    if( i > 0 )
                        str = append_to_string( str, " ## ", NULL, 0 );
                    str = append_to_string( str, tree_to_str_with_substitutions(node->u.kids.kids[i], subs), NULL, 0 );
                }
                _ind_depth -= 4;
                str = append_to_string( str, "}}", NULL, 0 );
                break;
        }
    }

    return str;
}

node_t *node_to_ptr(node_t node){
    node_t *tmp = (node_t *)calloc(1, sizeof(node_t));
    *tmp = node;
    return tmp;
}

