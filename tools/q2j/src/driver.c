/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "symtab.h"
#include "jdf.h"
#include "utility.h"
#include "q2j.y.h"

extern int yyparse (void);

char *q2j_input_file_name  = NULL;
int  _q2j_annot_API         = Q2J_ANN_UNSET;
int  _q2j_dump_mapping      = 0;
int  _q2j_paranoid_cond     = 0;
int  _q2j_antidep_level     = 0;
int  _q2j_direct_output     = 0;
int  _q2j_add_phony_tasks   = 1;
int  _q2j_verbose_warnings  = 0;
int  _q2j_produce_shmem_jdf = 1;
int  _q2j_finalize_antideps = 0;
int  _q2j_generate_line_numbers   = 0;
int  _q2j_check_unknown_functions = 0;

FILE *_q2j_output;
jdf_t _q2j_jdf;

static volatile int _keep_waiting = 1;

/*
 * Add the keyword _q2j_data_prefix infront of the matrix name to
 * differentiate the matrix from the data used in the kernels.
 */
char *_q2j_data_prefix = "data";

extern FILE *yyin;

node_t *_q2j_func_list_head = NULL;

void usage(char *pname);
static void  read_conf_file(void);
static char *read_line(FILE *ifp);
static void  parse_line(char *line);
static void  sig_handler(int signum);

void usage(char *pname){
    fprintf(stderr,"Usage: %s [-shmem] [-phony_tasks] [-line_numbers] [-anti] [-v] file_1.c [file_2.c ... file_N.c]\n",pname);
    exit(1);
}

static char *read_line(FILE *ifp){
    char *line, *rslt;
    size_t len=128, i;
    int c;

    line = (char *)calloc(len,sizeof(int));

    for(i=0; 1 ;i++){
        c = getc(ifp);
        if( '\n' == c || EOF == c || feof(ifp) )
            break;
        if( len-1 == i ){
            len *= 2;
            line = realloc(line, len);
        }
        line[i] = (char)c;
    }
    if( feof(ifp) )
        return NULL;

    line[i] = '\0';
    rslt = strdup(line);
    free(line);

    return rslt;
}

static void parse_line(char *line){
    char *key, *tmp;
    int value;

    assert( NULL != line );

    tmp = strstr(line, "//");
    /* If the line begins with a comment, our job here is done. */
    if( tmp == line )
        return;
    /* Erase the comment from the line. */
    if( NULL != tmp )
        tmp = '\0';

    /* The key cannot be larger than the whole line. */
    key = (char *)calloc(strlen(line), sizeof(char));

    /* Break the "key = value" formated input into two parts. */
    sscanf(line, " %[^ =] = %d", key, &value);

    if( !strlen(key) ) {
        free(key);
        return;
    }
    if( !strcmp(key,"produce_shmem_jdf") ){
        _q2j_produce_shmem_jdf = value;
    }else if( !strcmp(key,"add_phony_tasks") ){
        _q2j_add_phony_tasks = value;
    }else if( !strcmp(key,"generate_line_numbers") ){
        _q2j_generate_line_numbers = value;
    }else if( !strcmp(key,"finalize_antideps") ){
        _q2j_finalize_antideps = value;
    }else if( !strcmp(key,"dump_mapping") ){
        _q2j_dump_mapping = value;
    }else if( !strcmp(key,"verbose_warnings") ){
        _q2j_verbose_warnings = value;
    }else if( !strcmp(key,"direct_output") ){
        _q2j_direct_output = value;
    }else if( !strcmp(key,"generate_paranoid_conditions") ){
        _q2j_paranoid_cond = value;
    }else if( !strcmp(key,"antidepentency_finalization_level") ){
        _q2j_antidep_level = value;
    }else if( !strcmp(key,"check_unknown_functions") ){
        _q2j_check_unknown_functions = value;
    }

    /* ignore silently unrecognized keys */
    free(key);
    return;
}

static void read_conf_file(){
    char *line, *home, *path, *file_name=".q2jrc";
    FILE *ifp;

    home = getenv("HOME");
    path = (char *)calloc(strlen(home)+strlen(file_name)+2, sizeof(char));
    sprintf(path, "%s/%s", home, file_name);

    ifp = fopen(path, "r");

    /* If the file doesn't exist, or cannot be opened, ignore it silently. */
    if( NULL == ifp )
        return;

    for( line = read_line(ifp); NULL != line; line = read_line(ifp) ){
        parse_line(line);
    }

    return;
}

static void sig_handler(int signum) {
    (void)signum;
    _keep_waiting = 0;
}

pid_t fork_and_continue_in_child(void){
    pid_t child, parent = -1;
    struct sigaction action;

    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    action.sa_handler = &sig_handler;
    if (sigaction(SIGUSR1, &action, 0)){
        perror("sigaction");
        abort();
    }

    if ((child = fork()) != 0){
        // Parent
        // The parent will exit this loop when the child sends a SIGUSR1 signal
        while( _keep_waiting ){
            pause();
        }
        exit(0);
    }else{
        // Child
        parent = getppid();
    }

    return parent;
}

int main(int argc, char **argv){
    pid_t parent;
    int arg, func_count=0;
    node_t *tmp, *q2j_target_func = NULL;
    char *q2j_func_name = NULL;

    _q2j_output = stdout;

    read_conf_file();

    for(arg=1; arg<argc; arg++){
        if( argv[arg][0] == '-' ){
            if( !strcmp(argv[arg],"-shmem") ){
                _q2j_produce_shmem_jdf = 1;
            }else if( !strcmp(argv[arg],"-no_shmem") ){
                _q2j_produce_shmem_jdf = 0;
            }else if( !strcmp(argv[arg],"-phony_tasks") ){
                _q2j_add_phony_tasks = 1;
            }else if( !strcmp(argv[arg],"-no_phony_tasks") ){
                _q2j_add_phony_tasks = 0;
            }else if( !strcmp(argv[arg],"-line_numbers") ){
                _q2j_generate_line_numbers = 1;
            }else if( !strcmp(argv[arg],"-anti") ){
                _q2j_finalize_antideps = 1;
            }else if( !strcmp(argv[arg],"-advanced_anti") ){
                _q2j_antidep_level = 3;
            }else if( !strcmp(argv[arg],"-mapping") ){
                _q2j_dump_mapping = 1;
            }else if( !strcmp(argv[arg],"-check_unknown") ){
                _q2j_check_unknown_functions = 1;
            }else if( !strcmp(argv[arg],"-func") && (arg+1<argc) ){
                q2j_func_name = argv[arg+1];
                arg++;
            }else if( !strcmp(argv[arg],"-v") ){
                _q2j_verbose_warnings = 1;
            }else{
                usage(argv[0]);
            }
        }else{
            break;
        }
    }

    _q2j_jdf.prologue  = NULL;
    _q2j_jdf.epilogue  = NULL;
    _q2j_jdf.globals   = NULL;
    _q2j_jdf.global_properties = NULL;
    _q2j_jdf.functions = NULL;
    _q2j_jdf.data      = NULL;
    _q2j_jdf.datatypes = NULL;
    _q2j_jdf.inline_c_functions = NULL;

    _q2j_func_list_head = NULL;
    (void)st_init_symtab();


    parent = fork_and_continue_in_child();

    /* Parse all files and generate ASTs for all functions found */
    for(; arg<argc; arg++){
        if( argv[arg][0] == '-' ){
            usage(argv[0]);
        }
        q2j_input_file_name = argv[arg];

        yyin = fopen(q2j_input_file_name, "r");
        if( NULL == yyin ){
            fprintf(stderr,"Cannot open file \"%s\"\n", q2j_input_file_name);
            return -1;
        }

        if( yyparse() > 0 ) {
            fprintf(stderr,"Parse error during processing of file: %s\n", q2j_input_file_name);
            exit(1);
        }
        fclose( yyin );
    }

    /* Find the subtree of the function we are supposed to analyze */
    if( NULL == q2j_func_name ){
        q2j_func_name = DA_func_name(_q2j_func_list_head);
    }
    q2j_target_func = NULL;
    func_count=0;
    for(tmp=_q2j_func_list_head; NULL != tmp; tmp = tmp->next){
        char *tmp_name = DA_func_name(tmp);
        func_count++;
        if( (NULL != tmp_name) && !strcmp(tmp_name, q2j_func_name) ){
            q2j_target_func = tmp;
        }
    }
    if( NULL == q2j_target_func ){
        fprintf(stderr,"Cannot find target function: %s\n",q2j_func_name);
        usage(argv[0]);
    }

    /* Detect if we are using the PLASMA specific QUARK_Insert_Task(), or the generic Inert_Task() mode */
    detect_annotation_mode(q2j_target_func);

    /* If necessary, inline function calls (i.e., to PLASMA functions) and adjust the bounds of the
     submatrix appropriately, if necessary */
    inline_function_calls(q2j_target_func, _q2j_func_list_head);

    /* Do the necessary conversions to bring the code in canonical form */
    rename_induction_variables(q2j_target_func);
    convert_OUTPUT_to_INOUT(q2j_target_func);

    if( _q2j_add_phony_tasks )
        add_entry_and_exit_task_loops(q2j_target_func);

    /* Analyze the dependencies and output the JDF representation */
    if( Q2J_SUCCESS != analyze_deps(q2j_target_func) ){
        printf("\n\n%s\n\n",tree_to_str(q2j_target_func));
    }

    kill(parent, SIGUSR1); /* tell parent to exit. */

    exit(EXIT_SUCCESS);
}
