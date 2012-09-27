/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symtab.h"
#include "starpu_struct.h"

extern int yyparse (void);
char *q2j_input_file_name=NULL;
int _q2j_produce_shmem_jdf = 0;
int _q2j_verbose_warnings  = 0;
int _q2j_add_phony_tasks   = 0;
int _q2j_generate_line_numbers = 0;
extern FILE *yyin;

void usage(char *pname);

void usage(char *pname){
    fprintf(stderr,"Usage: %s [-shmem] [-phony_tasks] [-line_numbers] [-v] file_name.c\n",pname);
    exit(1);
}

int main(int argc, char **argv){
    int tmp_return;
    while(--argc > 0){
        if( argv[argc][0] == '-' ){
            if( !strcmp(argv[argc],"-shmem") ){
                _q2j_produce_shmem_jdf = 1;
            }else if( !strcmp(argv[argc],"-phony_tasks") ){
                _q2j_add_phony_tasks = 1;
            }else if( !strcmp(argv[argc],"-line_numbers") ){
                _q2j_generate_line_numbers = 1;
            }else if( !strcmp(argv[argc],"-v") ){
                _q2j_verbose_warnings = 1;
            }else{
                usage(argv[0]);
            }
        }else{
            q2j_input_file_name = argv[argc];
        }
    }

    yyin = fopen(q2j_input_file_name, "r");
    if( NULL == yyin ){
        fprintf(stderr,"Cannot open file \"%s\"\n",q2j_input_file_name);
        return -1;
    }
    
    (void)st_init_symtab();
    tmp_return = yyparse();
    return tmp_return;
}
