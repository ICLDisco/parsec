/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symtab.h"
#include "jdf.h"

extern int yyparse (void);

char *q2j_input_file_name      = NULL;
int _q2j_produce_shmem_jdf     = 0;
int _q2j_verbose_warnings      = 0;
int _q2j_add_phony_tasks       = 0;
int _q2j_finalize_antideps     = 0;
int _q2j_generate_line_numbers = 0;
int _q2j_dump_mapping          = 0;
int _q2j_direct_output         = 0;
FILE *_q2j_output;
jdf_t _q2j_jdf;

/* 
 * Add the keyword _q2j_data_prefix infront of the matrix name to
 * differentiate the matrix from the data used in the kernels.
 */
char *_q2j_data_prefix = "data";

extern FILE *yyin;

void usage(char *pname);

void usage(char *pname){
    fprintf(stderr,"Usage: %s [-shmem] [-phony_tasks] [-line_numbers] [-anti] [-v] file_name.c\n",pname);
    exit(1);
}

int main(int argc, char **argv){

    _q2j_output = stdout;

    while(--argc > 0){
        if( argv[argc][0] == '-' ){
            if( !strcmp(argv[argc],"-shmem") ){
                _q2j_produce_shmem_jdf = 1;
            }else if( !strcmp(argv[argc],"-phony_tasks") ){
                _q2j_add_phony_tasks = 1;
            }else if( !strcmp(argv[argc],"-line_numbers") ){
                _q2j_generate_line_numbers = 1;
            }else if( !strcmp(argv[argc],"-anti") ){
                _q2j_finalize_antideps = 1;
            }else if( !strcmp(argv[argc],"-mapping") ){
                _q2j_dump_mapping = 1;
            }else if( !strcmp(argv[argc],"-v") ){
                _q2j_verbose_warnings = 1;
            }else{
                usage(argv[0]);
            }
        }else{
            q2j_input_file_name = argv[argc];
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

    yyin = fopen(q2j_input_file_name, "r");
    if( NULL == yyin ){
        fprintf(stderr,"Cannot open file \"%s\"\n", q2j_input_file_name);
        return -1;
    }
    
/*     _q2j_output = fopen("output.jdf", "w"); */
/*     if( NULL == _q2j_output ){ */
/*         fprintf(stderr,"Cannot open file \"%s\"\n", q2j_input_file_name); */
/*         return -1; */
/*     } */

    (void)st_init_symtab();
    if( yyparse() > 0 ) {
        exit(1);
    }
    fclose( yyin );

    if (_q2j_output != stdout)
        fclose(_q2j_output);

    return EXIT_SUCCESS;
}
