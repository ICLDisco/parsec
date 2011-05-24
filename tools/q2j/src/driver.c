/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symtab.h"

extern int yyparse (void);
char *q2j_input_file_name=NULL;
int q2j_produce_shmem_jdf = 0;
extern FILE *yyin;

void usage(char *pname){
    fprintf(stderr,"Usage: %s [-shmem] file_name.c\n",pname);
    exit(1);
}

int main(int argc, char **argv){

    q2j_input_file_name = argv[1];
    if( argc == 3 ){
        if( !strcmp(argv[1],"-shmem") ){
            q2j_produce_shmem_jdf = 1;
            q2j_input_file_name = argv[2];
        }else if( !strcmp(argv[2],"-shmem") ){
            q2j_produce_shmem_jdf = 1;
            q2j_input_file_name = argv[1];
        }else{
            usage(argv[0]);
        }
    }
    if( argc < 2 ){
        usage(argv[0]);
    }

    yyin = fopen(q2j_input_file_name, "r");
    (void)st_init_symtab();
    return yyparse();
    fclose(yyin);
}
