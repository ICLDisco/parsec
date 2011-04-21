/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include "symtab.h"

extern int yyparse (void);
char *dague_input_file_name=NULL;
extern FILE *yyin;

int main(int argc, char **argv){
    if( argc < 2 ){
        fprintf(stderr,"Usage: %s file_name.c\n",argv[0]);
        exit(1);
    }
    dague_input_file_name = argv[1];

    yyin = fopen(dague_input_file_name, "r");
    (void)st_init_symtab();
    return yyparse();
    fclose(yyin);
}
