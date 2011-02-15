/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include "symtab.h"

extern int yyparse (void);

int main(int argc, char **argv){
    // Just to make the compiler shut up about unused parameters
    if( argc < 0 && argv[0][0] == '\0' )
        return yyparse();

    (void)st_init_symtab();
    return yyparse();
}
