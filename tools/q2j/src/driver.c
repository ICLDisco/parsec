/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>

extern int yyparse (void);

int main(int argc, char **argv){
    return yyparse();
}
