/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
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
static void read_conf_file(void);
static char *read_line(FILE *ifp);
static void parse_line(char *line);

void usage(char *pname){
    fprintf(stderr,"Usage: %s [-shmem] [-phony_tasks] [-line_numbers] [-anti] [-v] file_name.c\n",pname);
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

    if( !strlen(key) )
        return;

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
    }

    /* ignore silently unrecognized keys */

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

int main(int argc, char **argv){

    _q2j_output = stdout;

    read_conf_file();

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
