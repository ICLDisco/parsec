/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _PARSE_UTILITY_H_
#define _PARSE_UTILITY_H_

#define HASH_TAB_SIZE 1024

unsigned long hash(char *str);
char *lookup_type(char *new_type);
void add_type(char *new_type, char *old_type);

typedef struct _type_list type_list_t;

struct _type_list{
    type_list_t *next;
    char *new_type;
    char *old_type;
};
#endif
