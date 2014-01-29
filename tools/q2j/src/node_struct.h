/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DA_NODE_STRUCT_H_
#define _DA_NODE_STRUCT_H_
#include "symtab.h"
#include <stdint.h>

#define EMPTY       0x0000

#define ADDR_OF     0x1001 // 4097
#define STAR        0x1002 // 4098
#define PLUS        0x1003 // 4099
#define MINUS       0x1004 // 4100
#define TILDA       0x1005 // 4101
#define BANG        0x1006 // 4102
#define ASSIGN      0x1007 // 4103
#define COND        0x1008 // 4104
#define ARRAY       0x1009 // 4105
#define FCALL       0x100a // 4106
#define COND_DATA   0x100b // 4107
#define TASK_DEP    0x100c // 4108
#define PARAM_RANGE 0x100d // 4109

#define ENTRY       0x1010 // 4112
#define EXIT        0x1011 // 4113
#define BLKBOX_TASK 0x1012 // 4114
#define BLKBOX_TASK_ESPACE 0x1013 // 4115
#define BLKBOX_TASK_PARAMS 0x1014 // 4116
#define BLKBOX_TASK_DEPS   0x1015 // 4117


#define EXPR        0x1100 // 4352
#define ADD         0x1101 // 4353
#define SUB         0x1102 // 4354
#define MUL         0x1103 // 4355
#define DIV         0x1104 // 4356
#define MOD         0x1105 // 4357
#define B_AND       0x1106 // 4358
#define B_XOR       0x1107 // 4359
#define B_OR        0x1108 // 4360
#define LSHIFT      0x1109 // 4361
#define RSHIFT      0x110a // 4362
#define LT          0x110b // 4363
#define GT          0x110c // 4364
#define LE          0x110d // 4365
#define GE          0x110e // 4366
#define DEREF       0x110f // 4367
#define S_U_MEMBER  0x1110 // 4368
#define COMMA_EXPR  0x1111 // 4369
#define FUNC        0x1112 // 4370
#define COMMENT     0x1113 // 4371

#define TMP         0xFFFE // 65534
#define BLOCK       0xFFFF // 65535

typedef struct str_pair{
    const char *str1;
    const char *str2;
}str_pair_t;

typedef struct _task_t task_t;
typedef struct _node node_t;

typedef struct type_node{
    char *type;
    node_t *var;
}type_node_t;

struct _task_t{
    node_t *task_node;
    char ** ind_vars;
};

struct _node{
    uint32_t type;

    uint32_t lineno;

    node_t *next;
    node_t *prev;

    node_t *parent;
    node_t *enclosing_loop;
    node_t *enclosing_if;

    char *var_symname;

    symtab_t *symtab;

    task_t *task;
    jdf_function_entry_t *function;

    uint64_t trip_count;
    uint64_t loop_depth;

    union{
        uint64_t i64_value;
        double   f64_value;
        char     *str;
    }const_val;

    // This is a temporary hack since we don't have a symbol table
    char *var_type;

    union{
        struct{
            node_t *first;
            node_t *last;
        }block;

        struct{
            node_t **kids;
            int kid_count;
        }kids;

        char *var_name;
    }u;

    node_t *pragmas;
};

#endif
