#ifndef _DA_NODE_STRUCT_H_
#define _DA_NODE_STRUCT_H_
#include <stdint.h>

#define EMPTY       0x0000

#define ADDR_OF     0x1001 // 4096
#define STAR        0x1002
#define PLUS        0x1003
#define MINUS       0x1004
#define TILDA       0x1005
#define BANG        0x1006
#define ASSIGN      0x1007
#define COND        0x1008
#define ARRAY       0x1009
#define FCALL       0x100a

#define ENTRY       0x1010
#define EXIT        0x1011

#define EXPR        0x1100 // 4352
#define ADD         0x1101
#define SUB         0x1102
#define MUL         0x1103
#define DIV         0x1104
#define MOD         0x1105
#define B_AND       0x1106
#define B_XOR       0x1107
#define B_OR        0x1108
#define LSHIFT      0x1109
#define RSHIFT      0x110a
#define LT          0x110b
#define GT          0x110c
#define LE          0x110d
#define GE          0x111e
#define DEREF       0x111f
#define S_U_MEMBER  0x1110
#define COMMA_EXPR  0x1111

#define BLOCK       0xFFFF // 65535


typedef struct _task_t task_t;
typedef struct _node node_t;

struct _task_t{
    char *task_name;
    node_t *task_node;
    char ** ind_vars;
};

struct _node{
    uint32_t type;

    node_t *next;
    node_t *prev;

    node_t *parent;
    node_t *enclosing_loop;

    char *var_symname;

    task_t *task;

    uint64_t trip_count;
    uint64_t loop_depth;

    union{
        struct{
            node_t *first;
            node_t *last;
        }block;

        struct{
            node_t **kids;
            int kid_count;
        }kids;

        union{
            uint64_t i64_value;
            double   f64_value;
            char     *str;
        }const_val;

        char *var_name;
    }u;
};

#endif
