#ifndef _STARPU_STRUCTURES
#define _STARPU_STRUCTURES
#include <stdint.h>
#include <inttypes.h>
#include "node_struct.h"
#include "symtab.h"

#define CODELET_MODE   0
#define CODELET_WHERE  1
#define CODELET_NBUFF  2
#define CODELET_CPU    3
#define CODELET_CUDA   4

#define FLAG_CPU       1
#define FLAG_CUDA      0

typedef struct StarPU_codelet_t          StarPU_codelet;
typedef struct StarPU_param_list_t       StarPU_param_list;
typedef struct StarPU_param_t            StarPU_param;
typedef struct StarPU_codelet_list_t     StarPU_codelet_list;
typedef struct StarPU_function_list_t    StarPU_function_list;
typedef struct StarPU_translation_list_t StarPU_translation_list;
typedef struct StarPU_fun_decl_t         StarPU_fun_decl;
typedef struct StarPU_task_list_t        StarPU_task_list;
typedef struct StarPU_task_t             StarPU_task;
typedef struct StarPU_variable_list_t    StarPU_variable_list;


struct StarPU_function_list_t {
    char *name;
    StarPU_function_list *next;
};


struct StarPU_param_t {
    union {
	char *modes;
	char *where;
	uint64_t nbuffers;
	StarPU_function_list *l;
    } p;
    int type;
};


struct StarPU_param_list_t {
    StarPU_param      *p;
    StarPU_param_list *next;
};


struct StarPU_codelet_t {
    char              *name;
    StarPU_param_list *l;
};

struct StarPU_codelet_list_t {
    StarPU_codelet      *cl;
    StarPU_codelet_list *prev;
    StarPU_codelet_list *next; 
};

struct StarPU_translation_list_t {
    StarPU_translation_list *next;
    StarPU_translation_list *prev;
    StarPU_fun_decl         *tr;
};

struct StarPU_fun_decl_t {
    char *name;
    char *buffer;
    node_t *node;
};


struct StarPU_task_list_t {
    StarPU_task *t;
    StarPU_task_list *next;
};

struct StarPU_variable_list_t {
    char *call_name;
    char *new_name;
    char *cuda_name;
    uint64_t pos;
    StarPU_variable_list *next;
};

struct StarPU_task_t {
    char *name;
    char *buffer;
    symtab_t *symtab;
    StarPU_fun_decl *cpu;
    StarPU_fun_decl *cuda;
    StarPU_variable_list *vl;
    StarPU_variable_list *glob;
    StarPU_variable_list *cpudecl;
    StarPU_variable_list *cudadecl;
};

StarPU_codelet_list     *codelet_list;
StarPU_translation_list *trans_list;
StarPU_task_list        *starpu_task_list;

void print_param_list(StarPU_param_list *l);
void print_codelet(StarPU_codelet *cl);
void print_codelet_list(StarPU_codelet_list *l);



#endif // STARPU_STRUCTURES
