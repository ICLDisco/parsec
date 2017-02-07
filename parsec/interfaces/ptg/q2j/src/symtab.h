#ifndef SYMTAB_H_HAS_BEEN_INCLUDED
#define SYMTAB_H_HAS_BEEN_INCLUDED

typedef struct q2j_symbol_t q2j_symbol_t;

struct q2j_symbol_t{
    char *var_name;
    char *var_type;
    q2j_symbol_t *next;
};

typedef struct symtab_t symtab_t;
    
struct symtab_t{
    symtab_t *parent;
    q2j_symbol_t *symbols;
};

symtab_t *st_get_current_st(void);
void st_init_symtab(void);
symtab_t *st_enter_new_scope(void);
symtab_t *st_exit_scope(void);
void st_insert_new_variable(char *var, char *type);
char *st_type_of_variable(char *var, symtab_t *scope);
void dump_st(symtab_t *scope);

#endif
