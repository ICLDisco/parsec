#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <symtab.h>

static inline symbol_t *create_symtab_entry(char *var, char *type);
static symtab_t *_current_st=NULL;

symtab_t *st_get_current_st(void){
    return(_current_st);
}

////////////////////////////////////////////////////////////////////////////////
//
void st_init_symtab(void){
    // silently ignore multiple initialization requests
    if( NULL == _current_st ){
        _current_st = (symtab_t *)calloc(1, sizeof(symtab_t));
        _current_st->parent = NULL;
        _current_st->symbols = NULL;
    }
    return;
}

////////////////////////////////////////////////////////////////////////////////
//
symtab_t *st_enter_new_scope(void){
    symtab_t *new_scope;

    assert( NULL != _current_st );

    new_scope = (symtab_t *)calloc(1, sizeof(symtab_t));
    new_scope->parent = _current_st;
    new_scope->symbols = NULL;

    _current_st = new_scope;
    return(_current_st);
}

////////////////////////////////////////////////////////////////////////////////
//
symtab_t *st_exit_scope(void){

    assert( NULL != _current_st );
    // This should be true, because we never exit the global scope
    assert( NULL != _current_st->parent );

    _current_st = _current_st->parent;
    return(_current_st);
}

////////////////////////////////////////////////////////////////////////////////
//
void st_insert_new_variable(char *var, char *type){
    symbol_t *sym, *prev;
    
    assert( NULL != _current_st );

    // If this scope is empty, add the new variable to the head
    if( NULL == _current_st->symbols ){
        _current_st->symbols = create_symtab_entry(var, type);
        return;
    }

    // Otherwise, make sure it is not already there, and put it at the end of the list.
    sym = _current_st->symbols;
    prev = sym;
    for( ; NULL!=sym; prev=sym, sym=sym->next){
        if( !strcmp(sym->var_name, var) ){
            fprintf(stderr,"Variable: %s already defined as type: \"%s\"\n",var, sym->var_type);
        }
    }

    prev->next = create_symtab_entry(var, type);
    return;
}

////////////////////////////////////////////////////////////////////////////////
//
void dump_st(symtab_t *scope){
    symbol_t *sym;

    do{
        for(sym=scope->symbols; NULL!=sym; sym=sym->next){
            printf("%s [type = \"%s\"]\n",sym->var_name, sym->var_type);
        }
        scope = scope->parent;
    }while(NULL != scope);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//
char *st_type_of_variable(char *var, symtab_t *scope){
    symbol_t *sym;

    if( NULL == scope ){
        return NULL;
    }

    do{
        for(sym=scope->symbols; NULL!=sym; sym=sym->next){
            if( !strcmp(sym->var_name,var) ){
                return sym->var_type;
            }
        }
        scope = scope->parent;
    }while(NULL != scope);

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
//
static inline symbol_t *create_symtab_entry(char *var, char *type){
    symbol_t *sym;
    sym = (symbol_t *)calloc(1, sizeof(symbol_t));
    sym->var_name = strdup(var);
    sym->var_type = strdup(type);
    sym->next = NULL;
    return sym;
}
