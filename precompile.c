#include "precompile.h"
#include "dplasma.h"
#include "symbol.h"
#include "expr.h"

#include <strings.h>

#define INIT_FUNC_BODY_SIZE 4096
#define FNAME_SIZE            64
#define DEPENDENCY_SIZE     1024
#define DPLASMA_SIZE        4096
#define SYMBOL_CODE_SIZE    4096
#define PARAM_CODE_SIZE     4096
#define DEP_CODE_SIZE       4096
#define DPLASMA_ALL_SIZE    8192

#define COLORS_SIZE           54

static const char *colors[COLORS_SIZE] = { 
  "rgb(0xE5, 0x2B, 0x50)", 
  "rgb(0xFF, 0xBF, 0x00)", 
  "rgb(0x7F, 0xFF, 0xD4)", 
  "rgb(0x00, 0x7F, 0xFF)", 
  "rgb(0x00, 0x00, 0x00)", 
  "rgb(0x00, 0x00, 0xFF)", 
  "rgb(0x00, 0x95, 0xB6)", 
  "rgb(0x8A, 0x2B, 0xE2)", 
  "rgb(0xA5, 0x2A, 0x2A)", 
  "rgb(0x70, 0x29, 0x63)", 
  "rgb(0x96, 0x00, 0x18)", 
  "rgb(0xDE, 0x31, 0x63)", 
  "rgb(0x00, 0x7B, 0xA7)", 
  "rgb(0x7F, 0xFF, 0x00)", 
  "rgb(0xF8, 0x83, 0x79)", 
  "rgb(0xDC, 0x14, 0x3C)", 
  "rgb(0x00, 0xFF, 0xFF)", 
  "rgb(0x7D, 0xF9, 0xFF)", 
  "rgb(0xFF, 0xD7, 0x00)", 
  "rgb(0x80, 0x80, 0x80)", 
  "rgb(0x00, 0xCC, 0x00)", 
  "rgb(0x3F, 0xFF, 0x00)", 
  "rgb(0x4B, 0x00, 0x82)", 
  "rgb(0x00, 0xA8, 0x6B)", 
  "rgb(0xB5, 0x7E, 0xDC)", 
  "rgb(0xC8, 0xA2, 0xC8)", 
  "rgb(0xBF, 0xFF, 0x00)", 
  "rgb(0xFF, 0x00, 0xFF)", 
  "rgb(0x80, 0x00, 0x00)", 
  "rgb(0xE0, 0xB0, 0xFF)", 
  "rgb(0x00, 0x00, 0x80)", 
  "rgb(0x80, 0x80, 0x00)", 
  "rgb(0xFF, 0xA5, 0x00)", 
  "rgb(0xFF, 0x45, 0x00)", 
  "rgb(0xFF, 0xE5, 0xB4)", 
  "rgb(0x1C, 0x39, 0xBB)", 
  "rgb(0xFF, 0xC0, 0xCB)", 
  "rgb(0x84, 0x31, 0x79)", 
  "rgb(0xFF, 0x75, 0x18)", 
  "rgb(0x80, 0x00, 0x80)", 
  "rgb(0xFF, 0x00, 0x00)", 
  "rgb(0xC7, 0x15, 0x85)", 
  "rgb(0xFF, 0x00, 0x7F)", 
  "rgb(0xFA, 0x80, 0x72)", 
  "rgb(0xFF, 0x24, 0x00)", 
  "rgb(0xC0, 0xC0, 0xC0)", 
  "rgb(0x70, 0x80, 0x90)", 
  "rgb(0x00, 0xFF, 0x7F)", 
  "rgb(0x48, 0x3C, 0x32)", 
  "rgb(0x00, 0x80, 0x80)", 
  "rgb(0x40, 0xE0, 0xD0)", 
  "rgb(0xEE, 0x82, 0xEE)", 
  "rgb(0x40, 0x82, 0x6D)", 
  "rgb(0xFF, 0xFF, 0x00)" 
};

typedef struct preamble_list {
    const char *language;
    const char *code;
    struct preamble_list *next;
} preamble_list_t;
static preamble_list_t *preambles = NULL;

typedef struct symb_list {
    symbol_t *s;
    char *c_name;
    struct symb_list *next;
} symb_list_t;

typedef struct dumped_dep_list {
    const dep_t *dep;
    char        *name;
    struct dumped_dep_list *next;
} dumped_dep_list_t;

typedef struct dumped_param_list {
    const param_t *param;
    unsigned int   idx;
    char          *param_name;
    struct dumped_param_list *next;
} dumped_param_list_t;

static char *dump_c_symbol(FILE *out, const symbol_t *s, char *init_func_body, int init_func_body_size);
static char *dump_c_param(FILE *out, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it);

void dplasma_precompiler_add_preamble(const char *language, const char *code)
{
    preamble_list_t *n = (preamble_list_t*)calloc(1, sizeof(preamble_list_t));
    n->language = language;
    n->code = code;
    n->next = preambles;
    preambles = n;
}

static void dump_inline_c_expression(FILE *out, const expr_t *e)
{
    if( e == NULL ) {
        return;
    } else {
        if( EXPR_OP_CONST_INT == e->op ) {
            fprintf(out, "%d", e->value);
        } 
        else if( EXPR_OP_SYMB == e->op ) {
            fprintf(out, "%s", e->var->name);
        } else if( EXPR_IS_UNARY(e->op) ) {
            if( e->op == EXPR_OP_UNARY_NOT ) {
                fprintf(out, "!(");
                dump_inline_c_expression(out, e->uop1);
                fprintf(out, ")");
            }
        } else if( EXPR_IS_BINARY(e->op) ) {
            fprintf(out, "(");
            dump_inline_c_expression(out, e->bop1);
            switch( e->op ) {
            case EXPR_OP_BINARY_MOD:            
                fprintf(out, ")%%(");
                break;
            case EXPR_OP_BINARY_EQUAL:
                fprintf(out, ")==(");
                break;
            case EXPR_OP_BINARY_NOT_EQUAL:
                fprintf(out, ")!=(");
                break;
            case EXPR_OP_BINARY_PLUS:
                fprintf(out, ")+(");
                break;
            case EXPR_OP_BINARY_RANGE:
                fprintf(stderr, "cannot evaluate range expression here!\n");
                break;
            case EXPR_OP_BINARY_MINUS:
                fprintf(out, ")-(");
                break;
            case EXPR_OP_BINARY_TIMES:
                fprintf(out, ")*(");
                break;
            case EXPR_OP_BINARY_DIV:
                fprintf(out, ")/(");
                break;
            case EXPR_OP_BINARY_OR:
                fprintf(out, ")||(");
                break;
            case EXPR_OP_BINARY_AND:
                fprintf(out, ")&&(");
                break;
            case EXPR_OP_BINARY_XOR:
                fprintf(out, ")^(");
                break;
            }
            dump_inline_c_expression(out, e->bop2);
            fprintf(out, ")");
        } else {
            fprintf(stderr, "Unkown operand %d in expression", e->op);
        }
    }
}

static char *dump_c_expression(FILE *out, const expr_t *e, char *init_func_body, int init_func_body_size)
{
    static unsigned int expr_idx = 0;
    static char name[FNAME_SIZE];

    if( e == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        int my_id = expr_idx;
        expr_idx++;

        if( EXPR_OP_CONST_INT == e->op ) {
            fprintf(out, "static expr_t expr%d = { .op = EXPR_OP_CONST_INT, .flags = %d, .value = %d }; /* ",
                    my_id, e->flags, e->value);
            expr_dump(out, e);
            fprintf(out, " */\n");
        } 
        else if( EXPR_OP_SYMB == e->op ) {
            char sname[FNAME_SIZE];
            snprintf(sname, FNAME_SIZE, "%s", dump_c_symbol(out, e->var, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                fprintf(out, "static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s, .value = %d }; /* ",
                        my_id, e->flags, sname, e->value);
                expr_dump(out, e);
                fprintf(out, " */\n");
            } else {
                fprintf(out, "static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s }; /* ",
                        my_id, e->flags, sname);
                expr_dump(out, e);
                fprintf(out, " */\n");
            }
        } else if( EXPR_IS_UNARY(e->op) ) {
            char sn[FNAME_SIZE];
            snprintf(sn, FNAME_SIZE, "%s", dump_c_expression(out, e->uop1, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s, .value = %d }; /* ", 
                        my_id, e->op, e->flags, sn, e->value);
                expr_dump(out, e);
                fprintf(out, " */\n");
            } else {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s }; /* ", 
                        my_id, e->op, e->flags, sn);
                expr_dump(out, e);
                fprintf(out, " */\n");
            }
        } else if( EXPR_IS_BINARY(e->op) ) {
            char sn1[FNAME_SIZE];
            char sn2[FNAME_SIZE];
            snprintf(sn1, FNAME_SIZE, "%s", dump_c_expression(out, e->bop1, init_func_body, init_func_body_size));
            snprintf(sn2, FNAME_SIZE, "%s", dump_c_expression(out, e->bop2, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                 fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .bop1 = %s, .bop2 = %s, .value = %d }; /* ", 
                         my_id, e->op, e->flags, sn1, sn2, e->value);
                 expr_dump(out, e);
                fprintf(out, " */\n");
            } else {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .bop1 = %s, .bop2 = %s }; /* ", 
                        my_id, e->op, e->flags, sn1, sn2);
                expr_dump(out, e);
                fprintf(out, " */\n");
            }
        } else {
            fprintf(stderr, "Unkown operand %d in expression", e->op);
        }

        snprintf(name, FNAME_SIZE, "&expr%d", my_id);
    }

    return name;
}

static char *dump_c_dep(FILE *out, const dep_t *d, char *init_func_body, int init_func_body_size)
{
    static unsigned int dep_idx = 0;
    static char name[FNAME_SIZE];
    static dumped_dep_list_t *dumped_deps;
    dumped_dep_list_t *dumped;
    int i;
    unsigned int my_idx;
    
    if( d == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        char whole[DEP_CODE_SIZE];
        int p = 0;

        for(dumped = dumped_deps; dumped != NULL; dumped = dumped->next) {
            if( dumped->dep == d ) {
                return dumped->name;
            }
        }

        my_idx = dep_idx++;
        dumped = (dumped_dep_list_t*)calloc(1, sizeof(dumped_dep_list_t));
        dumped->dep = d;
        dumped->next = dumped_deps;
        asprintf(&dumped->name, "&dep%d", my_idx);
        dumped_deps = dumped;
        
        p += snprintf(whole + p, DEP_CODE_SIZE-p, 
                      "static dep_t dep%d = { .cond = %s, .dplasma = NULL,\n"
                      "                       .call_params = {",
                      my_idx, dump_c_expression(out, d->cond, init_func_body, init_func_body_size));
        i = snprintf(init_func_body + strlen(init_func_body), init_func_body_size - strlen(init_func_body),
                     "  dep%d.dplasma = &dplasma_array[%d];\n", my_idx, dplasma_dplasma_index( d->dplasma ));
        if(i + strlen(init_func_body) >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        i = snprintf(init_func_body + strlen(init_func_body), init_func_body_size - strlen(init_func_body),
                     "  dep%d.param = %s;\n", my_idx, dump_c_param(out, d->param, init_func_body, init_func_body_size, 0));
        if(i + strlen(init_func_body) >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        for(i = 0 ; i < MAX_CALL_PARAM_COUNT; i++) {
            p += snprintf(whole + p, DEP_CODE_SIZE-p, "%s%s", dump_c_expression(out, d->call_params[i], init_func_body, init_func_body_size), 
                          i < MAX_CALL_PARAM_COUNT-1 ? ", " : "}};\n");
        }
        fprintf(out, "%s", whole);
        snprintf(name, FNAME_SIZE, "&dep%d", my_idx);
    }
     
   return name;
}

static char *dump_c_param(FILE *out, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it)
{
    static unsigned int param_idx = 0;
    static dumped_param_list_t *dumped_params = NULL;
    static char name[FNAME_SIZE];
    dumped_param_list_t *dumped;
    char param[PARAM_CODE_SIZE];
    int  l = 0;
    int i;
    char *dep_name;
    unsigned int my_idx;

    if( p == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        for(dumped = dumped_params; dumped != NULL; dumped = dumped->next) {
            if( dumped->param == p ) {
                if( !dump_it ) {
                    return dumped->param_name;
                } else {
                    my_idx = dumped->idx;
                    break;
                }
            }
        }

        if( dumped == NULL ) {
            my_idx = param_idx++;
            dumped = (dumped_param_list_t*)calloc(1, sizeof(dumped_param_list_t));
            dumped->param = p;
            dumped->idx = my_idx;
            asprintf(&dumped->param_name, "&param%d", my_idx);
            dumped->next = dumped_params;
            dumped_params = dumped;
            if( !dump_it ) {
                return dumped->param_name;
            }
        }

        l += snprintf(param + l, PARAM_CODE_SIZE-l, 
                      "static param_t param%d = { .name = \"%s\", .sym_type = %d, .param_mask = 0x%02x,\n"
                      "     .dep_in  = {", my_idx, p->name, p->sym_type, p->param_mask);
        for(i = 0; i < MAX_DEP_IN_COUNT; i++) {
            dep_name = dump_c_dep(out, p->dep_in[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_IN_COUNT-1 ? ", " : "},\n"
                          "     .dep_out = {");
        }
        for(i = 0; i < MAX_DEP_OUT_COUNT; i++) {
            dep_name = dump_c_dep(out, p->dep_out[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_OUT_COUNT-1 ? ", " : "} };\n");
        }
        fprintf(out, "%s", param);
        snprintf(name, FNAME_SIZE, "&param%d", my_idx);
    }

    return name;
}

static char *dump_c_symbol(FILE *out, const symbol_t *s, char *init_func_body, int init_func_body_size)
{
    static symb_list_t *already_dumped = NULL;
    int i;
    symb_list_t *e;
    char mn[FNAME_SIZE];
    char mm[FNAME_SIZE];
    
    /* Did we already dump this symbol (pointer-wise)? */
    for(i = 0, e=already_dumped; e != NULL; i++, e = e->next ) {
        if(e->s == s) {
            return e->c_name;
        }
    }
    
    e = (symb_list_t*)calloc(1, sizeof(symb_list_t));
    e->s = (symbol_t*)s;
    e->c_name = (char*)malloc(FNAME_SIZE);
    snprintf(e->c_name, FNAME_SIZE, "&symb%d", i);
    e->next = already_dumped;
    already_dumped = e;

    snprintf(mn, FNAME_SIZE, "%s", dump_c_expression(out, s->min, init_func_body, init_func_body_size));
    snprintf(mm, FNAME_SIZE, "%s", dump_c_expression(out, s->max, init_func_body, init_func_body_size));
    
    fprintf(out, "static symbol_t symb%d = { .flags = 0x%08x, .name = \"%s\", .min = %s, .max = %s };\n",
            i,
            s->flags, s->name, mn, mm);

    return e->c_name;
}

static void dump_all_global_symbols_c(FILE *out, char *init_func_body, int init_func_body_size)
{
    int i;
    char whole[SYMBOL_CODE_SIZE];
    int l = 0;
    l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "static symbol_t *dplasma_symbols[] = {\n");
    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "   %s%s", 
                      dump_c_symbol(out, dplasma_symbol_get_element_at(i), init_func_body, init_func_body_size),
                      i < dplasma_symbol_get_count()-1 ? ",\n" : "};\n");
    }
    fprintf(out, "%s", whole);

    fprintf(out, "\n");

    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        if( (dplasma_symbol_get_element_at(i)->min != NULL) &&
            (dplasma_symbol_get_element_at(i)->max != NULL) &&
            (dplasma_symbol_get_element_at(i)->min->flags & EXPR_FLAG_CONSTANT) &&
            (dplasma_symbol_get_element_at(i)->max->flags & EXPR_FLAG_CONSTANT) &&
            (dplasma_symbol_get_element_at(i)->min->value == dplasma_symbol_get_element_at(i)->max->value) ) {
            /* strangely enough, this should be always the case... TODO: talk with the others -- Thomas */
            fprintf(out, "int %s = %d;\n", dplasma_symbol_get_element_at(i)->name, dplasma_symbol_get_element_at(i)->min->value);
        } else {
            const char *name = dplasma_symbol_get_element_at(i)->name;
            fprintf(out, "int %s;\n", name);

            snprintf(init_func_body + strlen(init_func_body),
                     init_func_body_size - strlen(init_func_body),
                     "  {\n"
                     "    int rc;\n"
                     "    rc = expr_eval( (%s)->min, NULL, 0, &%s);\n"
                     "    if( 0 != rc ) {\n"
                     "      return rc;\n"
                     "    }\n"
                     "  }\n",
                     dump_c_symbol(out, dplasma_symbol_get_element_at(i), init_func_body, init_func_body_size), name);
        }
    }
    fprintf(out, "\n");
}

static char *dump_c_dependency_list(FILE *out, dplasma_dependencies_t *d, char *init_func_body, int init_func_body_size)
{
    static char dname[FNAME_SIZE];
    static int  ndx = 0;
    int my_idx = ndx++;
    char b[DEPENDENCY_SIZE];
    int p = 0;

    if( d == NULL ) {
        snprintf(dname, FNAME_SIZE, "NULL");
    } else {
        my_idx = ndx++;
        p += snprintf(b+p, DEPENDENCY_SIZE-p, "static struct dplasma_dependencies_t deplist%d = {\n", my_idx);
        p += snprintf(b+p, DEPENDENCY_SIZE-p, "  .flags = 0x%04x, .min = %d, .max = %d, .symbol = %s,\n",
                      d->flags, d->min, d->max, dump_c_symbol(out, d->symbol, init_func_body, init_func_body_size));
        if(  DPLASMA_DEPENDENCIES_FLAG_NEXT & d->flags ) {
            p += snprintf(b+p, DEPENDENCY_SIZE-p, "  .u.next = {%s} };\n", dump_c_dependency_list(out, d->u.next[0], init_func_body, init_func_body_size));
        } else {
            p += snprintf(b+p, DEPENDENCY_SIZE-p, "  .u.dependencies = { 0x%02x } };\n", d->u.dependencies[0]);
        }
        fprintf(out, "%s", b);
        snprintf(dname, FNAME_SIZE, "&deplist%d", my_idx);
    }
    return dname;
}

static char *dplasma_dump_c(FILE *out, const dplasma_t *d,
                            char *init_func_body,
                            int init_func_body_size)
{
    static char dp_txt[DPLASMA_SIZE];
    int i;
    int p = 0;

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "    {\n");
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .name   = \"%s\",\n", d->name);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .flags  = 0x%02x,\n", d->flags);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .dependencies_mask = 0x%02x,\n", d->dependencies_mask);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .nb_locals = %d,\n", d->nb_locals);
    
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .locals = {");
    for(i = 0; i < d->nb_locals; i++) {
        if( symbol_c_index_lookup(d->locals[i]) > -1 ) {
            p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "dplasma_symbols[%d]%s", 
                          symbol_c_index_lookup(d->locals[i]),
                          i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
        } else {
            p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s", 
                          dump_c_symbol(out, d->locals[i], init_func_body, init_func_body_size),
                          i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
        }
    }
    for(; i < MAX_LOCAL_COUNT; i++) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "NULL%s",
                      i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .preds = {");
    for(i = 0; i < MAX_PRED_COUNT; i++) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s",
                      dump_c_expression(out, d->preds[i], init_func_body, init_func_body_size),
                      i < MAX_PRED_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .params = {");
    for(i = 0; i < MAX_PARAM_COUNT; i++) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s",
                      dump_c_param(out, d->params[i], init_func_body, init_func_body_size, 1),
                      i < MAX_PARAM_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .deps = %s,\n", dump_c_dependency_list(out, d->deps, init_func_body, init_func_body_size));
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .hook = NULL\n");
    //    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .body = \"%s\"\n", d->body);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "    }");
    
    /* d->body == NULL <=> IN or OUT. Is it the good test? */
    if( NULL != d->body ) {
        fprintf(out, 
                "int %s_hook(const dplasma_execution_context_t *exec_context)\n"
                "{\n",
                d->name);

        for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
            fprintf(out, "  int %s = exec_context->locals[%d].value;\n", d->locals[i]->name, i);
        }
            
        fprintf(out, 
                "  TAKE_TIME(%s_start_key);\n"
                "\n"
                "  %s\n"
                "\n"
                "  TAKE_TIME(%s_end_key);\n"
                "  return 0;\n"
                "}\n"
                "\n", d->name, d->body, d->name);
    }

    return dp_txt;
}

static void dump_tasks_enumerator(FILE *out, const dplasma_t *d, char *init_func_body, int init_func_body_size)
{
    int s;
    int p;
    char spaces[FNAME_SIZE];

    if(d->body == NULL)
        return;

    snprintf(spaces, FNAME_SIZE, "  ");
    fprintf(out, "%s/* %s */\n", spaces, d->name);        
    fprintf(out, "%s{\n", spaces);
    snprintf(spaces + strlen(spaces), FNAME_SIZE-strlen(spaces), "  ");
    for(s = 0; s < d->nb_locals; s++) {
        fprintf(out, "%sint %s, %s_start, %s_end;\n", spaces, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name );
    }
    for(p = 0; d->preds[p]!=NULL; p++) {
        fprintf(out, "%sint pred%d;\n", spaces, p);
    }
    for(s = 0; s < d->nb_locals; s++) {
        fprintf(out, "%s%s_start = ", spaces, d->locals[s]->name);
        dump_inline_c_expression(out, d->locals[s]->min); 
        fprintf(out, ";\n");
        fprintf(out, "%s%s_end = ", spaces, d->locals[s]->name);
        dump_inline_c_expression(out, d->locals[s]->max); 
        fprintf(out, ";\n");
        fprintf(out, "%sfor(%s = %s_start; %s <= %s_end; %s++) {\n",
                spaces, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name,  d->locals[s]->name, d->locals[s]->name);
        snprintf(spaces + strlen(spaces), FNAME_SIZE-strlen(spaces), "  ");
    }
    for(p = 0; d->preds[p] != NULL; p++) {
        fprintf(out, "%spred%d = ", spaces, p);
        dump_inline_c_expression(out, d->preds[p]);
        fprintf(out, ";\n");
    }
    fprintf(out, "%sif(1", spaces);
    for(p = 0; d->preds[p] != NULL; p++) {
        fprintf(out, " && pred%d", p);
    }
    fprintf(out, ") nbtasks++;\n");
    for(s = 0; s < d->nb_locals; s++) {
        spaces[strlen(spaces)-2] = '\0';        
        fprintf(out, "%s}\n", spaces);
    }
    spaces[strlen(spaces)-2] = '\0';        
    fprintf(out, "%s}\n", spaces);
}

void dplasma_dump_all_c(FILE *out)
{
    int i;
    char whole[DPLASMA_ALL_SIZE];
    char body[INIT_FUNC_BODY_SIZE];
    int p = 0;
    preamble_list_t *n;

    for(n = preambles; n != NULL; n = n->next) {
        if( strcasecmp(n->language, "C") == 0 ) {
            fprintf(out, "%s", n->code);
        }
    }

    body[0] = '\0';

    dump_all_global_symbols_c(out, body, INIT_FUNC_BODY_SIZE);

    fprintf(out, 
            "#ifdef DPLASMA_PROFILING\n"
            "#include \"profiling.h\"\n");
    for(i = 0; i < dplasma_nb_elements(); i++) {
        fprintf(out, "int %s_start_key, %s_end_key;\n", dplasma_element_at(i)->name, dplasma_element_at(i)->name);
    }
    fprintf(out,
            "#define TAKE_TIME(KEY)  dplasma_profiling_trace((KEY))\n"
            "#else\n"
            "#define TAKE_TIME(KEY)\n"
            "#endif  /* DPLASMA_PROFILING */\n\n");

    p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "static dplasma_t dplasma_array[%d] = {\n", dplasma_nb_elements());
    for(i = 0; i < dplasma_nb_elements(); i++) {
        p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "%s", dplasma_dump_c(out, dplasma_element_at(i), body, INIT_FUNC_BODY_SIZE));
        if( i < dplasma_nb_elements()-1) {
            p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, ",\n");
        }
    }
    p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "};\n");
    fprintf(out, 
            "%s\n"
            "\n"
            "int dplasma_init(void)\n"
            "{\n"
            "%s\n"
            "  return 0;\n"
            "}\n"
            , whole, body);

    fprintf(out,
            "int load_dplasma_objects( void )\n"
            "{\n"
            "  dplasma_load_array( dplasma_array, %d );\n"
            "  dplasma_load_symbols( dplasma_symbols, %d );\n"
            "  return 0;\n"
            "}\n\n",
            dplasma_nb_elements(),
            dplasma_symbol_get_count());

    fprintf(out, 
            "int load_dplasma_hooks( void )\n"
            "{\n"
            "  dplasma_t* object;\n"
            "  int nbtasks;\n"
            "\n"
            "  if( 0 != dplasma_init()) {\n"
            "     return -1;\n"
            "  }\n"
            "\n");
            
    for(i = 0; i < dplasma_nb_elements(); i++) {
        /* Specials IN and OUT test */
        if( dplasma_element_at(i)->body != NULL ) {
            fprintf(out, "  object = (dplasma_t*)dplasma_find(\"%s\");\n", dplasma_element_at(i)->name);
            fprintf(out, "  object->hook = %s_hook;\n\n", dplasma_element_at(i)->name);
        }
    }

    fprintf(out,
            "#ifdef DPLASMA_PROFILING\n"
            "  dplasma_profiling_init(1024);\n");
    for(i = 0; i < dplasma_nb_elements(); i++) {
        fprintf(out, 
                "  dplasma_profiling_add_dictionary_keyword( \"%s\", \"fill:%s;stroke:rgb(0,0,0)\",\n"
                "                                            &%s_start_key, &%s_end_key);\n",
                dplasma_element_at(i)->name, colors[i % COLORS_SIZE], dplasma_element_at(i)->name, dplasma_element_at(i)->name);
    }

    fprintf(out, 
            "#endif /* DPLASMA_PROFILING */\n"
            "\n"
            "  return 0;\n"
            "}\n");

    fprintf(out,
            "int enumerate_dplasma_tasks(void)\n"
            "{\n"
            "  int nbtasks = 0;\n");

    for(i = 0; i < dplasma_nb_elements(); i++) {
        dump_tasks_enumerator(out,  dplasma_element_at(i), NULL, 0);
    }

    fprintf(out,
            "  dplasma_register_nb_tasks(nbtasks);\n"
            "  return nbtasks;\n"
            "}\n\n");
}
