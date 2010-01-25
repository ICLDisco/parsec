#include "precompile.h"
#include "dplasma.h"
#include "symbol.h"
#include "expr.h"

#include <stdio.h>
#include <string.h>

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
  "#E52B50", 
  "#FFBF00", 
  "#7FFFD4", 
  "#007FFF", 
  "#000000", 
  "#0000FF", 
  "#0095B6", 
  "#8A2BE2", 
  "#A52A2A", 
  "#702963", 
  "#960018", 
  "#DE3163", 
  "#007BA7", 
  "#7FFF00", 
  "#F88379", 
  "#DC143C", 
  "#00FFFF", 
  "#7DF9FF", 
  "#FFD700", 
  "#808080", 
  "#00CC00", 
  "#3FFF00", 
  "#4B0082", 
  "#00A86B", 
  "#B57EDC", 
  "#C8A2C8", 
  "#BFFF00", 
  "#FF00FF", 
  "#800000", 
  "#E0B0FF", 
  "#000080", 
  "#808000", 
  "#FFA500", 
  "#FF4500", 
  "#FFE5B4", 
  "#1C39BB", 
  "#FFC0CB", 
  "#843179", 
  "#FF7518", 
  "#800080", 
  "#FF0000", 
  "#C71585", 
  "#FF007F", 
  "#FA8072", 
  "#FF2400", 
  "#C0C0C0", 
  "#708090", 
  "#00FF7F", 
  "#483C32", 
  "#008080", 
  "#40E0D0", 
  "#EE82EE", 
  "#40826D", 
  "#FFFF00" 
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
static char *dump_c_param(FILE *out, const dplasma_t *dplasma, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it);

static int current_line;
static char *out_name = "";

static int nblines(const char *p)
{
    int r = 0;
    for(; *p != '\0'; p++)
        if( *p == '\n' )
            r++;
    return r;
}

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
        } else if( EXPR_OP_SYMB == e->op ) {
            fflush(stdout);
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
                fprintf(out, "\n#error cannot evaluate range expression here!\n");
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
            case EXPR_OP_BINARY_LESS:
                fprintf(out, ")<(");
                break;
            case EXPR_OP_BINARY_MORE:
                fprintf(out, ")>(");
                break;
            default:
                fprintf(out, "\n#error unknown binary operand %d\n", e->op);
                fprintf(stderr, "Unkown binary operand %d\n", e->op);
            }
            dump_inline_c_expression(out, e->bop2);
            fprintf(out, ")");
        } else {
            fprintf(stderr, "Unkown operand %d in expression\n", e->op);
            fprintf(out, "\n#error Unknown operand %d in expression\n", e->op);
        }
    }
}

static char *dump_c_expression_inline(FILE *out, const expr_t *e,
                                      const symbol_t **symbols, int nbsymbols,
                                      char *init_func_body, int init_func_body_size)
{
    static unsigned int expr_idx = 0;
    static char name[FNAME_SIZE];

    if( e == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        int my_id = expr_idx;
        int i;
        expr_idx++;

        fprintf(out, 
                "static int inline_expr%d( const  assignment_t *assignments )\n"
                "{\n",
                my_id);
        current_line += 2;

        for(i = 0; i < nbsymbols; i++) {
            if( (NULL != symbols[i]) && ( EXPR_SUCCESS == expr_depend_on_symbol( e, symbols[i] ) ) ) {
                fprintf(out, "  int %s = assignments[%d].value;\n", symbols[i]->name, i);
                current_line++;
            } 
        }
        /*
        for(i = 0; i < nbsymbols; i++) {
            if( (NULL != symbols[i]) && ( EXPR_SUCCESS == expr_depend_on_symbol( e, symbols[i] ) ) ) {
                   fprintf(out, "  assert( (assignments[%d].sym != NULL) && (strcmp(assignments[%d].sym->name, \"%s\") == 0) );\n", i, i, symbols[i]->name);
                   current_line++;
            } 
        }
        */

        fprintf(out, "  return ");
        dump_inline_c_expression(out, e);
        fprintf(out, 
                ";\n"
                "}\n"
                "static expr_t inline%d = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr%d }; /* ",
                my_id, my_id);
        current_line += 2;
        expr_dump(out, e);
        fprintf(out, " */\n");
        current_line++;

        snprintf(name, FNAME_SIZE, "&inline%d", my_id);
    }

    return name;
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
            current_line++;
        } 
        else if( EXPR_OP_SYMB == e->op ) {
            char sname[FNAME_SIZE];
            snprintf(sname, FNAME_SIZE, "%s", dump_c_symbol(out, e->var, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                fprintf(out, "static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s, .value = %d }; /* ",
                        my_id, e->flags, sname, e->value);
                expr_dump(out, e);
                fprintf(out, " */\n");
                current_line++;                
            } else {
                fprintf(out, "static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s }; /* ",
                        my_id, e->flags, sname);
                expr_dump(out, e);
                fprintf(out, " */\n");
                current_line++;
            }
        } else if( EXPR_IS_UNARY(e->op) ) {
            char sn[FNAME_SIZE];
            snprintf(sn, FNAME_SIZE, "%s", dump_c_expression(out, e->uop1, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s, .value = %d }; /* ", 
                        my_id, e->op, e->flags, sn, e->value);
                expr_dump(out, e);
                fprintf(out, " */\n");
                current_line++;
            } else {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s }; /* ", 
                        my_id, e->op, e->flags, sn);
                expr_dump(out, e);
                fprintf(out, " */\n");
                current_line++;
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
                current_line++;
            } else {
                fprintf(out, "static expr_t expr%d = { .op = %d, .flags = %d, .bop1 = %s, .bop2 = %s }; /* ", 
                        my_id, e->op, e->flags, sn1, sn2);
                expr_dump(out, e);
                fprintf(out, " */\n");
                current_line++;
            }
        } else {
            fprintf(stderr, "Unkown operand %d in expression", e->op);
        }

        snprintf(name, FNAME_SIZE, "&expr%d", my_id);
    }

    return name;
}

static char *dump_c_dep(FILE *out, const dplasma_t *dplasma, const dep_t *d, char *init_func_body, int init_func_body_size)
{
    static unsigned int dep_idx = 0;
    static char name[FNAME_SIZE];
    static dumped_dep_list_t *dumped_deps;
    dumped_dep_list_t *dumped;
    unsigned int my_idx;
    size_t body_length;
    int i;
    
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
        asprintf(&dumped->name, "&dep%u", my_idx);
        dumped_deps = dumped;
        
        p += snprintf(whole + p, DEP_CODE_SIZE-p, 
                      "static dep_t dep%u = { .cond = %s, .dplasma = NULL,\n"
                      "                       .call_params = {",
                      my_idx, dump_c_expression_inline(out, d->cond, dplasma->locals, dplasma->nb_locals, init_func_body, init_func_body_size));
        body_length = strlen(init_func_body);
        i = snprintf(init_func_body + body_length, init_func_body_size - body_length,
                     "  dep%d.dplasma = &dplasma_array[%d];\n", my_idx, dplasma_dplasma_index( d->dplasma ));
        if(i + body_length >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        body_length = strlen(init_func_body);
        i = snprintf(init_func_body + body_length, init_func_body_size - body_length,
                     "  dep%d.param = %s;\n", my_idx, dump_c_param(out, d, d->param, init_func_body, init_func_body_size, 0));
        if(i + body_length >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        for(i = 0 ; i < MAX_CALL_PARAM_COUNT; i++) {
            /* params can have ranges here: don't use inline c expression */
            p += snprintf(whole + p, DEP_CODE_SIZE-p, "%s%s", dump_c_expression(out, d->call_params[i], init_func_body, init_func_body_size), 
                          i < MAX_CALL_PARAM_COUNT-1 ? ", " : "}};\n");
        }
        fprintf(out, "%s", whole);
        current_line += nblines(whole);
        snprintf(name, FNAME_SIZE, "&dep%u", my_idx);
    }
     
   return name;
}

static char *dump_c_param(FILE *out, const dplasma_t *dplasma, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it)
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
            asprintf(&dumped->param_name, "&param%u", my_idx);
            dumped->next = dumped_params;
            dumped_params = dumped;
            if( !dump_it ) {
                return dumped->param_name;
            }
        }

        l += snprintf(param + l, PARAM_CODE_SIZE-l, 
                      "static param_t param%u = { .name = \"%s\", .sym_type = %d, .param_mask = 0x%02x,\n"
                      "     .dep_in  = {", my_idx, p->name, p->sym_type, p->param_mask);
        for(i = 0; i < MAX_DEP_IN_COUNT; i++) {
            dep_name = dump_c_dep(out, dplasma, p->dep_in[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_IN_COUNT-1 ? ", " : "},\n"
                          "     .dep_out = {");
        }
        for(i = 0; i < MAX_DEP_OUT_COUNT; i++) {
            dep_name = dump_c_dep(out, dplasma, p->dep_out[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_OUT_COUNT-1 ? ", " : "} };\n");
        }
        fprintf(out, "%s", param);
        current_line += nblines(param);
        snprintf(name, FNAME_SIZE, "&param%u", my_idx);
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
    current_line++;

    return e->c_name;
}

static void dump_all_global_symbols_c(FILE *out, char *init_func_body, int init_func_body_size)
{
    int i, l = 0;
    char whole[SYMBOL_CODE_SIZE];
    const symbol_t* symbol;

    l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "static symbol_t *dplasma_symbols[] = {\n");
    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "   %s%s", 
                      dump_c_symbol(out, dplasma_symbol_get_element_at(i), init_func_body, init_func_body_size),
                      (i < (dplasma_symbol_get_count()-1)) ? ",\n" : "};\n");
    }
    fprintf(out, "%s", whole);
    current_line += nblines(whole);

    fprintf(out, "\n");
    current_line++;

    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        symbol = dplasma_symbol_get_element_at(i);
        if( (symbol->min != NULL) &&
            (symbol->max != NULL) &&
            ((symbol->min->flags & symbol->max->flags) & EXPR_FLAG_CONSTANT) &&
            (symbol->min->value == symbol->max->value) ) {
            fprintf(out, "int %s = %d;\n", symbol->name, symbol->min->value);
            current_line++;
        } else {
            fprintf(out, "int %s;\n", symbol->name);
            current_line++;

            snprintf(init_func_body + strlen(init_func_body),
                     init_func_body_size - strlen(init_func_body),
                     "  {\n"
                     "    int rc;\n"
                     "    if( 0 != (rc = expr_eval( (%s)->min, NULL, 0, &%s)) ) {\n"
                     "      return rc;\n"
                     "    }\n"
                     "  }\n",
                     dump_c_symbol(out, symbol, init_func_body, init_func_body_size), symbol->name);
        }
    }
    fprintf(out, "\n");
    current_line++;
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
        current_line += nblines(b);
        snprintf(dname, FNAME_SIZE, "&deplist%d", my_idx);
    }
    return dname;
}

#include "remote_dep.h"

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
                      dump_c_expression_inline(out, d->preds[i], d->locals, d->nb_locals, init_func_body, init_func_body_size),
                      i < MAX_PRED_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .inout= {");
    for(i = 0; i < MAX_PARAM_COUNT; i++) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s",
                      dump_c_param(out, d, d->inout[i], init_func_body, init_func_body_size, 1),
                      i < MAX_PARAM_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .deps = %s,\n", dump_c_dependency_list(out, d->deps, init_func_body, init_func_body_size));
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .hook = NULL\n");
    //    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .body = \"%s\"\n", d->body);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "    }");
    
    /* d->body == NULL <=> IN or OUT. Is it the good test? */
    if( NULL != d->body ) {
        int body_lines;

        fprintf(out, 
                "static int %s_hook(dplasma_execution_unit_t* context, const dplasma_execution_context_t *exec_context)\n"
                "{\n"
				"  (void)context;\n",
                d->name);
        current_line += 3;

        for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
            fprintf(out, "  int %s = exec_context->locals[%d].value;\n", d->locals[i]->name, i);
            current_line++;
        }
        fprintf(out, "  /* remove warnings in case the variable is not used later */\n");
        current_line++;
        for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
            fprintf(out, "  (void)%s;\n", d->locals[i]->name);
            current_line++;
        }
            
        body_lines = nblines(d->body);

        fprintf(out, 
                "  TAKE_TIME(context, %s_start_key);\n"
                "\n"
                "  %s\n"
                "#line %d \"%s\"\n"
                "\n"
                "  TAKE_TIME(context, %s_end_key);\n"
                "\n", d->name, d->body, body_lines+3+current_line, out_name, d->name);
        current_line += 6 + body_lines;

        for(i = 0; i < MAX_PARAM_COUNT; i++) {
            if( (NULL != d->inout[i]) && (d->inout[i]->sym_type & SYM_OUT) ) {
                char spaces[MAX_CALL_PARAM_COUNT * 2 + 3];
                int j;

                struct param *p = d->inout[i];

                fprintf(out, 
                        "  {\n"
                        "    struct dplasma_dependencies_t *placeholder = NULL;\n"
                        "    dplasma_execution_context_t new_context = { .function = NULL, .locals = {");
                current_line+=2;
                for(j = 0; j < MAX_LOCAL_COUNT; j++) {
                    fprintf(out, " {.sym = NULL}%s", j+1 == MAX_LOCAL_COUNT ? "}};\n" : ", ");
                }
                current_line++;

                sprintf(spaces, "  ");

                for(j = 0; j < MAX_DEP_OUT_COUNT; j++) {
                    if( (NULL != p->dep_out[j]) &&
                        (p->dep_out[j]->dplasma->nb_locals > 0) ) {
                        int k;
                        struct dep *dep = p->dep_out[j];
                        
                        fprintf(out, "    { /** iterate now on the params and dependencies to release OUT dependencies */\n");
                        current_line++;

                        for(k = 0; k < MAX_CALL_PARAM_COUNT; k++) {
                            if( NULL != dep->call_params[k] ) {
                                fprintf(out, "      int _p%d;\n", k);
                                current_line++;
                            }
                        }

                        fprintf(out, 
                                "      new_context.function = exec_context->function->inout[%d]->dep_out[%d]->dplasma; /* placeholder for %s */\n" 
                                "      assert( strcmp( new_context.function->name, \"%s\") == 0 );\n",
                                i, j, dep->dplasma->name, dep->dplasma->name);
                        current_line+=2;

                        for(k = 0; k < MAX_CALL_PARAM_COUNT; k++) {
                            if( NULL != dep->call_params[k] ) {

                                if( EXPR_OP_BINARY_RANGE == dep->call_params[k]->op ) {
                                    fprintf(out, "%s    for(_p%d = ", spaces, k);
                                    dump_inline_c_expression(out, dep->call_params[k]->bop1);
                                    fprintf(out, "; _p%d <= ", k);
                                    dump_inline_c_expression(out, dep->call_params[k]->bop2);
                                    fprintf(out, "; _p%d++) {\n", k);
                                    current_line++;
                                    snprintf(spaces + strlen(spaces), MAX_CALL_PARAM_COUNT * 2 + 3 - strlen(spaces), "  ");
                                } else {
                                    fprintf(out, "%s    _p%d = ", spaces, k);
                                    dump_inline_c_expression(out, dep->call_params[k]);
                                    fprintf(out, ";\n");
                                    current_line += 1;
                                }
                                fprintf(out, 
                                        "%s    new_context.locals[%d].value = _p%d;\n"
                                        "%s    new_context.locals[%d].min   = _p%d;\n"
                                        "%s    new_context.locals[%d].max   = _p%d;\n", 
                                        spaces, k, k,
                                        spaces, k, k,
                                        spaces, k, k);
                                current_line++;
                                fprintf(out, "%s    new_context.locals[%d].sym = new_context.function->locals[%d];\n", spaces, k, k);
                                current_line++;
                            }
                        }

#if defined(_DEBUG)
                        {
                            int l;
                            fprintf(out, 
                                    "%s    fprintf(stderr, \"%s", spaces, d->name);
                            for(l = 0; l < d->nb_locals; l++) {
                                fprintf(out, "_%s=%%d", d->locals[l]->name);
                            }
                            fprintf(out, "\\n\"");
                            for(l = 0; l < d->nb_locals; l++) {
                                fprintf(out, ", %s", d->locals[l]->name);
                            }
                            fprintf(out, ");\n");
                            current_line++;
                        }
#endif

                        if( NULL != dep->cond ) {
                            fprintf(out, "%s    if(", spaces);
                            dump_inline_c_expression(out, dep->cond);
                            fprintf(out, ") {\n");
                            current_line++;
                        } else {
                            fprintf(out, "%s    {\n", spaces);
                            current_line++;
                        }
                        for(k = 0; k < dep->dplasma->nb_locals; k++) {
                            fprintf(out, "%s      int %s = _p%d;\n", spaces, dep->dplasma->locals[k]->name, k);
                            current_line++;
                        }
                        for(k = 0; k < dep->dplasma->nb_locals; k++) {
                            fprintf(out, "%s      (void)%s;\n", spaces, dep->dplasma->locals[k]->name);
                            current_line++;
                        }

                        /******************************************************/
                        /* Compute predicates                                 */
                        fprintf(out, "%s      if( (1", spaces);
                        for(k = 0; k < MAX_PRED_COUNT; k++) {
                            if( NULL != dep->dplasma->preds[k] ) {
                                fprintf(out, ") && (");
                                dump_inline_c_expression(out, dep->dplasma->preds[k]);
                            }
                        }
                        fprintf(out, ") ) {\n");
                        current_line++;

                        fprintf(out,
                                "%s        dplasma_release_local_OUT_dependencies(context, exec_context, \n"
                                "%s                       exec_context->function->inout[%d/*i*/],\n"
                                "%s                       &new_context,\n"
                                "%s                       exec_context->function->inout[%d/*i*/]->dep_out[%d/*j*/]->param,\n"
                                "%s                       &placeholder);\n", 
                                spaces, spaces, i, spaces, spaces, i, j, spaces);
                        current_line += 5;

                        /* If predicates don't verify, this is remote, compute 
                         * target rank from predicate values
                         */
                        {
                            expr_t *rowpred; 
                            expr_t *colpred;
                            expr_t *rowsize;
                            expr_t *colsize;
                            
                            if(dplasma_remote_dep_get_rank_preds(dep->dplasma->preds, 
                                                                 &rowpred, 
                                                                 &colpred, 
                                                                 &rowsize,
                                                                 &colsize) < 0)
                            {
                               fprintf(out,
                                       "%s      } else {\n"
                                       "%s        DEBUG((\"GRID is not defined in JDF, but predicates are not verified. Your jdf is incomplete or your predicates false.\\n\"));\n"
                                       "%s      }\n", 
                                       spaces, spaces, spaces);
                                current_line += 3;
                            }
                            else 
                            {
                                fprintf(out, 
                                        "%s      } else {\n"
                                        "%s        int rank, rrank, crank, ncols;\n"
                                        "%s        rrank = ",
                                        spaces, spaces, spaces);
                                dump_inline_c_expression(out, rowpred);
                                fprintf(out, 
                                        "\n"
                                        "%s        crank = ", 
                                        spaces);
                                dump_inline_c_expression(out, colpred);
                                fprintf(out, 
                                        "\n"
                                        "%s        ncols = ",
                                        spaces);
                                dump_inline_c_expression(out, colsize);
                                fprintf(out, 
                                        "\n"
                                        "%s        rank = crank + rrank * ncols;\n"
                                        "%s        DEBUG((\"gridrank = %%d ( %%d + %%d x %%d )\\n\", rank, crank, rrank, ncols));\n"
                                        "%s        dplasma_remote_dep_activate_rank(context,\n"
                                        "%s                                         exec_context,\n"
                                        "%s                                         exec_context->function->inout[%d/*i*/],\n"
                                        "%s                                         new_context,\n"
                                        "%s                                         exec_context->function->inout[%d/*i*/]->dep_out[%d/*j*/]->param,\n"
                                        "%s                                         rank);\n"
                                        "%s      }\n",
                                        spaces, spaces, spaces, spaces, spaces, i, spaces, spaces, i, j, spaces, spaces);
                                current_line += 14;
                            }
                        }
                        fprintf(out, "%s    }\n", spaces);
                        current_line++;
                        
                        for(k = MAX_PARAM_COUNT-1; k >= 0; k--) {
                            if( NULL != dep->call_params[k] ) {
                                if( EXPR_OP_BINARY_RANGE == dep->call_params[k]->op ) {
                                    spaces[strlen(spaces)-2] = '\0';
                                    fprintf(out, "%s    }\n", spaces);
                                    current_line++;
                                    if( k == MAX_PARAM_COUNT-1 ) {
                                        fprintf(out, "%s  placeholder=NULL;\n", spaces);
                                        current_line++;
                                    }
                                }
                            }
                        }
                        fprintf(out, "    }\n");
                        current_line++;
                    }
                }

                fprintf(out, "  }\n");
                current_line++;
            }
        }

        fprintf(out, 
                "  return 0;\n"
                "}\n"
                "\n");
        current_line += 3;
    }

    return dp_txt;
}

static void dump_tasks_enumerator(FILE *out, const dplasma_t *d, char *init_func_body, int init_func_body_size)
{
    char spaces[FNAME_SIZE];
    size_t spaces_length;
    int s, p;

    if(d->body == NULL)
        return;

    snprintf(spaces, FNAME_SIZE, "  ");
    fprintf(out, "%s/* %s */\n", spaces, d->name);
    current_line ++;

    fprintf(out, "%s{\n", spaces);
    current_line++;

    spaces_length = strlen(spaces);
    snprintf(spaces + spaces_length, FNAME_SIZE-spaces_length, "  ");
    for(s = 0; s < d->nb_locals; s++) {
        fprintf(out, "%sint %s, %s_start, %s_end;\n", spaces, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name );
        current_line++;
    }
    for(p = 0; d->preds[p]!=NULL; p++) {
        fprintf(out, "%sint pred%d;\n", spaces, p);
        current_line++;
    }
    for(s = 0; s < d->nb_locals; s++) {
        fprintf(out, "%s%s_start = ", spaces, d->locals[s]->name);
        dump_inline_c_expression(out, d->locals[s]->min); 
        fprintf(out, ";\n");
        current_line++;
        fprintf(out, "%s%s_end = ", spaces, d->locals[s]->name);
        dump_inline_c_expression(out, d->locals[s]->max); 
        fprintf(out, ";\n");
        current_line++;
        fprintf(out, "%sfor(%s = %s_start; %s <= %s_end; %s++) {\n",
                spaces, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name,  d->locals[s]->name, d->locals[s]->name);
        current_line++;
        snprintf(spaces + strlen(spaces), FNAME_SIZE-strlen(spaces), "  ");
    }
    for(p = 0; d->preds[p] != NULL; p++) {
        fprintf(out, "%spred%d = ", spaces, p);
        dump_inline_c_expression(out, d->preds[p]);
        fprintf(out, ";\n");
        current_line++;
    }
    fprintf(out, "%sif(1", spaces);
    for(p = 0; d->preds[p] != NULL; p++) {
        fprintf(out, " && pred%d", p);
    }
    fprintf(out, ") nbtasks++;\n");
    current_line++;

    for(s = 0; s < d->nb_locals; s++) {
        spaces[strlen(spaces)-2] = '\0';
        fprintf(out, "%s}\n", spaces);
        current_line++;
    }

    spaces[strlen(spaces)-2] = '\0';
    fprintf(out, "%s}\n", spaces);
    current_line++;
}

int dplasma_dump_all_c(char *filename)
{
    char whole[DPLASMA_ALL_SIZE];
    char body[INIT_FUNC_BODY_SIZE];
    preamble_list_t *n;
    const dplasma_t* object;
    int i, p = 0;
    FILE *out;
    
    out = fopen(filename, "w");
    if( out == NULL ) {
        return -1;
    }
    out_name = filename;

    current_line = 1;
    
    for(n = preambles; n != NULL; n = n->next) {
        if( strcasecmp(n->language, "C") == 0 ) {
            int nb = nblines(n->code);
            fprintf(out, 
                    "%s\n"
                    "#line %d \"%s\"\n", 
                    n->code, nb+current_line+1, out_name);
            current_line += nb + 2;
        }
    }

    body[0] = '\0';
    
    dump_all_global_symbols_c(out, body, INIT_FUNC_BODY_SIZE);

    fprintf(out, 
            "#include <assert.h>\n"
            "#include <string.h>\n"
            "#ifdef DPLASMA_PROFILING\n"
            "#include \"profiling.h\"\n");
    current_line += 3;

    for(i = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        fprintf(out, "int %s_start_key, %s_end_key;\n", object->name, object->name);
        current_line++;
    }
    fprintf(out,
            "#define TAKE_TIME(EU_CONTEXT, KEY)  dplasma_profiling_trace((EU_CONTEXT), (KEY))\n"
            "#else\n"
            "#define TAKE_TIME(EU_CONTEXT, KEY)\n"
            "#endif  /* DPLASMA_PROFILING */\n"
            "\n"
            "#include \"scheduling.h\"\n"
            "\n");
    current_line += 7;

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
            "static int __dplasma_init(void)\n"
            "{\n"
            "%s\n"
            "  return 0;\n"
            "}\n"
            , whole, body);
    current_line += 7 + nblines(whole) + nblines(body);

    fprintf(out,
            "int load_dplasma_objects( dplasma_context_t* context )\n"
            "{\n"
			"  (void)context;\n"
            "  dplasma_load_array( dplasma_array, %d );\n"
            "  dplasma_load_symbols( dplasma_symbols, %d );\n"
            "  return 0;\n"
            "}\n"
            "\n",
            dplasma_nb_elements(),
            dplasma_symbol_get_count());
    current_line += 8;

    fprintf(out, 
            "int load_dplasma_hooks( dplasma_context_t* context )\n"
            "{\n"
            "  dplasma_t* object;\n"
            "\n"
			"  (void)context;\n"
            "  if( 0 != __dplasma_init()) {\n"
            "     return -1;\n"
            "  }\n"
            "\n");
    current_line += 9;

    for(i = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        /* Specials IN and OUT test */
        if( object->body != NULL ) {
            fprintf(out, "  object = (dplasma_t*)dplasma_find(\"%s\");\n"
                         "  object->hook = %s_hook;\n\n",
                    object->name, object->name);
            current_line += 2;
        }
    }

    fprintf(out, "#ifdef DPLASMA_PROFILING\n");
    current_line += 1;

    for(i = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        fprintf(out, 
                "  dplasma_profiling_add_dictionary_keyword( \"%s\", \"fill:%s\",\n"
                "                                            &%s_start_key, &%s_end_key);\n",
                object->name, colors[i % COLORS_SIZE], object->name, object->name);
        current_line += 2;
    }

    fprintf(out, 
            "#endif /* DPLASMA_PROFILING */\n"
            "\n"
            "  return 0;\n"
            "}\n");
    current_line += 4;

    fprintf(out,
            "int enumerate_dplasma_tasks(dplasma_context_t* context)\n"
            "{\n"
            "  int nbtasks = 0;\n");
    current_line += 3;

    for(i = 0; i < dplasma_nb_elements(); i++) {
        dump_tasks_enumerator(out,  dplasma_element_at(i), NULL, 0);
    }

    fprintf(out,
            "  dplasma_register_nb_tasks(context, nbtasks);\n"
            "  return nbtasks;\n"
            "}\n"
            "\n");
    current_line += 4;
    
    fclose(out);

    return 0;
}
