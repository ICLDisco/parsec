/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "precompile.h"
#include "symbol.h"
#include "expr.h"

#define INIT_FUNC_BODY_SIZE 4096
#define FNAME_SIZE            64
#define DEPENDENCY_SIZE     1024
#define DPLASMA_SIZE        4096
#define SYMBOL_CODE_SIZE    4096
#define PARAM_CODE_SIZE     4096
#define DEP_CODE_SIZE       4096
#define DPLASMA_ALL_SIZE    8192
#define DATA_REPO_LOOKUP_SIZE 4096
#define MAX_EXPR_LEN         512

#define COLORS_SIZE           54

static const char *colors[COLORS_SIZE] = { 
    "#E52B50", 
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

#define SHAPES_SIZE 14
static char *shapes[SHAPES_SIZE] = {
    "polygon",
    "ellipse",
    "egg",
    "diamond",
    "trapezium",
    "parallelogram",
    "hexagon",
    "octagon",
    "doublecircle",
    "tripleoctagon",
    "invtrapezium",
    "box",
    "triangle",
    "invtriangle"
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

static char *dump_c_symbol(const symbol_t *s, char *init_func_body, int init_func_body_size);
static char *dump_c_param(const dplasma_t *dplasma, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it);

static FILE *out;
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

#if defined(__GNUC__)
static void output(char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void output(char *format, ...)
{
    va_list ap;
    char *res;
    int len;

    va_start(ap, format);
    len = vasprintf(&res, format, ap);
    va_end(ap);

    if( len == -1 ) {
        fprintf(stderr, "Unable to ouptut a string: %s\n", strerror(errno));
    } else {
        fwrite(res, len, 1, out);
        current_line += nblines(res);
        free(res);
    }
}

void dplasma_precompiler_add_preamble(const char *language, const char *code)
{
    preamble_list_t *n = (preamble_list_t*)calloc(1, sizeof(preamble_list_t));
    n->language = language;
    n->code = code;
    n->next = preambles;
    preambles = n;
}

static char *expression_to_c_inline(const expr_t *e, char* prepend, char *res, int reslen)
{
    char lo[MAX_EXPR_LEN];
    char ro[MAX_EXPR_LEN];
    char mo[MAX_EXPR_LEN];
    char *pres;

#define WIR(p) do {                                                     \
        if( snprintf p > reslen ) {                                     \
            fprintf(stderr, "Stack overflow error: expression is too "  \
                    "long for limitation of %d bytes\n", reslen);       \
            return NULL;                                                \
        }                                                               \
    } while(0)
    
    if( e == NULL ) {
        return NULL;
    } else {
        if( EXPR_OP_CONST_INT == e->op ) {
            WIR((res, reslen, "%d", e->value));
        } else if( EXPR_OP_SYMB == e->op ) {
            if( e->var->flags & DPLASMA_SYMBOL_IS_GLOBAL ) {
                WIR((res, reslen, "%s", e->var->name));
            } else {
                WIR((res, reslen, "%s%s", prepend, e->var->name));
            }
        } else if( EXPR_IS_UNARY(e->op) ) {
            pres = expression_to_c_inline(e->uop1, prepend, lo, MAX_EXPR_LEN);
            if( NULL == pres )
                return NULL;

            if( e->op == EXPR_OP_UNARY_NOT ) {
                WIR((res, reslen, "!(%s)", lo));
            } else {
                fprintf(stderr, "Internal error: only defined unary operand is not\n");
                return NULL;
            }
        } else if( EXPR_IS_BINARY(e->op) ) {
            pres = expression_to_c_inline(e->bop1, prepend, lo, MAX_EXPR_LEN);
            if( NULL == pres )
                return NULL;

            pres = expression_to_c_inline(e->bop2, prepend, ro, MAX_EXPR_LEN);
            if( NULL == pres )
                return NULL;

            switch( e->op ) {
            case EXPR_OP_BINARY_MOD:            
                WIR((res, reslen, "(%s)%%(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_EQUAL:
                WIR((res, reslen, "(%s)==(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_NOT_EQUAL:
                WIR((res, reslen, "(%s)!=(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_PLUS:
                WIR((res, reslen, "(%s)+(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_RANGE:
                fprintf(stderr, "cannot evaluate range expression here!\n");
                return NULL;
                break;
            case EXPR_OP_BINARY_MINUS:
                WIR((res, reslen, "(%s)-(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_TIMES:
                WIR((res, reslen, "(%s)*(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_DIV:
                WIR((res, reslen, "(%s)/(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_OR:
                WIR((res, reslen, "(%s)||(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_AND:
                WIR((res, reslen, "(%s)&&(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_XOR:
                WIR((res, reslen, "(%s)(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_LESS:
                WIR((res, reslen, "(%s)<(%s)", lo, ro));
                break;
            case EXPR_OP_BINARY_MORE:
                WIR((res, reslen, "(%s)>(%s)", lo, ro));
                break;
            default:
                fprintf(stderr, "Unkown binary operand %d\n", e->op);
                return NULL;
            }
        } else if( EXPR_IS_TERTIAR(e->op) ) {
            pres =  expression_to_c_inline(e->tcond, prepend, lo, MAX_EXPR_LEN);
            if( NULL == pres )
                return NULL;

            pres = expression_to_c_inline(e->top1, prepend, mo, MAX_EXPR_LEN);
            if( NULL == pres )
                return NULL;

            pres = expression_to_c_inline(e->top2, prepend, ro, MAX_EXPR_LEN) ;
            if( NULL == pres )
                return NULL;

            WIR((res, reslen, "((%s)?(%s):(%s))", lo, mo, ro));
        } else {
            fprintf(stderr, "[%s:%d] Unkown operand %d in expression\n", __FILE__, __LINE__, e->op);
            return NULL;
        }
    }
    return res;
}

static char *dplasma_dep_dplasma_call_to_c(const dep_t *d, char *res, int reslen)
{
    int i;
    int p, r, dump = 0;
    char strexpr[MAX_EXPR_LEN];
    p = snprintf(res, reslen, "%s(", d->dplasma->name);
    if( p > reslen )
        goto error;

    for(i = 0; i < MAX_CALL_PARAM_COUNT; i++) {
        if( NULL != d->call_params[i] ) {
            r = snprintf(res+p, reslen-p, "%s%s", dump ? ", " : "", 
                         expression_to_c_inline(d->call_params[i], "", strexpr, MAX_EXPR_LEN));
            if( r > reslen-p )
                goto error;
            p += r;
            dump = 1;
        }
    }

    r = snprintf(res+p, reslen-p, ")");
    if( r > reslen-p ) 
        goto error;
    return res;

 error:
    fprintf(stderr, "Unable to create dplasma function call: a buffer of %d is not large enough\n",
            reslen);
    return NULL;
}

static char *dump_c_expression_inline(const expr_t *e,
                                      const symbol_t **symbols, int nbsymbols,
                                      char *init_func_body, int init_func_body_size)
{
    static unsigned int expr_idx = 0;
    static char name[FNAME_SIZE];
    char expr[MAX_EXPR_LEN];

    if( e == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        int my_id = expr_idx;
        int i;
        expr_idx++;

        output(
               "static inline int inline_expr%d( const  assignment_t *assignments )\n"
               "{\n",
               my_id);

        for(i = 0; i < nbsymbols; i++) {
            if( (NULL != symbols[i]) && ( EXPR_SUCCESS == expr_depend_on_symbol( e, symbols[i] ) ) ) {
                output("  int %s = assignments[%d].value;\n", symbols[i]->name, i);
            } 
        }
        output("  return %s;\n"
               "}\n"
               "static expr_t inline%d = { .op= EXPR_OP_INLINE, .flags = 0, .inline_func = inline_expr%d };\n",
               expression_to_c_inline(e, "", expr, MAX_EXPR_LEN), my_id, my_id);

        snprintf(name, FNAME_SIZE, "&inline%d", my_id);
    }

    return name;
}

static char *dump_c_expression(const expr_t *e, char *init_func_body, int init_func_body_size)
{
    static unsigned int expr_idx = 0;
    static char name[FNAME_SIZE];

    if( e == NULL ) {
        snprintf(name, FNAME_SIZE, "NULL");
    } else {
        int my_id = expr_idx;
        expr_idx++;

        if( EXPR_OP_CONST_INT == e->op ) {
            output("static expr_t expr%d = { .op = EXPR_OP_CONST_INT, .flags = %d, .value = %d }; /* ",
                   my_id, e->flags, e->value);
            expr_dump(out, e);
            output(" */\n");
        } 
        else if( EXPR_OP_SYMB == e->op ) {
            char sname[FNAME_SIZE];
            snprintf(sname, FNAME_SIZE, "%s", dump_c_symbol(e->var, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                output("static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s, .value = %d }; /* ",
                       my_id, e->flags, sname, e->value);
                expr_dump(out, e);
                output(" */\n");
            } else {
                output("static expr_t expr%d = { .op = EXPR_OP_SYMB, .flags = %d, .var = %s }; /* ",
                       my_id, e->flags, sname);
                expr_dump(out, e);
                output(" */\n");
            }
        } else if( EXPR_IS_UNARY(e->op) ) {
            char sn[FNAME_SIZE];
            snprintf(sn, FNAME_SIZE, "%s", dump_c_expression(e->uop1, init_func_body, init_func_body_size));
            if( e->flags & EXPR_FLAG_CONSTANT ) {
                output("static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s, .value = %d }; /* ", 
                       my_id, e->op, e->flags, sn, e->value);
                expr_dump(out, e);
                output(" */\n");
            } else {
                output("static expr_t expr%d = { .op = %d, .flags = %d, .uop1 = %s }; /* ", 
                       my_id, e->op, e->flags, sn);
                expr_dump(out, e);
                output(" */\n");
            }
        } else if( EXPR_IS_BINARY(e->op) ) {
            char sop1[FNAME_SIZE];
            char sop2[FNAME_SIZE];
            snprintf(sop1, FNAME_SIZE, "%s", dump_c_expression(e->bop1, init_func_body, init_func_body_size));
            snprintf(sop2, FNAME_SIZE, "%s", dump_c_expression(e->bop2, init_func_body, init_func_body_size));
            output("static expr_t expr%d = { .op = %d, .flags = %d, .bop1 = %s, .bop2 = %s, .value = %d }; /* ", 
                   my_id, e->op, e->flags, sop1, sop2,
                   ((e->flags & EXPR_FLAG_CONSTANT) ? e->value : 0));
            expr_dump(out, e);
            output(" */\n");
        } else if( EXPR_IS_TERTIAR(e->op) ) {
            char scond[FNAME_SIZE];
            char sop1[FNAME_SIZE];
            char sop2[FNAME_SIZE];
            snprintf(scond, FNAME_SIZE, "%s", dump_c_expression(e->tcond, init_func_body, init_func_body_size));
            snprintf(sop1, FNAME_SIZE, "%s", dump_c_expression(e->top1, init_func_body, init_func_body_size));
            snprintf(sop2, FNAME_SIZE, "%s", dump_c_expression(e->top2, init_func_body, init_func_body_size));

            output("static expr_t expr%d = { .op = %d, .flags = %d, .tcond = %s, .top1 = %s, .top2 = %s, .value = %d }; /* ", 
                   my_id, e->op, e->flags, scond, sop1, sop2,
                   ( (e->flags & EXPR_FLAG_CONSTANT) ? e->value : 0));
            expr_dump(out, e);
            output(" */\n");
        } else {
            fprintf(stderr, "[%s:%d] Unkown operand %d in expression\n", __FILE__, __LINE__, e->op);
        }

        snprintf(name, FNAME_SIZE, "&expr%d", my_id);
    }

    return name;
}

static char *dump_c_dep(const dplasma_t *dplasma, const dep_t *d, char *init_func_body, int init_func_body_size)
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
                      "static dep_t dep%u = { .cond = %s, .mpi_type = %c%s%c, .dplasma = NULL,\n"
                      "                       .call_params = {",
                      my_idx, dump_c_expression_inline(d->cond, (const symbol_t**)dplasma->locals, dplasma->nb_locals, init_func_body, init_func_body_size),
                      NULL == d->mpi_type ? ' ' : '"',
                      NULL == d->mpi_type ? "NULL" : d->mpi_type,
                      NULL == d->mpi_type ? ' ' : '"');
        body_length = strlen(init_func_body);
        i = snprintf(init_func_body + body_length, init_func_body_size - body_length,
                     "  dep%d.dplasma = &dplasma_array[%d];\n", my_idx, dplasma_dplasma_index( d->dplasma ));
        if(i + body_length >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        body_length = strlen(init_func_body);
        i = snprintf(init_func_body + body_length, init_func_body_size - body_length,
                     "  dep%d.param = %s;\n", my_idx, dump_c_param(dplasma, d->param, init_func_body, init_func_body_size, 0));
        if(i + body_length >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        for(i = 0 ; i < MAX_CALL_PARAM_COUNT; i++) {
            /* params can have ranges here: don't use inline c expression */
            p += snprintf(whole + p, DEP_CODE_SIZE-p, "%s%s", dump_c_expression(d->call_params[i], init_func_body, init_func_body_size), 
                          i < MAX_CALL_PARAM_COUNT-1 ? ", " : "}};\n");
        }
        output("%s", whole);
        snprintf(name, FNAME_SIZE, "&dep%u", my_idx);
    }
     
    return name;
}

static char *dump_c_param(const dplasma_t *dplasma, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it)
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
            dep_name = dump_c_dep(dplasma, p->dep_in[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_IN_COUNT-1 ? ", " : "},\n"
                          "     .dep_out = {");
        }
        for(i = 0; i < MAX_DEP_OUT_COUNT; i++) {
            dep_name = dump_c_dep(dplasma, p->dep_out[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, PARAM_CODE_SIZE-l, "%s%s", dep_name, i < MAX_DEP_OUT_COUNT-1 ? ", " : "} };\n");
        }
        output("%s", param);
        snprintf(name, FNAME_SIZE, "&param%u", my_idx);
    }

    return name;
}

static char *dump_c_symbol(const symbol_t *s, char *init_func_body, int init_func_body_size)
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

    snprintf(mn, FNAME_SIZE, "%s", dump_c_expression(s->min, init_func_body, init_func_body_size));
    snprintf(mm, FNAME_SIZE, "%s", dump_c_expression(s->max, init_func_body, init_func_body_size));
    
    output("static symbol_t symb%d = { .flags = 0x%08x, .name = \"%s\", .min = %s, .max = %s };\n",
           i,
           s->flags, s->name, mn, mm);

    return e->c_name;
}

static void dump_all_global_symbols_c(char *init_func_body, int init_func_body_size)
{
    int i, l = 0;
    char whole[SYMBOL_CODE_SIZE];
    const symbol_t* symbol;

    l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "static symbol_t *dplasma_symbols[] = {\n");
    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        l += snprintf(whole+l, SYMBOL_CODE_SIZE-l, "   %s%s", 
                      dump_c_symbol(dplasma_symbol_get_element_at(i), init_func_body, init_func_body_size),
                      (i < (dplasma_symbol_get_count()-1)) ? ",\n" : "};\n");
    }
    output("%s", whole);

    output("\n");

    for(i = 0; i < dplasma_symbol_get_count(); i++) {
        char* current_symbol_pointer;
        symbol = dplasma_symbol_get_element_at(i);
        current_symbol_pointer = dump_c_symbol(symbol, init_func_body, init_func_body_size);
        if( (symbol->min != NULL) &&
            (symbol->max != NULL) &&
            ((symbol->min->flags & symbol->max->flags) & EXPR_FLAG_CONSTANT) &&
            (symbol->min->value == symbol->max->value) ) {
            output("static int %s = %d;\n", symbol->name, symbol->min->value);
        } else {
            output("static int %s;\n", symbol->name);
        }
        snprintf(init_func_body + strlen(init_func_body),
                 init_func_body_size - strlen(init_func_body),
                 "  {\n"
                 "    int rc;\n"
                 "    symbol_t* symbol = dplasma_search_global_symbol((%s)->name);\n"
                 "    if( NULL == symbol ) symbol = %s;\n"
                 "    if( 0 != (rc = expr_eval( symbol->min, NULL, 0, &%s)) ) {\n"
                 "      return rc;\n"
                 "    }\n"
                 "  }\n",
                 current_symbol_pointer, current_symbol_pointer, symbol->name);
    }
    output("\n");
}

static char *dump_c_dependency_list(dplasma_dependencies_t *d, char *init_func_body, int init_func_body_size)
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
                      d->flags, d->min, d->max, dump_c_symbol(d->symbol, init_func_body, init_func_body_size));
        if(  DPLASMA_DEPENDENCIES_FLAG_NEXT & d->flags ) {
            p += snprintf(b+p, DEPENDENCY_SIZE-p, "  .u.next = {%s} };\n", dump_c_dependency_list(d->u.next[0], init_func_body, init_func_body_size));
        } else {
            p += snprintf(b+p, DEPENDENCY_SIZE-p, "  .u.dependencies = { 0x%02x } };\n", d->u.dependencies[0]);
        }
        output("%s", b);
        snprintf(dname, FNAME_SIZE, "&deplist%d", my_idx);
    }
    return dname;
}

#include "remote_dep.h"

static void dplasma_dump_context_holder(const dplasma_t *d,
                                        char *init_func_body,
                                        int init_func_body_size)
{
    int i, j, k;
    char minexpr[MAX_EXPR_LEN];
    char maxexpr[MAX_EXPR_LEN];

    output("static inline long int %s_hash(",
           d->name);

    for(i = 0; i < d->nb_params; i++) {
        output("int %s%s", d->params[i]->name, i == d->nb_params-1 ? "){\n" : ", ");
    }

    output("  return 0");          
    for(i = 0; i < d->nb_params; i++) {
        output("+ ( (%s-(%s))* 1",  d->params[i]->name, expression_to_c_inline(d->params[i]->min, "", minexpr, MAX_EXPR_LEN));
        for(j = 0; j < i; j++) {
            output("* ((%s)+1-(%s))", 
                   expression_to_c_inline(d->params[j]->max, "", maxexpr, MAX_EXPR_LEN), 
                   expression_to_c_inline(d->params[j]->min, "", minexpr, MAX_EXPR_LEN));
        }
        output(")");
    }
    output(";\n"
           "}\n"
           "\n");

    k = 0;
    for(i = 0; i < MAX_PARAM_COUNT; i++) {
        if( d->inout[i] != NULL &&
            d->inout[i]->sym_type & SYM_OUT ) {
            k++;
        }
    }

    output("static data_repo_t *%s_repo = NULL;\n",
           d->name);
    snprintf(init_func_body + strlen(init_func_body),
             init_func_body_size - strlen(init_func_body),
             "  %s_repo = data_repo_create_nothreadsafe(4*4096, %d);\n",
             d->name, k);
}

static char *dplasma_to_data_repo_lookup_entry( const dplasma_t *d, char* prepend )
{
    int i, p;
    static char res[DATA_REPO_LOOKUP_SIZE];

    p = snprintf(res, DATA_REPO_LOOKUP_SIZE, "data_repo_lookup_entry( %s_repo, %s_hash(",
                 d->name, d->name);
    for(i = 0; i < d->nb_locals; i++) {
        p += snprintf(res + p, DATA_REPO_LOOKUP_SIZE-p, "%s%s%s", 
                      prepend, d->locals[i]->name,
                      i == d->nb_locals - 1 ? "), 1 )" : ", ");
    }
    
    return res;
}

static char *dep_to_data_repo_lookup_entry( const dep_t *d, char* prepend )
{
    int i, p;
    static char res[DATA_REPO_LOOKUP_SIZE];
    char expr[MAX_EXPR_LEN];

    p = snprintf(res, DATA_REPO_LOOKUP_SIZE, "data_repo_lookup_entry( %s_repo, %s_hash(",
                 d->dplasma->name, d->dplasma->name);
    for(i = 0; i < d->dplasma->nb_locals; i++) {
        p += snprintf(res + p, DATA_REPO_LOOKUP_SIZE-p, "%s%s", 
                      expression_to_c_inline( d->call_params[i], prepend, expr, MAX_EXPR_LEN ),
                      i == d->dplasma->nb_locals - 1 ? "), 0 )" : ", ");
    }
    
    return res;
}

#define DUMP_DECLARATION         0x1
#define DUMP_ASSIGNMENT_LINE     0x2
#define DUMP_VOID_LINE           0x4
static void dplasma_dump_locals_from_context(const dplasma_t *d,
                                             char* prepend,
                                             int dump_what,
                                             int additional_spaces)
{
    int i;
    for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
        if( dump_what & DUMP_DECLARATION ) {
            if( dump_what & DUMP_ASSIGNMENT_LINE ) {
                output("%*s  int %s%s = exec_context->locals[%d].value;\n", additional_spaces, "", prepend, d->locals[i]->name, i);
            } else {
                output("%*s  int %s%s;\n", additional_spaces, "", prepend, d->locals[i]->name);
            }
        } else {
            if( dump_what & DUMP_ASSIGNMENT_LINE ) {
                output("%*s  %s%s = exec_context->locals[%d].value;\n", additional_spaces, "", prepend, d->locals[i]->name, i);
            }
        }
        if( dump_what & DUMP_VOID_LINE ) {
            output("%*s  (void)%s%s;\n", additional_spaces, "", prepend, d->locals[i]->name);
        }
    }
}

static void dplasma_dump_dependency_helper(const dplasma_t *d,
                                           char *init_func_body,
                                           int init_func_body_size)
{
    int i, j, cpt, output_deps;
    char strexpr1[MAX_EXPR_LEN];
    char strexpr2[MAX_EXPR_LEN];
    char local_prepend[MAX_EXPR_LEN];

    snprintf(local_prepend, MAX_EXPR_LEN, "%s_", d->name);
    output("\nstatic int %s_release_dependencies(dplasma_execution_unit_t *context,\n"
           "                                   const dplasma_execution_context_t *exec_context,\n"
           "                                   int action_mask,\n"
           "                                   struct dplasma_remote_deps_t* upstream_remote_deps,\n"
           "                                   gc_data_t **data)\n"
           "{\n"
           "  int ret = 0, remote_deps_count = 0;\n"
           "  data_repo_entry_t *e%s;\n", 
           d->name, d->name);

    output("  dplasma_execution_context_t*   ready_list = NULL;\n"
           "  dplasma_remote_deps_t* remote_deps = upstream_remote_deps;\n"
           "  uint32_t usage = 0;\n"
           "  dplasma_execution_context_t new_context = { .function = NULL, .locals = {");
    for(j = 0; j < MAX_LOCAL_COUNT; j++) {
        output(" {.sym = NULL}%s", j+1 == MAX_LOCAL_COUNT ? "}};\n" : ", ");
    }

    dplasma_dump_locals_from_context(d, local_prepend, DUMP_DECLARATION | DUMP_ASSIGNMENT_LINE, 0);
    output("  e%s = %s;\n", d->name, dplasma_to_data_repo_lookup_entry(d, local_prepend));

    for( i = cpt = 0; i < MAX_PARAM_COUNT; i++) {
        if( (d->inout[i] != NULL) && (d->inout[i]->sym_type & SYM_OUT) ) {
            output("  e%s->data[%d] = (NULL != data) ? data[%d] : NULL;\n", d->name, cpt, cpt);
            cpt++;
        }
    }

    for(i = output_deps = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
        if( (NULL != d->inout[i]) && (d->inout[i]->sym_type & SYM_OUT) ) {
            output_deps++;
        }
    }

    for(i = cpt = 0; i < MAX_PARAM_COUNT; i++) {
        if( (NULL != d->inout[i]) && (d->inout[i]->sym_type & SYM_OUT) ) {
            int spaces = 0;
            
            struct param *p = d->inout[i];
            
            for(j = 0; j < MAX_DEP_OUT_COUNT; j++) {
                if( (NULL != p->dep_out[j]) && (p->dep_out[j]->dplasma->nb_locals > 0) ) {
                    struct dep *dep = p->dep_out[j];
                    dplasma_t* target = dep->dplasma;
                    int k;

                    output("%*s  /**\n"
                           "%*s   * Release %s OUTPUT dependencies for %s(",
                           spaces, "", spaces, "", 
                           dep->param->name, dep->dplasma->name);
                    /* Prepare the list of locals on the target */
                    for(k = 0; k < MAX_CALL_PARAM_COUNT; k++) {
                        if( NULL == dep->call_params[k] ) break;
                        if( EXPR_OP_BINARY_RANGE != dep->call_params[k]->op ) {
                            output(" %s", expression_to_c_inline(dep->call_params[k], "", strexpr1, MAX_EXPR_LEN));
                        } else {
                            output(" %s", expression_to_c_inline(dep->call_params[k]->bop1, "", strexpr1, MAX_EXPR_LEN));
                            output("..");
                            output("%s", expression_to_c_inline(dep->call_params[k]->bop2, "", strexpr1, MAX_EXPR_LEN));
                        }
                        if( NULL != dep->call_params[k+1] )
                            output(",");
                        else
                            output(")\n");
                    }
                    output("%*s   */\n"
                           "%*s  if( action_mask & (1 << %d) ) {\n",
                           spaces, "", spaces, "", cpt);

                    output("%*s    assert( strcmp( exec_context->function->inout[%d]->dep_out[%d]->dplasma->name, \"%s\") == 0 );\n",
                           spaces, "", i, j, dep->dplasma->name);
                    output("%*s    new_context.function = exec_context->function->inout[%d]->dep_out[%d]->dplasma; /* %s */\n",
                           spaces, "", i, j, dep->dplasma->name);
                    if( NULL != dep->cond ) {
                        output("%*s    if(%s) {\n", spaces, "", expression_to_c_inline(dep->cond, local_prepend, strexpr1, MAX_EXPR_LEN));
                    }

                    /* Prepare the list of locals on the target */
                    for(k = 0; k < MAX_CALL_PARAM_COUNT; k++) {
                        if( NULL != dep->call_params[k] ) {
                            output("%*s    int %s", spaces, "", target->locals[k]->name);
                            if( EXPR_OP_BINARY_RANGE != dep->call_params[k]->op ) {
                                output(" = %s", expression_to_c_inline(dep->call_params[k], local_prepend, strexpr1, MAX_EXPR_LEN));
                            }
                            output(";  /* %s local variable %s */\n", dep->dplasma->name, target->locals[k]->name);
                        }
                    }

                    for(k = 0; k < MAX_CALL_PARAM_COUNT; k++) {
                        if( (NULL != dep->call_params[k]) && (EXPR_OP_BINARY_RANGE == dep->call_params[k]->op) ) {
                            output("%*s    for(%s = %s; %s <= %s; %s++) {\n", spaces, "", 
                                   target->locals[k]->name, expression_to_c_inline(dep->call_params[k]->bop1, local_prepend, strexpr1, MAX_EXPR_LEN),
                                   target->locals[k]->name, expression_to_c_inline(dep->call_params[k]->bop2, local_prepend, strexpr2, MAX_EXPR_LEN),
                                   target->locals[k]->name);
                            spaces += 2;
                        }
                    }

                    /******************************************************/
                    /* Compute predicates                                 */
                    output("%*s    if( (1)", spaces, "");
                    for(k = 0; NULL != target->preds[k]; k++) {
                        int l;
                        output(" && %s_pred%d(%s", target->name, k, target->locals[0]->name);
                        for( l = 1; l < target->nb_locals; l++ ) {
                            output(",%s", target->locals[l]->name);
                        }
                        output(")");
                    }
                    output(" ) {\n");

                    for(k = 0; k < dep->dplasma->nb_locals; k++) {
                        if( 0 == k ) {
                            output("%*s      struct dplasma_dependencies_t** %s_placeholder = &(new_context.function->deps);\n",
                                   spaces, "", target->locals[0]->name);
                        } else {
                            output("%*s      struct dplasma_dependencies_t** %s_placeholder = &((*%s_placeholder)->u.next[%s - (*%s_placeholder)->min]);\n",
                                   spaces, "", target->locals[k]->name, target->locals[k-1]->name, target->locals[k-1]->name, target->locals[k-1]->name);
                        }
                    }

                    for(k = 0; k < target->nb_locals; k++) {
                        output("%*s      new_context.locals[%d].sym = new_context.function->locals[%d]; /* task %s */\n",
                               spaces, "", k, k, target->name);
                        output("%*s      new_context.locals[%d].value = %s;  /* task %s local %s */\n",
                               spaces, "", k, target->locals[k]->name, target->name, target->locals[k]->name);
                    }

                    output( "%*s      usage++;\n"
                            "%*s      gc_data_ref( e%s->data[%d] /* %s of %s is used by %s */ );\n",
                            spaces, "",
                            spaces, "", d->name, cpt, p->name, d->name, target->name);

                    output( "%*s      ret += dplasma_release_local_OUT_dependencies(context, exec_context, \n"
                            "%*s                       exec_context->function->inout[%d/*i*/],\n"
                            "%*s                       &new_context,\n"
                            "%*s                       exec_context->function->inout[%d/*i*/]->dep_out[%d/*j*/]->param,\n"
                            "%*s                       %s_placeholder, &ready_list);\n",
                            spaces, "",
                            spaces, "", i,
                            spaces, "", 
                            spaces, "", i, j,
                            spaces, "", target->locals[target->nb_locals-1]->name);

                    /* If predicates don't verify, this is remote, compute 
                     * target rank from predicate values
                     */
                    {
                        expr_t *rowpred; 
                        expr_t *colpred;
                        symbol_t *rowsize;
                        symbol_t *colsize;
                            
                        if(dplasma_remote_dep_get_rank_preds((const expr_t **)target->preds, 
                                                             &rowpred, 
                                                             &colpred, 
                                                             &rowsize,
                                                             &colsize) < 0) {
                            output("%*s    } else if (action_mask & DPLASMA_ACTION_RELEASE_REMOTE_DEPS) {\n"
                                   "%*s      DEBUG((\"GRID is not defined in JDF, but predicates are not verified. Your jdf is incomplete or your predicates false.\\n\"));\n"
                                   "%*s    }\n", 
                                   spaces, "",
                                   spaces, "",
                                   spaces, "");
                        } else {
                            output( "#if defined(DISTRIBUTED)\n"                                                                 /* line  1 */
                                    "%*s    } else if (action_mask & DPLASMA_ACTION_RELEASE_REMOTE_DEPS ) {\n"                   /* line  2 */
                                    "%*s      int rank, rrank, crank, ncols, array_pos, array_mask;\n"                           /* line  3 */
                                    "%*s      rrank = %s;\n"                                                                     /* line  4 */
                                    "%*s      crank = %s;\n"                                                                     /* line  5 */
                                    "%*s      ncols = %s;\n"                                                                     /* line  6 */
                                    "%*s      rank = crank + rrank * ncols;\n"                                                   /* line  7 */
                                    "%*s      array_pos = rank / (8 * sizeof(uint32_t));\n"                                      /* line  8 */
                                    "%*s      array_mask = 1 << (rank %% (8 * sizeof(uint32_t)));\n"                             /* line  9 */
                                    "%*s      DPLASMA_ALLOCATE_REMOTE_DEPS_IF_NULL(remote_deps, exec_context, %d);\n"            /* line 10 */
                                    "%*s      if( !(remote_deps->output[%d].rank_bits[array_pos] & array_mask) ) {\n"            /* line 11 */
                                    "%*s        remote_deps->output[%d].data = data[%d];\n"                                      /* line 12 */
                                    "%*s        remote_deps->output[%d].type = &%s;\n"                                           /* line 13 */
                                    "%*s        remote_deps->output[%d].rank_bits[array_pos] |= array_mask;\n"                   /* line 14 */
                                    "%*s        remote_deps->output[%d].count++; remote_deps_count++;\n"                         /* line 15 */
                                    "%*s      }\n"                                                                               /* line 16 */
                                    "#endif  /* defined(DISTRIBUTED) */\n"                                                       /* line 17 */
                                    "%*s    }\n",                                                                                /* line 18 */
                                    /* line  2 */ spaces, "",
                                    /* line  3 */ spaces, "",
                                    /* line  4 */ spaces, "", expression_to_c_inline(rowpred, "", strexpr1, MAX_EXPR_LEN),
                                    /* line  5 */ spaces, "", expression_to_c_inline(colpred, "", strexpr2, MAX_EXPR_LEN),
                                    /* line  6 */ spaces, "", colsize->name,
                                    /* line  7 */ spaces, "",
                                    /* line  8 */ spaces, "",
                                    /* line  9 */ spaces, "",
                                    /* line 10 */ spaces, "", output_deps,
                                    /* line 11 */ spaces, "", cpt,
                                    /* line 12 */ spaces, "", cpt, cpt,
                                    /* line 13 */ spaces, "", cpt, NULL == p->dep_out[j]->mpi_type ? "DPLASMA_DEFAULT_DATA_TYPE" : p->dep_out[j]->mpi_type,
                                    /* line 14 */ spaces, "", cpt,
                                    /* line 15 */ spaces, "", cpt,
                                    /* line 16 */ spaces, "",
                                    /* line 17 */
                                    /* line 18 */ spaces, ""
                                    );
                        }
                    }                    
                    
                    for(k = MAX_PARAM_COUNT-1; k >= 0; k--) {
                        if( NULL != dep->call_params[k] ) {
                            if( EXPR_OP_BINARY_RANGE == dep->call_params[k]->op ) {
                                spaces -= 2;
                                output("%*s    }\n", spaces, "");
                            }
                        }
                    }

                    if( NULL != dep->cond ) {
                        output("%*s  }  /* if(%s) */\n", spaces, "", 
                               expression_to_c_inline(dep->cond, local_prepend, strexpr1, MAX_EXPR_LEN));
                    }

                    output("  }\n");
                }
            }
            cpt++;
        }
    }
    output("  data_repo_entry_set_usage_limit(%s_repo, e%s->key, usage);\n"                         /* line  1 */
           "  if( NULL != ready_list )\n"                                                           /* line  2 */
           "    __dplasma_schedule(context, ready_list);\n"                                         /* line  3 */
           "#if defined(DISTRIBUTED)\n"                                                             /* line  4 */
           "  if( (action_mask & DPLASMA_ACTION_RELEASE_REMOTE_DEPS) && remote_deps_count ) {\n"    /* line  5 */
           "    ret += dplasma_remote_dep_activate(context,\n"                                      /* line  6 */
           "                                       exec_context,\n"
           "                                       remote_deps,\n"                                  /* line  7 */
           "                                       remote_deps_count);\n"                           /* line  8 */
           "  }\n"                                                                                  /* line  9 */
           "#endif  /* defined(DISTRIBUTED) */\n"                                                   /* line 10 */
           "  return ret;\n"                                                                        /* line 11 */
           "}\n\n",                                                                                 /* line 12 */
           /* line  1 */ d->name, d->name);
}

#if defined(DPLASMA_CACHE_AWARENESS)
static void dplasma_dump_cache_evaluation_function(const dplasma_t *d,
                                                   char *init_func_body,
                                                   int init_func_body_size)
{
    int i, j, k, cpt, pointers_cpt;
    char strexpr1[MAX_EXPR_LEN];
    output( "static unsigned int %s_cache_rank(dplasma_execution_context_t *exec_context, const cache_t *cache, unsigned int reward)\n"
            "{\n"
            "  int result = 0, r;\n"
            "  const cache_t *c;\n",
            d->name);
    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) 
        if( d->inout[i]->sym_type & SYM_IN ) {
            output("  gc_data_t *%s;\n", d->inout[i]->name);
            output("  data_repo_entry_t *e%s;\n", d->inout[i]->name);
        }
	dplasma_dump_locals_from_context(d, "", DUMP_DECLARATION | DUMP_ASSIGNMENT_LINE, 0);

    output("  if( NULL == exec_context->pointers[1] ) {\n");

    pointers_cpt = 0;
    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
        if( d->inout[i]->sym_type & SYM_IN ) {
            for(k = 0; k < MAX_DEP_IN_COUNT; k++) {
                if( d->inout[i]->dep_in[k] != NULL ) {
                    if( NULL != d->inout[i]->dep_in[0]->cond ) {
                        output("    if(%s) {\n", expression_to_c_inline(d->inout[i]->dep_in[k]->cond, "", strexpr1, MAX_EXPR_LEN));
                        if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                            output("      e%s = %s;\n", 
                                   d->inout[i]->name,
                                   dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                            output("      exec_context->pointers[%d] = e%s;\n", 2*pointers_cpt, d->inout[i]->name);
                        } else {
                            output("      exec_context->pointers[%d] = NULL;\n", 2*pointers_cpt);
                        }
                        output("      exec_context->pointers[%d] = ", 2*pointers_cpt + 1);
                    } else {
                        if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                            output("    e%s = %s;\n", 
                                   d->inout[i]->name,
                                   dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                            output("    exec_context->pointers[%d] = e%s;\n", 2 * pointers_cpt, d->inout[i]->name);
                        } else {
                            output("    exec_context->pointers[%d] = NULL;\n", 2 * pointers_cpt);
                        }
                        output("    exec_context->pointers[%d] = ", 2 * pointers_cpt + 1);
                    }
                    if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                        cpt = 0;
                        for(j = 0; j < MAX_PARAM_COUNT; j++) {
                            if( d->inout[i]->dep_in[k]->dplasma->inout[j] == d->inout[i]->dep_in[k]->param )
                                break;
                            if( d->inout[i]->dep_in[k]->dplasma->inout[j]->sym_type & SYM_OUT )
                                cpt++;
                        }
                        output("e%s->data[%d];\n", d->inout[i]->name, cpt);
                    } else {
                        output( "%s;\n", dplasma_dep_dplasma_call_to_c( d->inout[i]->dep_in[k], strexpr1, MAX_EXPR_LEN) );
                    }
                    if( NULL != d->inout[i]->dep_in[k]->cond ) {
                        output("    }\n");
                    }
                }
            }
            pointers_cpt++;
        }         
    }
        
    output("  }\n");
    
    pointers_cpt = 0;
    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
        if( d->inout[i]->sym_type & SYM_IN ) {
            output("  e%s = exec_context->pointers[%d];\n", d->inout[i]->name, 2 * pointers_cpt);
            output("  %s = exec_context->pointers[%d];\n", d->inout[i]->name, 2 * pointers_cpt + 1);
            pointers_cpt++;
        }
    }

    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
        if( d->inout[i]->sym_type & SYM_IN ) {
            output("  r = reward;\n"
                   "  for( c = cache; NULL != c; c = c->parent ) {\n"
                   "    if( cache_buf_isLocal(c, %s) ) {\n"
                   "      result += r;\n"
                   "      break;\n"
                   "    }\n"
                   "    r = r / 2;\n"
                   "  }\n",
                   d->inout[i]->name);
        }
    }
    output("  return result;\n"
           "}\n");
}
#endif

static char *dplasma_dump_c(const dplasma_t *d,
                            char *init_func_body,
                            int init_func_body_size)
{
    static char dp_txt[DPLASMA_SIZE];
    static int next_shape_idx = 0;
    int i, j, k, cpt, pointers_cpt = 0;
    int p = 0;
    char strexpr1[MAX_EXPR_LEN];

    (void)pointers_cpt;

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "    {\n");
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .name   = \"%s\",\n", d->name);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .flags  = 0x%02x,\n", d->flags);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .dependencies_mask = 0x%02x,\n", d->dependencies_mask);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .nb_locals = %d,\n", d->nb_locals);
    
#if defined(DPLASMA_CACHE_AWARENESS)
    if( NULL != d->body ) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .cache_rank_function = %s_cache_rank,\n", d->name);
    } else {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .cache_rank_function = NULL,\n");
    }
#endif

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .locals = {");
    for(i = 0; i < d->nb_locals; i++) {
        if( symbol_c_index_lookup(d->locals[i]) > -1 ) {
            p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "dplasma_symbols[%d]%s", 
                          symbol_c_index_lookup(d->locals[i]),
                          i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
        } else {
            p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s", 
                          dump_c_symbol(d->locals[i], init_func_body, init_func_body_size),
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
                      dump_c_expression_inline(d->preds[i], (const symbol_t**)d->locals, d->nb_locals, init_func_body, init_func_body_size),
                      i < MAX_PRED_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .inout= {");
    for(i = 0; i < MAX_PARAM_COUNT; i++) {
        p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "%s%s",
                      dump_c_param(d, d->inout[i], init_func_body, init_func_body_size, 1),
                      i < MAX_PARAM_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .deps = %s,\n", dump_c_dependency_list(d->deps, init_func_body, init_func_body_size));
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .hook = NULL\n");
    //    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "      .body = \"%s\"\n", d->body);
    p += snprintf(dp_txt+p, DPLASMA_SIZE-p, "    }");
    
    /* d->body == NULL <=> IN or OUT. Is it the good test? */
    if( NULL != d->body ) {
        int body_lines;

        dplasma_dump_dependency_helper(d, init_func_body, init_func_body_size);

#if defined(DPLASMA_CACHE_AWARENESS)
        dplasma_dump_cache_evaluation_function(d, init_func_body, init_func_body_size);

        output( "static int %s_hook(dplasma_execution_unit_t* context, dplasma_execution_context_t *exec_context)\n"
                "{\n"
				"  (void)context;\n",
                d->name);

        dplasma_dump_locals_from_context(d, "", DUMP_DECLARATION, 0);
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++)  {
            output("  void *%s;\n", d->inout[i]->name);
            output("  gc_data_t *g%s;\n", d->inout[i]->name);
            output("  data_repo_entry_t *e%s;\n", d->inout[i]->name);
        }
        dplasma_dump_locals_from_context(d, "", DUMP_ASSIGNMENT_LINE, 0);
        
        output("  if( NULL == exec_context->pointers[1] ) {\n"
               "    /* remove warnings in case the variable is not used later */\n");
        dplasma_dump_locals_from_context(d, "", DUMP_VOID_LINE, 2);
        
        output("#warning \"This is untested Code\"\n");

        pointers_cpt = 0;
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_IN ) {
                for(k = 0; k < MAX_DEP_IN_COUNT; k++) {
                    if( d->inout[i]->dep_in[k] != NULL ) {
                        if( NULL != d->inout[i]->dep_in[0]->cond ) {
                            output("    if(%s) {\n", expression_to_c_inline(d->inout[i]->dep_in[k]->cond, "", strexpr1, MAX_EXPR_LEN));
                            if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                                output("      e%s = %s;\n",
                                       d->inout[i]->name,
                                       dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                                output("      exec_context->pointers[%d] = e%s;\n", 2*pointers_cpt, d->inout[i]->name);
                            } else {
                                output("      exec_context->pointers[%d] = NULL;\n", 2*pointers_cpt);
                            }
                            output("      g%s = ", d->inout[i]->name);
                        } else {
                            if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                                output("    e%s = %s;\n",
                                       d->inout[i]->name,
                                       dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                                output("    exec_context->pointers[%d] = e%s;\n", 2 * pointers_cpt, d->inout[i]->name);
                            } else {
                                output("    exec_context->pointers[%d] = NULL;\n", 2 * pointers_cpt);
                            }
                            output("    g%s = ", d->inout[i]->name);
                        }
                        if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                            cpt = 0;
                            for(j = 0; j < MAX_PARAM_COUNT; j++) {
                                if( d->inout[i]->dep_in[k]->dplasma->inout[j] == d->inout[i]->dep_in[k]->param )
                                    break;
                                if( d->inout[i]->dep_in[k]->dplasma->inout[j]->sym_type & SYM_OUT )
                                    cpt++;
                            }
                            output("e%s->data[%d];\n", d->inout[i]->name, cpt);
                        } else {
                            output( "gc_data_new(%s, 0);\n", dplasma_dep_dplasma_call_to_c( d->inout[i]->dep_in[k], strexpr1, MAX_EXPR_LEN) );
                        }
                        if( NULL != d->inout[i]->dep_in[k]->cond ) {
                            output("    }\n");
                        }
                        output("    exec_context->pointers[%d] = g%s;\n", 2*pointers_cpt + 1, d->inout[i]->name);
                        output("    gc_data_ref( g%s );\n", d->inout[i]->name);
                    }
                }
                pointers_cpt++;
            } else {
                output("    (void)%s;\n", d->inout[i]->name);
            }            
        }

        output("#warning \"End of untested Code\"\n");
        
        output("  }\n");
        
        pointers_cpt = 0;
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_IN ) {
                output("  e%s = exec_context->pointers[%d];\n", d->inout[i]->name, 2 * pointers_cpt);
                output("  %s = exec_context->pointers[%d];\n", d->inout[i]->name, 2 * pointers_cpt + 1);
                pointers_cpt++;
            }
        }
#else
        output( "static int %s_hook(dplasma_execution_unit_t* context, dplasma_execution_context_t *exec_context)\n"
                "{\n"
				"  (void)context;\n",
                d->name);

        dplasma_dump_locals_from_context(d, "", DUMP_DECLARATION | DUMP_ASSIGNMENT_LINE, 0);
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            output("  void *%s = NULL;\n", d->inout[i]->name);
            output("  gc_data_t *g%s = NULL;\n", d->inout[i]->name);
            output("  data_repo_entry_t *e%s = NULL;\n", d->inout[i]->name);
        }

        output("  /* remove warnings in case the variable is not used later */\n");
        dplasma_dump_locals_from_context(d, "", DUMP_VOID_LINE, 0);

        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_IN ) {
                for(k = 0; k < MAX_DEP_IN_COUNT; k++) {
                    if( d->inout[i]->dep_in[k] != NULL ) {
                       
                        if( NULL != d->inout[i]->dep_in[0]->cond ) {
                            output("  if(%s) {\n", expression_to_c_inline(d->inout[i]->dep_in[k]->cond, "", strexpr1, MAX_EXPR_LEN));
                            if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                                output("    e%s = %s;\n", 
                                       d->inout[i]->name,
                                       dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                            }
                            output("    g%s = ", d->inout[i]->name);
                        } else {
                            if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                                output("  e%s = %s;\n",
                                       d->inout[i]->name,
                                       dep_to_data_repo_lookup_entry(d->inout[i]->dep_in[k], ""));
                            }
                            output("  g%s = ", d->inout[i]->name);
                        }
                        if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                            cpt = 0;
                            for(j = 0; j < MAX_PARAM_COUNT; j++) {
                                if( d->inout[i]->dep_in[k]->dplasma->inout[j] == d->inout[i]->dep_in[k]->param )
                                    break;
                                if( d->inout[i]->dep_in[k]->dplasma->inout[j]->sym_type & SYM_OUT )
                                    cpt++;
                            }
                            output("e%s->data[%d];\n", d->inout[i]->name, cpt);
                        } else {
                            output( "gc_data_new(%s, 0);\n", dplasma_dep_dplasma_call_to_c( d->inout[i]->dep_in[k], strexpr1, MAX_EXPR_LEN) );
                        }
                        if( NULL != d->inout[i]->dep_in[k]->cond ) {
                            output("  }\n");
                        }
                    }
                }
                output("  %s = GC_DATA(g%s);\n", d->inout[i]->name, d->inout[i]->name);
            } else {
                output("  (void)%s;\n", d->inout[i]->name);
            }
            output("\n");
        }
#endif

        output( "\n"
                "#ifdef HAVE_PAPI\n"
                "  int i, num_events;\n"
                "  int events[MAX_EVENTS];\n"
                "  PAPI_list_events(eventSet, &events, &num_events);\n"
                "  long long values[num_events];\n"
                "  PAPI_start(eventSet);\n"
                "#endif\n"
                "\n");

        output( "#if defined(DPLASMA_CACHE_AWARENESS)\n");
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            output("  cache_buf_referenced(context->closest_cache, %s);\n", d->inout[i]->name);
        }
        output( "#endif /* DPLASMA_CACHE_AWARENESS */\n");

        output( "  TAKE_TIME(context, %s_start_key, %s_hash(",
                d->name, d->name);
        for(j = 0; j < d->nb_locals; j++) {
            output("%s", d->locals[j]->name );
            if( j == d->nb_locals - 1 ) 
                output("));\n");
            else
                output(", ");
        }
        body_lines = nblines(d->body);
        output( "  %s\n"
                "#line %d \"%s\"\n"
                "\n",
                d->body, body_lines+2+current_line, out_name);

        output("#if defined(DISTRIBUTED)\n"
               "  /** If not working on distributed, there is no risk that datas are not in place */ \n");
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_OUT ) {
                for(k = 0; k < MAX_DEP_OUT_COUNT; k++) {
                    if( d->inout[i]->dep_out[k] != NULL ) {
                        if( d->inout[i]->dep_out[k]->dplasma->nb_locals == 0 ) {
                            if( NULL != d->inout[i]->dep_out[k]->cond ) {
                                output("  if(%s) {\n  ", expression_to_c_inline(d->inout[i]->dep_out[k]->cond, "", strexpr1, MAX_EXPR_LEN));
                            }
                            dplasma_dep_dplasma_call_to_c( d->inout[i]->dep_out[k], strexpr1, MAX_EXPR_LEN);
                            output("%s  if(%s != %s)\n"
                                   "%s    dplasma_remote_dep_memcpy( %s, g%s, %s );\n",
                                   NULL != d->inout[i]->dep_out[k]->cond ? "  " : "", d->inout[i]->name, strexpr1,
                                   NULL != d->inout[i]->dep_out[k]->cond ? "  " : "", strexpr1, d->inout[i]->name,
                                   NULL == d->inout[i]->dep_out[k]->mpi_type ? "DPLASMA_DEFAULT_DATA_TYPE" : d->inout[i]->dep_out[k]->mpi_type);
                            if( NULL != d->inout[i]->dep_out[k]->cond ) {
                                output(  "}\n");
                            }
                        }
                    }
                }
            }
        }
        output("#endif /* defined(DISTRIBUTED) */\n");

        output( "  TAKE_TIME(context, %s_end_key, %s_hash(",
                d->name, d->name);
        for(j = 0; j < d->nb_locals; j++) {
            output("%s", d->locals[j]->name );
            if( j == d->nb_locals - 1 ) 
                output("));\n");
            else
                output(", ");
        }

        output( "\n"
                "#ifdef HAVE_PAPI\n"
                "  PAPI_stop(eventSet, &values);\n"
                "  if(num_events > 0) {\n"
                "    printf(\"PAPI counter values from %5s (thread=%%ld): \", context->eu_id);\n"  
                "    for(i=0; i<num_events; ++i) {\n"
                "      char event_name[PAPI_MAX_STR_LEN];\n"
                "      PAPI_event_code_to_name(events[i], &event_name);\n"
                "      printf(\"   %%s  %%lld \", event_name, values[i]);\n"
                "    }\n"
                "    printf(\"\\n\");\n"
                "  }\n"
                "#endif\n"
                "\n", d->name);
        
        output( "#if defined(DPLASMA_GRAPHER)\n"
                "if( NULL != __dplasma_graph_file ) {\n"
                "  char tmp[128];\n"
                "  dplasma_service_to_string(exec_context, tmp, 128);\n"
                "  fprintf(__dplasma_graph_file,\n"
                "          \"%%s [shape=\\\"%s\\\",style=filled,fillcolor=\\\"%%s\\\",fontcolor=\\\"black\\\",label=\\\"%%s\\\",tooltip=\\\"%s%%ld\\\"];\\n\",\n"
                "          tmp, colors[context->eu_id], tmp, %s_hash(",
                shapes[next_shape_idx++ % SHAPES_SIZE], d->name, d->name);
        for(j = 0; j < d->nb_locals; j++) {
            output("%s", d->locals[j]->name );
            if( j == d->nb_locals - 1 ) 
                output("));\n");
            else
                output(", ");
        }
        output("}\n"
               "#endif /* defined(DPLASMA_GRAPHER) */\n");

        cpt = 0;
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_OUT ) {
                cpt++;
            }
        }
        output("  {\n"
               "    gc_data_t *data[%d];\n",
               cpt);
        cpt=0;
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_OUT ) {
                output("    data[%d] = g%s;\n", cpt++, d->inout[i]->name);
            }
        }
        output("    %s_release_dependencies(context, exec_context, \n"
               "                            DPLASMA_ACTION_RELEASE_REMOTE_DEPS | DPLASMA_ACTION_DEPS_MASK, NULL, data);\n"
               "  }\n",
               d->name);
     
        for(i = 0; i < MAX_PARAM_COUNT && NULL != d->inout[i]; i++) {
            if( d->inout[i]->sym_type & SYM_IN ) {
                for(k = 0; k < MAX_DEP_IN_COUNT; k++) {
                    if( d->inout[i]->dep_in[k] != NULL ) {
                        if( d->inout[i]->dep_in[k]->dplasma->nb_locals != 0 ) {
                            if( NULL != d->inout[i]->dep_in[k]->cond ) {
                                output("  if(%s) {\n"
                                       "    data_repo_entry_used_once( %s_repo, e%s->key );\n",
                                       expression_to_c_inline(d->inout[i]->dep_in[k]->cond, "", strexpr1, MAX_EXPR_LEN),
                                       d->inout[i]->dep_in[k]->dplasma->name,
                                       d->inout[i]->name);
                                output("    (void)gc_data_unref(g%s);\n", d->inout[i]->name);
                                output("  }\n");
                            } else {
                                output("  data_repo_entry_used_once( %s_repo, e%s->key );\n",
                                       d->inout[i]->dep_in[k]->dplasma->name,
                                       d->inout[i]->name);
                                output("  (void)gc_data_unref(g%s);\n", d->inout[i]->name);
                            }
                        }
                    }
                }
            }
        }

        output( "  return 0;\n"
                "}\n"
                "\n");
    }

    return dp_txt;
}

static void dump_tasks_enumerator(const dplasma_t *d, char *init_func_body, int init_func_body_size)
{
    int spaces;
    size_t spaces_length;
    int s, p, has_preds;
    char strexpr1[MAX_EXPR_LEN];

    if(d->body == NULL)
        return;

    spaces = 2;
    output("%*s/* %s */\n", spaces, "", d->name);

    output("%*s{\n", spaces, "");

    spaces += 2;

    for(s = 0; s < d->nb_locals; s++) {
        output("%*sint %s, %s_start, %s_end;\n", spaces, "", d->locals[s]->name, d->locals[s]->name, d->locals[s]->name );
        output("%*sint %s_min, %s_max;\n", spaces, "", d->locals[s]->name, d->locals[s]->name );
        output("%*sdplasma_dependencies_t **%s_deps_location;\n", spaces, "", d->locals[s]->name);
    }
    for(p = 0; d->preds[p]!=NULL; p++) {
        output("%*sint pred%d;\n", spaces, "", p);
    }
    output("%*sfunction = (dplasma_t*)dplasma_find( \"%s\" );\n"
           "%*sfunction->deps = NULL;\n", 
           spaces, "", d->name, 
           spaces, "");
    output("%*sDEBUG((\"Prepare dependencies tracking for %s\\n\"));\n", 
           spaces, "", d->name );
    for(s = 0; s < d->nb_locals; s++) {
        output("%*s%s_start = %s;\n", spaces, "", d->locals[s]->name, expression_to_c_inline(d->locals[s]->min, "", strexpr1, MAX_EXPR_LEN));
        output("%*s%s_end = %s;\n", spaces, "", d->locals[s]->name, expression_to_c_inline(d->locals[s]->max, "", strexpr1, MAX_EXPR_LEN));
        output("%*s%s_min = 0x7fffffff;\n", spaces, "", d->locals[s]->name);
        output("%*s%s_max = -1;\n", spaces, "", d->locals[s]->name);
        if( 0 == s ) {
            output("%*s%s_deps_location = &(function->deps);\n",
                   spaces, "", d->locals[s]->name);
        }
        output("%*sfor(%s = %s_start; %s <= %s_end; %s++) {\n",
               spaces, "", d->locals[s]->name, d->locals[s]->name, d->locals[s]->name,  d->locals[s]->name, d->locals[s]->name);
        spaces += 2;
        has_preds = 0;
        for(p = 0; d->preds[p] != NULL; p++) {
            if( EXPR_SUCCESS == expr_depend_on_symbol(d->preds[p], d->locals[s]) ) {
                int l;
                output("%*spred%d = %s_pred%d(%s", spaces, "", p, d->name, p, d->locals[0]->name);
                for( l = 1; l < d->nb_locals; l++ ) {
                    output(",%s", d->locals[l]->name);
                }
                output(");\n");
                has_preds++;
            }
        }
        if( has_preds ) {
            output("%*sif( !(1", spaces, "");
            for(p = 0; d->preds[p] != NULL; p++) {
                if( EXPR_SUCCESS == expr_depend_on_symbol(d->preds[p], d->locals[s]) ) {
                    output(" && pred%d", p);
                }
            }
            output(") ) continue;\n");
        }
        if( s > 0 )
            output("%*s%s_deps_location = &((*%s_deps_location)->u.next[%s - (*%s_deps_location)->min]);\n",
                   spaces, "", d->locals[s]->name, d->locals[s-1]->name, d->locals[s-1]->name, d->locals[s-1]->name);
        output("%*sif( NULL == *%s_deps_location ) {\n", spaces, "", d->locals[s]->name);
        if( has_preds ) {
            output("%*s  {int _%s; for(_%s = %s_start; _%s <= %s_end; _%s++) {\n"
                   "%*s    int %s = _%s;\n",
                   spaces, "", d->locals[s]->name, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name,
                   spaces, "", d->locals[s]->name, d->locals[s]->name);
            for(p = 0; d->preds[p] != NULL; p++) {
                if( EXPR_SUCCESS == expr_depend_on_symbol(d->preds[p], d->locals[s]) ) {
                    int l;
                    output("%*s    pred%d = %s_pred%d(%s", spaces, "", p, d->name, p, d->locals[0]->name);
                    for( l = 1; l < d->nb_locals; l++ ) {
                        output(",%s", d->locals[l]->name);
                    }
                    output(");\n");
                    has_preds++;
                    /*output("%*s    pred%d = %s;\n", spaces, "", p, expression_to_c_inline(d->preds[p], "", strexpr1, MAX_EXPR_LEN));*/
                }
            }
            output("%*s    if( !(1", spaces, "");
            for(p = 0; d->preds[p] != NULL; p++) {
                if( EXPR_SUCCESS == expr_depend_on_symbol(d->preds[p], d->locals[s]) ) {
                    output(" && pred%d", p);
                }
            }
            output(") ) continue;\n");
            output("%*s    if( _%s < %s_min ) %s_min = %s;\n"
                   "%*s    %s_max = %s;\n"
                   "%*s  }}\n",
                   spaces, "", d->locals[s]->name, d->locals[s]->name, d->locals[s]->name, d->locals[s]->name,
                   spaces, "", d->locals[s]->name, d->locals[s]->name,
                   spaces, "" );
        } else {
            output("%*s  %s_min = %s_start;\n"
                   "%*s  %s_max = %s_end;\n",
                   spaces, "", d->locals[s]->name, d->locals[s]->name,
                   spaces, "", d->locals[s]->name, d->locals[s]->name);
        }
        output("%*s  assert( -1 != %s_max );\n"
               "%*s  ALLOCATE_DEP_TRACKING(deps, %s_min, %s_max, \"%s\", function->locals[%d], *%s_deps_location);\n"
               "%*s  *%s_deps_location = deps;  /* store the deps in the right location */\n"
               "%*s}\n",
               spaces, "", d->locals[s]->name,
               spaces, "", d->locals[s]->name, d->locals[s]->name, d->locals[s]->name, s, (0 == s ? d->locals[s]->name : d->locals[s-1]->name),
               spaces, "", d->locals[s]->name,
               spaces, "");
    }
    spaces -= 2;
    output("%*s  nbtasks++;\n", spaces, "");

    for(s = d->nb_locals-1; s >= 0; s--) {
        output("%*s}  /* for %s */\n", spaces, "", d->locals[s]->name);
        spaces -= 2;
    }
        
    output("%*s}\n", spaces, "");
}

int dplasma_dump_all_c(char *filename)
{
    char whole[DPLASMA_ALL_SIZE];
    char body[INIT_FUNC_BODY_SIZE];
    preamble_list_t *n;
    const dplasma_t* object;
    int i, j, k, p = 0, object_output_deps, max_output_deps;
    
    out = fopen(filename, "w");
    if( out == NULL ) {
        return -1;
    }
    out_name = filename;

    current_line = 1;
    
    for(n = preambles; n != NULL; n = n->next) {
        if( strcasecmp(n->language, "C") == 0 ) {
            int nb = nblines(n->code);
            output( "%s\n"
                    "#line %d \"%s\"\n", 
                    n->code, nb+current_line+2, out_name);
        }
    }

    body[0] = '\0';
    
    dump_all_global_symbols_c(body, INIT_FUNC_BODY_SIZE);

    output( "#include <assert.h>\n"
            "#include <string.h>\n"
            "#include \"lifo.h\"\n"
            "#include \"remote_dep.h\"\n"
            "#include \"datarepo.h\"\n\n"
            "#define TILE_SIZE (DPLASMA_TILE_SIZE*DPLASMA_TILE_SIZE*sizeof(double))\n"
            "#ifdef HAVE_PAPI\n"
            "#include \"papi.h\"\n"
            "extern int eventSet;\n"
            "#endif\n"
            "\n"
            "#if defined(DPLASMA_GRAPHER)\n"
            "#include <stdio.h>\n"
            "extern FILE *__dplasma_graph_file;\n"
            "#define COLORS_SIZE %d\n"
            "static char *colors[%d] = {\n",
            COLORS_SIZE, COLORS_SIZE);

    for(i = 0; i < COLORS_SIZE; i++) {
        output("  \"%s\"%s", colors[i], i==COLORS_SIZE-1 ? "\n};\n" : ",\n");
    }

    output( "#endif /* defined(DPLASMA_GRAPHER) */\n"
            "#ifdef DPLASMA_PROFILING\n"
            "#include \"profiling.h\"\n");
    for(i = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        output("static int %s_start_key, %s_end_key;\n", object->name, object->name);
    }
    output( "#define TAKE_TIME(EU_CONTEXT, KEY, ID)  dplasma_profiling_trace((EU_CONTEXT)->eu_profile, (KEY), (ID))\n"
            "#else\n"
            "#define TAKE_TIME(EU_CONTEXT, KEY, ID)\n"
            "#endif  /* DPLASMA_PROFILING */\n"
            "\n"
            "#include \"scheduling.h\"\n"
            "\n");
    /* Dump Macros for all predicates */
    for( i = 0; i < dplasma_nb_elements(); i++ ) {
        char strexpr1[MAX_EXPR_LEN];
        const dplasma_t *object = dplasma_element_at(i);
        for( j = 0; j < MAX_PRED_COUNT; j++ ) {
            if( NULL == object->preds[j] ) break;
            output("#define %s_pred%d(%s", object->name, j, object->locals[0]->name);
            for( k = 1; k < object->nb_locals; k++ ) {
                output(",%s", object->locals[k]->name);
            }
            output(") %s\n", expression_to_c_inline(object->preds[j], "", strexpr1, MAX_EXPR_LEN));
        }
    }

    p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "static dplasma_t dplasma_array[%d] = {\n", dplasma_nb_elements());

    for(i = 0; i < dplasma_nb_elements(); i++) {
        const dplasma_t *d = dplasma_element_at(i);
        if( d->nb_locals != 0 ) {
            dplasma_dump_context_holder(d, body, INIT_FUNC_BODY_SIZE);
        }
    }

    for(i = 0; i < dplasma_nb_elements(); i++) {
        p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "%s", dplasma_dump_c(dplasma_element_at(i), body, INIT_FUNC_BODY_SIZE));
        if( i < dplasma_nb_elements()-1) {
            p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, ",\n");
        }
    }
    p += snprintf(whole+p, DPLASMA_ALL_SIZE-p, "};\n");
    output( "%s\n"
            "\n"
            "static int __dplasma_init(void)\n"
            "{\n"
            "%s\n"
            "  return 0;\n"
            "}\n"
            , whole, body);

    output( "int load_dplasma_objects( dplasma_context_t* context )\n"
            "{\n"
			"  (void)context;\n"
            "  dplasma_load_array( dplasma_array, %d );\n"
            "  dplasma_load_symbols( dplasma_symbols, %d );\n"
            "  return 0;\n"
            "}\n"
            "\n",
            dplasma_nb_elements(),
            dplasma_symbol_get_count());

    output( "int load_dplasma_hooks( dplasma_context_t* context )\n"
            "{\n"
            "  dplasma_t* object;\n"
            "\n"
			"  (void)context;\n"
            "  if( 0 != __dplasma_init()) {\n"
            "     return -1;\n"
            "  }\n"
            "\n");

    for(i = max_output_deps = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        /* Specials IN and OUT test */
        if( object->body != NULL ) {
            output("  object = (dplasma_t*)dplasma_find(\"%s\");\n"
                   "  object->hook = %s_hook;\n"
                   "  object->release_deps = %s_release_dependencies;\n\n",
                   object->name, object->name, object->name);
        }
        /* Compute the maximum number of output dependencies */
        for( j = object_output_deps = 0; j < MAX_PARAM_COUNT; j++ ) {
            if( (NULL != object->inout[j]) && (object->inout[j]->sym_type & SYM_OUT) ) {
                object_output_deps++;
            }
        }
        if( max_output_deps < object_output_deps ) {
            max_output_deps = object_output_deps;
        }
    }
    output("#if defined(DISTRIBUTED)\n"
           "  remote_deps_allocation_init(context->nb_nodes, %d);\n"
           "#endif  /* defined(DISTRIBUTED) */\n\n", max_output_deps
           );

    output("#ifdef DPLASMA_PROFILING\n");

    for(i = 0; i < dplasma_nb_elements(); i++) {
        object = dplasma_element_at(i);
        output( "  dplasma_profiling_add_dictionary_keyword( \"%s\", \"fill:%s\",\n"
                "                                            &%s_start_key, &%s_end_key);\n",
                object->name, colors[i % COLORS_SIZE], object->name, object->name);
    }

    output( "#endif /* DPLASMA_PROFILING */\n"
            "\n"
            "  return 0;\n"
            "}\n");
    output("#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP) \\\n"
           "do { \\\n"
           "  int _vmin = (vMIN); \\\n"
           "  int _vmax = (vMAX); \\\n"
           "  (DEPS) = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) + \\\n"
           "                                           (_vmax - _vmin) * sizeof(dplasma_dependencies_union_t)); \\\n"
           "  /*DEBUG((\"Allocate %%d spaces for loop %%s (min %%d max %%d) 0x%%p last_dep 0x%%p\\n\", */\\\n"
           "  /*       (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP))); */\\\n"
           "  (DEPS)->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL; \\\n"
           "  (DEPS)->symbol = (vSYMBOL); \\\n"
           "  (DEPS)->min = _vmin; \\\n"
           "  (DEPS)->max = _vmax; \\\n"
           "  (DEPS)->prev = (PREVDEP); /* chain them backward */ \\\n"
           "  if( NULL != (PREVDEP) ) {\\\n"
           "    (PREVDEP)->flags = DPLASMA_DEPENDENCIES_FLAG_NEXT | DPLASMA_DEPENDENCIES_FLAG_ALLOCATED;\\\n"
           "  }\\\n"
           "} while (0)\\\n"
           "\n"
           "int enumerate_dplasma_tasks(dplasma_context_t* context)\n"
           "{\n"
           "  int nbtasks = 0;\n"
           "  dplasma_t* function;\n"
           "  dplasma_dependencies_t *deps;\n");

    for(i = 0; i < dplasma_nb_elements(); i++) {
        dump_tasks_enumerator(dplasma_element_at(i), NULL, 0);
    }

    output( "  dplasma_register_nb_tasks(context, nbtasks);\n"
            "  return nbtasks;\n"
            "}\n"
            "\n");
    
    fclose(out);

    return 0;
}
