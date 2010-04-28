#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"

extern const char *yyfilename;

static FILE *cfile;
static int   cfile_lineno;
static FILE *hfile;
static int   hfile_lineno;
static const char *jdf_basename;

/** Optional declarations of local functions */
static int jdf_expr_depends_on_symbol(const char *varname, const jdf_expr_t *expr);
static void jdf_code_hook(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);

static int nblines(const char *p)
{
    int r = 0;
    for(; *p != '\0'; p++)
        if( *p == '\n' )
            r++;
    return r;
}

#if defined(__GNUC__)
static void coutput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void coutput(const char *format, ...)
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
        fwrite(res, len, 1, cfile);
        cfile_lineno += nblines(res);
        free(res);
    }
}

#if defined(__GNUC__)
static void houtput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void houtput(const char *format, ...)
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
        fwrite(res, len, 1, hfile);
        hfile_lineno += nblines(res);
        free(res);
    }
}

/** UTIL HELPERS **/

/**
 * dump_string:
 *  general function to use with UTIL_DUMP_LIST_FIELD.
 *  Transforms a single field pointing to an existing char * in the char *
 * @param [IN] elt: pointer to the char * (format useable by UTIL_DUMP_LIST_FIELD)
 * @param [IN] _:   ignored pointer to abide by UTIL_DUMP_LIST_FIELD format
 * @return the char * pointed by elt
 */
static char *dump_string(void **elt, void *_)
{
    return (char*)*elt;
}

/**
 * dump_globals:
 *   Dump a global symbol like #define ABC (__DAGuE_object->ABC)
 */
static char* dump_globals(void** elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s (__DAGuE_object->%s)", (char*)*elem, (char*)*elem );
    return string_arena_get_string(sa);
}

/**
 * dump_data:
 *   Dump a global symbol like
 *     #define ABC(A0, A1) (__DAGuE_object->ABC->data_of(__DAGuE_object->ABC, A0, A1))
 */
static char* dump_data(void** elem, void *arg)
{
    jdf_data_entry_t* data = (jdf_data_entry_t*)elem;
    static char str[1024];
    int i, len = 0;

    len += snprintf( str, 1024, "%s(%s%d", data->dname, data->dname, 0 );
    for( i = 1; i < data->nbparams; i++ ) {
        len += snprintf( str+len, 1024 - len, ",%s%d", data->dname, i );
    }
    len += snprintf(str+len, 1024 - len, ")  (__DAGuE_object->%s->data_of(__DAGuE_object->%s", 
                    data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        len += snprintf( str+len, 1024 - len, ", (%s%d)", data->dname, i );
    }
    len += snprintf( str+len, 1024 - len, "))\n" );
    return str;
}

/**
 * Parameters for dump_expr helper
 */
typedef struct expr_info {
    string_arena_t* sa;
    const char* prefix;
} expr_info_t;

/**
 * dump_expr:
 *   dumps the jdf_expr* pointed to by elem into arg->sa, prefixing each 
 *   non-global variable with arg->prefix
 */
static char * dump_expr(void **elem, void *arg)
{
    expr_info_t* expr_info = (expr_info_t*)arg;
    expr_info_t li, ri;
    jdf_expr_t *e = *(jdf_expr_t**)elem;
    string_arena_t *sa = expr_info->sa;
    string_arena_t *la, *ra;

    string_arena_init(sa);

    la = string_arena_new(8);
    ra = string_arena_new(8);

    li.sa = la;
    li.prefix = expr_info->prefix;
    
    ri.sa = ra;
    ri.prefix = expr_info->prefix;

    switch( e->op ) {
    case JDF_VAR: {
        jdf_global_entry_t* item = current_jdf.globals;
        while( item != NULL ) {
            if( !strcmp(item->name, e->jdf_var) ) {
                string_arena_add_string(sa, "%s", e->jdf_var);
                break;
            }
            item = item->next;
        }
        if( NULL == item ) {
            string_arena_add_string(sa, "%s%s", expr_info->prefix, e->jdf_var);
        }
        break;
    }
    case JDF_EQUAL:
        string_arena_add_string(sa, "(%s == %s)", 
                                dump_expr((void**)&e->jdf_ba1, &li), 
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_NOTEQUAL:
        string_arena_add_string(sa, "(%s != %s)", 
                                dump_expr((void**)&e->jdf_ba1, &li), 
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_AND:
        string_arena_add_string(sa, "(%s && %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_OR:
        string_arena_add_string(sa, "(%s || %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_XOR:
        string_arena_add_string(sa, "(%s ^ %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_LESS:
        string_arena_add_string(sa, "(%s <  %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_LEQ:
        string_arena_add_string(sa, "(%s <= %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MORE:
        string_arena_add_string(sa, "(%s >  %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MEQ:
        string_arena_add_string(sa, "(%s >= %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_NOT:
        string_arena_add_string(sa, "!%s",
                                dump_expr((void**)&e->jdf_ua, &li));
        break;
    case JDF_PLUS:
        string_arena_add_string(sa, "(%s + %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MINUS:
        string_arena_add_string(sa, "(%s - %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_TIMES:
        string_arena_add_string(sa, "(%s * %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_DIV:
        string_arena_add_string(sa, "(%s / %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MODULO:
        string_arena_add_string(sa, "(%s %% %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_SHL:
        string_arena_add_string(sa, "(%s << %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_SHR:
        string_arena_add_string(sa, "(%s >> %s)",
                                dump_expr((void**)&e->jdf_ba1, &li),
                                dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_RANGE:
        break;
    case JDF_TERNARY: {
        expr_info_t ti;
        string_arena_t *ta;
        ta = string_arena_new(8);
        ti.sa = ta;
        ti.prefix = expr_info->prefix;

        string_arena_add_string(sa, "(%s ? %s : %s)",
                                dump_expr((void**)&e->jdf_tat, &ti),
                                dump_expr((void**)&e->jdf_ta1, &li),
                                dump_expr((void**)&e->jdf_ta2, &ri) );

        string_arena_free(ta);
        break;
    }
    case JDF_CST:
        string_arena_add_string(sa, "%d", e->jdf_cst);
        break;
    default:
        string_arena_add_string(sa, "DonKnow");
        break;
    }
    string_arena_free(la);
    string_arena_free(ra);
    
    return string_arena_get_string(sa);
}

/**
 * Dump a predicate like 
 *  #define F_pred(k, n, m) (__DAGuE_object->ABC->rank == __DAGuE_object->ABC->rank_of(__DAGuE_object->ABC, k, n, m))
 */
static char* dump_predicate(void** elem, void *arg)
{
    jdf_function_entry_t *f = (jdf_function_entry_t *)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    expr_info_t expr_info;

    jdf_call_t* call = f->predicate;
    int i, len = 0;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s_pred(%s) ",
                            f->fname,
                            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name,
                                                 dump_string, NULL, 
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
    string_arena_add_string(sa, "(__DAGuE_object->%s->myrank == __DAGuE_object->%s->rank_of(__DAGuE_object->%s, %s))", 
                            f->predicate->func_or_mem, f->predicate->func_or_mem, f->predicate->func_or_mem,
                            UTIL_DUMP_LIST_FIELD(sa2, f->predicate->parameters, next, expr,
                                                 dump_expr, &expr_info,
                                                 "", "", ", ", "")); 

    string_arena_free(sa2);
    string_arena_free(sa3);
    return string_arena_get_string(sa);
}

/**
 * Parameters of the dump_assignments function
 */
typedef struct assignment_info {
    string_arena_t *sa;
    int idx;
    const jdf_expr_t *expr;
} assignment_info_t;

/**
 * dump_assignments:
 *  Takes the pointer to the name of a parameter,
 *  a pointer to a dump_info, and display k = assignments[dump_info.idx] into assignment_info.sa
 *  for each variable k that belong to the expression that is going to be used. This expression
 *  is passed into assignment_info->expr. If assignment_info->expr is NULL, all variables
 *  are assigned.
 */
static char *dump_assignments(void **elem, void *arg)
{
    char *varname = *(char**)elem;
    assignment_info_t *info = (assignment_info_t*)arg;
    
    string_arena_init(info->sa);
    if( (NULL == info->expr) || jdf_expr_depends_on_symbol(varname, info->expr) )
        string_arena_add_string(info->sa, "  %s = assignments[%d].value;\n", varname, info->idx);
    info->idx++;
    return string_arena_get_string(info->sa);
}

/**
 * dump_dataflow:
 *  Takes the pointer to a jdf_flow, and a pointer to either "IN" or "OUT",
 *  and print the name of the variable for this flow if it's a variable as IN or as OUT
 *  NULL otherwise (and the UTIL_DUMP_LIST_FIELD macro will jump above these elements)
 */
static char *dump_dataflow(void **elem, void *arg)
{
    jdf_dataflow_t *d = *(jdf_dataflow_t**)elem;
    jdf_dep_type_t target;
    jdf_dep_list_t *l;

    if( !strcmp(arg, "IN") )
        target = JDF_DEP_TYPE_IN;
    else
        target = JDF_DEP_TYPE_OUT;

    for(l = d->deps; l != NULL; l = l->next)
        if( l->dep->type == target )
            break;
    if( l == NULL )
        return NULL;
    return d->varname;
}

/**
 * dump_resinit:
 *  Takes the pointer to a char *, which is a parameter name of the _new function,
 *  and a string_arena, and writes in the string_arena the assignment of this
 *  parameter in the res structure to the parameter passed to the _new function.
 */
static char *dump_resinit(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    char *varname = *(char**)elem;
    string_arena_init(sa);

    string_arena_add_string(sa, "res->%s = %s;", varname, varname);

    return string_arena_get_string(sa);
}

/**
 * Parameters of the dump_stringlist_by_index function
 */
typedef struct stringlist_by_index_info {
    string_arena_t *sa;
    int             idx;
    const char *    format;
} stringlist_by_index_info_t;

/**
 * dump_stringlist_by_index:
 *  This is almost a hack: it dumps a list of anything by applying a
 *  varying index to a given format. The varying element is ignored.
 *  Only the arguments vary. The only link with the list is that
 *  this is dumped once per element of the list.
 *  Takes as second argument the parameters for the function.
 */
static char *dump_stringlist_by_index(void **_, void *arg)
{
    stringlist_by_index_info_t *info = (stringlist_by_index_info_t*)arg;
    string_arena_init(info->sa);
    string_arena_add_string(info->sa, info->format, info->idx);
    info->idx++;
    return string_arena_get_string(info->sa);
}

static int jdf_symbol_is_global(const jdf_global_entry_t *globals, const char *name)
{
    const jdf_global_entry_t *g;
    for(g = globals; NULL != g; g = g->next)
        if( !strcmp(g->name, name) )
            return 1;
    return 0;
}

static int jdf_symbol_is_standalone(const char *name, const jdf_global_entry_t *globals, const jdf_expr_t *e)
{
    if( JDF_OP_IS_CST(e->op) )
        return 1;
    else if ( JDF_OP_IS_VAR(e->op) )
        return ((!strcmp(e->jdf_var, name)) ||
                jdf_symbol_is_global(globals, e->jdf_var));
    else if ( JDF_OP_IS_UNARY(e->op) )
        return jdf_symbol_is_standalone(name, globals, e->jdf_ua);
    else if ( JDF_OP_IS_TERNARY(e->op) )
        return jdf_symbol_is_standalone(name, globals, e->jdf_tat) &&
            jdf_symbol_is_standalone(name, globals, e->jdf_ta1) &&
            jdf_symbol_is_standalone(name, globals, e->jdf_ta2);
    else 
        return jdf_symbol_is_standalone(name, globals, e->jdf_ba1) &&
            jdf_symbol_is_standalone(name, globals, e->jdf_ba2);
}

static int jdf_expr_depends_on_symbol(const char *name, const jdf_expr_t *e)
{
    if( JDF_OP_IS_CST(e->op) )
        return 0;
    else if ( JDF_OP_IS_VAR(e->op) )
        return !strcmp(e->jdf_var, name);
    else if ( JDF_OP_IS_UNARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_ua);
    else if ( JDF_OP_IS_TERNARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_tat) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ta1) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ta2);
    else 
        return jdf_expr_depends_on_symbol(name, e->jdf_ba1) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ba2);
}

/** Structure Generators **/

static void jdf_generate_header_file(const jdf_t* jdf)
{
    string_arena_t *sa1, *sa2;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    houtput("#ifndef _%s_h_\n"
            "#define _%s_h_\n",
            jdf_basename, jdf_basename);
    houtput("#include <DAGuE.h>\n\n");
    houtput("DAGuE_object_t *DAGuE_%s_new(%s, %s);\n", jdf_basename,
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname,
                                  dump_string, NULL, "", "DAGUE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name,
                                  dump_string, NULL, "",  "int ", ", ", ""));
    string_arena_free(sa1);
    string_arena_free(sa2);
    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);
}

static void jdf_generate_structure(const jdf_t *jdf)
{
    int nbfunctions, nbdata;
    string_arena_t *sa1, *sa2;

    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);
    JDF_COUNT_LIST_ENTRIES(jdf->data, jdf_data_entry_t, next, nbdata);

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("#include <DAGuE.h>\n"
            "#include \"%s.h\"\n\n"
            "#define DAGuE_%s_NB_FUNCTIONS %d\n"
            "#define DAGuE_%s_NB_DATA %d\n", 
            jdf_basename, 
            jdf_basename, nbfunctions, 
            jdf_basename, nbdata);
    coutput("typedef struct DAGuE_%s {\n", jdf_basename);
    coutput("  /** All DAGuE_object_t structures hold these two arrays **/\n"
            "  int                    nb_functions;\n"
            "  const DAGuE_t        **functions_array;\n"
            "  DAGuE_dependencies_t **dependencies_array;\n"
            "  /*** Here begins the %s-specific part ***/\n",
            jdf_basename);
    coutput("  /* The list of globals */\n"
            "%s",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->globals, next, name,
                                  dump_string, NULL, "", "  int ", ";\n", ";\n"));
    coutput("  /* The list of data */\n"
            "%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname,
                                  dump_string, NULL, "", "  DAGuE_ddesc_t *", ";\n", ";\n"));
    coutput("  /* If profiling is enabled, the keys for profiling */\n"
            "#  if defined(DAGuE_PROFILING)\n"
            "%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname,
                                  dump_string, NULL, "", "  int ", "_start_key;\n", "_start_key;\n"));
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname,
                                  dump_string, NULL, "", "  int ", "_end_key;\n", "_end_key;\n"));
    coutput("#  endif /* defined(DAGuE_PROFILING) */\n");
    coutput("} __DAGuE_%s_t;\n"
            "\n", jdf_basename);

    /* dump the global symbols macros*/
    coutput("/* The globals */\n%s\n",
            UTIL_DUMP_LIST_FIELD(sa1, jdf->globals, next, name,
                                 dump_globals, sa2, "", "#define ", "\n", "\n"));

    /* dump the data access macros */
    coutput("/* The data access macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_data, sa2, "", "#define ", "\n", "\n"));

    /* dump the functions predicates */
    coutput("/* Functions Predicates */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->functions, next,
                           dump_predicate, sa2, "", "#define ", "\n", "\n"));

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_expression( const jdf_t *jdf, const jdf_def_list_t *context,
                                     const jdf_expr_t *e, const char *name)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    expr_info_t info;
    assignment_info_t ai;

    info.sa = sa;
    info.prefix = "";
    ai.sa = sa3;
    ai.idx = 0;
    ai.expr = e;
    coutput("static inline int %s_fct(const DAGuE_object_t *__DAGuE_object_parent, const assignment_t *assignments)\n"
            "{\n"
            "  const DAGuE_%s_object_t *__DAGuE_object = (const DAGuE_%s_object_t*)__DAGuE_object_parent;\n"
            "%s\n"
            "  return %s;\n"
            "}\n", name, jdf_basename, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa2, context, next, name, 
                                 dump_assignments, &ai, "", "", "", ""),
            dump_expr((void**)&e, &info));
    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);

    coutput("static const expr_t %s = {\n"
            "  .op = EXPR_OP_INLINE,\n"
            "  .flags = 0x0,\n"
            "  .u_expr.inline_func = %s_fct\n"
            "};\n", name, name);
}

static void jdf_generate_predicate_expr( const jdf_t *jdf, const jdf_def_list_t *context,
                                         const char *fname, const char *name)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    string_arena_t *sa4 = string_arena_new(64);
    expr_info_t info;
    assignment_info_t ai;

    info.sa = sa;
    info.prefix = "";
    ai.sa = sa3;
    ai.idx = 0;
    ai.expr = NULL;
    coutput("static inline int %s_fct(const DAGuE_object_t *__DAGuE_object_parent, const assignment_t *assignments)\n"
            "{\n"
            "  const DAGuE_%s_object_t *__DAGuE_object = (const DAGuE_%s_object_t*)__DAGuE_object_parent;\n"
            "%s\n"
            "  return %s_pred%s;\n"
            "}\n", name, jdf_basename, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa2, context, next, name, 
                                 dump_assignments, &ai, "", "", "", ""),
            fname, 
            UTIL_DUMP_LIST_FIELD(sa4, context, next, name,
                                 dump_string, NULL, "(", "", ", ", ")"));
    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);
    string_arena_free(sa4);

    coutput("static const expr_t %s = {\n"
            "  .op = EXPR_OP_INLINE,\n"
            "  .flags = 0x0,\n"
            "  .u_expr.inline_func = %s_fct\n"
            "};\n", name, name);
}

static void jdf_generate_symbols( const jdf_t *jdf, const jdf_def_list_t *def, const char *prefix )
{
    const jdf_def_list_t *d;
    char *exprname;
    string_arena_t *sa = string_arena_new(64);

    for(d = def; d != NULL; d = d->next) {
        exprname = (char*)malloc(strlen(d->name) + strlen(prefix) + 16);
        string_arena_init(sa);
        string_arena_add_string(sa, "static const symbol_t %s%s = {", prefix, d->name);
        if( d->expr->op == JDF_RANGE ) {
            sprintf(exprname, "minexpr_of_%s%s", prefix, d->name);
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            jdf_generate_expression(jdf, def, d->expr->jdf_ba1, exprname);

            sprintf(exprname, "maxexpr_of_%s%s", prefix, d->name);
            string_arena_add_string(sa, ".max = &%s, ", exprname);
            jdf_generate_expression(jdf, def, d->expr->jdf_ba2, exprname);
        } else {
            sprintf(exprname, "expr_of_%s%s", prefix, d->name);
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            string_arena_add_string(sa, ".max = &%s, ", exprname);
            jdf_generate_expression(jdf, def, d->expr, exprname);
        }

        if( jdf_symbol_is_global(jdf->globals, d->name) ) {
            string_arena_add_string(sa, " .flags = DPLASMA_SYMBOL_IS_GLOBAL");
        } else if ( jdf_symbol_is_standalone(d->name, jdf->globals, d->expr) ) {
            string_arena_add_string(sa, " .flags = DPLASMA_SYMBOL_IS_STANDALONE");
        } else {
            string_arena_add_string(sa, " .flags = 0x0");
        }
        string_arena_add_string(sa, "};");
        coutput("%s\n\n", string_arena_get_string(sa));
        free(exprname);
    }

    string_arena_free(sa);
}

static void jdf_generate_dependency( const jdf_t *jdf, const char *datatype, 
                                     const jdf_call_t *call, const char *depname,
                                     const char *condname, const jdf_def_list_t *context )
{
    string_arena_t *sa = string_arena_new(64);
    jdf_expr_list_t *le;
    char *exprname;
    int i;
    char pre[8];

    string_arena_add_string(sa, 
                            "static const dep_t %s = {\n"
                            "  .cond = %s,\n"
                            "  .dplasma = NULL; /**< To be filled when all structures are done to link to function %s */\n"
                            "  .param = NULL;   /**< To be filled when all structures are done to link to param %s of %s */\n"
                            "  .type  = (void*)%s%s%s; /**< Change this for C-code */\n"
                            "  .call_params = {\n",
                            depname, condname,
                            call->func_or_mem,
                            call->var != NULL ? call->var : "(null)", call->func_or_mem,
                            datatype != NULL ? "\"" : "", 
                            datatype != NULL ? datatype : "NULL",
                            datatype != NULL ? "\"" : "");

    exprname = (char *)malloc(strlen(depname) + 32);
    pre[0] = '\0';
    for( i = 1, le = call->parameters; le != NULL; i++, le = le->next ) {
        sprintf(exprname, "expr_of_p%d_for_%s", i, depname);
        string_arena_add_string(sa, "%s    &%s", pre, exprname);
        jdf_generate_expression(jdf, context, le->expr, exprname);
        sprintf(pre, ",\n");
    }
    free(exprname);

    string_arena_add_string(sa, 
                            "\n"
                            "  }\n"
                            "};\n");
    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
}

static void jdf_generate_dataflow( const jdf_t *jdf, const jdf_def_list_t *context,
                                   jdf_dataflow_t *flow, const char *prefix, unsigned char mask )
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa_dep_in = string_arena_new(64);
    string_arena_t *sa_dep_out = string_arena_new(64);
    string_arena_t *psa;
    int alldeps_type;
    jdf_dep_list_t *dl;
    char *sym_type;
    char *access_type;
    int depid;
    char *depname;
    char *condname;
    char sep_in[3], sep_out[3];
    char *sep;

    string_arena_init(sa_dep_in);
    string_arena_init(sa_dep_out);
    
    depname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    condname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    alldeps_type = 0;
    sep_in[0] = '\0';
    sep_out[0] = '\0';

    for(depid = 1, dl = flow->deps; dl != NULL; depid++, dl = dl->next) {
        alldeps_type |= dl->dep->type;
        if( dl->dep->type == JDF_DEP_TYPE_IN ) {
            psa = sa_dep_in;
            sep = sep_in;
        }
        else if ( dl->dep->type == JDF_DEP_TYPE_OUT ) {
            psa = sa_dep_out;
            sep = sep_out;
        } else {
            jdf_fatal(dl->dep->lineno, "This dependency is neither a DEP_IN or a DEP_OUT?\n");
            exit(1);
        }

        if( dl->dep->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
            sprintf(depname, "%s%s_dep%d_atline_%d", prefix, flow->varname, depid, dl->dep->lineno);
            sprintf(condname, "NULL");
            jdf_generate_dependency(jdf, dl->dep->datatype, dl->dep->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");
        } else if( dl->dep->guard->guard_type == JDF_GUARD_BINARY ) {
            sprintf(depname, "%s%s_dep%d_atline_%d", prefix, flow->varname, depid, dl->dep->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            jdf_generate_expression(jdf, context, dl->dep->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->dep->datatype, dl->dep->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");
        } else if( dl->dep->guard->guard_type == JDF_GUARD_TERNARY ) {
            jdf_expr_t not;

            sprintf(depname, "%s%s_dep%d_iftrue_atline_%d", prefix, flow->varname, depid, dl->dep->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            jdf_generate_expression(jdf, context, dl->dep->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->dep->datatype, dl->dep->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");

            sprintf(depname, "%s%s_dep%d_iffalse_atline_%d", prefix, flow->varname, depid, dl->dep->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            not.op = JDF_NOT;
            not.jdf_ua = dl->dep->guard->guard;
            jdf_generate_expression(jdf, context, &not, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->dep->datatype, dl->dep->guard->callfalse, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
        }
    }
    free(depname);
    free(condname);

    sym_type = ( (alldeps_type == JDF_DEP_TYPE_IN) ? "SYM_IN" :
                 ((alldeps_type == JDF_DEP_TYPE_OUT) ? "SYM_OUT" : "SYM_INOUT") );

    access_type = ( (flow->access_type == JDF_VAR_TYPE_READ) ? "ACCESS_READ" :
                    ((flow->access_type == JDF_VAR_TYPE_WRITE) ? "ACCESS_WRITE" : "ACCESS_RW") ); 

    string_arena_add_string(sa, 
                            "static const param_t %s%s = {\n"
                            "  .name = \"%s\",\n"
                            "  .function = NULL,\n"
                            "  .sym_type = %s,\n"
                            "  .access_type = %s,\n"
                            "  .param_mask = 0x%x,\n"
                            "  .dep_in  = { %s },\n"
                            "  .dep_out = { %s }\n"
                            "};\n\n", 
                            prefix, flow->varname, 
                            flow->varname, 
                            sym_type, 
                            access_type,
                            mask,
                            string_arena_get_string(sa_dep_in),
                            string_arena_get_string(sa_dep_out));
    string_arena_free(sa_dep_in);
    string_arena_free(sa_dep_out);

    coutput("%s", string_arena_get_string(sa));
    string_arena_free(sa);
}

static void jdf_generate_one_function( const jdf_t *jdf, const jdf_function_entry_t *f, int dep_index )
{
    string_arena_t *sa, *sa2;
    int nbparameters;
    int nbdataflow;
    int ls, rs, i;
    jdf_dataflow_list_t *fl;
    char *prefix;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    JDF_COUNT_LIST_ENTRIES(f->parameters, jdf_name_list_t, next, nbparameters);
    JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_list_t, next, nbdataflow);
    
    /* Pretty printing */
    ls = strlen(f->fname)/2;
    rs = strlen(f->fname)-ls;
    if( ls > 40 ) ls = 40;
    if( rs > 40 ) rs = 40;
    coutput("/**********************************************************************************\n"
            " *%*s%s%*s*\n"
            " **********************************************************************************/\n\n",
            40-ls, " ", f->fname, 40-rs, " ");
    /* End of Pretty printing */
    
    string_arena_add_string(sa, 
                            "static const DAGuE_t %s_%s = {\n"
                            "  .name = \"%s\",\n"
                            "  .deps = %d,\n"
                            "  .flags = %s,\n"
                            "  .dependencies_mask = 0x0,\n"
                            "  .nb_locals = %d,\n"
                            "  .nb_params = %d,\n",
                            jdf_basename, f->fname,
                            f->fname,
                            dep_index,
                            (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "DAGuE_HIGH_PRIORITY_TASK" : "0x0",
                            nbparameters,
                            nbdataflow);

    prefix = (char*)malloc(strlen(f->fname) + strlen(jdf_basename) + 32);

    sprintf(prefix, "symb_%s_%s_", jdf_basename, f->fname);
    jdf_generate_symbols(jdf, f->definitions, prefix);
    sprintf(prefix, "&symb_%s_%s_", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .params = { %s },\n", 
                            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL,
                                                 "", prefix, ", ", ""));
    string_arena_add_string(sa, "  .locals = { %s },\n",
                            UTIL_DUMP_LIST_FIELD(sa2, f->definitions, next, name, dump_string, NULL,
                                                 "", prefix, ", ", ""));

    sprintf(prefix, "pred_of_%s_%s_as_expr", jdf_basename, f->fname);
    jdf_generate_predicate_expr(jdf, f->definitions, f->fname, prefix);
    string_arena_add_string(sa, "  .pred = &%s;\n", prefix);

    sprintf(prefix, "priority_of_%s_%s_as_expr", jdf_basename, f->fname);
    jdf_generate_expression(jdf, f->definitions, f->priority, prefix);
    string_arena_add_string(sa, "  .priority = &%s;\n", prefix);

    sprintf(prefix, "param_of_%s_%s_for_", jdf_basename, f->fname);
    for(i = 0, fl = f->dataflow; fl != NULL; fl = fl->next, i++) {
        jdf_generate_dataflow(jdf, f->definitions, fl->flow, prefix, 1<<i);
    }
    sprintf(prefix, "&param_of_%s_%s_for_", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .in = { %s },\n",
                            UTIL_DUMP_LIST_FIELD(sa2, f->dataflow, next, flow, dump_dataflow, "IN",
                                                 "", prefix, ", ", ""));
    string_arena_add_string(sa, "  .out = { %s },\n",
                            UTIL_DUMP_LIST_FIELD(sa2, f->dataflow, next, flow, dump_dataflow, "OUT",
                                                 "", prefix, ", ", ""));

    sprintf(prefix, "hook_of_%s_%s", jdf_basename, f->fname);
    jdf_code_hook(jdf, f, prefix);
    string_arena_add_string(sa, "  .hook = %s;\n", prefix);

    sprintf(prefix, "release_deps_of_%s_%s", jdf_basename, f->fname);
    jdf_code_release_deps(jdf, f, prefix);
    string_arena_add_string(sa, "  .release_deps = %s;\n", prefix);

    free(prefix);

    string_arena_add_string(sa, "};\n");

    coutput("%s\n\n", string_arena_get_string(sa));

    string_arena_free(sa2);
    string_arena_free(sa);
}

static void jdf_generate_functions_statics( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    string_arena_t *sa;
    int i;

    sa = string_arena_new(64);
    string_arena_add_string(sa, "static const DAGuE_t *%s_functions[] = {\n",
                            jdf_basename);
    for(i = 0, f = jdf->functions; NULL != f; f = f->next, i++) {
        jdf_generate_one_function(jdf, f, i);
        string_arena_add_string(sa, "  &%s_%s%s\n", 
                                jdf_basename, f->fname, f->next != NULL ? "," : "");
    }
    string_arena_add_string(sa, "};\n\n");
    coutput("%s", string_arena_get_string(sa));
    string_arena_free(sa);
}

static void jdf_generate_initiator_body( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    int fli, depid;
    jdf_dataflow_list_t *fl;
    jdf_dep_list_t *dl;

    for(f = jdf->functions; f != NULL; f = f->next) {
        for(fl = f->dataflow; fl != NULL; fl = fl->next) {
            for(depid = 1, dl = fl->flow->deps; dl != NULL; depid++, dl = dl->next) {
                if( (dl->dep->guard->guard_type == JDF_GUARD_UNCONDITIONAL) ||
                    (dl->dep->guard->guard_type == JDF_GUARD_BINARY) ) {
                    coutput("  param_of_%s_%s_for_%s_dep%d_atline_%d.dplasma = &%s_%s;\n", 
                            jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                            jdf_basename, dl->dep->guard->calltrue->func_or_mem);
                    if( dl->dep->guard->calltrue->var != NULL ) {
                        coutput("  param_of_%s_%s_for_%s_dep%d_atline_%d.param = &param_of_%s_%s_for_%s;\n", 
                                jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                                jdf_basename, dl->dep->guard->calltrue->func_or_mem, dl->dep->guard->calltrue->var);
                    } 
                } else {
                    /* dl->dep->guard->guard_type == JDF_GUARD_TERNARY */
                    coutput("  param_of_%s_%s_for_%s_dep%d_iftrue_atline_%d.dplasma = &%s_%s;\n", 
                            jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                            jdf_basename, dl->dep->guard->calltrue->func_or_mem);
                    if( dl->dep->guard->calltrue->var != NULL ) {
                        coutput("  param_of_%s_%s_for_%s_dep%d_iftrue_atline_%d.param = &param_of_%s_%s_for_%s;\n", 
                                jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                                jdf_basename, dl->dep->guard->calltrue->func_or_mem, dl->dep->guard->calltrue->var);
                    } 
                    coutput("  param_of_%s_%s_for_%s_dep%d_iffalse_atline_%d.dplasma = &%s_%s;\n", 
                            jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                            jdf_basename, dl->dep->guard->callfalse->func_or_mem);
                    if( dl->dep->guard->callfalse->var != NULL ) {
                        coutput("  param_of_%s_%s_for_%s_dep%d_iffalse_atline_%d.param = &param_of_%s_%s_for_%s;\n", 
                                jdf_basename, f->fname, fl->flow->varname, depid, dl->dep->lineno,
                                jdf_basename, dl->dep->guard->callfalse->func_or_mem, dl->dep->guard->callfalse->var);
                    }
                }
            }
        }
    }
}

static void jdf_generate_constructor( const jdf_t* jdf )
{
    string_arena_t *sa1,*sa2;
    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("static int DAGuE_%s_initialized = 0;\n"
            "\n"
            "static void DAGuE_%s_init(void)\n"
            "{\n",
            jdf_basename, jdf_basename);
    
    jdf_generate_initiator_body( jdf );

    coutput("}\n"
            "\n"
            "DAGuE_object_t *DAGuE_%s_new(%s, %s)\n{\n", jdf_basename,
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname,
                                  dump_string, NULL, "", "DAGuE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name,
                                  dump_string, NULL, "",  "int ", ", ", ""));

    coutput("  DAGuE_%s_object_t *res = (DAGuE_%s_object_t *)calloc(sizeof(1, DAGuE_%s_object_t);\n",
            jdf_basename, jdf_basename, jdf_basename);
    coutput("  if( 0 == DAGuE_%s_initialized ) {\n"
            "    DAGuE_%s_init();\n"
            "    DAGuE_%s_initialized = 1;\n"
            "  }\n",
            jdf_basename, jdf_basename, jdf_basename);
    coutput("  res->nb_functions    = DAGuE_cholesky_NB_FUNCTIONS;\n", jdf_basename);
    coutput("  res->functions_array = (DAGuE_t**)malloc(DAGuE_%s_NB_FUNCTIONS * sizeof(DAGuE_t*));\n",
            jdf_basename);
    coutput("  res->dependencies_array = (DAGuE_dependencies_t **)\n"
            "             calloc(DAGuE_%s_NB_FUNCTIONS, sizeof(DAGuE_dependencies_t *));\n",
            jdf_basename);
    coutput("  memcpy(res->functions_array, %s_functions, DAGuE_%s_NB_FUNCTIONS * sizeof(DAGuE_t*));\n",
            jdf_basename, jdf_basename);
    coutput("%s", UTIL_DUMP_LIST_FIELD(sa1, jdf->data, next, dname,
                                       dump_resinit, sa2, "", "  ", "\n", "\n"));
    coutput("%s", UTIL_DUMP_LIST_FIELD(sa1, jdf->globals, next, name,
                                       dump_resinit, sa2, "", "  ", "\n", "\n"));
    coutput("  return (DAGuE_object_t*)res;\n"
            "}\n\n");

    string_arena_free(sa1);
    string_arena_free(sa2);
}

/** Code Generators */

static void jdf_code_hook(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{

}

static void jdf_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{

}

/** Main Function */

int jdf2c(const char *output_c, const char *output_h, const char *_jdf_basename, const jdf_t *jdf)
{
    int ret = 0;

    jdf_basename = _jdf_basename;
    cfile = NULL;
    hfile = NULL;

    cfile = fopen(output_c, "w");
    if( cfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", output_c, strerror(errno));
        ret = -1;
        goto err;
    }

    hfile = fopen(output_h, "w");
    if( hfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", output_h, strerror(errno));
        ret = -1;
        goto err;
    }

    cfile_lineno = 1;
    hfile_lineno = 1;
    
    jdf_generate_header_file(jdf);
    jdf_generate_structure(jdf);
    jdf_generate_functions_statics(jdf);

    /**
     * Generate the externally visible function.
     */
    jdf_generate_constructor(jdf);
 err:
    if( NULL != cfile ) 
        fclose(cfile);

    if( NULL != hfile )
        fclose(hfile);

    return ret;
}
