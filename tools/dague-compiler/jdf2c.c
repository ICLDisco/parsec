/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"
#include "jdf2c.h"
#include "dague/profiling.h"

extern const char *yyfilename;

static FILE *cfile;
static int   cfile_lineno;
static FILE *hfile;
static int   hfile_lineno;
static const char *jdf_basename;
static const char *jdf_cfilename;

/** Optional declarations of local functions */
static int jdf_expr_depends_on_symbol(const char *varname, const jdf_expr_t *expr);
static void jdf_generate_code_hooks(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_data_lookup(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_iterate_successors_or_predecessors(const jdf_t *jdf, const jdf_function_entry_t *f,
                                                                 const char *prefix, jdf_dep_flags_t flow_type);

static int jdf_property_get_int( const jdf_def_list_t* properties, const char* prop_name, int ret_if_not_found );

/** A coutput and houtput functions to write in the .h and .c files, counting the number of lines */

static inline int nblines(const char *p)
{
    int r = 0;
    for(; *p != '\0'; p++)
        if( *p == '\n' )
            r++;
    return r;
}

/**
 * This function is not thread-safe, not reentrant, and not pure. As such it
 * cannot be used twice on the same call to any oter function (including
 * printf's and friends). However, as a side effect, when it is called with
 * the same value for n, it is safe to be used in any of the previously
 * mentioned scenarios.
 */
static char *indent(int n)
{
    static char *istr    = NULL;
    static int   istrlen = 0;
    int i;

    if( n * 2 + 1 > istrlen ) {
        istrlen = n * 2 + 1;
        istr = (char*)realloc(istr, istrlen);
    }

    for(i = 0; i < n * 2; i++)
        istr[i] = ' ';
    istr[i] = '\0';
    return istr;
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

/**
 * Generate a semi-persistent string (kept in a circular buffer that will be reused after
 * dague_name_placeholders_max uses), representing the naming scheme behind the code
 * generator, aka. __dague_<JDF NAME>_<FUNC NAME>_%<others>.
 */
static char** dague_name_placeholders = NULL;
static int dague_name_placeholders_index = 0;
static const int dague_name_placeholders_max = 64;

static char*
dague_get_name(const jdf_t *jdf, const jdf_function_entry_t *f, char* fmt, ...)
{
    char* tmp = NULL; (void)jdf;
    va_list others;
    int rc;

    if( NULL == dague_name_placeholders ) {
        dague_name_placeholders = (char**)calloc(dague_name_placeholders_max, sizeof(char*));
    }
    if( NULL != dague_name_placeholders[dague_name_placeholders_index] ) {
        free(dague_name_placeholders[dague_name_placeholders_index]);
        dague_name_placeholders[dague_name_placeholders_index] = NULL;
    }
    rc = asprintf(&tmp, "__dague_%s_%s_%s", jdf_basename, f->fname, fmt);
    if( 0 > rc )
        return NULL;
    va_start(others, fmt);
    rc = vasprintf(&dague_name_placeholders[dague_name_placeholders_index],
                   tmp, others);
    va_end(others);
    free(tmp);
    if( 0 > rc )
        return NULL;
    tmp = dague_name_placeholders[dague_name_placeholders_index];
    dague_name_placeholders_index = (dague_name_placeholders_index + 1) % dague_name_placeholders_max;
    return tmp;
}

/**
 * Returns true if the function has any valid data output (not control).
 * Otherwise, returns false.
 */
static inline int function_has_data_output( const jdf_function_entry_t *f )
{
    jdf_dataflow_t* flow;
    jdf_dep_t *dep;

    for( flow = f->dataflow; flow != NULL; flow = flow->next) {
        if( !(flow->flow_flags & JDF_FLOW_TYPE_CTL) ) {
            for(dep = flow->deps; dep != NULL; dep = dep->next)
                if( JDF_DEP_FLOW_OUT & dep->dep_flags ) {
                    return 1;
                }
        }
    }
    return 0;
}

/** Utils: dumper functions for UTIL_DUMP_LIST_FIELD and UTIL_DUMP_LIST calls **/

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
    (void)_;
    return (char*)*elt;
}

/**
 * dump_globals:
 *   Dump a global symbol like #define ABC (__dague_handle->ABC)
 */
static char* dump_globals(void** elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;

    string_arena_init(sa);
    if( NULL != global->data )
        return NULL;
    string_arena_add_string(sa, "%s (__dague_handle->super.%s)",
                            global->name, global->name );
    return string_arena_get_string(sa);
}

/**
 * dump_data:
 *   Dump a global symbol like
 *     #define ABC(A0, A1) (__dague_handle->ABC->data_of(__dague_handle->ABC, A0, A1))
 */
static char* dump_data(void** elem, void *arg)
{
    jdf_data_entry_t* data = (jdf_data_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    int i;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s(%s%d", data->dname, data->dname, 0 );
    for( i = 1; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ",%s%d", data->dname, i );
    }
    string_arena_add_string(sa, ")  (((dague_ddesc_t*)__dague_handle->super.%s)->data_of((dague_ddesc_t*)__dague_handle->super.%s",
                            data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", (%s%d)", data->dname, i );
    }
    string_arena_add_string(sa, "))\n" );
    return string_arena_get_string(sa);
}

/**
 * dump_expr:
 *   dumps the jdf_expr* pointed to by elem into arg->sa, prefixing each
 *   non-global variable with arg->prefix
 */
char * dump_expr(void **elem, void *arg)
{
    expr_info_t* expr_info = (expr_info_t*)arg;
    expr_info_t li, ri;
    jdf_expr_t *e = (jdf_expr_t*)elem;
    string_arena_t *sa = expr_info->sa;
    string_arena_t *la, *ra;
    char *vc, *dot;

    string_arena_init(sa);

    la = string_arena_new(64);
    ra = string_arena_new(64);

    li.sa = la;
    li.prefix = expr_info->prefix;
    li.suffix = expr_info->suffix;
    li.assignments = expr_info->assignments;

    ri.sa = ra;
    ri.prefix = expr_info->prefix;
    ri.suffix = expr_info->suffix;
    ri.assignments = expr_info->assignments;

    switch( e->op ) {
    case JDF_VAR: {
        jdf_global_entry_t* item = current_jdf.globals;
        vc = strdup( e->jdf_var );
        dot = strchr(vc, '.');
        if( NULL != dot )
            *dot = '\0';
        /* Do not prefix if the variable is global */
        while( item != NULL ) {
            if( !strcmp(item->name, vc) ) {
                string_arena_add_string(sa, "%s", e->jdf_var);
                break;
            }
            item = item->next;
        }
        free(vc);
        if( NULL == item ) {
            string_arena_add_string(sa, "%s%s%s", expr_info->prefix, e->jdf_var, expr_info->suffix);
        }
        break;
    }
    case JDF_EQUAL:
        string_arena_add_string(sa, "(%s == %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_NOTEQUAL:
        string_arena_add_string(sa, "(%s != %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_AND:
        string_arena_add_string(sa, "(%s && %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_OR:
        string_arena_add_string(sa, "(%s || %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_XOR:
        string_arena_add_string(sa, "(%s ^ %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_LESS:
        string_arena_add_string(sa, "(%s < %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_LEQ:
        string_arena_add_string(sa, "(%s <= %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MORE:
        string_arena_add_string(sa, "(%s > %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MEQ:
        string_arena_add_string(sa, "(%s >= %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_NOT:
        string_arena_add_string(sa, "!%s",
                                dump_expr((void**)e->jdf_ua, &li));
        break;
    case JDF_PLUS:
        string_arena_add_string(sa, "(%s + %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MINUS:
        string_arena_add_string(sa, "(%s - %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_TIMES:
        string_arena_add_string(sa, "(%s * %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_DIV:
        string_arena_add_string(sa, "(%s / %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MODULO:
        string_arena_add_string(sa, "(%s %% %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_SHL:
        string_arena_add_string(sa, "(%s << %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_SHR:
        string_arena_add_string(sa, "(%s >> %s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_RANGE:
        break;
    case JDF_TERNARY: {
        expr_info_t ti;
        string_arena_t *ta;
        ta = string_arena_new(8);
        ti.sa = ta;
        ti.prefix = expr_info->prefix;
        ti.suffix = expr_info->suffix;
        ti.assignments = expr_info->assignments;

        string_arena_add_string(sa, "(%s ? %s : %s)",
                                dump_expr((void**)e->jdf_tat, &ti),
                                dump_expr((void**)e->jdf_ta1, &li),
                                dump_expr((void**)e->jdf_ta2, &ri) );

        string_arena_free(ta);
        break;
    }
    case JDF_CST:
        string_arena_add_string(sa, "%d", e->jdf_cst);
        break;
    case JDF_STRING:
        string_arena_add_string(sa, "%s", e->jdf_var);
        break;
    case JDF_C_CODE:
        if(  NULL == e->jdf_c_code.fname ) {
            string_arena_add_string(sa, "\n#error Expression %s has not been generated\n",
                                    e->jdf_c_code.code);
        } else {
            string_arena_add_string(sa, "%s(__dague_handle, %s)",
                                    e->jdf_c_code.fname, expr_info->assignments);
        }
        break;
    default:
        string_arena_add_string(sa, "DontKnow: %d", (int)e->op);
        break;
    }
    string_arena_free(la);
    string_arena_free(ra);

    return string_arena_get_string(sa);
}

/**
 * Dump a predicate like
 *  #define F_pred(k, n, m) (__dague_handle->ABC->rank == __dague_handle->ABC->rank_of(__dague_handle->ABC, k, n, m))
 */
static char* dump_predicate(void** elem, void *arg)
{
    jdf_function_entry_t *f = (jdf_function_entry_t *)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    expr_info_t expr_info;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s_pred(%s) ",
                            f->fname,
                            UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name,
                                                 dump_string, NULL,
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
    expr_info.suffix = "";
    expr_info.assignments = "assignments";
    string_arena_add_string(sa, "(((dague_ddesc_t*)(__dague_handle->super.%s))->myrank == ((dague_ddesc_t*)(__dague_handle->super.%s))->rank_of((dague_ddesc_t*)__dague_handle->super.%s, %s))",
                            f->predicate->func_or_mem, f->predicate->func_or_mem, f->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, f->predicate->parameters, next,
                                           dump_expr, &expr_info,
                                           "", "", ", ", ""));

    string_arena_free(sa2);
    string_arena_free(sa3);
    return string_arena_get_string(sa);
}

/**
 * Parameters of the dump_local_assignments function
 */
typedef struct assignment_info {
    string_arena_t *sa;
    const char *holder;
    const jdf_expr_t *expr;
} assignment_info_t;

/**
 * dump_local_assignments:
 * Takes the pointer to the name of a parameter, a pointer to a assignment_info, and prints
 * "const int %var% = <assignment_info.holder>%var%.value into assignment_info.sa.
 * If a local variable is not used by the expression then it is not generated. A special
 * corner case for inline_c code is handled by forcing the generation of all locals.
 * This function is usually used in order to generate the locals for the expression
 * associated with a particular task.
 * If assignment_info->expr is NULL, all variables are assigned.
 */
static char* dump_local_assignments( void** elem, void* arg )
{
    jdf_def_list_t *def = (jdf_def_list_t*)elem;
    assignment_info_t *info = (assignment_info_t*)arg;

    if( (NULL == info->expr) || jdf_expr_depends_on_symbol(def->name, info->expr) ) {
        string_arena_init(info->sa);
        string_arena_add_string(info->sa, "const int %s = %s%s.value;", def->name, info->holder, def->name);
        return string_arena_get_string(info->sa);
    }
    return NULL;
}

/**
 * dump_dataflow:
 *  Takes the pointer to a jdf_flow, and a pointer to either "IN" or "OUT",
 *  and print the name of the variable for this flow if it's a variable as IN or as OUT
 *  NULL otherwise (and the UTIL_DUMP_LIST_FIELD macro will jump above these elements)
 */
static char *dump_dataflow(void **elem, void *arg)
{
    jdf_dataflow_t *d = (jdf_dataflow_t*)elem;
    jdf_dep_flags_t target;
    jdf_dep_t *l;

    if( !strcmp(arg, "IN") )
        target = JDF_DEP_FLOW_IN;
    else
        target = JDF_DEP_FLOW_OUT;

    for(l = d->deps; l != NULL; l = l->next)
        if( l->dep_flags & target )
            break;
    if( l == NULL )
        return NULL;
    return d->varname;
}

/**
 * dump_data_declaration:
 *  Takes the pointer to a flow and produces the code
 *  needed for declaring the corresponding arena chunck and
 *  repo entry.
 * @return: the generated code or NULL if the flow is a control flow.
 */
static char *dump_data_declaration(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_dataflow_t *f = (jdf_dataflow_t*)elem;

    if(f->flow_flags & JDF_FLOW_TYPE_CTL) {
        return NULL;
    }

    string_arena_init(sa);
    string_arena_add_string(sa,
                            "  dague_arena_chunk_t *g%s;\n"
                            "  data_repo_entry_t *e%s = NULL; /**< repo entries can be NULL for memory data */\n",
                            f->varname,
                            f->varname);
    return string_arena_get_string(sa);
}

/**
 * dump_data_initialization_from_data_array:
 *  Takes the pointer to a flow *f, let say that f->varname == "A",
 *  this produces a string like
 *  dague_data_copy_t *gA = this_task->data[id].data_in;\n
 *  void *A = DAGUE_DATA_COPY_GET_PTR(gA); (void)A;\n
 */
static char *dump_data_initialization_from_data_array(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t *)arg;
    jdf_dataflow_t *f = (jdf_dataflow_t*)elem;
    char *varname = f->varname;

    if(f->flow_flags & JDF_FLOW_TYPE_CTL) {
        return NULL;
    }

    string_arena_init(sa);

    string_arena_add_string(sa,
                            "  dague_data_copy_t *g%s = this_task->data.%s.data_in;\n",
                            varname, f->varname);
    if( !(f->flow_flags & JDF_FLOW_TYPE_READ) ) {  /* if only write then we can locally have NULL */
        string_arena_add_string(sa,
                                "  void *%s = (NULL != g%s) ? DAGUE_DATA_COPY_GET_PTR(g%s) : NULL; (void)%s;\n",
                                varname, varname, varname, varname);
    } else {
        string_arena_add_string(sa,
                                "  void *%s = DAGUE_DATA_COPY_GET_PTR(g%s); (void)%s;\n",
                                varname, varname, varname);
    }
    return string_arena_get_string(sa);
}

/**
 * dump_dataflow_varname:
 *  Takes the pointer to a flow *f, and print the varname
 */
static char *dump_dataflow_varname(void **elem, void *_)
{
    (void)_;
    jdf_dataflow_t *f = (jdf_dataflow_t *)elem;
    if( f->flow_flags & JDF_FLOW_TYPE_CTL ) return NULL;
    return f->varname;
}

static void get_unique_rgb_color(float ratio, unsigned char *r, unsigned char *g, unsigned char *b)
{
    float h = ratio, s = 0.8, v = 0.8, r1, g1, b1;
    float var_h = ( fabs(h - 1.0f) < 1e-6 ) ? 0.0f : h * 6.0f;
    int var_i = (int)floor(var_h);
    float var_1 = (v * ( 1.0f - s ));
    float var_2 = (v * ( 1.0f - s * ( var_h - var_i )));
    float var_3 = (v * ( 1.0f - s * ( 1.0f - ( var_h - var_i ) ) ));

    if      ( var_i == 0 ) { r1 = v     ; g1 = var_3 ; b1 = var_1; }
    else if ( var_i == 1 ) { r1 = var_2 ; g1 = v     ; b1 = var_1; }
    else if ( var_i == 2 ) { r1 = var_1 ; g1 = v     ; b1 = var_3; }
    else if ( var_i == 3 ) { r1 = var_1 ; g1 = var_2 ; b1 = v;     }
    else if ( var_i == 4 ) { r1 = var_3 ; g1 = var_1 ; b1 = v;     }
    else                   { r1 = v     ; g1 = var_1 ; b1 = var_2; }

    *r = (unsigned char)(r1 * 255);
    *g = (unsigned char)(g1 * 255);
    *b = (unsigned char)(b1 * 255);
}

/**
 * Parameters of the dump_profiling_init function
 */
typedef struct profiling_init_info {
    string_arena_t *sa;
    int idx;
    int maxidx;
} profiling_init_info_t;

/**
 * dump_profiling_init:
 *  Takes the pointer to the name of a function, an index in
 *  a pointer to a dump_profiling_init, and prints
 *    dague_profiling_add_dictionary_keyword( "elem", attribute[idx], &elem_key_start, &elem_key_end);
 *  into profiling_init_info.sa
 */
static char *dump_profiling_init(void **elem, void *arg)
{
    profiling_init_info_t *info = (profiling_init_info_t*)arg;
    jdf_function_entry_t* f = (jdf_function_entry_t*)elem;
    char *fname = f->fname;
    unsigned char R, G, B;
    int nb_locals;
    string_arena_t *profiling_convertor_params;

    if( !jdf_property_get_int(f->properties, "profile", 1) ) {
        return NULL;
    }

    string_arena_init(info->sa);

    get_unique_rgb_color((float)info->idx / (float)info->maxidx, &R, &G, &B);
    info->idx++;

    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_def_list_t, next, nb_locals);
    profiling_convertor_params = string_arena_new(64);
    UTIL_DUMP_LIST_FIELD(profiling_convertor_params, f->locals, next, name, dump_string, NULL,
                         DAGUE_PROFILE_DDESC_INFO_CONVERTOR, ";", "{int32_t}", "{int32_t}");

    string_arena_add_string(info->sa,
                            "dague_profiling_add_dictionary_keyword(\"%s\", \"fill:%02X%02X%02X\",\n"
                            "                                       sizeof(dague_profile_ddesc_info_t)+%d*sizeof(assignment_t),\n"
                            "                                       \"%s\",\n"
                            "                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * %s_%s.function_id /* %s start key */],\n"
                            "                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * %s_%s.function_id /* %s end key */]);\n",
                            fname, R, G, B,
                            nb_locals,
                            string_arena_get_string(profiling_convertor_params),
                            jdf_basename, fname, fname,
                            jdf_basename, fname, fname);

    string_arena_free(profiling_convertor_params);

    return string_arena_get_string(info->sa);
}

/**
 * dump_startup_call:
 *  Takes a pointer to a function and print the call to add the startup tasks
 *  if the function can be a startup one (i.e. there is a set of values in the
 *  execution space that make all input come directly from the data instead of
 *  other tasks).
 */
static char *dump_startup_call(void **elem, void *arg)
{
    const jdf_function_entry_t *f = (const jdf_function_entry_t *)elem;
    string_arena_t* sa = (string_arena_t*)arg;

    if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
        string_arena_init(sa);
        string_arena_add_string(sa,
                                "_%s_startup_tasks(context, (__dague_%s_internal_handle_t*)__dague_handle, pready_list);",
                                f->fname, jdf_basename);
        return string_arena_get_string(sa);
    }
    return NULL;
}

/**
 * dump_globals_init:
 *  Takes a pointer to a global variables and generate the code used to initialize
 *  the global variable during *_New. If the variable has a default value or is
 *  marked as hidden the output code will not be generated.
 */
static char *dump_globals_init(void **elem, void *arg)
{
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_expr_t *hidden = jdf_find_property( global->properties, "hidden", NULL );
    jdf_expr_t *prop = jdf_find_property( global->properties, "default", NULL );

    string_arena_init(sa);
    /* We might have a default value */
    if( NULL == prop ) prop = global->expression;

    /* No default value ? */
    if( NULL == prop ) {
        if( NULL == hidden ) /* Hidden variable or not ? */
            string_arena_add_string(sa, "__dague_handle->super.%s = %s;", global->name, global->name);
    } else {
        expr_info_t info;
        info.sa = string_arena_new(8);
        info.prefix = "";
        info.suffix = "";
        info.assignments = "assignments";

        string_arena_add_string(sa, "__dague_handle->super.%s = %s = %s;",
                                global->name, global->name,
                                dump_expr((void**)prop, &info));
        string_arena_free(info.sa);
    }

    return string_arena_get_string(sa);
}

/**
 * dump_data_name
 *  Takes a pointer to a global variables and generate the code used to initialize
 *  the global variable during *_New. If the variable has a default value or is
 *  marked as hidden the output code will not be generated.
 */
static char *dump_data_name(void **elem, void *arg)
{
     jdf_global_entry_t* global = (jdf_global_entry_t*)elem;
     string_arena_t *sa = (string_arena_t*)arg;

     if( NULL == global->data ) return NULL;

     string_arena_init(sa);
     string_arena_add_string(sa, "%s", global->name);
     return string_arena_get_string(sa);
}

/**
 * Print global variables that have (or not) a certain property.
 */
typedef struct typed_globals_info {
    string_arena_t *sa;
    char* include;
    char* exclude;
} typed_globals_info_t;

static char* dump_typed_globals(void **elem, void *arg)
{
    typed_globals_info_t* prop = (typed_globals_info_t*)arg;
    string_arena_t *sa = prop->sa;
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;
    jdf_expr_t* type_str = jdf_find_property( global->properties, "type", NULL );
    jdf_expr_t *size_str = jdf_find_property( global->properties, "size", NULL );
    jdf_expr_t *prop_str;
    expr_info_t info;

    if( NULL != prop->include ) {
        prop_str = jdf_find_property( global->properties, prop->include, NULL );
        if( NULL == prop_str ) return NULL;
    } else if( NULL != prop->exclude ) {
        prop_str = jdf_find_property( global->properties, prop->exclude, NULL );
        if( NULL != prop_str ) return NULL;
    }
    string_arena_init(sa);

    info.sa = string_arena_new(8);
    info.prefix = "";
    info.suffix = "";
    info.assignments = "assignments";

    if( NULL == global->data ) {
        string_arena_add_string(sa, "%s %s",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), global->name);
    } else {
        string_arena_add_string(sa, "%s %s /* data %s */",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), global->name, global->name);
    }
    if( NULL != size_str ) {
        houtput("#define %s_%s_SIZE %s\n",
                jdf_basename, global->name, dump_expr((void**)size_str, &info));
    }
    string_arena_free(info.sa);

    return string_arena_get_string(sa);
}

/**
 * dump_hidden_globals_init:
 *  Takes a pointer to a global variables and generate the code used to initialize
 *  the global variable during *_New. If the variable is not marked as hidden
 *  the output code will not be generated.
 */
static char *dump_hidden_globals_init(void **elem, void *arg)
{
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_expr_t *hidden   = jdf_find_property( global->properties, "hidden", NULL );
    jdf_expr_t* type_str = jdf_find_property( global->properties, "type",   NULL );
    expr_info_t info;

    string_arena_init(sa);

    /* The property is hidden */
    if (NULL != hidden) {
        jdf_expr_t *prop = jdf_find_property( global->properties, "default", NULL );

        /* We might have a default value */
        if( NULL == prop ) prop = global->expression;

        /* No default value ? */
        if( NULL == prop ) return NULL;

        info.sa = string_arena_new(8);
        info.prefix = "";
        info.suffix = "";
        info.assignments = "assignments";

        string_arena_add_string(sa, "%s %s;",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)),
                                global->name);
        string_arena_free(info.sa);

        return string_arena_get_string(sa);
    }
    return NULL;
}

static char *dump_data_repository_constructor(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_function_entry_t *f = (jdf_function_entry_t *)elem;

    string_arena_init(sa);

    int nbdata = 0;
    JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_t, next, nbdata);
    string_arena_add_string(sa,
                            "  %s_nblocal_tasks = %s_%s_internal_init(__dague_handle);\n"
                            "  __dague_handle->repositories[%d] = data_repo_create_nothreadsafe(  /* %s */\n"
                            "          %s_nblocal_tasks, %d);\n",
                            f->fname, jdf_basename, f->fname,
                            f->function_id, f->fname,
                            f->fname, nbdata);

    return string_arena_get_string(sa);
}

/** Utils: observers for the jdf **/

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
    else if( JDF_OP_IS_BINARY(e->op) )
        return jdf_symbol_is_standalone(name, globals, e->jdf_ba1) &&
            jdf_symbol_is_standalone(name, globals, e->jdf_ba2);
    else
        return 0;
}

static int jdf_expr_depends_on_symbol(const char *name, const jdf_expr_t *e)
{
    if( JDF_OP_IS_CST(e->op) || JDF_OP_IS_STRING(e->op) )
        return 0;
    else if ( JDF_OP_IS_VAR(e->op) )
        return !strcmp(e->jdf_var, name);
    else if ( JDF_OP_IS_UNARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_ua);
    else if ( JDF_OP_IS_TERNARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_tat) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ta1) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ta2);
    else if( JDF_OP_IS_BINARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_ba1) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ba2);
    return 1;  /* by default assume the affirmative */
}

/**
 * Helpers to manipulate object properties (i.e. typed attributed assciated with
 * different concepts such as tasks, flows, dependencies and functions). The
 * existence of some of these properties will change the way the code is
 * generated, and allows for increased flexibility on driving the code
 * generator.
 *
 * When new properties are added, the list of existing properties must be updated.
 */
jdf_expr_t* jdf_find_property( const jdf_def_list_t* properties, const char* property_name, jdf_def_list_t** property )
{
    const jdf_def_list_t* current = properties;

    if( NULL != property ) *property = NULL;
    while( NULL != current ) {
        if( !strcmp(current->name, property_name) ) {
            if( NULL != property ) *property = (jdf_def_list_t*)current;
            return current->expr;
        }
        current = current->next;
    }
    return NULL;
}

/**
 * Accessors to get typed properties (int and string).
 */
static int jdf_property_get_int( const jdf_def_list_t* properties,
                                 const char* prop_name,
                                 int ret_if_not_found )
{
    jdf_def_list_t* property;
    jdf_expr_t* expr = jdf_find_property(properties, prop_name, &property);

    if( NULL != expr ) {
        if( JDF_CST == expr->op )
            return expr->jdf_cst;
        printf("Warning: property %s defined at line %d only support ON/OFF\n",
               prop_name, JDF_OBJECT_LINENO(property));
    }
    return ret_if_not_found;  /* ON by default */
}

static const char*
jdf_property_get_string( const jdf_def_list_t* properties,
                         const char* prop_name,
                         const char* ret_if_not_found )
{
    jdf_def_list_t* property;
    jdf_expr_t* expr = jdf_find_property(properties, prop_name, &property);

    if( NULL != expr ) {
        if( JDF_OP_IS_VAR(expr->op) )
            return expr->jdf_var;
        printf("Warning: property %s defined at line %d only support ON/OFF\n",
               prop_name, JDF_OBJECT_LINENO(property));
    }
    return ret_if_not_found;  /* the expected default */
}

static int jdf_dataflow_type(const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    int type = 0;
    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
            continue;  /* skip WRITE-only flows even if they have empty input deps (for datatype) */
        }
        type |= dl->dep_flags;
    }
    return type;
}

/**
 * Find the output flow corresponding to a particular input flow. This function
 * returns the flow and not a particular dependency.
 */
static const jdf_dataflow_t*
jdf_data_output_flow(const jdf_t *jdf, const char *fname, const char *varname)
{
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;

    for(f = jdf->functions; f != NULL; f = f->next) {
        if( strcmp(f->fname, fname) ) continue;
        for( fl = f->dataflow; fl != NULL; fl = fl->next) {
            if( jdf_dataflow_type(fl) & JDF_DEP_FLOW_OUT ) {
                if( !strcmp(fl->varname, varname) ) {
                    return fl;
                }
            }
        }
    }
    return NULL;
}

/**
 * Finds the index of the input flow that corresponds to a particular output flow.
 */
static int
jdf_data_input_index(const jdf_t *jdf, const char *fname, const char *varname)
{
    int i;
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;

    i = 0;
    for(f = jdf->functions; f != NULL; f = f->next) {
        if( strcmp(f->fname, fname) ) continue;
        for( fl = f->dataflow; fl != NULL; fl = fl->next) {
            if( jdf_dataflow_type(fl) & JDF_DEP_FLOW_IN ) {
                if( !strcmp(fl->varname, varname) ) {
                    return i;
                }
                i++;
            }
        }
        return -1;
    }
    return -2;
}

static void jdf_coutput_prettycomment(char marker, const char *format, ...)
{
    int ls, rs, i, length, vs;
    va_list ap, ap2;
    char *v;

    vs = 80;
    v = (char *)malloc(vs);

    va_start(ap, format);
    /* va_list might have pointer to internal state and using
       it twice is a bad idea.  So make a copy for the second
       use.  Copy order taken from Autoconf docs. */
#if defined(HAVE_VA_COPY)
    va_copy(ap2, ap);
#elif defined(HAVE_UNDERSCORE_VA_COPY)
    __va_copy(ap2, ap);
#else
    memcpy (&ap2, &ap, sizeof(va_list));
#endif

    length = vsnprintf(v, vs, format, ap);
    if( length >= vs ) {  /* realloc */
        vs = length + 1;
        v = (char*)realloc( v, vs );
        length = vsnprintf(v, vs, format, ap2);
    }

#if defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY) */
    va_end(ap);

    /* Pretty printing */
    if( length > 80 ) {
        ls = rs = 1;
    } else {
        ls = (80 - length) / 2;
        rs = 80 - length - ls;
    }
    coutput("/*");
    for(i = 0; i < 5; i++)
        coutput("%c", marker);
    coutput("%s%s", indent(ls/2), v);  /* indent drop two spaces */
    coutput("%s", indent(rs/2));       /* dont merge these two calls. Read the comment on the indent function */
    for(i = 0; i < 5; i++)
        coutput("%c", marker);
    coutput("*/\n\n");
    free(v);
}

/** Structure Generators **/

/**
 * Generate typedef for the tasks struct based on the locals and flows
 * of each task familly. Right now these tasks typedefs are not used
 * anywhere, instead we always use the generic task structure.
 */
static inline char* jdf_generate_task_typedef(void **elt, void* arg)
{
    const jdf_function_entry_t* f = (jdf_function_entry_t*)elt;
    string_arena_t *sa = (string_arena_t*)arg, *sa_locals, *sa_data;
    int nb_locals = 0, nb_flows = 0;

    sa_locals = string_arena_new(64);
    sa_data = string_arena_new(64);
    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_def_list_t, next, nb_locals);
    UTIL_DUMP_LIST_FIELD(sa_locals, f->locals, next, name, dump_string, NULL,
                         "", "  assignment_t ", ";\n", ";\n");

    JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_t, next, nb_flows);
    UTIL_DUMP_LIST_FIELD(sa_data, f->dataflow, next, varname, dump_string, NULL,
                         "", "  dague_data_pair_t ", ";\n", ";\n");

    string_arena_init(sa);
    /* Prepare the structure for the named assignments */
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "%s"
                            "  assignment_t unused[MAX_LOCAL_COUNT-%d];\n"
                            "} %s;\n\n",
                            dague_get_name(NULL, f, "assignment_s"),
                            string_arena_get_string(sa_locals),
                            nb_locals,
                            dague_get_name(NULL, f, "assignment_t"));
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "%s"
                            "  dague_data_pair_t unused[MAX_LOCAL_COUNT-%d];\n"
                            "} %s;\n\n",
                            dague_get_name(NULL, f, "data_s"),
                            string_arena_get_string(sa_data),
                            nb_flows,
                            dague_get_name(NULL, f, "data_t"));
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "    DAGUE_MINIMAL_EXECUTION_CONTEXT\n"
                            "#if defined(DAGUE_PROF_TRACE)\n"
                            "    dague_profile_ddesc_info_t prof_info;\n"
                            "#endif /* defined(DAGUE_PROF_TRACE) */\n"
                            "    struct __dague_%s_%s_assignment_s locals;\n"
                            "#if defined(PINS_ENABLE)\n"
                            "    int                        creator_core;\n"
                            "    int                        victim_core;\n"
                            "#endif /* defined(PINS_ENABLE) */\n"
                            "#if defined(DAGUE_SIM)\n"
                            "    int                        sim_exec_date;\n"
                            "#endif\n"
                            "    struct __dague_%s_%s_data_s data;\n"
                            "} %s;\n\n",
                            dague_get_name(NULL, f, "task_s"),
                            jdf_basename, f->fname,
                            jdf_basename, f->fname,
                            dague_get_name(NULL, f, "task_t"));
    string_arena_free(sa_locals);
    string_arena_free(sa_data);
    return string_arena_get_string(sa);
}

static void jdf_generate_header_file(const jdf_t* jdf)
{
    string_arena_t *sa1, *sa2, *sa3;
    struct jdf_name_list* g;
    int datatype_index = 0;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);

    houtput("#ifndef _%s_h_\n"
            "#define _%s_h_\n",
            jdf_basename, jdf_basename);
    houtput("#include \"dague.h\"\n"
            "#include \"dague/constants.h\"\n"
            "#include \"dague/data_distribution.h\"\n"
            "#include \"dague/data_internal.h\"\n"
            "#include \"dague/debug.h\"\n"
            "#include \"dague/ayudame.h\"\n"
            "#include \"dague/devices/device.h\"\n"
            "#include <assert.h>\n\n");
    houtput("BEGIN_C_DECLS\n\n");

    for( g = jdf->datatypes; NULL != g; g = g->next ) {
        houtput("#define DAGUE_%s_%s_ARENA    %d\n",
                jdf_basename, g->name, datatype_index);
        datatype_index++;
    }
    houtput("#define DAGUE_%s_ARENA_INDEX_MIN %d\n", jdf_basename, datatype_index);
    houtput("\ntypedef struct dague_%s_handle {\n", jdf_basename);
    houtput("  dague_handle_t super;\n");
    {
        typed_globals_info_t prop = { sa2, NULL, NULL };
        houtput("  /* The list of globals */\n"
                "%s",
                UTIL_DUMP_LIST( sa1, jdf->globals, next, dump_typed_globals, &prop,
                                "", "  ", ";\n", ";\n"));
    }
    houtput("  /* The array of datatypes (%s and co.) */\n"
            "  dague_arena_t** arenas;\n"
            "  int arenas_size;\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->datatypes, next, name,
                                  dump_string, NULL, "", "", ",", ""));

    houtput("} dague_%s_handle_t;\n\n", jdf_basename);

    {
        typed_globals_info_t prop = { sa3, NULL, "hidden" };
        houtput("extern dague_%s_handle_t *dague_%s_new(%s);\n\n", jdf_basename, jdf_basename,
                UTIL_DUMP_LIST( sa2, jdf->globals, next, dump_typed_globals, &prop,
                                "", "", ", ", ""));
    }

    /* TODO: Enable this once the task typedef are used in the code generation. */
    houtput("%s", UTIL_DUMP_LIST(sa1, jdf->functions, next, jdf_generate_task_typedef, sa3,
                                 "", "", "\n", "\n"));
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    houtput("END_C_DECLS\n\n");
    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);
}

static void jdf_generate_structure(const jdf_t *jdf)
{
    int nbfunctions, nbdata, need_profile = 0;
    string_arena_t *sa1, *sa2;
    jdf_function_entry_t* f;
    jdf_name_list_t *pl;

    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);
    JDF_COUNT_LIST_ENTRIES(jdf->data, jdf_data_entry_t, next, nbdata);

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("#include <dague.h>\n"
            "#include \"dague/debug.h\"\n"
            "#include \"dague/scheduling.h\"\n"
            "#include \"dague/mca/pins/pins.h\"\n"
            "#include \"dague/remote_dep.h\"\n"
            "#include \"dague/datarepo.h\"\n"
            "#include \"dague/data.h\"\n"
            "#include \"dague/mempool.h\"\n"
            "#include \"dague/utils/output.h\"\n"
            "#include \"%s.h\"\n\n"
            "#define DAGUE_%s_NB_FUNCTIONS %d\n"
            "#define DAGUE_%s_NB_DATA %d\n"
            "#if defined(DAGUE_PROF_GRAPHER)\n"
            "#include \"dague/dague_prof_grapher.h\"\n"
            "#endif  /* defined(DAGUE_PROF_GRAPHER) */\n"
            "#include <alloca.h>\n",
            jdf_basename,
            jdf_basename, nbfunctions,
            jdf_basename, nbdata);
    coutput("typedef struct __dague_%s_internal_handle {\n", jdf_basename);
    coutput(" dague_%s_handle_t super;\n",
            jdf_basename);

    coutput("  /* The ranges to compute the hash key */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        for(pl = f->parameters; pl != NULL; pl = pl->next) {
            coutput("  int %s_%s_range;\n", f->fname, pl->name);
        }
    }

    coutput("  /* The list of data repositories ");
    for(f = jdf->functions; NULL != f; f = f->next) {
        if( 0 != function_has_data_output(f) ) {
            coutput(" %s ", f->fname);
        }
    }
    coutput("*/\n");
    if(nbfunctions != 0 ) {
        coutput("  data_repo_t* repositories[%d];\n", nbfunctions );
    }

    coutput("} __dague_%s_internal_handle_t;\n"
            "\n", jdf_basename);

    for( f = jdf->functions; NULL != f; f = f->next ) {
        /* If the profile property is ON then enable the profiling array */
        need_profile += jdf_property_get_int(f->properties, "profile", 1);
    }
    if( need_profile )
        coutput("#if defined(DAGUE_PROF_TRACE)\n"
                "static int %s_profiling_array[2*DAGUE_%s_NB_FUNCTIONS] = {-1};\n"
                "#endif  /* defined(DAGUE_PROF_TRACE) */\n",
                jdf_basename, jdf_basename);

    UTIL_DUMP_LIST(sa1, jdf->globals, next,
                   dump_globals, sa2, "", "#define ", "\n", "\n");
    if( 1 < strlen(string_arena_get_string(sa1)) ) {
        coutput("/* Globals */\n%s\n", string_arena_get_string(sa1));
    }

    coutput("/* Data Access Macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_data, sa2, "", "#define ", "\n", "\n"));

    coutput("/* Functions Predicates */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->functions, next,
                           dump_predicate, sa2, "", "#define ", "\n", "\n"));

    coutput("/* Data Repositories */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            coutput("#define %s_repo (__dague_handle->repositories[%d])\n",
                    f->fname, f->function_id);
        }
    }

    coutput("/* Dependency Tracking Allocation Macro */\n"
            "#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \\\n"
            "do {                                                                                         \\\n"
            "  int _vmin = (vMIN);                                                                        \\\n"
            "  int _vmax = (vMAX);                                                                        \\\n"
            "  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \\\n"
            "                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \\\n"
            "  DEBUG3((\"Allocate %%d spaces for loop %%s (min %%d max %%d) 0x%%p last_dep 0x%%p\\n\",    \\\n"
            "           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP)));    \\\n"
            "  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \\\n"
            "  (DEPS)->symbol = (vSYMBOL);                                                                \\\n"
            "  (DEPS)->min = _vmin;                                                                       \\\n"
            "  (DEPS)->max = _vmax;                                                                       \\\n"
            "  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \\\n"
            "} while (0)\n\n"
            "static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };\n\n"
            "static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };\n\n");

    coutput("/* Release dependencies output macro */\n"
            "#if DAGUE_DEBUG_VERBOSE != 0\n"
            "#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\\\n"
            "  do { \\\n"
            "    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\\\n"
            "    DEBUG((\"thread %%d VP %%d explore deps from %%s:%%s to %%s:%%s (from rank %%d to %%d) base ptr %%p\\n\",\\\n"
            "           (NULL != (EU) ? (EU)->th_id : -1), (NULL != (EU) ? (EU)->virtual_process->vp_id : -1),\\\n"
            "           DEPO, dague_snprintf_execution_context(tmp1, 128, (TASKO)),\\\n"
            "           DEPI, dague_snprintf_execution_context(tmp2, 128, (TASKI)), (RSRC), (RDST), (DATA)));\\\n"
            "  } while(0)\n"
            "#define ACQUIRE_FLOW(TASKI, DEPI, FUNO, DEPO, LOCALS, PTR)\\\n"
            "  do { \\\n"
            "    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\\\n"
            "    DEBUG((\"task %%s acquires flow %%s from %%s %%s data ptr %%p\\n\",\\\n"
            "           dague_snprintf_execution_context(tmp1, 128, (TASKI)), (DEPI),\\\n"
            "           (DEPO), dague_snprintf_assignments(tmp2, 128, (FUNO), (LOCALS)), (PTR)));\\\n"
            "  } while(0)\n"
            "#else\n"
            "#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\n"
            "#define ACQUIRE_FLOW(TASKI, DEPI, TASKO, DEPO, LOCALS, PTR)\n"
            "#endif\n");
    string_arena_free(sa1);
    string_arena_free(sa2);
}

/**
 * Generates a highly optimized function for an expression. If the expression is
 * constant or an inlined code no local variables are generated. If the
 * expresion is a constant then it is directly returned, if the expression is an
 * inlined function then a call to the original accessor is generated
 * instead. This function only generates the code without generating the
 * corresponding expr_t.
 */
static void
jdf_generate_function_without_expression(const jdf_t *jdf,
                                         const jdf_function_entry_t* f,
                                         const jdf_expr_t *e,
                                         const char *name,
                                         const char *suffix,
                                         const char* rettype)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info;
    assignment_info_t ai;

    (void)jdf;

    assert(e->op != JDF_RANGE);

    coutput("static inline %s %s%s(const __dague_%s_internal_handle_t *__dague_handle, const %s *locals)\n"
            "{\n",
            rettype, name, suffix, jdf_basename, dague_get_name(jdf, f, "assignment_t"));
    if( !(JDF_OP_IS_C_CODE(e->op) || (JDF_OP_IS_CST(e->op))) ) {
        ai.sa = sa;
        ai.holder = "locals->";
        ai.expr = e;

        coutput("%s\n",
                UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_assignments, &ai,
                               "", "  ", "\n", "\n"));
    }
    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "locals";
    coutput("  (void)__dague_handle; (void)locals;\n"
            "  return %s;\n"
            "}\n",
            dump_expr((void**)e, &info));
    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_expression( const jdf_t *jdf, const jdf_function_entry_t *f,
                                     jdf_expr_t *e, const char *name)
{
    JDF_OBJECT_ONAME(e) = strdup(name);

    if( e->op == JDF_RANGE ) {
        char *subf = (char*)malloc(strlen(JDF_OBJECT_ONAME(e)) + 64);
        sprintf(subf, "rangemin_of_%s", JDF_OBJECT_ONAME(e));
        jdf_generate_expression(jdf, f, e->jdf_ta1, subf);
        sprintf(subf, "rangemax_of_%s", JDF_OBJECT_ONAME(e));
        jdf_generate_expression(jdf, f, e->jdf_ta2, subf);

        if( e->jdf_ta3->op == JDF_CST ) {
            coutput("static const expr_t %s = {\n"
                    "  .op = EXPR_OP_RANGE_CST_INCREMENT,\n"
                    "  .u_expr.range = {\n"
                    "    .op1 = &rangemin_of_%s,\n"
                    "    .op2 = &rangemax_of_%s,\n"
                    "    .increment.cst = %d\n"
                    "  }\n"
                    "};\n",
                    JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), e->jdf_ta3->jdf_cst );
        } else {
            sprintf(subf, "rangeincrement_of_%s", JDF_OBJECT_ONAME(e));
            jdf_generate_expression(jdf, f, e->jdf_ta3, subf);
            coutput("static const expr_t %s = {\n"
                    "  .op = EXPR_OP_RANGE_EXPR_INCREMENT,\n"
                    "  .u_expr.range = {\n"
                    "    .op1 = &rangemin_of_%s,\n"
                    "    .op2 = &rangemax_of_%s,\n"
                    "    .increment.expr = &rangeincrement_of_%s\n"
                    "  }\n"
                    "};\n",
                    JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e));
        }
    } else {
        jdf_generate_function_without_expression(jdf, f, e, JDF_OBJECT_ONAME(e), "_fct", "int");

        coutput("static const expr_t %s = {\n"
                "  .op = EXPR_OP_INLINE,\n"
                "  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)%s_fct }\n"
                "};\n", JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e));
    }
}

static void jdf_generate_affinity( const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    string_arena_t *sa4 = string_arena_new(64);
    string_arena_t *sa5 = string_arena_new(64);
    assignment_info_t ai;
    expr_info_t info;
    const jdf_call_t *data_affinity = f->predicate;

    (void)jdf;

    if( data_affinity->var != NULL ) {
        fprintf(stderr, "Internal Error: data affinity must reference a data, not a complete call (%s:%d).\n",
                data_affinity->super.filename, data_affinity->super.lineno);
        assert( NULL == data_affinity->var );
    }

    coutput("static inline int %s(%s *this_task,\n"
            "                     dague_data_ref_t *ref)\n"
            "{\n"
            "    const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)this_task->dague_handle;\n",
            name, dague_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename);

    info.sa = sa5;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("%s\n"
            "  /* Silent Warnings: should look into predicate to know what variables are usefull */\n"
            "  (void)__dague_handle;\n"
            "%s\n"
            "  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.%s;\n"
            "  /* Compute data key */\n"
            "  ref->key = ref->ddesc->data_key(ref->ddesc, %s);\n"
            "  return 1;\n"
            "}\n",
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"),
            UTIL_DUMP_LIST_FIELD(sa3, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";\n", ";"),
            data_affinity->func_or_mem,
            UTIL_DUMP_LIST(sa4, data_affinity->parameters, next,
                           dump_expr, (void*)&info,
                           "", "", ", ", ""));

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    string_arena_free(sa4);
    string_arena_free(sa5);
}

static void jdf_generate_initfinal_data_for_call(const jdf_call_t *call,
                                                 string_arena_t* sa,
                                                 int il)
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info;

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    assert( call->var == NULL );
    string_arena_add_string(sa, "%s    __d = (dague_ddesc_t*)__dague_handle->super.%s;\n"
                            "%s    refs[__flow_nb].ddesc = __d;\n",
                            indent(il), call->func_or_mem,
                            indent(il));
    string_arena_add_string(sa, "%s    refs[__flow_nb].key = __d->data_key(__d, %s);\n"
                            "%s    __flow_nb++;\n",
                            indent(il), UTIL_DUMP_LIST(sa1, call->parameters, next,
                                                       dump_expr, (void*)&info,
                                                       "", "", ", ", ""),
                            indent(il));

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static int jdf_generate_initfinal_data_for_dep(const jdf_dep_t *dep,
                                               string_arena_t* sa)
{
    string_arena_t *sa1 = string_arena_new(64);
    expr_info_t info;
    int ret = 0;

    info.sa = sa1;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    switch( dep->guard->guard_type ) {
    case JDF_GUARD_UNCONDITIONAL:
        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dep) ) {
            /* TODO */
            string_arena_add_string(sa,
                                    "    __d = (dague_ddesc_t*)NULL;\n"
                                    "    refs[__flow_nb].ddesc = NULL;\n"
                                    "    refs[__flow_nb].key = 0xffffffff;\n"
                                    "    __flow_nb++;\n");
                                    break;
        }
        if( dep->guard->calltrue->var == NULL ) {
            /* Unconditional direct memory reference: this is a init or final data */
            jdf_generate_initfinal_data_for_call(dep->guard->calltrue, sa, 0);
            ret++;
        }
        break;

    case JDF_GUARD_BINARY:
        if( dep->guard->calltrue->var == NULL ) {
            /* Conditional direct memory reference: this is a init or final data if the guard is true */
            string_arena_add_string(sa, "    if( %s ) {\n", dump_expr((void**)dep->guard->guard, &info));
            jdf_generate_initfinal_data_for_call(dep->guard->calltrue, sa, 2);
            string_arena_add_string(sa, "    }\n");
            ret++;
        }
        break;

    case JDF_GUARD_TERNARY:
        if( dep->guard->calltrue->var == NULL ||
            dep->guard->callfalse->var == NULL ) {
            /* Ternary direct memory reference: different cases. */
            if( dep->guard->calltrue->var == NULL &&
                dep->guard->callfalse->var == NULL ) {
                /* Direct memory reference in both cases, use if else to find which case */
                string_arena_add_string(sa, "    if( %s ) {\n", dump_expr((void**)dep->guard->guard, &info));
                jdf_generate_initfinal_data_for_call(dep->guard->calltrue, sa, 2);
                ret++;
                string_arena_add_string(sa, "    } else {\n");
                jdf_generate_initfinal_data_for_call(dep->guard->callfalse, sa, 2);
                ret++;
                string_arena_add_string(sa, "    }\n");
            } else {
                /* Direct memory reference only if guard true xor false */
                if( dep->guard->calltrue->var == NULL ) {
                    string_arena_add_string(sa, "    if( %s ) {\n", dump_expr((void**)dep->guard->guard, &info));
                    jdf_generate_initfinal_data_for_call(dep->guard->calltrue, sa, 2);
                    string_arena_add_string(sa, "    }\n");
                    ret++;
                } else {
                    string_arena_add_string(sa, "    if( !(%s) ) {\n", dump_expr((void**)dep->guard->guard, &info));
                    jdf_generate_initfinal_data_for_call(dep->guard->callfalse, sa, 2);
                    string_arena_add_string(sa, "    }\n");
                    ret++;
                }
            }
        }
        break;
    }

    return ret;
}

static int jdf_generate_initfinal_data_for_flow(char type,
                                                const jdf_dataflow_t *flow,
                                                string_arena_t* sa)
{
    jdf_dep_t *dl;
    int nbdep = 0;

    string_arena_add_string(sa, "    /** Flow of %s */\n", flow->varname);
    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( dl->dep_flags & type ) {
            nbdep += jdf_generate_initfinal_data_for_dep(dl, sa);
        }
    }
    return nbdep;
}

static int jdf_generate_initfinal_data( const jdf_t *jdf,
                                        char type,
                                        const jdf_function_entry_t *f,
                                        const char *name)
{
    string_arena_t *sa  = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    assignment_info_t ai;
    jdf_dataflow_t *fl;
    int nbdep = 0;

    for(fl = f->dataflow; fl != NULL; fl = fl->next) {
        nbdep += jdf_generate_initfinal_data_for_flow(type, fl, sa);
    }

    (void)jdf;

    if( 0 != nbdep ) {
        coutput("static inline int %s(%s *this_task,\n"
                "                     dague_data_ref_t *refs)\n"
                "{\n"
                "    const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)this_task->dague_handle;\n"
                "    dague_ddesc_t *__d;\n"
                "    int __flow_nb = 0;\n",
                name, dague_get_name(jdf, f, "task_t"),
                jdf_basename, jdf_basename);


        ai.sa = sa2;
        ai.holder = "this_task->locals.";
        ai.expr = NULL;
        coutput("%s\n"
                "    /* Silent Warnings: should look into predicate to know what variables are usefull */\n"
                "    (void)__dague_handle;\n"
                "%s\n",
                UTIL_DUMP_LIST(sa1, f->locals, next,
                               dump_local_assignments, &ai, "", "  ", "\n", "\n"),
                UTIL_DUMP_LIST_FIELD(sa3, f->locals, next, name,
                                     dump_string, NULL, "", "  (void)", ";\n", ";"));

        coutput("  %s\n"
                "    return __flow_nb;\n"
                "}\n\n",
                string_arena_get_string(sa));
    }

    string_arena_free(sa);
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    return nbdep;
}


static void jdf_generate_symbols( const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix )
{
    jdf_def_list_t *d;
    char *exprname;
    int id;
    string_arena_t *sa = string_arena_new(64);
    int rc;

    for(id = 0, d = f->locals; d != NULL; id++, d = d->next) {
        rc = asprintf( &JDF_OBJECT_ONAME(d), "%s%s", prefix, d->name );
        assert( rc != -1 );

        exprname = (char*)malloc(strlen(JDF_OBJECT_ONAME(d)) + 16);
        string_arena_init(sa);

        string_arena_add_string(sa, "static const symbol_t %s = { .name = \"%s\", .context_index = %d, ", JDF_OBJECT_ONAME(d), d->name, id);

        if( d->expr->op == JDF_RANGE ) {
            sprintf(exprname, "minexpr_of_%s", JDF_OBJECT_ONAME(d));
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            jdf_generate_expression(jdf, f, d->expr->jdf_ta1, exprname);

            sprintf(exprname, "maxexpr_of_%s", JDF_OBJECT_ONAME(d));
            string_arena_add_string(sa, ".max = &%s, ", exprname);
            jdf_generate_expression(jdf, f, d->expr->jdf_ta2, exprname);

            if( d->expr->jdf_ta3->op == JDF_CST ) {
                string_arena_add_string(sa, ".cst_inc = %d, .expr_inc = NULL, ", d->expr->jdf_ta3->jdf_cst);
            } else {
                sprintf(exprname, "incexpr_of_%s", JDF_OBJECT_ONAME(d));
                string_arena_add_string(sa, ".cst_inc = 0, .expr_inc = &%s, ", exprname);
                jdf_generate_expression(jdf, f, d->expr->jdf_ta3, exprname);
            }
        } else {
            sprintf(exprname, "expr_of_%s", JDF_OBJECT_ONAME(d));
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            string_arena_add_string(sa, ".max = &%s, .cst_inc = 0, .expr_inc = NULL, ", exprname);
            jdf_generate_expression(jdf, f, d->expr, exprname);
        }

        if( jdf_symbol_is_global(jdf->globals, d->name) ) {
            string_arena_add_string(sa, " .flags = DAGUE_SYMBOL_IS_GLOBAL");
        } else if ( jdf_symbol_is_standalone(d->name, jdf->globals, d->expr) ) {
            string_arena_add_string(sa, " .flags = DAGUE_SYMBOL_IS_STANDALONE");
        } else {
            string_arena_add_string(sa, " .flags = 0x0");
        }
        string_arena_add_string(sa, "};");
        coutput("%s\n\n", string_arena_get_string(sa));
        free(exprname);
    }

    string_arena_free(sa);
}

static void jdf_generate_ctl_gather_compute(const jdf_t *jdf, const jdf_function_entry_t* of,
                                            const char *tname, const char *fname,
                                            const jdf_expr_t *params)
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    expr_info_t info1, info2, info3;
    const jdf_expr_t *le;
    const jdf_function_entry_t *targetf;
    const jdf_name_list_t *pl;
    int i;
    assignment_info_t ai;

    for(targetf = jdf->functions; targetf != NULL; targetf = targetf->next) {
        if(!strcmp(tname, targetf->fname))
            break;
    }
    assert(targetf != NULL);

    coutput("static inline int %s_fct(const __dague_%s_internal_handle_t *__dague_handle, const %s *assignments)\n"
            "{\n"
            "  int   __nb_found = 0;\n"
            "  (void)__dague_handle;\n",
            fname, jdf_basename, dague_get_name(jdf, of, "assignment_t"));

    /* i = 0; */
    /* for(le = params, pl = f->parameters; NULL != le; pl = pl->next, le = le->next) { */
    /*     i++; */
    /* } */

    info1.sa = sa1;
    info1.prefix = "";
    info1.suffix = "";
    info1.assignments = "assignments";

    info2.sa = sa2;
    info2.prefix = "";
    info2.suffix = "";
    info2.assignments = "assignments";

    info3.sa = sa3;
    info3.prefix = "";
    info3.suffix = "";
    info3.assignments = "assignments";

    ai.sa = sa2;
    ai.holder = "assignments->";
    ai.expr = NULL;
    coutput( "%s",
             UTIL_DUMP_LIST(sa1, of->locals, next,
                            dump_local_assignments, &ai, "", "  ", "\n", "\n"));
    coutput( "%s\n",
             UTIL_DUMP_LIST_FIELD(sa2, of->locals, next, name, dump_string, NULL,
                                  "  ", "(void)", "; ", ";"));

    i = 0;
    for(pl = targetf->parameters, le = params; NULL != le; pl = pl->next, le = le->next) {
        if( le->op == JDF_RANGE ) {
            coutput("%s  {\n"
                    "%s    int %s_%s;\n"
                    "%s    for(%s_%s  = %s;\n"
                    "%s        %s_%s <= %s;\n"
                    "%s        %s_%s += %s) {\n",
                    indent(i),
                    indent(i), targetf->fname, pl->name,
                    indent(i), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta1, &info1 ),
                    indent(i), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta2, &info2 ),
                    indent(i), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta3, &info3 ));
            i+=2;
        }
    }
    coutput("%s  __nb_found++;\n", indent(i));
    i--;
    for(; i > -1; i--) {
        coutput("%s  }\n", indent(i));
    }
    coutput("  return __nb_found;\n"
            "}\n"
            "\n"
            "static const expr_t %s = {\n"
            "  .op = EXPR_OP_INLINE,\n"
            "  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)%s_fct }\n"
            "};\n\n", fname, fname);

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
}

static void jdf_generate_direct_data_function(const jdf_t *jdf, const char *mem,
                                              const jdf_expr_t *parameters,
                                              const jdf_function_entry_t *f,
                                              const char *function_name)
{
    assignment_info_t ai;
    expr_info_t info;
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    string_arena_t *sa4 = string_arena_new(64);
    string_arena_t *sa5 = string_arena_new(64);

    info.sa = sa5;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "assignments";

    ai.sa = sa2;
    ai.holder = "assignments->";
    ai.expr = NULL;

    UTIL_DUMP_LIST(sa4, parameters, next,
                   dump_expr, (void*)&info,
                   "", "", ", ", "");

    coutput("static dague_data_t *%s(const __dague_%s_internal_handle_t *__dague_handle, const %s *assignments)\n"
            "{\n"
            "  dague_ddesc_t *__ddesc;\n"
            "%s\n"
            "  /* Silent Warnings: should look into parameters to know what variables are usefull */\n"
            "%s\n"
            "  __ddesc = (dague_ddesc_t*)__dague_handle->super.%s;\n"
            "  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, %s) )\n"
            "    return __ddesc->data_of(__ddesc, %s);\n"
            "  return NULL;\n"
            "}\n"
            "\n",
            function_name, jdf_basename, dague_get_name(jdf, f, "assignment_t"),
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"),
            UTIL_DUMP_LIST_FIELD(sa3, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";\n", ";"),
            mem,
            string_arena_get_string(sa4),
            string_arena_get_string(sa4));

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    string_arena_free(sa4);
    string_arena_free(sa5);
    (void)jdf;
}

static int jdf_generate_dependency( const jdf_t *jdf, jdf_dataflow_t *flow, jdf_dep_t *dep,
                                    jdf_call_t *call, const char *depname,
                                    const char *condname, const jdf_function_entry_t* f )
{
    string_arena_t *sa = string_arena_new(64), *sa2 = string_arena_new(64);
    jdf_expr_t *le;
    int ret = 1, generate_stubs = 0;
    string_arena_t *tmp_fct_name;
    jdf_datatransfer_type_t* datatype = &dep->datatype;

    JDF_OBJECT_ONAME(call) = strdup(depname);

    if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
        for( le = call->parameters; le != NULL; le = le->next ) {
            if( le->op == JDF_RANGE ) {
                break;
            }
        }
        if( NULL != le ) {
            /* At least one range in input: must be a control gather */
            if( !(flow->flow_flags & JDF_FLOW_TYPE_CTL) ) {
                jdf_fatal(JDF_OBJECT_LINENO(dep), "This dependency features a range as input but is not a Control dependency\n");
                exit(1);
            }
            string_arena_add_string(sa2, "&ctl_gather_compute_for_dep_%s",
                                    JDF_OBJECT_ONAME(dep));
            /* skip the & at the beginning */
            jdf_generate_ctl_gather_compute(jdf, f, call->func_or_mem,
                                            string_arena_get_string(sa2)+1, call->parameters);
            ret = 0;
        } else {
            string_arena_add_string(sa2, "NULL");
        }
    }else {
        string_arena_add_string(sa2, "NULL");
    }

    string_arena_add_string(sa,
                            "static const dep_t %s = {\n"
                            "  .cond = %s,\n"
                            "  .ctl_gather_nb = %s,\n",
                            JDF_OBJECT_ONAME(call),
                            condname,
                            string_arena_get_string(sa2));

    if( NULL != call->var ) {
        jdf_function_entry_t* pf;
        for(pf = jdf->functions;
            strcmp(pf->fname, call->func_or_mem);
            pf = pf->next) /* nothing */;
        if( NULL == pf ) {
            fprintf(stderr, "Error: Can't identify the target function for the call at %s.jdf:%d: %s %s\n",
                   jdf_basename, call->super.lineno, call->var, call->func_or_mem);
            exit(-1);
        }
        string_arena_add_string(sa,
                                "  .function_id = %d, /* %s_%s */\n"
                                "  .direct_data = (direct_data_lookup_func_t)NULL,\n",
                                (NULL != pf ? pf->function_id : -1), jdf_basename, call->func_or_mem);
        string_arena_add_string(sa,
                                "  .flow = &flow_of_%s_%s_for_%s,\n",
                                jdf_basename, call->func_or_mem, call->var);
    } else {
        tmp_fct_name = string_arena_new(64);
        string_arena_add_string(tmp_fct_name, "%s_direct_access", JDF_OBJECT_ONAME(dep));
        jdf_generate_direct_data_function(jdf, call->func_or_mem, call->parameters, f,
                                          string_arena_get_string(tmp_fct_name));
        string_arena_add_string(sa,
                                "  .function_id = %d, /* %s_%s */\n"
                                "  .direct_data = (direct_data_lookup_func_t)&%s,\n",
                                -1, jdf_basename, call->func_or_mem,
                                string_arena_get_string(tmp_fct_name));
        string_arena_free(tmp_fct_name);
    }
    string_arena_add_string(sa,
                            "  .dep_index = %d,\n"
                            "  .dep_datatype_index = %d,\n",
                            dep->dep_index,
                            dep->dep_datatype_index);
    /**
     * Beware: There is a single datatype per dep_t, and several deps can reuse the same datatype
     *         as indicated by the dep_datatype_index field. Make sure we only create the datatype
     *         once.
     */
    if( NULL == JDF_OBJECT_ONAME(datatype) ) {
        string_arena_init(sa2);
        string_arena_add_string(sa2, "%s_datatype_%s%d", JDF_OBJECT_ONAME(flow),
                                (JDF_DEP_FLOW_IN & dep->dep_flags ? "in" : "out"),
                                dep->dep_datatype_index);
        JDF_OBJECT_ONAME(datatype) = strdup(string_arena_get_string(sa2));
        generate_stubs = (dep->dep_index == dep->dep_datatype_index);
    }

    /* Start with generating the type */
    if( (JDF_CST == datatype->type->op) || (JDF_VAR == datatype->type->op) || (JDF_STRING == datatype->type->op) ) {
        if( JDF_CST == datatype->type->op ) {
            string_arena_add_string(sa,
                                    "  .datatype = { .type   = { .cst = %d },\n",
                                    datatype->type->jdf_cst);
        } else {
            string_arena_add_string(sa,
                                    "  .datatype = { .type   = { .cst = DAGUE_%s_%s_ARENA },\n",
                                    jdf_basename, datatype->type->jdf_var);
        }
    } else {
        tmp_fct_name = string_arena_new(64);
        string_arena_add_string(tmp_fct_name, "%s_type_fct", JDF_OBJECT_ONAME(datatype));
        if( generate_stubs )
            jdf_generate_function_without_expression(jdf, f, datatype->type,
                                                     string_arena_get_string(tmp_fct_name), "", "int32_t");
        string_arena_add_string(sa,
                                "  .datatype = { .type   = { .fct = (expr_op_int32_inline_func_t)%s },\n",
                                string_arena_get_string(tmp_fct_name));
        string_arena_free(tmp_fct_name);
    }
    /* And the layout */
    if( datatype->type == datatype->layout ) {
        string_arena_add_string(sa,
                                "                .layout = { .fct = NULL },\n"
                                "                .count  = { .cst = 1 },\n"
                                "                .displ  = { .cst = 0 }\n");
    } else {
        if( (JDF_VAR == datatype->layout->op) || (JDF_STRING == datatype->layout->op) ) {
            string_arena_add_string(sa,
                                    "                .layout = { .cst = %s },\n",
                                    datatype->layout->jdf_var);
        } else {
            tmp_fct_name = string_arena_new(64);
            string_arena_add_string(tmp_fct_name, "%s_layout_fct", JDF_OBJECT_ONAME(datatype));
            if( generate_stubs )
                jdf_generate_function_without_expression(jdf, f, datatype->layout,
                                                         string_arena_get_string(tmp_fct_name), "", "dague_datatype_t");
            string_arena_add_string(sa,
                                    "                .layout = { .fct = %s },\n",
                                    string_arena_get_string(tmp_fct_name));
            string_arena_free(tmp_fct_name);
        }

        /* Now the count */
        if( JDF_CST == datatype->count->op ) {
            string_arena_add_string(sa,
                                    "                .count  = { .cst = %d },\n",
                                    datatype->count->jdf_cst);
        } else {
            tmp_fct_name = string_arena_new(64);
            string_arena_add_string(tmp_fct_name, "%s_cnt_fct", JDF_OBJECT_ONAME(datatype));
            if( generate_stubs )
                jdf_generate_function_without_expression(jdf, f, datatype->count,
                                                         string_arena_get_string(tmp_fct_name), "", "int64_t");
            string_arena_add_string(sa,
                                    "                .count  = { .fct = (expr_op_int64_inline_func_t)%s },\n",
                                    string_arena_get_string(tmp_fct_name));
            string_arena_free(tmp_fct_name);
        }

        /* And finally the displacement */
        if( JDF_CST == datatype->displ->op ) {
            string_arena_add_string(sa,
                                    "                .displ  = { .cst = %d }\n",
                                    datatype->displ->jdf_cst);
        } else {
            tmp_fct_name = string_arena_new(64);
            string_arena_add_string(tmp_fct_name, "%s_displ_fct", JDF_OBJECT_ONAME(datatype));
            if( generate_stubs )
                jdf_generate_function_without_expression(jdf, f, datatype->displ,
                                                         string_arena_get_string(tmp_fct_name), "", "int64_t");
            string_arena_add_string(sa,
                                    "                .displ  = { .fct = %s }\n",
                                    string_arena_get_string(tmp_fct_name));
            string_arena_free(tmp_fct_name);
        }
    }
    string_arena_add_string(sa,
                            "},\n"
                            "  .belongs_to = &%s,\n",
                            JDF_OBJECT_ONAME(flow));


    string_arena_add_string(sa,
                            "};\n");

    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
    string_arena_free(sa2);

    return ret;
}

static int jdf_generate_dataflow( const jdf_t *jdf, const jdf_function_entry_t* f,
                                  jdf_dataflow_t *flow, const char *prefix,
                                  int *has_control_gather )
{
    char *sym_type, *depname, *condname, *sep;
    int alldeps_type, depid, indepnorange = 1;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa_dep_in = string_arena_new(64);
    string_arena_t *sa_dep_out = string_arena_new(64);
    string_arena_t* flow_flags = string_arena_new(64);
    string_arena_t *psa;
    jdf_dep_t *dl;
    uint32_t flow_datatype_mask = 0;
    char sep_in[4], sep_out[4];  /* one char more to deal with '\n' special cases (Windows) */

    (void)jdf;

    /* The Object Name has already been pre-generated by the forward declaration procedure */
    assert( NULL != JDF_OBJECT_ONAME(flow) );

    string_arena_init(sa_dep_in);
    string_arena_init(sa_dep_out);

    depname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    condname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    sep_in[0] = '\0';
    sep_out[0] = '\0';

    for(depid = 1, dl = flow->deps; dl != NULL; depid++, dl = dl->next) {
        if( dl->dep_flags & JDF_DEP_FLOW_IN ) {
            psa = sa_dep_in;
            sep = sep_in;
        } else if ( dl->dep_flags & JDF_DEP_FLOW_OUT ) {
            psa = sa_dep_out;
            sep = sep_out;
            flow_datatype_mask |= (1U << dl->dep_datatype_index);
        } else {
            jdf_fatal(JDF_OBJECT_LINENO(dl), "This dependency is neither a DEP_IN or a DEP_OUT (flag 0x%x)\n");
            exit(1);
        }
        sprintf(depname, "%s_dep%d_atline_%d", JDF_OBJECT_ONAME(flow), depid, JDF_OBJECT_LINENO(dl));
        JDF_OBJECT_ONAME(dl) = strdup(depname);

        if( dl->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
            sprintf(condname, "NULL");
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
                continue;  /* skip type declaration for WRITE-only flows */
            }

            indepnorange = jdf_generate_dependency(jdf, flow, dl, dl->guard->calltrue,
                                                   JDF_OBJECT_ONAME(dl), condname, f) && indepnorange;
            string_arena_add_string(psa, "%s&%s", sep, JDF_OBJECT_ONAME(dl));
            sprintf(sep, ",\n ");
        } else if( dl->guard->guard_type == JDF_GUARD_BINARY ) {
            sprintf(condname, "expr_of_cond_for_%s", JDF_OBJECT_ONAME(dl));
            jdf_generate_expression(jdf, f, dl->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", JDF_OBJECT_ONAME(dl));
            indepnorange = jdf_generate_dependency(jdf, flow, dl, dl->guard->calltrue,
                                                   JDF_OBJECT_ONAME(dl), condname, f) && indepnorange;
            string_arena_add_string(psa, "%s&%s", sep, JDF_OBJECT_ONAME(dl));
            sprintf(sep, ",\n ");
        } else if( dl->guard->guard_type == JDF_GUARD_TERNARY ) {
            jdf_expr_t not;

            sprintf(depname, "%s_iftrue", JDF_OBJECT_ONAME(dl));
            sprintf(condname, "expr_of_cond_for_%s", depname);
            jdf_generate_expression(jdf, f, dl->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            indepnorange = jdf_generate_dependency(jdf, flow, dl, dl->guard->calltrue, depname, condname, f) && indepnorange;
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ",\n ");

            sprintf(depname, "%s_iffalse", JDF_OBJECT_ONAME(dl));
            sprintf(condname, "expr_of_cond_for_%s", depname);
            not.op = JDF_NOT;
            not.jdf_ua = dl->guard->guard;
            jdf_generate_expression(jdf, f, &not, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            indepnorange = jdf_generate_dependency(jdf, flow, dl, dl->guard->callfalse, depname, condname, f) && indepnorange;
            string_arena_add_string(psa, "%s&%s", sep, depname);
        }
    }
    free(depname);
    free(condname);

    alldeps_type = jdf_dataflow_type(flow);
    sym_type = ((alldeps_type & JDF_DEP_FLOW_IN) ?
                ((alldeps_type & JDF_DEP_FLOW_OUT) ? "SYM_INOUT" : "SYM_IN") : "SYM_OUT");

    string_arena_init(flow_flags);
    string_arena_add_string(flow_flags, "%s", ((flow->flow_flags & JDF_FLOW_TYPE_CTL) ? "FLOW_ACCESS_NONE" :
                                               ((flow->flow_flags & JDF_FLOW_TYPE_READ) ?
                                                ((flow->flow_flags & JDF_FLOW_TYPE_WRITE) ? "FLOW_ACCESS_RW" : "FLOW_ACCESS_READ") : "FLOW_ACCESS_WRITE")));
    if(flow->flow_flags & JDF_FLOW_HAS_IN_DEPS)
        string_arena_add_string(flow_flags, "|FLOW_HAS_IN_DEPS");

    if(strlen(string_arena_get_string(sa_dep_in)) == 0) {
        string_arena_add_string(sa_dep_in, "NULL");
    }
    if(strlen(string_arena_get_string(sa_dep_out)) == 0) {
        string_arena_add_string(sa_dep_out, "NULL");
    }
    string_arena_add_string(sa,
                            "static const dague_flow_t %s = {\n"
                            "  .name               = \"%s\",\n"
                            "  .sym_type           = %s,\n"
                            "  .flow_flags         = %s,\n"
                            "  .flow_index         = %u,\n"
                            "  .flow_datatype_mask = 0x%x,\n"
                            "  .dep_in     = { %s },\n"
                            "  .dep_out    = { %s }\n"
                            "};\n\n",
                            JDF_OBJECT_ONAME(flow),
                            flow->varname,
                            sym_type,
                            string_arena_get_string(flow_flags),
                            flow->flow_index,
                            flow_datatype_mask,
                            string_arena_get_string(sa_dep_in),
                            string_arena_get_string(sa_dep_out));
    string_arena_free(sa_dep_in);
    string_arena_free(sa_dep_out);
    string_arena_free(flow_flags);

    coutput("%s", string_arena_get_string(sa));
    string_arena_free(sa);

    /* This checks that the size of the dependency_t is big enough to
     * store all the flows, using the MASK method. Return false if
     * the MASK method must be discarded, and 1 if the MASK method
     * can be used. */
    *has_control_gather |= !indepnorange;
    return indepnorange && (((dague_dependency_t)(((1 << flow->flow_index) & 0x1fffffff /*~DAGUE_DEPENDENCIES_BITMASK*/))) != 0);
}

/**
 * Parse the whole dependency list and identify any possible combination
 * that will allow this task (based on its inputs) to be executed as a
 * startup task. In other words, if there is any tuple of the execution
 * space, which leads to all inputs being ready, either comming directly
 * from the matrix or due to write-only status.
 *
 * @Return: If the task cannot be a startup task, then the pint
 *          argument shall be set to zero.
 */
static char* has_ready_input_dependency(void **elt, void *pint)
{
    jdf_dataflow_t* flow = (jdf_dataflow_t*)elt;
    jdf_dep_t* dep = flow->deps;
    int can_be_startup = 0, has_input = 0;

    if( JDF_FLOW_TYPE_CTL & flow->flow_flags ) {
        has_input = 1;
        can_be_startup = 1;
        while( NULL != dep ) {
            if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
                if( dep->guard->guard_type != JDF_GUARD_BINARY )
                    can_be_startup = 0;
            }
            dep = dep->next;
        }
    } else {  /* This is a data */
        for( ; NULL != dep; dep = dep->next ) {
            if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
                /* Skip the default type declaration for WRITE-only dependencies */
                if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dep) )
                    continue;

                has_input = 1;

                if( NULL == dep->guard->calltrue->var ) {
                    can_be_startup = 1;
                }
                if( dep->guard->guard_type == JDF_GUARD_TERNARY ) {
                    if( NULL == dep->guard->callfalse->var ) {
                        can_be_startup = 1;
                    }
                }
            }
        }
    }
    if( (0 == can_be_startup) && has_input ) {
        *((int*)pint) = 0;
    }
    return NULL;
}

static char* dump_direct_input_conditions(void **elt, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg, *sa1;
    jdf_dataflow_t* flow = (jdf_dataflow_t*)elt;
    jdf_dep_t* dep = flow->deps;
    int already_added = 0;
    expr_info_t info;

    string_arena_init(sa);
    sa1 = string_arena_new(64);

    info.prefix = "";
    info.suffix = "";
    info.sa = sa;
    info.assignments = "assignments";

    while( NULL != dep ) {
        if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
            if( dep->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
                if( NULL == dep->guard->calltrue->var ) {
                    /* Always */
                }
            }
            if( dep->guard->guard_type == JDF_GUARD_BINARY ) {
                if( (NULL == dep->guard->calltrue->var) ||
                    (flow->flow_flags & JDF_FLOW_TYPE_CTL)) {
                    if( 0 == already_added ) {
                        info.sa = sa;
                        dump_expr((void**)dep->guard->guard, &info);
                        already_added = 1;
                    } else {
                        string_arena_init(sa1);
                        info.sa = sa1;
                        dump_expr((void**)dep->guard->guard, &info);
                        string_arena_add_string( sa, " || (%s) ", string_arena_get_string(sa1) );
                    }
                }
            }
            if( dep->guard->guard_type == JDF_GUARD_TERNARY ) {
                if( (NULL == dep->guard->calltrue->var) && (NULL == dep->guard->callfalse->var) ) {
                    /* Always */
                } else {
                    if( NULL == dep->guard->calltrue->var ) {
                        if( 0 == already_added ) {
                            info.sa = sa;
                            dump_expr((void**)dep->guard->guard, &info);
                            already_added = 1;
                        } else {
                            string_arena_init(sa1);
                            info.sa = sa1;
                            dump_expr((void**)dep->guard->guard, &info);
                            string_arena_add_string( sa, " || (%s) ", string_arena_get_string(sa1) );
                        }
                    } else if( NULL == dep->guard->callfalse->var ) {
                        string_arena_init(sa1);
                        info.sa = sa1;
                        dump_expr((void**)dep->guard->guard, &info);
                        if( 0 == already_added ) {
                            string_arena_add_string( sa, "(!(%s)) ", string_arena_get_string(sa1) );
                            already_added = 1;
                        } else {
                            string_arena_add_string( sa, " || (!(%s)) ", string_arena_get_string(sa1) );
                        }
                    }
                }
            }
        }
        dep = dep->next;
    }
    /* We need to prepend ! if we're dealing with control flows */
    if( already_added && (JDF_FLOW_TYPE_CTL & flow->flow_flags) ) {
        string_arena_init(sa1);
        string_arena_add_string( sa1, "!(%s)", string_arena_get_string(sa) );
        string_arena_init(sa);
        string_arena_add_string( sa, "%s", string_arena_get_string(sa1) );
    }
    string_arena_free(sa1);
    return (0 == already_added) ? NULL : string_arena_get_string(sa);
}

static void jdf_generate_startup_tasks(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{
    string_arena_t *sa1, *sa2, *sa3;
    jdf_def_list_t *dl;
    int nesting;
    expr_info_t info1, info2, info3;
    int idx;
    int nbdefinitions;

    assert( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP );
    (void)jdf;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);

    coutput("static int %s(dague_context_t *context, __dague_%s_internal_handle_t *__dague_handle, dague_execution_context_t** pready_list)\n"
            "{\n"
            "  %s* new_context, new_context_holder, *new_dynamic_context;\n"
            "  %s *locals = NULL;\n"
            "  int vpid = 0;\n"
            "%s\n"
            "%s\n"
            "  new_context = &new_context_holder;\n"
            "  locals = &new_context->locals;\n",
            fname, jdf_basename,
            dague_get_name(jdf, f, "task_t"),
            dague_get_name(jdf, f, "assignment_t"),
            UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name, dump_string, NULL,
                                 "  int32_t ", " ", " = -1,", " = -1;"),
            UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name, dump_string, NULL,
                                 "  ", "(void)", "; ", ";"));

    string_arena_init(sa1);
    string_arena_init(sa2);
    string_arena_init(sa3);

    coutput("  /* Parse all the inputs and generate the ready execution tasks */\n");
    coutput("  new_context->dague_handle = (dague_handle_t*)__dague_handle;\n"
            "  new_context->function = __dague_handle->super.super.functions_array[%s_%s.function_id];\n",
            jdf_basename, f->fname);

    info1.sa = sa1;
    info1.prefix = "";
    info1.suffix = "";
    info1.assignments = "locals";

    info2.sa = sa2;
    info2.prefix = "";
    info2.suffix = "";
    info2.assignments = "locals";

    info3.sa = sa3;
    info3.prefix = "";
    info3.suffix = "";
    info3.assignments = "locals";

    nesting = 0;
    idx = 0;
    for(dl = f->locals; dl != NULL; dl = dl->next, idx++) {
        if(dl->expr->op == JDF_RANGE) {
            coutput("%s  for(%s = %s;\n"
                    "%s      %s <= %s;\n"
                    "%s      %s+=%s) {\n"
                    "%s    locals->%s.value = %s;\n",
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta1, &info1),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta2, &info2),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta3, &info3),
                    indent(nesting), dl->name, dl->name);
            nesting++;
        } else {
            coutput("%s  locals->%s.value = %s = %s;\n",
                    indent(nesting), dl->name, dl->name, dump_expr((void**)dl->expr, &info1));
        }
    }
    coutput("%s  if( !%s_pred(%s) ) continue;\n",
            indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                                                            dump_string, NULL,
                                                            "", "", ", ", ""));
    /**
     * Dump all the conditions that can invalidate the startup propriety.
     */
    {
        char* condition = NULL;
        string_arena_init(sa2);

        condition = UTIL_DUMP_LIST(sa1, f->dataflow, next, dump_direct_input_conditions, sa2,
                                   "", "(", ") && ", ")");
        if( strlen(condition) > 1 )
            coutput("%s  if( !(%s) ) continue;\n", indent(nesting), condition );
    }

    coutput("%s  if( NULL != ((dague_ddesc_t*)__dague_handle->super.%s)->vpid_of ) {\n"
            "%s    vpid = ((dague_ddesc_t*)__dague_handle->super.%s)->vpid_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n"
            "%s    assert(context->nb_vp >= vpid);\n"
            "%s  }\n"
            "%s  new_dynamic_context = (%s*)dague_lifo_pop(&context->virtual_processes[vpid]->execution_units[0]->context_mempool->mempool);\n"
            "%s  if( NULL == new_dynamic_context)\n"
            "%s    new_dynamic_context = (%s*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );\n",
            indent(nesting), f->predicate->func_or_mem,
            indent(nesting), f->predicate->func_or_mem, f->predicate->func_or_mem,
            UTIL_DUMP_LIST(sa1, f->predicate->parameters, next,
                           dump_expr, (void*)&info2,
                           "", "", ", ", ""),
            indent(nesting),
            indent(nesting),
            indent(nesting), dague_get_name(jdf, f, "task_t"),
            indent(nesting),
            indent(nesting), dague_get_name(jdf, f, "task_t"));

    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_def_list_t, next, nbdefinitions);
    coutput("%s  /* Copy only the valid elements from new_context to new_dynamic one */\n"
            "%s  new_dynamic_context->dague_handle = new_context->dague_handle;\n"
            "%s  new_dynamic_context->function     = new_context->function;\n"
            "%s  new_dynamic_context->chore_id     = 0;\n"
            "%s  memcpy(&new_dynamic_context->locals, &new_context->locals, %d*sizeof(assignment_t));\n",
            indent(nesting),
            indent(nesting),
            indent(nesting),
            indent(nesting),
            indent(nesting), nbdefinitions);

    coutput("%s  DAGUE_LIST_ITEM_SINGLETON(new_dynamic_context);\n",
            indent(nesting));
    if( NULL != f->priority ) {
        coutput("%s  new_dynamic_context->priority = __dague_handle->super.super.priority + priority_of_%s_%s_as_expr_fct((__dague_%s_internal_handle_t*)new_dynamic_context->dague_handle, &new_dynamic_context->locals);\n",
                indent(nesting), jdf_basename, f->fname, jdf_basename);
    } else {
        coutput("%s  new_dynamic_context->priority = __dague_handle->super.super.priority;\n", indent(nesting));
    }

    {
        struct jdf_dataflow *dataflow = f->dataflow;
        for(idx = 0; NULL != dataflow; idx++, dataflow = dataflow->next ) {
            coutput("%s  new_dynamic_context->data.%s.data_repo = NULL;\n"
                    "%s  new_dynamic_context->data.%s.data_in   = NULL;\n"
                    "%s  new_dynamic_context->data.%s.data_out  = NULL;\n",
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname);
        }
    }

    coutput("#if DAGUE_DEBUG_VERBOSE != 0\n"
            "%s  {\n"
            "%s    char tmp[128];\n"
            "%s    DEBUG2((\"Add startup task %%s\\n\",\n"
            "%s           dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_dynamic_context)));\n"
            "%s  }\n"
            "#endif\n", indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting));

    coutput("%s  dague_dependencies_mark_task_as_startup( (dague_execution_context_t*)new_dynamic_context);\n", indent(nesting));

    coutput("        if( NULL != pready_list[vpid] ) {\n"
            "          dague_list_item_ring_merge((dague_list_item_t*)new_dynamic_context,\n"
            "                                     (dague_list_item_t*)(pready_list[vpid]));\n"
            "        }\n"
            "        pready_list[vpid] = (dague_execution_context_t*)new_dynamic_context;\n");

    for(; nesting > 0; nesting--) {
        coutput("%s}\n", indent(nesting));
    }

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);

    coutput("  return 0;\n"
            "}\n\n");
}

static void jdf_generate_internal_init(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{
    string_arena_t *sa1, *sa2, *sa3;
    jdf_def_list_t *dl;
    jdf_name_list_t *pl;
    int nesting, idx;
    expr_info_t info1, info2, info3;

    (void)jdf;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);

    coutput("static int %s(__dague_%s_internal_handle_t *__dague_handle)\n"
            "{\n"
            "  dague_dependencies_t *dep = NULL;\n"
            "  %s assignments;(void) assignments;\n"
            "  int nb_tasks = 0;\n"
            "%s",
            fname, jdf_basename,
            dague_get_name(jdf, f, "assignment_t"),
            UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name, dump_string, NULL,
                                 "  int32_t ", " ", ",", ";\n"));
    coutput("%s"
            "%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_min = 0x7fffffff,", "_min = 0x7fffffff;\n"),
            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_max = 0,", "_max = 0;\n"));

    coutput("  (void)__dague_handle;\n");
    if( NULL != f->parameters->next ) {
        for(pl = f->parameters; pl != NULL; pl = pl->next ) {
            for(dl = f->locals; dl != NULL; dl = dl->next) {
                if(!strcmp(pl->name, dl->name))
                    break;
            }
            /* This should be already checked by a sanity check */
            assert(NULL != dl);
            if(dl->expr->op == JDF_RANGE) {
                coutput("  int32_t %s_start, %s_end, %s_inc;\n", pl->name, pl->name, pl->name );
            }
        }
    }

    string_arena_init(sa1);
    string_arena_init(sa2);
    string_arena_init(sa3);

    info1.sa = sa1;
    info1.prefix = "";
    info1.suffix = "";
    info1.assignments = "&assignments";

    info2.sa = sa2;
    info2.prefix = "";
    info2.suffix = "";
    info2.assignments = "&assignments";

    info3.sa = sa3;
    info3.prefix = "";
    info3.suffix = "";
    info3.assignments = "&assignments";

    coutput("  /* First, find the min and max value for each of the dimensions */\n");

    idx = 0;
    nesting = 0;
    for(dl = f->locals; dl != NULL; dl = dl->next ) {
        if(dl->expr->op == JDF_RANGE) {
            coutput("%s  for(%s = %s;\n"
                    "%s      %s <= %s;\n"
                    "%s      %s += %s) {\n",
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta1, &info1),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta2, &info2),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta3, &info3));
            nesting++;
        } else {
            coutput("%s  %s = %s;\n",
                    indent(nesting), dl->name, dump_expr((void**)dl->expr, &info1));
        }
        coutput("%s  assignments.%s.value = %s;\n",
                indent(nesting), dl->name, dl->name);
        idx++;
    }

    for(pl = f->parameters; pl != NULL; pl = pl->next ) {
        coutput("%s  %s_max = dague_imax(%s_max, %s);\n"
                "%s  %s_min = dague_imin(%s_min, %s);\n",
                indent(nesting), pl->name, pl->name, pl->name,
                indent(nesting), pl->name, pl->name, pl->name);
    }

    string_arena_init(sa1);
    coutput("%s  if( !%s_pred(%s) ) continue;\n"
            "%s  nb_tasks++;\n",
            indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                                                            dump_string, NULL,
                                                            "", "", ", ", ""),
            indent(nesting));

    for(; nesting > 0; nesting--) {
        coutput("%s}\n", indent(nesting));
    }

    coutput("\n"
            "  /**\n"
            "   * Set the range variables for the collision-free hash-computation\n"
            "   */\n");
    for(pl = f->parameters; pl != NULL; pl = pl->next) {
        coutput("  __dague_handle->%s_%s_range = (%s_max - %s_min) + 1;\n",
                f->fname, pl->name, pl->name, pl->name);
    }

    coutput("\n"
            "  /**\n"
            "   * Now, for each of the dimensions, re-iterate on the space,\n"
            "   * and if at least one value is defined, allocate arrays to point\n"
            "   * to it. Array dimensions are defined by the (rough) observation above\n"
            "   **/\n"
            "  DEBUG3((\"Allocating dependencies array for %s\\n\"));\n", fname);

    if( f->parameters->next == NULL ) {
        coutput("  if( 0 != nb_tasks ) {\n"
                "    ALLOCATE_DEP_TRACKING(dep, %s_min, %s_max, \"%s\", &symb_%s_%s_%s, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);\n"
                "  }\n",
                f->parameters->name, f->parameters->name, f->parameters->name,
                jdf_basename, f->fname, f->parameters->name);
    } else {
        coutput("  dep = NULL;\n");

        nesting = 0;
        idx = 0;
        for(dl = f->locals; dl != NULL; dl = dl->next) {

            for(pl = f->parameters; pl != NULL; pl = pl->next) {
                if(!strcmp(pl->name, dl->name))
                    break;
            }

            if(dl->expr->op == JDF_RANGE) {
                coutput("%s  %s_start = %s;\n",
                        indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta1, &info1));
                coutput("%s  %s_end = %s;\n",
                        indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta2, &info2));
                coutput("%s  %s_inc = %s;\n",
                        indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ta3, &info3));
                coutput("%s  for(%s = dague_imax(%s_start, %s_min); %s <= dague_imin(%s_end, %s_max); %s+=%s_inc) {\n",
                        indent(nesting), dl->name, dl->name, dl->name, dl->name, dl->name, dl->name, dl->name, dl->name);
                nesting++;
            } else {
                coutput("%s  %s = %s;\n",
                        indent(nesting), dl->name, dump_expr((void**)dl->expr, &info1));
            }

            coutput("%s  assignments.%s.value = %s;\n",
                    indent(nesting), dl->name, dl->name);
            idx++;
        }

        coutput("%s  if( %s_pred(%s) ) {\n"
                "%s    /* We did find one! Allocate the dependencies array. */\n",
                indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name,
                                                                dump_string, NULL,
                                                                "", "", ", ", ""),
                indent(nesting));
        nesting++;

        string_arena_init(sa1);
        string_arena_add_string(sa1, "dep");
        for(pl = f->parameters; pl != NULL; pl = pl->next) {
            for(dl = f->locals; dl != NULL; dl = dl->next) {
                if(!strcmp(pl->name, dl->name))
                    break;
            }
            assert(NULL != dl);
            coutput("%s  if( %s == NULL ) {\n"
                    "%s    ALLOCATE_DEP_TRACKING(%s, %s_min, %s_max, \"%s\", &symb_%s_%s_%s, %s, %s);\n"
                    "%s  }\n",
                    indent(nesting), string_arena_get_string(sa1),
                    indent(nesting), string_arena_get_string(sa1), dl->name, dl->name, dl->name,
                                     jdf_basename, f->fname, dl->name,
                                     pl == f->parameters ? "NULL" : string_arena_get_string(sa2),
                                     pl->next == NULL ? "DAGUE_DEPENDENCIES_FLAG_FINAL" : "DAGUE_DEPENDENCIES_FLAG_NEXT",
                    indent(nesting));
            string_arena_init(sa2);
            string_arena_add_string(sa2, "%s", string_arena_get_string(sa1));
            string_arena_add_string(sa1, "->u.next[%s-%s_min]", dl->name, dl->name);
        }
        coutput("%s  }\n", indent(nesting));
        nesting--;

        for(; nesting > 0; nesting--) {
            coutput("%s}\n", indent(nesting));
        }
    }

    /* Quiet the compiler by using the variables */
    if( NULL != f->parameters->next ) {
        for(pl = f->parameters; pl != NULL; pl = pl->next) {
            for(dl = f->locals; dl != NULL; dl = dl->next) {
                if(!strcmp(pl->name, dl->name))
                    break;
            }
            assert(NULL != dl);
            if(dl->expr->op == JDF_RANGE) {
                coutput("  (void)%s_start; (void)%s_end; (void)%s_inc;", dl->name, dl->name, dl->name);
            }
        }
    }

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    coutput("\n  AYU_REGISTER_TASK(&%s_%s);\n", jdf_basename, f->fname);
    coutput("  __dague_handle->super.super.dependencies_array[%d] = dep;\n"
            "  __dague_handle->super.super.nb_local_tasks += nb_tasks;\n"
            "  return nb_tasks;\n"
            "}\n"
            "\n",
            f->function_id);
}

static void jdf_generate_simulation_cost_fct(const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix)
{
    assignment_info_t ai;
    expr_info_t info;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);

    (void)jdf;

    ai.sa = sa;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;

    coutput("#if defined(DAGUE_SIM)\n"
            "static int %s(const %s *this_task)\n"
            "{\n"
            "  const dague_handle_t *__dague_handle = (const dague_handle_t*)this_task->dague_handle;\n"
            "%s"
            "  (void)__dague_handle;\n",
            prefix, dague_get_name(jdf, f, "task_t"),
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    string_arena_init(sa);
    coutput("%s",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    string_arena_init(sa);
    info.prefix = "";
    info.suffix = "";
    info.sa = sa;
    info.assignments = "&this_task->locals";
    coutput("  return %s;\n", dump_expr((void**)f->simcost, &info));
    coutput("}\n"
            "#endif\n"
            "\n");

    string_arena_free(sa);
    string_arena_free(sa1);
}

static void
jdf_generate_function_incarnation_list( const jdf_t *jdf,
                                        const jdf_function_entry_t *f,
                                        string_arena_t *sa,
                                        char* base_name)
{
    jdf_body_t* body = f->bodies;
    jdf_def_list_t* type_property;
    jdf_def_list_t* dyld_property;

    (void)jdf;
    string_arena_add_string(sa, "static const __dague_chore_t __%s_chores[] ={\n", base_name);
    do {
        jdf_find_property(body->properties, "type", &type_property);
        jdf_find_property(body->properties, "dyld", &dyld_property);
        if( NULL == type_property) {
            string_arena_add_string(sa, "    { .type     = DAGUE_DEV_CPU,\n");
            string_arena_add_string(sa, "      .evaluate = %s,\n", "NULL");
            string_arena_add_string(sa, "      .hook     = (dague_hook_t*)hook_of_%s },\n", base_name);
        } else {
            string_arena_add_string(sa, "#if defined(HAVE_%s)\n", type_property->expr->jdf_var);
            string_arena_add_string(sa, "    { .type     = DAGUE_DEV_%s,\n", type_property->expr->jdf_var);
            if( NULL == dyld_property ) {
                string_arena_add_string(sa, "      .dyld     = NULL,\n");
            } else {
                string_arena_add_string(sa, "      .dyld     = \"%s\",\n", dyld_property->expr->jdf_var);
            }
            string_arena_add_string(sa, "      .evaluate = %s,\n", "NULL");
            string_arena_add_string(sa, "      .hook     = (dague_hook_t*)hook_of_%s_%s },\n", base_name, type_property->expr->jdf_var);
            string_arena_add_string(sa, "#endif  /* defined(HAVE_%s) */\n", type_property->expr->jdf_var);
        }
        body = body->next;
    } while (NULL != body);
    string_arena_add_string(sa,
                            "    { .type     = DAGUE_DEV_NONE,\n"
                            "      .evaluate = NULL,\n"
                            "      .hook     = (dague_hook_t*)NULL },  /* End marker */\n"
                            "};\n\n");
}

static void jdf_generate_one_function( const jdf_t *jdf, jdf_function_entry_t *f)
{
    string_arena_t *sa, *sa2;
    int nbparameters, nbdefinitions;
    int inputmask, nb_input, nb_output, input_index;
    int i, has_in_in_dep, has_control_gather, ret, use_mask;
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;
    char *prefix;
    int rc;

    rc = asprintf( &JDF_OBJECT_ONAME(f), "%s_%s", jdf_basename, f->fname);
    assert(rc != -1);

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    JDF_COUNT_LIST_ENTRIES(f->parameters, jdf_name_list_t, next, nbparameters);
    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_def_list_t, next, nbdefinitions);

    inputmask = 0;
    nb_input = nb_output = 0;
    has_in_in_dep = 0;
    for( input_index = 0, fl = f->dataflow;
         NULL != fl;
         fl = fl->next, input_index++ ) {
        int foundin = 0, foundout = 0;

        for( dl = fl->deps; NULL != dl; dl = dl->next ) {
            if( dl->dep_flags & JDF_DEP_FLOW_IN ) {

                if( JDF_FLOW_TYPE_CTL & fl->flow_flags ) {
                    if( JDF_GUARD_BINARY == dl->guard->guard_type )
                        fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                } else {
                    switch( dl->guard->guard_type ) {
                    case JDF_GUARD_TERNARY:
                        if( NULL == dl->guard->callfalse->var )
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;

                    case JDF_GUARD_UNCONDITIONAL:
                        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                            break;
                        }

                    case JDF_GUARD_BINARY:
                        if( NULL == dl->guard->calltrue->var )
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                    }
                }
                if( foundin == 0 ) {
                    inputmask |= (1 << fl->flow_index);
                    nb_input++;
                    foundin = 1;
                }
            } else if( dl->dep_flags & JDF_DEP_FLOW_OUT )
                if( 0 == foundout ) { nb_output++; foundout = 1; }
        }
        has_in_in_dep |= (fl->flow_flags & JDF_FLOW_HAS_IN_DEPS);
    }

    jdf_coutput_prettycomment('*', "%s", f->fname);

    prefix = (char*)malloc(strlen(f->fname) + strlen(jdf_basename) + 32);

    sprintf(prefix, "%s_%s", jdf_basename, f->fname);
    jdf_generate_function_incarnation_list(jdf, f, sa, prefix);

    string_arena_add_string(sa,
                            "static const dague_function_t %s = {\n"
                            "  .name = \"%s\",\n"
                            "  .function_id = %d,\n"
                            "  .nb_flows = %d,\n"
                            "  .nb_parameters = %d,\n"
                            "  .nb_locals = %d,\n",
                            JDF_OBJECT_ONAME(f),
                            f->fname,
                            f->function_id,
                            input_index,
                            nbparameters,
                            nbdefinitions);

    sprintf(prefix, "symb_%s_%s_", jdf_basename, f->fname);
    jdf_generate_symbols(jdf, f, prefix);
    sprintf(prefix, "&symb_%s_%s_", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .params = { %s },\n",
                            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL,
                                                 "", prefix, ", ", ", NULL"));
    string_arena_add_string(sa, "  .locals = { %s },\n",
                            UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name, dump_string, NULL,
                                                 "", prefix, ", ", ", NULL"));

    sprintf(prefix, "affinity_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_affinity(jdf, f, prefix);
    string_arena_add_string(sa, "  .data_affinity = (dague_data_ref_fn_t*)%s,\n", prefix);

    sprintf(prefix, "initial_data_of_%s_%s", jdf_basename, f->fname);
    ret = jdf_generate_initfinal_data(jdf, JDF_DEP_FLOW_IN, f, prefix);
    string_arena_add_string(sa, "  .initial_data = (dague_data_ref_fn_t*)%s,\n", (0 != ret ? prefix : "NULL"));

    sprintf(prefix, "final_data_of_%s_%s", jdf_basename, f->fname);
    ret = jdf_generate_initfinal_data(jdf, JDF_DEP_FLOW_OUT, f, prefix);
    string_arena_add_string(sa, "  .final_data = (dague_data_ref_fn_t*)%s,\n", (0 != ret ? prefix : "NULL"));

    if( NULL != f->priority ) {
        sprintf(prefix, "priority_of_%s_%s_as_expr", jdf_basename, f->fname);
        jdf_generate_expression(jdf, f, f->priority, prefix);
        string_arena_add_string(sa, "  .priority = &%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .priority = NULL,\n");
    }

#if defined(DAGUE_SCHED_DEPS_MASK)
    use_mask = 1;
#else
    use_mask = 0;
#endif
    if( jdf_property_get_int(f->properties, "count_deps", 0) )
        use_mask = 0;
    if( jdf_property_get_int(f->properties, "mask_deps", 0) )
        use_mask = 1;
    sprintf(prefix, "flow_of_%s_%s_for_", jdf_basename, f->fname);
    has_control_gather = 0;
    for(i = 0, fl = f->dataflow; fl != NULL; fl = fl->next, i++)
        use_mask &= jdf_generate_dataflow(jdf, f, fl, prefix, &has_control_gather);

    if( jdf_property_get_int(f->properties, "mask_deps", 0) && (use_mask == 0) ) {
        jdf_warn(JDF_OBJECT_LINENO(f),
                 "In task %s, mask_deps was requested, but this method cannot be provided\n"
                 "  Either the function uses too many flows to store in a mask\n"
                 "  Or it uses control gather, which must be counted\n"
                 "  Falling back to the counting method for dependency managing.\n",
                 f->fname);
    }
    sprintf(prefix, "&flow_of_%s_%s_for_", jdf_basename, f->fname);
    UTIL_DUMP_LIST(sa2, f->dataflow, next, dump_dataflow, "IN", "", prefix, ", ", "");
    if( 0 == strlen(string_arena_get_string(sa2)) )
        string_arena_add_string(sa2, "NULL");
    else
        string_arena_add_string(sa2, ", NULL");
    string_arena_add_string(sa, "  .in = { %s },\n",
                            string_arena_get_string(sa2));
    UTIL_DUMP_LIST(sa2, f->dataflow, next, dump_dataflow, "OUT", "", prefix, ", ", "");
    if( 0 == strlen(string_arena_get_string(sa2)) )
        string_arena_add_string(sa2, "NULL");
    else
        string_arena_add_string(sa2, ", NULL");
    string_arena_add_string(sa, "  .out = { %s },\n",
                            string_arena_get_string(sa2));

    if( use_mask ) {
        string_arena_add_string(sa,
                                "  .flags = %s%s%s | DAGUE_USE_DEPS_MASK,\n"
                                "  .dependencies_goal = 0x%x,\n",
                                (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "DAGUE_HIGH_PRIORITY_TASK" : "0x0",
                                has_in_in_dep ? " | DAGUE_HAS_IN_IN_DEPENDENCIES" : "",
                                jdf_property_get_int(f->properties, "immediate", 0) ? " | DAGUE_IMMEDIATE_TASK" : "",
                                inputmask);
    } else {
        string_arena_add_string(sa,
                                "  .flags = %s%s%s%s,\n"
                                "  .dependencies_goal = %d,\n",
                                (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "DAGUE_HIGH_PRIORITY_TASK" : "0x0",
                                has_in_in_dep ? " | DAGUE_HAS_IN_IN_DEPENDENCIES" : "",
                                jdf_property_get_int(f->properties, "immediate", 0) ? " | DAGUE_IMMEDIATE_TASK" : "",
                                has_control_gather ? "|DAGUE_HAS_CTL_GATHER" : "",
                                nb_input);
    }

    string_arena_add_string(sa, "  .key = (dague_functionkey_fn_t*)%s_hash,\n", f->fname);
    string_arena_add_string(sa, "  .fini = (dague_hook_t*)%s,\n", "NULL");

    sprintf(prefix, "%s_%s", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .incarnations = __%s_chores,\n", prefix);

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) ) {
        sprintf(prefix, "iterate_successors_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_code_iterate_successors_or_predecessors(jdf, f, prefix, JDF_DEP_FLOW_OUT);
        string_arena_add_string(sa, "  .iterate_successors = (dague_traverse_function_t*)%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .iterate_successors = (dague_traverse_function_t*)NULL,\n");
    }

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_PREDECESSORS) ) {
        sprintf(prefix, "iterate_predecessors_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_code_iterate_successors_or_predecessors(jdf, f, prefix, JDF_DEP_FLOW_IN);
        string_arena_add_string(sa, "  .iterate_predecessors = (dague_traverse_function_t*)%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .iterate_predecessors = (dague_traverse_function_t*)NULL,\n");
    }

    sprintf(prefix, "release_deps_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_release_deps(jdf, f, prefix);
    string_arena_add_string(sa, "  .release_deps = (dague_release_deps_t*)%s,\n", prefix);

    sprintf(prefix, "data_lookup_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_data_lookup(jdf, f, prefix);
    string_arena_add_string(sa, "  .prepare_input = (dague_hook_t*)%s,\n", prefix);
    string_arena_add_string(sa, "  .prepare_output = (dague_hook_t*)%s,\n", "NULL");

    sprintf(prefix, "hook_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_hooks(jdf, f, prefix);
    string_arena_add_string(sa, "  .complete_execution = (dague_hook_t*)complete_%s,\n", prefix);

    string_arena_add_string(sa, "  .release_task = (dague_hook_t*)dague_release_task_to_mempool,\n");

    if( NULL != f->simcost ) {
        sprintf(prefix, "simulation_cost_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_simulation_cost_fct(jdf, f, prefix);
        string_arena_add_string(sa,
                                "#if defined(DAGUE_SIM)\n"
                                "  .sim_cost_fct =(dague_sim_cost_fct_t*) %s,\n"
                                "#endif\n", prefix);
    } else {
        string_arena_add_string(sa,
                                "#if defined(DAGUE_SIM)\n"
                                "  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,\n"
                                "#endif\n");
    }

    sprintf(prefix, "%s_%s_internal_init", jdf_basename, f->fname);
    jdf_generate_internal_init(jdf, f, prefix);

    if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
        sprintf(prefix, "%s_%s_startup_tasks", jdf_basename, f->fname);
        jdf_generate_startup_tasks(jdf, f, prefix);
    }

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
    string_arena_add_string(sa, "static const dague_function_t *%s_functions[] = {\n",
                            jdf_basename);
    /* We need to put the function in the array based on their function_id */
    {
        jdf_function_entry_t** array;
        int max_id;

        for(max_id = 0, f = jdf->functions; NULL != f; f = f->next) {
            jdf_generate_one_function(jdf, f);
            if( max_id < f->function_id ) max_id = f->function_id;
        }
        max_id++;  /* allow one more space */
        array = (jdf_function_entry_t**)malloc(max_id * sizeof(jdf_function_entry_t*));
        for(i = 0; i < max_id; array[i] = NULL, i++);
        for(f = jdf->functions; NULL != f; f = f->next)
            array[f->function_id] = f;
        for(i = 0; i < max_id; array[i] = NULL, i++) {
            if( NULL == (f = array[i]) ) {
                string_arena_add_string(sa, "  NULL%s\n",
                                        i != (max_id - 1) ? "," : "");
            } else {
                string_arena_add_string(sa, "  &%s_%s%s\n",
                                        jdf_basename, f->fname, i != (max_id - 1) ? "," : "");
            }
        }
    }
    string_arena_add_string(sa, "};\n\n");
    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
}

static void jdf_generate_predeclarations( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    int rc;

    coutput("/** Predeclarations of the dague_function_t */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        rc = asprintf(&JDF_OBJECT_ONAME( f ), "%s_%s", jdf_basename, f->fname);
        assert(rc != -1);
        coutput("static const dague_function_t %s;\n", JDF_OBJECT_ONAME( f ));
        if( NULL != f->priority ) {
            coutput("static inline int priority_of_%s_as_expr_fct(const __dague_%s_internal_handle_t *__dague_handle, const %s *assignments);\n",
                    JDF_OBJECT_ONAME( f ), jdf_basename, dague_get_name(jdf, f, "assignment_t"));
        }
    }
    string_arena_free(sa);
    string_arena_free(sa2);
    coutput("/** Predeclarations of the parameters */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        for(fl = f->dataflow; fl != NULL; fl = fl->next) {
            rc = asprintf(&JDF_OBJECT_ONAME( fl ), "flow_of_%s_%s_for_%s", jdf_basename, f->fname, fl->varname);
            assert(rc != -1);
            coutput("static const dague_flow_t %s;\n",
                    JDF_OBJECT_ONAME( fl ));
        }
    }
}

static void jdf_generate_startup_hook( const jdf_t *jdf )
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);

    coutput("static void %s_startup(dague_context_t *context, __dague_%s_internal_handle_t *__dague_handle, dague_execution_context_t** pready_list)\n"
            "{\n"
            "  uint32_t supported_dev = 0;\n"
            " \n"
            "  uint32_t wanted_devices = __dague_handle->super.super.devices_mask; __dague_handle->super.super.devices_mask = 0;\n"
            "  uint32_t _i;\n"
            "  for( _i = 0; _i < dague_nb_devices; _i++ ) {\n"
            "    if( !(wanted_devices & (1<<_i)) ) continue;\n"
            "    dague_device_t* device = dague_devices_get(_i);\n"
            "    dague_ddesc_t* dague_ddesc;\n"
            " \n"
            "    if(NULL == device) continue;\n"
            "    if(NULL != device->device_handle_register)\n"
            "      if( DAGUE_SUCCESS != device->device_handle_register(device, (dague_handle_t*)__dague_handle) ) {\n"
            "        dague_output_verbose(1, 0, \"Device %%s refused to register handle %%p\\n\", device->name, __dague_handle);\n"
            "        continue;\n"
            "      }\n"
            "    if(NULL != device->device_memory_register) {  /* Register all the data */\n"
            "%s"
            "    }\n"
            "    supported_dev |= (1 << device->type);\n"
            "    __dague_handle->super.super.devices_mask |= (1 << _i);\n"
            "  }\n",
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa1, jdf->globals, next,
                           dump_data_name, sa2, "",
                           "      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.",
                           ";\n"
                           "      if(DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) {\n"
                           "        dague_output_verbose(1, 0, \"Device %s refused to register memory for data %s (%p) from handle %p\",\n"
                           "                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);\n"
                           "        continue;\n"
                           "      }\n",
                           ";\n"
                           "      if(DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) {\n"
                           "        dague_output_verbose(1, 0, \"Device %s refused to register memory for data %s (%p) from handle %p\",\n"
                           "                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);\n"
                           "        continue;\n"
                           "      }\n"));
    coutput("  /* Remove all the chores without a backend device */\n"
            "  uint32_t i;\n"
            "  for( i = 0; i < __dague_handle->super.super.nb_functions; i++ ) {\n"
            "    dague_function_t* func = (dague_function_t*)__dague_handle->super.super.functions_array[i];\n"
            "    __dague_chore_t* chores = (__dague_chore_t*)func->incarnations;\n"
            "    uint32_t index = 0;\n"
            "    uint32_t j;\n"
            "    for( j = 0; NULL != chores[j].hook; j++ ) {\n"
            "      if(supported_dev & (1 << chores[j].type)) {\n"
            "          if( j != index ) {\n"
            "            chores[index] = chores[j];\n"
            "            dague_output_verbose(1, 0, \"Device type %%i disabled for function %%s\"\n, chores[j].type, func->name);\n"
            "          }\n"
            "          index++;\n"
            "      }\n"
            "    }\n"
            "    chores[index].type     = DAGUE_DEV_NONE;\n"
            "    chores[index].evaluate = NULL;\n"
            "    chores[index].hook     = NULL;\n"
            "  }\n"
            );

    coutput("%s\n"
            "}\n",
            UTIL_DUMP_LIST( sa1, jdf->functions, next, dump_startup_call, sa2,
                            "  ", jdf_basename, "\n  ", "") );

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_destructor( const jdf_t *jdf )
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);

    coutput("static void %s_destructor( __dague_%s_internal_handle_t *handle )\n"
            "{\n"
            "  uint32_t i;\n",
            jdf_basename, jdf_basename);

    coutput("  for( i = 0; i < handle->super.super.nb_functions; i++ ) {\n"
            "    dague_function_t* func = (dague_function_t*)handle->super.super.functions_array[i];\n"
            "    free((void*)func->incarnations);\n"
            "    free(func);"
            "  }\n"
            "  free(handle->super.super.functions_array);\n"
            "  handle->super.super.functions_array = NULL;\n"
            "  handle->super.super.nb_functions = 0;\n");

    coutput("  for(i = 0; i < (uint32_t)handle->super.arenas_size; i++) {\n"
            "    if( handle->super.arenas[i] != NULL ) {\n"
            "      dague_arena_destruct(handle->super.arenas[i]);\n"
            "      free(handle->super.arenas[i]);\n"
            "      handle->super.arenas[i] = NULL;\n"
            "    }\n"
            "  }\n"
            "  free( handle->super.arenas );\n"
            "  handle->super.arenas = NULL;\n"
            "  handle->super.arenas_size = 0;\n");

    coutput("  /* Destroy the data repositories for this object */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            coutput("   data_repo_destroy_nothreadsafe(handle->repositories[%d]);  /* %s */\n",
                    f->function_id, f->fname);
        }
    }

    coutput("  for(i = 0; i < DAGUE_%s_NB_FUNCTIONS; i++) {\n"
            "    dague_destruct_dependencies( handle->super.super.dependencies_array[i] );\n"
            "    handle->super.super.dependencies_array[i] = NULL;\n"
            "  }\n"
            "  free( handle->super.super.dependencies_array );\n"
            "  handle->super.super.dependencies_array = NULL;\n",
            jdf_basename);

    coutput("  /* Unregister all the data */\n"
            "  uint32_t _i;\n"
            "  for( _i = 0; _i < dague_nb_devices; _i++ ) {\n"
            "    dague_device_t* device;\n"
            "    dague_ddesc_t* dague_ddesc;\n"
            "    if(!(handle->super.super.devices_mask & (1 << _i))) continue;\n"
            "    if((NULL == (device = dague_devices_get(_i))) || (NULL == device->device_memory_unregister)) continue;\n"
            "  %s"
            "}\n",
            UTIL_DUMP_LIST(sa, jdf->globals, next,
                           dump_data_name, sa1, "",
                           "  dague_ddesc = (dague_ddesc_t*)handle->super.",
                           ";\n  (void)dague_ddesc->unregister_memory(dague_ddesc, device);\n",
                           ";\n  (void)dague_ddesc->unregister_memory(dague_ddesc, device);\n"));

    coutput("  /* Unregister the handle from the devices */\n"
            "  for( i = 0; i < dague_nb_devices; i++ ) {\n"
            "    if(!(handle->super.super.devices_mask & (1 << i))) continue;\n"
            "    handle->super.super.devices_mask ^= (1 << i);\n"
            "    dague_device_t* device = dague_devices_get(i);\n"
            "    if((NULL == device) || (NULL == device->device_handle_unregister)) continue;\n"
            "    if( DAGUE_SUCCESS != device->device_handle_unregister(device, &handle->super.super) ) continue;\n"
            "  }\n");

    coutput("  dague_handle_unregister( &handle->super.super );\n"
            "  free(handle);\n");

    coutput("}\n"
            "\n");

    string_arena_free(sa);
    string_arena_free(sa1);
}

static void jdf_generate_constructor( const jdf_t* jdf )
{
    string_arena_t *sa1,*sa2;
    profiling_init_info_t pi;
    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->globals, next, name,
                                  dump_string, NULL, "", "#undef ", "\n", "\n"));

    {
        typed_globals_info_t prop = { sa2, NULL, "hidden" };
        coutput("dague_%s_handle_t *dague_%s_new(%s)\n{\n",
                jdf_basename, jdf_basename,
                UTIL_DUMP_LIST( sa1, jdf->globals, next, dump_typed_globals, &prop,
                                "", "", ", ", ""));
    }

    coutput("  __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)calloc(1, sizeof(__dague_%s_internal_handle_t));\n",
            jdf_basename, jdf_basename, jdf_basename);

    string_arena_init(sa1);
    string_arena_init(sa2);
    {
        coutput("  /* Dump the hidden parameters */\n"
                "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                     dump_hidden_globals_init, sa2, "", "  ", "\n", "\n"));
    }

    string_arena_init(sa1);
    coutput("  int i, j;\n"
            "%s\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname,
                                  dump_string, NULL, "", "  int ", "_nblocal_tasks;\n", "_nblocal_tasks;\n") );

    coutput("  __dague_handle->super.super.nb_functions    = DAGUE_%s_NB_FUNCTIONS;\n", jdf_basename);
    coutput("  __dague_handle->super.super.dependencies_array = (dague_dependencies_t **)\n"
            "              calloc(DAGUE_%s_NB_FUNCTIONS, sizeof(dague_dependencies_t *));\n",
            jdf_basename);
    /* Prepare the functions */
    coutput("  __dague_handle->super.super.devices_mask = DAGUE_DEVICES_ALL;\n");
    coutput("  __dague_handle->super.super.functions_array = (const dague_function_t**)malloc(DAGUE_%s_NB_FUNCTIONS * sizeof(dague_function_t*));\n",
            jdf_basename);
    coutput("  for( i = 0; i < (int)__dague_handle->super.super.nb_functions; i++ ) {\n"
            "    dague_function_t* func;\n"
            "    __dague_handle->super.super.functions_array[i] = malloc(sizeof(dague_function_t));\n"
            "    memcpy((dague_function_t*)__dague_handle->super.super.functions_array[i], %s_functions[i], sizeof(dague_function_t));\n"
            "    func = (dague_function_t*)__dague_handle->super.super.functions_array[i];\n"
            "    for( j = 0; NULL != func->incarnations[j].hook; j++);\n"
            "    func->incarnations = (__dague_chore_t*)malloc((j+1) * sizeof(__dague_chore_t));\n"
            "    memcpy((__dague_chore_t*)func->incarnations, %s_functions[i]->incarnations, (j+1) * sizeof(__dague_chore_t));\n"
            "  }\n",
            jdf_basename,
            jdf_basename);
    {
        struct jdf_name_list* g;
        int datatype_index = 0;
        jdf_expr_t *arena_strut = NULL;
        jdf_def_list_t* prop;

        coutput("  /* Compute the number of arenas: */\n");

        for( g = jdf->datatypes; NULL != g; g = g->next ) {
            coutput("  /*   DAGUE_%s_%s_ARENA  ->  %d */\n",
                    jdf_basename, g->name, datatype_index);
            datatype_index++;
        }
        arena_strut = jdf_find_property(jdf->global_properties, "DAGUE_ARENA_STRUT", &prop);
        if( NULL != arena_strut ) {
            expr_info_t info;

            coutput("  /* and add to that the ARENA_STRUT */\n");

            info.prefix = "";
            info.suffix = "";
            info.sa = string_arena_new(64);
            info.assignments = "NULL";

            coutput("  __dague_handle->super.arenas_size = %d + %s;\n",
                    datatype_index, dump_expr((void**)arena_strut, &info));

            string_arena_free(info.sa);
        } else {
            coutput("  __dague_handle->super.arenas_size = %d;\n", datatype_index);
        }

        coutput("  __dague_handle->super.arenas = (dague_arena_t **)malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t*));\n"
                "  for(i = 0; i < __dague_handle->super.arenas_size; i++) {\n"
                "    __dague_handle->super.arenas[i] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));\n"
                "  }\n");
    }

    coutput("  /* Now the Parameter-dependent structures: */\n"
            "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                 dump_globals_init, sa2, "", "  ", "\n", "\n"));

    pi.sa = sa2;
    pi.idx = 0;
    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, pi.maxidx);
    {
        char* prof = UTIL_DUMP_LIST(sa1, jdf->functions, next,
                                    dump_profiling_init, &pi, "", "    ", "\n", "\n");
        coutput("  /* If profiling is enabled, the keys for profiling */\n"
                "#  if defined(DAGUE_PROF_TRACE)\n");

        if( strcmp(prof, "\n") ) {
            coutput("  __dague_handle->super.super.profiling_array = %s_profiling_array;\n"
                    "  if( -1 == %s_profiling_array[0] ) {\n"
                    "%s"
                    "  }\n",
                    jdf_basename,
                    jdf_basename,
                    prof);
        } else {
            coutput("  __dague_handle->super.super.profiling_array = NULL;\n");
        }
        coutput("#  endif /* defined(DAGUE_PROF_TRACE) */\n");
    }

    coutput("  /* Populate the data repositories for this handle */\n"
            "%s",
            UTIL_DUMP_LIST( sa1, jdf->functions, next, dump_data_repository_constructor, sa2,
                            "", "", "\n", ""));
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            coutput("  __dague_handle->super.super.repo_array = __dague_handle->repositories;\n\n");
            break;
        }
        if( NULL == f )
            coutput("  __dague_handle->super.super.repo_array = NULL;\n");
    }

    coutput("  __dague_handle->super.super.startup_hook = (dague_startup_fn_t)%s_startup;\n"
            "  __dague_handle->super.super.destructor   = (dague_destruct_fn_t)%s_destructor;\n"
            "  (void)dague_handle_reserve_id((dague_handle_t*)__dague_handle);\n",
            jdf_basename, jdf_basename);

    coutput("  return (dague_%s_handle_t*)__dague_handle;\n"
            "}\n\n",
            jdf_basename);

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static jdf_name_list_t *definition_is_parameter(const jdf_function_entry_t *f, const jdf_def_list_t *dl)
{
    jdf_name_list_t *pl;
    for( pl = f->parameters; pl != NULL; pl = pl->next )
        if( strcmp(pl->name, dl->name) == 0 )
            return pl;
    return NULL;
}

static void jdf_generate_hashfunction_for(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    string_arena_t *sa = string_arena_new(64);
    jdf_def_list_t *dl;
    expr_info_t info;
    int idx;

    (void)jdf;

    coutput("static inline uint64_t %s_hash(const __dague_%s_internal_handle_t *__dague_handle,\n"
            "                               const %s *assignments)\n"
            "{\n"
            "  uint64_t __h = 0;\n",
            f->fname, jdf_basename,
            dague_get_name(jdf, f, "assignment_t"));

    info.prefix = "";
    info.suffix = "";
    info.sa = sa;
    info.assignments = "assignments";

    idx = 0;
    for(dl = f->locals; dl != NULL; dl = dl->next) {
        string_arena_init(sa);

        coutput("  const int %s = assignments->%s.value;\n",
                dl->name, dl->name);

        if( definition_is_parameter(f, dl) != NULL ) {
            if( dl->expr->op == JDF_RANGE ) {
                coutput("  int %s_min = %s;\n", dl->name, dump_expr((void**)dl->expr->jdf_ta1, &info));
            } else {
                coutput("  int %s_min = %s;\n", dl->name, dump_expr((void**)dl->expr, &info));
            }
        } else {
            /* Hash functions should depend only on the parameters of the
             * function. However, we might need the other definitions because
             * the min expression of the parameters might depend on them. If
             * this is not the case, a quick "(void)" removes the warning.
             */
            coutput("  (void)%s;\n", dl->name);
        }
        idx++;
    }

    string_arena_init(sa);
    for(dl = f->locals; dl != NULL; dl = dl->next) {
        if( definition_is_parameter(f, dl) != NULL ) {
            coutput("  __h += (%s - %s_min)%s;\n", dl->name, dl->name, string_arena_get_string(sa));
            string_arena_add_string(sa, " * __dague_handle->%s_%s_range", f->fname, dl->name);
        }
    }

    coutput(" (void)__dague_handle; return __h;\n");
    coutput("}\n\n");
    string_arena_free(sa);
}

static void jdf_generate_hashfunctions(const jdf_t *jdf)
{
    jdf_function_entry_t *f;

    for(f = jdf->functions; f != NULL; f = f->next) {
        jdf_generate_hashfunction_for(jdf, f);
    }
}

/** Helper for sanity checker **/
char *malloc_and_dump_jdf_expr_list(const jdf_expr_t *el)
{
    char *res;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info;

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "assignments";

    UTIL_DUMP_LIST(sa, el, next,
                   dump_expr, (void*)&info, "", "", ", ", "");
    res = strdup( string_arena_get_string(sa) );
    string_arena_free(sa);
    string_arena_free(sa2);

    return res;
}

/** Code Generators */

static char *jdf_create_code_assignments_calls(string_arena_t *sa, int spaces,
                                               const jdf_t *jdf, const char *name, const jdf_call_t *call)
{
  const jdf_expr_t *el;
  expr_info_t infodst, infosrc;
  string_arena_t *sa2;
  jdf_expr_t *params = call->parameters;
  jdf_def_list_t *dl;
  jdf_name_list_t *pl;
  const jdf_function_entry_t *f;

  for(f = jdf->functions; f != NULL; f = f->next) {
      if(!strcmp(call->func_or_mem, f->fname))
          break;
  }

  assert(f != NULL);

  string_arena_init(sa);
  sa2 = string_arena_new(64);

  infodst.sa = sa2;
  infodst.prefix = f->fname;
  infodst.suffix = "";
  infodst.assignments = (char*)name;
  infosrc.sa = sa2;
  infosrc.prefix = "";
  infosrc.suffix = "";
  infosrc.assignments = "assignments";

  for(dl = f->locals; dl != NULL; dl = dl->next) {
      /* Is this definition a parameter or a value? */
      /* If it is a parameter, find the corresponding param in the call */
      for(el = params, pl = f->parameters; pl != NULL; el = el->next, pl = pl->next) {
          if( NULL == el ) {  /* Badly formulated call */
              string_arena_t *sa_caller, *sa_callee;
              expr_info_t caller;

              sa_caller = string_arena_new(64);
              sa_callee = string_arena_new(64);

              caller.sa = sa;
              caller.prefix = "";
              caller.suffix = "";
              caller.assignments = "";

              string_arena_init(sa);
              UTIL_DUMP_LIST_FIELD(sa_callee, f->parameters, next, name,
                                   dump_string, sa,
                                   "(", "", ", ", ")");
              string_arena_init(sa);
              UTIL_DUMP_LIST(sa_caller, params, next,
                             dump_expr, (void*)&caller,
                             "(", "", ", ", ")");
              fprintf(stderr, "%s.jdf:%d Badly formulated call %s%s instead of %s%s\n",
                      jdf_basename, call->super.lineno,
                      f->fname, string_arena_get_string(sa_caller),
                      f->fname, string_arena_get_string(sa_callee));
              exit(-1);
          }
          assert( el != NULL );
          if(!strcmp(pl->name, dl->name))
              break;
      }
      string_arena_init(sa2);
      /**
       * In addition to the locals in the context of the current task we need to
       * generate the target task "locals" otherwise the usage of expressions in
       * it's local will be broken. Moreover, we also need to generate the full
       * assignment context, including target locals, or the usage of %inline
       * will be broken.
       */
      if( NULL == pl ) {
          /* It is a value. Let's dump it's expression in the destination context */
          string_arena_add_string(sa,
                                  "%sconst int %s%s = %s->%s.value = %s; (void)%s%s;\n",
                                  indent(spaces), f->fname, dl->name, name, dl->name, dump_expr((void**)dl->expr, &infodst), f->fname, dl->name);
      } else {
          /* It is a parameter. Let's dump it's expression in the source context */
          assert(el != NULL);
          string_arena_add_string(sa,
                                  "%sconst int %s%s = %s->%s.value = %s; (void)%s%s;\n",
                                  indent(spaces), f->fname, dl->name, name, dl->name, dump_expr((void**)el, &infosrc), f->fname, dl->name);
      }
  }

  string_arena_free(sa2);

  return string_arena_get_string(sa);
}

static void
jdf_generate_code_call_initialization(const jdf_t *jdf, const jdf_call_t *call,
                                      const char *fname, const jdf_dataflow_t *f,
                                      const char *spaces)
{
    string_arena_t *sa, *sa2;
    expr_info_t info;
    const jdf_dataflow_t* tflow;
    const jdf_function_entry_t* targetf;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "assignments";

    if( call->var != NULL ) {
        /* Find the target function */
        for(targetf = jdf->functions; NULL != targetf; targetf = targetf->next)
            if( !strcmp(targetf->fname, call->func_or_mem) )
                break;
        if( NULL == targetf ) {
            jdf_fatal(JDF_OBJECT_LINENO(f),
                      "During code generation: unable to find the source function %s\n"
                      "required by function %s flow %s to satisfy INPUT dependency at line %d\n",
                      call->func_or_mem,
                      fname, call->var, JDF_OBJECT_LINENO(f));
            exit(1);
        }

        tflow = jdf_data_output_flow(jdf, call->func_or_mem, call->var);
        if( NULL == tflow ) {
            jdf_fatal(JDF_OBJECT_LINENO(f),
                      "During code generation: unable to find an output flow for variable %s in function %s,\n"
                      "which is requested by function %s to satisfy Input dependency at line %d\n",
                      call->var, call->func_or_mem,
                      fname, JDF_OBJECT_LINENO(f));
            exit(1);
        }
        coutput("%s *target_locals = (%s*)&generic_locals;\n",
                dague_get_name(jdf, targetf, "assignment_t"), dague_get_name(jdf, targetf, "assignment_t"));
        coutput("%s",  jdf_create_code_assignments_calls(sa, strlen(spaces)+1, jdf, "target_locals", call));

        coutput("%s    entry = data_repo_lookup_entry( %s_repo, %s_hash( __dague_handle, target_locals ));\n"
                "%s    chunk = entry->data[%d];  /* %s:%s <- %s:%s */\n",
                spaces, call->func_or_mem, call->func_or_mem,
                spaces, tflow->flow_index, f->varname, fname, call->var, call->func_or_mem);
        coutput("%s    ACQUIRE_FLOW(this_task, \"%s\", &%s_%s, \"%s\", target_locals, chunk);\n",
                spaces, f->varname, jdf_basename, call->func_or_mem, call->var);
    } else {
        coutput("%s    chunk = dague_data_get_copy(%s(%s), target_device);\n"
                "%s    OBJ_RETAIN(chunk);\n",
                spaces, call->func_or_mem,
                UTIL_DUMP_LIST(sa, call->parameters, next,
                               dump_expr, (void*)&info, "", "", ", ", ""),
                spaces);
    }

    string_arena_free(sa);
    string_arena_free(sa2);
}

/**
 * A pure output data. We need to allocate the data base on the output flow that
 * will be followed upon completion.
 */
static void jdf_generate_code_call_init_output(const jdf_t *jdf, const jdf_call_t *call,
                                               int lineno, const char *fname,
                                               const char *spaces, const char *arena, const char* count)
{
    int dataindex;

    if( (NULL != call) && (NULL != call->var) ) {
        dataindex = jdf_data_input_index(jdf, call->func_or_mem, call->var);
        if( dataindex < 0 ) {
            if( dataindex == -1 ) {
                jdf_fatal(lineno,
                          "During code generation: unable to find an input flow for variable %s in function %s,\n"
                          "which is requested by function %s to satisfy Output dependency at line %d\n",
                          call->var, call->func_or_mem,
                          fname, lineno);
                exit(1);
            } else {
                jdf_fatal(lineno,
                          "During code generation: unable to find function %s,\n"
                          "which is requested by function %s to satisfy Output dependency at line %d\n",
                          call->func_or_mem,
                          fname, lineno);
                exit(1);
            }
        }
    }
    coutput("%s    chunk = dague_arena_get_copy(%s, %s, target_device);\n",
            spaces, arena, count);
    return;
}

static void
create_arena_from_datatype(string_arena_t *sa,
                           jdf_datatransfer_type_t datatype)
{
    expr_info_t info;
    string_arena_t *sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    string_arena_add_string(sa, "__dague_handle->super.arenas[");
    if( JDF_CST == datatype.type->op ) {
        string_arena_add_string(sa, "%d", datatype.type->jdf_cst);
    } else if( (JDF_VAR == datatype.type->op) || (JDF_STRING == datatype.type->op) ) {
        string_arena_add_string(sa, "DAGUE_%s_%s_ARENA", jdf_basename, datatype.type->jdf_var);
    } else {
        string_arena_add_string(sa, "%s", dump_expr((void**)datatype.type, &info));
    }
    string_arena_add_string(sa, "]");
    string_arena_free(sa2);
}

static void jdf_generate_code_flow_initialization(const jdf_t *jdf,
                                                  const jdf_function_entry_t* f,
                                                  const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    expr_info_t info;
    string_arena_t *sa, *sa2, *sa_count;
    int cond_index = 0;
    char* condition[] = {"    if( %s ) {\n", "    else if( %s ) {\n"};

    if( JDF_FLOW_TYPE_CTL & flow->flow_flags ) {
        coutput("  /* %s is a control flow */\n"
                "  this_task->data.%s.data_in   = NULL;\n"
                "  this_task->data.%s.data_repo = NULL;\n",
                flow->varname,
                flow->varname,
                flow->varname);
        return;
    }
    coutput( "  if( NULL == (chunk = this_task->data.%s.data_in) ) {  /* flow %s */\n"
             "    entry = NULL;\n",
             flow->varname, flow->varname);

    sa  = string_arena_new(64);

    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "    &this_task->locals";

    if ( flow->flow_flags & JDF_FLOW_TYPE_READ ) {
        int check = 1;
        for(dl = flow->deps; dl != NULL; dl = dl->next) {
            if( dl->dep_flags & JDF_DEP_FLOW_OUT ) continue;

            check = 0;
            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                if( 0 != cond_index ) coutput("    else {\n");
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f->fname, flow,
                                                       (0 != cond_index ? "  " : "") );
                if( 0 != cond_index ) coutput("    }\n");
                goto done_with_input;
            case JDF_GUARD_BINARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f->fname, flow, "  " );
                coutput("    }\n");
                cond_index++;
                break;
            case JDF_GUARD_TERNARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f->fname, flow, "  " );
                coutput("    } else {\n");
                jdf_generate_code_call_initialization( jdf, dl->guard->callfalse, f->fname, flow, "  " );
                coutput("    }\n");
                goto done_with_input;
            }
        }
        if ( check ) {
            jdf_fatal(JDF_OBJECT_LINENO(flow),
                      "During code generation: unable to find an input flow for variable %s marked as RW or READ\n",
                      flow->varname );
        }
        goto done_with_input;
    }
    if ( flow->flow_flags & JDF_FLOW_TYPE_WRITE ) {
        sa2 = string_arena_new(64);
        sa_count = string_arena_new(64);
        for(dl = flow->deps; dl != NULL; dl = dl->next) {
            /* Special case for the arena definition for WRITE-only flows */
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
                assert(JDF_GUARD_UNCONDITIONAL == dl->guard->guard_type);
            } else {
                if ( !(dl->dep_flags & JDF_DEP_FLOW_OUT) ) {
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "During code generation: unable to find an output flow for variable %s marked as WRITE\n",
                              flow->varname );
                    break;
                }
            }

            string_arena_init(sa2);
            create_arena_from_datatype(sa2, dl->datatype);

            assert( dl->datatype.count != NULL );
            string_arena_init(sa_count);
            string_arena_add_string(sa_count, "%s", dump_expr((void**)dl->datatype.count, &info));


            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                if( 0 != cond_index ) coutput("    else {\n");
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, JDF_OBJECT_LINENO(flow), f->fname, "  ",
                                                   string_arena_get_string(sa2), string_arena_get_string(sa_count));
                if( 0 != cond_index ) coutput("    }\n");
                goto done_with_input;
            case JDF_GUARD_BINARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, JDF_OBJECT_LINENO(flow), f->fname, "  ",
                                                   string_arena_get_string(sa2), string_arena_get_string(sa_count));
                coutput("    }\n");
                cond_index++;
                break;
            case JDF_GUARD_TERNARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, JDF_OBJECT_LINENO(flow), f->fname, "  ",
                                                   string_arena_get_string(sa2), string_arena_get_string(sa_count));
                coutput("    } else {\n");
                jdf_generate_code_call_init_output(jdf, dl->guard->callfalse, JDF_OBJECT_LINENO(flow), f->fname, "  ",
                                                   string_arena_get_string(sa2), string_arena_get_string(sa_count));
                coutput("    }\n");
                goto done_with_input;
            }
        }
        string_arena_free(sa2);
        string_arena_free(sa_count);
    }
 done_with_input:
    coutput("      this_task->data.%s.data_in   = chunk;   /* flow %s */\n"
            "      this_task->data.%s.data_repo = entry;\n"
            "    }\n",
            flow->varname, flow->varname,
            flow->varname);
    {
        int has_output_deps = 0;
        for(dl = flow->deps; dl != NULL; dl = dl->next) {
            if ( dl->dep_flags & JDF_DEP_FLOW_OUT ) {
                has_output_deps = 1;
                break;
            }
        }
        if( has_output_deps ) {
            coutput("    /* Now get the local version of the data to be worked on */\n"
                    "    %sthis_task->data.%s.data_out = dague_data_get_copy(chunk->original, target_device);\n\n",
                    (flow->flow_flags & JDF_FLOW_TYPE_WRITE ? "if( NULL != chunk )\n  " : ""),
                    flow->varname);
        } else
            coutput("    this_task->data.%s.data_out = NULL;  /* input only */\n\n", flow->varname);
    }
    string_arena_free(sa);
}

static void jdf_generate_code_call_final_write(const jdf_t *jdf, const jdf_call_t *call,
                                               jdf_datatransfer_type_t datatype,
                                               const char *spaces,
                                               const jdf_dataflow_t *flow)
{
    string_arena_t *sa, *sa2, *sa3, *sa4;
    expr_info_t info;

    (void)jdf;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);
    sa4 = string_arena_new(64);

    if( call->var == NULL ) {
        info.sa = sa2;
        info.prefix = "";
        info.suffix = "";
        info.assignments = "&this_task->locals";

        UTIL_DUMP_LIST(sa, call->parameters, next,
                       dump_expr, (void*)&info, "", "", ", ", "");

        string_arena_init(sa2);
        string_arena_add_string(sa3, "%s", dump_expr((void**)datatype.count, &info));
        string_arena_add_string(sa4, "%s", dump_expr((void**)datatype.displ, &info));

        string_arena_init(sa2);
        create_arena_from_datatype(sa2, datatype);
        coutput("%s  if( this_task->data.%s.data_out->original != %s(%s) ) {\n"
                "%s    dague_dep_data_description_t data;\n"
                "%s    data.data   = this_task->data.%s.data_out;\n"
                "%s    data.arena  = %s;\n"
                "%s    data.layout = data.arena->opaque_dtt;\n"
                "%s    data.count  = %s;\n"
                "%s    data.displ  = %s;\n"
                "%s    assert( data.count > 0 );\n"
                "%s    dague_remote_dep_memcpy(context, this_task->dague_handle,\n"
                "%s                            dague_data_get_copy(%s(%s), 0),\n"
                "%s                            this_task->data.%s.data_out, &data);\n"
                "%s  }\n",
                spaces, flow->varname, call->func_or_mem, string_arena_get_string(sa),
                spaces,
                spaces, flow->varname,
                spaces, string_arena_get_string(sa2),
                spaces,
                spaces, string_arena_get_string(sa3),
                spaces, string_arena_get_string(sa4),
                spaces,
                spaces,
                spaces, call->func_or_mem, string_arena_get_string(sa),
                spaces, flow->varname,
                spaces);
    }

    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);
}

static void
jdf_generate_code_flow_final_writes(const jdf_t *jdf,
                                    const jdf_function_entry_t* f,
                                    const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    expr_info_t info;
    string_arena_t *sa;

    (void)jdf; (void)f;
    sa = string_arena_new(64);
    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "assignments";

    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( dl->dep_flags & JDF_DEP_FLOW_IN )
            /** No final write for input-only flows */
            continue;

        switch( dl->guard->guard_type ) {
        case JDF_GUARD_UNCONDITIONAL:
            if( dl->guard->calltrue->var == NULL ) {
                jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype, "", flow );
            }
            break;
        case JDF_GUARD_BINARY:
            if( dl->guard->calltrue->var == NULL ) {
                coutput("  if( %s ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype, "  ", flow );
                coutput("  }\n");
            }
            break;
        case JDF_GUARD_TERNARY:
            if( dl->guard->calltrue->var == NULL ) {
                coutput("  if( %s ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype, "  ", flow );
                if( dl->guard->callfalse->var == NULL ) {
                    coutput("  } else {\n");
                    jdf_generate_code_call_final_write( jdf, dl->guard->callfalse, dl->datatype, "  ", flow );
                }
                coutput("  }\n");
            } else if ( dl->guard->callfalse->var == NULL ) {
                coutput("  if( !(%s) ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, dl->guard->callfalse, dl->datatype, "  ", flow );
                coutput("  }\n");
            }
            break;
        }
    }

    string_arena_free(sa);
}

static void jdf_generate_code_dry_run_before(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    (void)jdf;
    (void)f;

    coutput("\n\n#if !defined(DAGUE_PROF_DRY_BODY)\n\n");
}

static void jdf_generate_code_dry_run_after(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    (void)jdf;
    (void)f;

    coutput("\n\n#endif /*!defined(DAGUE_PROF_DRY_BODY)*/\n\n");
}

static void jdf_generate_code_grapher_task_done(const jdf_t *jdf, const jdf_function_entry_t *f, const char* context_name)
{
    (void)jdf;

    coutput("#if defined(DAGUE_PROF_GRAPHER)\n"
            "  dague_prof_grapher_task(%s, context->th_id, context->virtual_process->vp_id, %s_hash(__dague_handle, %s->locals));\n"
            "#endif  /* defined(DAGUE_PROF_GRAPHER) */\n",
            context_name, f->fname, context_name);
}

static void jdf_generate_code_cache_awareness_update(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    string_arena_t *sa;
    sa = string_arena_new(64);

    (void)jdf;
    UTIL_DUMP_LIST(sa, f->dataflow, next,
                   dump_dataflow_varname, NULL,
                   "", "  cache_buf_referenced(context->closest_cache, ", ");\n", "");
    if( strlen(string_arena_get_string(sa)) ) {
            coutput("  /** Cache Awareness Accounting */\n"
                    "#if defined(DAGUE_CACHE_AWARENESS)\n"
                    "%s);\n"
                    "#endif /* DAGUE_CACHE_AWARENESS */\n",
                    string_arena_get_string(sa));
    }
    string_arena_free(sa);
}

static void jdf_generate_code_call_release_dependencies(const jdf_t *jdf,
                                                        const jdf_function_entry_t *function,
                                                        const char* context_name)
{
    uint32_t complete_mask = 0;
    jdf_dataflow_t* dl;
    (void)jdf;

    for( dl = function->dataflow; dl != NULL; dl = dl->next ) {
        complete_mask |= dl->flow_dep_mask_out;
    }
    coutput("  release_deps_of_%s_%s(context, %s,\n"
            "      DAGUE_ACTION_RELEASE_REMOTE_DEPS |\n"
            "      DAGUE_ACTION_RELEASE_LOCAL_DEPS |\n"
            "      DAGUE_ACTION_RELEASE_LOCAL_REFS |\n"
            "      0x%x,  /* mask of all dep_index */ \n"
            "      NULL);\n",
            jdf_basename, function->fname, context_name, complete_mask);
}

static void
jdf_generate_code_data_lookup(const jdf_t *jdf,
                              const jdf_function_entry_t *f,
                              const char *name)
{
    string_arena_t *sa, *sa2;
    assignment_info_t ai;
    jdf_dataflow_t *fl;

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);

    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("static int %s(dague_execution_unit_t *context, %s *this_task)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle;\n"
            "  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */\n"
            "  int target_device = 0; (void)target_device;\n"
            "  (void)__dague_handle; (void)generic_locals; (void)context;\n"
            "  dague_data_copy_t *chunk = NULL;\n"
            "  data_repo_entry_t *entry = NULL;\n"
            "%s",
            name, dague_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    UTIL_DUMP_LIST(sa, f->dataflow, next,
                   dump_data_declaration, sa2, "", "", "", "");
    coutput("  /** Lookup the input data, and store them in the context if any */\n");
    for( fl = f->dataflow; fl != NULL; fl = fl->next ) {
        jdf_generate_code_flow_initialization(jdf, f, fl);
    }

    /* If the function has the property profile turned off do not generate the profiling code */
    if( jdf_property_get_int(f->properties, "profile", 1) ) {
        string_arena_t *sa3 = string_arena_new(64);
        expr_info_t linfo;

        linfo.prefix = "";
        linfo.suffix = "";
        linfo.sa = sa2;
        linfo.assignments = "&this_task->locals";

        coutput("  /** Generate profiling information */\n"
                "#if defined(DAGUE_PROF_TRACE)\n"
                "  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.%s;\n"
                "  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.%s))->data_key((dague_ddesc_t*)__dague_handle->super.%s, %s);\n"
                "#endif  /* defined(DAGUE_PROF_TRACE) */\n",
                f->predicate->func_or_mem,
                f->predicate->func_or_mem, f->predicate->func_or_mem,
                UTIL_DUMP_LIST(sa3, f->predicate->parameters, next,
                               dump_expr, (void*)&linfo,
                               "", "", ", ", "") );
        string_arena_free(sa3);
    } else {
        coutput("  /** No profiling information */\n");
    }
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", "; (void)chunk; (void)entry;\n"));

    coutput("  return DAGUE_HOOK_RETURN_DONE;\n"
            "}\n\n");
    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_code_hook(const jdf_t *jdf,
                                   const jdf_function_entry_t *f,
                                   const jdf_body_t* body,
                                   const char *name)
{
    jdf_def_list_t* type_property;
    string_arena_t *sa, *sa2;
    assignment_info_t ai;
    jdf_dataflow_t *fl;
    int di;
    int profile_on;
    char* output;

    profile_on = jdf_property_get_int(f->properties, "profile", 1);
    profile_on = jdf_property_get_int(body->properties, "profile", profile_on);

    jdf_find_property(body->properties, "type", &type_property);
    if(NULL != type_property) {
        if(JDF_VAR != type_property->expr->op) {
            expr_info_t ei;

            ei.sa = string_arena_new(64);
            ei.prefix = "";
            ei.suffix = "";
            ei.assignments = NULL;

            jdf_fatal(body->super.lineno,
                      "Type property set to unknown value for function %s in file %s:%d\n"
                      "Currently set to [%s]<%d>\n",
                      f->fname, body->super.filename, body->super.lineno,
                      dump_expr((void**)type_property->expr, (void*)&ei), type_property->expr->op);
            string_arena_free(ei.sa);
            exit(1);
        }
    }
    if( NULL != type_property)
        coutput("#if defined(HAVE_%s)\n", type_property->expr->jdf_var);

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    if(NULL == type_property)
        coutput("static int %s(dague_execution_unit_t *context, %s *this_task)\n",
                name, dague_get_name(jdf, f, "task_t"));
    else
        coutput("static int %s_%s(dague_execution_unit_t *context, %s *this_task)\n",
                name, type_property->expr->jdf_var, dague_get_name(jdf, f, "task_t"));

    coutput("{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle;\n"
            "  assignment_t tass[MAX_PARAM_COUNT];  /* generic locals */\n"
            "  (void)context; (void)__dague_handle; (void)tass;\n"
            "%s",
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    output = UTIL_DUMP_LIST(sa, f->dataflow, next,
                            dump_data_initialization_from_data_array, sa2, "", "", "", "");
    if( 0 != strlen(output) ) {
        coutput("  /** Declare the variables that will hold the data, and all the accounting for each */\n"
                "%s\n",
                output);
    }

    /**
     * Generate code for the simulation.
     */
    coutput("  /** Update starting simulation date */\n"
            "#if defined(DAGUE_SIM)\n"
            "  {\n"
            "    this_task->sim_exec_date = 0;\n");
    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {

        if(fl->flow_flags & JDF_FLOW_TYPE_CTL) continue;  /* control flow, nothing to store */

        coutput("    data_repo_entry_t *e%s = this_task->data.%s.data_repo;\n"
                "    if( (NULL != e%s) && (e%s->sim_exec_date > this_task->sim_exec_date) )\n"
                "      this_task->sim_exec_date = e%s->sim_exec_date;\n",
                fl->varname, fl->varname,
                fl->varname, fl->varname,
                fl->varname);
    }
    coutput("    if( this_task->function->sim_cost_fct != NULL ) {\n"
            "      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);\n"
            "    }\n"
            "    if( context->largest_simulation_date < this_task->sim_exec_date )\n"
            "      context->largest_simulation_date = this_task->sim_exec_date;\n"
            "  }\n"
            "#endif\n");

    jdf_generate_code_cache_awareness_update(jdf, f);

    jdf_generate_code_dry_run_before(jdf, f);
    jdf_coutput_prettycomment('-', "%s BODY", f->fname);

    if( profile_on ) {
        coutput("  DAGUE_TASK_PROF_TRACE(context->eu_profile,\n"
                "                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],\n"
                "                        this_task);\n");
    }

    coutput("%s\n", body->external_code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }
    jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname);
    jdf_generate_code_dry_run_after(jdf, f);
    coutput("  return DAGUE_HOOK_RETURN_DONE;\n"
            "}\n");

    if( NULL != type_property)
        coutput("#endif  /*  defined(HAVE_%s) */\n", type_property->expr->jdf_var);

    string_arena_free(sa);
    string_arena_free(sa2);
}

static void
jdf_generate_code_complete_hook(const jdf_t *jdf,
                                const jdf_function_entry_t *f,
                                const char *name)
{
    string_arena_t *sa, *sa2;
    int di;
    int profile_on;
    jdf_dataflow_t *fl;
    assignment_info_t ai;

    profile_on = jdf_property_get_int(f->properties, "profile", 1);

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("static int complete_%s(dague_execution_unit_t *context, %s *this_task)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle;\n"
            "#if defined(DISTRIBUTED)\n"
            "  %s"
            "#endif  /* defined(DISTRIBUTED) */\n"
            "  (void)context; (void)__dague_handle;\n",
            name, dague_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {
        if(JDF_FLOW_TYPE_CTL & fl->flow_flags) continue;
        if(fl->flow_flags & JDF_FLOW_TYPE_WRITE) {
            if(fl->flow_flags & JDF_FLOW_TYPE_READ)
                coutput("this_task->data.%s.data_out->version++;  /* %s */\n", fl->varname, fl->varname);
            else
                coutput("if( NULL !=  this_task->data.%s.data_out) this_task->data.%s.data_out->version++;\n",
                        fl->varname, fl->varname);
        }
    }

    if( profile_on ) {
        coutput("  DAGUE_TASK_PROF_TRACE(context->eu_profile,\n"
                "                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],\n"
                "                        this_task);\n");
    }

    coutput("#if defined(DISTRIBUTED)\n"
            "  /** If not working on distributed, there is no risk that data is not in place */\n");
    for( fl = f->dataflow; fl != NULL; fl = fl->next ) {
        jdf_generate_code_flow_final_writes(jdf, f, fl);
    }
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));
    coutput("#endif /* DISTRIBUTED */\n");

    jdf_generate_code_grapher_task_done(jdf, f, "this_task");

    jdf_generate_code_call_release_dependencies(jdf, f, "this_task");

    coutput("  return 0;\n"
            "}\n\n");
    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_code_hooks(const jdf_t *jdf,
                                    const jdf_function_entry_t *f,
                                    const char *name)
{
    jdf_body_t* body = f->bodies;
    do {
        jdf_generate_code_hook(jdf, f, body, name);
        body = body->next;
    } while (NULL != body);
    jdf_generate_code_complete_hook(jdf, f, name);
}

static void jdf_generate_code_free_hash_table_entry(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    jdf_dataflow_t *dl;
    jdf_dep_t *dep;
    expr_info_t info;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa_local = string_arena_new(64);
    string_arena_t *sa_code = string_arena_new(64);
    int cond_index, need_locals = 0;
    char* condition[] = {"    if( %s ) {\n", "    else if( %s ) {\n"};
    assignment_info_t ai;

    ai.sa = sa;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;

    UTIL_DUMP_LIST(sa_local, f->locals, next,
                   dump_local_assignments, &ai, "", "    ", "\n", "\n");
    /* Quiet the unused variable warnings */
    UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                         dump_string, NULL, "   ", " (void)", ";", ";\n");
    string_arena_add_string(sa_local, "\n%s\n", string_arena_get_string(sa1));
    coutput("  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {\n");

    info.prefix = "";
    info.suffix = "";
    info.sa = sa1;
    info.assignments = "&this_task->locals";

    for( dl = f->dataflow; dl != NULL; dl = dl->next ) {
        if( dl->flow_flags & JDF_FLOW_TYPE_CTL ) continue;
        cond_index = 0;

        if( dl->flow_flags & JDF_FLOW_TYPE_READ ) {
            for( dep = dl->deps; dep != NULL; dep = dep->next ) {
                if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
                    switch( dep->guard->guard_type ) {
                    case JDF_GUARD_UNCONDITIONAL:
                        if( NULL != dep->guard->calltrue->var ) {  /* this is a dataflow not a data access */
                            if( 0 != cond_index ) string_arena_add_string(sa_code, "    else {\n");
                            string_arena_add_string(sa_code, "    data_repo_entry_used_once( eu, %s_repo, this_task->data.%s.data_repo->key );\n",
                                                    dep->guard->calltrue->func_or_mem, dl->varname);
                            if( 0 != cond_index ) string_arena_add_string(sa_code, "    }\n");
                        }
                        goto next_dependency;
                    case JDF_GUARD_BINARY:
                        string_arena_add_string(sa_code, (0 == cond_index ? condition[0] : condition[1]),
                                                dump_expr((void**)dep->guard->guard, &info));
                        need_locals++;
                        if( NULL != dep->guard->calltrue->var ) {   /* this is a dataflow not a data access */
                            string_arena_add_string(sa_code, "      data_repo_entry_used_once( eu, %s_repo, this_task->data.%s.data_repo->key );\n",
                                                    dep->guard->calltrue->func_or_mem, dl->varname);
                        }
                        string_arena_add_string(sa_code, "    }\n");
                        cond_index++;
                        break;
                    case JDF_GUARD_TERNARY:
                        need_locals++;
                        string_arena_add_string(sa_code, (0 == cond_index ? condition[0] : condition[1]),
                                                dump_expr((void**)dep->guard->guard, &info));
                        if( NULL != dep->guard->calltrue->var ) {    /* this is a dataflow not a data access */
                            string_arena_add_string(sa_code, "      data_repo_entry_used_once( eu, %s_repo, this_task->data.%s.data_repo->key );\n",
                                                    dep->guard->calltrue->func_or_mem, dl->varname);
                        }
                        string_arena_add_string(sa_code, "    } else {\n");
                        if( NULL != dep->guard->callfalse->var ) {    /* this is a dataflow not a data access */
                            string_arena_add_string(sa_code,
                                                    "      data_repo_entry_used_once( eu, %s_repo, this_task->data.%s.data_repo->key );\n",
                                                    dep->guard->callfalse->func_or_mem, dl->varname);
                        }
                        string_arena_add_string(sa_code, "    }\n");
                        goto next_dependency;
                    }
                }
            }
        }

    next_dependency:
        if( dl->flow_flags & (JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE) ) {
            if( !(dl->flow_flags & JDF_FLOW_TYPE_READ) )
                string_arena_add_string(sa_code, "    if(NULL != this_task->data.%s.data_in)\n", dl->varname);
            if(need_locals) {
                coutput("%s", string_arena_get_string(sa_local));
                string_arena_init(sa_local);  /* reset the sa_local */
            }
            coutput("%s", string_arena_get_string(sa_code));
            string_arena_init(sa_code);
            coutput("    DAGUE_DATA_COPY_RELEASE(this_task->data.%s.data_in);\n", dl->varname);
        }
        (void)jdf;  /* just to keep the compilers happy regarding the goto to an empty statement */
    }
    coutput("  }\n");

    string_arena_free(sa);
    string_arena_free(sa1);
    string_arena_free(sa_local);
    string_arena_free(sa_code);
}

static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    coutput("static int %s(dague_execution_unit_t *eu, %s *this_task, uint32_t action_mask, dague_remote_deps_t *deps)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t *)this_task->dague_handle;\n"
            "  dague_release_dep_fct_arg_t arg;\n"
            "  int __vp_id;\n"
            "  arg.action_mask = action_mask;\n"
            "  arg.output_usage = 0;\n"
            "  arg.output_entry = NULL;\n"
            "#if defined(DISTRIBUTED)\n"
            "  arg.remote_deps = deps;\n"
            "#endif  /* defined(DISTRIBUTED) */\n"
            "  assert(NULL != eu);\n"
            "  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);\n"
            "  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );\n"
            "  (void)__dague_handle; (void)deps;\n",
            name, dague_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename);

    coutput("  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {\n"
            "    arg.output_entry = data_repo_lookup_entry_and_create( eu, %s_repo, %s_hash(__dague_handle, (%s*)(&this_task->locals)) );\n"
            "    arg.output_entry->generator = (void*)this_task;  /* for AYU */\n"
            "#if defined(DAGUE_SIM)\n"
            "    assert(arg.output_entry->sim_exec_date == 0);\n"
            "    arg.output_entry->sim_exec_date = this_task->sim_exec_date;\n"
            "#endif\n"
            "  }\n",
            f->fname, f->fname, dague_get_name(jdf, f, "assignment_t"));

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) ) {
        coutput("  iterate_successors_of_%s_%s(eu, this_task, action_mask, dague_release_dep_fct, &arg);\n"
                "\n",
                jdf_basename, f->fname);

        coutput("#if defined(DISTRIBUTED)\n"
                "  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {\n"
                "    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);\n"
                "  }\n"
                "#endif\n"
                "\n");
    }
    coutput("  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {\n"
            "    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;\n");
    coutput("    data_repo_entry_addto_usage_limit(%s_repo, arg.output_entry->key, arg.output_usage);\n",
            f->fname);
    coutput("    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {\n"
            "      if( NULL == arg.ready_lists[__vp_id] ) continue;\n"
            "      if(__vp_id == eu->virtual_process->vp_id) {\n"
            "        __dague_schedule(eu, arg.ready_lists[__vp_id]);\n"
            "      } else {\n"
            "        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);\n"
            "      }\n"
            "      arg.ready_lists[__vp_id] = NULL;\n"
            "    }\n"
            "  }\n");

    jdf_generate_code_free_hash_table_entry(jdf, f);

    coutput(
        "  return 0;\n"
        "}\n"
        "\n");
}

static char *jdf_dump_context_assignment(string_arena_t *sa_open,
                                         const jdf_t *jdf,
                                         const jdf_dataflow_t *flow,
                                         const char *calltext,
                                         const jdf_call_t *call,
                                         int lineno,
                                         const char *prefix,
                                         const char *var)
{
    jdf_def_list_t *def;
    const jdf_function_entry_t *targetf;
    jdf_expr_t *el;
    jdf_name_list_t *nl;
    expr_info_t local_info, dest_info;
    string_arena_t *sa2, *sa1, *sa_close;
    int i, nbopen;
    int nbparam_given, nbparam_required;

    string_arena_init(sa_open);

    /* Find the target function */
    for(targetf = jdf->functions; targetf != NULL; targetf = targetf->next)
        if( !strcmp(targetf->fname, call->func_or_mem) )
            break;

    if( NULL == targetf ) {
        jdf_fatal(lineno,
                  "During code generation: unable to find function %s referenced in this call.\n",
                  call->func_or_mem);
        exit(1);
    }

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    dest_info.prefix      = "ncc->locals.";
    dest_info.suffix      = ".value";
    dest_info.sa          = sa1;
    dest_info.assignments = "&ncc->locals";

    local_info.prefix      = "";
    local_info.suffix      = "";
    local_info.sa          = sa2;
    local_info.assignments = "&this_task->locals";

    sa_close = string_arena_new(64);

    nbopen = 0;

    string_arena_add_string(sa_open, "%s%s%s* ncc = (%s*)&%s;\n",
                            prefix, indent(nbopen), dague_get_name(jdf, targetf, "task_t"), dague_get_name(jdf, targetf, "task_t"), var);
    string_arena_add_string(sa_open, "%s%s%s.function = __dague_handle->super.super.functions_array[%s_%s.function_id];\n",
                            prefix, indent(nbopen), var, jdf_basename, targetf->fname);

    nbparam_given = 0;
    for(el = call->parameters; el != NULL; el = el->next) {
        nbparam_given++;
    }

    nbparam_required = 0;
    for(nl = targetf->parameters; nl != NULL; nl = nl->next) {
        nbparam_required++;
    }

    if( nbparam_given != nbparam_required ) {
        fprintf(stderr,
                "Internal Error: Wrong number of arguments when calling %s at line %d (%d instead of %d)\n",
                targetf->fname, JDF_OBJECT_LINENO(flow), nbparam_given, nbparam_required );
        assert( nbparam_given == nbparam_required );
    }

    for(def = targetf->locals, i = 0;
        def != NULL;
        def = def->next, i++) {

        for(el  = call->parameters, nl = targetf->parameters;
            nl != NULL;
            nl = nl->next, el = el->next) {
            assert(el != NULL);
            if( !strcmp(nl->name, def->name) )
                break;
        }

        if( NULL == nl ) {
            /* This definition is not a parameter: just dump it's computation. */
            /**
             * If we have to execute code possibly comming from the user then we need to instantiate
             * the entire stack of the target function, including the local variables.
             */
            assert(el == NULL);
            string_arena_add_string(sa_open,
                                    "%s%s  const int %s_%s = %s;\n",
                                    prefix, indent(nbopen), targetf->fname, def->name, dump_expr((void**)def->expr, &dest_info));
            string_arena_add_string(sa_open, "%s%s  assert(&%s.locals[%d].value == &ncc->locals.%s.value);\n",
                                    prefix, indent(nbopen), var, i, def->name);
            string_arena_add_string(sa_open, "%s%s  ncc->locals.%s.value = %s_%s;\n",
                                    prefix, indent(nbopen), def->name,
                                    targetf->fname, def->name);
        } else {
            /* This definition is a parameter */
            assert(el != NULL);
            if( el->op == JDF_RANGE ) {
                string_arena_add_string(sa_open,
                                        "%s%s  int %s_%s;\n",
                                        prefix, indent(nbopen), targetf->fname, nl->name);

                string_arena_add_string(sa_open,
                                        "%s%sfor( %s_%s = %s;",
                                        prefix, indent(nbopen), targetf->fname, nl->name, dump_expr((void**)el->jdf_ta1, &local_info));
                string_arena_add_string(sa_open, "%s_%s <= %s; %s_%s+=",
                                        targetf->fname, nl->name, dump_expr((void**)el->jdf_ta2, &local_info), targetf->fname, nl->name);
                string_arena_add_string(sa_open, "%s) {\n",
                                        dump_expr((void**)el->jdf_ta3, &local_info));
                nbopen++;
            } else {
                string_arena_add_string(sa_open,
                                        "%s%s  const int %s_%s = %s;\n",
                                        prefix, indent(nbopen), targetf->fname, nl->name, dump_expr((void**)el, &local_info));
            }

            if( def->expr->op == JDF_RANGE ) {
                string_arena_add_string(sa_open,
                                        "%s%s  if( (%s_%s >= (%s))",
                                        prefix, indent(nbopen), targetf->fname, nl->name,
                                        dump_expr((void**)def->expr->jdf_ta1, &dest_info));
                string_arena_add_string(sa_open, " && (%s_%s <= (%s)) ) {\n",
                                        targetf->fname, nl->name,
                                        dump_expr((void**)def->expr->jdf_ta2, &dest_info));
                nbopen++;
            } else {
                string_arena_add_string(sa_open,
                                        "%s%s  if( (%s_%s == (%s)) ) {\n",
                                        prefix, indent(nbopen), targetf->fname, nl->name,
                                        dump_expr((void**)def->expr, &dest_info));
                nbopen++;
            }

            string_arena_add_string(sa_open, "%s%s  assert(&%s.locals[%d].value == &ncc->locals.%s.value);\n",
                                    prefix, indent(nbopen), var, i, nl->name);
            string_arena_add_string(sa_open,
                                    "%s%s  ncc->locals.%s.value = %s_%s;\n",
                                    prefix, indent(nbopen), nl->name,
                                    targetf->fname, nl->name);
        }
    }

    string_arena_add_string(sa_open,
                            "#if defined(DISTRIBUTED)\n"
                            "%s%s  rank_dst = ((dague_ddesc_t*)__dague_handle->super.%s)->rank_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n",
                            prefix, indent(nbopen), targetf->predicate->func_or_mem, targetf->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, targetf->predicate->parameters, next,
                                           dump_expr, (void*)&dest_info,
                                           "", "", ", ", ""));
    string_arena_add_string(sa_open,
                            "%s%s  if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )\n"
                            "#endif /* DISTRIBUTED */\n"
                            "%s%s    vpid_dst = ((dague_ddesc_t*)__dague_handle->super.%s)->vpid_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n",
                            prefix, indent(nbopen),
                            prefix, indent(nbopen), targetf->predicate->func_or_mem, targetf->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, targetf->predicate->parameters, next,
                                           dump_expr, (void*)&dest_info,
                                           "", "", ", ", ""));

    if( NULL != targetf->priority ) {
        string_arena_add_string(sa_open,
                                "%s%s  %s.priority = __dague_handle->super.super.priority + priority_of_%s_%s_as_expr_fct(__dague_handle, &ncc->locals);\n",
                                prefix, indent(nbopen), var, jdf_basename, targetf->fname);
    } else {
        string_arena_add_string(sa_open, "%s%s  %s.priority = __dague_handle->super.super.priority;\n",
                                prefix, indent(nbopen), var);
    }

    string_arena_add_string(sa_open,
                            "%s%sRELEASE_DEP_OUTPUT(eu, \"%s\", this_task, \"%s\", &%s, rank_src, rank_dst, &data);\n",
                            prefix, indent(nbopen), flow->varname, call->var, var);
    dest_info.assignments = NULL;
    dest_info.prefix = NULL;

    string_arena_add_string(sa_open,
                            "%s%s%s", prefix, indent(nbopen), calltext);

    for(i = nbopen; i > 0; i--) {
        string_arena_add_string(sa_close, "%s%s  }\n", prefix, indent(nbopen));
        nbopen--;
    }
    string_arena_add_string(sa_open, "%s", string_arena_get_string(sa_close));

    string_arena_free(sa_close);
    string_arena_free(sa2);

    return string_arena_get_string(sa_open);
}

/**
 * If this function has no predecessors or successors (depending on the
 * parameters), tag it as such. This will prevent us from generating useless
 * code.
 */
static void jdf_check_relatives( jdf_function_entry_t *f, jdf_dep_flags_t flow_type, jdf_flags_t flag)
{
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;

    for(fl = f->dataflow; fl != NULL; fl = fl->next) {
        for(dl = fl->deps; dl != NULL; dl = dl->next) {
            if( !(dl->dep_flags & flow_type) ) continue;

            /* Skip the default type declaration for WRITE-only dependencies */
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) )
                continue;

            if( (NULL != dl->guard->calltrue->var) ||
                ((JDF_GUARD_TERNARY == dl->guard->guard_type) &&
                 (NULL != dl->guard->callfalse->var)) ) {
                return;  /* we do have a relative of type flow_type */
            }
        }
    }
    /* We don't have a relative of type flow_type, let's tag it with flag */
    f->flags |= flag;
}

#define OUTPUT_PREV_DEPS(MASK, SA_DATATYPE, SA_DEPS)                    \
    if( strlen(string_arena_get_string((SA_DEPS))) ) {                  \
        if( strlen(string_arena_get_string((SA_DATATYPE))) ) {          \
            string_arena_add_string(sa_coutput,                         \
                                    "  %s",                             \
                                    string_arena_get_string((SA_DATATYPE))); \
        }                                                               \
        if( (JDF_DEP_FLOW_OUT & flow_type) && fl->flow_dep_mask_out == (MASK) ) { \
            string_arena_add_string(sa_coutput,                         \
                                    "  %s",                             \
                                    string_arena_get_string((SA_DEPS))); \
        } else {                                                        \
            string_arena_add_string(sa_coutput,                         \
                                    "  if( action_mask & 0x%x ) {\n"    \
                                    "    %s"                            \
                                    "  }\n",                            \
                                    MASK, string_arena_get_string((SA_DEPS))); \
        }                                                               \
        string_arena_init((SA_DEPS));                                   \
        string_arena_init((SA_DATATYPE));                               \
    }

static void
jdf_generate_code_iterate_successors_or_predecessors(const jdf_t *jdf,
                                                     const jdf_function_entry_t *f,
                                                     const char *name,
                                                     jdf_dep_flags_t flow_type)
{
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;
    int flowempty, flowtomem;
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa_ontask     = string_arena_new(64);
    string_arena_t *sa_coutput    = string_arena_new(1024);
    string_arena_t *sa_deps       = string_arena_new(1024);
    string_arena_t *sa_datatype   = string_arena_new(1024);
    string_arena_t *sa_type       = string_arena_new(256);
    string_arena_t *sa_tmp_type   = string_arena_new(256);
    string_arena_t *sa_nbelt      = string_arena_new(256);
    string_arena_t *sa_tmp_nbelt  = string_arena_new(256);
    string_arena_t *sa_displ      = string_arena_new(256);
    string_arena_t *sa_tmp_displ  = string_arena_new(256);
    string_arena_t *sa_layout     = string_arena_new(256);
    string_arena_t *sa_tmp_layout = string_arena_new(256);
    string_arena_t *sa_temp       = string_arena_new(1024);
    int depnb, last_datatype_idx;
    assignment_info_t ai;
    expr_info_t info;

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("static void\n"
            "%s(dague_execution_unit_t *eu, const %s *this_task,\n"
            "               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)this_task->dague_handle;\n"
            "  dague_execution_context_t nc;  /* generic placeholder for locals */\n"
            "  dague_dep_data_description_t data;\n"
            "  int vpid_dst = -1, rank_src = 0, rank_dst = 0;\n"
            "%s"
            "  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;\n",
            name, dague_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));
    coutput("%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    coutput("  nc.dague_handle = this_task->dague_handle;\n"
            "  nc.priority     = this_task->priority;\n"
            "  nc.chore_id     = 0;\n");
    coutput("#if defined(DISTRIBUTED)\n"
            "  rank_src = ((dague_ddesc_t*)__dague_handle->super.%s)->rank_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n"
            "#endif\n",
            f->predicate->func_or_mem, f->predicate->func_or_mem,
            UTIL_DUMP_LIST(sa1, f->predicate->parameters, next,
                           dump_expr, (void*)&info,
                           "", "", ", ", ""));

    for(fl = f->dataflow; fl != NULL; fl = fl->next) {
        flowempty = 1;
        flowtomem = 0;
        depnb = 0;
        last_datatype_idx = -1;
        string_arena_init(sa_coutput);
        string_arena_init(sa_deps);
        string_arena_init(sa_datatype);
        string_arena_init(sa_type);
        string_arena_init(sa_nbelt);
        string_arena_init(sa_displ);
        string_arena_init(sa_layout);

        string_arena_add_string(sa_coutput, "    data.data   = this_task->data.%s.data_out;\n", fl->varname);

        for(dl = fl->deps; dl != NULL; dl = dl->next) {
            if( !(dl->dep_flags & flow_type) ) continue;
            /* Special case for the arena definition for WRITE-only flows */
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) )
                continue;

            string_arena_init(sa_tmp_type);
            string_arena_init(sa_tmp_nbelt);
            string_arena_init(sa_tmp_layout);
            string_arena_init(sa_tmp_displ);
            if( JDF_FLOW_TYPE_CTL & fl->flow_flags ) {
                string_arena_add_string(sa_tmp_type, "NULL");
                string_arena_add_string(sa_tmp_nbelt, "  /* Control: always empty */ 0");
                string_arena_add_string(sa_tmp_layout, "DAGUE_DATATYPE_NULL");
                string_arena_add_string(sa_tmp_displ, "0");
            } else {
                create_arena_from_datatype(sa_tmp_type, dl->datatype);

                assert( dl->datatype.count != NULL );
                string_arena_add_string(sa_tmp_nbelt, "%s", dump_expr((void**)dl->datatype.count, &info));
                if( dl->datatype.layout == dl->datatype.type ) { /* no specific layout */
                    string_arena_add_string(sa_tmp_layout, "data.arena->opaque_dtt");
                } else {
                    string_arena_add_string(sa_tmp_layout, "%s", dump_expr((void**)dl->datatype.layout, &info));
                }
                string_arena_add_string(sa_tmp_displ, "%s", dump_expr((void**)dl->datatype.displ, &info));
            }

            if( last_datatype_idx != dl->dep_datatype_index ) {
                /* Prepare the memory layout of the output dependency. */
                if( strcmp(string_arena_get_string(sa_tmp_type), string_arena_get_string(sa_type)) ) {
                    string_arena_init(sa_type);
                    /* The type might change (possibly from undefined), so let's output */
                    string_arena_add_string(sa_type, "%s", string_arena_get_string(sa_tmp_type));
                    string_arena_add_string(sa_temp, "    data.arena  = %s;\n", string_arena_get_string(sa_type));
                    /* As we change the arena force the reset of the layout */
                    string_arena_init(sa_layout);
                }
                if( strcmp(string_arena_get_string(sa_tmp_layout), string_arena_get_string(sa_layout)) ) {
                    /* Same thing: the memory layout may change at anytime */
                    string_arena_init(sa_layout);
                    string_arena_add_string(sa_layout, "%s", string_arena_get_string(sa_tmp_layout));
                    string_arena_add_string(sa_temp, "    data.layout = %s;\n", string_arena_get_string(sa_tmp_layout));
                }
                if( strcmp(string_arena_get_string(sa_tmp_nbelt), string_arena_get_string(sa_nbelt)) ) {
                    /* Same thing: the number of transmitted elements may change at anytime */
                    string_arena_init(sa_nbelt);
                    string_arena_add_string(sa_nbelt, "%s", string_arena_get_string(sa_tmp_nbelt));
                    string_arena_add_string(sa_temp, "    data.count  = %s;\n", string_arena_get_string(sa_tmp_nbelt));
                }
                if( strcmp(string_arena_get_string(sa_tmp_displ), string_arena_get_string(sa_displ)) ) {
                    /* Same thing: the displacement may change at anytime */
                    string_arena_init(sa_displ);
                    string_arena_add_string(sa_displ, "%s", string_arena_get_string(sa_tmp_displ));
                    string_arena_add_string(sa_temp, "    data.displ  = %s;\n", string_arena_get_string(sa_tmp_displ));
                }
                if( strlen(string_arena_get_string(sa_temp)) ) {
                    string_arena_add_string(sa_datatype,
                                            "%s", string_arena_get_string(sa_temp));
                    string_arena_init(sa_temp);
                }
                last_datatype_idx = dl->dep_datatype_index;
            }
            string_arena_init(sa_ontask);
            string_arena_add_string(sa_ontask,
                                    "if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )\n"
                                    "  return;\n",
                                    JDF_OBJECT_ONAME(dl->guard->calltrue));

            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                if( NULL != dl->guard->calltrue->var) {
                    flowempty = 0;

                    string_arena_add_string(sa_deps,
                                            "%s",
                                            jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa_ontask), dl->guard->calltrue, JDF_OBJECT_LINENO(dl),
                                                                        "    ", "nc") );
                } else {
                    UTIL_DUMP_LIST(sa_temp, dl->guard->calltrue->parameters, next,
                                   dump_expr, (void*)&info, "", "", ", ", "");
                    string_arena_add_string(sa_coutput,
                                            "    /* action_mask & 0x%x goes to data %s(%s) */\n",
                                            (1U << dl->dep_index), dl->guard->calltrue->func_or_mem,
                                            string_arena_get_string(sa_temp));
                    string_arena_init(sa_temp);
                    flowtomem = 1;
                }
                break;
            case JDF_GUARD_BINARY:
                if( NULL != dl->guard->calltrue->var ) {
                    flowempty = 0;
                    string_arena_add_string(sa_deps,
                                            "    if( %s ) {\n"
                                            "%s"
                                            "    }\n",
                                            dump_expr((void**)dl->guard->guard, &info),
                                            jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa_ontask), dl->guard->calltrue, JDF_OBJECT_LINENO(dl),
                                                                        "      ", "nc") );
                } else {
                    UTIL_DUMP_LIST(sa_temp, dl->guard->calltrue->parameters, next,
                                   dump_expr, (void*)&info, "", "", ", ", "");
                    string_arena_add_string(sa_coutput,
                                            "    /* action_mask & 0x%x goes to data %s(%s) */\n",
                                            (1U << dl->dep_index), dl->guard->calltrue->func_or_mem,
                                            string_arena_get_string(sa_temp));
                    string_arena_init(sa_temp);
                    flowtomem = 1;
                }
                break;
            case JDF_GUARD_TERNARY:
                if( NULL != dl->guard->calltrue->var ) {
                    flowempty = 0;
                    string_arena_add_string(sa_deps,
                                            "    if( %s ) {\n"
                                            "%s"
                                            "    }",
                                            dump_expr((void**)dl->guard->guard, &info),
                                            jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa_ontask), dl->guard->calltrue, JDF_OBJECT_LINENO(dl),
                                                                        "      ", "nc"));
                    depnb++;

                    string_arena_init(sa_ontask);
                    string_arena_add_string(sa_ontask,
                                            "if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )\n"
                                            "  return;\n",
                                            JDF_OBJECT_ONAME(dl->guard->callfalse));

                    if( NULL != dl->guard->callfalse->var ) {
                        string_arena_add_string(sa_deps,
                                                " else {\n"
                                                "%s"
                                                "    }\n",
                                                jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa_ontask), dl->guard->callfalse, JDF_OBJECT_LINENO(dl),
                                                                            "      ", "nc") );
                    } else {
                        string_arena_add_string(sa_deps,
                                                "\n");
                    }
                } else {
                    depnb++;
                    string_arena_init(sa_ontask);
                    string_arena_add_string(sa_ontask,
                                            "if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )\n"
                                            "  return;\n",
                                            JDF_OBJECT_ONAME(dl->guard->callfalse));

                    if( NULL != dl->guard->callfalse->var ) {
                        flowempty = 0;
                        string_arena_add_string(sa_deps,
                                                "    if( !(%s) ) {\n"
                                                "%s"
                                                "    }\n",
                                                dump_expr((void**)dl->guard->guard, &info),
                                                jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa_ontask), dl->guard->callfalse, JDF_OBJECT_LINENO(dl),
                                                                            "      ", "nc") );
                    } else {
                        UTIL_DUMP_LIST(sa_temp, dl->guard->callfalse->parameters, next,
                                       dump_expr, (void*)&info, "", "", ", ", "");
                        string_arena_add_string(sa_coutput,
                                                "    /* action_mask & 0x%x goes to data %s(%s) */\n",
                                                (1U << dl->dep_index), dl->guard->callfalse->func_or_mem,
                                                string_arena_get_string(sa_temp));
                        string_arena_init(sa_temp);
                        flowtomem = 1;
                    }
                }
                break;
            }
            depnb++;
            /* Dump the previous dependencies */
            OUTPUT_PREV_DEPS((1U << dl->dep_index), sa_datatype, sa_deps);
        }

        if( (1 == flowempty) && (0 == flowtomem) ) {
            coutput("  /* Flow of data %s has only IN dependencies */\n", fl->varname);
        } else if( 1 == flowempty ) {
            coutput("  /* Flow of data %s has only OUTPUT dependencies to Memory */\n", fl->varname);
        } else {
            coutput("  if( action_mask & 0x%x ) {  /* Flow of Data %s */\n"
                    "%s"
                    "  }\n",
                    (flow_type & JDF_DEP_FLOW_OUT) ? fl->flow_dep_mask_out : fl->flow_dep_mask_in,
                    fl->varname, string_arena_get_string(sa_coutput));
        }
    }
    coutput("  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;\n");
    coutput("}\n\n");

    string_arena_free(sa_ontask);
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa_coutput);
    string_arena_free(sa_deps);
    string_arena_free(sa_datatype);
    string_arena_free(sa_type);
    string_arena_free(sa_tmp_type);
    string_arena_free(sa_nbelt);
    string_arena_free(sa_tmp_nbelt);
    string_arena_free(sa_displ);
    string_arena_free(sa_tmp_displ);
    string_arena_free(sa_layout);
    string_arena_free(sa_tmp_layout);
    string_arena_free(sa_temp);
}

/**
 * Generates the code corresponding to inline_c expressions. If the inline_c was
 * defined in the context of a function, then it uses the function name and the
 * function arguments to build it's scope. Otherwise, it uses the global scope
 * of the handle.
 */
static void jdf_generate_inline_c_function(jdf_expr_t *expr)
{
    static int inline_c_functions = 0;
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    assignment_info_t ai;
    int rc;

    assert(JDF_OP_IS_C_CODE(expr->op));
    if( NULL != expr->jdf_c_code.function_context ) {
        rc = asprintf(&expr->jdf_c_code.fname, "%s_%s_inline_c_expr%d_line_%d",
                      jdf_basename, expr->jdf_c_code.function_context->fname,
                      ++inline_c_functions, expr->jdf_c_code.lineno);
        assert(rc != -1);
        coutput("static inline int %s(const __dague_%s_internal_handle_t *__dague_handle, const %s *assignments)\n"
                "{\n"
                "  (void)__dague_handle;\n",
                expr->jdf_c_code.fname, jdf_basename,
                dague_get_name(NULL, expr->jdf_c_code.function_context, "assignment_t"));

        coutput("  /* This inline C function was declared in the context of the task %s */\n",
                expr->jdf_c_code.function_context->fname);

        ai.sa = sa1;
        ai.holder = "assignments->";
        ai.expr = NULL;
        coutput("%s\n",
                UTIL_DUMP_LIST(sa2, expr->jdf_c_code.function_context->locals, next,
                               dump_local_assignments, &ai, "", "  ", "\n", "\n"));
         coutput("%s\n",
                UTIL_DUMP_LIST_FIELD(sa2, expr->jdf_c_code.function_context->locals, next, name,
                                     dump_string, NULL, "", "  (void)", ";", ";\n"));
    } else {
        rc = asprintf(&expr->jdf_c_code.fname, "%s_inline_c_expr%d_line_%d",
                      jdf_basename, ++inline_c_functions, expr->jdf_c_code.lineno);
        assert(rc != -1);
        coutput("static inline int %s(const __dague_%s_internal_handle_t *__dague_handle, const %s *assignments)\n"
                "{\n"
                "  /* This inline C function was declared in the global context: no variables */\n"
                "  (void)assignments;\n"
                "  (void)__dague_handle;\n",
                expr->jdf_c_code.fname, jdf_basename,
                dague_get_name(NULL, expr->jdf_c_code.function_context, "assignment_t"));
    }

    string_arena_free(sa1);
    string_arena_free(sa2);

    coutput("%s\n", expr->jdf_c_code.code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline )
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    coutput("}\n"
            "\n");
}

static void jdf_generate_inline_c_functions(jdf_t* jdf)
{
    jdf_function_entry_t *f;
    jdf_expr_t *le;
    if( NULL != jdf->inline_c_functions )
        jdf_generate_inline_c_function(jdf->inline_c_functions);

    for(f = jdf->functions; NULL != f; f = f->next) {
        for( le = f->inline_c_functions; NULL != le; le = le->next_inline ) {
            jdf_generate_inline_c_function(le);
        }
    }
}


/**
 * Analyze the code to optimize the output
 */
int jdf_optimize( jdf_t* jdf )
{
    jdf_function_entry_t *f;
    string_arena_t *sa;
    int i, can_be_startup, high_priority, has_displacement;
    jdf_dataflow_t* flow;
    jdf_dep_t *dep;

    sa = string_arena_new(64);
    /**
     * Check if any function is marked as high priority (via the properties) or if all
     * arguments are loaded from the original data. If yes, then mark the function as
     * potential startup.
     */
    for(i = 0, f = jdf->functions; NULL != f; f = f->next, i++) {
        /* Check if the function has the HIGH_PRIORITY property on */
        high_priority = jdf_property_get_int(f->properties, "high_priority", 0);
        if( high_priority ) {
            f->flags |= JDF_FUNCTION_FLAG_HIGH_PRIORITY;
        }
        /* Check if the function has any successors and predecessors */
        jdf_check_relatives(f, JDF_DEP_FLOW_OUT, JDF_FUNCTION_FLAG_NO_SUCCESSORS);
        jdf_check_relatives(f, JDF_DEP_FLOW_IN, JDF_FUNCTION_FLAG_NO_PREDECESSORS);

        can_be_startup = 1;
        UTIL_DUMP_LIST(sa, f->dataflow, next, has_ready_input_dependency, &can_be_startup, NULL, NULL, NULL, NULL);
        if( can_be_startup ) {
            f->flags |= JDF_FUNCTION_FLAG_CAN_BE_STARTUP;
        }
        /* Do the flow has explicit displacement */
        for( flow = f->dataflow; NULL != flow; flow = flow->next ) {
            has_displacement = 0;
            for( dep = flow->deps; NULL != dep; dep = dep->next ) {
                has_displacement |= dep->dep_flags;
            }
            if( JDF_DEP_HAS_DISPL & has_displacement )
                flow->flow_flags |= JDF_FLOW_HAS_DISPL;
        }
    }
    string_arena_free(sa);
    return 0;
}

/** Main Function */

int jdf2c(const char *output_c, const char *output_h, const char *_jdf_basename, jdf_t *jdf)
{
    int ret = 0;

    jdf_cfilename = output_c;
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

    /**
     * Now generate the code.
     */
    jdf_generate_header_file(jdf);

    /**
     * Dump all the prologue sections
     */
    if( NULL != jdf->prologue ) {
        coutput("%s\n", jdf->prologue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }

    jdf_generate_structure(jdf);
    jdf_generate_inline_c_functions(jdf);
    jdf_generate_hashfunctions(jdf);
    jdf_generate_predeclarations( jdf );
    jdf_generate_functions_statics(jdf); // PETER generates startup tasks
    jdf_generate_startup_hook( jdf );

    /**
     * Generate the externally visible function.
     */
    jdf_generate_destructor( jdf );
    jdf_generate_constructor(jdf);

    /**
     * Dump all the epilogue sections
     */
    if( NULL != jdf->epilogue ) {
        coutput("%s\n", jdf->epilogue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            coutput("#line %d \"%s\"\n",cfile_lineno+1, jdf_cfilename);
    }

 err:
    if( NULL != cfile )
        fclose(cfile);

    if( NULL != hfile )
        fclose(hfile);

#if defined(HAVE_INDENT)
    {
        char* command;

#if !defined(HAVE_AWK)
        asprintf(&command, "%s %s %s", DAGUE_INDENT_PREFIX, DAGUE_INDENT_OPTIONS, output_c );
        system(command);
        asprintf(&command, "%s %s %s", DAGUE_INDENT_PREFIX, DAGUE_INDENT_OPTIONS, output_h );
        system(command);
#else
        asprintf(&command,
                 "%s %s %s -st | "
                 "%s '$1==\"#line\" && $3==\"\\\"%s\\\"\" {printf(\"#line %%d \\\"%s\\\"\\n\", NR+1); next} {print}'"
                 "> %s.indent.awk",
                 DAGUE_INDENT_PREFIX, DAGUE_INDENT_OPTIONS, output_c,
                 DAGUE_AWK_PREFIX, output_c, output_c,
                 output_c);
        system(command);
        asprintf(&command, "%s.indent.awk", output_c);
        rename(command, output_c);

        asprintf(&command,
                 "%s %s %s -st | "
                 "%s '$1==\"#line\" && $3==\"\\\"%s\\\"\" {printf(\"#line %%d \\\"%s\\\"\\n\", NR+1); next} {print}'"
                 "> %s.indent.awk",
                 DAGUE_INDENT_PREFIX, DAGUE_INDENT_OPTIONS, output_h,
                 DAGUE_AWK_PREFIX, output_h, output_h,
                 output_h);
        system(command);
        asprintf(&command, "%s.indent.awk", output_h);
        rename(command, output_h);
#endif
    }
#endif  /* defined(HAVE_INDENT) */

    return ret;
}
