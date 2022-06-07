/**
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#if defined(PARSEC_HAVE_SYS_TYPES_H)
#include <sys/types.h>
#endif
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"
#include "jdf2c.h"
#include "parsec/profiling.h"

extern const char *yyfilename;

static FILE *cfile;
static int   cfile_lineno;
static FILE *hfile;
static int   hfile_lineno;
static const char *jdf_basename;
static const char *jdf_cfilename;

/* Optional declarations of local functions */
static int jdf_expr_depends_on_symbol(const char *varname, const jdf_expr_t *expr);
static void jdf_generate_code_hooks(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_data_lookup(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_iterate_successors_or_predecessors(const jdf_t *jdf, const jdf_function_entry_t *f,
                                                                 const char *prefix, jdf_dep_flags_t flow_type);
static void jdf_generate_code_datatype_lookup(const jdf_t *jdf,  const jdf_function_entry_t *f, const char *name);
static void
jdf_generate_code_find_deps(const jdf_t *jdf,
                            const jdf_function_entry_t *f,
                            const char *name);
static void jdf_generate_inline_c_functions(jdf_t* jdf);

/* local constants */

/* Define datatypes that JDF_C_CODE functions can return. */
static const char *full_type[]  = { "int32_t", "int64_t", "float", "double", "parsec_arena_datatype_t*" };
static const char *short_type[] = {   "int32",   "int64", "float", "double", "parsec_arena_datatype_t*" };

/* C99 doesn't help us initialize a structure to zero easily */
static const jdf_expr_t jdf_empty_expr;

#define TASKPOOL_GLOBAL_PREFIX  "__parsec_tp->super."

/* A coutput and houtput functions to write in the .h and .c files, counting the number of lines */

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
    } else if( 0 < len ) {
#if (defined(__WINDOWS__) || defined(__CYGWIN__)) && !defined(__MINGW64__)
        char *start = res, *end;
        while( NULL != (end = strchr(start, '\n'))) {
            if( (end != start) && (end[-1] != '\r')) {
                fwrite(start, (end - start), 1, cfile);
                fwrite("\r\n", 2, 1, cfile);
            } else {
                fwrite(start, (end - start) + 1, 1, cfile);
            }
            len -= (end - start) + 1;
            start = end + 1;  /* skip the current \n */
        }
        fwrite(start, len, 1, cfile);
#else
        fwrite(res, len, 1, cfile);
#endif  /* (defined(__WINDOWS__) || defined(__CYGWIN__)) && !defined(__MINGW64__) */
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
    } else if( 0 < len ) {
#if (defined(__WINDOWS__) || defined(__CYGWIN__)) && !defined(__MINGW64__)
        char *start = res, *end;
        while( NULL != (end = strchr(start, '\n'))) {
            if( (end != start) && (end[-1] != '\r')) {
                fwrite(start, (end - start), 1, hfile);
                fwrite("\r\n", 2, 1, hfile);
            } else {
                fwrite(start, (end - start + 1), 1, hfile);
            }
            len -= (end - start) + 1;
            start = end + 1;  /* skip the current \n */
        }
        fwrite(start, len, 1, hfile);
#else
        fwrite(res, len, 1, hfile);
#endif  /* (defined(__WINDOWS__) || defined(__CYGWIN__)) && !defined(__MINGW64__) */
        hfile_lineno += nblines(res);
        free(res);
    }
}

static void var_to_c_code(jdf_expr_t* expr)
{
    assert(JDF_VAR == expr->op);
    char *fname = expr->jdf_var;

    JDF_OBJECT_ONAME(expr)            = fname;
    expr->jdf_c_code.fname            = fname;
    expr->op                          = JDF_C_CODE;
    expr->jdf_c_code.lineno           = -1;
    expr->jdf_c_code.code             = NULL;
    expr->jdf_c_code.function_context = NULL;
}


/**
 * Generate a semi-persistent string (kept in a circular buffer that will be reused after
 * parsec_name_placeholders_max uses), representing the naming scheme behind the code
 * generator, aka. __parsec_<JDF NAME>_<FUNC NAME>_%<others>.
 */
static char** parsec_name_placeholders = NULL;
static int parsec_name_placeholders_index = 0;
static const int parsec_name_placeholders_max = 64;

static void free_name_placeholders(void) {
    int i;
    if(NULL == parsec_name_placeholders)
        return;
    for(i = 0; i < parsec_name_placeholders_max; i++) {
        if(NULL == parsec_name_placeholders[i]) continue;
        free(parsec_name_placeholders[i]);
    }
    free(parsec_name_placeholders);
}

static char*
parsec_get_name(const jdf_t *jdf, const jdf_function_entry_t *f, char* fmt, ...)
{
    char* tmp = NULL; (void)jdf;
    va_list others;
    int rc;

    if( NULL == parsec_name_placeholders ) {
        parsec_name_placeholders = (char**)calloc(parsec_name_placeholders_max, sizeof(char*));
    }
    if( NULL != parsec_name_placeholders[parsec_name_placeholders_index] ) {
        free(parsec_name_placeholders[parsec_name_placeholders_index]);
        parsec_name_placeholders[parsec_name_placeholders_index] = NULL;
    }
    rc = asprintf(&tmp, "__parsec_%s_%s_%s", jdf_basename, f->fname, fmt);
    if( 0 > rc )
        return NULL;
    va_start(others, fmt);
    rc = vasprintf(&parsec_name_placeholders[parsec_name_placeholders_index],
                   tmp, others);
    va_end(others);
    free(tmp);
    if( 0 > rc )
        return NULL;
    tmp = parsec_name_placeholders[parsec_name_placeholders_index];
    parsec_name_placeholders_index = (parsec_name_placeholders_index + 1) % parsec_name_placeholders_max;
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
static char* dump_string(void **elt, void *_)
{
    (void)_;
    return (char*)*elt;
}

/**
 * dump_C99_struct_initialization:
 * dump a correct code to initialize a structure as indicated in
 * 6.7.8 Initialization of ISO/IEC 9899:1999 (aka. .name = val).
 */
static char* dump_C99_struct_initialization(void **elt, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    char* name = (char*)*elt;
    string_arena_init(sa);
    string_arena_add_string(sa, ".%s.value = %s", name, name);
    return string_arena_get_string(sa);
}

/**
 * dump_globals:
 *   Dump a global symbol like #define ABC (__parsec_tp->ABC)
 */
static char* dump_globals(void** elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;

    string_arena_init(sa);
    if( NULL != global->data ) {
        string_arena_add_string(sa, "%s "TASKPOOL_GLOBAL_PREFIX"_g_%s",
                                global->name, global->name );
    } else {
        string_arena_add_string(sa, "%s ("TASKPOOL_GLOBAL_PREFIX"_g_%s)",
                                global->name, global->name );
    }
    return string_arena_get_string(sa);
}

/**
 * dump_data:
 *   Dump a global symbol like
 *     #define data_of_ABC(A0, A1) (__parsec_tp->super.ABC->data_of(__parsec_tp->super.ABC, A0, A1))
 */
static char* dump_data(void** elem, void *arg)
{
    jdf_data_entry_t* data = (jdf_data_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    int i;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s(%s_d%d", data->dname, data->dname, 0 );
    for( i = 1; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", %s_d%d", data->dname, i );
    }
    string_arena_add_string(sa, ")  (((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->data_of((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s",
                            data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", (%s_d%d)", data->dname, i );
    }
    string_arena_add_string(sa, "))" );
    return string_arena_get_string(sa);
}

static int expr_requires_assignment(const jdf_expr_t *expr, int inc_locals)
{
    if( inc_locals && NULL != expr->local_variables ) {
        return 1;
    }
    if(expr->op == JDF_C_CODE)
        return 1;
    if( JDF_OP_IS_UNARY( expr->op ) )
        return expr_requires_assignment(expr->jdf_ua, inc_locals);
    if( JDF_OP_IS_TERNARY( expr->op ) )
        return expr_requires_assignment(expr->jdf_ta1, inc_locals) ||
            expr_requires_assignment(expr->jdf_ta2, inc_locals) ||
            expr_requires_assignment(expr->jdf_ta3, inc_locals);
    if( JDF_OP_IS_CST( expr->op ) )
        return 0;
    if( JDF_OP_IS_STRING( expr->op ) )
        return 0;
    if( JDF_OP_IS_VAR( expr->op ) )
        return 0;
    if( JDF_OP_IS_BINARY( expr->op ) )
        return expr_requires_assignment( expr->jdf_ba1, inc_locals ) ||
            expr_requires_assignment( expr->jdf_ba2, inc_locals );
    assert(0);
    return 1;
}

/**
 * dump_rank:
 *   Dump a global symbol like
 *     #define rank_of_ABC(A0, A1) (__parsec_tp->super.ABC->rank_of(__parsec_tp->super.ABC, A0, A1))
 */
static char* dump_rank(void** elem, void *arg)
{
    jdf_data_entry_t* data = (jdf_data_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    int i;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s(%s_d%d", data->dname, data->dname, 0 );
    for( i = 1; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", %s_d%d", data->dname, i );
    }
    string_arena_add_string(sa, ")  (((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->rank_of((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s",
                            data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", (%s_d%d)", data->dname, i );
    }
    string_arena_add_string(sa, "))" );
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
    expr_info_t li = EMPTY_EXPR_INFO, ri = EMPTY_EXPR_INFO;
    jdf_expr_t *e = (jdf_expr_t*)elem;
    string_arena_t *sa = expr_info->sa;
    string_arena_t *la, *ra;
    char *vc, *dot;

    if(e == NULL ) return "NULL";
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
        if( NULL != (dot = strchr(vc, '.')) )
            *dot = '\0';
        if( NULL != (dot = strstr(vc, "->")) )
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
        string_arena_add_string(sa, "(%s) == (%s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_NOTEQUAL:
        string_arena_add_string(sa, "(%s) != (%s)",
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
        string_arena_add_string(sa, "(%s) < (%s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_LEQ:
        string_arena_add_string(sa, "(%s) <= (%s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MORE:
        string_arena_add_string(sa, "(%s) > (%s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_MEQ:
        string_arena_add_string(sa, "(%s) >= (%s)",
                                dump_expr((void**)e->jdf_ba1, &li),
                                dump_expr((void**)e->jdf_ba2, &ri) );
        break;
    case JDF_NOT:
        string_arena_add_string(sa, "!(%s)",
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
        string_arena_add_string(sa,"\n#error ptg-compiler tried to dump a range expression\n");
        break;
    case JDF_TERNARY: {
        expr_info_t ti = EMPTY_EXPR_INFO;
        string_arena_t *ta;
        ta = string_arena_new(8);
        ti.sa = ta;
        ti.prefix = expr_info->prefix;
        ti.suffix = expr_info->suffix;
        ti.assignments = expr_info->assignments;

        string_arena_add_string(sa, "((%s) ? (%s) : (%s))",
                                dump_expr((void**)e->jdf_tat, &ti),
                                dump_expr((void**)e->jdf_ta1, &li),
                                dump_expr((void**)e->jdf_ta2, &ri) );

        string_arena_free(ta);
        break;
    }
    case JDF_CST:
        switch(e->jdf_type) {
        case EXPR_TYPE_INT32:
            string_arena_add_string(sa, "%d", e->jdf_cst);
            break;
        case EXPR_TYPE_INT64:
            string_arena_add_string(sa, "%"PRId64, e->jdf_cst64);
            break;
        case EXPR_TYPE_FLOAT:
            string_arena_add_string(sa, "%f", e->jdf_cstfloat);
            break;
        case EXPR_TYPE_DOUBLE:
            string_arena_add_string(sa, "%lf", e->jdf_cstdouble);
            break;
        default:
            string_arena_add_string(sa, "%d", e->jdf_cst);
            break;
        }
        break;
    case JDF_STRING:
        string_arena_add_string(sa, "%s", e->jdf_var);
        break;
    case JDF_C_CODE:
        if(  NULL == e->jdf_c_code.fname ) {
            /* Keep the output clean. Don't add anything to could make this string unique,
             * as it is used for validation purposes in the jdf_sanity_check_remote_memory_references.
             */
            string_arena_add_string(sa, "\n#error Expression %s has not been generated\n",
                                    e->jdf_c_code.code);
        } else {
            string_arena_add_string(sa, "%s(__parsec_tp, %s)",
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
 *  #define F_pred(k, n, m) (__parsec_tp->ABC->rank == __parsec_tp->ABC->rank_of(__parsec_tp->ABC, k, n, m))
 * In case one of the arguments is an inline C we need to prepare the assignments to match the expected call
 * &(__parsec_%s_%s_parsec_assignment_t){.%s.value = %s, .%s.value = %s, .%s.value = %s}.
 */
static char* dump_predicate(void** elem, void *arg)
{
    jdf_function_entry_t *f = (jdf_function_entry_t *)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    expr_info_t expr_info = EMPTY_EXPR_INFO;

    /* Prepare the assignment field for the complex calls (where we have inline_C functions) */
    string_arena_init(sa);
    string_arena_add_string(sa, "&(__parsec_%s_%s_parsec_assignment_t){%s}",
                            jdf_basename, f->fname,
                            UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name,
                                                 dump_C99_struct_initialization, sa3,
                                                 "", "", ", ", ""));
    expr_info.assignments = strdup(string_arena_get_string(sa));

    string_arena_init(sa);
    string_arena_add_string(sa, "%s_pred(%s) ",
                            f->fname,
                            UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name,
                                                 dump_string, NULL,
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
    expr_info.suffix = "";
    string_arena_add_string(sa, "(((parsec_data_collection_t*)("TASKPOOL_GLOBAL_PREFIX"_g_%s))->myrank == "
                            "rank_of_%s(%s))",
                            f->predicate->func_or_mem, f->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, f->predicate->parameters, next,
                                           dump_expr, &expr_info,
                                           "", "", ", ", ""));

    string_arena_free(sa2);
    string_arena_free(sa3);
    free(expr_info.assignments);  /* free the temporary assignment */
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
    int dos = jdf_expr_depends_on_symbol(def->name, info->expr);

    if( dos > 0 ) {
        string_arena_init(info->sa);
        string_arena_add_string(info->sa, "const int %s = %s%s.value;", def->name, info->holder, def->name);
        if( dos > 1 )
            string_arena_add_string(info->sa, " (void)%s;", def->name);
        return string_arena_get_string(info->sa);
    }
    return NULL;
}

/**
 * dump_local_used_in_expr:
 * Generate the name of a variable but only if it appears on an expression.
 */
static char* dump_local_used_in_expr( void** elem, void* arg )
{
    jdf_def_list_t *def = (jdf_def_list_t*)elem;
    assignment_info_t *info = (assignment_info_t*)arg;
    int dos = jdf_expr_depends_on_symbol(def->name, info->expr);

    if( dos > 0 ) {
        string_arena_init(info->sa);
        string_arena_add_string(info->sa, "%s", def->name);
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
 *  needed for declaring the corresponding arena chunk and
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
                            "  parsec_arena_chunk_t *g%s;\n"
                            "  data_repo_entry_t *e%s = NULL; /**< repo entries can be NULL for memory data */\n",
                            f->varname,
                            f->varname);
    return string_arena_get_string(sa);
}

/**
 * Parameters of the dump_data_initialization_from_data_array
 */
typedef struct init_from_data_info {
    string_arena_t *sa;
    const char *where;
} init_from_data_info_t;

/**
 * dump_data_initialization_from_data_array:
 *  Takes the pointer to a flow *f, let say that f->varname == "A", and where ==
 *  "in", this produces a string like
 *  parsec_data_copy_t *_f_A = this_task->data[id].data_in;\n
 *  void *A = PARSEC_DATA_COPY_GET_PTR(gA); (void)A;\n
 */
static char *dump_data_initialization_from_data_array(void **elem, void *arg)
{
    init_from_data_info_t *info = (init_from_data_info_t*)arg;
    string_arena_t *sa = info->sa;
    const char *where = info->where;
    jdf_dataflow_t *f = (jdf_dataflow_t*)elem;
    char *varname = f->varname;

    if(f->flow_flags & JDF_FLOW_TYPE_CTL) {
        return NULL;
    }

    string_arena_init(sa);

    string_arena_add_string(sa,
                            "  parsec_data_copy_t *_f_%s = this_task->data._f_%s.data_%s;\n",
                            varname, f->varname, where);
    string_arena_add_string(sa,
                            "  void *%s = PARSEC_DATA_COPY_GET_PTR(_f_%s); (void)%s;\n",
                            varname, varname, varname);
    return string_arena_get_string(sa);
}

static char *dump_data_copy_init_for_inline_function(void **elem, void *arg)
{
    init_from_data_info_t *info = (init_from_data_info_t*)arg;
    string_arena_t *sa = info->sa;
    const char *where = info->where;
    jdf_dataflow_t *f = (jdf_dataflow_t*)elem;
    char *varname = f->varname;

    if(f->flow_flags & JDF_FLOW_TYPE_CTL) {
        return NULL;
    }

    string_arena_init(sa);
    string_arena_add_string(sa,
                            "  parsec_data_copy_t *_f_%s = this_task->data._f_%s.data_%s;\n"
                            "  (void)_f_%s;",
                            varname, f->varname, where,
                            varname
                            );
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

static int profile_enabled(jdf_def_list_t *dl)
{
    return jdf_property_get_int(dl, "profile", 1);
}

/**
 * dump_profiling_init:
 *  Takes the pointer to the name of a function, an index in
 *  a pointer to a dump_profiling_init, and prints
 *    parsec_profiling_add_dictionary_keyword( "elem", attribute[idx], &elem_key_start, &elem_key_end);
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

    if( !profile_enabled(f->properties) ) {
        return NULL;
    }

    string_arena_init(info->sa);

    get_unique_rgb_color((float)info->idx / (float)info->maxidx, &R, &G, &B);
    info->idx++;

    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_variable_list_t, next, nb_locals);
    profiling_convertor_params = string_arena_new(64);
    UTIL_DUMP_LIST_FIELD(profiling_convertor_params, f->locals, next, name, dump_string, NULL,
                         PARSEC_TASK_PROF_INFO_CONVERTOR, ";", "{int32_t}", "{int32_t}");

    string_arena_add_string(info->sa,
                            "#if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)\n"
                            "parsec_profiling_add_dictionary_keyword(\"%s::%s::internal init\", \"fill:%02X%02X%02X\",\n"
                            "                                       0,\n"
                            "                                       NULL,\n"
                            "                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * %s_%s.task_class_id  + 2 * PARSEC_%s_NB_TASK_CLASSES/* %s (internal init) start key */],\n"
                            "                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * %s_%s.task_class_id  + 2 * PARSEC_%s_NB_TASK_CLASSES/* %s (internal init) end key */]);\n"
                            "#endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */\n",
                            jdf_basename, fname, 256-R, 256-G, 256-B,
                            jdf_basename, fname, jdf_basename, fname,
                            jdf_basename, fname, jdf_basename, fname);
    string_arena_add_string(info->sa,
                            "parsec_profiling_add_dictionary_keyword(\"%s::%s\", \"fill:%02X%02X%02X\",\n"
                            "                                       sizeof(parsec_task_prof_info_t)+%d*sizeof(parsec_assignment_t),\n"
                            "                                       \"%s\",\n"
                            "                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * %s_%s.task_class_id /* %s start key */],\n"
                            "                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * %s_%s.task_class_id /* %s end key */]);\n",
                            jdf_basename, fname, R, G, B,
                            nb_locals,
                            string_arena_get_string(profiling_convertor_params),
                            jdf_basename, fname, fname,
                            jdf_basename, fname, fname);

    string_arena_free(profiling_convertor_params);

    return string_arena_get_string(info->sa);
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
            string_arena_add_string(sa, TASKPOOL_GLOBAL_PREFIX"_g_%s = %s;", global->name, global->name);
    } else {
        expr_info_t info = EMPTY_EXPR_INFO;
        info.sa = string_arena_new(8);
        info.prefix = "";
        info.suffix = "";
        info.assignments = "assignments";

        string_arena_add_string(sa, TASKPOOL_GLOBAL_PREFIX"_g_%s = %s = %s;",
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
    char* prefix;
} typed_globals_info_t;

static char* dump_typed_globals(void **elem, void *arg)
{
    typed_globals_info_t* prop = (typed_globals_info_t*)arg;
    string_arena_t *sa = prop->sa;
    jdf_global_entry_t* global = (jdf_global_entry_t*)elem;
    jdf_expr_t* type_str = jdf_find_property( global->properties, "type", NULL );
    jdf_expr_t *size_str = jdf_find_property( global->properties, "size", NULL );
    jdf_expr_t *prop_str;
    expr_info_t info = EMPTY_EXPR_INFO;

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
        string_arena_add_string(sa, "%s %s%s",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), prop->prefix, global->name);
    } else {
        string_arena_add_string(sa, "%s %s%s /* data %s */",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), prop->prefix, global->name, global->name);
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
 *  Generate the default initializer for all hidden variables. As we generate the
 *  code in the order in which the variables appear in the JDF, and as we dump their
 *  default value as is, they can reference other globals that have been already
 *  defined. Hidden variables without default value are not generated in order to
 *  protect the user from accessing them by mistake.
 */
static char *dump_hidden_globals_init(void **elem, void *arg)
{
    jdf_global_entry_t* global_var = (jdf_global_entry_t*)elem;
    jdf_expr_t *hidden = jdf_find_property( global_var->properties, "hidden", NULL );

    if (NULL != hidden) { /* The hidden property is set */
        jdf_expr_t *prop = jdf_find_property( global_var->properties, "default", NULL );

        if( NULL == prop )  /* Do we have a default value or expression? */
            if( NULL == (prop = global_var->expression) ) return NULL;

        expr_info_t info = EMPTY_EXPR_INFO;
        jdf_expr_t* type_str = jdf_find_property( global_var->properties, "type",   NULL );

        info.sa = string_arena_new(8);
        info.prefix = "";
        info.suffix = "";
        info.assignments = "assignments";

        string_arena_t *sa = (string_arena_t*)arg;
        string_arena_init(sa);
        string_arena_add_string(sa, "%s %s;",
                                (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)),
                                global_var->name);
        string_arena_free(info.sa);

        return string_arena_get_string(sa);
    }
    return NULL;
}

/**
 * dump_name_of_hidden_globals_without_default:
 *  Only dump the hidden variables with default values. This function is
 *  only used in _new to generate the (void) statement to prevent warnings
 *  about unused variables.
 */
static char *dump_name_of_hidden_globals_without_default(void **elem, void *_)
{
    jdf_global_entry_t* global_var = (jdf_global_entry_t*)elem;
    jdf_expr_t *prop = jdf_find_property( global_var->properties, "hidden", NULL );

    if (NULL == prop) return NULL; /* Not a hidden global variable */
    if( NULL == global_var->expression )  /* No default expression */
        if( NULL == jdf_find_property( global_var->properties, "default", NULL ) )  /* No default initializer */
            return NULL;
    (void)_;
    return global_var->name;
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
    int u, v;
    if(NULL == e)
        return 2;
    if( JDF_OP_IS_CST(e->op) || JDF_OP_IS_STRING(e->op) )
        return 0;
    else if ( JDF_OP_IS_VAR(e->op) && 0 == strcmp(e->jdf_var, name) )
        return 1;
    else if ( JDF_OP_IS_UNARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_ua);
    else if ( JDF_OP_IS_TERNARY(e->op) ) {
        u = jdf_expr_depends_on_symbol(name, e->jdf_tat);
        if(u > 1)
            return u;
        v = jdf_expr_depends_on_symbol(name, e->jdf_ta1);
        if(v > u) u = v;
        if(u > 1)
            return u;
        v = jdf_expr_depends_on_symbol(name, e->jdf_ta2);
        if(v > u) u = v;
        return u;
    } else if( JDF_OP_IS_BINARY(e->op) ) {
        u = jdf_expr_depends_on_symbol(name, e->jdf_ba1);
        if(u > 1)
            return u;
        v = jdf_expr_depends_on_symbol(name, e->jdf_ba2);
        if(v > u)
            u = v;
        return u;
    }
    return 2;  /* by default assume we might need all of them */
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
    const jdf_name_list_t *nl;

    if( NULL != property ) *property = NULL;
    for(nl = JDF_COMPILER_GLOBAL_ARGS.ignore_properties;
        nl != NULL;
        nl = nl->next) {
        if( !strcmp(nl->name, property_name) ) {
            *property = NULL;
            return NULL;
        }
    }
    while( NULL != current ) {
        if( !strcmp(current->name, property_name) ) {
            if( NULL != property ) *property = (jdf_def_list_t*)current;
            return current->expr;
        }
        current = current->next;
    }
    return NULL;
}

static int jdf_dataflow_type(const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    int type = 0;
    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
            continue;  /* Skip empty flows that are used to define datatype in WRITE-only flow */
        }
        type |= dl->dep_flags;
    }
    return type;
}

static jdf_function_entry_t *find_target_function(const jdf_t *jdf, const char *name)
{
    jdf_function_entry_t *targetf;
    for(targetf = jdf->functions; targetf != NULL; targetf = targetf->next)
        if( !strcmp(targetf->fname, name) )
            break;
    return targetf;
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

    f = find_target_function(jdf, fname);
    if( NULL == f )
        return NULL;
    for( fl = f->dataflow; fl != NULL; fl = fl->next) {
        if( jdf_dataflow_type(fl) & JDF_DEP_FLOW_OUT ) {
            if( !strcmp(fl->varname, varname) ) {
                return fl;
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
    f = find_target_function(jdf, fname);
    if( NULL == f )
        return -2;
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

/**
 * iterators for the stack of local variables:
 *   local variables are stored as a stack, so managing their scope is simplified
 *   during parsing; however when generating the code, we need to go from the
 *   outermost variable to the innermost, so we need to reverse the stack
 */
static jdf_expr_t *jdf_expr_lv_first(jdf_expr_t *head)
{
    if(NULL == head)
        return NULL;
    while(head->next != NULL) head = head->next;
    return head;
}

static jdf_expr_t *jdf_expr_lv_next(jdf_expr_t *head, jdf_expr_t *cur)
{
    if( cur == head ) return NULL;
    while(head->next != cur) head = head->next;
    return head;
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
#if defined(PARSEC_HAVE_VA_COPY)
    va_copy(ap2, ap);
#elif defined(PARSEC_HAVE_UNDERSCORE_VA_COPY)
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

#if defined(PARSEC_HAVE_VA_COPY) || defined(PARSEC_HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(PARSEC_HAVE_VA_COPY) || defined(PARSEC_HAVE_UNDERSCORE_VA_COPY) */
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
    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_variable_list_t, next, nb_locals);
    UTIL_DUMP_LIST_FIELD(sa_locals, f->locals, next, name, dump_string, NULL,
                         "", "  parsec_assignment_t ", ";\n", ";\n");

    if(f->nb_max_local_def>0) {
        string_arena_add_string(sa_locals, "  parsec_assignment_t ldef[%d];\n", f->nb_max_local_def);
        nb_locals += f->nb_max_local_def;
    }

    if( nb_locals > MAX_LOCAL_COUNT ) {
        jdf_fatal(JDF_OBJECT_LINENO(f),
                  "Task class %s uses %d locals (%d of them being used in local definitions) in its definition space. Current stack can manage up to %d variables only.\n",
                  f->fname, nb_locals, f->nb_max_local_def, MAX_LOCAL_COUNT);
        exit(1);
    }
    
    JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_t, next, nb_flows);
    UTIL_DUMP_LIST_FIELD(sa_data, f->dataflow, next, varname, dump_string, NULL,
                         "", "  parsec_data_pair_t _f_", ";\n", ";\n");
    string_arena_init(sa);
    /* Prepare the structure for the named assignments */
    string_arena_add_string(sa,
                            "#if MAX_LOCAL_COUNT < %d  /* number of parameters and locals %s */\n"
                            "  #error Too many parameters and local variables (%d out of MAX_LOCAL_COUNT) for task %s\n"
                            "#endif  /* MAX_PARAM_COUNT */\n",
                            nb_locals, f->fname, nb_locals, f->fname);
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "%s"
                            "  parsec_assignment_t reserved[MAX_LOCAL_COUNT-%d];\n"
                            "} %s;\n\n",
                            parsec_get_name(NULL, f, "assignment_s"),
                            string_arena_get_string(sa_locals),
                            nb_locals,
                            parsec_get_name(NULL, f, "parsec_assignment_t"));
    string_arena_add_string(sa,
                            "#if MAX_PARAM_COUNT < %d  /* total number of flows for task %s */\n"
                            "  #error Too many flows (%d out of MAX_PARAM_COUNT) for task %s\n"
                            "#endif  /* MAX_PARAM_COUNT */\n",
                            nb_flows, f->fname, nb_flows, f->fname);
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "%s"
                            "  parsec_data_pair_t unused[MAX_LOCAL_COUNT-%d];\n"
                            "} %s;\n\n",
                            parsec_get_name(NULL, f, "data_s"),
                            string_arena_get_string(sa_data),
                            nb_flows,
                            parsec_get_name(NULL, f, "data_t"));
    string_arena_add_string(sa, "typedef struct %s {\n"
                            "    PARSEC_MINIMAL_EXECUTION_CONTEXT\n"
                            "#if defined(PARSEC_PROF_TRACE)\n"
                            "    parsec_task_prof_info_t prof_info;\n"
                            "#endif /* defined(PARSEC_PROF_TRACE) */\n"
                            "    struct __parsec_%s_%s_assignment_s locals;\n"
                            "#if defined(PARSEC_SIM)\n"
                            "    int                        sim_exec_date;\n"
                            "#endif\n"
                            "    struct __parsec_%s_%s_data_s data;\n"
                            "} %s;\n\n",
                            parsec_get_name(NULL, f, "task_s"),
                            jdf_basename, f->fname,
                            jdf_basename, f->fname,
                            parsec_get_name(NULL, f, "task_t"));
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
    houtput("#include \"parsec.h\"\n"
            "#include \"parsec/parsec_internal.h\"\n"
            "#include \"parsec/remote_dep.h\"\n\n"
            );
    houtput("BEGIN_C_DECLS\n\n");

    for( g = jdf->datatypes; NULL != g; g = g->next ) {
        houtput("#define PARSEC_%s_%s_ADT_IDX    %d\n",
                jdf_basename, g->name, datatype_index);
        datatype_index++;
    }
    houtput("#define PARSEC_%s_ADT_IDX_MAX       %d\n", jdf_basename, datatype_index);
    houtput("typedef struct parsec_%s_taskpool_s {\n", jdf_basename);
    houtput("  parsec_taskpool_t super;\n");
    {
        typed_globals_info_t prop = { sa2, NULL, NULL, .prefix = "_g_" };
        houtput("  /* The list of globals */\n"
                "%s",
                UTIL_DUMP_LIST( sa1, jdf->globals, next, dump_typed_globals, &prop,
                                "", "  ", ";\n", ";\n"));
    }
    houtput("  /* The array of datatypes (%s and co.) */\n"
            "  parsec_arena_datatype_t arenas_datatypes[PARSEC_%s_ADT_IDX_MAX];\n"
            "  uint32_t arenas_datatypes_size;\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->datatypes, next, name,
                                  dump_string, NULL, "", "", ",", ""),
            jdf_basename);

    houtput("} parsec_%s_taskpool_t;\n\n", jdf_basename);

    /* Generate the class declaration. Unlike in a normal class hierarchy the class in the
     * wrapper is not a superset of the internal class but a subset. Thus, we should never
     * directly call PARSEC_OBJ_NEW on the wrapper class or we might overwrite memory.
     */
    houtput("/** Beware that this object is not a superset of the internal object, and as a result\n"
            " *  we should never create one directly. Instead we should rely on the creation of the\n"
            " *  internal object (__parsec_%s_taskpool_t).\n"
            " */\n"
            "PARSEC_OBJ_CLASS_DECLARATION(parsec_%s_taskpool_t);\n",
            jdf_basename,
            jdf_basename);

    {
        typed_globals_info_t prop = { sa3, NULL, "hidden", .prefix = "_g_" };
        houtput("extern parsec_%s_taskpool_t *parsec_%s_new(%s);\n\n", jdf_basename, jdf_basename,
                UTIL_DUMP_LIST( sa2, jdf->globals, next, dump_typed_globals, &prop,
                                "", "", ", ", ""));
    }

    houtput("%s", UTIL_DUMP_LIST(sa1, jdf->functions, next, jdf_generate_task_typedef, sa3,
                                 "", "", "\n", "\n"));
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    houtput("END_C_DECLS\n\n");
    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);
}

/**
 * Dump the definitions of all functions and flows. This function must be
 * called early or the name of the functions and flows will not be defined.
 */
static void jdf_generate_predeclarations( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    int rc;

    coutput("/** Predeclarations of the parsec_task_class_t */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        rc = asprintf(&JDF_OBJECT_ONAME( f ), "%s_%s", jdf_basename, f->fname);
        assert(rc != -1);
        coutput("static const parsec_task_class_t %s;\n", JDF_OBJECT_ONAME( f ));
    }
    string_arena_free(sa);
    string_arena_free(sa2);
    coutput("/** Predeclarations of the parameters */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        for(fl = f->dataflow; fl != NULL; fl = fl->next) {
            rc = asprintf(&JDF_OBJECT_ONAME( fl ), "flow_of_%s_%s_for_%s", jdf_basename, f->fname, fl->varname);
            assert(rc != -1);
            coutput("static const parsec_flow_t %s;\n",
                    JDF_OBJECT_ONAME( fl ));
        }
    }
    (void)rc;
}

/**
 * Dump a minimalistic code including all the includes and all the defines that
 * can be used in the prologue. Keep this small so that we don't generate code
 * for structures that are not yet defind, such as those where the corresponding
 * header will only be included in the prologue.
 */
static void jdf_minimal_code_before_prologue(const jdf_t *jdf)
{
    int nbfunctions, nbdata;
    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);
    JDF_COUNT_LIST_ENTRIES(jdf->data, jdf_data_entry_t, next, nbdata);
    coutput("#include \"parsec.h\"\n"
            "#include \"parsec/parsec_internal.h\"\n"
            "#include \"parsec/ayudame.h\"\n"
            "#include \"parsec/execution_stream.h\"\n"
            "#if defined(PARSEC_HAVE_CUDA)\n"
            "#include \"parsec/mca/device/cuda/device_cuda.h\"\n"
            "#endif  /* defined(PARSEC_HAVE_CUDA) */\n"
            "#if defined(_MSC_VER) || defined(__MINGW32__)\n"
            "#  include <malloc.h>\n"
            "#else\n"
            "#  include <alloca.h>\n"
            "#endif  /* defined(_MSC_VER) || defined(__MINGW32__) */\n\n"
            "#define PARSEC_%s_NB_TASK_CLASSES %d\n"
            "#define PARSEC_%s_NB_DATA %d\n\n"
            "typedef struct __parsec_%s_internal_taskpool_s __parsec_%s_internal_taskpool_t;\n"
            "struct parsec_%s_internal_taskpool_s;\n\n",
            jdf_basename, nbfunctions,
            jdf_basename, nbdata,
            jdf_basename, jdf_basename,
            jdf_basename);
    jdf_generate_predeclarations(jdf);
}

static void jdf_generate_structure(jdf_t *jdf)
{
    int nbfunctions, need_profile = 0;
    string_arena_t *sa1, *sa2;
    jdf_function_entry_t* f;
    jdf_param_list_t *pl;

    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("#include \"%s.h\"\n\n"
            "struct __parsec_%s_internal_taskpool_s {\n"
            " parsec_%s_taskpool_t super;\n"
            " volatile int32_t sync_point;\n"
            " parsec_task_t* startup_queue;\n",
            jdf_basename, jdf_basename, jdf_basename);

    coutput("  /* The ranges to compute the hash key */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        if( 0 == (f->user_defines & JDF_FUNCTION_HAS_UD_MAKE_KEY) ) {
            for(pl = f->parameters; pl != NULL; pl = pl->next) {
                coutput("  int %s_%s_range;\n", f->fname, pl->name);
            }
        } else {
            coutput("  /* nothing for %s as it gets a user-defined make_key */\n",
                    f->fname);
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

    coutput("};\n\n");

    for( f = jdf->functions; need_profile == 0 && NULL != f; f = f->next ) {
        /* If the profile property is ON then enable the profiling array */
        need_profile = profile_enabled(f->properties);
    }
    if( need_profile )
        coutput("#if defined(PARSEC_PROF_TRACE)\n"
                "#  if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)\n"
                "static int %s_profiling_array[4*PARSEC_%s_NB_TASK_CLASSES] = {-1}; /* 2 pairs (begin, end) per task, times two because each task class has an internal_init task */\n"
                "#  else /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */\n"
                "static int %s_profiling_array[2*PARSEC_%s_NB_TASK_CLASSES] = {-1}; /* 2 pairs (begin, end) per task */\n"
                "#  endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */\n"
                "#endif  /* defined(PARSEC_PROF_TRACE) */\n",
                jdf_basename, jdf_basename,
                jdf_basename, jdf_basename);

    UTIL_DUMP_LIST(sa1, jdf->globals, next,
                   dump_globals, sa2, "", "#define ", "\n", "\n");
    if( 1 < strlen(string_arena_get_string(sa1)) ) {
        coutput("/* Globals */\n%s\n", string_arena_get_string(sa1));
    }

    coutput("static inline int parsec_imin(int a, int b) { return (a <= b) ? a : b; };\n\n"
            "static inline int parsec_imax(int a, int b) { return (a >= b) ? a : b; };\n\n");

    /**
     * Generate the inline_c functions as soon as possible, or they will not be usable
     * during the macro generation.
     */
    jdf_generate_inline_c_functions(jdf);

    coutput("/* Data Access Macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_data, sa2, "", "#define data_of_", "\n", "\n"));
    coutput("%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_rank, sa2, "", "#define rank_of_", "\n", "\n"));

    coutput("/* Functions Predicates */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->functions, next,
                           dump_predicate, sa2, "", "#define ", "\n", "\n"));

    coutput("/* Data Repositories */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            /* Generate the repo for all tasks classes, they can be used when:
             * - a predecessor sets up a reshape promised for a datacopy
             * - the own tasks use it when reshaping a datacopy directly read from desc
             * No longer only when if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) )
             */
            coutput("#define %s_repo (__parsec_tp->repositories[%d])\n",
                    f->fname, f->task_class_id);
        }
    }

    if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
        coutput("/* Dependency Tracking Allocation Macro */\n"
                "#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, FLAG)                  \\\n"
                "do {                                                                          \\\n"
                "  int _vmin = (vMIN);                                                         \\\n"
                "  int _vmax = (vMAX);                                                         \\\n"
                "  (DEPS) = (parsec_dependencies_t*)calloc(1, sizeof(parsec_dependencies_t) +  \\\n"
                "                   (_vmax - _vmin) * sizeof(parsec_dependencies_union_t));    \\\n"
                "  PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, \"Allocate %%d spaces for loop %%s (min %%d max %%d) 0x%%p\",    \\\n"
                "           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS));        \\\n"
                "  (DEPS)->flags = PARSEC_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                \\\n"
                "  (DEPS)->min = _vmin;                                                        \\\n"
                "  (DEPS)->max = _vmax;                                                        \\\n"
                "} while (0)\n\n");
    }

    coutput("/* Release dependencies output macro */\n"
            "#if defined(PARSEC_DEBUG_NOISIER)\n"
            "#define RELEASE_DEP_OUTPUT(ES, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\\\n"
            "  do { \\\n"
            "    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\\\n"
            "    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, \"thread %%d VP %%d explore deps from %%s:%%s to %%s:%%s (from rank %%d to %%d) base ptr %%p\",\\\n"
            "           (NULL != (ES) ? (ES)->th_id : -1), (NULL != (ES) ? (ES)->virtual_process->vp_id : -1),\\\n"
            "           DEPO, parsec_task_snprintf(tmp1, 128, (parsec_task_t*)(TASKO)),\\\n"
            "           DEPI, parsec_task_snprintf(tmp2, 128, (parsec_task_t*)(TASKI)), (RSRC), (RDST), (DATA));\\\n"
            "  } while(0)\n"
            "#define ACQUIRE_FLOW(TASKI, DEPI, FUNO, DEPO, LOCALS, PTR)\\\n"
            "  do { \\\n"
            "    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\\\n"
            "    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, \"task %%s acquires flow %%s from %%s %%s data ptr %%p\",\\\n"
            "           parsec_task_snprintf(tmp1, 128, (parsec_task_t*)(TASKI)), (DEPI),\\\n"
            "           (DEPO), parsec_snprintf_assignments(tmp2, 128, (FUNO), (parsec_assignment_t*)(LOCALS)), (PTR));\\\n"
            "  } while(0)\n"
            "#else\n"
            "#define RELEASE_DEP_OUTPUT(ES, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\n"
            "#define ACQUIRE_FLOW(TASKI, DEPI, TASKO, DEPO, LOCALS, PTR)\n"
            "#endif\n");
    string_arena_free(sa1);
    string_arena_free(sa2);
}

static int jdf_expr_is_range( const jdf_expr_t *e )
{
    jdf_expr_t *ld;
    if( JDF_RANGE == e->op )
        return 1;
    for(ld = e->local_variables; NULL != ld; ld = ld->next) {
        if( ld->op == JDF_RANGE )
            return 1;
    }
    return 0;
}

/**
 * Generates a highly optimized function for an expression. If the expression is
 * constant or an inlined code no local variables are generated. If the
 * expresion is a constant then it is directly returned, if the expression is an
 * inlined function then a call to the original accessor is generated
 * instead. This function only generates the code without generating the
 * corresponding parsec_expr_t.
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
    expr_info_t info = EMPTY_EXPR_INFO;
    assignment_info_t ai;

    (void)jdf;

    assert( !jdf_expr_is_range(e) );

    coutput("static inline %s %s%s(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *locals)\n"
            "{\n",
            rettype, name, suffix, jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"));
    if( !(JDF_OP_IS_C_CODE(e->op) || (JDF_OP_IS_CST(e->op))) ) {
        ai.sa = sa;
        ai.holder = "locals->";
        ai.expr = e;

        coutput("%s\n",
                UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_assignments, &ai,
                               "", "  ", "\n", "\n"));
        ai.holder = "";
        coutput("%s\n",
                UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_used_in_expr, &ai,
                               "", "  (void)", ";\n", ";"));
    }

    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "locals";
    
    coutput("  (void)__parsec_tp; (void)locals;\n"
            "  return %s;\n"
            "}\n",
            dump_expr((void**)e, &info));
    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_range_min_without_fn(const jdf_t *jdf, const jdf_expr_t *expr,
                                              const char *ret_name, char *asname)
{
    string_arena_t *sa = string_arena_new(64);
    expr_info_t info = EMPTY_EXPR_INFO;
    jdf_expr_t *ld;

    (void)jdf;
    
    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = asname;

    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        assert(ld->ldef_index != -1);
        if( JDF_RANGE == ld->op )
            coutput("  { /* New scope for local definition '%s' */ \n"
                    "    int %s = %s;\n"
                    "    %s->ldef[%d].value = %s;\n",
                    ld->alias,
                    ld->alias, dump_expr((void**)ld->jdf_ta1, &info),
                    asname, ld->ldef_index, ld->alias);
        else
             coutput("  { /* New scope for local definition '%s' */ \n"
                     "    int %s = %s;\n"
                     "    %s->ldef[%d].value = %s;\n",
                     ld->alias,
                     ld->alias, dump_expr((void**)ld, &info),
                     asname, ld->ldef_index, ld->alias);
    }
    if( JDF_RANGE == expr->op )
        coutput("  %s = %s;\n", ret_name, dump_expr((void**)expr->jdf_ta1, &info));
    else
        coutput("  %s = %s;\n", ret_name, dump_expr((void**)expr, &info));
    
    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        coutput("  }\n");
    }

    string_arena_free(sa);
}

static void jdf_generate_range_min(const jdf_t *jdf, const jdf_function_entry_t *f, const jdf_expr_t *expr,
                                   const char *fn_name)
{
    assignment_info_t ai;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);

    coutput("static inline int32_t %s_fct(const __parsec_%s_internal_taskpool_t *__parsec_tp, %s *locals)\n"
            "{\n"
            "  int32_t __parsec_ret;\n",
            fn_name, jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"));

    ai.sa = sa;
    ai.holder = "locals->";
    ai.expr = expr;
    coutput("%s\n",
            UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_assignments, &ai,
                           "", "  ", "\n", "\n"));
    string_arena_free(sa);
    string_arena_free(sa2);

    jdf_generate_range_min_without_fn(jdf, expr, "__parsec_ret", "locals");
    coutput("  (void)__parsec_tp;\n"
            "  return __parsec_ret;\n"
            "}\n");

    coutput("static const parsec_expr_t %s = {\n"
            "  .op = PARSEC_EXPR_OP_INLINE,\n"
            "  .u_expr.v_func = { .type = %s, /* PARSEC_RETURN_TYPE_INT32 */\n"
            "                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)%s_fct }\n"
            "                   }\n"
            "};\n",
            fn_name, enum_type_name(0), fn_name);
}

static void jdf_generate_range_max(const jdf_t *jdf, const jdf_function_entry_t *f, const jdf_expr_t *expr,
                                   const char *fn_name)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info = EMPTY_EXPR_INFO;
    assignment_info_t ai;
    jdf_expr_t *ld;

    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "locals";

    coutput("static inline int32_t %s_fct(const __parsec_%s_internal_taskpool_t *__parsec_tp, %s *locals)\n"
            "{\n"
            "  int32_t __parsec_ret;\n",
            fn_name, jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"));

    ai.sa = sa;
    ai.holder = "locals->";
    ai.expr = expr;
    coutput("%s\n",
            UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_assignments, &ai,
                           "", "  ", "\n", "\n"));
    
    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        assert(ld->ldef_index != -1);
        if( JDF_RANGE == ld->op )
            coutput("  { /* New scope for local definition '%s' */ \n"
                    "    int %s = %s;\n"
                    "    locals->ldef[%d].value = %s;\n",
                    ld->alias,
                    ld->alias, dump_expr((void**)ld->jdf_ta2, &info),
                    ld->ldef_index, ld->alias);
        else
            coutput("  { /* New scope for local definition '%s' */ \n"
                    "    int %s = %s;\n"
                    "    locals->ldef[%d].value = %s;\n",
                    ld->alias,
                    ld->alias, dump_expr((void**)ld, &info),
                    ld->ldef_index, ld->alias);
    }
    if( JDF_RANGE == expr->op )
        coutput("  __parsec_ret = %s;\n", dump_expr((void**)expr->jdf_ta2, &info));
    else
        coutput("  __parsec_ret = %s;\n", dump_expr((void**)expr, &info));
    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        coutput("  }\n");
    }
    coutput("  (void)__parsec_tp; (void)locals;\n"
            "  return __parsec_ret;\n"
            "}\n");

    coutput("static const parsec_expr_t %s = {\n"
            "  .op = PARSEC_EXPR_OP_INLINE,\n"
            "  .u_expr.v_func = { .type = %s, /* PARSEC_RETURN_TYPE_INT32 */\n"
            "                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)%s_fct }\n"
            "                   }\n"
            "};\n",
            fn_name, enum_type_name(0), fn_name);

    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_range_increment(const jdf_t *jdf, const jdf_function_entry_t *f,
                                         jdf_expr_t *expr, const char *fn_name)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info = EMPTY_EXPR_INFO;
    assignment_info_t ai;
    jdf_expr_t *ld;

    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "locals";

    coutput("static inline int32_t %s_fct(const __parsec_%s_internal_taskpool_t *__parsec_tp, %s *locals)\n"
            "{\n"
            "  int32_t __parsec_ret;\n",
            fn_name, jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"));

    ai.sa = sa;
    ai.holder = "locals->";
    ai.expr = expr;
    coutput("%s\n",
            UTIL_DUMP_LIST(sa2, f->locals, next, dump_local_assignments, &ai,
                           "", "  ", "\n", "\n"));
    
    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        assert(-1 != ld->ldef_index);
        coutput("  { /* New scope for local index '%s' */ \n"
                "    int %s = locals->ldef[%d].value;\n",
                ld->alias,
                ld->alias, ld->ldef_index);
        if( JDF_RANGE == ld->op ) 
            coutput("    %s += %s;\n"
                    "    locals->ldef[%d].value = %s;\n",
                    ld->alias, dump_expr((void**)ld->jdf_ta3, &info),
                    ld->ldef_index, ld->alias);            
    }
    coutput("  __parsec_ret = %s;\n", dump_expr((void**)expr, &info));
    for(ld = jdf_expr_lv_first(expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(expr->local_variables, ld)) {
        coutput("  }\n");
    }
    coutput("  (void)__parsec_tp;\n"
            "  return __parsec_ret;\n"
            "}\n");

    coutput("static const parsec_expr_t %s = {\n"
            "  .op = PARSEC_EXPR_OP_INLINE,\n"
            "  .u_expr.v_func = { .type = %s, /* PARSEC_RETURN_TYPE_INT32 */\n"
            "                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)%s_fct }\n"
            "                   }\n"
            "};\n",
            fn_name, enum_type_name(0), fn_name);

    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_expression( const jdf_t *jdf, const jdf_function_entry_t *f,
                                     jdf_expr_t *e, const char *name)
{
    if (NULL != JDF_OBJECT_ONAME(e)) return;

    JDF_OBJECT_ONAME(e) = strdup(name);

    if( jdf_expr_is_range(e) ) {
        char *subf = (char*)malloc(strlen(JDF_OBJECT_ONAME(e)) + 64);
        sprintf(subf, "rangemin_of_%s", JDF_OBJECT_ONAME(e));
        jdf_generate_range_min(jdf, f, e, subf);
        sprintf(subf, "rangemax_of_%s", JDF_OBJECT_ONAME(e));
        jdf_generate_range_max(jdf, f, e, subf);

        if( e->op == JDF_RANGE && e->jdf_ta3->op == JDF_CST ) {
            coutput("static const parsec_expr_t %s = {\n"
                    "  .op = PARSEC_EXPR_OP_RANGE_CST_INCREMENT,\n"
                    "  .u_expr.range = {\n"
                    "    .op1 = &rangemin_of_%s,\n"
                    "    .op2 = &rangemax_of_%s,\n"
                    "    .increment.cst = %d\n"
                    "  }\n"
                    "};\n",
                    JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), e->jdf_ta3->jdf_cst );
        } else {
            sprintf(subf, "rangeincrement_of_%s", JDF_OBJECT_ONAME(e));
            jdf_generate_range_increment(jdf, f, e, subf);
            coutput("static const parsec_expr_t %s = {\n"
                    "  .op = PARSEC_EXPR_OP_RANGE_EXPR_INCREMENT,\n"
                    "  .u_expr.range = {\n"
                    "    .op1 = &rangemin_of_%s,\n"
                    "    .op2 = &rangemax_of_%s,\n"
                    "    .increment.expr = &rangeincrement_of_%s\n"
                    "  }\n"
                    "};\n",
                    JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e), JDF_OBJECT_ONAME(e));
        }
    } else if (e->op == JDF_C_CODE || e->op == JDF_CST) {
        jdf_generate_function_without_expression(jdf, f, e, JDF_OBJECT_ONAME(e), "_fct", full_type[e->jdf_type]);

        coutput("static const parsec_expr_t %s = {\n"
                "  .op = PARSEC_EXPR_OP_INLINE,\n"
                "  .u_expr.v_func = { .type = %s,\n"
                "                     .func = { .inline_func_%s = (parsec_expr_op_%s_inline_func_t)%s_fct }\n"
                "                   }\n"
                "};\n", JDF_OBJECT_ONAME(e), enum_type_name(e->jdf_type), short_type[e->jdf_type],
                short_type[e->jdf_type], JDF_OBJECT_ONAME(e));
    } else {
        jdf_generate_function_without_expression(jdf, f, e, JDF_OBJECT_ONAME(e), "_fct", "int");

        coutput("static const parsec_expr_t %s = {\n"
                "  .op = PARSEC_EXPR_OP_INLINE,\n"
                "  .u_expr.v_func = { .type = %s, /* PARSEC_RETURN_TYPE_INT32 */\n"
                "                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)%s_fct }\n"
                "                   }\n"
                "};\n",
                JDF_OBJECT_ONAME(e),
                enum_type_name(0),
                JDF_OBJECT_ONAME(e));
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
    expr_info_t info = EMPTY_EXPR_INFO;
    const jdf_call_t *data_affinity = f->predicate;

    (void)jdf;

    if( data_affinity->var != NULL ) {
        fprintf(stderr, "Internal Error: data affinity must reference a data, not a complete call (%s:%d).\n",
                data_affinity->super.filename, data_affinity->super.lineno);
        assert( NULL == data_affinity->var );
    }

    coutput("static inline int %s(%s *this_task,\n"
            "                     parsec_data_ref_t *ref)\n"
            "{\n"
            "    const __parsec_%s_internal_taskpool_t *__parsec_tp = (const __parsec_%s_internal_taskpool_t*)this_task->taskpool;\n",
            name, parsec_get_name(jdf, f, "task_t"),
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
            "%s\n"
            "  ref->dc = (parsec_data_collection_t *)"TASKPOOL_GLOBAL_PREFIX"_g_%s;\n"
            "  /* Compute data key */\n"
            "  ref->key = ref->dc->data_key(ref->dc, %s);\n"
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

#if 0
static void jdf_generate_initfinal_data_for_call(const jdf_call_t *call,
                                                 string_arena_t* sa,
                                                 int il)
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    expr_info_t info = EMPTY_EXPR_INFO;

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    assert( call->var == NULL );
    if ( call->parameters != NULL ) {
        string_arena_add_string(sa, "%s    __d = (parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s;\n"
                                "%s    refs[__flow_nb].dc = __d;\n",
                                indent(il), call->func_or_mem,
                                indent(il));
        string_arena_add_string(sa, "%s    refs[__flow_nb].key = __d->data_key(__d, %s);\n"
                                "%s    __flow_nb++;\n",
                                indent(il), UTIL_DUMP_LIST(sa1, call->parameters, next,
                                                           dump_expr, (void*)&info,
                                                           "", "", ", ", ""),
                                indent(il));
    }
    else {
        /* TODO */
        string_arena_add_string(sa,
                                "%s    refs[__flow_nb].dc = NULL;\n"
                                "%s    refs[__flow_nb].key = 0xffffffff;\n"
                                "%s    __flow_nb++;\n"
                                "%s    (void)__d;\n",
                                indent(il), indent(il), indent(il), indent(il));
    }

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static int jdf_generate_initfinal_data_for_dep(const jdf_dep_t *dep,
                                               string_arena_t* sa)
{
    string_arena_t *sa1 = string_arena_new(64);
    expr_info_t info = EMPTY_EXPR_INFO;
    int ret = 0;

    info.sa = sa1;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    switch( dep->guard->guard_type ) {
    case JDF_GUARD_UNCONDITIONAL:
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

    string_arena_free(sa1);
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
                "                     parsec_data_ref_t *refs)\n"
                "{\n"
                "    const __parsec_%s_internal_taskpool_t *__parsec_tp = (const __parsec_%s_internal_taskpool_t*)this_task->taskpool;\n"
                "    parsec_data_collection_t *__d = NULL;\n"
                "    int __flow_nb = 0;\n",
                name, parsec_get_name(jdf, f, "task_t"),
                jdf_basename, jdf_basename);


        ai.sa = sa2;
        ai.holder = "this_task->locals.";
        ai.expr = NULL;
        coutput("%s\n"
                "    /* Silent Warnings: should look into predicate to know what variables are usefull */\n"
                "    (void)__parsec_tp;\n"
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
#endif  /* 0 */

static void jdf_generate_symbols( const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix )
{
    string_arena_t *sa = string_arena_new(64);
    jdf_variable_list_t *v;
    char *exprname;
    int id, rc;

    for(id = 0, v = f->locals; v != NULL; id++, v = v->next) {
        rc = asprintf( &JDF_OBJECT_ONAME(v), "%s%s", prefix, v->name );
        assert( rc != -1 );

        exprname = (char*)malloc(strlen(JDF_OBJECT_ONAME(v)) + 16);
        string_arena_init(sa);

        string_arena_add_string(sa, "static const parsec_symbol_t %s = { .name = \"%s\", .context_index = %d, ", JDF_OBJECT_ONAME(v), v->name, id);

        if( v->expr->op == JDF_RANGE ) {
            sprintf(exprname, "minexpr_of_%s", JDF_OBJECT_ONAME(v));
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            jdf_generate_expression(jdf, f, v->expr->jdf_ta1, exprname);

            sprintf(exprname, "maxexpr_of_%s", JDF_OBJECT_ONAME(v));
            string_arena_add_string(sa, ".max = &%s, ", exprname);
            jdf_generate_expression(jdf, f, v->expr->jdf_ta2, exprname);

            if( v->expr->jdf_ta3->op == JDF_CST ) {
                string_arena_add_string(sa, ".cst_inc = %d, .expr_inc = NULL, ", v->expr->jdf_ta3->jdf_cst);
            } else {
                sprintf(exprname, "incexpr_of_%s", JDF_OBJECT_ONAME(v));
                string_arena_add_string(sa, ".cst_inc = 0, .expr_inc = &%s, ", exprname);
                jdf_generate_expression(jdf, f, v->expr->jdf_ta3, exprname);
            }
        } else {
            sprintf(exprname, "expr_of_%s", JDF_OBJECT_ONAME(v));
            string_arena_add_string(sa, ".min = &%s, ", exprname);
            string_arena_add_string(sa, ".max = &%s, .cst_inc = 0, .expr_inc = NULL, ", exprname);
            jdf_generate_expression(jdf, f, v->expr, exprname);
        }

        if( jdf_symbol_is_global(jdf->globals, v->name) ) {
            string_arena_add_string(sa, " .flags = PARSEC_SYMBOL_IS_GLOBAL");
        } else if ( jdf_symbol_is_standalone(v->name, jdf->globals, v->expr) ) {
            string_arena_add_string(sa, " .flags = PARSEC_SYMBOL_IS_STANDALONE");
        } else {
            string_arena_add_string(sa, " .flags = 0x0");
        }
        string_arena_add_string(sa, "};");
        coutput("%s\n\n", string_arena_get_string(sa));
        free(exprname);
    }

    string_arena_free(sa);
    (void)rc;
}

static void jdf_generate_ctl_gather_compute(const jdf_t *jdf, const jdf_function_entry_t* of,
                                            const char *tname, const char *fname,
                                            const jdf_call_t *call, const jdf_dep_t *dep)
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    const jdf_expr_t *params = call->parameters;
    expr_info_t info1 = EMPTY_EXPR_INFO, info2 = EMPTY_EXPR_INFO, info3 = EMPTY_EXPR_INFO;
    const jdf_expr_t *le;
    const jdf_function_entry_t *targetf;
    const jdf_param_list_t *pl;
    int nbopen, depdone;
    assignment_info_t ai;
    jdf_expr_t *ld, *local_defs;

    targetf = find_target_function(jdf, tname);
    assert(targetf != NULL);

    coutput("static inline int %s_fct(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *assignments)\n"
            "{\n"
            "  int   __nb_found = 0;\n"
            "  %s "JDF2C_NAMESPACE"_tmp_locals = *assignments;\n"
            "  (void)__parsec_tp;\n",
            fname, jdf_basename, parsec_get_name(jdf, of, "parsec_assignment_t"), parsec_get_name(jdf, of, "parsec_assignment_t"));

    info1.sa = sa1;
    info1.prefix = "";
    info1.suffix = "";
    info1.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    info2.sa = sa2;
    info2.prefix = "";
    info2.suffix = "";
    info2.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    info3.sa = sa3;
    info3.prefix = "";
    info3.suffix = "";
    info3.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    ai.sa = sa2;
    ai.holder = JDF2C_NAMESPACE"_tmp_locals.";
    ai.expr = NULL;
    coutput( "%s",
             UTIL_DUMP_LIST(sa1, of->locals, next,
                            dump_local_assignments, &ai, "", "  ", "\n", "\n"));
    coutput( "%s\n",
             UTIL_DUMP_LIST_FIELD(sa2, of->locals, next, name, dump_string, NULL,
                                  "  ", "(void)", "; ", ";"));

    if( NULL != dep->local_defs ) {
        local_defs = dep->local_defs;
        depdone = 0;
    } else {
        local_defs = call->local_defs;
        depdone = 1;
    }
    nbopen=0;
    for(ld = jdf_expr_lv_first(local_defs); ld != NULL; ld = jdf_expr_lv_next(local_defs, ld)) {
        assert(ld->alias != NULL);
        assert(-1 != ld->ldef_index);
        if( !depdone ) {
            if( ld == call->local_defs ) {
                depdone=1;
            } else {
                coutput("%s  int %s = "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value;\n"
                        "%s  (void)%s;\n",
                        indent(nbopen), ld->alias, ld->ldef_index,
                        indent(nbopen), ld->alias);
                continue;
            }
        }
        if( ld->op == JDF_RANGE ) {
            coutput("%s  {\n"
                    "%s    int %s;\n"
                    "%s    for(%s  = %s;\n"
                    "%s        %s <= %s;\n"
                    "%s        %s += %s) {\n"
                    "%s      "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                    indent(nbopen),
                    indent(nbopen), ld->alias,
                    indent(nbopen), ld->alias, dump_expr( (void**)ld->jdf_ta1, &info1 ),
                    indent(nbopen), ld->alias, dump_expr( (void**)ld->jdf_ta2, &info2 ),
                    indent(nbopen), ld->alias, dump_expr( (void**)ld->jdf_ta3, &info3 ),
                    indent(nbopen), ld->ldef_index, ld->alias);
            nbopen+=2;
        } else {
            coutput("%s  {\n"
                    "%s    int %s = "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                    indent(nbopen),
                    indent(nbopen), ld->alias, ld->ldef_index, dump_expr( (void**)ld, &info1 ));
            nbopen+=1;
        }
        
    }
    for(pl = targetf->parameters, le = params; NULL != le; pl = pl->next, le = le->next) {
        if( le->op == JDF_RANGE ) {
            coutput("%s  {\n"
                    "%s    int %s_%s;\n"
                    "%s    for(%s_%s  = %s;\n"
                    "%s        %s_%s <= %s;\n"
                    "%s        %s_%s += %s) {\n",
                    indent(nbopen),
                    indent(nbopen), targetf->fname, pl->name,
                    indent(nbopen), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta1, &info1 ),
                    indent(nbopen), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta2, &info2 ),
                    indent(nbopen), targetf->fname, pl->name, dump_expr( (void**)le->jdf_ta3, &info3 ));
            nbopen+=2;
        } else {
            coutput("%s  int %s_%s = %s;\n"
                    "%s  (void)%s_%s;\n",
                    indent(nbopen), targetf->fname, pl->name, dump_expr( (void**)le, &info1 ),
                    indent(nbopen), targetf->fname, pl->name);
        }                    
    }
    coutput("%s  __nb_found++;\n", indent(nbopen));
    nbopen--;
    for(; nbopen > -1; nbopen--) {
        coutput("%s  }\n", indent(nbopen));
    }
    coutput("  return __nb_found;\n"
            "}\n"
            "\n"
            "static const parsec_expr_t %s = {\n"
            "  .op = PARSEC_EXPR_OP_INLINE,\n"
            "  .u_expr.v_func = { .type = %s, /* PARSEC_RETURN_TYPE_INT32 */\n"
            "                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)%s_fct }\n"
            "                   }\n"
            "};\n\n", fname, enum_type_name(0), fname);

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
    expr_info_t info = EMPTY_EXPR_INFO;
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

    coutput("static parsec_data_t *%s(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *assignments)\n"
            "{\n"
            "%s\n"
            "  /* Silence Warnings: should look into parameters to know what variables are useful */\n"
            "%s\n"
            "  if( __parsec_tp->super.super.context->my_rank == (int32_t)rank_of_%s(%s) )\n"
            "    return data_of_%s(%s);\n"
            "  return NULL;\n"
            "}\n"
            "\n",
            function_name, jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"),
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"),
            UTIL_DUMP_LIST_FIELD(sa3, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";\n", ";"),
            mem, string_arena_get_string(sa4),
            mem, string_arena_get_string(sa4));

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
    string_arena_t *sa = string_arena_new(64), *sa2 = string_arena_new(64), *sa3 = string_arena_new(64);
    jdf_expr_t *le;
    int ret = 1;
    string_arena_t *tmp_fct_name;

    JDF_OBJECT_ONAME(call) = strdup(depname);

    if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
        /* First: do we have a control gather because of the parameters? */
        for( le = call->parameters; le != NULL; le = le->next ) {
            if( le->op == JDF_RANGE ) {
                break;
            }
        }
        if( NULL == le ) {
            /* If not, do we have one because the call has a ranged local definition ? */
            for(le = call->local_defs; le != NULL; le = le->next) {
                if( le->op == JDF_RANGE ) {
                    break;
                }
            }
            if( NULL == le ) {
                /* Last, do we have one because the dep has a ranged local definition ? */
                for(le = dep->local_defs; le != NULL; le = le->next) {
                    if( le->op == JDF_RANGE ) {
                        break;
                    }
                }
            }
        }
        if( NULL != le ) {
            /* At least one range in input: must be a control gather */
            if( !(flow->flow_flags & JDF_FLOW_TYPE_CTL) ) {
                jdf_fatal(JDF_OBJECT_LINENO(dep), "This dependency features a range as input but is not a Control dependency\n");
                exit(1);
            }
            string_arena_add_string(sa2, "&ctl_gather_compute_for_dep_%s", depname);
            jdf_generate_ctl_gather_compute(jdf, f, call->func_or_mem,
                                            string_arena_get_string(sa2)+1, call, dep);
            ret = 0;
        } else {
            string_arena_add_string(sa2, "NULL");
        }
    }else {
        string_arena_add_string(sa2, "NULL");
    }

    if( NULL != dep->guard->guard ) {  /* Dump a comment with the dep condition */
        expr_info_t info = { .prefix = "", .suffix = "", .assignments = "assignments", .nb_bound_locals = 0, .bound_locals = NULL};
        string_arena_init(sa3);
        info.sa = sa3;
        dump_expr((void**)dep->guard->guard, &info);
    }
    string_arena_add_string(sa,
                            "static const parsec_dep_t %s = {\n"
                            "  .cond = %s,  /* %s%s */\n"
                            "  .ctl_gather_nb = %s,\n",
                            JDF_OBJECT_ONAME(call),
                            condname, (call == dep->guard->calltrue ? "" : "!"), string_arena_get_string(sa3),
                            string_arena_get_string(sa2));

    if( NULL != call->var ) {
        jdf_function_entry_t* pf;
        pf = find_target_function(jdf, call->func_or_mem);
        if( NULL == pf ) {
            fprintf(stderr, "Error: Can't identify the target function for the call at %s.jdf:%d: %s %s\n",
                   jdf_basename, call->super.lineno, call->var, call->func_or_mem);
            exit(-1);
        }
        if( NULL != pf ) {
            string_arena_add_string(sa,
                                    "  .task_class_id = %d, /* %s_%s */\n",
                                    pf->task_class_id, jdf_basename, call->func_or_mem);
        } else {
            string_arena_add_string(sa,
                                    "  .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID, /* %s_%s */\n",
                                    jdf_basename, call->func_or_mem);
        }
        string_arena_add_string(sa,
                                "  .direct_data = (parsec_data_lookup_func_t)NULL,\n"
                                "  .flow = &flow_of_%s_%s_for_%s,\n",
                                jdf_basename, call->func_or_mem, call->var);
    } else {
        if ( NULL != call->parameters ) {
            tmp_fct_name = string_arena_new(64);
            string_arena_add_string(tmp_fct_name, "%s_direct_access", JDF_OBJECT_ONAME(dep));
            jdf_generate_direct_data_function(jdf, call->func_or_mem, call->parameters, f,
                                              string_arena_get_string(tmp_fct_name));
            string_arena_add_string(sa,
                                    "  .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID, /* %s_%s */\n"
                                    "  .direct_data = (parsec_data_lookup_func_t)&%s,\n",
                                    jdf_basename, call->func_or_mem,
                                    string_arena_get_string(tmp_fct_name));
            string_arena_free(tmp_fct_name);
        }
        else {
            string_arena_add_string(sa,
                                    "  .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID, /* %s_%s */\n"
                                    "  .direct_data = (parsec_data_lookup_func_t)NULL,\n",
                                    jdf_basename, call->func_or_mem);
        }
    }
    string_arena_add_string(sa,
                            "  .dep_index = %d,\n"
                            "  .dep_datatype_index = %d,\n"
                            "  .belongs_to = &%s,\n"
                            "};\n",
                            dep->dep_index,
                            dep->dep_datatype_index,
                            JDF_OBJECT_ONAME(flow));

    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);

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
        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
            continue; /* Skip type declaration for WRITE-only flows */
        }

        sprintf(depname, "%s_dep%d_atline_%d", JDF_OBJECT_ONAME(flow), depid, JDF_OBJECT_LINENO(dl));
        JDF_OBJECT_ONAME(dl) = strdup(depname);

        if( dl->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
            sprintf(condname, "NULL");
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
            jdf_expr_t not = jdf_empty_expr;

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
                ((alldeps_type & JDF_DEP_FLOW_OUT) ? "PARSEC_SYM_INOUT" : "PARSEC_SYM_IN") : "PARSEC_SYM_OUT");

    string_arena_init(flow_flags);
    string_arena_add_string(flow_flags, "%s", ((flow->flow_flags & JDF_FLOW_TYPE_CTL) ? "PARSEC_FLOW_ACCESS_NONE" :
                                               ((flow->flow_flags & JDF_FLOW_TYPE_READ) ?
                                                ((flow->flow_flags & JDF_FLOW_TYPE_WRITE) ? "PARSEC_FLOW_ACCESS_RW" : "PARSEC_FLOW_ACCESS_READ") : "PARSEC_FLOW_ACCESS_WRITE")));
    if(flow->flow_flags & JDF_FLOW_HAS_IN_DEPS)
        string_arena_add_string(flow_flags, "|PARSEC_FLOW_HAS_IN_DEPS");

    if(strlen(string_arena_get_string(sa_dep_in)) == 0) {
        string_arena_add_string(sa_dep_in, "NULL");
    } else {
        int deps_in = 0;
        for(dl = flow->deps; NULL != dl; dl = dl->next) {
            deps_in += !!(dl->dep_flags & JDF_DEP_FLOW_IN);
        }
        string_arena_add_string(sa,
                                "#if MAX_DEP_IN_COUNT < %d  /* number of input dependencies */\n"
                                "    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=%d] but found %d). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.\n"
                                "#endif\n",
                                deps_in, MAX_DEP_IN_COUNT, deps_in);
    }
    if(strlen(string_arena_get_string(sa_dep_out)) == 0) {
        string_arena_add_string(sa_dep_out, "NULL");
    } else {
        int deps_out = 0;
        for(dl = flow->deps; NULL != dl; dl = dl->next) {
            deps_out += !!(dl->dep_flags & JDF_DEP_FLOW_OUT);
        }
        string_arena_add_string(sa,
                                "#if MAX_DEP_OUT_COUNT < %d  /* number of output dependencies */\n"
                                "    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=%d] but found %d). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.\n"
                                "#endif\n",
                                deps_out, MAX_DEP_OUT_COUNT, deps_out);
    }
    string_arena_add_string(sa,
                            "\nstatic const parsec_flow_t %s = {\n"
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
    return indepnorange && (((parsec_dependency_t)(((1 << flow->flow_index) & 0x1fffffff /*~PARSEC_DEPENDENCIES_BITMASK*/))) != 0);
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

static void jdf_generate_direct_input_conditions(const jdf_t *jdf, const jdf_function_entry_t *f, const jdf_dataflow_t *dataflow)
{
    const jdf_dataflow_t *flow, *next_flow;
    jdf_dep_t *dep;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa = string_arena_new(64);
    int write_next_label, continue_if_true, goto_if_false, goto_if_true, skip_continue;

    coutput("  %s "JDF2C_NAMESPACE"_tmp_locals = *(%s*)&this_task->locals;\n"
            "  (void)"JDF2C_NAMESPACE"_tmp_locals;\n",
            parsec_get_name(jdf, f, "parsec_assignment_t"),
            parsec_get_name(jdf, f, "parsec_assignment_t"));

    info.prefix = "";
    info.suffix = "";
    info.sa = sa;
    info.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    write_next_label = 0;

    /* First, we point to the first flow with a DEP_IN */
    for(flow = dataflow; NULL != flow; flow = flow->next) {
        for(dep = flow->deps; NULL != dep; dep = dep->next)
            if( JDF_DEP_FLOW_IN & dep->dep_flags )
                break;
        if(NULL == dep) continue;
        break;
    }

    for(; NULL != flow; flow = next_flow) {
        /* Then, we compute next_flow as the next one with a DEP_IN. This flow is
         * needed to be able to generate the goto label to jump from one flow to
         * the next.
         */
        for(next_flow = flow->next; NULL != next_flow; next_flow = next_flow->next) {
            for(dep = next_flow->deps; NULL != dep; dep = dep->next)
                if( JDF_DEP_FLOW_IN & dep->dep_flags )
                    break;
            if(NULL == dep) continue;
            break;
        }

        if(write_next_label) {
            coutput(" %s_check_flow:\n", flow->varname);
            write_next_label = 0;
        }
        skip_continue = 0;

        for(dep = flow->deps; NULL != dep; dep = dep->next) {
            if( ! (dep->dep_flags & JDF_DEP_FLOW_IN) ) {
                continue;
            }
            continue_if_true = 0;
            goto_if_true = 0;
            goto_if_false = 0;

            if( dep->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
                /* We cannot be a control flow, or has_ready_input_dependency would have returned false */
                assert( 0 == (JDF_FLOW_TYPE_CTL & flow->flow_flags) );
                /* We are necessarily depending on a direct memory, for the same reason */
                assert( NULL == dep->guard->calltrue->var );
                coutput("  /* Flow for %s is always a memory reference */\n", flow->varname);
                skip_continue = 1;  /* no need to complete the flow with a continue */
                break; /* No need to go check other cases, no need to print the flow label, or the continue */
            } else if( dep->guard->guard_type == JDF_GUARD_BINARY ) {
                if(NULL != dep->guard->calltrue->var) {
                    /* Either we have a <- cond ? A task(), in which case we want to discard the task if cond is true
                     *     or we have a <- cond ? CTL task(), in which case we also want to discard the task if cond is true */
                    continue_if_true = 1;
                } else {
                    /* Here, we have found a direct memory access */
                    assert( 0 == (JDF_FLOW_TYPE_CTL & flow->flow_flags) );
                    goto_if_true = 1;
                }
            } else if( dep->guard->guard_type == JDF_GUARD_TERNARY ) {
                /* We cannot be a control flow in ternary, or there would always be a case where we cannot be a startup task */
                assert( 0 == (JDF_FLOW_TYPE_CTL & flow->flow_flags) );

                if( (NULL == dep->guard->calltrue->var) && (NULL == dep->guard->callfalse->var) ) {
                    /* No condition, we always depend on a direct memory reference */
                    coutput("  /* Flow for %s is always a memory reference */\n", flow->varname);
                    skip_continue = 1;  /* no need to complete the flow with a continue */
                    break; /* No need to go check other cases, no need to print the flow label, or the continue */
                } else {
                    if( NULL == dep->guard->calltrue->var ) {
                        assert( NULL != dep->guard->callfalse->var );
                        goto_if_true = 1;
                    } else {
                        assert( NULL == dep->guard->callfalse->var );
                        goto_if_false = 1;
                    }
                }
            }

            assert( continue_if_true || goto_if_false || goto_if_true );
            jdf_expr_t *ld;
            if( NULL != dep->guard->guard->local_variables ) {
                for(ld = jdf_expr_lv_first(dep->guard->guard->local_variables);
                    ld != NULL; ld = jdf_expr_lv_next(dep->guard->guard->local_variables, ld)) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    coutput("  int %s;\n", ld->alias);
                    if(JDF_RANGE == ld->op) {
                        coutput("  for( %s = %s;",
                                ld->alias, dump_expr((void**)ld->jdf_ta1, &info));
                        coutput("%s <= %s; %s+=",
                                ld->alias, dump_expr((void**)ld->jdf_ta2, &info), ld->alias);
                        coutput("%s) {\n"
                                "     "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                                dump_expr((void**)ld->jdf_ta3, &info),
                                ld->ldef_index, ld->alias);
                    } else {
                        coutput("  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s = %s;\n",
                                ld->ldef_index, ld->alias, dump_expr((void**)ld, &info));
                    }
                }
            }

            if( goto_if_false || goto_if_true ) {
                char *nextname = (NULL == next_flow) ? JDF2C_NAMESPACE"done" : next_flow->varname;
                coutput("  if( %s%s%s ) goto %s_check_flow;\n",
                        goto_if_false ? "!(" : "",
                        dump_expr((void**)dep->guard->guard, &info),
                        goto_if_false ? ")" : "",
                        nextname);
                write_next_label = 1;
            } if( continue_if_true) {
                coutput("  if( %s ) continue; /* %s %s() is not a memory reference for flow %s */\n",
                        dump_expr((void**)dep->guard->guard, &info),
                        dep->guard->calltrue->var, dep->guard->calltrue->func_or_mem,
                        flow->varname);
            }
            if( NULL != dep->guard->guard->local_variables ) {
                for(ld = jdf_expr_lv_first(dep->guard->guard->local_variables);
                    ld != NULL; ld = jdf_expr_lv_next(dep->guard->guard->local_variables, ld)) {
                    if(JDF_RANGE == ld->op) {
                        coutput("  }\n");
                    }
                }
            }
        }

        if(write_next_label && !skip_continue) {
            coutput("  continue; /* All other cases are not startup tasks */\n");
        }
    }
    if(write_next_label) {
        coutput(" "JDF2C_NAMESPACE"done_check_flow:\n");
    }

    string_arena_free(sa);
}

/**
 * Note about the lifecycle of tasks coming from JDF:
 * For each task class, we generate a task class to execute its initializations
 * and creation of initial tasks in parallel.
 * The initialization of the structures (dependency tracking and data flow
 * repositories) are executed in the %s_internal_init functions, bound to the
 * prepare_input hook, and the creation of the initial tasks executed in the
 * %s_startup_tasks functions bound to the incarnation hook.
 *
 * internal_init is supposed to return ASYNC until the last prepare_input has
 * been executed, so that the hook is not triggered before everything is prepared.
 * Then, when the last prepare_input has been triggered, it parsec_taskpool_enable
 * the startup_queue, on which all the startup tasks are chained.
 * parsec_taskpool_enable changes their status to PARSEC_TASK_STATUS_HOOK, which
 * is higher than PREPARE_INPUT, and put them back in the scheduling list. Thus,
 * when they are selected again, they skip the prepare_input step, and go
 * directly to the hook step, that executes the creation of the initial tasks.
 */
static void jdf_generate_startup_tasks(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{
    string_arena_t *sa1, *sa2, *sa_properties;
    jdf_variable_list_t *vl, *inner_vl = NULL;
    int nesting = 0, idx, nb_locals;
    expr_info_t info1 = EMPTY_EXPR_INFO;
    jdf_expr_t *ld;
    int ctx_level = 0;

    assert( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP );
    if( f->user_defines & JDF_FUNCTION_HAS_UD_STARTUP_TASKS_FUN )
        return;
    (void)jdf;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa_properties = string_arena_new(64);

    coutput("static int %s(parsec_execution_stream_t * es, %s *this_task)\n"
            "{\n"
            "  %s* new_task;\n"
            "  __parsec_%s_internal_taskpool_t* __parsec_tp = (__parsec_%s_internal_taskpool_t*)this_task->taskpool;\n"
            "  parsec_context_t *context = __parsec_tp->super.super.context;\n"
            "  int vpid = 0, nb_tasks = 0;\n"
            "  size_t total_nb_tasks = 0;\n"
            "  parsec_list_item_t* pready_ring[context->nb_vp];\n"
            "  int restore_context = 0;\n",
            fname, parsec_get_name(jdf, f, "task_t"),
            parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename);

    for(vl = f->locals; vl != NULL; vl = vl->next)
        coutput("  int %s = this_task->locals.%s.value;  /* retrieve value saved during the last iteration */\n", vl->name, vl->name);

    coutput("  for(int _i = 0; _i < context->nb_vp; pready_ring[_i++] = NULL );\n"
            "  if( 0 != this_task->locals.reserved[0].value ) {\n"
            "    this_task->locals.reserved[0].value = 1; /* reset the submission process */\n"
            "    restore_context = 1;\n"
            "    goto restore_context_0;\n"
            "  }\n"
            "  this_task->locals.reserved[0].value = 1; /* a sane default value */\n");

    string_arena_init(sa1);
    string_arena_init(sa2);

    info1.sa = sa1;
    info1.prefix = "";
    info1.suffix = "";
    info1.assignments = "&this_task->locals";

    idx = 0;
    for(vl = f->locals; vl != NULL; vl = vl->next, idx++) {
        if(vl->expr->op == JDF_RANGE) {
            coutput("%s  for(this_task->locals.%s.value = %s = %s;\n",
                    indent(nesting), vl->name, vl->name, dump_expr((void**)vl->expr->jdf_ta1, &info1));
            coutput("%s      this_task->locals.%s.value <= %s;\n",
                    indent(nesting), vl->name, dump_expr((void**)vl->expr->jdf_ta2, &info1));
            coutput("%s      this_task->locals.%s.value += %s, %s = this_task->locals.%s.value) {\n",
                    indent(nesting), vl->name, dump_expr((void**)vl->expr->jdf_ta3, &info1), vl->name, vl->name);
            nesting++;
        } else if( NULL != vl->expr->local_variables ) {
            for(ld = jdf_expr_lv_first(vl->expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(vl->expr->local_variables, ld)) {
                assert(-1 != ld->ldef_index);
                assert(NULL != ld->alias);
                if( ld->op == JDF_RANGE ) {
                    coutput("%s  { /* block for the local variable '%s' */\n"
                            "%s    int %s_loop_%s = 1;\n"
                            "%s    int %s;\n"
                            "%s    for(this_task->locals.ldef[%d].value = %s; %s_loop_%s == 1;",
                            indent(nesting), ld->alias,
                            indent(nesting), JDF2C_NAMESPACE, ld->alias,
                            indent(nesting), ld->alias,
                            indent(nesting), ld->ldef_index, dump_expr((void**)ld->jdf_ta1, &info1), JDF2C_NAMESPACE, ld->alias);
                    coutput(" this_task->locals.ldef[%d].value += %s ) { /* Arbitrary iterator on %s */ \n",
                            ld->ldef_index, dump_expr((void**)ld->jdf_ta3, &info1), ld->alias);
                    coutput("%s    restore_context_%d:\n"
                            "%s      %s = this_task->locals.ldef[%d].value;\n"
                            "%s      %s_loop_%s = (%s != %s); /* Execute once only when reaching the end; recompute every time in case we are restoring the context */\n"
                            "%s      if( restore_context ) goto restore_context_%d;\n",
                            indent(nesting), ctx_level,
                            indent(nesting), ld->alias, ld->ldef_index,
                            indent(nesting), JDF2C_NAMESPACE, ld->alias, ld->alias, dump_expr((void**)ld->jdf_ta2, &info1),
                            indent(nesting), ctx_level+1);
                    ctx_level++;
                    nesting+=2;
                } else {
                    coutput("%s  { /* block for the local variable '%s' */\n"
                            "%s     this_task->locals.ldef[%d].value = %s;\n"
                            "%s   restore_context_%d:\n"
                            "%s     %s = this_task->locals.ldef[%d].value;\n"
                            "%s     if( restore_context ) goto restore_context_%d;\n",
                            indent(nesting), ld->alias,
                            indent(nesting), ld->ldef_index, dump_expr((void**)ld, &info1),
                            indent(nesting), ctx_level,
                            indent(nesting), ld->alias, ld->ldef_index,
                            indent(nesting), ctx_level+1);
                    nesting++;
                    ctx_level++;
                }
            }
            coutput("%s    this_task->locals.%s.value = %s = %s;\n",
                    indent(nesting), vl->name, vl->name, dump_expr((void**)vl->expr, &info1));
        } else {
            coutput("%s  this_task->locals.%s.value = %s = %s;\n",
                    indent(nesting), vl->name, vl->name, dump_expr((void**)vl->expr, &info1));
        }
        inner_vl = vl;
    }
    coutput("%s  if( !%s_pred(%s) ) continue;\n",
            indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                                                              dump_string, NULL,
                                                              "", "", ", ", ""));

    /**
     * Dump all the conditions that can invalidate the startup propriety.
     */
    jdf_generate_direct_input_conditions(jdf, f, f->dataflow);

    coutput("%s  if( NULL != ((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->vpid_of ) {\n"
            "%s    vpid = ((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->vpid_of((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s, %s);\n"
            "%s    assert(context->nb_vp >= vpid);\n"
            "%s  }\n"
            "%s  new_task = (%s*)parsec_thread_mempool_allocate( context->virtual_processes[vpid]->execution_streams[0]->context_mempool );\n"
            "%s  new_task->status = PARSEC_TASK_STATUS_NONE;\n",
            indent(nesting), f->predicate->func_or_mem,
            indent(nesting), f->predicate->func_or_mem, f->predicate->func_or_mem,
            UTIL_DUMP_LIST(sa2, f->predicate->parameters, next,
                           dump_expr, (void*)&info1,
                           "", "", ", ", ""),
            indent(nesting),
            indent(nesting),
            indent(nesting), parsec_get_name(jdf, f, "task_t"),
            indent(nesting));

    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_variable_list_t, next, nb_locals);
    coutput("%s  /* Copy only the valid elements from this_task to new_task one */\n"
            "%s  new_task->taskpool   = this_task->taskpool;\n"
            "%s  new_task->task_class = __parsec_tp->super.super.task_classes_array[%s_%s.task_class_id];\n"
            "%s  new_task->chore_mask   = PARSEC_DEV_ALL;\n",
            indent(nesting),
            indent(nesting),
            indent(nesting), jdf_basename, f->fname,
            indent(nesting));
    for(vl = f->locals; vl != NULL; vl = vl->next, idx++)
        coutput("%s  new_task->locals.%s.value = this_task->locals.%s.value;\n", indent(nesting), vl->name, vl->name);

    coutput("%s  PARSEC_LIST_ITEM_SINGLETON(new_task);\n",
            indent(nesting));
    if( NULL != f->priority ) {
        coutput("%s  new_task->priority = __parsec_tp->super.super.priority + priority_of_%s_%s_as_expr_fct((__parsec_%s_internal_taskpool_t*)new_task->taskpool, &new_task->locals);\n",
                indent(nesting), jdf_basename, f->fname, jdf_basename);
    } else {
        coutput("%s  new_task->priority = __parsec_tp->super.super.priority;\n", indent(nesting));
    }

    coutput("%s  new_task->repo_entry = NULL;\n",
            indent(nesting));
    {
        struct jdf_dataflow *dataflow = f->dataflow;
        for(idx = 0; NULL != dataflow; idx++, dataflow = dataflow->next ) {
            coutput("%s  new_task->data._f_%s.source_repo_entry = NULL;\n"
                    "%s  new_task->data._f_%s.source_repo       = NULL;\n"
                    "%s  new_task->data._f_%s.data_in         = NULL;\n"
                    "%s  new_task->data._f_%s.data_out        = NULL;\n"
                    "%s  new_task->data._f_%s.fulfill         = 0;\n",
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname,
                    indent(nesting), dataflow->varname);
        }
    }

    coutput("#if defined(PARSEC_DEBUG_NOISIER)\n"
            "%s  {\n"
            "%s    char tmp[128];\n"
            "%s    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, \"Add startup task %%s to vpid %%d\",\n"
            "%s           parsec_task_snprintf(tmp, 128, (parsec_task_t*)new_task), vpid);\n"
            "%s  }\n"
            "#endif\n", indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting));

    coutput("%s  parsec_dependencies_mark_task_as_startup((parsec_task_t*)new_task, es);\n"
            "%s  pready_ring[vpid] = parsec_list_item_ring_push_sorted(pready_ring[vpid],\n"
            "%s                                                        (parsec_list_item_t*)new_task,\n"
            "%s                                                        parsec_execution_context_priority_comparator);\n"
            "%s  nb_tasks++;\n", indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting));    
    coutput("%s restore_context_%d:  /* we jump here just so that we have code after the label */\n", indent(nesting), ctx_level);
    coutput("%s  restore_context = 0;\n"
            "%s  (void)restore_context;\n"
            "%s  if( nb_tasks > this_task->locals.reserved[0].value ) {\n"
            "%s    if( (size_t)this_task->locals.reserved[0].value < parsec_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;\n"
            "%s    for(int _i = 0; _i < context->nb_vp; _i++ ) {\n"
            "%s      if( NULL == pready_ring[_i] ) continue;\n"
            "%s      __parsec_schedule(context->virtual_processes[_i]->execution_streams[0], (parsec_task_t*)pready_ring[_i], 0);\n"
            "%s      pready_ring[_i] = NULL;\n"
            "%s    }\n"
            "%s    total_nb_tasks += nb_tasks;\n"
            "%s    nb_tasks = 0;\n"
            "%s    if( total_nb_tasks > parsec_task_startup_chunk ) {  /* stop here and request to be rescheduled */\n"
            "%s      return PARSEC_HOOK_RETURN_AGAIN;\n"
            "%s    }\n"
            "%s  }\n",
            indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting),
            indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting),
            indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting));

    /* We close all variables, in reverse order to manage the local indices */
    while( NULL != inner_vl ) {
        if( inner_vl->expr->op == JDF_RANGE ) {
            coutput("%s} /* Loop on normal range %s */\n", indent(nesting--), inner_vl->name);
        } else if (NULL != inner_vl->expr->local_variables) {
            for(ld = inner_vl->expr->local_variables; NULL != ld; ld = ld->next) {
                coutput("%s  %s = this_task->locals.ldef[%d].value; /* restore variable in its nesting level, if needed */\n", indent(nesting), ld->alias, ld->ldef_index);
                if( JDF_RANGE == ld->op )
                    coutput("%s  } /* Arbitrary iterator on %s */ \n", indent(nesting--), ld->alias);
                coutput("%s  } /* Block for local definition of %s */ \n", indent(nesting--), ld->alias);
            }
        }
        /* Find previous local */
        for(vl = f->locals; NULL != vl; vl = vl->next)
            if( vl->next == inner_vl )
                break;
        inner_vl = vl;
    }

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa_properties);

    coutput("  (void)vpid;\n"
            "  if( 0 != nb_tasks ) {\n"
            "    for(int _i = 0; _i < context->nb_vp; _i++ ) {\n"
            "      if( NULL == pready_ring[_i] ) continue;\n"
            "      __parsec_schedule(context->virtual_processes[_i]->execution_streams[0],\n"
            "                        (parsec_task_t*)pready_ring[_i], 0);\n"
            "      pready_ring[_i] = NULL;\n"
            "    }\n"
            "    nb_tasks = 0;\n"
            "  }\n"
            "  return PARSEC_HOOK_RETURN_DONE;\n"
            "}\n\n");
}

/* structure to handle the correspondence between local variables and function parameters */
typedef struct jdf_l2p_s {
    const jdf_variable_list_t    *vl;
    const jdf_param_list_t       *pl;
    struct jdf_l2p_s             *next;
} jdf_l2p_t;

/**
 * Return a list of all the local variables that are also parameters of the
 * function, in the order in which they appear on the stack (sequential order
 * in the source code).
 */
jdf_l2p_t* build_l2p( const jdf_function_entry_t *f )
{
    jdf_l2p_t *l2p = NULL, *item = NULL;
    jdf_variable_list_t   *vl;
    jdf_param_list_t  *pl;

    for(vl = f->locals; vl != NULL; vl = vl->next) {
        for(pl = f->parameters; pl != NULL; pl = pl->next ) {
            if(0 == strcmp(pl->name, vl->name)) {
                if( NULL == l2p ) {
                    /* we need to maintain the list in the same order as the arguments */
                    item = (jdf_l2p_t*)malloc(sizeof(jdf_l2p_t));
                    l2p = item;
                } else {
                    item->next = (jdf_l2p_t*)malloc(sizeof(jdf_l2p_t));
                    item = item->next;
                }
                item->vl = vl;
                item->pl = pl;
                item->next = NULL;
                break;  /* move to the next local */
            }
        }
    }
    return l2p;
}

void free_l2p( jdf_l2p_t* l2p )
{
    jdf_l2p_t* item;
    while( NULL != l2p ) {
        item = l2p->next;
        free(l2p);
        l2p = item;
    }
}

static jdf_param_list_t *local_is_parameter(const jdf_function_entry_t *f, const jdf_variable_list_t *vl)
{
    jdf_param_list_t *pl;
    for( pl = f->parameters; pl != NULL; pl = pl->next )
        if( strcmp(pl->name, vl->name) == 0 )
            return pl;
    return NULL;
}

static  void jdf_generate_deps_key_functions(const jdf_t *jdf, const jdf_function_entry_t *f, const char *sname)
{
    jdf_variable_list_t *vl;

    (void)jdf;
    
    if( f->parameters == NULL || (0 != (f->user_defines & JDF_FUNCTION_HAS_UD_MAKE_KEY)) ) {
        /* There are no parameters for this task class, or we don't know how to inverse the key.
         * If the user knows how to print properly, she can define the hash struct entirely;
         * as this is only for debug / human-readable error messages, we bailout and just print the key
         * instead of forcing to compute the range only for that. */
        coutput("static char *%s_key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data)\n"
                "{\n"
                "  (void)user_data;\n"
                "  snprintf(buffer, buffer_size, \"(%%llu)\", (unsigned long long int)k);\n"
                "  return buffer;\n"
                "}\n"
                "\n",
                sname);
    } else {
        string_arena_t *sa1 = string_arena_new(64);
        string_arena_t *sa_format = string_arena_new(64);
        string_arena_t *sa_params = string_arena_new(64);
        string_arena_t *sa_info = string_arena_new(64);
        expr_info_t info;
        int first_param = 1;
        int need_assignment = 0;
        
        info.sa = sa_info;
        info.prefix = "";
        info.suffix = "";
        info.assignments = "&"JDF2C_NAMESPACE"_assignments";

        for(vl = f->locals; vl != NULL; vl = vl->next) {
            if( expr_requires_assignment(vl->expr, 1) ) {
                need_assignment = 1;
                break;
            }
        }
        
        coutput("static char *%s_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)\n"
                "{\n"
                "  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;\n"
                "  __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)user_data;\n",
                sname,
                jdf_basename, jdf_basename);
        if(need_assignment)
            coutput("  %s "JDF2C_NAMESPACE"_assignments = {%s};\n",
                parsec_get_name(jdf, f, "parsec_assignment_t"),
                UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name, dump_string, NULL,
                                     "  ", ".", ".value = 0, ", ".value = 0 "));
        string_arena_free(sa1);
        
        for(vl = f->locals; vl != NULL; vl = vl->next) {
            if( local_is_parameter(f, vl) != NULL ) {
                int have_min = 0;
                if( vl->expr->op == JDF_RANGE ) {
                    coutput("  int %s%s_min = %s;\n", JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr->jdf_ta1, &info));
                    have_min = 1;
                } else {
                    if( vl->expr->local_variables != NULL ) {
                        char *vname;
                        if( asprintf(&vname, "%s%s_min", JDF2C_NAMESPACE, vl->name) <= 0 ) {
                            fprintf(stderr, "Cannot allocate internal memory for the PTG compiler\n");
                            exit(-1);
                        }
                        coutput("  int %s;\n", vname);
                        jdf_generate_range_min_without_fn(jdf, vl->expr, vname, "(&"JDF2C_NAMESPACE"_assignments)");
                        free(vname);
                        have_min = 1;
                    }
                }
                if(have_min) {
                    coutput("  int %s = (__parsec_key) %% __parsec_tp->%s_%s_range + %s%s_min;\n",
                            vl->name, f->fname, vl->name, JDF2C_NAMESPACE, vl->name);
                } else {
                    coutput("  int %s = (__parsec_key) %% __parsec_tp->%s_%s_range;\n",
                            vl->name, f->fname, vl->name);
                }
                string_arena_add_string(sa_format, "%s%%d", first_param?"":", ");
                string_arena_add_string(sa_params, "%s%s", first_param?"":", ", vl->name);
                first_param = 0;
                coutput("  __parsec_key = __parsec_key / __parsec_tp->%s_%s_range;\n",
                        f->fname, vl->name);
            } else {
                /* IDs should depend only on the parameters of the
                 * function. However, we might need the other definitions because
                 * the min expression of the parameters might depend on them. If
                 * this is not the case, a quick "(void)" removes the warning.
                 */
                coutput("  int %s = %s;\n"
                        "  (void)%s;\n",
                        vl->name, dump_expr((void**)vl->expr, &info), vl->name);
            }
            if(need_assignment)
                coutput("  "JDF2C_NAMESPACE"_assignments.%s.value = %s;\n", vl->name, vl->name);
        }
        if(need_assignment)
            coutput("  (void)"JDF2C_NAMESPACE"_assignments;\n");
        coutput("  snprintf(buffer, buffer_size, \"%s(%s)\", %s);\n"
                "  return buffer;\n"
                "}\n"
                "\n", f->fname, string_arena_get_string(sa_format), string_arena_get_string(sa_params));
        string_arena_free(sa_format);
        string_arena_free(sa_params);
        string_arena_free(sa_info);
    }

    coutput("static parsec_key_fn_t %s = {\n"
            "   .key_equal = parsec_hash_table_generic_64bits_key_equal,\n"
            "   .key_print = %s_key_print,\n"
            "   .key_hash  = parsec_hash_table_generic_64bits_key_hash\n"
            "};\n"
            "\n",
            sname, sname);
}

static void jdf_generate_internal_init(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{
    string_arena_t *sa1, *sa2, *sa_end;
    const jdf_variable_list_t *vl, *inner_vl = NULL;
    jdf_expr_t *ld;
    const jdf_param_list_t *pl;
    expr_info_t info = EMPTY_EXPR_INFO;
    int need_to_iterate, need_min_max, need_to_count_tasks;
    int nesting = 0, idx;
    jdf_l2p_t *l2p = build_l2p(f), *l2p_item;
    char *dep_key_fn_name = NULL;
    (void)jdf;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa_end = string_arena_new(64);

    need_min_max = (0 == (f->user_defines & JDF_FUNCTION_HAS_UD_MAKE_KEY));
    need_to_count_tasks = (0 == (f->user_defines & JDF_HAS_UD_NB_LOCAL_TASKS));
    need_to_iterate = need_min_max || need_to_count_tasks;

    if( 0 != (f->user_defines & JDF_FUNCTION_HAS_UD_HASH_STRUCT) ) {
        dep_key_fn_name = strdup( jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL) );
    } else {
        if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE) {
            if( asprintf(&dep_key_fn_name, "%s_%s_deps_key_functions", jdf_basename, fname) <= 0 ) {
                fprintf(stderr, "Cannot allocate internal memory for the PTG compiler\n");
                exit(-1);
            }
            jdf_generate_deps_key_functions(jdf, f, dep_key_fn_name);
        }
    }

    coutput("/* Needs: %s %s %s */\n"
            "static int %s(parsec_execution_stream_t * es, %s * this_task)\n"
            "{\n"
            "  __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t*)this_task->taskpool;\n",
            need_min_max ? "min-max" : "-",
            need_to_count_tasks ? "count-tasks" : "-",
            need_to_iterate ? "iterate" : "-",
            fname, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename);

    if(need_to_count_tasks) {
        coutput("  int32_t nb_tasks = 0, saved_nb_tasks = 0;\n");
        /* prepare the epilog output to prevent compiler from complaining about initialized but unused data */
        string_arena_add_string(sa_end, "(void)saved_nb_tasks;\n");
    } else {
        coutput("  int32_t nb_tasks = 0;\n");
    }
    if( need_min_max ) {
        for(l2p_item = l2p; NULL != l2p_item; l2p_item = l2p_item->next) {
            vl = l2p_item->vl; assert(NULL != vl);
            if( NULL == (pl = l2p_item->pl) )
                continue;
            if(vl->expr->op != JDF_RANGE && vl->expr->local_variables == NULL)
                continue;
            coutput("int32_t __%s_min = 0x7fffffff, __%s_max = 0;", vl->name, vl->name);
            coutput("int32_t %s%s_min = 0x7fffffff, %s%s_max = 0;",
                    JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name);
            /* prepare the epilog output to prevent compiler from complaining about initialized but unused data */
            string_arena_add_string(sa_end, "(void)__%s_min; (void)__%s_max;",
                                    vl->name, vl->name);
        }
    }

    if( need_to_iterate ) {
        coutput("  %s assignments = {%s};\n",
            parsec_get_name(jdf, f, "parsec_assignment_t"),
            UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name, dump_string, NULL,
                                     "  ", ".", ".value = 0, ", ".value = 0 "));
        if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
            coutput("  parsec_dependencies_t *dep = NULL;\n");
        }
        coutput("%s",
                UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name, dump_string, NULL,
                                     "  int32_t ", " ", ", ", ";\n"));
        for(l2p_item = l2p; NULL != l2p_item; l2p_item = l2p_item->next) {
            vl = l2p_item->vl; assert(NULL != vl);

            if(vl->expr->op == JDF_RANGE || NULL != vl->expr->local_variables) {
                coutput("  int32_t %s%s_start, %s%s_end, %s%s_inc;\n", JDF2C_NAMESPACE, vl->name,
                        JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name );
                /* prepare the epilog output to prevent compiler from complaining about initialized but unused data */
                string_arena_add_string(sa_end, "  (void)%s%s_start; (void)%s%s_end; (void)%s%s_inc;",
                                        JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE,
                                        vl->name, JDF2C_NAMESPACE, vl->name);
            }
        }
    }

    string_arena_init(sa1);
    string_arena_init(sa2);

    if( profile_enabled(f->properties) ) {
        coutput("#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)\n"
                "  PARSEC_PROFILING_TRACE(es->es_profile,\n"
                "                         this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id],\n"
                "                         0,\n"
                "                         this_task->taskpool->taskpool_id, NULL);\n"
                "#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */\n");
    }

    info.sa = sa1;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&assignments";

    if( need_to_iterate || need_min_max ) {
        for(vl = f->locals; vl != NULL; vl = vl->next) {
            if(vl->expr->op == JDF_RANGE) {
                coutput("%s    %s%s_start = %s;\n",
                        indent(nesting), JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr->jdf_ta1, &info));
                coutput("%s    %s%s_end = %s;\n",
                        indent(nesting), JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr->jdf_ta2, &info));
                coutput("%s    %s%s_inc = %s;\n",
                        indent(nesting), JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr->jdf_ta3, &info));

                if( need_min_max ) {
                    coutput("%s    __%s_min = parsec_imin(%s%s_start, %s%s_end);\n",
                            indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name);
                    coutput("%s    __%s_max = parsec_imax(%s%s_start, %s%s_end);\n",
                            indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name);
                    coutput("%s    %s%s_min = parsec_imin(%s%s_min, __%s_min);\n",
                            indent(nesting), JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name, vl->name);
                    coutput("%s    %s%s_max = parsec_imax(%s%s_max, __%s_max);\n",
                            indent(nesting), JDF2C_NAMESPACE, vl->name, JDF2C_NAMESPACE, vl->name, vl->name);
                }

                /* Adapt the loop condition depending on the value of the increment. We can
                 * now handle both increasing and decreasing execution spaces. */
                coutput("%s    for(%s =  %s%s_start;\n",
                    indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name);
                if( JDF_OP_IS_CST(vl->expr->jdf_ta3->op) ) {
                    if( vl->expr->jdf_ta3->jdf_cst >= 0 ) {
                        coutput("%s        %s <= %s%s_end;\n",
                            indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name);
                    } else {
                        coutput("%s        %s >= %s%s_end;\n",
                            indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name);
                    }
                } else {
                    coutput("%s        (((%s%s_inc >= 0) && (%s <= %s%s_end)) ||\n"
                            "%s         ((%s%s_inc <  0) && (%s >= %s%s_end)));\n",
                            indent(nesting), JDF2C_NAMESPACE, vl->name, vl->name, JDF2C_NAMESPACE, vl->name,
                            indent(nesting), JDF2C_NAMESPACE, vl->name, vl->name, JDF2C_NAMESPACE, vl->name);
                }
                coutput("%s        %s += %s%s_inc) {\n",
                    indent(nesting), vl->name, JDF2C_NAMESPACE, vl->name);
            } else if ( NULL != vl->expr->local_variables) {
                for(ld = jdf_expr_lv_first(vl->expr->local_variables); NULL != ld; ld = jdf_expr_lv_next(vl->expr->local_variables, ld)) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    if( ld->op == JDF_RANGE ) {
                        coutput("%s  { /* block for the local variable '%s' */\n"
                                "%s    int %s_loop_%s = 1;\n"
                                "%s    int %s;\n"
                                "%s    for(assignments.ldef[%d].value = %s = %s; %s_loop_%s == 1;",
                                indent(nesting), ld->alias,
                                indent(nesting), JDF2C_NAMESPACE, ld->alias,
                                indent(nesting), ld->alias,
                                indent(nesting), ld->ldef_index, ld->alias, dump_expr((void**)ld->jdf_ta1, &info), JDF2C_NAMESPACE, ld->alias);
                        coutput(" assignments.ldef[%d].value = %s += %s ) { /* Arbitrary iterator on %s */ \n",
                                ld->ldef_index, ld->alias, dump_expr((void**)ld->jdf_ta3, &info), ld->alias);
                        coutput("%s      if( %s == %s ) %s_loop_%s = 0; /* Execute once only when reaching the end */\n",
                                indent(nesting), ld->alias, dump_expr((void**)ld->jdf_ta2, &info), JDF2C_NAMESPACE, ld->alias);
                        nesting+=2;
                    } else {
                        coutput("%s  { /* block for the local variable '%s' */\n"
                                "%s    int %s = assignments.ldef[%d].value = %s;\n",
                                indent(nesting), ld->alias,
                                indent(nesting), ld->alias, ld->ldef_index, dump_expr((void**)ld, &info));
                        nesting++;
                    } 
                }
                coutput("%s    %s = %s;\n", indent(nesting), vl->name, dump_expr((void**)vl->expr, &info));
            } else {
                /* We need to start a new code block to have a similar layout as the cases above.
                 * Otherwise the } few lines below will match the wrong loop.
                 */
                coutput("%s{  /* block for the non-range variable %s */\n",
                        indent(nesting), vl->name);
                coutput("%s    %s = %s;\n", indent(nesting), vl->name, dump_expr((void**)vl->expr, &info));
            }
            coutput("%s    assignments.%s.value = %s;\n",
                    indent(nesting), vl->name, vl->name);
            nesting++;
            inner_vl = vl; /* remember what is the last local seen */
        }
        
        if( need_min_max ) {
            for(vl = f->locals; vl != NULL; vl = vl->next) {
                if ( NULL != vl->expr->local_variables) {
                    coutput("%s    %s%s_min = parsec_imin(__jdf2c_%s_min, %s);\n",
                            indent(nesting), JDF2C_NAMESPACE, vl->name, vl->name, vl->name);
                    coutput("%s    %s%s_max = parsec_imax(__jdf2c_%s_max, %s);\n",
                            indent(nesting), JDF2C_NAMESPACE, vl->name, vl->name, vl->name);
                }
            }
        }
        
        string_arena_init(sa1);
        string_arena_init(sa2);
        coutput("%s  if( !%s_pred(%s) ) continue;\n",
                indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa2, f->locals, next, name,
                                                                dump_string, NULL,
                                                                "", "", ", ", ""));
        if( need_to_count_tasks ) {
            coutput("%s  nb_tasks++;\n",
                    indent(nesting));
        }

        /* We close all non-range variables */
        while( NULL != inner_vl &&
               inner_vl->expr->op != JDF_RANGE &&
               inner_vl->expr->local_variables == NULL ) {
            coutput("%s} /* block for the non-range variable %s */ \n", indent(nesting--), inner_vl->name);
            for(vl = f->locals; NULL != vl; vl = vl->next)
                if( vl->next == inner_vl )
                    break;
            inner_vl = vl;
        }
        if( NULL != inner_vl ) {
            /* and we close the most inner loop */
            if( inner_vl->expr->op == JDF_RANGE ) {
                coutput("%s} /* Loop on normal range %s */\n", indent(nesting--), inner_vl->name);
            } else {
                assert(inner_vl->expr->local_variables != NULL);
                for(ld = inner_vl->expr->local_variables; NULL != ld; ld = ld->next) {
                    if( JDF_RANGE == ld->op )
                        coutput("%s  } /* Arbitrary iterator on %s */ \n", indent(nesting--), ld->alias);
                    coutput("%s  } /* Block for local definition of %s */ \n", indent(nesting--), ld->alias);
                }
            }
            for(vl = f->locals; NULL != vl; vl = vl->next)
                if( vl->next == inner_vl )
                    break;
            inner_vl = vl;
            if(NULL != inner_vl) {
                /* As we can re-use a local definition alias for different deps or calls,
                 * and all that gets nested but stored in the same cell of the assignment,
                 * we need to restore last value left by the level above */
                for(ld = inner_vl->expr->local_variables; NULL != ld; ld = ld->next) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    coutput("%s  assignments.ldef[%d].value = %s;\n", indent(nesting), ld->ldef_index, ld->alias);
                }
            }
        }

        if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
            /* If no tasks have been generated during the last loop, there is no need
             * to have any dependencies.
             */
            if( need_to_count_tasks ) {
                coutput("%s  if( saved_nb_tasks != nb_tasks ) {\n", indent(nesting));
                coutput("%s    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, \"Allocating dependencies array for %s (partial nb_tasks = %%d)\", nb_tasks);\n",
                        indent(nesting), f->fname);
            }

            string_arena_add_string(sa1, "dep");
            for(l2p_item = l2p; NULL != l2p_item; l2p_item = l2p_item->next) {
                vl = l2p_item->vl;
                coutput("%s    if( %s == NULL ) {\n",
                        indent(nesting), string_arena_get_string(sa1));
                if(vl->expr->op == JDF_RANGE) {
                    coutput("%s      ALLOCATE_DEP_TRACKING(%s, __%s_min, __%s_max,\n",
                            indent(nesting), string_arena_get_string(sa1), vl->name, vl->name);
                } else {
                    coutput("%s      ALLOCATE_DEP_TRACKING(%s, %s, %s,\n",
                            indent(nesting), string_arena_get_string(sa1), vl->name, vl->name);
                }
                coutput("%s                            \"%s\", %s);\n"
                        "%s    }\n",
                        indent(nesting), vl->name,
                        NULL == l2p_item->next ? "PARSEC_DEPENDENCIES_FLAG_FINAL" : "PARSEC_DEPENDENCIES_FLAG_NEXT",  /* last item */
                        indent(nesting));
                string_arena_init(sa2);
                string_arena_add_string(sa2, "%s", string_arena_get_string(sa1));
                string_arena_add_string(sa1, "->u.next[%s-__%s_min]", vl->name, vl->name);
            }
            /* Save the current number of tasks for the optimization of the next iteration */
            if( need_to_count_tasks ) {
                coutput("%s    saved_nb_tasks = nb_tasks;\n"
                        "%s }\n",
                        indent(nesting),
                        indent(nesting));
            }
        }

        while( NULL != inner_vl ) {
            /* We close all the other loops in reverse order */
            if( inner_vl->expr->local_variables == NULL ) {
                /* If it is a range or a simple expression without local indices, it's easy */
                if( vl->expr->op == JDF_RANGE ) {
                    coutput("%s} /* For loop of %s */ \n", indent(nesting--), inner_vl->name);
                } else {
                    coutput("%s} /* Non-range variable %s */ \n", indent(nesting--), inner_vl->name);
                }
            } else {
                for(ld = inner_vl->expr->local_variables; NULL != ld; ld = ld->next) {
                    if( JDF_RANGE == ld->op )
                        coutput("%s  } /* Arbitrary iterator on %s */ \n", indent(nesting--), ld->alias);
                    coutput("%s  } /* Block for local definition of %s */ \n", indent(nesting--), ld->alias);
                }
            }
            for(vl = f->locals; NULL != vl; vl = vl->next)
                if( vl->next == inner_vl )
                    break;
            inner_vl = vl;
            if(NULL != inner_vl) {
                /* As we can re-use a local definition alias for different deps or calls,
                 * and all that gets nested but stored in the same cell of the assignment,
                 * we need to restore last value left by the level above */
                for(ld = inner_vl->expr->local_variables; NULL != ld; ld = ld->next) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    coutput("%s  assignments.ldef[%d].value = %s;\n", indent(nesting), ld->ldef_index, ld->alias);
                }
            }
        }
        
        if(need_to_count_tasks) {
            coutput("%s   if( 0 != nb_tasks ) {\n"
                    "%s     (void)parsec_atomic_fetch_add_int32(&__parsec_tp->super.super.nb_tasks, nb_tasks);\n"
                    "%s   }\n"
                    "%s   saved_nb_tasks = nb_tasks;\n",
                    indent(nesting),
                    indent(nesting),
                    indent(nesting),
                    indent(nesting));
        }
        if( need_min_max ) {
            coutput("  /* Set the range variables for the collision-free hash-computation */\n");
            for(l2p_item = l2p; NULL != l2p_item; l2p_item = l2p_item->next) {
                vl = l2p_item->vl;
                if( NULL == (pl = l2p_item->pl) ) continue;
                if(vl->expr->op == JDF_RANGE || NULL != vl->expr->local_variables) {
                    coutput("  __parsec_tp->%s_%s_range = (%s%s_max - %s%s_min) + 1;\n",
                            f->fname, pl->name, JDF2C_NAMESPACE, pl->name, JDF2C_NAMESPACE, pl->name);
                } else {
                    coutput("  __parsec_tp->%s_%s_range = 1;  /* single value, not a range */\n", f->fname, pl->name);
                }
            }
        }
    }
    /* If this startup task belongs to a task class that has the potential to generate initial tasks,
     * we should be careful to delay the generation of these tasks until all initializations tasks
     * have been completed (or we might face a race condition between creating the
     * dependencies arrays and accessing them). We synchronize the initial tasks via a sync. Until
     * the sync trigger all task-generation tasks will be enqueued on the taskpool's startup_queue.
     */
    if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
        coutput("%s    do {\n"
                "%s      this_task->super.list_next = (parsec_list_item_t*)__parsec_tp->startup_queue;\n"
                "%s    } while(!parsec_atomic_cas_ptr(&__parsec_tp->startup_queue, (parsec_list_item_t*)this_task->super.list_next, this_task));\n"
                "%s    this_task->status = PARSEC_TASK_STATUS_HOOK;\n",
                indent(nesting), indent(nesting), indent(nesting), indent(nesting));
    } else {
        /* Assume that all startup tasks complete right away, without going through the
         * second stage.
         */
        coutput("  this_task->status = PARSEC_TASK_STATUS_COMPLETE;\n");
    }

    string_arena_free(sa1);
    string_arena_free(sa2);
    coutput("\n  PARSEC_AYU_REGISTER_TASK(&%s_%s);\n", jdf_basename, f->fname);
    idx = 0;
    JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_t, next, idx);
    if( f->user_defines & JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS ) {
        coutput("  PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, \"Allocating dependencies array for %s (user-defined allocator)\\n\");\n",
                fname);
        coutput("  __parsec_tp->super.super.dependencies_array[%d] = %s(__parsec_tp);\n",
                f->task_class_id, jdf_property_get_function(f->properties, JDF_PROP_UD_ALLOC_DEPS_FN_NAME, NULL));
    } else {
        if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
            coutput("  __parsec_tp->super.super.dependencies_array[%d] = dep;\n",
                    f->task_class_id);
        } else if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ||
                   0 != (f->user_defines & JDF_FUNCTION_HAS_UD_HASH_STRUCT)) {
            coutput("  __parsec_tp->super.super.dependencies_array[%d] = PARSEC_OBJ_NEW(parsec_hash_table_t);\n"
                    "  parsec_hash_table_init(__parsec_tp->super.super.dependencies_array[%d], offsetof(parsec_hashable_dependency_t, ht_item), 10, %s, this_task->taskpool);\n",
                    f->task_class_id, f->task_class_id, dep_key_fn_name);
            free(dep_key_fn_name);
            dep_key_fn_name = NULL;
        }
    }

    /* Generate the repo for all tasks classes, they can be used when:
     * - a predecessor sets up a reshape promised for a datacopy
     * - the own tasks use it when reshaping a datacopy directly read from desc
     * No longer only when if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) )
     */
    coutput("  __parsec_tp->repositories[%d] = data_repo_create_nothreadsafe(%s, %s, (parsec_taskpool_t*)__parsec_tp, %d);\n",
            f->task_class_id, need_to_count_tasks ? "nb_tasks" : "PARSEC_DEFAULT_DATAREPO_HASH_LENGTH",
            jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL),
            idx );

    coutput("%s"
            "  %s (void)__parsec_tp; (void)es;\n",
            string_arena_get_string(sa_end),
            need_to_iterate ? "(void)assignments;" : "");
    /**
     * Generate the code to deal nicely with the case where the JDF provide an undermined
     * number of tasks. Don't waste time atomically counting the tasks, let the JDF decide
     * when everything is completed.
     */
    coutput("  if(1 == parsec_atomic_fetch_dec_int32(&__parsec_tp->sync_point)) {\n"
            "    /* Last initialization task complete. Update the number of tasks. */\n");
    if(!need_to_count_tasks) {
        /* We only need to update the number of tasks according to the user provided count */
        coutput("    nb_tasks = __parsec_tp->super.super.nb_tasks = %s(__parsec_tp);\n",
                jdf_property_get_function(jdf->global_properties, JDF_PROP_UD_NB_LOCAL_TASKS_FN_NAME, NULL));
    } else {
        coutput("  nb_tasks = parsec_atomic_fetch_dec_int32(&__parsec_tp->super.super.nb_tasks);\n");
        /* TODO coutput("    __parsec_tp->super.super.nb_tasks = __parsec_tp->super.super.initial_number_tasks;\n"); */
    }
    coutput("    parsec_mfence();  /* write memory barrier to guarantee that the scheduler gets the correct number of tasks */\n"
            "    parsec_taskpool_enable((parsec_taskpool_t*)__parsec_tp, &__parsec_tp->startup_queue,\n"
            "                           (parsec_task_t*)this_task, es, (1 <= nb_tasks));\n");
    if( profile_enabled(f->properties) ) {
        coutput("#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)\n"
                "    PARSEC_PROFILING_TRACE(es->es_profile,\n"
                "                           this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id + 1],\n"
                "                           0,\n"
                "                           this_task->taskpool->taskpool_id, NULL);\n"
                "#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */\n");
    }
    if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
        coutput("    if( 1 >= nb_tasks ) {\n"
                "        /* if no tasks will be generated let's prevent the runtime from calling the hook and instead go directly to complete the task */\n"
                "        this_task->status = PARSEC_TASK_STATUS_COMPLETE;\n"
                "    }\n");
    }
    coutput("    return PARSEC_HOOK_RETURN_DONE;\n"
            "  }\n"
            "  return %s;\n"
            "}\n\n",
            (f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP) ? "PARSEC_HOOK_RETURN_ASYNC" : "PARSEC_HOOK_RETURN_DONE");

    string_arena_free(sa_end);
    free_l2p(l2p);
}

static void jdf_generate_simulation_cost_fct(const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix)
{
    assignment_info_t ai;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);

    (void)jdf;

    ai.sa = sa;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;

    coutput("#if defined(PARSEC_SIM)\n"
            "static int %s(const %s *this_task)\n"
            "{\n"
            "  const parsec_taskpool_t *__parsec_tp = (const parsec_taskpool_t*)this_task->taskpool;\n"
            "%s"
            "  (void)__parsec_tp;\n",
            prefix, parsec_get_name(jdf, f, "task_t"),
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
    string_arena_add_string(sa, "static const __parsec_chore_t __%s_chores[] ={\n", base_name);
    do {
        jdf_find_property(body->properties, "type", &type_property);
        jdf_find_property(body->properties, "dyld", &dyld_property);
        if( NULL == type_property) {
            string_arena_add_string(sa, "    { .type     = PARSEC_DEV_CPU,\n");
            string_arena_add_string(sa, "      .evaluate = %s,\n", "NULL");
            string_arena_add_string(sa, "      .hook     = (parsec_hook_t*)hook_of_%s },\n", base_name);
        } else {
            string_arena_add_string(sa, "#if defined(PARSEC_HAVE_%s)\n", type_property->expr->jdf_var);
            string_arena_add_string(sa, "    { .type     = PARSEC_DEV_%s,\n", type_property->expr->jdf_var);
            if( NULL == dyld_property ) {
                string_arena_add_string(sa, "      .dyld     = NULL,\n");
            } else {
                jdf_def_list_t* dyld_proptotype_property;
                jdf_find_property(body->properties, "dyldtype", &dyld_proptotype_property);
                if ( NULL == dyld_proptotype_property ) {
                    fprintf(stderr,
                            "Internal Error: function prototype (dyldtype) of dyld function (%s) is not defined in %s body of task %s at line %d\n",
                            dyld_property->expr->jdf_var, type_property->expr->jdf_var, f->fname, JDF_OBJECT_LINENO( body ) );
                    assert( NULL != dyld_proptotype_property );
                }
                string_arena_add_string(sa, "      .dyld     = \"%s\",\n", dyld_property->expr->jdf_var);
            }
            string_arena_add_string(sa, "      .evaluate = %s,\n", "NULL");
            string_arena_add_string(sa, "      .hook     = (parsec_hook_t*)hook_of_%s_%s },\n", base_name, type_property->expr->jdf_var);
            string_arena_add_string(sa, "#endif  /* defined(PARSEC_HAVE_%s) */\n", type_property->expr->jdf_var);
        }
        body = body->next;
    } while (NULL != body);
    string_arena_add_string(sa,
                            "    { .type     = PARSEC_DEV_NONE,\n"
                            "      .evaluate = NULL,\n"
                            "      .hook     = (parsec_hook_t*)NULL },  /* End marker */\n"
                            "};\n\n");
}

static void jdf_generate_release_task_fct(const jdf_t *jdf, jdf_function_entry_t *f, const char *prefix)
{
    (void)jdf;
    coutput("static parsec_hook_return_t %s(parsec_execution_stream_t *es, parsec_task_t *this_task)\n"
            "{\n"
            "    const __parsec_%s_internal_taskpool_t *__parsec_tp =\n"
            "        (const __parsec_%s_internal_taskpool_t *)this_task->taskpool;\n",
            prefix,
            jdf_basename,
            jdf_basename);
    if( !(f->user_defines & JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS) ) {
        coutput("    parsec_hash_table_t *ht = (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[%d];\n"
                "    parsec_key_t key = this_task->task_class->make_key((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals);\n"
                "    parsec_hashable_dependency_t *hash_dep = (parsec_hashable_dependency_t *)parsec_hash_table_remove(ht, key);\n"
                "    parsec_thread_mempool_free(hash_dep->mempool_owner, hash_dep);\n",
                f->task_class_id);
    }
    if( f->user_defines & JDF_HAS_UD_NB_LOCAL_TASKS ) {
        coutput("    if( (PARSEC_UNDETERMINED_NB_TASKS == __parsec_tp->super.super.nb_tasks) ||\n"
                "        (0 == __parsec_tp->super.super.nb_tasks) ||\n"
                "        (PARSEC_RUNTIME_RESERVED_NB_TASKS == __parsec_tp->super.super.nb_tasks) ) {\n"
                "        /* don't spend time counting */\n"
                "        return parsec_release_task_to_mempool(es, this_task);\n"
                "    }\n");
    }
    coutput("    return parsec_release_task_to_mempool_update_nbtasks(es, this_task);\n"
            "}\n"
            "\n");
}

static int jdf_function_property_has_duplicate_name(const jdf_def_list_t *cp)
{
    if (NULL == cp->next) return 0;
    const jdf_def_list_t *cp2;
    for (cp2 = cp->next; cp2 != NULL; cp2 = cp2->next)
        if (!strcmp(cp->name, cp2->name))
            return 1;
    return 0;
}

static void jdf_generate_function_properties( const jdf_t *jdf, jdf_function_entry_t *f, string_arena_t *sa )
{
    string_arena_t *sa2;
    char prefix[512];
    int i;
    const jdf_def_list_t* cp;

    sa2 = string_arena_new(64);
    i=1;
    for(cp = f->properties; cp != NULL; cp = cp->next) {
        if(jdf_function_property_is_keyword(cp->name))
            continue;
        i++;
    }
    string_arena_add_string(sa2, "static const parsec_property_t properties_of_%s_%s[%d] = {\n",
                            jdf_basename, f->fname, i);
    for(cp = f->properties; cp != NULL; cp = cp->next) {
        if(jdf_function_property_is_keyword(cp->name))
            continue;
        if(jdf_function_property_has_duplicate_name(cp)) {
            fprintf(stdout, "Internal Warning: Property %s defined at line %d is overloaded by the next one.\n", cp->name, JDF_OBJECT_LINENO(cp));
            continue;
        }

        sprintf(prefix, "property_%d_%s_of_%s_%s_as_expr", i, cp->name, jdf_basename, f->fname);
        jdf_generate_expression(jdf, f, cp->expr, prefix);
        string_arena_add_string(sa2, "  {.name = \"%s\", .expr = &%s},\n", cp->name, prefix);
        i++;
    }
    string_arena_add_string(sa2,
                            "  {.name = NULL, .expr = NULL}\n"
                            "};\n");
    coutput("%s", string_arena_get_string(sa2));
    string_arena_add_string(sa, "  .properties = properties_of_%s_%s,\n", jdf_basename, f->fname);
    string_arena_free(sa2);
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

    JDF_COUNT_LIST_ENTRIES(f->parameters, jdf_param_list_t, next, nbparameters);
    JDF_COUNT_LIST_ENTRIES(f->locals, jdf_variable_list_t, next, nbdefinitions);

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
                    if( JDF_GUARD_BINARY == dl->guard->guard_type ) {
                        fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                    }
                } else {
                    switch( dl->guard->guard_type ) {
                    case JDF_GUARD_TERNARY:
                        if( NULL == dl->guard->callfalse->var ) {
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                        }
                        /* fallthrough */
                    case JDF_GUARD_UNCONDITIONAL:
                        if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) ) {
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                            break;
                        }
                        /* fallthrough */
                    case JDF_GUARD_BINARY:
                        if( NULL == dl->guard->calltrue->var ) {
                            fl->flow_flags |= JDF_FLOW_HAS_IN_DEPS;
                        }
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
                            "static const parsec_task_class_t %s = {\n"
                            "  .name = \"%s\",\n"
                            "  .task_class_id = %d,\n"
                            "  .nb_flows = %d,\n"
                            "  .nb_parameters = %d,\n"
                            "  .nb_locals = %d,\n"
                            "  .task_class_type = PARSEC_TASK_CLASS_TYPE_PTG,\n",
                            JDF_OBJECT_ONAME(f),
                            f->fname,
                            f->task_class_id,
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
    string_arena_add_string(sa, "  .data_affinity = (parsec_data_ref_fn_t*)%s,\n", prefix);

    /**
     * As of 05/2020 I disabled the generation of the initial_data and final_data
     * because 1) they are incorrect with regard to the local indices, and 2) are
     * not necessary right now for the engine (we use data_lookup to get the
     * data in all cases).
     */
#if 0
    sprintf(prefix, "initial_data_of_%s_%s", jdf_basename, f->fname);
    ret = jdf_generate_initfinal_data(jdf, JDF_DEP_FLOW_IN, f, prefix);
#else
    ret = -1;  /* to get NULL for .initial_data */
#endif
    string_arena_add_string(sa, "  .initial_data = (parsec_data_ref_fn_t*)%s,\n", (0 != ret ? prefix : "NULL"));

#if 0
    sprintf(prefix, "final_data_of_%s_%s", jdf_basename, f->fname);
    ret = jdf_generate_initfinal_data(jdf, JDF_DEP_FLOW_OUT, f, prefix);
#else
    ret = -1;  /* to get NULL for .final_data */
#endif
    string_arena_add_string(sa, "  .final_data = (parsec_data_ref_fn_t*)%s,\n", (0 != ret ? prefix : "NULL"));

    if( NULL != f->priority ) {
        sprintf(prefix, "priority_of_%s_%s_as_expr", jdf_basename, f->fname);
        jdf_generate_expression(jdf, f, f->priority, prefix);
        string_arena_add_string(sa, "  .priority = &%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .priority = NULL,\n");
    }

    jdf_generate_function_properties( jdf, f, sa );

#if defined(PARSEC_SCHED_DEPS_MASK)
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
    {
        int in_flows = 0, out_flows = 0;
        for(i = 0, fl = f->dataflow; fl != NULL; fl = fl->next, i++) {
            use_mask &= jdf_generate_dataflow(jdf, f, fl, prefix, &has_control_gather);
            in_flows += !!(fl->flow_flags & JDF_FLOW_TYPE_READ);
            out_flows += !!(fl->flow_flags & JDF_FLOW_TYPE_WRITE);
        }
        string_arena_add_string(sa,
                                "#if MAX_PARAM_COUNT < %d  /* number of read flows of %s */\n"
                                "  #error Too many read flows for task %s\n"
                                "#endif  /* MAX_PARAM_COUNT */\n"
                                "#if MAX_PARAM_COUNT < %d  /* number of write flows of %s */\n"
                                "  #error Too many write flows for task %s\n"
                                "#endif  /* MAX_PARAM_COUNT */\n",
                                in_flows, f->fname, f->fname,
                                out_flows, f->fname, f->fname);
    }
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
                                "  .flags = %s%s%s | PARSEC_USE_DEPS_MASK,\n"
                                "  .dependencies_goal = 0x%x,\n",
                                (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "PARSEC_HIGH_PRIORITY_TASK" : "0x0",
                                has_in_in_dep ? " | PARSEC_HAS_IN_IN_DEPENDENCIES" : "",
                                jdf_property_get_int(f->properties, "immediate", 0) ? " | PARSEC_IMMEDIATE_TASK" : "",
                                inputmask);
    } else {
        string_arena_add_string(sa,
                                "  .flags = %s%s%s%s,\n"
                                "  .dependencies_goal = %d,\n",
                                (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "PARSEC_HIGH_PRIORITY_TASK" : "0x0",
                                has_in_in_dep ? " | PARSEC_HAS_IN_IN_DEPENDENCIES" : "",
                                jdf_property_get_int(f->properties, "immediate", 0) ? " | PARSEC_IMMEDIATE_TASK" : "",
                                has_control_gather ? "|PARSEC_HAS_CTL_GATHER" : "",
                                nb_input);
    }

    string_arena_add_string(sa,
                            "  .make_key = %s,\n"
                            "  .key_functions = &%s,\n",
                            jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL),
                            jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL));
    string_arena_add_string(sa, "  .fini = (parsec_hook_t*)%s,\n", "NULL");

    sprintf(prefix, "%s_%s", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .incarnations = __%s_chores,\n", prefix);

    if( !(f->user_defines & JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS) ) {
        if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
            sprintf(prefix, "find_deps_%s_%s", jdf_basename, f->fname);
            jdf_generate_code_find_deps(jdf, f, prefix);
            (void)jdf_add_function_property(&f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, prefix);
        } else if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ) {
            (void)jdf_add_function_property(&f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, "parsec_hash_find_deps");
        }
    }
    string_arena_add_string(sa, "  .find_deps = %s,\n", jdf_property_get_function(f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, NULL));

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) ) {
        sprintf(prefix, "iterate_successors_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_code_iterate_successors_or_predecessors(jdf, f, prefix, JDF_DEP_FLOW_OUT);
        string_arena_add_string(sa, "  .iterate_successors = (parsec_traverse_function_t*)%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .iterate_successors = (parsec_traverse_function_t*)NULL,\n");
    }

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_PREDECESSORS) ) {
        sprintf(prefix, "iterate_predecessors_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_code_iterate_successors_or_predecessors(jdf, f, prefix, JDF_DEP_FLOW_IN);
        string_arena_add_string(sa, "  .iterate_predecessors = (parsec_traverse_function_t*)%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .iterate_predecessors = (parsec_traverse_function_t*)NULL,\n");
    }

    sprintf(prefix, "release_deps_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_release_deps(jdf, f, prefix);
    string_arena_add_string(sa, "  .release_deps = (parsec_release_deps_t*)%s,\n", prefix);

    sprintf(prefix, "data_lookup_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_data_lookup(jdf, f, prefix);
    string_arena_add_string(sa, "  .prepare_input = (parsec_hook_t*)%s,\n", prefix);
    string_arena_add_string(sa, "  .prepare_output = (parsec_hook_t*)%s,\n", "NULL");
    sprintf(prefix, "datatype_lookup_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_datatype_lookup(jdf, f, prefix);
    string_arena_add_string(sa, "  .get_datatype = (parsec_datatype_lookup_t*)%s,\n", prefix);

    sprintf(prefix, "hook_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_hooks(jdf, f, prefix);
    string_arena_add_string(sa, "  .complete_execution = (parsec_hook_t*)complete_%s,\n", prefix);

    /**
     * By default assume that the even if the JDF writer provides a specialized function to count
     * the tasks, the value returned from this function is not PARSEC_UNDETERMINED_NB_TASKS
     * (which means the runtime will have to count the completed tasks).
     */
    if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
        string_arena_add_string(sa, "  .release_task = (parsec_hook_t*)parsec_release_task_to_mempool_update_nbtasks,\n");
    } else if ( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ) {
        /* If we have a user-defined find_deps function, don't generate the hashtable_dep release task, keep
         * just counting, if needed */
        sprintf(prefix, "release_task_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_release_task_fct(jdf, f, prefix);
        string_arena_add_string(sa, "  .release_task = &%s,\n", prefix);
    }

    if( NULL != f->simcost ) {
        sprintf(prefix, "simulation_cost_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_simulation_cost_fct(jdf, f, prefix);
        string_arena_add_string(sa,
                                "#if defined(PARSEC_SIM)\n"
                                "  .sim_cost_fct =(parsec_sim_cost_fct_t*) %s,\n"
                                "#endif\n", prefix);
    } else {
        string_arena_add_string(sa,
                                "#if defined(PARSEC_SIM)\n"
                                "  .sim_cost_fct = (parsec_sim_cost_fct_t*)NULL,\n"
                                "#endif\n");
    }

    sprintf(prefix, "%s_%s_internal_init", jdf_basename, f->fname);
    jdf_generate_internal_init(jdf, f, prefix);
    free(prefix);

    if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
        jdf_generate_startup_tasks(jdf, f, jdf_property_get_function(f->properties, JDF_PROP_UD_STARTUP_TASKS_FN_NAME, NULL));
    }

    string_arena_add_string(sa, "};\n");

    coutput("%s\n\n", string_arena_get_string(sa));

    string_arena_free(sa2);
    string_arena_free(sa);
    (void)rc;
}

static void jdf_generate_functions_statics( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    string_arena_t *sa;
    int i;

    sa = string_arena_new(64);
    string_arena_add_string(sa, "static const parsec_task_class_t *%s_task_classes[] = {\n",
                            jdf_basename);
    /* We need to put the function in the array based on their task_class_id */
    {
        jdf_function_entry_t** array;
        int max_id;

        for(max_id = 0, f = jdf->functions; NULL != f; f = f->next) {
            jdf_generate_one_function(jdf, f);
            if( max_id < f->task_class_id ) max_id = f->task_class_id;
        }
        max_id++;  /* allow one more space */
        array = (jdf_function_entry_t**)malloc(max_id * sizeof(jdf_function_entry_t*));
        for(i = 0; i < max_id; array[i] = NULL, i++);
        for(f = jdf->functions; NULL != f; f = f->next)
            array[f->task_class_id] = f;
        for(i = 0; i < max_id; array[i] = NULL, i++) {
            if( NULL == (f = array[i]) ) {
                string_arena_add_string(sa, "  NULL%s\n",
                                        i != (max_id - 1) ? "," : "");
            } else {
                string_arena_add_string(sa, "  &%s_%s%s\n",
                                        jdf_basename, f->fname, i != (max_id - 1) ? "," : "");
            }
        }
        free(array);
    }
    string_arena_add_string(sa, "};\n\n");
    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
}

static void jdf_generate_priority_prototypes( const jdf_t *jdf )
{
    jdf_function_entry_t *f;

    for(f = jdf->functions; f != NULL; f = f->next) {
        if( NULL == f->priority ) continue;
        coutput("static inline int priority_of_%s_as_expr_fct(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *assignments);\n",
                JDF_OBJECT_ONAME( f ), jdf_basename, parsec_get_name(jdf, f, "parsec_assignment_t"));
    }
}

static void jdf_generate_startup_hook( const jdf_t *jdf )
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);

    coutput("static void %s_startup(parsec_context_t *context, __parsec_%s_internal_taskpool_t *__parsec_tp, parsec_list_item_t ** ready_tasks)\n"
            "{\n"
            "  uint32_t i, supported_dev = 0;\n"
            " \n"
            "  for( i = 0; i < parsec_nb_devices; i++ ) {\n"
            "    if( !(__parsec_tp->super.super.devices_index_mask & (1<<i)) ) continue;\n"
            "    parsec_device_module_t* device = parsec_mca_device_get(i);\n"
            "    parsec_data_collection_t* parsec_dc;\n"
            " \n"
            "    if(NULL == device) continue;\n"
            "    if(NULL != device->taskpool_register)\n"
            "      if( PARSEC_SUCCESS != device->taskpool_register(device, (parsec_taskpool_t*)__parsec_tp) ) {\n"
            "        parsec_debug_verbose(5, parsec_debug_output, \"Device %%s refused to register taskpool %%p\", device->name, __parsec_tp);\n"
            "        __parsec_tp->super.super.devices_index_mask &= ~(1 << device->device_index);\n"
            "        continue;\n"
            "      }\n"
            "    if(NULL != device->memory_register) {  /* Register all the data */\n"
            "%s"
            "    }\n"
            "    supported_dev |= device->type;\n"
            "  }\n",
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa1, jdf->globals, next,
                           dump_data_name, sa2, "",
                           "      parsec_dc = (parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_",
                           ";\n"
                           "      if( (NULL != parsec_dc->register_memory) &&\n"
                           "          (PARSEC_SUCCESS != parsec_dc->register_memory(parsec_dc, device)) ) {\n"
                           "        parsec_debug_verbose(3, parsec_debug_output, \"Device %s refused to register memory for data %s (%p) from taskpool %p\",\n"
                           "                     device->name, parsec_dc->key_base, parsec_dc, __parsec_tp);\n"
                           "        __parsec_tp->super.super.devices_index_mask &= ~(1 << device->device_index);\n"
                           "      }\n",
                           ";\n"
                           "      if( (NULL != parsec_dc->register_memory) &&\n"
                           "          (PARSEC_SUCCESS != parsec_dc->register_memory(parsec_dc, device)) ) {\n"
                           "        parsec_debug_verbose(3, parsec_debug_output, \"Device %s refused to register memory for data %s (%p) from taskpool %p\",\n"
                           "                     device->name, parsec_dc->key_base, parsec_dc, __parsec_tp);\n"
                           "        __parsec_tp->super.super.devices_index_mask &= ~(1 << device->device_index);\n"
                           "      }\n"));
    coutput("  /* Remove all the chores without a backend device */\n"
            "  for( i = 0; i < PARSEC_%s_NB_TASK_CLASSES; i++ ) {\n"
            "    parsec_task_class_t* tc = (parsec_task_class_t*)__parsec_tp->super.super.task_classes_array[i];\n"
            "    __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;\n"
            "    uint32_t idx = 0, j;\n"
            "    for( j = 0; NULL != chores[j].hook; j++ ) {\n"
            "      if( !(supported_dev & chores[j].type) ) continue;\n"
            "      if( j != idx ) {\n"
            "        chores[idx] = chores[j];\n"
            "        parsec_debug_verbose(20, parsec_debug_output, \"Device type %%i disabledfor function %%s\"\n, chores[j].type, tc->name);\n"
            "      }\n"
            "      idx++;\n"
            "    }\n"
            "    chores[idx].type     = PARSEC_DEV_NONE;\n"
            "    chores[idx].evaluate = NULL;\n"
            "    chores[idx].hook     = NULL;\n"
            "    parsec_task_t* task = (parsec_task_t*)parsec_thread_mempool_allocate(context->virtual_processes[0]->execution_streams[0]->context_mempool);\n"
            "    task->taskpool = (parsec_taskpool_t *)__parsec_tp;\n"
            "    task->chore_mask = PARSEC_DEV_ALL;\n"
            "    task->status = PARSEC_TASK_STATUS_NONE;\n"
            "    memset(&task->locals, 0, sizeof(parsec_assignment_t) * MAX_LOCAL_COUNT);\n"
            "    PARSEC_LIST_ITEM_SINGLETON(task);\n"
            "    task->priority = -1;\n"
            "    task->task_class = task->taskpool->task_classes_array[PARSEC_%s_NB_TASK_CLASSES + i];\n"
            "    int where = i %% context->nb_vp;\n"
            "    if( NULL == ready_tasks[where] ) ready_tasks[where] = &task->super;\n"
            "    else ready_tasks[where] = parsec_list_item_ring_push(ready_tasks[where], &task->super);\n"
            "  }\n",
            jdf_basename, jdf_basename);
    /**
     *  Takes a pointer to a function and if the function can generate startup tasks
     *  creates a single task to iterate over the entire execution space and generate
     *  the startup tasks.
     */
    coutput("}\n");

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_destructor( const jdf_t *jdf )
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);
    jdf_function_entry_t* f;

    coutput("static void __parsec_%s_internal_destructor( __parsec_%s_internal_taskpool_t *__parsec_tp )\n"
            "{\n"
            "  uint32_t i;\n",
            jdf_basename, jdf_basename);
    if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
        coutput("  size_t dependencies_size = 0;\n");
    }

    coutput("  parsec_taskpool_unregister( &__parsec_tp->super.super );\n");

    coutput("  for( i = 0; i < (uint32_t)(2 * __parsec_tp->super.super.nb_task_classes); i++ ) {  /* Extra startup function added at the end */\n"
            "    parsec_task_class_t* tc = (parsec_task_class_t*)__parsec_tp->super.super.task_classes_array[i];\n"
            "    free((void*)tc->incarnations);\n"
            "    free(tc);\n"
            "  }\n"
            "  free(__parsec_tp->super.super.task_classes_array); __parsec_tp->super.super.task_classes_array = NULL;\n"
            "  __parsec_tp->super.super.nb_task_classes = 0;\n"
            "\n"
            "  for(i = 0; i < (uint32_t)__parsec_tp->super.arenas_datatypes_size; i++) {\n"
            "    if( NULL != __parsec_tp->super.arenas_datatypes[i].arena ) {\n"
            "      PARSEC_OBJ_RELEASE(__parsec_tp->super.arenas_datatypes[i].arena);\n"
            "    }\n"
            "  }\n");

    coutput("  /* Destroy the data repositories for this object */\n");
    for( f = jdf->functions; NULL != f; f = f->next ) {
        /* Destruct the repo for all tasks classes, they can be used when:
         * - a predecessor sets up a reshape promised for a datacopy
         * - the own tasks use it when reshaping a datacopy directly read from desc
         * No longer only when if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) )
         */
        coutput("   data_repo_destroy_nothreadsafe(__parsec_tp->repositories[%d]);  /* %s */\n",
                f->task_class_id, f->fname);
    }

    coutput("  /* Release the dependencies arrays for this object */\n");
    for(f = jdf->functions; NULL != f; f = f->next) {
        if( !( f->user_defines & JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS ) ) {
            if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
                coutput("  if(NULL != __parsec_tp->super.super.dependencies_array[%d])\n"
                        "    dependencies_size += parsec_destruct_dependencies( __parsec_tp->super.super.dependencies_array[%d] );\n",
                        f->task_class_id, f->task_class_id);
            } else if (JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ) {
                coutput("  parsec_hash_table_fini( (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[%d] );\n"
                        "  PARSEC_OBJ_RELEASE(__parsec_tp->super.super.dependencies_array[%d]);\n",
                        f->task_class_id, f->task_class_id);
            } 
        } else {
            coutput("  %s(__parsec_tp, __parsec_tp->super.super.dependencies_array[%d]);\n",
                    jdf_property_get_function(f->properties, JDF_PROP_UD_FREE_DEPS_FN_NAME, NULL),
                    f->task_class_id);

        }
        coutput("  __parsec_tp->super.super.dependencies_array[%d] = NULL;\n",
                f->task_class_id);
    }
    coutput("  free( __parsec_tp->super.super.dependencies_array );\n"
            "  __parsec_tp->super.super.dependencies_array = NULL;\n");

    if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
        coutput("#if defined(PARSEC_PROF_TRACE)\n"
                "  {\n"
                "    char meminfo[128];\n"
                "    snprintf(meminfo, 128, \"INDEX_ARRAY - Taskpool %%d - Dependencies - %%zu bytes\",\n"
                "             __parsec_tp->super.super.taskpool_id, dependencies_size);\n"
                "    parsec_profiling_add_information(\"MEMORY_USAGE\", meminfo);\n"
                "  }\n"
                "#endif\n");
    }

    coutput("  /* Unregister all the data */\n"
            "  uint32_t _i;\n"
            "  for( _i = 0; _i < parsec_nb_devices; _i++ ) {\n"
            "    parsec_device_module_t* device;\n"
            "    parsec_data_collection_t* parsec_dc;\n"
            "    if(!(__parsec_tp->super.super.devices_index_mask & (1 << _i))) continue;\n"
            "    if((NULL == (device = parsec_mca_device_get(_i))) || (NULL == device->memory_unregister)) continue;\n"
            "  %s"
            "}\n",
            UTIL_DUMP_LIST(sa, jdf->globals, next,
                           dump_data_name, sa1, "",
                           "  parsec_dc = (parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_",
                           ";\n  if( NULL != parsec_dc->unregister_memory ) { (void)parsec_dc->unregister_memory(parsec_dc, device); };\n",
                           ";\n  if( NULL != parsec_dc->unregister_memory ) { (void)parsec_dc->unregister_memory(parsec_dc, device); };\n"));

    coutput("  /* Unregister the taskpool from the devices */\n"
            "  for( i = 0; i < parsec_nb_devices; i++ ) {\n"
            "    if(!(__parsec_tp->super.super.devices_index_mask & (1 << i))) continue;\n"
            "    __parsec_tp->super.super.devices_index_mask ^= (1 << i);\n"
            "    parsec_device_module_t* device = parsec_mca_device_get(i);\n"
            "    if((NULL == device) || (NULL == device->taskpool_unregister)) continue;\n"
            "    if( PARSEC_SUCCESS != device->taskpool_unregister(device, &__parsec_tp->super.super) ) continue;\n"
            "  }\n");

    coutput("  free(__parsec_tp->super.super.taskpool_name); __parsec_tp->super.super.taskpool_name = NULL;\n");

    coutput("}\n\n");

    string_arena_free(sa);
    string_arena_free(sa1);
}

static void jdf_generate_constructor( const jdf_t* jdf )
{
    string_arena_t *sa1,*sa2;
    profiling_init_info_t pi;
    int idx = 0;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("void __parsec_%s_internal_constructor(__parsec_%s_internal_taskpool_t* __parsec_tp)\n{\n"
            "  parsec_task_class_t* tc;\n"
            "  uint32_t i, j;\n\n",
            jdf_basename, jdf_basename);

    string_arena_init(sa1);

    coutput("  __parsec_tp->super.super.nb_task_classes = PARSEC_%s_NB_TASK_CLASSES;\n"
            "  __parsec_tp->super.super.devices_index_mask = PARSEC_DEVICES_ALL;\n"
            "  __parsec_tp->super.super.taskpool_name = strdup(\"%s\");\n"
            "  __parsec_tp->super.super.update_nb_runtime_task = parsec_add_fetch_runtime_task;\n"
            "  __parsec_tp->super.super.dependencies_array = (void **)\n"
            "              calloc(__parsec_tp->super.super.nb_task_classes, sizeof(void*));\n"
            "  /* Twice the size to hold the startup tasks function_t */\n"
            "  __parsec_tp->super.super.task_classes_array = (const parsec_task_class_t**)\n"
            "              malloc(2 * __parsec_tp->super.super.nb_task_classes * sizeof(parsec_task_class_t*));\n"
            "  __parsec_tp->super.super.nb_tasks = 1;\n"
            "  __parsec_tp->super.super.taskpool_type = PARSEC_TASKPOOL_TYPE_PTG;\n"
            "  __parsec_tp->super.super.nb_pending_actions = 1 + __parsec_tp->super.super.nb_task_classes;  /* for the startup tasks */\n"
            "  __parsec_tp->sync_point = __parsec_tp->super.super.nb_task_classes;\n"
            "  __parsec_tp->startup_queue = NULL;\n"
            "%s",
            jdf_basename, jdf_basename, string_arena_get_string(sa1));
    /* Prepare the functions */
    coutput("  for( i = 0; i < __parsec_tp->super.super.nb_task_classes; i++ ) {\n"
            "    __parsec_tp->super.super.task_classes_array[i] = tc = malloc(sizeof(parsec_task_class_t));\n"
            "    memcpy(tc, %s_task_classes[i], sizeof(parsec_task_class_t));\n"
            "    for( j = 0; NULL != tc->incarnations[j].hook; j++);  /* compute the number of incarnations */\n"
            "    tc->incarnations = (__parsec_chore_t*)malloc((j+1) * sizeof(__parsec_chore_t));\n    "
            "    memcpy((__parsec_chore_t*)tc->incarnations, %s_task_classes[i]->incarnations, (j+1) * sizeof(__parsec_chore_t));\n\n"
            "    /* Add a placeholder for initialization and startup task */\n"
            "    __parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+i] = tc = (parsec_task_class_t*)malloc(sizeof(parsec_task_class_t));\n"
            "    memcpy(tc, (void*)&__parsec_generic_startup, sizeof(parsec_task_class_t));\n"
            "    tc->task_class_id = __parsec_tp->super.super.nb_task_classes + i;\n"
            "    tc->incarnations = (__parsec_chore_t*)malloc(2 * sizeof(__parsec_chore_t));\n"
            "    memcpy((__parsec_chore_t*)tc->incarnations, (void*)__parsec_generic_startup.incarnations, 2 * sizeof(__parsec_chore_t));\n"
            "    tc->release_task = parsec_release_task_to_mempool_and_count_as_runtime_tasks;\n"
            "  }\n",
            jdf_basename,
            jdf_basename);
    /**
     * Prepare the function_t structure for the startup tasks. Count the total
     * number of types of startup tasks to be used to correctly allocate the
     * array of function_t.
     */
    string_arena_init(sa1);
    idx = 0;
    for(jdf_function_entry_t *f = jdf->functions; f != NULL; f = f->next) {
        coutput("  /* Startup task for %s */\n"
                "  tc = (parsec_task_class_t *)__parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+%d];\n"
                "  tc->name = \"Startup for %s\";\n",
                f->fname, idx, f->fname);
        idx++;
        coutput("  tc->prepare_input = (parsec_hook_t*)%s_%s_internal_init;\n",
                jdf_basename, f->fname);
        if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
            coutput("  ((__parsec_chore_t*)&tc->incarnations[0])->hook = (parsec_hook_t *)%s;\n",
                    jdf_property_get_function(f->properties, JDF_PROP_UD_STARTUP_TASKS_FN_NAME, NULL));
        }
    }

    {
        struct jdf_name_list* g;
        int datatype_index = 0;
        jdf_expr_t *arena_strut = NULL;
        jdf_def_list_t* prop;

        coutput("  /* Compute the number of arenas_datatypes: */\n");

        for( g = jdf->datatypes; NULL != g; g = g->next ) {
            coutput("  /*   PARSEC_%s_%s_ARENA  ->  %d */\n",
                    jdf_basename, g->name, datatype_index);
            datatype_index++;
        }
        arena_strut = jdf_find_property(jdf->global_properties, "PARSEC_ARENA_STRUT", &prop);
        if( NULL != arena_strut ) {
            expr_info_t info = EMPTY_EXPR_INFO;

            coutput("  /* and add to that the ARENA_STRUT */\n");

            info.prefix = "";
            info.suffix = "";
            info.sa = string_arena_new(64);
            info.assignments = "NULL";

            coutput("  __parsec_tp->super.arenas_datatypes_size = %d + %s;\n",
                    datatype_index, dump_expr((void**)arena_strut, &info));

            string_arena_free(info.sa);
        } else {
            coutput("  __parsec_tp->super.arenas_datatypes_size = %d;\n", datatype_index);
        }
        coutput("  memset(&__parsec_tp->super.arenas_datatypes[0], 0, __parsec_tp->super.arenas_datatypes_size*sizeof(parsec_arena_datatype_t));\n");
    }

    string_arena_init(sa2);
    pi.sa = sa2;
    pi.idx = 0;
    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, pi.maxidx);
    {
        char* prof = UTIL_DUMP_LIST(sa1, jdf->functions, next,
                                    dump_profiling_init, &pi, "", "    ", "\n", "\n");
        coutput("  /* If profiling is enabled, the keys for profiling */\n"
                "#  if defined(PARSEC_PROF_TRACE)\n");

        if( strcmp(prof, "\n") ) {
            coutput("  __parsec_tp->super.super.profiling_array = %s_profiling_array;\n"
                    "  if( -1 == %s_profiling_array[0] ) {\n"
                    "%s"
                    "  }\n",
                    jdf_basename,
                    jdf_basename,
                    prof);
        } else {
            coutput("  __parsec_tp->super.super.profiling_array = NULL;\n");
        }
        coutput("#  endif /* defined(PARSEC_PROF_TRACE) */\n");
    }

    coutput("  __parsec_tp->super.super.repo_array = %s;\n",
            (NULL != jdf->functions) ? "__parsec_tp->repositories" : "NULL");

    coutput("  __parsec_tp->super.super.startup_hook = (parsec_startup_fn_t)%s_startup;\n"
            "  (void)parsec_taskpool_reserve_id((parsec_taskpool_t*)__parsec_tp);\n"
            "}\n\n",
            jdf_basename);

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_new_function( const jdf_t* jdf )
{
    string_arena_t *sa1,*sa2;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->globals, next, name,
                                  dump_string, NULL, "", "#undef ", "\n", "\n"));

    {
        typed_globals_info_t prop = { sa2, NULL, "hidden", .prefix = "" };
        coutput("parsec_%s_taskpool_t *parsec_%s_new(%s)\n{\n",
                jdf_basename, jdf_basename,
                UTIL_DUMP_LIST( sa1, jdf->globals, next, dump_typed_globals, &prop,
                                "", "", ", ", ""));
    }

    coutput("  __parsec_%s_internal_taskpool_t *__parsec_tp = PARSEC_OBJ_NEW(__parsec_%s_internal_taskpool_t);\n",
            jdf_basename, jdf_basename);

    string_arena_init(sa1);
    string_arena_init(sa2);
    coutput("  /* Dump the hidden parameters with default values */\n"
            "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                 dump_hidden_globals_init, sa2, "", "  ", "\n", "\n"));

    string_arena_init(sa1);
    string_arena_init(sa2);
    coutput("  /* Now the Parameter-dependent structures: */\n"
            "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                 dump_globals_init, sa2, "", "  ", "\n", "\n"));

    for(jdf_function_entry_t *f = jdf->functions; f != NULL; f = f->next) {
        coutput("  PARSEC_AYU_REGISTER_TASK(&%s_%s);\n",
                jdf_basename, f->fname);
    }

    coutput("  __parsec_tp->super.super.startup_hook = (parsec_startup_fn_t)%s_startup;\n"
            "  (void)parsec_taskpool_reserve_id((parsec_taskpool_t*)__parsec_tp);\n",
            jdf_basename);

    string_arena_init(sa1);
    string_arena_init(sa2);
    coutput("/* Prevent warnings related to not used hidden global variables */\n"
            "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                 dump_name_of_hidden_globals_without_default, sa2, "", "  (void)", ";", ";\n"));

    coutput("  return (parsec_%s_taskpool_t*)__parsec_tp;\n"
            "}\n\n",
            jdf_basename);

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_hashfunction_for(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    string_arena_t *sa_range_multiplier = string_arena_new(64);
    jdf_variable_list_t *vl;
    expr_info_t info = EMPTY_EXPR_INFO;
    int idx;

    if( !(f->user_defines & JDF_FUNCTION_HAS_UD_MAKE_KEY) ) {
        coutput("static inline parsec_key_t %s(const parsec_taskpool_t *tp, const parsec_assignment_t *as)\n"
                "{\n",
                jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL));
        if( f->parameters == NULL ) {
            coutput("  return (parsec_key_t)0;\n"
                    "}\n");
        } else {
            coutput( "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (const __parsec_%s_internal_taskpool_t *)tp;\n"
                     "  %s ascopy, *assignment = &ascopy;\n"
                     "  uintptr_t __parsec_id = 0;\n"
                     "  memcpy(assignment, as, sizeof(%s));\n",
                     jdf_basename, jdf_basename,
                     parsec_get_name(jdf, f, "parsec_assignment_t"),
                     parsec_get_name(jdf, f, "parsec_assignment_t"));
            
            info.prefix = "";
            info.suffix = "";
            info.sa = sa_range_multiplier;
            info.assignments = "assignment";

            idx = 0;
            for(vl = f->locals; vl != NULL; vl = vl->next) {
                string_arena_init(sa_range_multiplier);
                
                coutput("  const int %s = assignment->%s.value;\n",
                        vl->name, vl->name);
                
                if( local_is_parameter(f, vl) != NULL ) {
                    if( vl->expr->op == JDF_RANGE ) {
                        coutput("  int %s%s_min = %s;\n", JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr->jdf_ta1, &info));
                    } else {
                        if( vl->expr->local_variables != NULL ) {
                            char *vname;
                            if( asprintf(&vname, "%s%s_min", JDF2C_NAMESPACE, vl->name) <= 0 ) {
                                fprintf(stderr, "Cannot allocate internal memory for the PTG compiler\n");
                                exit(-1);
                            }
                            coutput("  int %s;\n", vname);
                            jdf_generate_range_min_without_fn(jdf, vl->expr, vname, "assignment");
                            free(vname);
                        } else {
                            coutput("  int %s%s_min = %s;\n", JDF2C_NAMESPACE, vl->name, dump_expr((void**)vl->expr, &info));
                        }
                    }
                } else {
                    /* IDs should depend only on the parameters of the
                     * function. However, we might need the other definitions because
                     * the min expression of the parameters might depend on them. If
                     * this is not the case, a quick "(void)" removes the warning.
                     */
                    coutput("  (void)%s;\n", vl->name);
                }
                idx++;
            }

            string_arena_init(sa_range_multiplier);
            for(vl = f->locals; vl != NULL; vl = vl->next) {
                if( local_is_parameter(f, vl) != NULL ) {
                    coutput("  __parsec_id += (%s - %s%s_min)%s;\n", vl->name, JDF2C_NAMESPACE, vl->name, string_arena_get_string(sa_range_multiplier));
                    string_arena_add_string(sa_range_multiplier, " * __parsec_tp->%s_%s_range", f->fname, vl->name);
                }
            }

            coutput("  (void)__parsec_tp;\n"
                    "  return (parsec_key_t)__parsec_id;\n"
                    "}\n");
        }
    }

    if( !(f->user_defines & JDF_FUNCTION_HAS_UD_HASH_STRUCT) ) {
        jdf_generate_deps_key_functions(jdf, f, jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL));
    }

    string_arena_free(sa_range_multiplier);
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
    expr_info_t info = EMPTY_EXPR_INFO;

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
  expr_info_t infodst = EMPTY_EXPR_INFO, infosrc = EMPTY_EXPR_INFO;
  string_arena_t *sa2;
  jdf_expr_t *params = call->parameters;
  jdf_variable_list_t *vl;
  jdf_param_list_t *pl;
  const jdf_function_entry_t *f;

  f = find_target_function(jdf, call->func_or_mem);

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
  infosrc.assignments = "&this_task->locals";

  for(vl = f->locals; vl != NULL; vl = vl->next) {
      /* Is this definition a parameter or a value? */
      /* If it is a parameter, find the corresponding param in the call */
      for(el = params, pl = f->parameters; pl != NULL; el = el->next, pl = pl->next) {
          if( NULL == el ) {  /* Badly formulated call */
              string_arena_t *sa_caller, *sa_callee;
              expr_info_t caller = EMPTY_EXPR_INFO;

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
          if(!strcmp(pl->name, vl->name))
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
                                  indent(spaces), f->fname, vl->name, name, vl->name, dump_expr((void**)vl->expr, &infodst), f->fname, vl->name);
      } else {
          /* It is a parameter. Let's dump it's expression in the source context */
          assert(el != NULL);
          string_arena_add_string(sa,
                                  "%sconst int %s%s = %s->%s.value = %s; (void)%s%s;\n",
                                  indent(spaces), f->fname, vl->name, name, vl->name, dump_expr((void**)el, &infosrc), f->fname, vl->name);
      }
  }

  string_arena_free(sa2);

  return string_arena_get_string(sa);
}

static void
jdf_generate_arena_string_from_datatype(string_arena_t *sa,
                                        jdf_datatransfer_type_t datatype)
{
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals, this_task ";
    if( JDF_CST == datatype.type->op ) {
        string_arena_add_string(sa, "%d", datatype.type->jdf_cst);
    } else if( (JDF_VAR == datatype.type->op) || (JDF_STRING == datatype.type->op) ) {
        string_arena_add_string(sa, "PARSEC_%s_%s_ADT", jdf_basename, datatype.type->jdf_var);
    } else {
        string_arena_add_string(sa, "%s", dump_expr((void**)datatype.type, &info));
    }
    string_arena_free(sa2);
}

static void
jdf_generate_code_fillup_datatypes(string_arena_t * sa_tmp_arena,     string_arena_t * sa_arena,
                                   string_arena_t * sa_tmp_type_src,  string_arena_t * sa_type_src,
                                   string_arena_t * sa_tmp_displ_src, string_arena_t * sa_displ_src,
                                   string_arena_t * sa_tmp_count_src, string_arena_t * sa_count_src,
                                   string_arena_t * sa_tmp_type_dst,
                                   string_arena_t * sa_tmp_displ_dst,
                                   string_arena_t * sa_tmp_count_dst,
                                   string_arena_t * sa_out,
                                   char * access, char * target, int dump_all)
{
    /* Prepare the memory layout of the output dependency. */
    if( dump_all || (sa_arena == NULL) || strcmp(string_arena_get_string(sa_tmp_arena), string_arena_get_string(sa_arena))) {
        if( strcmp(target, "local") == 0) {
            if(sa_arena != NULL) string_arena_add_string(sa_out, "    data%sdata_future  = NULL;\n", access);
        }
        /* The type might change (possibly from undefined), so let's output */
        if(sa_arena != NULL){
            string_arena_init(sa_arena);
            string_arena_add_string(sa_arena, "%s", string_arena_get_string(sa_tmp_arena));
        }
        string_arena_add_string(sa_out, "    data%s%s.arena  = %s;\n", access, target, string_arena_get_string(sa_tmp_arena));
        /* As we change the arena force the reset of the layout */
        if(sa_type_src != NULL){
            string_arena_init(sa_type_src);
        }
    }
    if( dump_all || (sa_type_src == NULL) || strcmp(string_arena_get_string(sa_tmp_type_src), string_arena_get_string(sa_type_src))) {
        /* Same thing: the memory layout may change at anytime */
        if(sa_type_src != NULL){
            string_arena_init(sa_type_src);
            string_arena_add_string(sa_type_src, "%s", string_arena_get_string(sa_tmp_type_src));
        }
        string_arena_add_string(sa_out, "    data%s%s.src_datatype = %s;\n"
                                        "    data%s%s.dst_datatype = %s;\n",
                                        access, target, string_arena_get_string(sa_tmp_type_src),
                                        access, target, (sa_tmp_type_dst == NULL) ? string_arena_get_string(sa_tmp_type_src) :
                                                            string_arena_get_string(sa_tmp_type_dst));
    }
    if( dump_all || ( sa_count_src == NULL ) || strcmp(string_arena_get_string(sa_tmp_count_src), string_arena_get_string(sa_count_src))) {
        /* Same thing: the number of transmitted elements may change at anytime */
        if( sa_count_src != NULL ){
            string_arena_init(sa_count_src);
            string_arena_add_string(sa_count_src, "%s", string_arena_get_string(sa_tmp_count_src));
        }
        string_arena_add_string(sa_out, "    data%s%s.src_count  = %s;\n"
                                        "    data%s%s.dst_count  = %s;\n",
                                        access, target, string_arena_get_string(sa_tmp_count_src),
                                        access, target, (sa_tmp_count_dst == NULL) ? string_arena_get_string(sa_tmp_count_src) :
                                                            string_arena_get_string(sa_tmp_count_dst));
    }
    if( dump_all || ( sa_displ_src == NULL) || strcmp(string_arena_get_string(sa_tmp_displ_src), string_arena_get_string(sa_displ_src))) {
        /* Same thing: the displacement may change at anytime */
        if( sa_displ_src != NULL){
            string_arena_init(sa_displ_src);
            string_arena_add_string(sa_displ_src, "%s", string_arena_get_string(sa_tmp_displ_src));
        }
        string_arena_add_string(sa_out, "    data%s%s.src_displ    = %s;\n"
                                        "    data%s%s.dst_displ    = %s;\n",
                                        access, target, string_arena_get_string(sa_tmp_displ_src),
                                        access, target, (sa_tmp_displ_dst == NULL) ? string_arena_get_string(sa_tmp_displ_src) :
                                                            string_arena_get_string(sa_tmp_displ_dst));
    }
}

/* After fulfilling the reshaping set up by the predecessor, we may need to
 * fulfill a second local reshaping, because of the reshaping
 * type on this input dependency. Thus, we need to:
 * - clean up the stuff from the repo that we have gotten this copy from
 * - do an inline reshaping, creating our own reshaping promise, and fulfill it
 *   and keep it on the repo if it will be clean up during release_deps_of
 */
static void
jdf_generate_code_reshape_input_from_desc(const jdf_t *jdf,
                                          const jdf_function_entry_t* f, const jdf_dataflow_t *flow,
                                          const jdf_dep_t *dl,
                                          const char *spaces)
{
    (void)jdf; (void)f; (void)flow;
    string_arena_t *sa, *sa2;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa_datatype;
    string_arena_t *sa_tmp_arena;
    string_arena_t *sa_type_src;
    string_arena_t *sa_tmp_type_src, *sa_tmp_displ_src, *sa_tmp_count_src;

    string_arena_t *sa_type_dst;
    string_arena_t *sa_tmp_type_dst, *sa_tmp_displ_dst, *sa_tmp_count_dst;

    sa_datatype = string_arena_new(64);

    sa_type_src = string_arena_new(64);
    sa_type_dst = string_arena_new(64);

    sa_tmp_arena = string_arena_new(64);
    sa_tmp_type_src = string_arena_new(64);
    sa_tmp_displ_src = string_arena_new(64);
    sa_tmp_count_src = string_arena_new(64);
    sa_tmp_type_dst = string_arena_new(64);
    sa_tmp_displ_dst = string_arena_new(64);
    sa_tmp_count_dst = string_arena_new(64);

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    string_arena_init(sa_datatype);

    string_arena_init(sa_type_src);
    string_arena_init(sa_type_dst);
    string_arena_init(sa_tmp_arena);
    string_arena_init(sa_tmp_type_src);
    string_arena_init(sa_tmp_displ_src);
    string_arena_init(sa_tmp_count_src);
    string_arena_init(sa_tmp_type_dst);
    string_arena_init(sa_tmp_displ_dst);
    string_arena_init(sa_tmp_count_dst);

    jdf_datatransfer_type_t reshape_dtt_dst, reshape_dtt_src;

    /* Reading from matrix: allowing inline reshaping:
     *
     *   (1) If !undef(type) && !undef(type_data)
     *           Pack desc(m,n) type_data, Unpack type
     *
     *   (2) If undef(type) && !undef(type_data)
     *           Pack desc(m,n) type_data, Unpack type_data
     *
     *   (3) If !undef(type) && undef(type_data)
     *           Pack desc(m,n) desc(m,n).type, Unpack type
     *
     *   (4) If undef(type) && undef(type_data)  DON'T DO ANYTHING
     */
    assert( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dl->datatype_remote)) );

    /* If we don't have a reshaping type, we want to consume the default
     * reshaping of the future. We mark that with NULL and MPI_DATYPE_NULL.
     * Otherwise, we use SRC and DST dt.
    */
    coutput("%s    data.data = chunk;", spaces);
    reshape_dtt_dst = dl->datatype_local;
    reshape_dtt_src = dl->datatype_data;

    if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_dst))
            &&(DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_src) )){
        /* Don't do reshape*/
        string_arena_add_string(sa_tmp_arena, "NULL");
        string_arena_add_string(sa_tmp_type_src, "PARSEC_DATATYPE_NULL");
        string_arena_add_string(sa_tmp_type_dst, "PARSEC_DATATYPE_NULL");
        string_arena_add_string(sa_tmp_displ_src, "0");
        string_arena_add_string(sa_tmp_count_src, "1");
        string_arena_add_string(sa_tmp_displ_dst, "0");
        string_arena_add_string(sa_tmp_count_dst, "1");
    }else{
        /* packing */
        if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_src)) /* undefined type on dependency */
                && (NULL == reshape_dtt_src.layout) ){ /* User didn't specify a custom layout*/
            string_arena_add_string(sa_tmp_displ_src, "0");
            string_arena_add_string(sa_tmp_count_src, "1");
            string_arena_add_string(sa_tmp_type_src, "data.data->dtt");
        }else{
            jdf_generate_arena_string_from_datatype(sa_type_src, reshape_dtt_src);
            if( NULL == reshape_dtt_src.layout ){ /* User didn't specify a custom layout*/
                string_arena_add_string(sa_tmp_type_src, "%s->opaque_dtt", string_arena_get_string(sa_type_src));
            }else{
                string_arena_add_string(sa_tmp_type_src, "%s", dump_expr((void**)reshape_dtt_src.layout, &info));
            }
            assert( reshape_dtt_src.count != NULL );
            string_arena_add_string(sa_tmp_displ_src, "%s", dump_expr((void**)reshape_dtt_src.displ, &info));
            string_arena_add_string(sa_tmp_count_src, "%s", dump_expr((void**)reshape_dtt_src.count, &info));
        }

        /* unpacking */
        if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_dst)) /* undefined type on dependency */
                && (NULL == reshape_dtt_dst.layout) ){ /* User didn't specify a custom layout*/
            reshape_dtt_dst = reshape_dtt_src;/* we use the type_data too*/
        }
        jdf_generate_arena_string_from_datatype(sa_type_dst, reshape_dtt_dst);

        string_arena_add_string(sa_tmp_arena, "%s->arena", string_arena_get_string(sa_type_dst));

        if( NULL == reshape_dtt_dst.layout ){ /* User didn't specify a custom layout*/
            string_arena_add_string(sa_tmp_type_dst, "%s->opaque_dtt", string_arena_get_string(sa_type_dst));
        }else{
            string_arena_add_string(sa_tmp_type_dst, "%s", dump_expr((void**)reshape_dtt_dst.layout, &info));
        }
        assert( reshape_dtt_dst.count != NULL );
        string_arena_add_string(sa_tmp_displ_dst, "%s", dump_expr((void**)reshape_dtt_dst.displ, &info));
        string_arena_add_string(sa_tmp_count_dst, "%s", dump_expr((void**)reshape_dtt_dst.count, &info));
    }

    jdf_generate_code_fillup_datatypes(sa_tmp_arena,     NULL,
                                       sa_tmp_type_src,  NULL,
                                       sa_tmp_displ_src, NULL,
                                       sa_tmp_count_src, NULL,
                                       sa_tmp_type_dst,
                                       sa_tmp_displ_dst,
                                       sa_tmp_count_dst,
                                       sa_datatype,
                                       ".", "local", 1);
    coutput("%s    %s", spaces, string_arena_get_string(sa_datatype));

    /*INLINE RESHAPING WHEN READING FROM MATRIX NEW COPY EACH THREAD*/
    coutput("%s    data.data_future   = NULL;\n", spaces);
    coutput("%s    if( (ret = parsec_get_copy_reshape_from_desc(es, this_task->taskpool, (parsec_task_t *)this_task, %d, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){\n"
            "%s        return ret;\n"
            "%s    }\n",
            spaces, flow->flow_index,
            spaces,
            spaces);

    string_arena_free(sa);
    string_arena_free(sa2);

    string_arena_free(sa_datatype);
    string_arena_free(sa_type_src);
    string_arena_free(sa_type_dst);
    string_arena_free(sa_tmp_arena);
    string_arena_free(sa_tmp_type_src);
    string_arena_free(sa_tmp_displ_src);
    string_arena_free(sa_tmp_count_src);
    string_arena_free(sa_tmp_type_dst);
    string_arena_free(sa_tmp_displ_dst);
    string_arena_free(sa_tmp_count_dst);
}

/* After fulfilling the reshaping set up by the predecessor, we may need to
 * fulfill a second local reshaping, because of the reshaping
 * type on this input dependency. Thus, we need to:
 * - clean up the stuff from the repo that we have gotten this copy from
 * - do an inline reshaping, creating our own reshaping promise, and fulfill it
 *   and keep it on the repo if it will be clean up during release_deps_of
 */
static void
jdf_generate_code_reshape_input_from_dep(const jdf_t *jdf,
                                         const jdf_function_entry_t* f, const jdf_dataflow_t *flow,
                                         const jdf_dep_t *dl, const char *spaces)
{
    (void)jdf; (void)f; (void)flow;
    string_arena_t *sa, *sa2;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa_datatype;
    string_arena_t *sa_tmp_arena;
    string_arena_t *sa_type_src;
    string_arena_t *sa_tmp_type_src, *sa_tmp_displ_src, *sa_tmp_count_src;

    string_arena_t *sa_type_dst;
    string_arena_t *sa_tmp_type_dst, *sa_tmp_displ_dst, *sa_tmp_count_dst;

    sa_datatype = string_arena_new(64);

    sa_type_src = string_arena_new(64);
    sa_type_dst = string_arena_new(64);

    sa_tmp_arena = string_arena_new(64);
    sa_tmp_type_src = string_arena_new(64);
    sa_tmp_displ_src = string_arena_new(64);
    sa_tmp_count_src = string_arena_new(64);
    sa_tmp_type_dst = string_arena_new(64);
    sa_tmp_displ_dst = string_arena_new(64);
    sa_tmp_count_dst = string_arena_new(64);

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    string_arena_init(sa_datatype);

    string_arena_init(sa_type_src);
    string_arena_init(sa_type_dst);
    string_arena_init(sa_tmp_arena);
    string_arena_init(sa_tmp_type_src);
    string_arena_init(sa_tmp_displ_src);
    string_arena_init(sa_tmp_count_src);
    string_arena_init(sa_tmp_type_dst);
    string_arena_init(sa_tmp_displ_dst);
    string_arena_init(sa_tmp_count_dst);

    jdf_datatransfer_type_t reshape_dtt_dst, reshape_dtt_src;

    /* Reshaping input dependency from another task
     * Format: type = XX   type_remote = ... -> pack XX unpack XX
     * */
    coutput("%s    data.data   = NULL;\n", spaces);
    coutput("%s    data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];\n",
            spaces);

    /* Source and dst datatypes are [type] */
    reshape_dtt_dst = reshape_dtt_src = dl->datatype_local;

    jdf_generate_arena_string_from_datatype(sa_type_dst, reshape_dtt_dst);
    jdf_generate_arena_string_from_datatype(sa_type_src, reshape_dtt_src);

    assert( reshape_dtt_dst.count != NULL );
    assert( reshape_dtt_src.count != NULL );

    /* If we don't have a reshaping type, we want to consume the default
     * reshaping of the future. We mark that with NULL and MPI_DATYPE_NULL.
     * Otherwise, we use SRC and DST dt.
     * */

    if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_dst)) /* undefined type on dependency */
            && (NULL == reshape_dtt_dst.layout) ){ /* User didn't specify a custom layout*/
        string_arena_add_string(sa_tmp_arena, "NULL");
    }else{/* Custom type on dependency -> using deptype arena*/
        string_arena_add_string(sa_tmp_arena, "%s->arena", string_arena_get_string(sa_type_dst));
    }

    if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_src)) /* undefined type on dependency */
            && (NULL == reshape_dtt_src.layout) ){ /* User didn't specify a custom layout*/
        string_arena_add_string(sa_tmp_type_src, "PARSEC_DATATYPE_NULL");
    }else{
        if( NULL == reshape_dtt_src.layout ){ /* User didn't specify a custom layout*/
            string_arena_add_string(sa_tmp_type_src, "%s->opaque_dtt", string_arena_get_string(sa_type_src));
        }else{
            string_arena_add_string(sa_tmp_type_src, "%s", dump_expr((void**)reshape_dtt_src.layout, &info));
        }
    }

    if( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_dst)) /* undefined type on dependency */
            && (NULL == reshape_dtt_dst.layout) ){ /* User didn't specify a custom layout*/
        string_arena_add_string(sa_tmp_type_dst, "PARSEC_DATATYPE_NULL");
    }else{
        if( NULL == reshape_dtt_dst.layout ){ /* User didn't specify a custom layout*/
            string_arena_add_string(sa_tmp_type_dst, "%s->opaque_dtt", string_arena_get_string(sa_type_dst));
        }else{
            string_arena_add_string(sa_tmp_type_dst, "%s", dump_expr((void**)reshape_dtt_dst.layout, &info));
        }
    }

    string_arena_add_string(sa_tmp_displ_src, "%s", dump_expr((void**)reshape_dtt_src.displ, &info));
    string_arena_add_string(sa_tmp_count_src, "%s", dump_expr((void**)reshape_dtt_src.count, &info));
    string_arena_add_string(sa_tmp_displ_dst, "%s", dump_expr((void**)reshape_dtt_dst.displ, &info));
    string_arena_add_string(sa_tmp_count_dst, "%s", dump_expr((void**)reshape_dtt_dst.count, &info));

    jdf_generate_code_fillup_datatypes(sa_tmp_arena,     NULL,
                                       sa_tmp_type_src,  NULL,
                                       sa_tmp_displ_src, NULL,
                                       sa_tmp_count_src, NULL,
                                       sa_tmp_type_dst,
                                       sa_tmp_displ_dst,
                                       sa_tmp_count_dst,
                                       sa_datatype,
                                       ".", "local", 1);
    coutput("%s    %s", spaces, string_arena_get_string(sa_datatype));

    coutput("%s    if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, %d, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){\n"
            "%s        return ret;\n"
            "%s    }\n",
            spaces, flow->flow_index,
            spaces,
            spaces);

    string_arena_free(sa);
    string_arena_free(sa2);

    string_arena_free(sa_datatype);
    string_arena_free(sa_type_src);
    string_arena_free(sa_type_dst);
    string_arena_free(sa_tmp_arena);
    string_arena_free(sa_tmp_type_src);
    string_arena_free(sa_tmp_displ_src);
    string_arena_free(sa_tmp_count_src);
    string_arena_free(sa_tmp_type_dst);
    string_arena_free(sa_tmp_displ_dst);
    string_arena_free(sa_tmp_count_dst);
}

static void
jdf_generate_code_consume_predecessor_setup(const jdf_t *jdf, const jdf_call_t *call,
                                                  const jdf_function_entry_t* f, const jdf_dataflow_t *flow,
                                                  const jdf_dep_t *dl, const char *spaces,
                                                  int final_check/*1 = Final check predecessor setting repo & stuff */,
                                                  const char *pred_var, const char *pred_name,
                                                  const jdf_function_entry_t* pred_f, const jdf_dataflow_t *pred_flow)
{
    (void)jdf; (void)call; (void)dl; (void)f;
    (void)pred_var;
    /*final_check = case predecessor has set up stuff on our data */

    if( final_check ){

        coutput("%s    consumed_repo = this_task->data._f_%s.source_repo;\n"
                "%s    consumed_entry = this_task->data._f_%s.source_repo_entry;\n"
                "%s    consumed_entry_key = this_task->data._f_%s.source_repo_entry->ht_item.key;\n",
                spaces, flow->varname,
                spaces, flow->varname,
                spaces, flow->varname);
        /* Is this current task repo or predecessor repo? Different flow index */

        coutput("%s    if( (reshape_entry != NULL) && (reshape_entry->data[%d] != NULL) ){\n"
                "%s        /* Reshape promise set up on input by predecessor is on this task repo */\n"
                "%s        consumed_flow_index = %d;\n",
                spaces, flow->flow_index,
                spaces,
                spaces, flow->flow_index);
        coutput("%s        assert( (this_task->data._f_%s.source_repo == reshape_repo)\n"
                "%s             && (this_task->data._f_%s.source_repo_entry == reshape_entry));\n",
                spaces, flow->varname,
                spaces, flow->varname);
        coutput("%s    }else{\n"
                "%s        /* Reshape promise set up on input by predecessor is the predecesssor task repo */\n"
                "%s        consumed_flow_index = %d;\n"
                "%s    }\n",
                spaces,
                spaces,
                spaces, pred_flow->flow_index,
                spaces);
    }else{
        coutput("%s    if( (reshape_entry != NULL) && (reshape_entry->data[%d] != NULL) ){\n"
                "%s        /* Reshape promise set up on this task repo by predecessor */\n"
                "%s        consumed_repo = reshape_repo;\n"
                "%s        consumed_entry = reshape_entry;\n"
                "%s        consumed_entry_key = reshape_entry_key;\n"
                "%s        consumed_flow_index = %d;\n",
                spaces, flow->flow_index,
                spaces,
                spaces,
                spaces,
                spaces,
                spaces, flow->flow_index);
        coutput("%s    }else{\n"
                "%s        /* Consume from predecessor's repo */\n"
                "%s        consumed_repo = %s_repo;\n"
                "%s        consumed_entry_key = %s((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;\n"
                "%s        consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );\n"
                "%s        consumed_flow_index = %d;\n"
                "%s    }\n",
                spaces,
                spaces,
                spaces, pred_name,
                spaces, jdf_property_get_string(pred_f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL),
                spaces,
                spaces, pred_flow->flow_index,
                spaces);
    }
}

static void
jdf_generate_code_call_initialization(const jdf_t *jdf, const jdf_call_t *call,
                                      const jdf_function_entry_t* f, const jdf_dataflow_t *flow,
                                      const jdf_dep_t *dl, const char *spaces)
{
    string_arena_t *sa, *sa2;
    expr_info_t info = EMPTY_EXPR_INFO;
    const jdf_dataflow_t* tflow = NULL;
    const jdf_function_entry_t* targetf = NULL;
    const char *fname = f->fname;
    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    if( call->var != NULL ) {
        targetf = find_target_function(jdf, call->func_or_mem);
        if( NULL == targetf ) {
            jdf_fatal(JDF_OBJECT_LINENO(flow),
                      "During code generation: unable to find the source function %s\n"
                      "required by function %s flow %s to satisfy INPUT dependency at line %d\n",
                      call->func_or_mem,
                      fname, call->var, JDF_OBJECT_LINENO(flow));
            exit(1);
        }

        tflow = jdf_data_output_flow(jdf, call->func_or_mem, call->var);
        if( NULL == tflow ) {
            jdf_fatal(JDF_OBJECT_LINENO(flow),
                      "During code generation: unable to find an output flow for variable %s in function %s,\n"
                      "which is requested by function %s to satisfy Input dependency at line %d\n",
                      call->var, call->func_or_mem,
                      fname, JDF_OBJECT_LINENO(flow));
            exit(1);
        }
    }

    /* Function calls */
    if( call->var != NULL ) {
        string_arena_add_string(sa, "from predecessor %s", targetf->fname);
    } else {
        /* Memory references */
        if ( call->parameters != NULL) {
            string_arena_add_string(sa, "from memory %s", call->func_or_mem);
        } else {/* NEW or NULL data */
            string_arena_add_string(sa, "from memory NEW or NULL data");
        }
    }

    /********************************************/
    /* NO DATA SET ON OUR INPUT BY PREDECESSOR  */
    /********************************************/
    coutput("%s  /* Flow %s [%d] dependency [%d] %s */\n",
             spaces, flow->varname, flow->flow_index, dl->dep_index, string_arena_get_string(sa) );
    string_arena_init(sa);

    coutput("%s  if( NULL == (chunk = this_task->data._f_%s.data_in) ) {\n"
            "%s  /* No data set up by predecessor on this task input flow */\n",
             spaces, flow->varname,
             spaces);

    /* Function calls */
    if( call->var != NULL ) {
        coutput("%s *target_locals = (%s*)&generic_locals;\n",
                parsec_get_name(jdf, targetf, "parsec_assignment_t"), parsec_get_name(jdf, targetf, "parsec_assignment_t"));
        coutput("%s", jdf_create_code_assignments_calls(sa, strlen(spaces)+1, jdf, "target_locals", call));

        /* Code to fulfill a reshape promise set up by predecessor if there's one */
        jdf_generate_code_consume_predecessor_setup(jdf, call,
                                                    f, flow,
                                                    dl, spaces, 0 /*1 = Final check predecessor setting repo & stuff */,
                                                    call->var, call->func_or_mem,
                                                    targetf, tflow);

        /* Code to create & fulfill a reshape promise locally in case this input dependency is typed */
        jdf_generate_code_reshape_input_from_dep(jdf, f, flow, dl, spaces);

        coutput("%s    ACQUIRE_FLOW(this_task, \"%s\", &%s_%s, \"%s\", target_locals, chunk);\n"
                "%s    this_task->data._f_%s.data_out = chunk;\n",
                spaces, flow->varname, jdf_basename, call->func_or_mem, call->var,
                spaces, flow->varname);

    }
    else {
        /* Memory references */
        if ( call->parameters != NULL) {
            /* Code to create & fulfill a reshape promise locally in case this input dependency is typed */
            coutput("#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)\n"
                    "%s  parsec_prof_grapher_data_input(data_of_%s(%s), (parsec_task_t*)this_task, &%s, 1);\n"
                    "#endif\n",
                    spaces,
                    call->func_or_mem, UTIL_DUMP_LIST(sa, call->parameters, next,
                                                      dump_expr, (void*)&info, "", "", ", ", ""),
                    JDF_OBJECT_ONAME( flow ));
            coutput("%s    chunk = parsec_data_get_copy(data_of_%s(%s), target_device);\n",
                    spaces, call->func_or_mem,
                    UTIL_DUMP_LIST(sa, call->parameters, next,
                                   dump_expr, (void*)&info, "", "", ", ", ""));

            jdf_generate_code_reshape_input_from_desc(jdf, f, flow, dl, spaces);

            coutput("%s    this_task->data._f_%s.data_out = chunk;\n"
                    "%s    PARSEC_OBJ_RETAIN(chunk);\n",
                    spaces, flow->varname,
                    spaces);

        }
        /* NEW or NULL data */
        else {
            assert( JDF_IS_CALL_WITH_NO_INPUT(call) &&
                    (0 == strcmp( PARSEC_WRITE_MAGIC_NAME, call->func_or_mem ) ||
                     0 == strcmp( PARSEC_NULL_MAGIC_NAME, call->func_or_mem )) );

            if ( strcmp( PARSEC_WRITE_MAGIC_NAME, call->func_or_mem ) == 0 ) {
                info.sa = string_arena_new(64);
                info.prefix = "";
                info.suffix = "";
                info.assignments = "    &this_task->locals";

                /* New datacopy type must be specified on the [type].
                 */
                assert( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dl->datatype_remote))
                        && (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dl->datatype_data)));

                jdf_generate_arena_string_from_datatype(sa, dl->datatype_local);
                assert( dl->datatype_local.count != NULL );
                string_arena_add_string(sa2, "%s", dump_expr((void**)dl->datatype_local.count, &info));

                coutput("%s    chunk = parsec_arena_get_copy(%s->arena, %s, target_device, %s->opaque_dtt);\n"
                        "%s    chunk->original->owner_device = target_device;\n"
                        "%s    this_task->data._f_%s.data_out = chunk;\n",
                        spaces, string_arena_get_string(sa), string_arena_get_string(sa2), string_arena_get_string(sa),
                        spaces,
                        spaces, flow->varname);

                string_arena_free(info.sa);
            }
        }
    }

    /**********************************************/
    /* SOME DATA SET ON OUR INPUT BY PREDECESSOR  */
    /**********************************************/
    coutput("%s  } \n", spaces);
    if( call->var != NULL ) { /* dep from predecessor: check if we have to reshape */
        coutput("%s  else {\n"
                "%s    /* Data set up by predecessor on this task input flow */\n",
                spaces,
                spaces);

        /* Code to fulfill a reshape promise set up by predecessor if there's one */
        jdf_generate_code_consume_predecessor_setup(jdf, call,
                                                    f, flow,
                                                    dl, spaces, 1 /*1 = Final check predecessor setting repo & stuff */,
                                                    call->var, call->func_or_mem,
                                                    targetf, tflow);

        /* Code to create & fulfill a reshape promise locally in case this input dependency is typed */
        jdf_generate_code_reshape_input_from_dep(jdf, f, flow, dl, spaces);
        coutput("%s    this_task->data._f_%s.data_out = parsec_data_get_copy(chunk->original, target_device);\n"
                "#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)\n"
                "%s    parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &%s, 0);\n"
                "#endif\n"
                "%s  }\n",
                spaces, flow->varname,
                spaces, JDF_OBJECT_ONAME( flow ),
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
                                               const jdf_dataflow_t *flow, const char *fname,
                                               const jdf_dep_t *dl,
                                               const char *spaces)
{
    int dataindex;
    string_arena_t *sa, *sa_arena, *sa_datatype, *sa_count ;
    expr_info_t info = EMPTY_EXPR_INFO;

    if( (NULL != call) && (NULL != call->var) ) {
        dataindex = jdf_data_input_index(jdf, call->func_or_mem, call->var);
        if( dataindex < 0 ) {
            if( dataindex == -1 ) {
                jdf_fatal(JDF_OBJECT_LINENO(flow),
                          "During code generation: unable to find an input flow for variable %s in function %s,\n"
                          "which is requested by function %s to satisfy Output dependency at line %d\n",
                          call->var, call->func_or_mem,
                          fname, JDF_OBJECT_LINENO(flow));
                exit(1);
            } else {
                jdf_fatal(JDF_OBJECT_LINENO(flow),
                          "During code generation: unable to find function %s,\n"
                          "which is requested by function %s to satisfy Output dependency at line %d\n",
                          call->func_or_mem,
                          fname, JDF_OBJECT_LINENO(flow));
                exit(1);
            }
        }
    }

    sa  = string_arena_new(64);

    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "    &this_task->locals";

    sa_arena  = string_arena_new(64);
    sa_datatype = string_arena_new(64);
    sa_count = string_arena_new(64);
    /* This is the case of a pure output data. There's no reshaping. New datacopy should be directly
     * allocated using the correct arena/datatype.
     */
    jdf_datatransfer_type_t dtt = dl->datatype_local;

    jdf_generate_arena_string_from_datatype(sa_arena, dtt);
    if( NULL == dtt.layout ){ /* User didn't specify a custom layout*/
        string_arena_add_string(sa_datatype, "%s->opaque_dtt", string_arena_get_string(sa_arena));
    }else{
        string_arena_add_string(sa_datatype, "%s", dump_expr((void**)dtt.layout, &info));
    }
    assert( dtt.count != NULL );
    string_arena_add_string(sa_count, "%s", dump_expr((void**)dtt.count, &info));

    coutput("%s  /* Flow %s [%d] dependency [%d] pure output data  */\n",
             spaces, flow->varname, flow->flow_index, dl->dep_index);

    coutput("%s  if( NULL == (chunk = this_task->data._f_%s.data_in) ) {\n"
            "%s     /* No data set up by predecessor */\n",
             spaces, flow->varname,
             spaces);

    coutput("%s    chunk = parsec_arena_get_copy(%s->arena, %s, target_device, %s);\n"
            "%s    chunk->original->owner_device = target_device;\n",
            spaces, string_arena_get_string(sa_arena), string_arena_get_string(sa_count), string_arena_get_string(sa_datatype),
            spaces);
    coutput("%s    this_task->data._f_%s.data_out = chunk;\n"
            "%s  }\n",
            spaces, flow->varname,
            spaces);

#if defined(PARSEC_DEBUG_NOISIER) || defined(PARSEC_DEBUG_PARANOID)
    coutput("%s  else {\n"
            "%s     /* Data set up by predecessor */\n"
            "%s     parsec_fatal(\"Data shouldn't be set up by predecessor on an pure output flow\");\n"
            "%s  }\n",
            spaces,
            spaces,
            spaces,
            spaces);
#endif

    string_arena_free(sa);
    string_arena_free(sa_arena);
    string_arena_free(sa_datatype);
    string_arena_free(sa_count);
}

static void jdf_generate_code_flow_initialization(const jdf_t *jdf,
                                                  const jdf_function_entry_t* f,
                                                  const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa, *sa2 = NULL;
    int cond_index = 0, has_output_deps = 0;
    char* condition[] = {"    if( %s ) {\n", "    else if( %s ) {\n"};

    if( JDF_FLOW_TYPE_CTL & flow->flow_flags ) {
        coutput("  /* %s is a control flow */\n"
                "  this_task->data._f_%s.data_in         = NULL;\n"
                "  this_task->data._f_%s.data_out        = NULL;\n"
                "  this_task->data._f_%s.source_repo       = NULL;\n"
                "  this_task->data._f_%s.source_repo_entry = NULL;\n",
                flow->varname,
                flow->varname,
                flow->varname,
                flow->varname,
                flow->varname);
        return;
    }

    coutput("\n"
            "if(! this_task->data._f_%s.fulfill ){ /* Flow %s */\n", flow->varname, flow->varname);

    coutput("  consumed_repo = NULL;\n"
            "  consumed_entry_key = 0;\n"
            "  consumed_entry = NULL;\n"
            "  chunk = NULL;\n");
    
    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if ( dl->dep_flags & JDF_DEP_FLOW_OUT ) {
            has_output_deps = 1;
            break;
        }
    }
    if( !has_output_deps ) {
        coutput("\n    this_task->data._f_%s.data_out = NULL;  /* input only */\n", flow->varname);
    } else {
        coutput("\n    this_task->data._f_%s.data_out = NULL;  /* By default, if nothing matches */\n", flow->varname);
    }
    
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
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f, flow, dl,
                                                       (0 != cond_index ? "  " : "") );
                if( 0 != cond_index ) coutput("    }\n");
                goto done_with_input;
            case JDF_GUARD_BINARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f, flow, dl, "  " );
                coutput("    }\n");
                cond_index++;
                break;
            case JDF_GUARD_TERNARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, f, flow, dl, "  " );
                coutput("    } else {\n");
                jdf_generate_code_call_initialization( jdf, dl->guard->callfalse, f, flow, dl, "  " );
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
        for(dl = flow->deps; dl != NULL; dl = dl->next) {
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

            /* This is the case of a pure output data.
             */
            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                if( 0 != cond_index ) coutput("    else {\n");
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, flow, f->fname, dl, "  ");
                if( 0 != cond_index ) coutput("    }\n");
                goto done_with_input;
            case JDF_GUARD_BINARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, flow, f->fname, dl, "  ");
                coutput("    }\n");
                cond_index++;
                break;
            case JDF_GUARD_TERNARY:
                coutput( (0 == cond_index ? condition[0] : condition[1]),
                         dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_init_output(jdf, dl->guard->calltrue, flow, f->fname, dl, "  ");
                coutput("    } else {\n");
                jdf_generate_code_call_init_output(jdf, dl->guard->callfalse, flow, f->fname, dl, "  ");
                coutput("    }\n");
                goto done_with_input;
            }
        }
    }
 done_with_input:
     coutput("    this_task->data._f_%s.data_in     = chunk;\n"
             "    this_task->data._f_%s.source_repo       = consumed_repo;\n"
             "    this_task->data._f_%s.source_repo_entry = consumed_entry;\n",
             flow->varname,
             flow->varname,
             flow->varname);
     if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) ) {
        /* If we have consume from our repo, we clean up the repo entry for that flow,
         * so we don't have old stuff during release_deps_of when we will set up outputs
         * on our repo as a predecessor.
         */
        coutput("    if( this_task->data._f_%s.source_repo_entry == this_task->repo_entry ){\n"
                "      /* in case we have consume from this task repo entry for the flow,\n"
                "       * it is cleaned up, avoiding having old stuff during release_deps_of\n"
                "       */\n"
                "      this_task->repo_entry->data[%d] = NULL;\n"
                "    }\n",
                flow->varname,
                flow->flow_index);
    }
    coutput("    this_task->data._f_%s.fulfill = 1;\n"
            "}\n\n",
            flow->varname);/*    coutput("if(! this_task->data._f_%s.fulfill ){\n", flow->varname);*/

    
    string_arena_free(sa);
    if( NULL != sa2 )
        string_arena_free(sa2);
}

static void jdf_generate_code_call_final_write(const jdf_t *jdf,
                                               const jdf_function_entry_t *f,
                                               const jdf_call_t *call,
                                               jdf_datatransfer_type_t datatype_local,
                                               jdf_datatransfer_type_t datatype_remote,
                                               jdf_datatransfer_type_t datatype_data,
                                               const char *spaces,
                                               const jdf_dataflow_t *flow)
{
    string_arena_t *sa, *sa2;
    string_arena_t *sa_arena, *sa_tmp_type_src, *sa_tmp_type_dst;
    string_arena_t *sa_type_src, *sa_displ_src, *sa_count_src;
    string_arena_t *sa_type_dst, *sa_displ_dst, *sa_count_dst;

    expr_info_t info = EMPTY_EXPR_INFO;

    (void)jdf;
    (void)datatype_local;
    (void)datatype_remote;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    sa_tmp_type_src = string_arena_new(64);
    sa_tmp_type_dst = string_arena_new(64);
    sa_arena = string_arena_new(64);
    sa_type_src = string_arena_new(64);
    sa_displ_src = string_arena_new(64);
    sa_count_src = string_arena_new(64);
    sa_type_dst = string_arena_new(64);
    sa_displ_dst = string_arena_new(64);
    sa_count_dst = string_arena_new(64);

    if( call->var == NULL ) {
        info.sa = sa2;
        info.prefix = "";
        info.suffix = "";
        info.assignments = "&this_task->locals";

        UTIL_DUMP_LIST(sa, call->parameters, next,
                       dump_expr, (void*)&info, "", "", ", ", "");

        string_arena_init(sa_arena);
        string_arena_init(sa_tmp_type_src);
        string_arena_init(sa_tmp_type_dst);
        string_arena_init(sa_type_src);
        string_arena_init(sa_displ_src);
        string_arena_init(sa_count_src);
        string_arena_init(sa_type_dst);
        string_arena_init(sa_displ_dst);
        string_arena_init(sa_count_dst);

        /* Writing to matrix: allowing inline reshaping:
         *
         *   (1) If !undef(type) && !undef(type_data)
         *           Pack A type, Unpack type_data on descA(m,n)
         *
         *   (2) If undef(type) && !undef(type_data)
         *           Pack A A.type, Unpack type_data on descA(m,n)
         *
         *   (3) If !undef(type) && undef(type_data)
         *           Pack A type, Unpack descA(m,n).type on descA(m,n)
         *
         *   (4) If undef(type) && undef(type_data)
         *           Pack A A.type, Unpack descA(m,n).type on descA(m,n)
         */
        assert( (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(datatype_remote)) );

        jdf_datatransfer_type_t reshape_dtt_dst, reshape_dtt_src;

        reshape_dtt_src = datatype_local;
        reshape_dtt_dst = datatype_data;

        string_arena_add_string(sa_arena, "NULL"); /* We don't care about arena when writing back */

        /* Selecting packing type */
        if( DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_src) ){/* undefined type_data on dependency */
            string_arena_add_string(sa_displ_src, "0");
            string_arena_add_string(sa_count_src, "1");
            string_arena_add_string(sa_type_src, "data.data->dtt");
        }else{
            string_arena_add_string(sa_displ_src, "%s", dump_expr((void**)reshape_dtt_src.displ, &info));
            string_arena_add_string(sa_count_src, "%s", dump_expr((void**)reshape_dtt_src.count, &info));
            jdf_generate_arena_string_from_datatype(sa_tmp_type_src, reshape_dtt_src);
            assert( reshape_dtt_src.count != NULL );

            if( NULL == reshape_dtt_src.layout ){ /* User didn't specify a custom layout*/
                string_arena_add_string(sa_type_src, "%s->opaque_dtt", string_arena_get_string(sa_tmp_type_src));
            }else{
                string_arena_add_string(sa_type_src, "%s", dump_expr((void**)reshape_dtt_src.layout, &info));
            }
        }

        /* Selecting unpacking type */
        if( DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt_dst) ){/* undefined type_data on dependency */
            string_arena_add_string(sa_displ_dst, "0");
            string_arena_add_string(sa_count_dst, "1");
            string_arena_add_string(sa_type_dst, "parsec_data_get_copy(data_t_desc, 0)->dtt");
        }else{

            string_arena_add_string(sa_displ_dst, "%s", dump_expr((void**)reshape_dtt_dst.displ, &info));
            string_arena_add_string(sa_count_dst, "%s", dump_expr((void**)reshape_dtt_dst.count, &info));
            jdf_generate_arena_string_from_datatype(sa_tmp_type_dst, reshape_dtt_dst);
            assert( reshape_dtt_dst.count != NULL );

            if( NULL == reshape_dtt_dst.layout ){ /* User didn't specify a custom layout*/
                string_arena_add_string(sa_type_dst, "%s->opaque_dtt", string_arena_get_string(sa_tmp_type_dst));
            }else{
                string_arena_add_string(sa_type_dst, "%s", dump_expr((void**)reshape_dtt_dst.layout, &info));
            }
        }

        coutput("%s  data_t_desc = data_of_%s(%s);\n"
                "%s  if( (NULL != this_task->data._f_%s.data_out) && (this_task->data._f_%s.data_out->original != data_t_desc) ) {\n"
                "%s    /* Writting back using remote_type */;\n"
                "%s    parsec_dep_data_description_t data;\n"
                "%s    data.data         = this_task->data._f_%s.data_out;\n"
                "%s    data.local.arena        = %s;\n"
                "%s    data.local.src_datatype = %s;\n"
                "%s    data.local.src_count    = %s;\n"
                "%s    data.local.src_displ    = %s;\n"
                "%s    data.local.dst_datatype = %s;\n"
                "%s    data.local.dst_count    = %s;\n"
                "%s    data.local.dst_displ    = %s;\n"
                "%s    data.data_future  = NULL;\n"
                "%s    assert( data.local.src_count > 0 );\n"
                "%s    parsec_remote_dep_memcpy(es,\n"
                "%s                            this_task->taskpool,\n"
                "%s                            parsec_data_get_copy(data_of_%s(%s), 0),\n"
                "%s                            this_task->data._f_%s.data_out, &data);\n"
                "%s  }\n",
                spaces, call->func_or_mem, string_arena_get_string(sa),
                spaces, flow->varname, flow->varname,
                spaces,
                spaces,
                spaces, flow->varname,
                spaces, string_arena_get_string(sa_arena),
                spaces, string_arena_get_string(sa_type_src),
                spaces, string_arena_get_string(sa_count_src),
                spaces, string_arena_get_string(sa_displ_src),
                spaces, string_arena_get_string(sa_type_dst),
                spaces, string_arena_get_string(sa_count_dst),
                spaces, string_arena_get_string(sa_displ_dst),
                spaces,
                spaces,
                spaces,
                spaces,
                spaces, call->func_or_mem, string_arena_get_string(sa),
                spaces, flow->varname,
                spaces);

        coutput("#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)\n"
                "%s  parsec_prof_grapher_data_output((parsec_task_t*)this_task, data_of_%s(%s), &flow_of_%s_%s_for_%s);\n"
                "#endif\n",
                spaces, call->func_or_mem, string_arena_get_string(sa), jdf_basename, f->fname, flow->varname);
    }

    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa_arena);
    string_arena_free(sa_tmp_type_src);
    string_arena_free(sa_tmp_type_dst);
    string_arena_free(sa_type_src);
    string_arena_free(sa_displ_src);
    string_arena_free(sa_count_src);
    string_arena_free(sa_type_dst);
    string_arena_free(sa_displ_dst);
    string_arena_free(sa_count_dst);
}

static void
jdf_generate_code_flow_final_writes(const jdf_t *jdf,
                                    const jdf_function_entry_t* f,
                                    const jdf_dataflow_t *flow)
{
    jdf_dep_t *dl;
    expr_info_t info = EMPTY_EXPR_INFO;
    string_arena_t *sa;

    (void)f;
    sa = string_arena_new(64);
    info.sa = sa;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( dl->dep_flags & JDF_DEP_FLOW_IN )
            /** No final write for input-only flows */
            continue;

        switch( dl->guard->guard_type ) {
        case JDF_GUARD_UNCONDITIONAL:
            if( dl->guard->calltrue->var == NULL ) {
                jdf_generate_code_call_final_write( jdf, f, dl->guard->calltrue, dl->datatype_local, dl->datatype_remote, dl->datatype_data, "", flow );
            }
            break;
        case JDF_GUARD_BINARY:
            if( dl->guard->calltrue->var == NULL ) {
                coutput("  if( %s ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, f, dl->guard->calltrue, dl->datatype_local, dl->datatype_remote, dl->datatype_data, "  ", flow );
                coutput("  }\n");
            }
            break;
        case JDF_GUARD_TERNARY:
            if( dl->guard->calltrue->var == NULL ) {
                coutput("  if( %s ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, f, dl->guard->calltrue, dl->datatype_local, dl->datatype_remote, dl->datatype_data, "  ", flow );
                if( dl->guard->callfalse->var == NULL ) {
                    coutput("  } else {\n");
                    jdf_generate_code_call_final_write( jdf, f, dl->guard->callfalse, dl->datatype_local, dl->datatype_remote, dl->datatype_data, "  ", flow );
                }
                coutput("  }\n");
            } else if ( dl->guard->callfalse->var == NULL ) {
                coutput("  if( !(%s) ) {\n",
                        dump_expr((void**)dl->guard->guard, &info));
                jdf_generate_code_call_final_write( jdf, f, dl->guard->callfalse, dl->datatype_local, dl->datatype_remote, dl->datatype_data, "  ", flow );
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

    coutput("\n\n#if !defined(PARSEC_PROF_DRY_BODY)\n\n");
}

static void jdf_generate_code_dry_run_after(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    (void)jdf;
    (void)f;

    coutput("\n\n#endif /*!defined(PARSEC_PROF_DRY_BODY)*/\n\n");
}

static void jdf_generate_code_grapher_task_done(const jdf_t *jdf, const jdf_function_entry_t *f, const char* context_name)
{
    (void)jdf;

    coutput("#if defined(PARSEC_PROF_GRAPHER)\n"
            "  parsec_prof_grapher_task((parsec_task_t*)%s, es->th_id, es->virtual_process->vp_id,\n"
            "     %s.key_hash(%s->task_class->make_key( (parsec_taskpool_t*)%s->taskpool, ((parsec_task_t*)%s)->locals), NULL));\n"
            "#endif  /* defined(PARSEC_PROF_GRAPHER) */\n",
            context_name,
            jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL), context_name, context_name, context_name);
}

static void jdf_generate_code_cache_awareness_update(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    string_arena_t *sa;
    sa = string_arena_new(64);

    (void)jdf;
    UTIL_DUMP_LIST(sa, f->dataflow, next,
                   dump_dataflow_varname, NULL,
                   "", "  cache_buf_referenced(es->closest_cache, ", ");\n", "");
    if( strlen(string_arena_get_string(sa)) ) {
            coutput("  /** Cache Awareness Accounting */\n"
                    "#if defined(PARSEC_CACHE_AWARENESS)\n"
                    "%s);\n"
                    "#endif /* PARSEC_CACHE_AWARENESS */\n",
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
    coutput("  release_deps_of_%s_%s(es, %s,\n"
            "      PARSEC_ACTION_RELEASE_REMOTE_DEPS |\n"
            "      PARSEC_ACTION_RELEASE_LOCAL_DEPS |\n"
            "      PARSEC_ACTION_RELEASE_LOCAL_REFS |\n"
            "      PARSEC_ACTION_RESHAPE_ON_RELEASE |\n"
            "      0x%x,  /* mask of all dep_index */ \n"
            "      NULL);\n",
            jdf_basename, function->fname, context_name, complete_mask);
}

/**
 * Dump the code corresponding to a set of conditions. The usage of this macro
 * allows us to delay the code generation in order to merge together multiple deps
 * using the same datatype, count and displacement.
 */
#define JDF_CODE_DATATYPE_DUMP(SA_WHERE, MASK, SA_COND, SA_DATATYPE, SKIP_COND)    \
    do {                                                                \
        if( strlen(string_arena_get_string((SA_DATATYPE))) ) {          \
            string_arena_add_string((SA_WHERE),                         \
                                    "    if( ((*flow_mask) & 0x%xU)",   \
                                    (MASK));                            \
            if( strlen(string_arena_get_string((SA_COND))) ) {          \
                if( !(SKIP_COND) ) {                                    \
                    string_arena_add_string((SA_WHERE),                 \
                                            "\n && (%s)",               \
                                            string_arena_get_string((SA_COND))); \
                }                                                       \
            }                                                           \
            string_arena_add_string((SA_WHERE),                         \
                                    " ) {%s\n",                         \
                                    ((SKIP_COND) ? "  /* Have unconditional! */" : "")); \
            (SKIP_COND) = 0;                                            \
            string_arena_add_string((SA_WHERE),                         \
                                    "%s"                                \
                                    "      (*flow_mask) &= ~0x%xU;\n"   \
                                    "      return PARSEC_HOOK_RETURN_NEXT;\n", \
                                    string_arena_get_string((SA_DATATYPE)), \
                                    (MASK));                            \
            string_arena_add_string((SA_WHERE), "    }\n");             \
            if( strlen(string_arena_get_string((SA_COND))) ) {          \
                string_arena_init((SA_COND));                           \
            }                                                           \
            (MASK) = 0;                                                 \
            string_arena_init((SA_DATATYPE));                           \
        }                                                               \
    } while(0)


static void
jdf_generate_code_datatype_lookup(const jdf_t *jdf,
                                  const jdf_function_entry_t *f,
                                  const char *name)
{
    string_arena_t *sa, *sa2;
    assignment_info_t ai;
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;
    string_arena_t *sa_coutput    = string_arena_new(1024);
    string_arena_t *sa_deps       = string_arena_new(1024);
    string_arena_t *sa_datatype   = string_arena_new(1024);
    string_arena_t *sa_arena      = string_arena_new(256);
    string_arena_t *sa_tmp_arena  = string_arena_new(256);
    string_arena_t *sa_count      = string_arena_new(256);
    string_arena_t *sa_tmp_count  = string_arena_new(256);
    string_arena_t *sa_displ      = string_arena_new(256);
    string_arena_t *sa_tmp_displ  = string_arena_new(256);
    string_arena_t *sa_type       = string_arena_new(256);
    string_arena_t *sa_tmp_type   = string_arena_new(256);
    string_arena_t *sa_cond       = string_arena_new(256);
    string_arena_t *sa_temp       = string_arena_new(256);

    int last_datatype_idx, continue_dependencies, type, skip_condition, generate_exit_label = 0;
    uint32_t current_mask = 0;
    expr_info_t info = EMPTY_EXPR_INFO;

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("static int %s(parsec_execution_stream_t *es, const %s *this_task,\n"
            "              uint32_t* flow_mask, parsec_dep_data_description_t* data)\n"
            "{\n"
            "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  (void)__parsec_tp; (void)es; (void)this_task; (void)data;\n"
            "%s",
            name, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    coutput("  data->local.arena = data->remote.arena = NULL;\n"
            "  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;\n"
            "  data->local.src_count     = data->local.dst_count = 0;\n"
            "  data->local.src_displ     = data->local.dst_displ = 0;\n"
            "  data->data_future  = NULL;\n");
    /* First time we generate the IN flows and then we do a second pass and generate the rest of the flows */
    type = JDF_DEP_FLOW_IN;

 redo:  /* we come back here to iterate over the output flows (depending on the variable type) */
    string_arena_init(sa_coutput);
    for( fl = f->dataflow; fl != NULL; fl = fl->next ) {
        if( JDF_FLOW_TYPE_CTL & fl->flow_flags ) continue;

        if( type == JDF_DEP_FLOW_IN ) {
            if( !(fl->flow_flags & JDF_FLOW_IS_IN) ) continue;
        } else {
            if( !(fl->flow_flags & JDF_FLOW_IS_OUT) ) continue;
        }
        skip_condition = 0;  /* Assume we have a valid not-yet-optimized condition */
        string_arena_add_string(sa_coutput, "if( (*flow_mask) & 0x%xU ) {  /* Flow %s */\n",
                                (type == JDF_DEP_FLOW_OUT ? fl->flow_dep_mask_out : (1U << fl->flow_index)), fl->varname);

        last_datatype_idx = -1;
        continue_dependencies = 1;
        string_arena_init(sa_deps);
        string_arena_init(sa_datatype);
        string_arena_init(sa_arena);
        string_arena_init(sa_count);
        string_arena_init(sa_displ);
        string_arena_init(sa_type);
        string_arena_init(sa_cond);

        for(dl = fl->deps; NULL != dl; dl = dl->next) {
            if( !(dl->dep_flags & type) ) continue;

            /* Prepare the memory layout of the output dependency. */
            if( last_datatype_idx != dl->dep_datatype_index ) {
                JDF_CODE_DATATYPE_DUMP(sa_coutput, current_mask, sa_cond, sa_datatype, skip_condition);
                /************************************/
                /* REMOTE DATATYPE USED FOR RECV    */
                /************************************/
                string_arena_init(sa_tmp_arena);
                string_arena_init(sa_tmp_type);
                string_arena_init(sa_temp);
                jdf_generate_arena_string_from_datatype(sa_temp, dl->datatype_remote);
                string_arena_add_string(sa_tmp_arena, " %s->arena ", string_arena_get_string(sa_temp));

                if( NULL == dl->datatype_remote.layout ) { /* no specific layout */
                    string_arena_add_string(sa_tmp_type, "%s->opaque_dtt", string_arena_get_string(sa_temp));
                } else {
                    string_arena_add_string(sa_tmp_type, "%s", dump_expr((void**)dl->datatype_remote.layout, &info));
                }
                string_arena_init(sa_tmp_count);
                string_arena_add_string(sa_tmp_count, "%s", dump_expr((void**)dl->datatype_remote.count, &info));
                string_arena_init(sa_tmp_displ);
                string_arena_add_string(sa_tmp_displ, "%s", dump_expr((void**)dl->datatype_remote.displ, &info));

                jdf_generate_code_fillup_datatypes(sa_tmp_arena,    sa_arena,
                                                   sa_tmp_type,     sa_type,
                                                   sa_tmp_displ,    sa_displ,
                                                   sa_tmp_count,    sa_count,
                                                   NULL,
                                                   NULL,
                                                   NULL,
                                                   sa_datatype,
                                                   "->", "remote", 1);

                last_datatype_idx = dl->dep_datatype_index;
            }

            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                skip_condition = 1;
                /* fallthrough */
                /* No break; process case for JDF_GUARD_TERNARY */
            case JDF_GUARD_TERNARY:
                if( type == JDF_DEP_FLOW_IN ) {
                    skip_condition = 1;  /* we are forced to always match, especially as we do not
                                          * support encapsulation in the tertiaries */
                    continue_dependencies = 0;
                }
                break;
            case JDF_GUARD_BINARY:
                if( strlen(string_arena_get_string(sa_cond)) )
                    string_arena_add_string(sa_cond, " || ");
                string_arena_add_string(sa_cond,
                                        "%s",
                                        dump_expr((void**)dl->guard->guard, &info));
                break;
            }

            /* update the mask before the next dump */
            current_mask |= (type == JDF_DEP_FLOW_OUT ? (1U << dl->dep_index) : (1U << fl->flow_index));

            if( !continue_dependencies ) break;
        }
        JDF_CODE_DATATYPE_DUMP(sa_coutput, current_mask, sa_cond, sa_datatype, skip_condition);

        string_arena_add_string(sa_coutput, "}  /* (flow_mask & 0x%xU) */\n",
                                (type == JDF_DEP_FLOW_OUT ? fl->flow_dep_mask_out : (1U << fl->flow_index)));
    }

    if( type == JDF_DEP_FLOW_IN ) {
        if( 0 < strlen(string_arena_get_string(sa_coutput)) ) {
            coutput("  if( (*flow_mask) & 0x80000000U ) { /* these are the input dependencies remote datatypes  */\n"
                    "%s"
                    "    goto no_mask_match;\n"
                    "  }\n",
                    string_arena_get_string(sa_coutput));
            generate_exit_label = 1;
        }
        type = JDF_DEP_FLOW_OUT;
        goto redo;  /* generate the code for the output dependencies */
    } else {
        coutput("\n  /* these are the output dependencies remote datatypes */\n%s", string_arena_get_string(sa_coutput));
    }
    if( generate_exit_label )
        coutput(" no_mask_match:\n");

    coutput("  data->data                             = NULL;\n"
            "  data->local.arena = data->remote.arena = NULL;\n"
            "  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;\n"
            "  data->remote.src_datatype = data->remote.dst_datatype = PARSEC_DATATYPE_NULL;\n"
            "  data->local.src_count     = data->local.dst_count = 0;\n"
            "  data->remote.src_count    = data->remote.dst_count = 0;\n"
            "  data->local.src_displ     = data->local.dst_displ = 0;\n"
            "  data->remote.src_displ    = data->remote.dst_displ = 0;\n"
            "  data->data_future  = NULL;\n"
            "  (*flow_mask) = 0;  /* nothing left */\n"
            "%s"
            "  return PARSEC_HOOK_RETURN_DONE;\n"
            "}\n",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));
    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa_displ);
    string_arena_free(sa_coutput);
    string_arena_free(sa_datatype);
    string_arena_free(sa_cond);
    string_arena_free(sa_type);
    string_arena_free(sa_deps);
    string_arena_free(sa_arena);
    string_arena_free(sa_tmp_arena);
    string_arena_free(sa_count);
    string_arena_free(sa_tmp_count);
    string_arena_free(sa_tmp_displ);
    string_arena_free(sa_tmp_type);
    string_arena_free(sa_temp);
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
    coutput("static int %s(parsec_execution_stream_t *es, %s *this_task)\n"
            "{\n"
            "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  parsec_assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */\n"
            "  int target_device = 0; (void)target_device;\n"
            "  (void)__parsec_tp; (void)generic_locals; (void)es;\n"
            "  parsec_data_copy_t *chunk = NULL;\n"
            "  data_repo_t        *reshape_repo  = NULL, *consumed_repo  = NULL;\n"
            "  data_repo_entry_t  *reshape_entry = NULL, *consumed_entry = NULL;\n"
            "  parsec_key_t        reshape_entry_key = 0, consumed_entry_key = 0;\n"
            "  uint8_t             consumed_flow_index;\n"
            "  parsec_dep_data_description_t data;\n"
            "  int ret;\n"
            "  (void)reshape_repo; (void)reshape_entry; (void)reshape_entry_key;\n"
            "  (void)consumed_repo; (void)consumed_entry; (void)consumed_entry_key;\n"
            "  (void)consumed_flow_index;\n"
            "  (void)chunk; (void)data; (void)ret;\n"
            "%s",
            name, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    UTIL_DUMP_LIST(sa, f->dataflow, next,
                   dump_data_declaration, sa2, "", "", "", "");

    /* We need to set up +1 as usage limit because we are creating it during datalookup *ONLY* when it hasn't been
     * advanced by the predecessor.
     * When we do data_repo_lookup_entry_and_create the repo is retained until a usage limit set.
     * We need to have it retained during release_deps_of.
     * If we don't increase the usage limit always during datalookup, in one execution path on release_deps it will
     * be retained once and in the other twice.
     * This way, it's only retained once during release_deps.
     */
    coutput("  if( NULL == this_task->repo_entry ){\n"
            "    this_task->repo_entry = data_repo_lookup_entry_and_create(es, %s_repo, "
            "                                      %s((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));\n"
            "    data_repo_entry_addto_usage_limit(%s_repo, this_task->repo_entry->ht_item.key, 1);"
            "    this_task->repo_entry ->generator = (void*)this_task;  /* for AYU */\n"
            "#if defined(PARSEC_SIM)\n"
            "    assert(this_task->repo_entry ->sim_exec_date == 0);\n"
            "    this_task->repo_entry ->sim_exec_date = this_task->sim_exec_date;\n"
            "#endif\n"
            "  }\n",
            f->fname,
            jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL),
            f->fname);

    coutput("  /* The reshape repo is the current task repo. */"
            "  reshape_repo = %s_repo;\n"
            "  reshape_entry_key = %s((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals) ;\n"
            "  reshape_entry = this_task->repo_entry;\n",
            f->fname,
            jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL));

    coutput("  /* Lookup the input data, and store them in the context if any */\n");
    for( fl = f->dataflow; fl != NULL; fl = fl->next ) {
        jdf_generate_code_flow_initialization(jdf, f, fl);
    }

    /* If the function has the property profile turned off do not generate the profiling code */
    if( profile_enabled(f->properties) ) {
        string_arena_t *sa3 = string_arena_new(64);
        expr_info_t linfo = EMPTY_EXPR_INFO;

        linfo.prefix = "";
        linfo.suffix = "";
        linfo.sa = sa2;
        linfo.assignments = "&this_task->locals";

        coutput("  /** Generate profiling information */\n"
                "#if defined(PARSEC_PROF_TRACE)\n"
                "  this_task->prof_info.desc = (parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s;\n"
                "  this_task->prof_info.priority = this_task->priority;\n"
                "  this_task->prof_info.data_id   = ((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->data_key((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s, %s);\n"
                "  this_task->prof_info.task_class_id = this_task->task_class->task_class_id;\n"
                "  this_task->prof_info.task_return_code = -1;\n"
                "#endif  /* defined(PARSEC_PROF_TRACE) */\n",
                f->predicate->func_or_mem,
                f->predicate->func_or_mem, f->predicate->func_or_mem,
                UTIL_DUMP_LIST(sa3, f->predicate->parameters, next,
                               dump_expr, (void*)&linfo,
                               "", "", ", ", "") );
        string_arena_free(sa3);
    } else {
        coutput("  /** No profiling information */\n");
    }
    coutput("  return PARSEC_HOOK_RETURN_DONE;\n"
            "}\n\n");
    string_arena_free(sa);
    string_arena_free(sa2);
}

static int jdf_has_cuda_chore(const jdf_t *jdf, const char *fname)
{
    jdf_function_entry_t *f;
    jdf_body_t* body;
    jdf_def_list_t *type_property;

    for(f = jdf->functions; f != NULL; f = f->next) {
        if( strcmp(f->fname, fname) ) continue;
        for(body = f->bodies; body != NULL; body = body->next) {
            jdf_find_property(body->properties, "type", &type_property);
            if( NULL != type_property && !strcmp(type_property->expr->jdf_var, "CUDA"))
                return 1;
        }
        return 0;
    }
    return 0;
}

static void jdf_generate_code_hook_cuda(const jdf_t *jdf,
                                        const jdf_function_entry_t *f,
                                        const jdf_body_t* body,
                                        const char *name)
{
    jdf_def_list_t *type_property;
    jdf_def_list_t *stage_in_property;
    jdf_def_list_t *stage_out_property;
    jdf_def_list_t *size_property;
    jdf_def_list_t *desc_property;
    jdf_def_list_t *weight_property;
    jdf_def_list_t *device_property;
    const char *dyld;
    const char *dyldtype;
    const char *device;
    const char *weight;
    string_arena_t *sa, *sa2, *sa3;
    assignment_info_t ai;
    init_from_data_info_t ai2;
    jdf_dataflow_t *fl;
    expr_info_t info = EMPTY_EXPR_INFO;
    int di;
    int profile_on;
    char* output;

    profile_on = profile_enabled(f->properties) && profile_enabled(body->properties);

    jdf_find_property(body->properties, "type", &type_property);

    /* Get the dynamic function properties */
    dyld = jdf_property_get_string(body->properties, "dyld", NULL);
    dyldtype = jdf_property_get_string(body->properties, "dyldtype", "void*");

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);

    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;

    string_arena_add_string(sa3, "%s",
                            UTIL_DUMP_LIST(sa, f->locals, next,
                                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    string_arena_add_string(sa3, "%s",
                            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    /* Generate the cuda_kernel_submit structure and function */
    coutput("struct parsec_body_cuda_%s_%s_s {\n"
            "  uint8_t      index;\n"
            "  cudaStream_t stream;\n"
            "  %s           dyld_fn;\n"
            "};\n"
            "\n"
            "static int cuda_kernel_submit_%s_%s(parsec_device_gpu_module_t  *gpu_device,\n"
            "                                    parsec_gpu_task_t           *gpu_task,\n"
            "                                    parsec_gpu_exec_stream_t    *gpu_stream )\n"
            "{\n"
            "  %s *this_task = (%s *)gpu_task->ec;\n"
            "  parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;\n"
            "  parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;\n"
            "  __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  struct parsec_body_cuda_%s_%s_s parsec_body = { cuda_device->cuda_index, cuda_stream->cuda_stream, NULL };\n"
            "%s\n"
            "  (void)gpu_device; (void)gpu_stream; (void)__parsec_tp; (void)parsec_body; (void)cuda_device; (void)cuda_stream;\n",
            jdf_basename, f->fname,
            dyldtype,
            jdf_basename, f->fname,
            parsec_get_name(jdf, f, "task_t"), parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            jdf_basename, f->fname,
            string_arena_get_string( sa3 ));

    ai2.sa = sa2;
    ai2.where = "out";
    output = UTIL_DUMP_LIST(sa, f->dataflow, next,
                            dump_data_initialization_from_data_array, &ai2, "", "", "", "");
    if( 0 != strlen(output) ) {
        coutput("  /** Declare the variables that will hold the data, and all the accounting for each */\n"
                "%s\n",
                output);
    }

    /**
     * Generate code for the simulation.
     */
    coutput("  /** Update starting simulation date */\n"
            "#if defined(PARSEC_SIM)\n"
            "  {\n"
            "    this_task->sim_exec_date = 0;\n");
    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {

        if(fl->flow_flags & JDF_FLOW_TYPE_CTL) continue;  /* control flow, nothing to store */

        coutput("    data_repo_entry_t *e%s = this_task->data._f_%s.source_repo_entry;\n"
                "    if( (NULL != e%s) && (e%s->sim_exec_date > this_task->sim_exec_date) )\n"
                "      this_task->sim_exec_date = e%s->sim_exec_date;\n",
                fl->varname, fl->varname,
                fl->varname, fl->varname,
                fl->varname);
    }
    coutput("    if( this_task->task_class->sim_cost_fct != NULL ) {\n"
            "      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);\n"
            "    }\n"
            "    if( es->largest_simulation_date < this_task->sim_exec_date )\n"
            "      es->largest_simulation_date = this_task->sim_exec_date;\n"
            "  }\n"
            "#endif\n");

    jdf_generate_code_cache_awareness_update(jdf, f);

    coutput("#if defined(PARSEC_DEBUG_NOISIER)\n"
            "  {\n"
            "    char tmp[MAX_TASK_STRLEN];\n"
            "    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, \"GPU[%%s]:\\tEnqueue on device %%s priority %%d\", gpu_device->super.name, \n"
            "           parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t *)this_task),\n"
            "           this_task->priority );\n"
            "  }\n"
            "#endif /* defined(PARSEC_DEBUG_NOISIER) */\n" );

    jdf_generate_code_dry_run_before(jdf, f);
    jdf_coutput_prettycomment('-', "%s BODY", f->fname);

    if( profile_on ) {
        coutput("#if defined(PARSEC_PROF_TRACE)\n"
                "  if(gpu_stream->prof_event_track_enable) {\n"
                "    PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,\n"
                "                           PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,\n"
                "                                     this_task->task_class->task_class_id),\n"
                "                           (parsec_task_t*)this_task);\n"
                "    gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool,\n"
                "                                   this_task->task_class->task_class_id);\n"
                "    gpu_task->prof_event_id = this_task->task_class->key_functions->\n"
                "           key_hash(this_task->task_class->make_key(this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL);\n"
                "    gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;\n"
                "  }\n"
                "#endif /* PARSEC_PROF_TRACE */\n");
    }

    dyld = jdf_property_get_string(body->properties, "dyld", NULL);
    dyldtype = jdf_property_get_string(body->properties, "dyldtype", "void*");
    if ( NULL != dyld ) {
        coutput("  /* Pointer to dynamic gpu function */\n"
                "  {\n"
                "    int chore_idx = 0;\n"
                "    for ( ; PARSEC_DEV_NONE != this_task->task_class->incarnations[chore_idx].type; ++chore_idx) {\n"
                "      if (this_task->task_class->incarnations[chore_idx].type == PARSEC_DEV_CUDA) break;\n"
                "    }\n"
                "    /* The void* cast prevents the compiler from complaining about the type change */\n"
                "    parsec_body.dyld_fn = (%s)(void*)this_task->task_class->incarnations[chore_idx].dyld_fn;\n"
                "  }\n\n",
                dyldtype );
    }

    coutput("%s\n", body->external_code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }
    jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname);
    jdf_generate_code_dry_run_after(jdf, f);
    coutput("  return PARSEC_HOOK_RETURN_DONE;\n"
            "}\n\n");

    /* Generate the hook_cuda */
    coutput("static int %s_%s(parsec_execution_stream_t *es, %s *this_task)\n"
            "{\n"
            "  __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  parsec_gpu_task_t *gpu_task;\n"
            "  double ratio;\n"
            "  int dev_index;\n"
            "  %s\n"
            "  (void)es; (void)__parsec_tp;\n"
            "\n",
            name, type_property->expr->jdf_var, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            string_arena_get_string( sa3 ));

    info.sa = string_arena_new(64);
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&this_task->locals";

    /* Get the ratio to  apply on the weight for this task */
    jdf_find_property( body->properties, "weight", &weight_property );
    if ( NULL != weight_property ) {
        weight = dump_expr((void**)weight_property->expr, &info);
    } else {
        weight = "1.";
    }
    coutput("  ratio = %s;\n", weight);

    /* Get the hint for statix and/or external gpu scheduling */
    jdf_find_property( body->properties, "device", &device_property );
    if ( NULL != device_property ) {
        device = dump_expr((void**)device_property->expr, &info);
        coutput("  dev_index = %s;\n"
                "  if (dev_index < -1) {\n"
                "    return PARSEC_HOOK_RETURN_NEXT;\n"
                "  } else if (dev_index == -1) {\n"
                "    dev_index = parsec_get_best_device((parsec_task_t*)this_task, ratio);\n"
                "  } else {\n"
                "    dev_index = (dev_index %% (parsec_mca_device_enabled()-2)) + 2;\n"
                "  }\n",
                device);
    } else {
        coutput("  dev_index = parsec_get_best_device((parsec_task_t*)this_task, ratio);\n");
    }
    coutput("  assert(dev_index >= 0);\n"
            "  if( dev_index < 2 ) {\n"
            "    return PARSEC_HOOK_RETURN_NEXT;  /* Fall back */\n"
            "  }\n"
            "\n"
            "  gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));\n"
            "  PARSEC_OBJ_CONSTRUCT(gpu_task, parsec_list_item_t);\n"
            "  gpu_task->ec = (parsec_task_t*)this_task;\n"
            "  gpu_task->submit = &cuda_kernel_submit_%s_%s;\n"
            "  gpu_task->task_type = 0;\n"
            "  gpu_task->load = ratio * parsec_device_sweight[dev_index];\n"
            "  gpu_task->last_data_check_epoch = -1;  /* force at least one validation for the task */\n",
            jdf_basename, f->fname);

    /* Set up stage in/out callbacks */
    jdf_find_property(body->properties, "stage_in", &stage_in_property);
    jdf_find_property(body->properties, "stage_out", &stage_out_property);

    if(stage_in_property == NULL) {
        coutput("  gpu_task->stage_in  = parsec_default_cuda_stage_in;\n");
    }else{
        coutput("  gpu_task->stage_in  = %s;\n", dump_expr((void**)stage_in_property->expr, &info));
    }

    if(stage_out_property == NULL) {
        coutput("  gpu_task->stage_out = parsec_default_cuda_stage_out;\n");
    }else{
        coutput("  gpu_task->stage_out = %s;\n", dump_expr((void**)stage_out_property->expr, &info));
    }

    /* Dump the dataflow */
    coutput("  gpu_task->pushout = 0;\n");
    for(fl = f->dataflow, di = 0; fl != NULL; fl = fl->next, di++) {
        coutput("  gpu_task->flow[%d]         = &%s;\n",
                di, JDF_OBJECT_ONAME( fl ));

        sprintf(sa->ptr, "%s.dc", fl->varname);
        jdf_find_property(body->properties, sa->ptr, &desc_property);
        if(desc_property == NULL){
            coutput("  gpu_task->flow_dc[%d] = NULL;\n", di);
        }else{
            coutput("  gpu_task->flow_dc[%d] = (parsec_data_collection_t *)%s;\n", di,
                        dump_expr((void**)desc_property->expr, &info));
        }

        sprintf(sa->ptr, "%s.size", fl->varname);
        jdf_find_property(body->properties, sa->ptr, &size_property);

        if(fl->flow_flags & JDF_FLOW_TYPE_CTL) {
            if(size_property != NULL){
                fprintf(stderr, "Error: specifying GPU buffer size for CTL flow %s at line %d\n",
                        fl->varname, JDF_OBJECT_LINENO(fl));
                exit(-1);
            }
            coutput("  gpu_task->flow_nb_elts[%d] = 0;\n", di);
        }else{
            if(size_property == NULL){
                coutput("  gpu_task->flow_nb_elts[%d] = gpu_task->ec->data[%d].data_in->original->nb_elts;\n", di, di);
            }else{
                coutput("  gpu_task->flow_nb_elts[%d] = %s;\n",
                        di, dump_expr((void**)size_property->expr, &info));
                if( (stage_in_property == NULL) || ( stage_out_property == NULL )){
                    coutput("  assert(gpu_task->ec->data[%d].data_in->original->nb_elts <= %s);\n",
                            di, dump_expr((void**)size_property->expr, &info));
                }

            }
        }

        if (fl->flow_flags & JDF_FLOW_TYPE_WRITE) {
            jdf_dep_t *dl;
            int testtrue, testfalse;

            /**
             * We force the pushout for every data that is not only going to the
             * same kind of kernel in the future.
             * (TODO: could be avoided with different GPU compliant kernels)
             */
            for(dl = fl->deps; dl != NULL; dl = dl->next) {
                if( dl->dep_flags & JDF_DEP_FLOW_IN )
                    continue;

                testtrue = (dl->guard->calltrue != NULL) &&
                    ((dl->guard->calltrue->var == NULL ) ||
                     (!jdf_has_cuda_chore(jdf, dl->guard->calltrue->func_or_mem)));

                testfalse = (dl->guard->callfalse != NULL) &&
                    ((dl->guard->callfalse->var == NULL ) ||
                     (!jdf_has_cuda_chore(jdf, dl->guard->callfalse->func_or_mem)));

                switch( dl->guard->guard_type ) {
                case JDF_GUARD_UNCONDITIONAL:
                    if(testtrue) {
                        coutput("  gpu_task->pushout |= (1 << %d);\n", di);
                        goto nextflow;
                    }
                    break;
                case JDF_GUARD_BINARY:
                    if(testtrue) {
                        coutput("  if( %s ) {\n"
                                "    gpu_task->pushout |= (1 << %d);\n"
                                "  }",
                                dump_expr((void**)dl->guard->guard, &info), di);
                    }
                    break;
                case JDF_GUARD_TERNARY:
                    if( testtrue ) {
                        if( testfalse ) {
                            coutput("  gpu_task->pushout |= (1 << %d);\n", di);
                        } else {
                            coutput("  if( %s ) {\n"
                                    "    gpu_task->pushout |= (1 << %d);\n"
                                    "  }\n",
                                    dump_expr((void**)dl->guard->guard, &info), di);
                        }
                    } else if ( testfalse ) {
                        coutput("  if( !(%s) ) {\n"
                                "    gpu_task->pushout |= (1 << %d);\n"
                                "  }\n",
                                dump_expr((void**)dl->guard->guard, &info), di);
                    }
                    break;
                }
            }
          nextflow:
            ;
        }
    }
    string_arena_free(info.sa);

    coutput("  parsec_device_load[dev_index] += gpu_task->load;\n"
            "\n"
            "  return parsec_cuda_kernel_scheduler( es, gpu_task, dev_index );\n"
            "}\n\n");

    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);
}

static void jdf_generate_code_hook(const jdf_t *jdf,
                                   const jdf_function_entry_t *f,
                                   const jdf_body_t* body,
                                   const char *name)
{
    jdf_def_list_t* type_property;
    string_arena_t *sa, *sa2;
    assignment_info_t ai;
    init_from_data_info_t ai2;
    jdf_dataflow_t *fl;
    int di;
    char* output;

    jdf_find_property(body->properties, "type", &type_property);
    if(NULL != type_property) {
        if(JDF_VAR != type_property->expr->op) {
            expr_info_t ei = EMPTY_EXPR_INFO;

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
    if( NULL != type_property) {
        coutput("#if defined(PARSEC_HAVE_%s)\n", type_property->expr->jdf_var);

        if (!strcmp(type_property->expr->jdf_var, "CUDA")) {
            jdf_generate_code_hook_cuda(jdf, f, body, name);
            goto hook_end_block;
        }
    }
    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;

    if(NULL == type_property)
        coutput("static int %s(parsec_execution_stream_t *es, %s *this_task)\n",
                name, parsec_get_name(jdf, f, "task_t"));
    else
        coutput("static int %s_%s(parsec_execution_stream_t *es, %s *this_task)\n",
                name, type_property->expr->jdf_var, parsec_get_name(jdf, f, "task_t"));

    coutput("{\n"
            "  __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  (void)es; (void)__parsec_tp;\n"
            "%s",
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    ai2.sa = sa2;
    ai2.where = "in";
    output = UTIL_DUMP_LIST(sa, f->dataflow, next,
                            dump_data_initialization_from_data_array, &ai2, "", "", "", "");
    if( 0 != strlen(output) ) {
        coutput("  /** Declare the variables that will hold the data, and all the accounting for each */\n"
                "%s\n",
                output);
    }

    /**
     * Generate code for the simulation.
     */
    coutput("  /** Update starting simulation date */\n"
            "#if defined(PARSEC_SIM)\n"
            "  {\n"
            "    this_task->sim_exec_date = 0;\n");
    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {

        if(fl->flow_flags & JDF_FLOW_TYPE_CTL) continue;  /* control flow, nothing to store */

        coutput("    data_repo_entry_t *e%s = this_task->data._f_%s.source_repo_entry;\n"
                "    if( (NULL != e%s) && (e%s->sim_exec_date > this_task->sim_exec_date) )\n"
                "      this_task->sim_exec_date = e%s->sim_exec_date;\n",
                fl->varname, fl->varname,
                fl->varname, fl->varname,
                fl->varname);
    }
    coutput("    if( this_task->task_class->sim_cost_fct != NULL ) {\n"
            "      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);\n"
            "    }\n"
            "    if( es->largest_simulation_date < this_task->sim_exec_date )\n"
            "      es->largest_simulation_date = this_task->sim_exec_date;\n"
            "  }\n"
            "#endif\n");

    if ((NULL == type_property) ||
        (!strcmp(type_property->expr->jdf_var, "RECURSIVE"))) {
        coutput("  /** Transfer the ownership to the CPU */\n"
                "#if defined(PARSEC_HAVE_CUDA)\n");

        for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {
            /* Update the ownership of read/write data */
            /* Applied only on the Write data, since the number of readers is not atomically increased yet */
            if ((fl->flow_flags & JDF_FLOW_TYPE_READ) &&
                (fl->flow_flags & JDF_FLOW_TYPE_WRITE) ) {
               coutput("    if ( NULL != _f_%s ) {\n"
                       "      parsec_data_transfer_ownership_to_copy( _f_%s->original, 0 /* device */,\n"
                       "                                           %s);\n"
                       "    }\n",
                       fl->varname,
                       fl->varname,
                       ((fl->flow_flags & JDF_FLOW_TYPE_CTL) ? "PARSEC_FLOW_ACCESS_NONE" :
                        ((fl->flow_flags & JDF_FLOW_TYPE_READ) ?
                         ((fl->flow_flags & JDF_FLOW_TYPE_WRITE) ? "PARSEC_FLOW_ACCESS_RW" : "PARSEC_FLOW_ACCESS_READ") : "PARSEC_FLOW_ACCESS_WRITE")));
            }
        }
        coutput("#endif  /* defined(PARSEC_HAVE_CUDA) */\n");
    }
    jdf_generate_code_cache_awareness_update(jdf, f);

    jdf_generate_code_dry_run_before(jdf, f);
    jdf_coutput_prettycomment('-', "%s BODY", f->fname);

    coutput("%s\n", body->external_code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }
    jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname);
    jdf_generate_code_dry_run_after(jdf, f);
    coutput("  return PARSEC_HOOK_RETURN_DONE;\n"
            "}\n");

    string_arena_free(sa);
    string_arena_free(sa2);

  hook_end_block:
    if( NULL != type_property)
        coutput("#endif  /*  defined(PARSEC_HAVE_%s) */\n", type_property->expr->jdf_var);
}

static void
jdf_generate_code_complete_hook(const jdf_t *jdf,
                                const jdf_function_entry_t *f,
                                const char *name)
{
    string_arena_t *sa, *sa2;
    int di;
    jdf_dataflow_t *fl;
    assignment_info_t ai;

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.holder = "this_task->locals.";
    ai.expr = NULL;
    coutput("static int complete_%s(parsec_execution_stream_t *es, %s *this_task)\n"
            "{\n"
            "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (__parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "#if defined(DISTRIBUTED)\n"
            "  %s"
            "#endif  /* defined(DISTRIBUTED) */\n"
            "  (void)es; (void)__parsec_tp;\n",
            name, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST(sa, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    coutput("parsec_data_t* data_t_desc = NULL; (void)data_t_desc;\n");

    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {
        if(JDF_FLOW_TYPE_CTL & fl->flow_flags) continue;
        if(fl->flow_flags & JDF_FLOW_TYPE_WRITE) {
            /**
             * The data_out might be NULL if we don't forward anything.
             */
            coutput("  if ( NULL != this_task->data._f_%s.data_out ) {\n"
                    "#if defined(PARSEC_DEBUG_NOISIER)\n"
                    "     char tmp[128];\n"
                    "#endif\n"
                    "     this_task->data._f_%s.data_out->version++;  /* %s */\n"
                    "     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,\n"
                    "                          \"Complete hook of %%s: change Data copy %%p to version %%d at %%s:%%d\",\n"
                    "                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),\n"
                    "                          this_task->data._f_%s.data_out, this_task->data._f_%s.data_out->version, __FILE__, __LINE__);\n"
                    "  }\n",
                    fl->varname,
                    fl->varname, fl->varname,
                    fl->varname, fl->varname );
        }
    }

    /* TODO: The data could be on the GPU */
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

    coutput("  return PARSEC_HOOK_RETURN_DONE;\n"
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

static void jdf_generate_code_free_hash_table_entry(const jdf_t *jdf, const jdf_function_entry_t *f, int consume_repo, int release_inputs)
{
    jdf_dataflow_t *dl;

    coutput("  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {\n");

    for( dl = f->dataflow; dl != NULL; dl = dl->next ) {
        if( dl->flow_flags & JDF_FLOW_TYPE_CTL ) continue;
        if(consume_repo){
            coutput("    if( NULL != this_task->data._f_%s.source_repo_entry ) {\n"
                    "        data_repo_entry_used_once( this_task->data._f_%s.source_repo, this_task->data._f_%s.source_repo_entry->ht_item.key );\n"
                    "    }\n",
                    dl->varname,
                    dl->varname, dl->varname);
        }
        if(release_inputs){
            if( dl->flow_flags & (JDF_FLOW_TYPE_READ | JDF_FLOW_TYPE_WRITE) ) {
                coutput("    if( NULL != this_task->data._f_%s.data_in ) {\n"
                        "        PARSEC_DATA_COPY_RELEASE(this_task->data._f_%s.data_in);\n"
                        "    }\n",
                        dl->varname, dl->varname);
            }
        }
        (void)jdf;  /* just to keep the compilers happy regarding the goto to an empty statement */
    }
    coutput("  }\n");
}

static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    coutput("static int %s(parsec_execution_stream_t *es, %s *this_task, uint32_t action_mask, parsec_remote_deps_t *deps)\n"
            "{\n"
            "PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *)this_task);"
            "{\n"
            "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (const __parsec_%s_internal_taskpool_t *)this_task->taskpool;\n"
            "  parsec_release_dep_fct_arg_t arg;\n"
            "  int __vp_id;\n"
            "  int consume_local_repo = 0;\n"
            "  arg.action_mask = action_mask;\n"
            "  arg.output_entry = NULL;\n"
            "  arg.output_repo = NULL;\n"
            "#if defined(DISTRIBUTED)\n"
            "  arg.remote_deps = deps;\n"
            "#endif  /* defined(DISTRIBUTED) */\n"
            "  assert(NULL != es);\n"
            "  arg.ready_lists = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);\n"
            "  for( __vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );\n"
            "  (void)__parsec_tp; (void)deps;\n",
            name, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename);

    jdf_generate_code_free_hash_table_entry(jdf, f, 1/*consume_repo*/, 0/*release_inputs*/);

    coutput("  consume_local_repo = (this_task->repo_entry != NULL);\n");

    if( !(f->flags & JDF_FUNCTION_FLAG_NO_SUCCESSORS) ) {

        coutput("  arg.output_repo = %s_repo;\n", f->fname);
        coutput("  arg.output_entry = this_task->repo_entry;\n");
        coutput("  arg.output_usage = 0;\n");

        coutput("  if( action_mask & (PARSEC_ACTION_RELEASE_LOCAL_DEPS | PARSEC_ACTION_GET_REPO_ENTRY) ) {\n"
                 "    arg.output_entry = data_repo_lookup_entry_and_create( es, arg.output_repo, %s((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));\n"
                 "    arg.output_entry->generator = (void*)this_task;  /* for AYU */\n"
                 "#if defined(PARSEC_SIM)\n"
                 "    assert(arg.output_entry->sim_exec_date == 0);\n"
                 "    arg.output_entry->sim_exec_date = this_task->sim_exec_date;\n"
                 "#endif\n"
                 "  }\n",
                 jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL));

        /* We need 2 iterate_successors calls so that all reshapping info is
         * setup before it is consumed by any local successor.
         */
        coutput("  if(action_mask & ( PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE ) ){\n");
        coutput("    /* Generate the reshape promise for thet outputs that need it */\n");
        coutput("    iterate_successors_of_%s_%s(es, this_task, action_mask, parsec_set_up_reshape_promise, &arg);\n"
                "   }\n",
                jdf_basename, f->fname);

        coutput("  iterate_successors_of_%s_%s(es, this_task, action_mask, parsec_release_dep_fct, &arg);\n"
                "\n",
                jdf_basename, f->fname);

        coutput("#if defined(DISTRIBUTED)\n"
                "  if( (action_mask & PARSEC_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {\n"
                "    parsec_remote_dep_activate(es, (parsec_task_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);\n"
                "  }\n"
                "#endif\n"
                "\n");
        coutput("  if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {\n"
                "    struct parsec_vp_s** vps = es->virtual_process->parsec_context->virtual_processes;\n");
        coutput("    data_repo_entry_addto_usage_limit(%s_repo, arg.output_entry->ht_item.key, arg.output_usage);\n",
                f->fname);

        coutput("    for(__vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; __vp_id++) {\n"
                "      if( NULL == arg.ready_lists[__vp_id] ) continue;\n"
                "      if(__vp_id == es->virtual_process->vp_id) {\n"
                "        __parsec_schedule(es, arg.ready_lists[__vp_id], 0);\n"
                "      } else {\n"
                "        __parsec_schedule(vps[__vp_id]->execution_streams[0], arg.ready_lists[__vp_id], 0);\n"
                "      }\n"
                "      arg.ready_lists[__vp_id] = NULL;\n"
                "    }\n"
                "  }\n");
    } else {
        coutput("  /* No successors, don't call iterate_successors and don't release any local deps */\n");
    }

    /* We need to consume from the repo because we are creating it during datalookup *ONLY* when it hasn't been
     * advanced by the predecessor.
     * When we do data_repo_lookup_entry_and_create the repo is retained until a usage limit set.
     * We need to have it retained during release_deps_of.
     * If we don't increase the usage limit always during datalookup, in one execution path on release_deps it will
     * be retained once and in the other twice.
     * This way, it's only retained once during release_deps.
     *
     * /!\ We run release deps for "legit" tasks, and for fake remote task that has sent us the data.
     * Only legit have run datalookup & +1 usage limit.
     *
     */
    coutput("      if (consume_local_repo) {\n"
            "         data_repo_entry_used_once( %s_repo, this_task->repo_entry->ht_item.key );\n"
            "      }\n",
            f->fname);

    jdf_generate_code_free_hash_table_entry(jdf, f, 0/*consume_repo*/, 1/*release_inputs*/);

    coutput(
        "PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *)this_task);"
        "}\n"
        "  return 0;\n"
        "}\n"
        "\n");
}

static char *jdf_dump_context_assignment(string_arena_t *sa_open,
                                         const jdf_t *jdf,
                                         const jdf_function_entry_t *sourcef,
                                         const jdf_dataflow_t *flow,
                                         const char *calltext,
                                         const jdf_call_t *call,
                                         const jdf_dep_t *dep,
                                         int lineno,
                                         const char *prefix,
                                         const char *var)
{
    expr_info_t local_info = EMPTY_EXPR_INFO, dest_info = EMPTY_EXPR_INFO;
    int nbparam_given, nbparam_required, i, nbopen;
    const jdf_function_entry_t *targetf;
    string_arena_t *sa2, *sa1, *sa_close;
    jdf_variable_list_t *vl;
    jdf_param_list_t *nl;
    jdf_expr_t *el;

    (void)sourcef;
    
    string_arena_init(sa_open);

    /* Find the target function */
    targetf = find_target_function(jdf, call->func_or_mem);

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
    local_info.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    sa_close = string_arena_new(64);

    nbopen = 0;

    string_arena_add_string(sa_open, "%s%s%s* ncc = (%s*)&%s;\n",
                            prefix, indent(nbopen), parsec_get_name(jdf, targetf, "task_t"), parsec_get_name(jdf, targetf, "task_t"), var);
    string_arena_add_string(sa_open, "%s%s%s.task_class = __parsec_tp->super.super.task_classes_array[%s_%s.task_class_id];\n",
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

    /* Now we generate the local definitions that are defined *only* at the call
     * level, ignoring the ones defined at the dep level, that are generated above */
    if( NULL != call->local_defs ) {
        jdf_expr_t *ld;
        jdf_expr_t *dep_ld;
        dep_ld = jdf_expr_lv_first(dep->local_defs);
        for(ld = jdf_expr_lv_first(call->local_defs); ld != NULL; ld = jdf_expr_lv_next(call->local_defs, ld)) {
            assert(NULL != ld->alias);
            assert(-1 != ld->ldef_index);
            if( NULL != dep_ld ) {
                assert( dep_ld == ld);
                dep_ld = jdf_expr_lv_next(dep->local_defs, dep_ld);
                continue; /* This local define was already issued as part of the dep */
            }
            string_arena_add_string(sa_open, "%s%s  int %s;\n", prefix, indent(nbopen), ld->alias);
            if(JDF_RANGE == ld->op) {
                string_arena_add_string(sa_open,
                                        "%s%sfor( %s = %s;",
                                        prefix, indent(nbopen), ld->alias, dump_expr((void**)ld->jdf_ta1, &local_info));
                string_arena_add_string(sa_open, "%s <= %s; %s+=",
                                        ld->alias, dump_expr((void**)ld->jdf_ta2, &local_info), ld->alias);
                string_arena_add_string(sa_open, "%s) {\n"
                                        "%s%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                                        dump_expr((void**)ld->jdf_ta3, &local_info),
                                        prefix, indent(nbopen), ld->ldef_index, ld->alias);
                nbopen++;
            } else {
                string_arena_add_string(sa_open,
                                        "%s%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s = %s;\n",
                                        prefix, indent(nbopen), ld->ldef_index, ld->alias, dump_expr((void**)ld, &local_info));
            }
        }
    }
    
    for(vl = targetf->locals, i = 0; vl != NULL; vl = vl->next, i++) {

        for(el  = call->parameters, nl = targetf->parameters;
            nl != NULL;
            nl = nl->next, el = el->next) {
            assert(el != NULL);
            if( !strcmp(nl->name, vl->name) )
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
                                    prefix, indent(nbopen), targetf->fname, vl->name, dump_expr((void**)vl->expr, &dest_info));
            string_arena_add_string(sa_open, "%s%s  assert(&%s.locals[%d].value == &ncc->locals.%s.value);\n",
                                    prefix, indent(nbopen), var, i, vl->name);
            string_arena_add_string(sa_open, "%s%s  ncc->locals.%s.value = %s_%s;\n",
                                    prefix, indent(nbopen), vl->name,
                                    targetf->fname, vl->name);
        } else {
            /* This definition is a parameter */
            assert(el != NULL);

            if( NULL != el->local_variables ) {
                /* Now, there can be additional local definitions: the definitions that appear directly
                 * inside the parameters themselves. But of course, we need to skip the local definitions
                 * of the call or dep... */
                jdf_expr_t *dep_ld = NULL, *call_ld = NULL, *ld;
                if(NULL != dep->local_defs)  dep_ld  = jdf_expr_lv_first(dep->local_defs);
                if(NULL != call->local_defs) call_ld = jdf_expr_lv_first(call->local_defs);
                for(ld = jdf_expr_lv_first(el->local_variables); ld != NULL; ld = jdf_expr_lv_next(el->local_variables, ld)) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    if( NULL != dep_ld ) {
                        assert( dep_ld == ld);
                        dep_ld = jdf_expr_lv_next(dep->local_defs, dep_ld);
                        if( NULL != call_ld ) {
                            assert( call_ld == ld );
                            call_ld = jdf_expr_lv_next(call->local_defs, call_ld);
                        }
                        continue; /* This local define was already issued as part of the dep */
                    }
                    if( NULL != call_ld ) {
                        assert( call_ld == ld );
                        call_ld = jdf_expr_lv_next(call->local_defs, call_ld);
                        continue; /* This local define was alredy issued above as part of the call */
                    }
                    string_arena_add_string(sa_open, "%s%s  {\n"
                                            "%s%s    int %s;\n",
                                            prefix, indent(nbopen),
                                            prefix, indent(nbopen), ld->alias);
                    nbopen++;
                    if(JDF_RANGE == ld->op) {
                        string_arena_add_string(sa_open,
                                                "%s%s  for( %s = %s;",
                                                prefix, indent(nbopen), ld->alias, dump_expr((void**)ld->jdf_ta1, &local_info));
                        string_arena_add_string(sa_open, "%s <= %s; %s+=",
                                                ld->alias, dump_expr((void**)ld->jdf_ta2, &local_info), ld->alias);
                        string_arena_add_string(sa_open, "%s) {\n"
                                                "%s%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                                                dump_expr((void**)ld->jdf_ta3, &local_info),
                                                prefix, indent(nbopen), ld->ldef_index, ld->alias);
                        nbopen++;
                    } else {
                        string_arena_add_string(sa_open,
                                                "%s%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s = %s;\n",
                                                prefix, indent(nbopen), ld->ldef_index, ld->alias, dump_expr((void**)ld, &local_info));
                    }
                }
            }
            
            if( JDF_RANGE == el->op ) {
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

            if( vl->expr->op == JDF_RANGE ) {
                /* This is a place where we consider iterators must be from low to high */
                string_arena_add_string(sa_open,
                                        "%s%s  if( (%s_%s >= (%s))",
                                        prefix, indent(nbopen), targetf->fname, nl->name,
                                        dump_expr((void**)vl->expr->jdf_ta1, &dest_info));
                string_arena_add_string(sa_open, " && (%s_%s <= (%s)) ) {\n",
                                        targetf->fname, nl->name,
                                        dump_expr((void**)vl->expr->jdf_ta2, &dest_info));
                nbopen++;
            } else if( NULL != vl->expr->local_variables ) {
                string_arena_add_string(sa_open, "%s%s  /* We cannot check if %s_%s is within the iterator space, because that space is defined with local indices. We need to trust */\n",
                                        prefix, indent(nbopen), targetf->fname, nl->name);
            } else {
                string_arena_add_string(sa_open,
                                        "%s%s  if( (%s_%s == (%s)) ) {\n",
                                        prefix, indent(nbopen), targetf->fname, nl->name,
                                        dump_expr((void**)vl->expr, &dest_info));
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
                            "%s%s  rank_dst = rank_of_%s(%s);\n",
                            prefix, indent(nbopen), targetf->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, targetf->predicate->parameters, next,
                                           dump_expr, (void*)&dest_info,
                                           "", "", ", ", ""));
    string_arena_add_string(sa_open,
                            "%s%s  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )\n"
                            "#endif /* DISTRIBUTED */\n"
                            "%s%s    vpid_dst = ((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s)->vpid_of((parsec_data_collection_t*)"TASKPOOL_GLOBAL_PREFIX"_g_%s, %s);\n",
                            prefix, indent(nbopen),
                            prefix, indent(nbopen), targetf->predicate->func_or_mem, targetf->predicate->func_or_mem,
                            UTIL_DUMP_LIST(sa2, targetf->predicate->parameters, next,
                                           dump_expr, (void*)&dest_info,
                                           "", "", ", ", ""));

    if( NULL != targetf->priority ) {
        string_arena_add_string(sa_open,
                                "%s%s  %s.priority = __parsec_tp->super.super.priority + priority_of_%s_%s_as_expr_fct(__parsec_tp, &ncc->locals);\n",
                                prefix, indent(nbopen), var, jdf_basename, targetf->fname);
    } else {
        string_arena_add_string(sa_open, "%s%s  %s.priority = __parsec_tp->super.super.priority;\n",
                                prefix, indent(nbopen), var);
    }

    if( !(targetf->flags & JDF_FUNCTION_FLAG_NO_PREDECESSORS) ){
        string_arena_add_string(sa_open, "%s%s  successor_repo = %s_repo;\n",
                                prefix, indent(nbopen), targetf->fname);
        string_arena_add_string(sa_open, "%s%s  successor_repo_key = "
                                "%s((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);\n",
                                prefix, indent(nbopen),
                                jdf_property_get_string(targetf->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL));
    }else{
        string_arena_add_string(sa_open, "%s%s  successor_repo = NULL;\n",
                                prefix, indent(nbopen));

        string_arena_add_string(sa_open, "%s%s  successor_repo_key = 0;\n",
                                prefix, indent(nbopen));
    }

    string_arena_add_string(sa_open,
                            "%s%sRELEASE_DEP_OUTPUT(es, \"%s\", this_task, \"%s\", &%s, rank_src, rank_dst, &data);\n",
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
    string_arena_free(sa1);
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
    
    string_arena_t *sa_arena       = string_arena_new(256);
    string_arena_t *sa_tmp_arena   = string_arena_new(256);
    string_arena_t *sa_count      = string_arena_new(256);
    string_arena_t *sa_tmp_count  = string_arena_new(256);
    string_arena_t *sa_displ      = string_arena_new(256);
    string_arena_t *sa_tmp_displ  = string_arena_new(256);
    string_arena_t *sa_type       = string_arena_new(256);
    string_arena_t *sa_tmp_type   = string_arena_new(256);
    string_arena_t *sa_temp       = string_arena_new(1024);

    string_arena_t *sa_arena_r       = string_arena_new(256);
    string_arena_t *sa_tmp_arena_r   = string_arena_new(256);
    string_arena_t *sa_count_r      = string_arena_new(256);
    string_arena_t *sa_tmp_count_r  = string_arena_new(256);
    string_arena_t *sa_displ_r      = string_arena_new(256);
    string_arena_t *sa_tmp_displ_r  = string_arena_new(256);
    string_arena_t *sa_type_r     = string_arena_new(256);
    string_arena_t *sa_tmp_type_r = string_arena_new(256);
    string_arena_t *sa_temp_r       = string_arena_new(1024);

    int depnb, last_datatype_idx;
    assignment_info_t ai;
    expr_info_t info = EMPTY_EXPR_INFO;
    int nb_open_ldef;

    info.sa = sa2;
    info.prefix = "";
    info.suffix = "";
    info.assignments = "&"JDF2C_NAMESPACE"_tmp_locals";

    ai.sa = sa2;
    ai.holder = JDF2C_NAMESPACE"_tmp_locals.";
    ai.expr = NULL;
    coutput("static void\n"
            "%s(parsec_execution_stream_t *es, const %s *this_task,\n"
            "               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)\n"
            "{\n"
            "  const __parsec_%s_internal_taskpool_t *__parsec_tp = (const __parsec_%s_internal_taskpool_t*)this_task->taskpool;\n"
            "  parsec_task_t nc;  /* generic placeholder for locals */\n"
            "  parsec_dep_data_description_t data;\n"
            "  %s "JDF2C_NAMESPACE"_tmp_locals = *(%s*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */\n"
            "  int vpid_dst = -1, rank_src = 0, rank_dst = 0;\n"
            "%s"
            "  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;\n",
            name, parsec_get_name(jdf, f, "task_t"),
            jdf_basename, jdf_basename,
            parsec_get_name(jdf, f, "parsec_assignment_t"), parsec_get_name(jdf, f, "parsec_assignment_t"),
            UTIL_DUMP_LIST(sa1, f->locals, next,
                           dump_local_assignments, &ai, "", "  ", "\n", "\n"));

    coutput("   data_repo_t *successor_repo; parsec_key_t successor_repo_key;");

    coutput("%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->locals, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    coutput("  nc.taskpool  = this_task->taskpool;\n"
            "  nc.priority  = this_task->priority;\n"
            "  nc.chore_mask  = PARSEC_DEV_ALL;\n");
    coutput("#if defined(DISTRIBUTED)\n"
            "  rank_src = rank_of_%s(%s);\n"
            "#endif\n",
            f->predicate->func_or_mem,
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
        string_arena_init(sa_arena);
        string_arena_init(sa_count);
        string_arena_init(sa_displ);
        string_arena_init(sa_type);
        string_arena_init(sa_arena_r);
        string_arena_init(sa_count_r);
        string_arena_init(sa_displ_r);
        string_arena_init(sa_type_r);
        nb_open_ldef = 0;

        string_arena_add_string(sa_coutput,
                "    data.data   = this_task->data._f_%s.data_out;\n",
                fl->varname);

        for(dl = fl->deps; dl != NULL; dl = dl->next) {
            if( !(dl->dep_flags & flow_type) ) continue;
            /* Special case for the arena definition for WRITE-only flows */
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dl) )
                continue;

            string_arena_init(sa_tmp_arena);
            string_arena_init(sa_tmp_count);
            string_arena_init(sa_tmp_type);
            string_arena_init(sa_tmp_displ);

            string_arena_init(sa_tmp_arena_r);
            string_arena_init(sa_tmp_count_r);
            string_arena_init(sa_tmp_type_r);
            string_arena_init(sa_tmp_displ_r);


            /*********************************/
            /* LOCAL DATATYPE FOR RESHAPPING */
            /* always checked if !=; deps are 
             * grouped by type_remote, thus no 
             * checking for last_datatype_idx
             * dl->dep_datatype_index
             * Change that to minimize the 
             * number of reshapings? /!\
             */
            /*********************************/
            if( JDF_FLOW_TYPE_CTL & fl->flow_flags ) {
                string_arena_add_string(sa_tmp_arena, "NULL");
                string_arena_add_string(sa_tmp_count, "  /* Control: always empty */ 0");
                string_arena_add_string(sa_tmp_type, "PARSEC_DATATYPE_NULL");
                string_arena_add_string(sa_tmp_displ, "0");
            } else {

                jdf_datatransfer_type_t reshape_dtt = dl->datatype_local;

                string_arena_init(sa_temp);
                jdf_generate_arena_string_from_datatype(sa_temp, reshape_dtt);

                /* Always select the arena from the dependency. There's no relation
                 * between datatype and arena, therefore, there's no possible way to select
                 * an arena that's more appropriate than that. */
                /* We need to define the arena, otherwise, this would be considered a CTL flow. */
                string_arena_add_string(sa_tmp_arena, "%s->arena", string_arena_get_string(sa_temp));

                /* We select the dtt considering [type] on dependency or dtt on datacopy.
                 */
                if( DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(reshape_dtt) /* undefined type on dependency */
                        && (NULL == reshape_dtt.layout) ){ /* User didn't specify a custom layout*/

                    string_arena_add_string(sa_tmp_type, "PARSEC_DATATYPE_NULL");
                }else{
                    if( NULL == reshape_dtt.layout ){ /* User didn't specify a custom layout*/
                        string_arena_add_string(sa_tmp_type, "%s->opaque_dtt", string_arena_get_string(sa_temp));
                    }else{
                        string_arena_add_string(sa_tmp_type, "%s", dump_expr((void**)reshape_dtt.layout, &info));
                    }
                }

                assert( reshape_dtt.count != NULL );
                string_arena_add_string(sa_tmp_count, "%s", dump_expr((void**)reshape_dtt.count, &info));
                string_arena_add_string(sa_tmp_displ, "%s", dump_expr((void**)reshape_dtt.displ, &info));
            }

            string_arena_add_string(sa_datatype,"  if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {\n");
            jdf_generate_code_fillup_datatypes(sa_tmp_arena,    sa_arena,
                                               sa_tmp_type,     sa_type,
                                               sa_tmp_displ,    sa_displ,
                                               sa_tmp_count,    sa_count,
                                               NULL,
                                               NULL,
                                               NULL,
                                               sa_datatype,
                                               ".", "local", 0);

            /* Generate the remote datatype info only during releasing deps of
             * a real task. That is, avoid it when after reception during
             * release deps of a fake predecessor task.
             */
            /*********************************/
            /* REMOTE DATATYPE FOR SENDING */
            /*********************************/
            if( JDF_FLOW_TYPE_CTL & fl->flow_flags ) {
                string_arena_add_string(sa_tmp_arena_r, "NULL");
                string_arena_add_string(sa_tmp_count_r, "  /* Control: always empty */ 0");
                string_arena_add_string(sa_tmp_type_r, "PARSEC_DATATYPE_NULL");
                string_arena_add_string(sa_tmp_displ_r, "0");
            } else {
                string_arena_init(sa_temp);
                jdf_generate_arena_string_from_datatype(sa_temp, dl->datatype_remote);
                string_arena_add_string(sa_tmp_arena_r, "%s->arena", string_arena_get_string(sa_temp));

                /* We select the dtt considering [type_remote] on dependency or dtt on datacopy.
                 */
                if( DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dl->datatype_remote) /* undefined type on dependency */
                        && (NULL == dl->datatype_remote.layout) ){ /* User didn't specify a custom layout*/
                    /* using data dtt or using PARSEC_DATATYPE_NULL if we don't have a data (this is the case
                     * of iterate_successors -> get_datatype to recv the data; we are running over "fake predecessor task"
                     * and the goal is to check the successor datatype) */
                    string_arena_add_string(sa_tmp_type_r, "(data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL )");
                }else{
                    if( NULL == dl->datatype_remote.layout ){ /* User didn't specify a custom layout*/
                        string_arena_add_string(sa_tmp_type_r, "%s->opaque_dtt", string_arena_get_string(sa_temp));
                    }else{
                        string_arena_add_string(sa_tmp_type_r, "%s", dump_expr((void**)dl->datatype_remote.layout, &info));
                    }
                }
                assert( dl->datatype_remote.count != NULL );
                string_arena_add_string(sa_tmp_count_r, "%s", dump_expr((void**)dl->datatype_remote.count, &info));
                string_arena_add_string(sa_tmp_displ_r, "%s", dump_expr((void**)dl->datatype_remote.displ, &info));
            }

            if( last_datatype_idx != dl->dep_datatype_index ) {
                jdf_generate_code_fillup_datatypes(sa_tmp_arena_r, sa_arena_r,
                                                   sa_tmp_type_r,  sa_type_r,
                                                   sa_tmp_displ_r, sa_displ_r,
                                                   sa_tmp_count_r, sa_count_r,
                                                   NULL,
                                                   NULL,
                                                   NULL,
                                                   sa_datatype,
                                                   ".", "remote", 0);

                last_datatype_idx = dl->dep_datatype_index;

            }
            //end if of string_arena_add_string(sa_datatype,"  if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {\n");
            string_arena_add_string(sa_datatype,"  }\n");

            string_arena_init(sa_ontask);
            string_arena_add_string(sa_ontask,
                                    "if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst,"
                                    " successor_repo, successor_repo_key, ontask_arg) )\n"
                                    "  return;\n",
                                    JDF_OBJECT_ONAME(dl->guard->calltrue));

            if( NULL != dl->local_defs ) {
                jdf_expr_t *ld;
                for(ld = jdf_expr_lv_first(dl->local_defs); ld != NULL; ld = jdf_expr_lv_next(dl->local_defs, ld)) {
                    assert(NULL != ld->alias);
                    assert(-1 != ld->ldef_index);
                    string_arena_add_string(sa_coutput, "%s  int %s;\n", indent(nb_open_ldef), ld->alias);
                    if(JDF_RANGE == ld->op) {
                        string_arena_add_string(sa_coutput,
                                                "%s  for( %s = %s;",
                                                indent(nb_open_ldef), ld->alias, dump_expr((void**)ld->jdf_ta1, &info));
                        string_arena_add_string(sa_coutput, "%s <= %s; %s+=",
                                                ld->alias, dump_expr((void**)ld->jdf_ta2, &info), ld->alias);
                        string_arena_add_string(sa_coutput, "%s) {\n"
                                                "%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s;\n",
                                                dump_expr((void**)ld->jdf_ta3, &info),
                                                indent(nb_open_ldef), ld->ldef_index, ld->alias);
                        nb_open_ldef++;
                    } else {
                        string_arena_add_string(sa_coutput,
                                                "%s  "JDF2C_NAMESPACE"_tmp_locals.ldef[%d].value = %s = %s;\n",
                                                indent(nb_open_ldef), ld->ldef_index, ld->alias, dump_expr((void**)ld, &info));
                    }
                }
            }
            
            switch( dl->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
                if( NULL != dl->guard->calltrue->var) {
                    flowempty = 0;

                    string_arena_add_string(sa_deps,
                                            "%s",
                                            jdf_dump_context_assignment(sa1, jdf, f, fl, string_arena_get_string(sa_ontask),
                                                                        dl->guard->calltrue, dl, JDF_OBJECT_LINENO(dl),
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
                                            jdf_dump_context_assignment(sa1, jdf, f, fl, string_arena_get_string(sa_ontask),
                                                                        dl->guard->calltrue, dl, JDF_OBJECT_LINENO(dl),
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
                                            jdf_dump_context_assignment(sa1, jdf, f, fl, string_arena_get_string(sa_ontask),
                                                                        dl->guard->calltrue, dl, JDF_OBJECT_LINENO(dl),
                                                                        "      ", "nc"));
                    depnb++;

                    string_arena_init(sa_ontask);
                    string_arena_add_string(sa_ontask,
                                            "if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst,"
                                            " successor_repo, successor_repo_key, ontask_arg) )\n"
                                            "  return;\n",
                                            JDF_OBJECT_ONAME(dl->guard->callfalse));

                    if( NULL != dl->guard->callfalse->var ) {
                        string_arena_add_string(sa_deps,
                                                " else {\n"
                                                "%s"
                                                "    }\n",
                                                jdf_dump_context_assignment(sa1, jdf, f, fl, string_arena_get_string(sa_ontask),
                                                                            dl->guard->callfalse, dl, JDF_OBJECT_LINENO(dl),
                                                                            "      ", "nc") );
                    } else {
                        string_arena_add_string(sa_deps,
                                                "\n");
                    }
                } else {
                    depnb++;
                    string_arena_init(sa_ontask);
                    string_arena_add_string(sa_ontask,
                                            "if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &%s, &data, rank_src, rank_dst, vpid_dst,"
                                            " successor_repo, successor_repo_key, ontask_arg) )\n"
                                            "  return;\n",
                                            JDF_OBJECT_ONAME(dl->guard->callfalse));

                    if( NULL != dl->guard->callfalse->var ) {
                        flowempty = 0;
                        string_arena_add_string(sa_deps,
                                                "    if( !(%s) ) {\n"
                                                "%s"
                                                "    }\n",
                                                dump_expr((void**)dl->guard->guard, &info),
                                                jdf_dump_context_assignment(sa1, jdf, f, fl, string_arena_get_string(sa_ontask),
                                                                            dl->guard->callfalse, dl, JDF_OBJECT_LINENO(dl),
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

            while(nb_open_ldef > 0) {
                string_arena_add_string(sa_coutput, "%s  }\n", indent(nb_open_ldef));
                nb_open_ldef--;
            }
        }

        if( (1 == flowempty) && (0 == flowtomem) ) {
            coutput("  /* Flow of data %s has only IN dependencies */\n", fl->varname);
        } else if( 1 == flowempty ) {
            coutput("  /* Flow of data %s has only OUTPUT dependencies to Memory */\n", fl->varname);
        } else {
            coutput("  if( action_mask & 0x%x ) {  /* Flow of data %s [%d] */\n"
                    "%s"
                    "  }\n",
                    (flow_type & JDF_DEP_FLOW_OUT) ? fl->flow_dep_mask_out : fl->flow_dep_mask_in/*mask*/, fl->varname, fl->flow_index,
                    string_arena_get_string(sa_coutput)/*IFBODY*/);
        }
    }
    coutput("  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;\n");
    coutput("}\n\n");

    string_arena_free(sa_ontask);
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa_coutput);
    string_arena_free(sa_deps);
    string_arena_free(sa_datatype);
    string_arena_free(sa_arena);
    string_arena_free(sa_tmp_arena);
    string_arena_free(sa_count);
    string_arena_free(sa_tmp_count);
    string_arena_free(sa_displ);
    string_arena_free(sa_tmp_displ);
    string_arena_free(sa_type);
    string_arena_free(sa_tmp_type);
    string_arena_free(sa_temp);

    string_arena_free(sa_arena_r);
    string_arena_free(sa_tmp_arena_r);
    string_arena_free(sa_count_r);
    string_arena_free(sa_tmp_count_r);
    string_arena_free(sa_displ_r);
    string_arena_free(sa_tmp_displ_r);
    string_arena_free(sa_type_r);
    string_arena_free(sa_tmp_type_r);
    string_arena_free(sa_temp_r);

}

/**
 * Generates the code corresponding to inline_c expressions. If the inline_c was
 * defined in the context of a function, then it uses the function name and the
 * function arguments to build it's scope. Otherwise, it uses the global scope
 * of the taskpool.
 */
static void jdf_generate_inline_c_function(jdf_expr_t *expr)
{
    static int inline_c_functions = 0;
    string_arena_t *sa1, *sa2;
    assignment_info_t ai;
    int rc;
    jdf_expr_t *lbv;
    int i;

    /* Make sure we generate an inline only once (this allows for shortcuts while identifying identical parsec_expr_t */
    if( NULL != expr->jdf_c_code.fname )
        return;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    assert(JDF_OP_IS_C_CODE(expr->op));
    if( NULL != expr->jdf_c_code.function_context ) {
        rc = asprintf(&expr->jdf_c_code.fname, "%s_%s_inline_c_expr%d_line_%d",
                      jdf_basename, expr->jdf_c_code.function_context->fname,
                      ++inline_c_functions, expr->jdf_c_code.lineno);
        assert(rc != -1);

        if(expr->jdf_type == PARSEC_RETURN_TYPE_ARENA_DATATYPE_T){
            coutput("static inline %s %s(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *assignments, const __parsec_%s_%s_task_t *this_task)\n"
                    "{\n"
                    "  (void)__parsec_tp;\n",
                    full_type[expr->jdf_type],
                    expr->jdf_c_code.fname, jdf_basename,
                    parsec_get_name(NULL, expr->jdf_c_code.function_context, "parsec_assignment_t"),
                    jdf_basename, expr->jdf_c_code.function_context->fname);
        }else{
            coutput("static inline %s %s(const __parsec_%s_internal_taskpool_t *__parsec_tp, const %s *assignments)\n"
                    "{\n"
                    "  (void)__parsec_tp;\n",
                    full_type[expr->jdf_type],
                    expr->jdf_c_code.fname, jdf_basename,
                    parsec_get_name(NULL, expr->jdf_c_code.function_context, "parsec_assignment_t"));
        }

        coutput("  /* This inline C function was declared in the context of the task %s */\n",
                expr->jdf_c_code.function_context->fname);

        ai.sa = sa1;
        ai.holder = "assignments->";
        ai.expr = NULL;
        for(lbv = jdf_expr_lv_first(expr->local_variables); NULL != lbv; lbv = jdf_expr_lv_next(expr->local_variables, lbv)) {
            assert(NULL != lbv->alias);
            assert(-1 != lbv->ldef_index);
            coutput("  const int %s = assignments->ldef[%d].value;\n", lbv->alias, lbv->ldef_index);
        }
        coutput("%s\n",
                UTIL_DUMP_LIST(sa2, expr->jdf_c_code.function_context->locals, next,
                               dump_local_assignments, &ai, "", "  ", "\n", "\n"));

        for(i = 0, lbv = jdf_expr_lv_first(expr->local_variables); NULL != lbv; lbv = jdf_expr_lv_next(expr->local_variables, lbv), i++) {
            coutput("  (void)%s;\n", lbv->alias);
        }
        coutput("%s\n",
                UTIL_DUMP_LIST_FIELD(sa2, expr->jdf_c_code.function_context->locals, next, name,
                                     dump_string, NULL, "", "  (void)", ";", ";\n"));

        if(expr->jdf_type == PARSEC_RETURN_TYPE_ARENA_DATATYPE_T){
            init_from_data_info_t ai2;
            ai2.sa = sa1;
            ai2.where = "out";
            coutput("  /** Declare the data flow variables for the task */\n"
                    "%s\n",
                    UTIL_DUMP_LIST(sa2, expr->jdf_c_code.function_context->dataflow, next,
                                    dump_data_copy_init_for_inline_function, &ai2, "", "", "", ""));

        }
    } else {
        rc = asprintf(&expr->jdf_c_code.fname, "%s_inline_c_expr%d_line_%d",
                      jdf_basename, ++inline_c_functions, expr->jdf_c_code.lineno);
        assert(rc != -1);
        coutput("static inline int %s(const __parsec_%s_internal_taskpool_t *__parsec_tp, const parsec_assignment_t *assignments)\n"
                "{\n"
                "  /* This inline C function was declared in the global context: no variables */\n"
                "  (void)assignments;\n"
                "  (void)__parsec_tp;\n",
                expr->jdf_c_code.fname, jdf_basename);
    }

    
    string_arena_free(sa1);
    string_arena_free(sa2);

    coutput("%s\n", expr->jdf_c_code.code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline )
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    coutput("}\n"
            "\n");
    (void)rc;
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

static void jdf_check_user_defined_internals(jdf_t *jdf)
{
    jdf_function_entry_t *f;
    jdf_def_list_t* property;
    jdf_expr_t* expr;
    char *tmp;
    int rc;
    int global_nb_local_tasks = 0;

    if( NULL != (expr = jdf_find_property(jdf->global_properties, JDF_PROP_UD_NB_LOCAL_TASKS_FN_NAME, &property)) ) {
        var_to_c_code(expr);
        global_nb_local_tasks = JDF_HAS_UD_NB_LOCAL_TASKS;
    }

    for(f = jdf->functions; NULL != f; f = f->next) {
        /* We store the global UD flag about counting the number of local tasks in each
         * task class, for easy and unified checks, but all tasks have the same flag on
         * xor off, as nb_local_tasks must be define globally for the taskpool in order
         * to enable the undetermined number of tasks feature. */
        f->user_defines |= global_nb_local_tasks;
        
        if( NULL == jdf_property_get_string(f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, NULL) ) {
            rc = asprintf(&tmp, JDF2C_NAMESPACE"key_fns_%s", f->fname);
            if (rc == -1) {
                jdf_fatal(JDF_OBJECT_LINENO(f->properties),
                          "Out of ressource to generate the function name key_fns_%s\n", f->fname);
                exit(1);
            }
            (void)jdf_add_string_property(&f->properties, JDF_PROP_UD_HASH_STRUCT_NAME, tmp);
            free(tmp);
            f->user_defines &= ~JDF_FUNCTION_HAS_UD_HASH_STRUCT;
        } else {
            f->user_defines |= JDF_FUNCTION_HAS_UD_HASH_STRUCT;
        }
        
        if( NULL == jdf_property_get_string(f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, NULL) ) {
            rc = asprintf(&tmp, JDF2C_NAMESPACE"make_key_%s", f->fname);
            if (rc == -1) {
                jdf_fatal(JDF_OBJECT_LINENO(f->properties),
                          "Out of ressource to generate the function name make_key_%s\n", f->fname);
                exit(1);
            }
            (void)jdf_add_string_property(&f->properties, JDF_PROP_UD_MAKE_KEY_FN_NAME, tmp);
            free(tmp);
            f->user_defines &= ~JDF_FUNCTION_HAS_UD_MAKE_KEY;
        } else {
            f->user_defines |= JDF_FUNCTION_HAS_UD_MAKE_KEY;
        }
        
        if( NULL == (expr = jdf_find_property(f->properties, JDF_PROP_UD_STARTUP_TASKS_FN_NAME, &property)) ) {
            if( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP ) {
                rc = asprintf(&tmp, JDF2C_NAMESPACE"startup_%s", f->fname);
                if (rc == -1) {
                    jdf_fatal(JDF_OBJECT_LINENO(f->properties),
                              "Out of resources to generate the startup function name startup_%s\n", f->fname);
                    exit(1);
                }
                (void)jdf_add_function_property(&f->properties, JDF_PROP_UD_STARTUP_TASKS_FN_NAME, tmp);
            }
            f->user_defines &= ~JDF_FUNCTION_HAS_UD_STARTUP_TASKS_FUN;
        } else {
            var_to_c_code(expr);
            f->flags |= JDF_FUNCTION_FLAG_CAN_BE_STARTUP;
            f->user_defines |= JDF_FUNCTION_HAS_UD_STARTUP_TASKS_FUN;
        }

        if( NULL != (expr = jdf_find_property(f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, &property)) ) {
            var_to_c_code(expr);
            if( NULL == (expr = jdf_find_property(f->properties, JDF_PROP_UD_ALLOC_DEPS_FN_NAME, &property)) ) {
                jdf_fatal(JDF_OBJECT_LINENO(f->properties),
                          "Users who want to define a user-specific find_deps function ('%s') must also define a alloc_deps function\n",
                          expr->jdf_var);
                exit(1);
            } else {
                var_to_c_code(expr);
            }
            if( NULL == (expr = jdf_find_property(f->properties, JDF_PROP_UD_FREE_DEPS_FN_NAME, &property)) ) {
                jdf_fatal(JDF_OBJECT_LINENO(f->properties),
                          "Users who want to define a user-specific find_deps function ('%s') must also define a free_deps function\n",
                          expr->jdf_var);
                exit(1);
            } else {
                var_to_c_code(expr);
            }
            f->user_defines |= JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS;
        } else {
            if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_INDEX_ARRAY ) {
                (void)jdf_add_function_property(&f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, "parsec_default_find_deps");
            } else if( JDF_COMPILER_GLOBAL_ARGS.dep_management == DEP_MANAGEMENT_DYNAMIC_HASH_TABLE ) {
                (void)jdf_add_function_property(&f->properties, JDF_PROP_UD_FIND_DEPS_FN_NAME, "parsec_hash_find_deps");
            } else {
                assert(0);
            }
            f->user_defines &= ~JDF_FUNCTION_HAS_UD_DEPENDENCIES_FUNS;
        }
    }
}

static void
jdf_generate_code_find_deps(const jdf_t *jdf,
                            const jdf_function_entry_t *f,
                            const char *name)
{
    jdf_l2p_t *l2p = NULL, *l2p_item;
    coutput("parsec_dependency_t*\n"
            "%s(const parsec_taskpool_t*__tp,\n"
            "   parsec_execution_stream_t *es,\n"
            "   const parsec_task_t* restrict __task)\n"
            "{\n"
            "  parsec_dependencies_t *deps;\n"
            "  (void)es;\n"
            "  const parsec_%s_taskpool_t *__parsec_tp = (parsec_%s_taskpool_t*)__tp;\n"
            "  const __parsec_%s_%s_task_t* task = (__parsec_%s_%s_task_t*)__task;\n"
            "  deps = %sdependencies_array[task->task_class->task_class_id];\n",
            name,
            jdf_basename, jdf_basename,
            jdf_basename, f->fname, jdf_basename, f->fname,
            TASKPOOL_GLOBAL_PREFIX);

    l2p = build_l2p(f);
    for(l2p_item = l2p; NULL != l2p_item->next; l2p_item = l2p_item->next) {
        coutput("  assert( (deps->flags & PARSEC_DEPENDENCIES_FLAG_NEXT) != 0 );\n");
        coutput("  deps = deps->u.next[task->locals.%s.value - deps->min];\n"
                "  assert( NULL != deps );\n",
                l2p_item->pl->name);
    }
    coutput("  return &(deps->u.dependencies[task->locals.%s.value - deps->min]);\n",
            l2p_item->pl->name);
    free_l2p(l2p);
    coutput("}\n\n");
    (void)jdf;
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
        UTIL_DUMP_LIST(sa, f->dataflow, next, has_ready_input_dependency, &can_be_startup, "", "", "", "");
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

#if defined(PARSEC_HAVE_INDENT) && !(defined(__WINDOWS__) || defined(__MING64__) || defined(__CYGWIN__))
#include <sys/wait.h>
#endif

int jdf2c(const char *output_c, const char *output_h, const char *_jdf_basename, jdf_t *jdf)
{
    int ret = 0;

    jdf_cfilename = output_c;
    jdf_basename = _jdf_basename;
    cfile = NULL;
    hfile = NULL;

#if defined(PARSEC_HAVE_INDENT) && !(defined(__WINDOWS__) || defined(__MING64__) || defined(__CYGWIN__))
    /* When we apply indent/awk to the output of jdf2c, we need to make 
     * sure that the resultant file is flushed onto the filesystem before 
     * the rest of the compilation chain can takeover. An original version
     * was using rename(2) and temporary files to apply the indent/awk, but
     * it turns out to be very difficult to portably ensure visibility of 
     * the rename in subsequent operations (see PR#32 for the discussion).
     * As an alternative, we use pipes between jdf2c and the system spawned
     * indent/awk commands, so that we can spare the rename and rely on a 
     * classic fsync on the output to ensure visibilitiy. 
     */
    int child = -1;
    int cpipefd[2] = {-1,-1};
    int hpipefd[2] = {-1,-1};
    ret = pipe(cpipefd);
    if( -1 == ret ) {
        perror("Creating pipe between jdf2c and indent");
        goto err;
    }
    ret = pipe(hpipefd);
    if( -1 == ret ) {
        perror("Creating pipe between jdf2c and indent");
        goto err;
    }
    child = fork();
    if( -1 == child ) {
        perror("Creating fork to run indent");
        goto err;
    }
    if( 0 == child ) {
        char *command;
        close(cpipefd[1]);
        close(hpipefd[1]);
#if !defined(PARSEC_HAVE_AWK)
        asprintf(&command, "%s %s -o %s <&%d",
            PARSEC_INDENT_PREFIX, PARSEC_INDENT_OPTIONS, output_c, cpipefd[0]);
        system(command);
        free(command);
        asprintf(&command, "%s %s -o %s <&%d",
            PARSEC_INDENT_PREFIX, PARSEC_INDENT_OPTIONS, output_h, hpipefd[0]);
        system(command);
        free(command);
#else
        asprintf(&command,
             "%s %s <&%d -st | "
             "%s '$1==\"#line\" && $3==\"\\\"%s\\\"\" {printf(\"#line %%d \\\"%s\\\"\\n\", NR+1); next} {print}'"
             ">%s",
             PARSEC_INDENT_PREFIX, PARSEC_INDENT_OPTIONS, cpipefd[0],
             PARSEC_AWK_PREFIX, output_c, output_c,
             output_c);
        system(command);
        free(command);

        asprintf(&command,
             "%s %s <&%d -st | "
             "%s '$1==\"#line\" && $3==\"\\\"%s\\\"\" {printf(\"#line %%d \\\"%s\\\"\\n\", NR+1); next} {print}'"
             ">%s",
             PARSEC_INDENT_PREFIX, PARSEC_INDENT_OPTIONS, hpipefd[0],
             PARSEC_AWK_PREFIX, output_h, output_h,
             output_h);
        system(command);
        free(command);
#endif /* !defined(PARSEC_HAVE_AWK) */
        exit(0);
    }
    cfile = fdopen(cpipefd[1], "w");
    close(hpipefd[0]);
    hfile = fdopen(hpipefd[1], "w");
#else /* defined(PARSEC_HAVE_INDENT) */
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
#endif /* defined(PARSEC_HAVE_INDENT) */

    cfile_lineno = 1;
    hfile_lineno = 1;

    /**
     * Now generate the code.
     */
    jdf_generate_header_file(jdf);

    /**
     * Look for user-defined internal functions
     */
    jdf_check_user_defined_internals(jdf);

    jdf_minimal_code_before_prologue(jdf);

    /**
     * Dump the prologue section
     */
    if( NULL != jdf->prologue ) {
        coutput("%s", jdf->prologue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }

    /* Dump references to arenas_datatypes array */
    struct jdf_name_list* g;
    int datatype_index = 0;
    for( g = jdf->datatypes; NULL != g; g = g->next ) {
        coutput("#define PARSEC_%s_%s_ADT    (&__parsec_tp->super.arenas_datatypes[PARSEC_%s_%s_ADT_IDX])\n",
                jdf_basename, g->name,
                jdf_basename, g->name);
        datatype_index++;
    }

    jdf_generate_structure(jdf);
    jdf_generate_hashfunctions(jdf);
    jdf_generate_priority_prototypes(jdf);
    jdf_generate_functions_statics(jdf); // PETER generates startup tasks
    jdf_generate_startup_hook(jdf);

    /**
     * Generate the externally visible function.
     */
    jdf_generate_destructor(jdf);
    jdf_generate_constructor(jdf);
    coutput("  /** Generate class declaration and instance using the above constructor and destructor */\n"
            "  PARSEC_OBJ_CLASS_DECLARATION(__parsec_%s_internal_taskpool_t);\n"
            "  PARSEC_OBJ_CLASS_INSTANCE(__parsec_%s_internal_taskpool_t, parsec_%s_taskpool_t,\n"
            "                            __parsec_%s_internal_constructor, __parsec_%s_internal_destructor);\n\n",
            jdf_basename,
            jdf_basename, jdf_basename,
            jdf_basename, jdf_basename);
    
    jdf_generate_new_function(jdf);

    free_name_placeholders();

    /**
     * Dump all the epilogue sections
     */
    if( NULL != jdf->epilogue ) {
        coutput("%s", jdf->epilogue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            coutput("#line %d \"%s\"\n",cfile_lineno+1, jdf_cfilename);
    }

    /**
     * Automatically generate the taskpool instance.
     */
    if( NULL == jdf_find_property(jdf->global_properties, JDF_PROP_NO_AUTOMATIC_TASKPOOL_INSTANCE, NULL) ) {
        coutput("PARSEC_OBJ_CLASS_INSTANCE(parsec_%s_taskpool_t, parsec_taskpool_t, NULL, NULL);\n\n",
                jdf_basename);
    }

 err:
    if( NULL != cfile ) {
        fclose(cfile);
    }

    if( NULL != hfile ) {
        fclose(hfile);
    }

#if defined(PARSEC_HAVE_INDENT) && !(defined(__WINDOWS__) || defined(__MING64__) || defined(__CYGWIN__))
    /* wait for the indent command to generate the output files for us */
    if( -1 != child ) {
        waitpid(child, NULL, 0);
    }
#endif
    return ret;
}
