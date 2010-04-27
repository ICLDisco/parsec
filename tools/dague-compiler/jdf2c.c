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
static char *jdf_basename;

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
    len += snprintf(str+len, 1024 - len, ")  (__DAGuE_object->%s->data_of(__DAGuE_object->%s", data->dname, data->dname);
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
 *   dumps the jdf_expr* pointed to by elem into arg->sa, prefixing each non-global variable with arg->prefix
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
        string_arena_add_string(sa, "(%s == %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_NOTEQUAL:
        string_arena_add_string(sa, "(%s != %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_AND:
        string_arena_add_string(sa, "(%s && %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_OR:
        string_arena_add_string(sa, "(%s || %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_XOR:
        string_arena_add_string(sa, "(%s ^ %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_LESS:
        string_arena_add_string(sa, "(%s <  %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_LEQ:
        string_arena_add_string(sa, "(%s <= %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MORE:
        string_arena_add_string(sa, "(%s >  %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MEQ:
        string_arena_add_string(sa, "(%s >= %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_NOT:
        string_arena_add_string(sa, "!%s", dump_expr((void**)&e->jdf_ua, &li));
        break;
    case JDF_PLUS:
        string_arena_add_string(sa, "(%s + %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MINUS:
        string_arena_add_string(sa, "(%s - %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_TIMES:
        string_arena_add_string(sa, "(%s * %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_DIV:
        string_arena_add_string(sa, "(%s / %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_MODULO:
        string_arena_add_string(sa, "(%s %% %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_SHL:
        string_arena_add_string(sa, "(%s << %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_SHR:
        string_arena_add_string(sa, "(%s >> %s)", dump_expr((void**)&e->jdf_ba1, &li), dump_expr((void**)&e->jdf_ba2, &ri) );
        break;
    case JDF_RANGE:
        break;
    case JDF_TERNARY: {
        expr_info_t ti;
        string_arena_t *ta;
        ta = string_arena_new(8);
        ti.sa = ta;
        ti.prefix = expr_info->prefix;

        string_arena_add_string(sa, "(%s ? %s : %s)", dump_expr((void**)&e->jdf_tat, &ti), dump_expr((void**)&e->jdf_ta1, &li), dump_expr((void**)&e->jdf_ta2, &ri) );

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
 * Dump a predicate like #define F_pred(k, n, m) (rank == __DAGuE_object->ABC->rank_of(__DAGuE_object->ABC, k, n, m))
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
                            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL, 
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
    string_arena_add_string(sa, "(rank == __DAGuE_object->%s->rank_of(__DAGuE_object->%s, %s))", f->predicate->func_or_mem, f->predicate->func_or_mem,
                            UTIL_DUMP_LIST_FIELD(sa2, f->predicate->parameters, next, expr, dump_expr, &expr_info,
                                                 "", "", ", ", "")); 

    string_arena_free(sa2);
    string_arena_free(sa3);
    return string_arena_get_string(sa);
}

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
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dump_string, NULL, "", "DAGUE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name, dump_string, NULL, "",  "int ", ", ", ""));
    string_arena_free(sa1);
    string_arena_free(sa2);
    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);
}

static void typedef_structure(const jdf_t *jdf)
{
    int nbfunctions, nbmatrices;
    string_arena_t *sa1, *sa2;

    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);
    JDF_COUNT_LIST_ENTRIES(jdf->data, jdf_data_entry_t, next, nbmatrices);

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("#include <DAGuE.h>\n"
            "#include \"%s.h\"\n\n"
            "#define DAGuE_%s_NB_FUNCTIONS %d\n"
            "#define DAGuE_%s_NB_MATRICES %d\n", jdf_basename, jdf_basename, nbfunctions, jdf_basename, nbmatrices);
    coutput("typedef struct DAGuE_%s {\n", jdf_basename);
    coutput("  const DAGuE_t *functions_array[DAGuE_%s_NB_FUNCTIONS];\n", jdf_basename);
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->globals, next, name, dump_string, NULL, "", "  int ", ";\n", ";\n"));
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dump_string, NULL, "", "  DAGuE_ddesc_t *", ";\n", ";\n"));
    coutput("#  if defined(DAGuE_PROFILING)\n");
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname, dump_string, NULL, "", "  int ", "_start_key;\n", "_start_key;\n"));
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname, dump_string, NULL, "", "  int ", "_end_key;\n", "_end_key;\n"));
    coutput("#  endif /* defined(DAGuE_PROFILING) */\n");
    coutput("} __DAGuE_%s_t;\n"
            "\n", jdf_basename);

    /* dump the global symbols macros*/
    coutput("/* The globals */\n%s\n",
            UTIL_DUMP_LIST_FIELD(sa1, jdf->globals, next, name, dump_globals, sa2, "", "#define ", "\n", "\n"));

    /* dump the data access macros */
    coutput("/* The data access macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next, dump_data, sa2, "", "#define ", "\n", "\n"));

    /* dump the functions predicates */
    coutput("/* Functions Predicates */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->functions, next, dump_predicate, sa2, "", "#define ", "\n", "\n"));

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_constructor( const jdf_t* jdf )
{
    string_arena_t *sa1,*sa2;
    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("DAGuE_object_t *DAGuE_%s_new(%s, %s)\n{\n", jdf_basename,
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dump_string, NULL, "", "DAGuE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name, dump_string, NULL, "",  "int ", ", ", ""));

    coutput("}\n\n");

    string_arena_free(sa1);
    string_arena_free(sa2);
}

int jdf2c(char *_jdf_basename, const jdf_t *jdf)
{
    char filename[strlen(_jdf_basename)+4];
    int ret = 0;

    jdf_basename = _jdf_basename;
    cfile = NULL;
    hfile = NULL;

    sprintf(filename, "%s.c", jdf_basename);
    cfile = fopen(filename, "w");
    if( cfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", filename, strerror(errno));
        ret = -1;
        goto err;
    }

    sprintf(filename, "%s.h", jdf_basename);
    hfile = fopen(filename, "w");
    if( hfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", filename, strerror(errno));
        ret = -1;
        goto err;
    }

    cfile_lineno = 1;
    hfile_lineno = 1;
    
    jdf_generate_header_file(jdf);

    typedef_structure(jdf);

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
