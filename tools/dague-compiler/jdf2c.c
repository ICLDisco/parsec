#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "jdf.h"

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

typedef struct string_arena {
    char *ptr;
    int   pos;
    int   size;
} string_arena_t;

static string_arena_t *string_arena_new(int base_size)
{
    string_arena_t *sa;
    sa = (string_arena_t*)calloc(1, sizeof(string_arena_t));
    sa->ptr  = (char*)malloc(base_size);
    sa->pos  = 0;
    sa->size = base_size;
    return sa;
}

static void string_arena_free(string_arena_t *sa)
{
    free(sa->ptr);
    sa->pos  = -1;
    sa->size = -1;
    free(sa);
}

static void string_arena_add_string(string_arena_t *sa, const char *format, ...)
{
    va_list ap, ap2;
    int length;

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

    length = vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap);
    if( length >= (sa->size - sa->pos) ) {
        /* realloc */
        sa->size = sa->pos + length + 1;
        sa->ptr = (char*)realloc( sa->ptr, sa->size );
        length = vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap2);
    }
    sa->pos += length;

#if defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY) */
    va_end(ap);
}

typedef char *(*dumper_function_t)(void **elt, void *arg);

static char *dumpstring(void **elt, void *_)
{
    return (char*)*elt;
}

/**
 * util_dump_list:
 *  @param [IN] structure_ptr: pointer to a structure that implement any list
 *  @param [IN] nextfield:     the name of a field pointing to the next structure pointer
 *  @param [IN] eltfield:      the name of a field pointing to an element to print
 *  @param [IN] before:        string (of characters) representing what must appear before the list
 *  @param [IN] prefix:        string (of characters) representing what must appear before each element
 *  @param [IN] fct:           a function that transforms a pointer to an element to a string of characters
 *  @param [IN] separator:     string (of characters) that will be put between each element, but not at the end 
 *                             or before the first
 *  @param [IN] after:         string (of characters) that will be put at the end of the list, after the last
 *                             element
 *
 *  @return a string (of characters) with the list formed so. This string is useable until the next
 *          call to UTIL_DUMP_LIST
 *
 *  @example: to create the list of expressions that is a parameter call, use
 *    UTIL_DUMP_LIST(jdf->functions->predicates, next, expr, "(", "", dump_expr_inline, ", ", ")")
 *  @example: to create the list of declarations of globals, use
 *    UTIL_DUMP_LIST(jdf->globals, next, name, "", "  int ", dumpstring, ";\n", ";\n");
 */
#define UTIL_DUMP_LIST_FIELD(arena, structure_ptr, nextfield, eltfield, fct, fctarg, before, prefix, separator, after) \
    util_dump_list_fct( arena, structure_ptr,                           \
                        (char *)&(structure_ptr->nextfield)-(char *)structure_ptr, \
                        (char *)&(structure_ptr->eltfield)-(char *)structure_ptr, \
                            fct, fctarg, before, prefix, separator, after)
#define UTIL_DUMP_LIST(arena, structure_ptr, nextfield, fct, fctarg, before, prefix, separator, after) \
    util_dump_list_fct( arena, structure_ptr,                           \
                        (char *)&(structure_ptr->nextfield)-(char *)structure_ptr, \
                        0, \
                        fct, fctarg, before, prefix, separator, after)
static char *util_dump_list_fct( string_arena_t *sa, 
                                 void *firstelt, unsigned int next_offset, unsigned int elt_offset, 
                                 dumper_function_t fct, void *fctarg,
                                 const char *before, const char *prefix, const char *separator, const char *after)
{
    char *eltstr;
    void *elt;
    int pos = 0;
    
    sa->pos = 0;
    sa->ptr[0] = '\0';

    string_arena_add_string(sa, "%s", before);

    while(firstelt != NULL) {
        elt = ((void **)((char*)(firstelt) + elt_offset));
        eltstr = fct(elt, fctarg);

        firstelt = *((void **)((char *)(firstelt) + next_offset));
        if( firstelt != NULL ) {
            string_arena_add_string(sa, "%s%s%s", prefix, eltstr, separator);
        } else {
            string_arena_add_string(sa, "%s%s", prefix, eltstr);
        }
    }
    
    string_arena_add_string(sa, "%s", after);

    return sa->ptr;
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
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dumpstring, NULL, "", "DAGUE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name, dumpstring, NULL, "",  "int ", ", ", ""));
    string_arena_free(sa1);
    string_arena_free(sa2);
    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);
}

/**
 * Dump a global symbol like #define ABC (jdf->ABC)
 */
static char* dump_globals(void** elem, void *arg)
{
    static char str[512];
    snprintf( str, 512, "%s (jdf->%s)", (char*)*elem, (char*)*elem );
    return str;
}

/**
 * Dump a global symbol like #define ABC(A0, A1) (jdf->ABC->data_of(jdf->ABC, A0, A1))
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
    len += snprintf(str+len, 1024 - len, ")  (jdf->%s->data_of(jdf->%s", data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        len += snprintf( str+len, 1024 - len, ", (%s%d)", data->dname, i );
    }
    len += snprintf( str+len, 1024 - len, "))\n" );
    return str;
}

typedef struct expr_info {
    string_arena_t* sa;
    const char* prefix;
} expr_info_t;

static char * dump_expr(void **elem, void *arg)
{
    expr_info_t* expr_info = (expr_info_t*)arg;
    expr_info_t li, ri;
    jdf_expr_t *e = *(jdf_expr_t**)elem;
    string_arena_t *sa = expr_info->sa;
    string_arena_t *la, *ra;

    sa->pos = 0;
    sa->ptr[0] = '\0';

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
    
    return sa->ptr;
}

/**
 * Dump a predicate like #define F_pred(k, n, m) (rank == jdf->ABC->rank_of(jdf->ABC, k, n, m))
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

    sa->pos = 0;
    sa->ptr[0] = '\0';

    string_arena_add_string(sa, "%s_pred(%s) ",
                            f->fname,
                            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dumpstring, NULL, 
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
    string_arena_add_string(sa, "(rank == jdf->%s->rank_of(jdf->%s, %s))", f->predicate->func_or_mem, f->predicate->func_or_mem,
                            UTIL_DUMP_LIST_FIELD(sa2, f->predicate->parameters, next, expr, dump_expr, &expr_info,
                                                 "", "", ", ", "")); 

    string_arena_free(sa2);
    string_arena_free(sa3);
    return sa->ptr;
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
            UTIL_DUMP_LIST_FIELD( sa1, jdf->globals, next, name, dumpstring, NULL, "", "  int ", ";\n", ";\n"));
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dumpstring, NULL, "", "  DAGuE_ddesc_t *", ";\n", ";\n"));
    coutput("#  if defined(DAGuE_PROFILING)\n");
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname, dumpstring, NULL, "", "  int ", "_start_key;\n", "_start_key;\n"));
    coutput("%s", 
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname, dumpstring, NULL, "", "  int ", "_end_key;\n", "_end_key;\n"));
    coutput("#  endif /* defined(DAGuE_PROFILING) */\n");
    coutput("} DAGuE_%s_t;\n"
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
            UTIL_DUMP_LIST_FIELD( sa1, jdf->data, next, dname, dumpstring, NULL, "", "DAGuE_ddesc_t *", ", ", ""),
            UTIL_DUMP_LIST_FIELD( sa2, jdf->globals, next, name, dumpstring, NULL, "",  "int ", ", ", ""));

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
