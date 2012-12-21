/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"

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

extern const char *yyfilename;

static FILE *cudafile;
static int   cudafile_lineno;
static FILE *cfile;
static int   cfile_lineno;
static FILE *hfile;
static int   hfile_lineno;
static const char *jdf_basename;
static const char *jdf_cfilename;
static char       *jdf_cudafilename;

/** Optional declarations of local functions */
static int jdf_expr_depends_on_symbol(const char *varname, const jdf_expr_t *expr);
static void jdf_generate_code_hook(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname);
static void jdf_generate_code_iterate_successors(const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix);

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

#if defined(__GNUC__)
static void cudaoutput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void cudaoutput(const char *format, ...)
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
        fwrite(res, len, 1, cudafile);
        cudafile_lineno += nblines(res);
        free(res);
    }
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
        if(flow->access_type != JDF_VAR_TYPE_CTL) {
            for(dep = flow->deps; dep != NULL; dep = dep->next)
                if( JDF_DEP_TYPE_OUT == dep->type ) {
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
    string_arena_add_string(sa, "%s (__dague_handle->super.%s)", global->name, global->name );
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
 * dump_data_handle:
 *  Dump a global symbol like
 *  #define A_HANDLE(A0, A1) (__dague_handle->ABC->data_handle_of(__dague_handle->ABC, A0, A1))   
 */
static char*
dump_data_handle(void** elem, void *arg)
{
    jdf_data_entry_t* data = (jdf_data_entry_t*)elem;
    string_arena_t *sa = (string_arena_t*)arg;
    int i;

    string_arena_init(sa);
    string_arena_add_string(sa, "%s_handle(%s%d", data->dname, data->dname, 0 );
    for( i = 1; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ",%s%d", data->dname, i );
    }
    string_arena_add_string(sa, ")  (((dague_ddesc_t*)__dague_handle->super.%s)->data_handle_of((dague_ddesc_t*)__dague_handle->super.%s", 
                            data->dname, data->dname);
    for( i = 0; i < data->nbparams; i++ ) {
        string_arena_add_string(sa, ", (%s%d)", data->dname, i );
    }
    string_arena_add_string(sa, "))\n" );
    return string_arena_get_string(sa);
}





/**
 * Parameters for dump_expr helper
 */
typedef struct expr_info {
    string_arena_t* sa;
    const char* prefix;
    char *assignments;
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
    jdf_expr_t *e = (jdf_expr_t*)elem;
    string_arena_t *sa = expr_info->sa;
    string_arena_t *la, *ra;
    char *vc, *dot;

    string_arena_init(sa);

    la = string_arena_new(8);
    ra = string_arena_new(8);

    li.sa = la;
    li.prefix = expr_info->prefix;
    li.assignments = expr_info->assignments;
    
    ri.sa = ra;
    ri.prefix = expr_info->prefix;
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
            string_arena_add_string(sa, "%s%s", expr_info->prefix, e->jdf_var);
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
        if (  NULL == e->jdf_c_code.fname ) {
            fprintf(stderr, "Internal Error: Function for the inline C expression of line %d has not been generated.\n",
                    e->jdf_c_code.lineno);
            assert( NULL != e->jdf_c_code.fname );
        }
        string_arena_add_string(sa, "%s((const dague_handle_t*)__dague_handle, %s)", 
                                e->jdf_c_code.fname, expr_info->assignments);
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
                            UTIL_DUMP_LIST_FIELD(sa2, f->definitions, next, name,
                                                 dump_string, NULL, 
                                                 "", "", ", ", ""));
    expr_info.sa = sa3;
    expr_info.prefix = "";
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
 * Parameters of the dump_assignments function
 */
typedef struct assignment_info {
    string_arena_t *sa;
    int idx;
    const char *holder;
    const jdf_expr_t *expr;
} assignment_info_t;

/**
 * dump_assignments:
 *  Takes the pointer to the name of a parameter,
 *  a pointer to a dump_info, and prints <assignment_info.holder>k = assignments[<assignment_info.idx>] 
 *  into assignment_info.sa for each variable k that belong to the expression that is going
 *  to be used. This expression is passed into assignment_info->expr. If assignment_info->expr 
 *  is NULL, all variables are assigned.
 */
static char *dump_assignments(void **elem, void *arg)
{
    char *varname = *(char**)elem;
    assignment_info_t *info = (assignment_info_t*)arg;
    
    string_arena_init(info->sa);
    if( (NULL == info->expr) || jdf_expr_depends_on_symbol(varname, info->expr) ) {
        string_arena_add_string(info->sa, "%s = %s[%d].value; (void) %s", varname, info->holder, info->idx, varname);
        info->idx++;
        return string_arena_get_string(info->sa);
    } else {
        info->idx++;
        return NULL;
    }
}


static char *dump_retrieving_param(void **elem, void *arg)
{
    char *varname = *(char**)elem;
    assignment_info_t *info = (assignment_info_t*)arg;
    
    string_arena_init(info->sa);
    
    if( (NULL == info->expr) || jdf_expr_depends_on_symbol(varname, info->expr) ) 
    {
        string_arena_add_string(info->sa, "%s = args->var[%d]; (void) %s", varname, info->idx, varname);
        info->idx++;
        return string_arena_get_string(info->sa);
    } 
    else 
    {
        info->idx++;
        return NULL;
    }


}

static char*
dump_registering_global(string_arena_t *sa, jdf_global_entry_t *globals)
{
    jdf_global_entry_t *glob = globals;
    int cpt = 0;
    
    string_arena_init(sa);
    
    for(; glob != NULL; glob = glob->next)
    {
	if(glob->data == NULL)
	    string_arena_add_string(sa, "    arg->glob[%d] = (void*)(&(%s));\n", cpt++, glob->name); 
    }
    return string_arena_get_string(sa);
}


static char*
dump_retrieving_global(string_arena_t *sa, jdf_global_entry_t *globals)
{
    jdf_global_entry_t *glob = globals;
    jdf_expr_t* type_str;
    int cpt = 0;
    expr_info_t info;
    
    string_arena_init(sa);

    info.sa = string_arena_new(8);
    info.prefix = "";
    info.assignments = "assignments";


    for(; glob != NULL; glob = glob->next)
    {
	if(glob->data == NULL)
	{
	    type_str = jdf_find_property( glob->properties, "type", NULL );
	    string_arena_add_string(sa, "    %s %s = *((%s*) args->glob[%d]);\n", 
				    (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), 
				    glob->name,  
				    (NULL == type_str ? "int" : dump_expr((void**)type_str, &info)), cpt++);
	}
    }
    return string_arena_get_string(sa);
}


static char*
dump_registering_param(void **elem, void *arg)
{
    char *varname = *(char**)elem;
    assignment_info_t *info = (assignment_info_t*)arg;
    
    string_arena_init(info->sa);
    if( (NULL == info->expr) || jdf_expr_depends_on_symbol(varname, info->expr) ) {
        string_arena_add_string(info->sa, "arg->var[%d] = %s[%d].value", info->idx, info->holder, info->idx);
        info->idx++;
        return string_arena_get_string(info->sa);
    } else {
        info->idx++;
        return NULL;
    }
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
    jdf_dep_type_t target;
    jdf_dep_t *l;

    if( !strcmp(arg, "IN") )
        target = JDF_DEP_TYPE_IN;
    else
        target = JDF_DEP_TYPE_OUT;

    for(l = d->deps; l != NULL; l = l->next)
        if( l->type == target )
            break;
    if( l == NULL )
        return NULL;
    return d->varname;
}

/* /\** */
/*  * dump_all_handle_assignments: */
/*  *  Takes the pointer to a flow *f, let say that f->varname == "A", */
/*  *  this produces a string like t->handles[i] = super.A_handle; */
/*  *  And so on, for the flow while f->next == NULL */
/*  *\/ */
/* static char *dump_all_handle_assignments(void **elem, void *arg) */
/* { */
/*     int cmp = 0; */
/*     string_arena_t *sa = (string_arena_t*) arg; */
/*     jdf_dataflow_t *f = (jdf_dataflow_t*) elem; */
/*     string_arena_init(sa); */
    
/*     for(; f != NULL; f = f->next) */
/*     { */
/* 	string_arena_add_string(sa, */
/* 				"    t->handles[%d] = ->%s_handle;\n", */
/* 				cmp, f->varname); */
/* 	cmp++; */
/*     } */

/*     return string_arena_get_string(sa); */
/* } */


/**
 * dump_data_declaration:
 *  Takes the pointer to a flow *f, let say that f->varname == "A",
 *  this produces a string like void *A = NULL;\n  
 *  dague_arena_chunk_t *gT = NULL;\n  data_repo_entry_t *eT = NULL;\n
 */
static char *dump_data_declaration(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_dataflow_t *f = (jdf_dataflow_t*)elem;
    char *varname = f->varname;

    if(f->access_type == JDF_VAR_TYPE_CTL) 
        return NULL;

    string_arena_init(sa);

    string_arena_add_string(sa, 
                            "    starpu_data_handle_t *%s = NULL; (void)%s;\n"
                            "    dague_arena_chunk_t *g%s = NULL; (void)g%s;\n"
                            "    data_repo_entry_t *e%s = NULL; (void)e%s;\n", 
                            varname, varname, varname,
                            varname, varname, varname);

    return string_arena_get_string(sa);
}

/* /\** */
/*  * dump_data_handle_assign: */
/*  * Create the handlers for the cpu function codelet parameters and assign them with the corresponding parameter */
/*  *\/ */
/* static char *dump_data_handle_assign(jdf_dataflow_t *f, string_arena_t *sa) */
/* { */
/*     int num_arg = 0; */
        
/*     string_arena_init(sa); */
    
/*     for(; f != NULL; f = f->next) */
/*     { */
/* 	if(f->access_type == JDF_VAR_TYPE_CTL)  */
/* 	    continue; */
	
/* 	string_arena_add_string(sa, */
/* 				"    starpu_data_handle_t *%s = (starpu_data_handle_t*) buffer[%d];\n" */
/* 				"    (void) %s;\n", */
/* 				f->varname, num_arg++, f->varname); */
/*     } */
/*     return string_arena_get_string(sa); */
/* } */


/**
 * dump_dataflow_varname:
 *  Takes the pointer to a flow *f, and print the varname
 */
static char *dump_dataflow_varname(void **elem, void *_)
{
    (void)_;
    jdf_dataflow_t *f = (jdf_dataflow_t *)elem;
    if( f->access_type == JDF_VAR_TYPE_CTL ) return NULL;
    return f->varname;
}

#if 0
 /* TODO: Thomas needs to check if this is old junk or WIP */
static double unique_rgb_color_saturation;
static double unique_rgb_color_value;

static void init_unique_rgb_color(void)
{
    unique_rgb_color_value = 0.5 + (0.5 * (double)rand() / (double)RAND_MAX);
    unique_rgb_color_saturation = 0.5 + (0.5 * (double)rand() / (double)RAND_MAX);
}
#endif

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

    if( !jdf_property_get_int(f->properties, "profile", 1) ) {
        return NULL;
    }

    string_arena_init(info->sa);

    get_unique_rgb_color((float)info->idx / (float)info->maxidx, &R, &G, &B);
    info->idx++;

    string_arena_add_string(info->sa,
                            "dague_profiling_add_dictionary_keyword(\"%s\", \"fill:%02X%02X%02X\",\n"
                            "                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,\n"
                            "                                       (int*)&_res->super.super.profiling_array[0 + 2 * %s_%s.function_id /* %s start key */],\n"
                            "                                       (int*)&_res->super.super.profiling_array[1 + 2 * %s_%s.function_id /* %s end key */]);",
                            fname, R, G, B,
                            jdf_basename, fname, fname,
                            jdf_basename, fname, fname);

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
                                "_%s_startup_tasks(context, (__dague_%s_internal_handle_t*)dague_handle, pready_list);",
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
    jdf_expr_t *prop = jdf_find_property( global->properties, "default", NULL );
    expr_info_t info;

    string_arena_init(sa);
    /* We might have a default value */
    if( NULL == prop ) prop = global->expression;

    /* No default value ? */
    if( NULL == prop ) {
        prop = jdf_find_property( global->properties, "hidden", NULL );
        /* Hidden variable or not ? */
        if( NULL == prop )
            string_arena_add_string(sa, "_res->super.%s = %s;", global->name, global->name);
    } else {
        info.sa = string_arena_new(8);
        info.prefix = "";
        info.assignments = "assignments";
        string_arena_add_string(sa, "_res->super.%s = %s;",
                                global->name, dump_expr((void**)prop, &info));
        string_arena_free(info.sa);
    }

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


static char* dump_starpu_handle(void **elem, void *arg)
{
  //  (void **) elem;                                                                                                                                
  // (void *) arg;                                                                                                                                   
  typed_globals_info_t* prop = (typed_globals_info_t*) arg;
  string_arena_t *sa=prop->sa;
  jdf_global_entry_t* global = (jdf_global_entry_t*)elem;

  string_arena_init(sa);
  string_arena_add_string(sa, "starpu_data_handle_t %s_handle", global->name);

  return string_arena_get_string(sa);
}

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

static char *dump_data_repository_constructor(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t*)arg;
    jdf_function_entry_t *f = (jdf_function_entry_t *)elem;

    string_arena_init(sa);

    if( 0 == function_has_data_output(f) ) {
        string_arena_add_string(sa, 
                                "  %s_nblocal_tasks = %s_%s_internal_init(_res);\n"
                                "  (void)%s_nblocal_tasks;\n",
                                f->fname, jdf_basename, f->fname,
                                f->fname);
    } else {
        int nbdata = 0;
        JDF_COUNT_LIST_ENTRIES(f->dataflow, jdf_dataflow_t, next, nbdata);
        string_arena_add_string(sa, 
                                "  %s_nblocal_tasks = %s_%s_internal_init(_res);\n"
                                "  if( 0 == %s_nblocal_tasks ) %s_nblocal_tasks = 10;\n"
                                "  _res->%s_repository = data_repo_create_nothreadsafe(\n"
                                "          ((unsigned int)(%s_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?\n"
                                "          MAX_DATAREPO_HASH :\n"
                                "          ((unsigned int)(%s_nblocal_tasks * 1.5)), %d);\n",
                                f->fname, jdf_basename, f->fname,
                                f->fname, f->fname,
                                f->fname,
                                f->fname, f->fname, nbdata);
    }

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
    else if( JDF_OP_IS_BINARY(e->op) )
        return jdf_expr_depends_on_symbol(name, e->jdf_ba1) ||
            jdf_expr_depends_on_symbol(name, e->jdf_ba2);
    else
        return 0;
}

jdf_expr_t* jdf_find_property( const jdf_def_list_t* properties, const char* property_name, jdf_def_list_t** property )
{
    const jdf_def_list_t* current = properties;

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
        type |= dl->type;
    }
    return type;
}

static int jdf_data_output_index(const jdf_t *jdf, const char *fname, const char *varname)
{
    int i;
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;
    
    i = 0;
    for(f = jdf->functions; f != NULL; f = f->next) {
        if( !strcmp(f->fname, fname) ) {
            for( fl = f->dataflow; fl != NULL; fl = fl->next) {
                if( jdf_dataflow_type(fl) & JDF_DEP_TYPE_OUT ) {
                    if( !strcmp(fl->varname, varname) ) {
                        return i;
                    }
                    i++;
                }
            }
            return -1;
        }
    }
    return -2;
}

static int jdf_data_input_index(const jdf_function_entry_t *f, const char *varname)
{
    int i;
    jdf_dataflow_t *fl;
    
    i = 0;
    for( fl = f->dataflow; fl != NULL; fl = fl->next) {
        if( jdf_dataflow_type(fl) & JDF_DEP_TYPE_IN ) {
            if( !strcmp(fl->varname, varname) ) {
                return i;
            }
            i++;
        }
    }
    return -1;
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
    for(i = 0; i < 80; i++)
        coutput("%c", marker);
    coutput("*\n *%s%s", indent(ls/2), v);  /* indent drop two spaces */
    coutput("%s*\n *", indent(rs/2));       /* dont merge these two calls. Read the comment on the indent function */
    for(i = 0; i < 80; i++)
        coutput("%c", marker);
    coutput("*/\n\n");
    free(v);
}

/** Structure Generators **/

static void jdf_generate_header_file(const jdf_t* jdf)
{
    string_arena_t *sa1, *sa2, *sa3, *sa4, *sa5;
    struct jdf_name_list* g;
    int datatype_index = 0;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);
    sa3 = string_arena_new(64);
    sa4 = string_arena_new(64);
    sa5 = string_arena_new(64);


    houtput("#ifndef _%s_h_\n"
            "#define _%s_h_\n",
            jdf_basename, jdf_basename);
    houtput("#include <dague.h>\n"
            "#include <debug.h>\n"
            "#include <assert.h>\n\n");

    houtput("#define MAX_GLOBAL_COUNT 50\n");
    for( g = jdf->datatypes; NULL != g; g = g->next ) {
        houtput("#define DAGUE_%s_%s_ARENA    %d\n",
                jdf_basename, g->name, datatype_index);
        datatype_index++;
    }
    houtput("\ntypedef struct dague_%s_handle {\n", jdf_basename);
    houtput("  dague_handle_t super;\n");
    {
        typed_globals_info_t prop = { sa2, NULL, NULL };
        houtput("  /* The list of globals */\n"
                "%s",
                UTIL_DUMP_LIST( sa1, jdf->globals, next, dump_typed_globals, &prop,
                                "", "  ", ";\n", ";\n"));
    }
    {
	typed_globals_info_t prop = { sa5, NULL, NULL };
	houtput("  /* The list of global handles */\n"
		"%s",
		UTIL_DUMP_LIST( sa4, jdf->globals, next, dump_starpu_handle, &prop,
				"",  "  ", ";\n", ";\n"));
    }
    houtput("  /* The array of datatypes %s */\n"
            "  dague_arena_t* arenas[%d];\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->datatypes, next, name,
                                  dump_string, NULL, "", "", ",", ""),
            datatype_index);

    houtput("} dague_%s_handle_t;\n\n", jdf_basename);
    
    {
        typed_globals_info_t prop = { sa3, NULL, "hidden" };
        houtput("extern dague_%s_handle_t *dague_%s_new(%s);\n", jdf_basename, jdf_basename,
                UTIL_DUMP_LIST( sa2, jdf->globals, next, dump_typed_globals, &prop,
                                "", "", ", ", ""));
    }

    houtput("extern void dague_%s_destroy( dague_%s_handle_t *o );\n\n",
            jdf_basename, jdf_basename);


    // on génère la structure pour les arguments des fonctions callback
    houtput("struct callback_args {\n"
	    "    dague_execution_unit_t     *exec_unit;\n"
	    "    dague_execution_context_t  *exec_context;\n"
	    "};\n");
    // on génère la structure pour les arguments des fonctions (cpu, gpu)
    houtput("struct func_args {\n"
	    "    int var[MAX_PARAM_COUNT];\n"
	    "    //struct callback_args *callback;\n"
	    "    void *obj;\n"
	    "    void *glob[MAX_GLOBAL_COUNT];\n"
	    "};\n");


    houtput("void generic_scheduling_func(dague_execution_unit_t *context, dague_list_item_t *elt);\n");

    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa3);
    string_arena_free(sa4);
    string_arena_free(sa5);
}

static void jdf_generate_structure(const jdf_t *jdf)
{
    int nbfunctions, nbdata;
    string_arena_t *sa1, *sa2;

    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, nbfunctions);
    JDF_COUNT_LIST_ENTRIES(jdf->data, jdf_data_entry_t, next, nbdata);

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("#include <dague.h>\n"
            "#include <scheduling.h>\n"
            "#include <remote_dep.h>\n"
            "#if defined(HAVE_PAPI)\n"
            "#include <papime.h>\n"
            "#endif\n"
            "#include \"%s.h\"\n\n"
            "#define DAGUE_%s_NB_FUNCTIONS %d\n"
            "#define DAGUE_%s_NB_DATA %d\n"
            "#if defined(DAGUE_PROF_TRACE)\n"
            "int %s_profiling_array[2*DAGUE_%s_NB_FUNCTIONS] = {-1};\n"
            "#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \\\n"
            "   dague_profile_ddesc_info_t info;                         \\\n"
            "   info.desc = (dague_ddesc_t*)refdesc;                     \\\n"
            "   info.id = refid;                                         \\\n"
            "   dague_profiling_trace(context->eu_profile,               \\\n"
            "                         __dague_handle->super.super.profiling_array[(key)],\\\n"
            "                         eid, (void*)&info);                \\\n"
            "  } while(0);\n"
            "#else\n"
            "#define TAKE_TIME(context, key, id, refdesc, refid)\n"
            "#endif\n"
            "#include \"dague_prof_grapher.h\"\n"
            "#include <mempool.h>\n", 
            jdf_basename, 
            jdf_basename, nbfunctions, 
            jdf_basename, nbdata,
            jdf_basename, jdf_basename);
    coutput("typedef struct __dague_%s_internal_handle {\n", jdf_basename);
    coutput(" dague_%s_handle_t super;\n",
            jdf_basename);
    coutput("  /* The list of data repositories */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            if( 0 != function_has_data_output(f) )
                coutput("  data_repo_t *%s_repository;\n", f->fname);
        }
    }
    coutput("} __dague_%s_internal_handle_t;\n"
            "\n", jdf_basename);

    
    
    UTIL_DUMP_LIST(sa1, jdf->globals, next,
                   dump_globals, sa2, "", "#define ", "\n", "\n");
   
    
    if( 1 < strlen(string_arena_get_string(sa1)) ) {
        coutput("/* Globals */\n%s\n", string_arena_get_string(sa1));
    }

    coutput("/* Data Access Macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_data, sa2, "", "#define ", "\n", "\n"));

    coutput("/* Data handles Access Macros */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->data, next,
                           dump_data_handle, sa2, "", "#define ", "\n", "\n"));
    

    coutput("/* Functions Predicates */\n%s\n",
            UTIL_DUMP_LIST(sa1, jdf->functions, next,
                           dump_predicate, sa2, "", "#define ", "\n", "\n"));

    coutput("/* Data Repositories */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            if( 0 != function_has_data_output(f) )
                coutput("#define %s_repo (__dague_handle->%s_repository)\n",
                        f->fname, f->fname);
        }
    }

    coutput("/* Dependency Tracking Allocation Macro */\n"
            "#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \\\n"
            "do {                                                                                         \\\n"
            "  int _vmin = (vMIN);                                                                        \\\n"
            "  int _vmax = (vMAX);                                                                        \\\n"
            "  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \\\n"
            "                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \\\n"
            "  /*DEBUG((\"Allocate %%d spaces for loop %%s (min %%d max %%d) 0x%%p last_dep 0x%%p\\n\", */         \\\n"
            "  /*       (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP))); */ \\\n"
            "  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \\\n"
            "  DAGUE_STAT_INCREASE(mem_bitarray,  sizeof(dague_dependencies_t) + STAT_MALLOC_OVERHEAD +   \\\n"
            "                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \\\n"
            "  (DEPS)->symbol = (vSYMBOL);                                                                \\\n"
            "  (DEPS)->min = _vmin;                                                                       \\\n"
            "  (DEPS)->max = _vmax;                                                                       \\\n"
            "  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \\\n"
            "} while (0)                                                                                  \n\n"
            "static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };                     \n\n"
            "static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };                     \n\n");
    

    cudaoutput("#include <starpu.h>\n"
	       "#include <starpu_cuda.h>\n"
	       "#include \"%s.h\"\n"
	       "#include \"dague.h\"\n\n", jdf_basename);
    
    
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

    if( e->op == JDF_RANGE ) {
        char *subf = (char*)malloc(strlen(name) + 64);
        sprintf(subf, "rangemin_of_%s", name);
        jdf_generate_expression(jdf, context, e->jdf_ba1, subf);
        sprintf(subf, "rangemax_of_%s", name);
        jdf_generate_expression(jdf, context, e->jdf_ba2, subf);

        coutput("static const expr_t %s = {\n"
                "  .op = EXPR_OP_BINARY_RANGE,\n"
                "  .u_expr.binary = {\n"
                "    .op1 = &rangemin_of_%s,\n"
                "    .op2 = &rangemax_of_%s\n"
                "  }\n"
                "};\n",
                name, name, name);
    } else {
        info.sa = sa;
        info.prefix = "";
        info.assignments = "assignments";
        ai.sa = sa3;
        ai.idx = 0;
        ai.holder = "assignments";
        ai.expr = e;
        coutput("static inline int %s_fct(const dague_handle_t *__dague_handle_parent, const assignment_t *assignments)\n"
                "{\n"
                "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)__dague_handle_parent;\n"
                "%s\n"
                "  (void)__dague_handle; (void)assignments;\n"
                "  return %s;\n"
                "}\n", name, jdf_basename, jdf_basename,
                UTIL_DUMP_LIST_FIELD(sa2, context, next, name, 
                                     dump_assignments, &ai, "", "  int ", ";\n", ";\n"),
                dump_expr((void**)e, &info));

        coutput("static const expr_t %s = {\n"
                "  .op = EXPR_OP_INLINE,\n"
                "  .inline_func = %s_fct\n"
                "};\n", name, name);
    }

    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);
}

static void jdf_generate_predicate_expr( const jdf_t *jdf, const jdf_def_list_t *context,
                                         const char *fname, const char *name)
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa3 = string_arena_new(64);
    string_arena_t *sa4 = string_arena_new(64);
    string_arena_t *sa5 = string_arena_new(64);
    assignment_info_t ai;

    (void)jdf;

    ai.sa = sa3;
    ai.idx = 0;
    ai.holder = "assignments";
    ai.expr = NULL;
    coutput("static inline int %s_fct(const dague_handle_t *__dague_handle_parent, const assignment_t *assignments)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)__dague_handle_parent;\n"
            "%s\n"
            "  /* Silent Warnings: should look into predicate to know what variables are usefull */\n"
            "  (void)__dague_handle;\n"
            "%s\n"
            "  /* Compute Predicate */\n"
            "  return %s_pred%s;\n"
            "}\n", name, jdf_basename, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa2, context, next, name, 
                                 dump_assignments, &ai, "", "  int ", ";\n", ";\n"),
            UTIL_DUMP_LIST_FIELD(sa5, context, next, name,
                                 dump_string, NULL, "", "  (void)", ";\n", ";"),
            fname, 
            UTIL_DUMP_LIST_FIELD(sa4, context, next, name,
                                 dump_string, NULL, "(", "", ", ", ")"));
    string_arena_free(sa);
    string_arena_free(sa2);
    string_arena_free(sa3);
    string_arena_free(sa4);
    string_arena_free(sa5);

    coutput("static const expr_t %s = {\n"
            "  .op = EXPR_OP_INLINE,\n"
            "  .inline_func = %s_fct\n"
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

static void jdf_generate_dependency( const jdf_t *jdf, const char* datatype_name,
                                     const jdf_call_t *call, const char *depname,
                                     const char *condname, const jdf_def_list_t *context )
{
    string_arena_t *sa = string_arena_new(64);
    jdf_expr_t *le;
    char *exprname;
    int i;
    char pre[8];

    string_arena_add_string(sa, 
                            "static const dep_t %s = {\n"
                            "  .cond = %s,\n"
                            "  .dague = &%s_%s,\n",
                            depname, 
                            condname,
                            jdf_basename, call->func_or_mem);

    if( call->var != NULL ) {
            string_arena_add_string(sa, 
                                    "  .flow = &flow_of_%s_%s_for_%s,\n",
                                    jdf_basename, call->func_or_mem, call->var);
    }
    string_arena_add_string(sa, 
                            "  .datatype_index = DAGUE_%s_%s_ARENA,"
                            "  .call_params = {\n",
                            jdf_basename, datatype_name);

    exprname = (char *)malloc(strlen(depname) + 32);
    pre[0] = '\0';
    for( i = 1, le = call->parameters; le != NULL; i++, le = le->next ) {
        sprintf(exprname, "expr_of_p%d_for_%s", i, depname);
        string_arena_add_string(sa, "%s    &%s", pre, exprname);
        jdf_generate_expression(jdf, context, le, exprname);
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
                                   jdf_dataflow_t *flow, const char *prefix, uint32_t flow_index )
{
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa_dep_in = string_arena_new(64);
    string_arena_t *sa_dep_out = string_arena_new(64);
    string_arena_t *psa;
    int alldeps_type;
    jdf_dep_t *dl;
    char *sym_type;
    char *access_type;
    int depid;
    char *depname;
    char *condname;
    char sep_in[3], sep_out[3];
    char *sep;

    (void)jdf;
#if !defined(DAGUE_SCHED_DEPS_MASK)
    assert((1<< flow_index) && (((1 << flow_index) & ~DAGUE_DEPENDENCIES_BITMASK) == 0)); 
    (void)flow_index;
#endif

    string_arena_init(sa_dep_in);
    string_arena_init(sa_dep_out);
    
    depname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    condname = (char*)malloc(strlen(prefix) + strlen(flow->varname) + 128);
    sep_in[0] = '\0';
    sep_out[0] = '\0';

    for(depid = 1, dl = flow->deps; dl != NULL; depid++, dl = dl->next) {
        if( dl->type == JDF_DEP_TYPE_IN ) {
            psa = sa_dep_in;
            sep = sep_in;
        }
        else if ( dl->type == JDF_DEP_TYPE_OUT ) {
            psa = sa_dep_out;
            sep = sep_out;
        } else {
            jdf_fatal(dl->lineno, "This dependency is neither a DEP_IN or a DEP_OUT?\n");
            exit(1);
        }

        if( dl->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
            sprintf(depname, "%s%s_dep%d_atline_%d", prefix, flow->varname, depid, dl->lineno);
            sprintf(condname, "NULL");
            jdf_generate_dependency(jdf, dl->datatype_name, dl->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");
        } else if( dl->guard->guard_type == JDF_GUARD_BINARY ) {
            sprintf(depname, "%s%s_dep%d_atline_%d", prefix, flow->varname, depid, dl->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            jdf_generate_expression(jdf, context, dl->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->datatype_name, dl->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");
        } else if( dl->guard->guard_type == JDF_GUARD_TERNARY ) {
            jdf_expr_t not;

            sprintf(depname, "%s%s_dep%d_iftrue_atline_%d", prefix, flow->varname, depid, dl->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            jdf_generate_expression(jdf, context, dl->guard->guard, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->datatype_name, dl->guard->calltrue, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
            sprintf(sep, ", ");

            sprintf(depname, "%s%s_dep%d_iffalse_atline_%d", prefix, flow->varname, depid, dl->lineno);
            sprintf(condname, "expr_of_cond_for_%s", depname);
            not.op = JDF_NOT;
            not.jdf_ua = dl->guard->guard;
            jdf_generate_expression(jdf, context, &not, condname);
            sprintf(condname, "&expr_of_cond_for_%s", depname);
            jdf_generate_dependency(jdf, dl->datatype_name, dl->guard->callfalse, depname, condname, context);
            string_arena_add_string(psa, "%s&%s", sep, depname);
        }
    }
    free(depname);
    free(condname);

    alldeps_type = jdf_dataflow_type(flow);
    sym_type = ( (alldeps_type == JDF_DEP_TYPE_IN) ? "SYM_IN" :
                 ((alldeps_type == JDF_DEP_TYPE_OUT) ? "SYM_OUT" : "SYM_INOUT") );

    access_type = (  (flow->access_type == JDF_VAR_TYPE_CTL) ? "ACCESS_NONE" :
                    ((flow->access_type == JDF_VAR_TYPE_READ) ? "ACCESS_READ" :
                    ((flow->access_type == JDF_VAR_TYPE_WRITE) ? "ACCESS_WRITE" : "ACCESS_RW") ) ); 
    
    if(strlen(string_arena_get_string(sa_dep_in)) == 0) {
        string_arena_add_string(sa_dep_in, "NULL");
    }
    if(strlen(string_arena_get_string(sa_dep_out)) == 0) {
        string_arena_add_string(sa_dep_out, "NULL");
    }

    string_arena_add_string(sa, 
                            "static const dague_flow_t %s%s = {\n"
                            "  .name = \"%s\",\n"
                            "  .sym_type = %s,\n"
                            "  .access_type = %s,\n"
                            "  .flow_index = %u,\n"
                            "  .dep_in  = { %s },\n"
                            "  .dep_out = { %s }\n"
                            "};\n\n", 
                            prefix, flow->varname, 
                            flow->varname, 
                            sym_type, 
                            access_type,
                            flow_index,
                            string_arena_get_string(sa_dep_in),
                            string_arena_get_string(sa_dep_out));
    string_arena_free(sa_dep_in);
    string_arena_free(sa_dep_out);


    coutput("%s", string_arena_get_string(sa));
    string_arena_free(sa);
}

/**
 * Parse the whole dependency and try to figure out if there is any
 * combination that will allow this task (based on its inputs) to be
 * executed as a startup task. In other words, if there is any tuple
 * of the execution space, which leads to all inputs comming directly
 * from the matrix.
 * We should ignore all dependencies that lack an input argument.
 *
 * @Return: If the task cannot be a startup task, then the pint
 *          argument should be set to zero.
 */
static char* has_ready_input_dependency(void **elt, void *pint)
{
    jdf_dataflow_t* flow = (jdf_dataflow_t*)elt;
    jdf_dep_t* dep = flow->deps;
    int can_be_startup = 0, has_input = 0;

    if( JDF_VAR_TYPE_CTL == flow->access_type ) {
        has_input = 1;
        can_be_startup = 1;
        while( NULL != dep ) {
            if( dep->type == JDF_DEP_TYPE_IN ) {
                if( dep->guard->guard_type != JDF_GUARD_BINARY )
                    can_be_startup = 0;
            }
            dep = dep->next;
        }
    } else {  /* This is a data */
        while( NULL != dep ) {
            if( dep->type == JDF_DEP_TYPE_IN ) {
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
            dep = dep->next;
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
    info.sa = sa;
    info.assignments = "assignments";

    while( NULL != dep ) {
        if( dep->type == JDF_DEP_TYPE_IN ) {
            if( dep->guard->guard_type == JDF_GUARD_UNCONDITIONAL ) {
                if( NULL == dep->guard->calltrue->var ) {
                    /* Always */
                }
            }
            if( dep->guard->guard_type == JDF_GUARD_BINARY ) {
                if( (NULL == dep->guard->calltrue->var) ||
                    (flow->access_type == JDF_VAR_TYPE_CTL)) {
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
    if( already_added && (JDF_VAR_TYPE_CTL == flow->access_type) ) {
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
    string_arena_t *sa1, *sa2;
    jdf_def_list_t *dl;
    int nesting;
    expr_info_t info1, info2;
    int idx;

    assert( f->flags & JDF_FUNCTION_FLAG_CAN_BE_STARTUP );
    (void)jdf;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("static int %s(dague_context_t *context, const __dague_%s_internal_handle_t *__dague_handle, dague_execution_context_t** pready_list)\n"
            "{\n"
            "  dague_execution_context_t* new_context;\n"
            "  assignment_t *assignments = NULL;\n"
            "%s\n"
            "%s\n"
            "  new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );\n"
            "  assignments = new_context->locals;\n",
            fname, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name, dump_string, NULL,
                                 "  int32_t ", " ", " = -1,", " = -1;"),
            UTIL_DUMP_LIST_FIELD(sa2, f->definitions, next, name, dump_string, NULL, 
                                 "  ", "(void)", "; ", ";"));

    string_arena_init(sa1);
    string_arena_init(sa2);

    info1.sa = sa1;
    info1.prefix = "";
    info1.assignments = "assignments";

    info2.sa = sa2;
    info2.prefix = "";
    info2.assignments = "assignments";

    coutput("  /* Parse all the inputs and generate the ready execution tasks */\n");

    nesting = 0;
    idx = 0;
    for(dl = f->definitions; dl != NULL; dl = dl->next, idx++) {
        if(dl->expr->op == JDF_RANGE) {
            coutput("%s  for(%s = %s;\n"
                    "%s      %s <= %s;\n"
                    "%s      %s++) {\n"
                    "%s    assignments[%d].value = %s;\n",
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba1, &info1),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba2, &info2),
                    indent(nesting), dl->name,
                    indent(nesting), idx, dl->name);
            nesting++;
        } else {
            coutput("%s  assignments[%d].value = %s = %s;\n",
                    indent(nesting), idx, dl->name, dump_expr((void**)dl->expr, &info1));
        }
    }
    coutput("%s  if( !%s_pred(%s) ) continue;\n",
            indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name,
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

    coutput("%s  DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);\n"
            "%s  DAGUE_LIST_ITEM_SINGLETON( new_context );\n",
            indent(nesting),
            indent(nesting));
    coutput("%s  new_context->dague_handle = (dague_handle_t*)__dague_handle;\n"
            "%s  new_context->function = (const dague_function_t*)&%s_%s;\n",
            indent(nesting),
            indent(nesting), jdf_basename, f->fname);
    if( NULL != f->priority ) {
        coutput("%s  new_context->priority = priority_of_%s_%s_as_expr_fct(new_context->dague_handle, new_context->locals);\n",
                indent(nesting), jdf_basename, f->fname);
    } else {
        coutput("%s  new_context->priority = 0;\n", indent(nesting));
    }
    {
        struct jdf_dataflow *dataflow = f->dataflow;
        for(idx = 0; NULL != dataflow; idx++, dataflow = dataflow->next ) {
            coutput("    new_context->data[%d].data_repo = NULL;\n"
                    "    new_context->data[%d].data      = NULL;\n",
                    idx, idx);
        }
    }
    coutput("#if defined(DAGUE_DEBUG)\n"
            "%s  {\n"
            "%s    char tmp[128];\n"
            "%s    DEBUG((\"Add startup task %%s\\n\",\n"
            "%s           dague_service_to_string(new_context, tmp, 128)));\n"
            "%s  }\n"
            "#endif\n", indent(nesting), indent(nesting), indent(nesting), indent(nesting), indent(nesting));

    coutput("%s  dague_list_add_single_elem_by_priority( pready_list, new_context );\n", indent(nesting));

    coutput("%s  new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );\n"
            "%s  assignments = new_context->locals;\n",
            indent(nesting),
            indent(nesting));
    /* Dump all assignments except the last one */
    for(idx = 0, dl = f->definitions; NULL != dl->next; dl = dl->next, idx++) {
        coutput("%s  assignments[%d].value = %s;\n", 
                indent(nesting), idx, dl->name);
    }

    for(; nesting > 0; nesting--) {
        coutput("%s}\n", indent(nesting));
    }

    /* Quiet the compiler by using the varibales */
    string_arena_free(sa1);    
    string_arena_free(sa2);

    coutput("  dague_thread_mempool_free( context->execution_units[0]->context_mempool, new_context );\n");

    coutput("  return 0;\n"
            "}\n\n");
}

static void jdf_generate_internal_init(const jdf_t *jdf, const jdf_function_entry_t *f, const char *fname)
{
    string_arena_t *sa1, *sa2;
    jdf_def_list_t *dl;
    jdf_name_list_t *pl;
    int nesting, idx;
    const jdf_function_entry_t *pf;
    expr_info_t info1, info2;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    coutput("static int %s(__dague_%s_internal_handle_t *__dague_handle)\n"
            "{\n"
            "  dague_dependencies_t *dep = NULL;\n"
            "  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;\n"
            "  int nb_tasks = 0, __foundone = 0;\n"
            "%s",
            fname, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name, dump_string, NULL,
                                 "  int32_t ", " ", ",", ";\n"));
    coutput("%s"
            "%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_min = 0x7fffffff,", "_min = 0x7fffffff;\n"),
            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_max = 0,", "_max = 0;\n"));
#if 0
    coutput("%s"
            "%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_start,", "_start;\n"),
            UTIL_DUMP_LIST_FIELD(sa2, f->parameters, next, name, dump_string, NULL,
                                 "  int32_t ", " ", "_end,", "_end;\n"));
#endif
    coutput("  (void)__dague_handle; (void)__foundone;\n");
    /* The first definitions are the parameter definitions, and only those */
    if( NULL != f->parameters->next ) {
        for(dl = f->definitions, pl = f->parameters; pl != NULL; dl = dl->next, pl = pl->next ) {
            if(dl->expr->op == JDF_RANGE) {
                coutput("  int32_t %s_start, %s_end;", dl->name, dl->name );
            }
        }
    }

    string_arena_init(sa1);
    string_arena_init(sa2);

    info1.sa = sa1;
    info1.prefix = "";
    info1.assignments = "assignments";

    info2.sa = sa2;
    info2.prefix = "";
    info2.assignments = "assignments";

    coutput("  /* First, find the min and max value for each of the dimensions */\n");

    idx = 0;
    nesting = 0;
    pl = f->parameters;
    for(dl = f->definitions; dl != NULL; dl = dl->next ) {
        if(dl->expr->op == JDF_RANGE) {
            assert( pl != NULL );
            coutput("%s  for(%s = %s;\n"
                    "%s      %s <= %s;\n"
                    "%s      %s++) {\n",
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba1, &info1),
                    indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba2, &info2),
                    indent(nesting), dl->name);
            nesting++;
        } else {
            coutput("%s  %s = %s;\n", 
                    indent(nesting), dl->name, dump_expr((void**)dl->expr, &info1));
        }
        coutput("%s  assignments[%d].value = %s;\n", 
                indent(nesting), idx, dl->name);
        idx++;

        if ( pl != NULL )
            pl = pl->next;
    }

    string_arena_init(sa1);
    coutput("%s  if( !%s_pred(%s) ) continue;\n"
            "%s  nb_tasks++;\n",
            indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name,
                                                            dump_string, NULL, 
                                                            "", "", ", ", ""),
            indent(nesting));
    for(dl = f->definitions, pl = f->parameters; pl != NULL; dl = dl->next, pl = pl->next ) {
        coutput("%s  %s_max = dague_imax(%s_max, %s);\n"
                "%s  %s_min = dague_imin(%s_min, %s);\n",
                indent(nesting), dl->name, dl->name, dl->name,
                indent(nesting), dl->name, dl->name, dl->name);
    }

    for(; nesting > 0; nesting--) {
        coutput("%s}\n", indent(nesting));
    }

    coutput("\n"
            "  /**\n"
            "   * Now, for each of the dimensions, re-iterate on the space,\n"
            "   * and if at least one value is defined, allocate arrays to point\n"
            "   * to it. Array dimensions are defined by the (rough) observation above\n"
            "   **/\n");

    if( f->parameters->next == NULL ) {
        coutput("  if( 0 != nb_tasks ) {\n"
                "    ALLOCATE_DEP_TRACKING(dep, %s_min, %s_max, \"%s\", &symb_%s_%s_%s, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);\n"
                "  }\n",
                f->parameters->name, f->parameters->name, f->parameters->name,
                jdf_basename, f->fname, f->parameters->name);
    } else {
        int last_dimension_is_a_range = 0;

        coutput("  dep = NULL;\n");

        nesting = 0;
        idx = 0;
        for(dl = f->definitions; dl != NULL; dl = dl->next) {
            if( dl->next == NULL ) {
                coutput("%s  __foundone = 0;\n", indent(nesting));
            }
            if(dl->expr->op == JDF_RANGE) {
                last_dimension_is_a_range = 1;
                coutput("%s  %s_start = %s;\n", 
                        indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba1, &info1));
                coutput("%s  %s_end = %s;\n", 
                        indent(nesting), dl->name, dump_expr((void**)dl->expr->jdf_ba2, &info2));
                coutput("%s  for(%s = dague_imax(%s_start, %s_min); %s <= dague_imin(%s_end, %s_max); %s++) {\n",
                        indent(nesting), dl->name, dl->name, dl->name, dl->name, dl->name, dl->name, dl->name);
                nesting++;
            } else {
                last_dimension_is_a_range = 0;
                coutput("%s  %s = %s;\n", 
                        indent(nesting), dl->name, dump_expr((void**)dl->expr, &info1));
            }

            coutput("%s  assignments[%d].value = %s;\n", 
                    indent(nesting), idx, dl->name);
            idx++;
        }

        coutput("%s  if( %s_pred(%s) ) {\n"
                "%s    /* We did find one! Allocate the dependencies array. */\n",
                indent(nesting), f->fname, UTIL_DUMP_LIST_FIELD(sa2, f->definitions, next, name,
                                                                dump_string, NULL, 
                                                                "", "", ", ", ""),
                indent(nesting));

        string_arena_init(sa1);
        string_arena_add_string(sa1, "dep");
        for(dl = f->definitions, pl = f->parameters; pl != NULL; dl = dl->next, pl = pl->next ) {
            coutput("%s  if( %s == NULL ) {\n"
                    "%s    ALLOCATE_DEP_TRACKING(%s, %s_min, %s_max, \"%s\", &symb_%s_%s_%s, %s, %s);\n"
                    "%s  }\n",
                    indent(nesting), string_arena_get_string(sa1),
                    indent(nesting), string_arena_get_string(sa1), dl->name, dl->name, dl->name, 
                                   jdf_basename, f->fname, dl->name,
                                   dl == f->definitions ? "NULL" : string_arena_get_string(sa2),
                                   pl->next == NULL ? "DAGUE_DEPENDENCIES_FLAG_FINAL" : "DAGUE_DEPENDENCIES_FLAG_NEXT",
                    indent(nesting));
            string_arena_init(sa2);
            string_arena_add_string(sa2, "%s", string_arena_get_string(sa1));
            string_arena_add_string(sa1, "->u.next[%s-%s_min]", dl->name, dl->name);
        }
        if( last_dimension_is_a_range )
            coutput("%s    break;\n", indent(nesting));
        coutput("%s  }\n", indent(nesting));
        
        for(; nesting > 0; nesting--) {
            coutput("%s}\n", indent(nesting));
        }
    }

    /* Quiet the compiler by using the varibales */
    if( NULL != f->parameters->next ) {
        for(dl = f->definitions, pl = f->parameters; pl != NULL; dl = dl->next, pl = pl->next ) {
            if(dl->expr->op == JDF_RANGE) {
                coutput("  (void)%s_start; (void)%s_end;", dl->name, dl->name );
            }
        }
    }

    string_arena_free(sa1);    
    string_arena_free(sa2);

    for(nesting = 0, pf = jdf->functions;
        strcmp( pf->fname, f->fname);
        nesting++, pf = pf->next) /* nothing */;

    coutput("  __dague_handle->super.super.dependencies_array[%d] = dep;\n"
            "  __dague_handle->super.super.nb_local_tasks += nb_tasks;\n"
            "  return nb_tasks;\n"
            "}\n"
            "\n",
            nesting);
}

static void jdf_generate_simulation_cost_fct(const jdf_t *jdf, const jdf_function_entry_t *f, const char *prefix)
{
    assignment_info_t ai;
    expr_info_t info;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);

    (void)jdf;

    ai.sa = sa;
    ai.idx = 0;
    ai.holder = "this_task->locals";
    ai.expr = NULL;

    coutput("#if defined(DAGUE_SIM)\n"
            "static int %s(const dague_execution_context_t *this_task)\n"
            "{\n"
            "  const dague_handle_t *__dague_handle = (const dague_handle_t*)this_task->dague_handle;\n"
            "%s"
            "  (void)__dague_handle;\n",
            prefix, UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name, 
                                         dump_assignments, &ai, "", "  int ", ";\n", ";\n"));
    
    string_arena_init(sa);
    coutput("%s",
            UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    string_arena_init(sa);
    info.prefix = "";
    info.sa = sa;
    info.assignments = "this_task->locals";
    coutput("  return %s;\n", dump_expr((void**)f->simcost, &info));
    coutput("}\n"
            "#endif\n"
            "\n");

    string_arena_free(sa);
    string_arena_free(sa1);
}

static void jdf_generate_one_function( const jdf_t *jdf, const jdf_function_entry_t *f, int dep_index )
{
    string_arena_t *sa, *sa2;
    int nbparameters, nbdefinitions;
    int inputmask, nbinput;
    int i, has_in_in_dep, foundin;
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;
    char *prefix;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    JDF_COUNT_LIST_ENTRIES(f->parameters, jdf_name_list_t, next, nbparameters);
    JDF_COUNT_LIST_ENTRIES(f->definitions, jdf_def_list_t, next, nbdefinitions);

    
    inputmask = 0;
    nbinput = 0;
    has_in_in_dep = 0;
    for( fl = f->dataflow; NULL != fl; fl = fl->next ) {

        foundin = 0;
        for( dl = fl->deps; NULL != dl; dl = dl->next ) {
            if( dl->type & JDF_DEP_TYPE_IN ) {
                
                if( JDF_VAR_TYPE_CTL == fl->access_type ) {
                    if( JDF_GUARD_BINARY == dl->guard->guard_type )
                        has_in_in_dep = 1;
                } else {
                    switch( dl->guard->guard_type ) {
                    case JDF_GUARD_TERNARY:
                        if( NULL == dl->guard->callfalse->var )
                            has_in_in_dep = 1;

                    case JDF_GUARD_UNCONDITIONAL:
                    case JDF_GUARD_BINARY:
                        if( NULL == dl->guard->calltrue->var )
                            has_in_in_dep = 1;
                    }
                }
                if( foundin == 0 ) {
                    inputmask |= (1 << nbinput);
                    nbinput++;
                    foundin = 1;
                }
            }
        }
    }

    jdf_coutput_prettycomment('*', "%s", f->fname);
    
    string_arena_add_string(sa, 
                            "static const dague_function_t %s_%s = {\n"
                            "  .name = \"%s\",\n"
                            "  .deps = %d,\n"
                            "  .flags = %s%s,\n"
                            "  .function_id = %d,\n"
#if !defined(DAGUE_SCHED_DEPS_MASK)
                            "  .dependencies_goal = %d,\n"
#else
                            "  .dependencies_goal = 0x%x,\n"
#endif
                            "  .nb_parameters = %d,\n"
                            "  .nb_definitions = %d,\n",
                            jdf_basename, f->fname,
                            f->fname,
                            dep_index,
                            (f->flags & JDF_FUNCTION_FLAG_HIGH_PRIORITY) ? "DAGUE_HIGH_PRIORITY_TASK" : "0x0",
                            has_in_in_dep ? " | DAGUE_HAS_IN_IN_DEPENDENCIES" : "",
                            dep_index,
#if !defined(DAGUE_SCHED_DEPS_MASK)
                            nbinput,
#else
                            inputmask,
#endif
                            nbparameters,
                            nbdefinitions);

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
    string_arena_add_string(sa, "  .pred = &%s,\n", prefix);

    if( NULL != f->priority ) {
        sprintf(prefix, "priority_of_%s_%s_as_expr", jdf_basename, f->fname);
        jdf_generate_expression(jdf, f->definitions, f->priority, prefix);
        string_arena_add_string(sa, "  .priority = &%s,\n", prefix);
    } else {
        string_arena_add_string(sa, "  .priority = NULL,\n");
    }

    sprintf(prefix, "flow_of_%s_%s_for_", jdf_basename, f->fname);
    for(i = 0, fl = f->dataflow; fl != NULL; fl = fl->next, i++) {
        jdf_generate_dataflow(jdf, f->definitions, fl, prefix, (uint32_t)i);
    }
    sprintf(prefix, "&flow_of_%s_%s_for_", jdf_basename, f->fname);
    string_arena_add_string(sa, "  .in = { %s },\n",
                            UTIL_DUMP_LIST(sa2, f->dataflow, next, dump_dataflow, "IN",
                                           "", prefix, ", ", ""));
    string_arena_add_string(sa, "  .out = { %s },\n",
                            UTIL_DUMP_LIST(sa2, f->dataflow, next, dump_dataflow, "OUT",
                                           "", prefix, ", ", ""));

    sprintf(prefix, "iterate_successors_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_iterate_successors(jdf, f, prefix);
    string_arena_add_string(sa, "  .iterate_successors = %s,\n", prefix);

    

    sprintf(prefix, "release_deps_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_release_deps(jdf, f, prefix);
    string_arena_add_string(sa, "  .release_deps = %s,\n", prefix);

    sprintf(prefix, "hook_of_%s_%s", jdf_basename, f->fname);
    jdf_generate_code_hook(jdf, f, prefix);
    string_arena_add_string(sa, "  .hook = %s_task_create,\n", prefix);
    string_arena_add_string(sa, "  .complete_execution = complete_%s,\n", prefix);
    
    if( NULL != f->simcost ) {
        sprintf(prefix, "simulation_cost_of_%s_%s", jdf_basename, f->fname);
        jdf_generate_simulation_cost_fct(jdf, f, prefix);
        string_arena_add_string(sa, 
                                "#if defined(DAGUE_SIM)\n"
                                "  .sim_cost_fct = %s,\n"
                                "#endif\n", prefix);
    } else {
        string_arena_add_string(sa, 
                                "#if defined(DAGUE_SIM)\n"
                                "  .sim_cost_fct = NULL,\n"
                                "#endif\n");
    }

    string_arena_add_string(sa, "  .key = (dague_functionkey_fn_t*)%s_hash,\n", f->fname);

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
    for(i = 0, f = jdf->functions; NULL != f; f = f->next, i++) {
        jdf_generate_one_function(jdf, f, i);
        string_arena_add_string(sa, "  &%s_%s%s\n", 
                                jdf_basename, f->fname, f->next != NULL ? "," : "");
    }
    string_arena_add_string(sa, "};\n\n");
    coutput("%s", string_arena_get_string(sa));

    string_arena_free(sa);
}

static char *dump_pseudodague(void **elem, void *arg)
{
    string_arena_t *sa = (string_arena_t *)arg;
    char *name = *(char**)elem;
    string_arena_init(sa);
    string_arena_add_string(sa,
                            "static const dague_function_t %s_%s = {\n"
                            "  .name = \"%s\",\n"
                            "  .flags = 0x0,\n"
                            "  .dependencies_goal = 0x0,\n"
                            "  .nb_parameters = 0,\n"
                            "  .nb_definitions = 0,\n"
                            "  .params = { NULL, },\n"
                            "  .locals = { NULL, },\n"
                            "  .pred = NULL,\n"
                            "  .in = { NULL, },\n"
                            "  .out = { NULL, },\n"
                            "  .priority = NULL,\n"
                            "  .deps = -1,\n"
                            "  .hook = NULL,\n"
                            "  .release_deps = NULL,\n"
                            "  .body = NULL,\n"
                            "#if defined(DAGUE_SCHED_CACHE_AWARE)\n"
                            "  .cache_rank_function = NULL,\n"
                            "#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */\n"
                            "};\n",
                            jdf_basename, name, name);
    return string_arena_get_string(sa);
}

static void jdf_generate_predeclarations( const jdf_t *jdf )
{
    jdf_function_entry_t *f;
    jdf_dataflow_t *fl;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    


    houtput("#ifdef __cplusplus\n"
	    "extern \"C\" {\n"
	    "#endif\n");
     
    coutput("/** Predeclarations of the dague_function_t objects */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
	houtput("void hook_of_%s_%s_callback_function(dague_execution_unit_t*, dague_execution_context_t*);\n", jdf_basename, f->fname);
	coutput("static int hook_of_%s_%s_task_create(dague_execution_unit_t*, dague_execution_context_t*);\n",
		jdf_basename, f->fname); 
	coutput("static void func_cpu_hook_of_%s_%s(void* buffers[], void* cl_arg);\n",
		jdf_basename, f->fname); 
	if(f->body_gpu != NULL)
	    coutput("extern void hook_of_%s_%s_cuda_wrapper(void* buffers[], void* cl_arg);\n", 
		    jdf_basename, f->fname); 
        coutput("static const dague_function_t %s_%s;\n", jdf_basename, f->fname);
        if( NULL != f->priority ) {
            coutput("static inline int priority_of_%s_%s_as_expr_fct(const dague_handle_t *__dague_handle_parent, const assignment_t *assignments);\n", 
                    jdf_basename, f->fname);
        }
    }

    houtput("#ifdef __cplusplus\n"
	    "}\n"
	    "#endif\n");


    coutput("/** Declarations of the pseudo-dague_function_t objects for data */\n"
            "%s\n",
            UTIL_DUMP_LIST_FIELD(sa2, jdf->data, next, dname,
                                 dump_pseudodague, sa, "", "", "", ""));
    string_arena_free(sa);
    string_arena_free(sa2);
    coutput("/** Predeclarations of the parameters */\n");
    for(f = jdf->functions; f != NULL; f = f->next) {
        for(fl = f->dataflow; fl != NULL; fl = fl->next) {

            coutput("static const dague_flow_t flow_of_%s_%s_for_%s;\n", 
                    jdf_basename, f->fname, fl->varname);
        }
    }
}

static void jdf_generate_startup_hook( const jdf_t *jdf )
{
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);

    coutput("static void %s_startup(dague_context_t *context, dague_handle_t *dague_handle, dague_execution_context_t** pready_list)\n"
            "{\n"
            "%s\n"
            "}\n",
            jdf_basename, 
            UTIL_DUMP_LIST( sa1, jdf->functions, next, dump_startup_call, sa2,
                            "  ", jdf_basename, "\n  ", "") );

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_destructor( const jdf_t *jdf )
{
    struct jdf_name_list* g;
    string_arena_t *sa = string_arena_new(64);

    coutput("void dague_%s_destroy( dague_%s_handle_t *o )\n"
            "{\n"
            "  dague_handle_t *d = (dague_handle_t *)o;\n"
            "  __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t*)o; (void)__dague_handle;\n"
            "  int i;\n",
            jdf_basename, jdf_basename,
            jdf_basename,
            jdf_basename);

    coutput("  free(d->functions_array);\n"
            "  d->functions_array = NULL;\n"
            "  d->nb_functions = 0;\n");

    for( g = jdf->datatypes; NULL != g; g = g->next ) {
        coutput("  if( o->arenas[DAGUE_%s_%s_ARENA] != NULL ) {\n"
                "    dague_arena_destruct(o->arenas[DAGUE_%s_%s_ARENA]);\n"
                "    free(o->arenas[DAGUE_%s_%s_ARENA]);\n"
                "  }\n",
                jdf_basename, g->name,
                jdf_basename, g->name,
                jdf_basename, g->name);
    }

    coutput("  /* Destroy the data repositories for this object */\n");
    {
        jdf_function_entry_t* f;

        for( f = jdf->functions; NULL != f; f = f->next ) {
            if( 0 != function_has_data_output(f) )
                coutput("   data_repo_destroy_nothreadsafe(__dague_handle->%s_repository);\n",
                        f->fname);
        }
    }

    coutput("  for(i = 0; i < DAGUE_%s_NB_FUNCTIONS; i++)\n"
            "    dague_destruct_dependencies( d->dependencies_array[i] );\n"
            "  free( d->dependencies_array );\n",
            jdf_basename);

    coutput("  free(o);\n");

    coutput("}\n"
            "\n");

    string_arena_free(sa);
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

    coutput("  __dague_%s_internal_handle_t *_res = (__dague_%s_internal_handle_t *)calloc(1, sizeof(__dague_%s_internal_handle_t));\n",
            jdf_basename, jdf_basename, jdf_basename);

    string_arena_init(sa1);
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD( sa1, jdf->functions, next, fname,
                                  dump_string, NULL, "", "  int ", "_nblocal_tasks;\n", "_nblocal_tasks;\n") );

    coutput("  _res->super.super.nb_functions    = DAGUE_%s_NB_FUNCTIONS;\n", jdf_basename);
    coutput("  _res->super.super.functions_array = (const dague_function_t**)malloc(DAGUE_%s_NB_FUNCTIONS * sizeof(dague_function_t*));\n",
            jdf_basename);
    coutput("  _res->super.super.dependencies_array = (dague_dependencies_t **)\n"
            "              calloc(DAGUE_%s_NB_FUNCTIONS, sizeof(dague_dependencies_t *));\n",
            jdf_basename);
    coutput("  memcpy(_res->super.super.functions_array, %s_functions, DAGUE_%s_NB_FUNCTIONS * sizeof(dague_function_t*));\n",
            jdf_basename, jdf_basename);
    {
        struct jdf_name_list* g;

        for( g = jdf->datatypes; NULL != g; g = g->next ) {
            coutput("  _res->super.arenas[DAGUE_%s_%s_ARENA] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));\n",
                    jdf_basename, g->name);
        }
    }

    coutput("  /* Now the Parameter-dependent structures: */\n"
            "%s", UTIL_DUMP_LIST(sa1, jdf->globals, next,
                                 dump_globals_init, sa2, "", "  ", "\n", "\n"));

    pi.sa = sa2;
    pi.idx = 0;
    JDF_COUNT_LIST_ENTRIES(jdf->functions, jdf_function_entry_t, next, pi.maxidx);
    coutput("  /* If profiling is enabled, the keys for profiling */\n"
            "#  if defined(DAGUE_PROF_TRACE)\n"
            "  _res->super.super.profiling_array = %s_profiling_array;\n"
            "  if( -1 == %s_profiling_array[0] ) {\n"
            "%s"
            "  }\n"
            "#  endif /* defined(DAGUE_PROF_TRACE) */\n", 
            jdf_basename,
            jdf_basename,
            UTIL_DUMP_LIST( sa1, jdf->functions, next,
                            dump_profiling_init, &pi, "", "    ", "\n", "\n"));

    coutput("  /* Create the data repositories for this object */\n"
            "%s",
            UTIL_DUMP_LIST( sa1, jdf->functions, next, dump_data_repository_constructor, sa2,
                            "", "", "\n", "\n"));

    coutput("  _res->super.super.startup_hook = %s_startup;\n"
            "  (void)dague_handle_register((dague_handle_t*)_res);\n"
            "  return (dague_%s_handle_t*)_res;\n"
            "}\n\n",
            jdf_basename, jdf_basename);

    string_arena_free(sa1);
    string_arena_free(sa2);
}

static void jdf_generate_hashfunction_for(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    string_arena_t *sa = string_arena_new(64);
    jdf_def_list_t *dl;
    jdf_name_list_t *pl;
    expr_info_t info;
    int idx;

    (void)jdf;

    coutput("static inline uint64_t %s_hash(const __dague_%s_internal_handle_t *__dague_handle, const assignment_t *assignments)\n"
            "{\n"
            "  uint64_t __h = 0;\n"
            "  (void)__dague_handle;\n",
            f->fname, jdf_basename);

    info.prefix = "";
    info.sa = sa;
    info.assignments = "assignments";

    pl = f->parameters;
    idx = 0;
    for(dl = f->definitions; dl != NULL; dl = dl->next) {
        string_arena_init(sa);
        coutput("  int %s = assignments[%d].value;\n",
                dl->name, idx);
        idx++;

        if( !strcmp(dl->name, pl->name) ) {
            if( dl->expr->op == JDF_RANGE ) {
                coutput("  int %s_min = %s;\n", dl->name, dump_expr((void**)dl->expr->jdf_ba1, &info));
                if( pl->next != NULL ) {
                    coutput("  int %s_range = %s - %s_min + 1;\n", 
                            dl->name, dump_expr((void**)dl->expr->jdf_ba2, &info), dl->name);
                }
            } else {
                coutput("  int %s_min = %s;\n", dl->name, dump_expr((void**)dl->expr, &info));
                if( dl->next != NULL ) {
                    coutput("  int %s_range = 1;\n", dl->name);
                }
            }
        }

        /* Hash functions depends only on the parameters of the function.
         * It is not necessary to define things after the last parameter
         */
        if( !strcmp(dl->name, pl->name) ) {
            pl = pl->next;
            if( NULL == pl )
                break;
        }
    }

    string_arena_init(sa);
    for(pl = f->parameters; pl != NULL; pl = pl->next) {
        coutput("  __h += (%s - %s_min)%s;\n",pl->name, pl->name, string_arena_get_string(sa));
        string_arena_add_string(sa, " * %s_range", pl->name);
    }

    coutput("  return __h;\n");
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
    info.assignments = "assignments";

    UTIL_DUMP_LIST(sa, el, next,
                   dump_expr, (void**)&info, "", "", ", ", "");
    res = strdup( string_arena_get_string(sa) );
    string_arena_free(sa);
    string_arena_free(sa2);

    return res;
}

/** Code Generators */

static char *jdf_create_code_assignments_calls(string_arena_t *sa, int spaces,
                                               const jdf_t *jdf, const char *name, const jdf_expr_t *param)
{
  int idx = 0;
  const jdf_expr_t *el;
  expr_info_t info;
  string_arena_t *sa2;

  (void)jdf;

  string_arena_init(sa);
  sa2 = string_arena_new(64);

  info.sa = sa2;
  info.prefix = "";
  info.assignments = "assignments";

  for(el = param; NULL != el; el = el->next) {
    string_arena_init(sa2);
    string_arena_add_string(sa, "%s  %s[%d].value = %s;\n",
                            indent(spaces), name, idx, 
                            dump_expr((void**)el, &info));
    idx++;
  }

  string_arena_free(sa2);

  return string_arena_get_string(sa);
}

static void jdf_generate_code_call_initialization(const jdf_t *jdf, const jdf_call_t *call, 
                                                  int lineno, const char *fname, const jdf_dataflow_t *f,
                                                  const char *spaces)
{
    string_arena_t *sa, *sa2;
    expr_info_t info;
    int dataindex;

    sa = string_arena_new(64);
    sa2 = string_arena_new(64);

    info.sa = sa2;
    info.prefix = "";
    info.assignments = "assignments";

    if( call->var != NULL ) {
        dataindex = jdf_data_output_index(jdf, call->func_or_mem,
                                          call->var);
        if( dataindex < 0 ) {
            if( dataindex == -1 ) {
                jdf_fatal(lineno, 
                          "During code generation: unable to find an output flow for variable %s in function %s,\n"
                          "which is requested by function %s to satisfy Input dependency at line %d\n",
                          call->var, call->func_or_mem,
                          fname, lineno);
                exit(1);
            } else {
                jdf_fatal(lineno, 
                          "During code generation: unable to find function %s,\n"
                          "which is requested by function %s to satisfy Input dependency at line %d\n",
                          call->func_or_mem,
                          fname, lineno);
                exit(1);
            }
        }
        coutput("%s",  jdf_create_code_assignments_calls(sa, strlen(spaces), jdf, "tass", call->parameters));
        coutput("%s  e%s = data_repo_lookup_entry( %s_repo, %s_hash( __dague_handle, tass ));\n"
                "%s  g%s = e%s->data[%d];\n",
                spaces, f->varname, call->func_or_mem, call->func_or_mem, 
                spaces, f->varname, f->varname, dataindex);
    } else {
        coutput("%s  g%s = (dague_arena_chunk_t*) %s(%s);\n",
                spaces, f->varname, call->func_or_mem,
                UTIL_DUMP_LIST(sa, call->parameters, next,
                               dump_expr, (void**)&info, "", "", ", ", ""));
    }

    string_arena_free(sa);
    string_arena_free(sa2);
}

static void
jdf_generate_code_flow_retrieving(const jdf_dataflow_t *flow, uint32_t flow_index)
{
    coutput("    void *%s =  get_data_handle_of(buffers[%d]); (void) %s;\n",
	    flow->varname, flow_index, flow->varname);
}

static void
jdf_generate_cuda_code_flow_retrieving(const jdf_dataflow_t *flow, uint32_t flow_index)
{
    cudaoutput("    void *%s =  get_data_handle_of(buffers[%d]); (void) %s;\n",
	       flow->varname, flow_index, flow->varname);
}




static void jdf_generate_code_flow_initialization(const jdf_t *jdf,
                                                  const char *fname,
                                                  const jdf_dataflow_t *flow,
                                                  uint32_t flow_index)
{
    jdf_dep_t *dl;
    expr_info_t info;
    string_arena_t *sa;
    int cond_index = 0;
    char* condition[] = {"  if( %s ) {\n", "  else if( %s ) {\n"};

    if( JDF_VAR_TYPE_CTL == flow->access_type ) {
        coutput("    /* this_task->data[%u] is a control flow */\n", flow_index);
        return;
    }
    coutput( "  e%s = this_task->data[%u].data_repo;\n"
             "  g%s = this_task->data[%u].data;\n"
             "  if( NULL == g%s ) {\n",
             flow->varname, flow_index,
             flow->varname, flow_index,
             flow->varname);
    sa = string_arena_new(64);
    info.sa = sa;
    info.prefix = "";
    info.assignments = "  this_task->locals";

    for(dl = flow->deps; dl != NULL; dl = dl->next) {
        if( dl->type == JDF_DEP_TYPE_OUT )
            /** No initialization for output-only flows */
            continue;

        switch( dl->guard->guard_type ) {
        case JDF_GUARD_UNCONDITIONAL:
            if( 0 != cond_index ) coutput("  else {\n");
            jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, flow->lineno, fname, flow,
                                                   (0 != cond_index ? "  " : "") );
            if( 0 != cond_index ) coutput("  }\n");
            goto done_with_input;
        case JDF_GUARD_BINARY:
            coutput( (0 == cond_index ? condition[0] : condition[1]),
                     dump_expr((void**)dl->guard->guard, &info));
            jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, flow->lineno, fname, flow, "  " );
            coutput("  }\n");
            cond_index++;
            break;
        case JDF_GUARD_TERNARY:
            coutput( (0 == cond_index ? condition[0] : condition[1]),
                     dump_expr((void**)dl->guard->guard, &info));
            jdf_generate_code_call_initialization( jdf, dl->guard->calltrue, flow->lineno, fname, flow, "  " );
            coutput("  } else {\n");
            jdf_generate_code_call_initialization( jdf, dl->guard->callfalse, flow->lineno, fname, flow, "  " );
            coutput("  }\n");
            goto done_with_input;
        }
    }

 done_with_input:
    coutput("    this_task->data[%u].data = g%s;\n"
            "    this_task->data[%u].data_repo = e%s;\n"
            "  }\n"
            "  %s = (starpu_data_handle_t*)ADATA(g%s);\n",
            flow_index, flow->varname,
            flow_index, flow->varname,
            flow->varname, flow->varname);
    string_arena_free(sa);
}

/* static void jdf_generate_code_call_final_write(const jdf_t *jdf, const jdf_call_t *call, const char* datatype_name, */
/*                                                const char *spaces, */
/*                                                int dataflow_index) */
/* { */
/*     string_arena_t *sa, *sa2; */
/*     expr_info_t info; */

/*     (void)jdf; */

/*     sa = string_arena_new(64); */
/*     sa2 = string_arena_new(64); */

/*     info.sa = sa2; */
/*     info.prefix = ""; */
/*     info.assignments = "assignments"; */

/*     if( call->var == NULL ) { */
/*         UTIL_DUMP_LIST(sa, call->parameters, next, */
/*                        dump_expr, (void**)&info, "", "", ", ", ""); */
/*         coutput("%s  if( ADATA(this_task->data[%d].data) != %s(%s) ) {\n" */
/*                 "%s    dague_remote_dep_memcpy( context, this_task->dague_handle, %s(%s), this_task->data[%d].data, \n" */
/*                 "%s                             __dague_handle->super.arenas[DAGUE_%s_%s_ARENA]->opaque_dtt );\n" */
/*                 "%s  }\n",                 */
/*                 spaces, dataflow_index, call->func_or_mem, string_arena_get_string(sa), */
/*                 spaces, call->func_or_mem, string_arena_get_string(sa), dataflow_index,  */
/*                 spaces, jdf_basename, datatype_name, */
/*                 spaces); */
/*     } */

/*     string_arena_free(sa); */
/*     string_arena_free(sa2); */
/* } */

/* static void jdf_generate_code_flow_final_writes(const jdf_t *jdf, const jdf_dataflow_t *f, int dataflow_index) */
/* { */
/*     jdf_dep_t *dl; */
/*     expr_info_t info; */
/*     string_arena_t *sa; */

/*     (void)jdf; */

/*     sa = string_arena_new(64); */
/*     info.sa = sa; */
/*     info.prefix = ""; */
/*     info.assignments = "assignments"; */

/*     for(dl = f->deps; dl != NULL; dl = dl->next) { */
/*         if( dl->type == JDF_DEP_TYPE_IN ) */
/*             /\** No final write for input-only flows *\/ */
/*             continue; */

/*         switch( dl->guard->guard_type ) { */
/*         case JDF_GUARD_UNCONDITIONAL: */
/*             if( dl->guard->calltrue->var == NULL ) { */
/*                 jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype_name, "", dataflow_index ); */
/*             } */
/*             break; */
/*         case JDF_GUARD_BINARY: */
/*             if( dl->guard->calltrue->var == NULL ) { */
/*                 coutput("  if( %s ) {\n", */
/*                         dump_expr((void**)dl->guard->guard, &info)); */
/*                 jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype_name, "  ", dataflow_index ); */
/*                 coutput("  }\n"); */
/*             } */
/*             break; */
/*         case JDF_GUARD_TERNARY: */
/*             if( dl->guard->calltrue->var == NULL ) { */
/*                 coutput("  if( %s ) {\n", */
/*                         dump_expr((void**)dl->guard->guard, &info)); */
/*                 jdf_generate_code_call_final_write( jdf, dl->guard->calltrue, dl->datatype_name, "  ", dataflow_index ); */
/*                 if( dl->guard->callfalse->var == NULL ) { */
/*                     coutput("  } else {\n"); */
/*                     jdf_generate_code_call_final_write( jdf, dl->guard->callfalse, dl->datatype_name, "  ", dataflow_index ); */
/*                 } */
/*                 coutput("  }\n"); */
/*             } else if ( dl->guard->callfalse->var == NULL ) { */
/*                 coutput("  if( !(%s) ) {\n", */
/*                         dump_expr((void**)dl->guard->guard, &info)); */
/*                 jdf_generate_code_call_final_write( jdf, dl->guard->callfalse, dl->datatype_name, "  ", dataflow_index ); */
/*                 coutput("  }\n"); */
/*             } */
/*             break; */
/*         } */
/*     } */

/*     string_arena_free(sa); */
/* } */

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


static void jdf_generate_code_papi_events_before(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    (void)jdf;
    (void)f;

    coutput("  /** PAPI events */\n"
	    "#if defined(HAVE_PAPI)\n"
	    "  papime_start_thread_counters();\n"
	    "#endif\n");
}

static void jdf_generate_code_papi_events_after(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    (void)jdf;
    (void)f;

    coutput("  /** PAPI events */\n"
            "#if defined(HAVE_PAPI)\n"
	    "  papime_stop_thread_counters();\n"
	    "#endif\n");
}

static void jdf_generate_code_grapher_task_done(const jdf_t *jdf, const jdf_function_entry_t *f, const char* context_name)
{
    (void)jdf;

    coutput("  dague_prof_grapher_task(%s, context->eu_id, %s_hash(__dague_handle, %s->locals));\n",
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
    (void)jdf;

    coutput("  release_deps_of_%s_%s(context, %s,\n"
            "      DAGUE_ACTION_RELEASE_REMOTE_DEPS |\n"
            "      DAGUE_ACTION_RELEASE_LOCAL_DEPS |\n"
            "      DAGUE_ACTION_RELEASE_LOCAL_REFS |\n"
            "      DAGUE_ACTION_DEPS_MASK,\n"
            "      NULL);\n",
            jdf_basename, function->fname, context_name);
}

static int jdf_property_get_int( const jdf_def_list_t* properties, const char* prop_name, int ret_if_not_found )
{
    jdf_def_list_t* property;
    jdf_expr_t* expr = jdf_find_property(properties, prop_name, &property);

    if( NULL != expr ) {
        if( JDF_CST == expr->op )
            return expr->jdf_cst;
        printf("Warning: property %s defined at line %d only support ON/OFF\n",
               prop_name, property->lineno);
    }
    return ret_if_not_found;  /* ON by default */
}

static void
jdf_generate_code_generic_scheduling_func(const jdf_t *jdf)
{
    jdf_function_entry_t *f;

    coutput("void generic_scheduling_func(dague_execution_unit_t *context, dague_list_item_t *elt)\n"
	    "{\n");


    coutput("    dague_execution_context_t *task2exec;\n"
	    "    dague_list_item_t *next = elt;\n"
	    "    while( NULL != elt ) {\n"
	    "/* Assume that we're going to push elt.\n"
	    "* Remove the first element from the list, keeping the rest of the list in next\n"
	    "*/\n"
	    "        next = (dague_list_item_t *)elt->list_next;\n"
	    "        if(next == elt) {\n"
            "            next = NULL;\n"
	    "        } else {\n"
            "            elt->list_next->list_prev = elt->list_prev;\n"
            "            elt->list_prev->list_next = elt->list_next;\n"
            "            elt->list_prev = elt;\n"
            "            elt->list_next = elt;\n"
            "        }\n"
	    "        task2exec = (dague_execution_context_t*) elt;\n"	    
	    "        elt = next;\n");
	    
    // Liste des fonctions :
    for(f = jdf->functions; f != NULL; f = f->next)
    {
	coutput("        if(strcmp((task2exec->function)->name, \"%s\") == 0) hook_of_%s_%s_task_create(context, task2exec);\n", f->fname, jdf_basename, f->fname);
    }
    coutput("        elt = next;\n"
	    "    }\n");
    coutput("}\n\n");
}


static void
jdf_generate_code_callback_function(const char *name)
{
    coutput("void %s_callback_function(dague_execution_unit_t *exec_unit, dague_execution_context_t *exec_context)\n"
	    "{\n"
	    "    //fprintf(stderr, \"Début de la fonction callback %s\\n\");\n"
	    "    dague_handle_t *dague_handle = exec_context->dague_handle;\n"
	    "    int rc = 0;\n"
	    "    if( NULL != exec_context->function->complete_execution )\n"
	    "        rc = exec_context->function->complete_execution( exec_unit, exec_context );\n\n"
	    "    if( 0 ==  dague_atomic_dec_32b( &(dague_handle->nb_local_tasks) )){\n"
	    "        /* A dague object has been completed. Call the attached callback if\n"    
	    "         * necessary, then update the main engine.*/ \n"
	    "        if( NULL != dague_handle->complete_cb ) {\n"
            "            (void)dague_handle->complete_cb( dague_handle, dague_handle->complete_cb_data );\n"
	    "        }\n"
	    "        dague_atomic_dec_32b( &(exec_unit->master_context->active_objects) );\n"
	    "    }\n"
	    "    DEBUG_MARK_EXE( eu_context->eu_id, exec_context );\n"
	    "    DAGUE_STAT_DECREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);\n"
	    "    dague_thread_mempool_free( exec_unit->context_mempool, exec_context );\n"
	    "    //fprintf(stderr, \"FIN de la fonction callback %s\\n\");\n"
	    "}\n\n", 
	    name, name, name);
}


static void 
jdf_generate_code_starpu_task_create(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    string_arena_t *sa, *sa2;
    assignment_info_t ai;
    int profile_on;
    int di;
    jdf_dataflow_t *fl;
    int num_handle = 0;
    (void) jdf;

    /* If the function has the property profile turned off do not generate the profiling code */
    profile_on = jdf_property_get_int(f->properties, "profile", 1);

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.idx = 0;
    ai.holder = "this_task->locals";
    ai.expr = NULL;
    coutput("static int %s_task_create(dague_execution_unit_t *context, dague_execution_context_t *this_task)\n"
            "{\n"
	    "   // fprintf(stderr, \"début de la fonction task_create %s\\n\");\n"
	    "    struct starpu_task *task = starpu_task_create();\n"
//	    "    __dague_%s_internal_handle_t *obj_temp = malloc(sizeof(struct __dague_%s_internal_handle));\n"
	    /* "    struct callback_args *callback_arg = malloc(sizeof(struct callback_args));\n" */
	    "    struct func_args *arg = malloc(sizeof(struct func_args));\n"
	    /* "    callback_arg->exec_unit     = context;\n" */
	    /* "    callback_arg->exec_context  = this_task;\n" */
            "    const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle; (void) __dague_handle;\n"
            "    assignment_t tass[MAX_PARAM_COUNT];\n"
            "    (void)context; (void)__dague_handle; (void)tass;\n"
            "#if defined(DAGUE_SIM)\n"
            "    int __dague_simulation_date = 0;\n"
            "#endif\n"
	    "/* Registering local params  */\n"
	    "%s",
            name, name, jdf_basename, jdf_basename,// jdf_basename, 
            UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, 
                                 dump_registering_param, &ai, "", "    ", ";\n", ";\n"));
    ai.idx = 0;
    coutput("\n\n/* Local Variables */ \n"
	    "%s",
	    UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, 
                                 dump_assignments, &ai, "", "    int ", ";\n", ";\n"));
    

    coutput("/* Registering global variables for cuda codelet */ \n"
	    "%s\n", dump_registering_global(sa, jdf->globals));
       
    /* coutput("\n /\* Declare the variables that will hold the data *\/ \n" */
    /* 	    "%s\n", */
    /* 	    dump_data_holding_decl(f->dataflow, sa)); */
    {
        char* output = UTIL_DUMP_LIST(sa, f->dataflow, next,
                                      dump_data_declaration, sa2, "", "", "", "");
        if( 0 != strlen(output) )
            coutput("  /** Declare the variables that will hold the data, and all the accounting for each */\n"
                    "%s\n",
                    output);
    }
    
    for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) {
	
        if(fl->access_type == JDF_VAR_TYPE_CTL) continue;  /* control flow, nothing to store */
	
	jdf_generate_code_flow_initialization(jdf, f->fname, fl, di);
	
//	coutput("    starpu_data_set_sequential_consistency_flag(*(%s), 0);\n", fl->varname);
	coutput("    task->handles[%d] = *(%s);\n\n", num_handle++, fl->varname);
	
	coutput("#if defined(DAGUE_SIM)\n"
                "  if( (NULL != e%s) && (e%s->sim_exec_date > __dague_simulation_date) )\n"
                "    __dague_simulation_date =  e%s->sim_exec_date;\n"
                "#endif\n",
                fl->varname,
                fl->varname,
                fl->varname);
    }

//    coutput("    arg->obj = (void*) obj_temp;\n");
    coutput("    arg->obj = (void*) __dague_handle;\n");
  
    coutput("    /* Definition of the task */\n"
	    "    task->cl = &%s_cl;\n"
	    "    task->cl_arg = (void*) arg;\n"
	    "    task->cl_arg_size = sizeof(struct func_args);\n\n", name);
    

    /* coutput("    /\* Global variables must be saved *\/ \n" */
    /* 	    "    memcpy(obj_temp, __dague_handle, sizeof(struct __dague_%s_internal_handle));\n", */
    /* 	    jdf_basename); */

    /* coutput("    /\* Callback function for task completion *\/ \n" */
    /* 	    "    task->callback_func = %s_callback_function;\n" */
    /* 	    "    task->callback_arg  = (void*) callback_arg;\n", */
    /* 	    name); */

    coutput("    /* Task submit */\n"
	    "    if(starpu_task_submit(task) == -ENODEV)\n"
	    "    {\n"
	    "        fprintf(stderr, \"Task'll not execute.\\n\");\n"
	    "        return 1;\n"
	    "    }\n"
	    "    %s_callback_function(context, this_task);\n"
	    "    //fprintf(stderr, \"FIN de la fonction task_create %s\\n\");\n"
	    "    return 0;\n"
	    "}\n\n", name, name);
    
}

static void
jdf_generate_code_gpu_function(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    (void) jdf;
    (void) f;
    (void) name;

    string_arena_t *sa, *sa2, *sa3, *sa4, *sa5;
    expr_info_t linfo;
    int di;
    jdf_dataflow_t *fl;
    assignment_info_t ai;
    int profile_on = jdf_property_get_int(f->properties, "profile", 1);
    sa = string_arena_new(64);
    sa2 =  string_arena_new(64);
    sa4 = string_arena_new(64);
    sa5 = string_arena_new(64);

//    typed_globals_info_t prop = { sa2, NULL, NULL };


    /* cudaoutput("static __global__ void %s_cuda(%s)\n" */
    /* 	       "{\n", name, dump_cuda_args_proto(f, sa)); */
			
    /* cudaoutput("%s\n", dump_cuda_args_void(f, sa)); */
    /* cudaoutput("%s\n", f->body_gpu); */
    /* cudaoutput("}\n"); */

    
    
    cudaoutput("extern \"C\" void %s_cuda_wrapper(void *buffers[], void *cl_arg)\n"
	       "{\n"
	       "    struct func_args *args = (struct func_args*) cl_arg;\n",
	       name);
    
    cudaoutput("/* Retrieving global variables */\n");

    /* UTIL_DUMP_LIST(sa4, jdf->globals, next, */
    /*                dump_globals, sa5, "", "    ", "\n", "\n"); */
    
    /* if(strlen(string_arena_get_string(sa4)) > 1) */
    /* 	cudaoutput("%s\n", string_arena_get_string(sa4)); */

    cudaoutput("%s\n", dump_retrieving_global(sa, jdf->globals));


    cudaoutput("\n\n /* Parameters */ \n");
    
    for(di = 0, fl = f->dataflow; fl != NULL; fl = fl->next)
    {
	if(fl->access_type == JDF_VAR_TYPE_CTL) continue;
	
	jdf_generate_cuda_code_flow_retrieving(fl, di++);

    }    
    ai.idx = 0;
    cudaoutput(" /* Local variables */ \n"
	    "%s\n\n",
	    UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, 
				 dump_retrieving_param, &ai, "", "    int ", ";\n", ";\n"));
    


    // EXPRESSION DU BODY
    /* jdf_generate_code_papi_events_before(jdf, f); */
    /* jdf_generate_code_cache_awareness_update(jdf, f); */


    /* jdf_generate_code_dry_run_before(jdf, f); */
//    jdf_cudaoutput_prettycomment('-', "%s BODY", f->fname);

    if( profile_on ) {
        sa3 = string_arena_new(64);
        linfo.prefix = "";
        linfo.sa = sa2;
        linfo.assignments = "this_task->locals";

        /* cudaoutput("  TAKE_TIME(context, 2*this_task->function->function_id, %s_hash( __dague_handle, this_task->locals), __dague_handle->super.%s, ((dague_ddesc_t*)(__dague_handle->super.%s))->data_key((dague_ddesc_t*)__dague_handle->super.%s, %s) );\n", */
        /*         f->fname, */
        /*         f->predicate->func_or_mem, f->predicate->func_or_mem, f->predicate->func_or_mem, */
        /*         UTIL_DUMP_LIST(sa3, f->predicate->parameters, next, */
        /*                        dump_expr, (void**)&linfo, */
        /*                        "", "", ", ", "") ); */
        string_arena_free(sa3);
    }

    cudaoutput("%s\n", f->body_gpu);


//    cudaoutput("%s\n", f->body_gpu);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
        cudaoutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }
//    jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname);
    /* jdf_generate_code_dry_run_after(jdf, f); */

    ai.idx = 0;

    string_arena_free(sa);
    string_arena_free(sa2);

    /* cudaoutput("    %s_callback_function((void*)(args->callback));\n", name); */
    cudaoutput("    free(cl_arg);\n"
	       "}\n\n");
}


static void
jdf_generate_code_cpu_function(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    (void) jdf;
    (void) f;
    (void) name;

    string_arena_t *sa, *sa2, *sa3;
    expr_info_t linfo;
    int di;
    jdf_dataflow_t *fl;
    assignment_info_t ai;
    int profile_on = jdf_property_get_int(f->properties, "profile", 1);


    sa = string_arena_new(64);
    sa2 =  string_arena_new(64);


    coutput("static void func_cpu_%s(void *buffers[], void *cl_arg)"
            "{\n"
	    "    assignment_t tass[MAX_PARAM_COUNT];\n"
	    "    (void) tass;\n"
	    "    struct func_args *args = (struct func_args*) cl_arg;\n"
	    "    const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)(args->obj);\n"
	    "    (void) __dague_handle;\n"
	    "#if defined(DAGUE_SIM)\n"
            "  int __dague_simulation_date = 0;\n"
            "#endif\n",
	    name, jdf_basename, jdf_basename);

    coutput("\n\n /* Parameters */ \n");
    
    for(di = 0, fl = f->dataflow; fl != NULL; fl = fl->next)
    {
	if(fl->access_type == JDF_VAR_TYPE_CTL) continue;
	
	jdf_generate_code_flow_retrieving(fl, di++);

    }
	   
    ai.idx = 0;
    coutput("\n\n /* Local variables */ \n"
	    "%s\n\n",
	    UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, 
				 dump_retrieving_param, &ai, "", "    int ", ";\n", ";\n"));
    

    /* What is that ? */
    /* UTIL_DUMP_LIST(sa, f->dataflow, next, */
    /* 		   dump_data_declaration, sa2, "", "", "", ""); */
    jdf_generate_code_papi_events_before(jdf, f);
    jdf_generate_code_cache_awareness_update(jdf, f);


    // EXPRESSION DU BODY
    jdf_generate_code_dry_run_before(jdf, f);
    jdf_coutput_prettycomment('-', "%s BODY", f->fname);

    if( profile_on ) {
        sa3 = string_arena_new(64);
        linfo.prefix = "";
        linfo.sa = sa2;
        linfo.assignments = "this_task->locals";

        /* coutput("  TAKE_TIME(context, 2*this_task->function->function_id, %s_hash( __dague_handle, this_task->locals), __dague_handle->super.%s, ((dague_ddesc_t*)(__dague_handle->super.%s))->data_key((dague_ddesc_t*)__dague_handle->super.%s, %s) );\n", */
        /*         f->fname, */
        /*         f->predicate->func_or_mem, f->predicate->func_or_mem, f->predicate->func_or_mem, */
        /*         UTIL_DUMP_LIST(sa3, f->predicate->parameters, next, */
        /*                        dump_expr, (void**)&linfo, */
        /*                        "", "", ", ", "") ); */
        string_arena_free(sa3);
    }
    coutput("%s\n", f->body);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline ) {
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }
    jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname);
    jdf_generate_code_dry_run_after(jdf, f);

    ai.idx = 0;

    string_arena_free(sa);
    string_arena_free(sa2);

    /* coutput("    %s_callback_function((void*)(args->callback));\n", name); */

//    coutput("    free((__dague_%s_internal_handle_t*)__dague_handle);\n", jdf_basename);
    coutput("    free(cl_arg);\n"
	    "}\n\n");
    
}

static int
dump_nb_buffers(const jdf_function_entry_t *fun)
{
    int cmp = 0;
    jdf_dataflow_t *f = fun->dataflow;

    for(; f != NULL; f = f->next)
	cmp++;

    return cmp;
}


static void
jdf_generate_codelet(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    string_arena_t *sa;
    int nbuffers = dump_nb_buffers(f);
    (void) jdf;
    jdf_dataflow_t *df;

    coutput("static struct starpu_perfmodel %s_model = {\n"
		"    .type   = STARPU_HISTORY_BASED,\n"
		"    .symbol = \"%s_model\"\n"
		"};\n\n", name, name);

    coutput("\nstruct starpu_codelet %s_cl =\n"
	    "{\n"
	    "    .where = STARPU_CPU", name);

    if(f->body_gpu != NULL)
	coutput("|STARPU_CUDA");
    
    coutput(",\n");
    coutput("    .cpu_funcs = { func_cpu_%s, NULL },\n", name);

    if(f->body_gpu != NULL)
	coutput("    .cuda_funcs = { %s_cuda_wrapper, NULL },\n", name);

    coutput("    .nbuffers = %d", nbuffers);


    if(nbuffers > 0)
    {
	sa = string_arena_new(64);
	string_arena_init(sa);

	for(df = f->dataflow; df != NULL; df = df->next)
	{
	    if(df->access_type == JDF_VAR_TYPE_READ)
		string_arena_add_string(sa, "STARPU_R");
	    else 
	    if(df->access_type == JDF_VAR_TYPE_WRITE)
		string_arena_add_string(sa, "STARPU_W");
	    else
		string_arena_add_string(sa, "STARPU_RW");
	    if(df->next != NULL)
		string_arena_add_string(sa, ", ");
	}
	coutput(",\n    .modes = { %s }", string_arena_get_string(sa));
    }
    
    coutput(",\n    .model = &%s_model\n", name);

    
    coutput("\n};\n\n");
}


static void jdf_generate_code_hook(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    string_arena_t *sa, *sa2; //, *sa3;
    //    expr_info_t linfo;
    assignment_info_t ai;
    /* jdf_dataflow_t *fl; */
    int /* di, */ profile_on;

    /* If the function has the property profile turned off do not generate the profiling code */
    profile_on = jdf_property_get_int(f->properties, "profile", 1);

    sa  = string_arena_new(64);
    sa2 = string_arena_new(64);
    ai.sa = sa2;
    ai.idx = 0;
    ai.holder = "this_task->locals";
    ai.expr = NULL;


    jdf_generate_codelet(jdf, f, name);
    
    jdf_generate_code_callback_function(name);

    jdf_generate_code_starpu_task_create(jdf, f, name);

    jdf_generate_code_cpu_function(jdf, f, name);
    
    if(f->body_gpu != NULL)
	jdf_generate_code_gpu_function(jdf, f, name);
    /* coutput("static int %s(dague_execution_unit_t *context, dague_execution_context_t *this_task)\n" */
    /*         "{\n" */
    /*         "  const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle;\n" */
    /*         "  assignment_t tass[MAX_PARAM_COUNT];\n" */
    /*         "  (void)context; (void)__dague_handle; (void)tass;\n" */
    /*         "#if defined(DAGUE_SIM)\n" */
    /*         "  int __dague_simulation_date = 0;\n" */
    /*         "#endif\n" */
    /*         "%s", */
    /*         name, jdf_basename, jdf_basename, */
    /*         UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name,  */
    /*                              dump_assignments, &ai, "", "  int ", ";\n", ";\n")); */
    /* coutput("%s\n", */
    /*         UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, */
    /*                              dump_string, NULL, "", "  (void)", ";", ";\n")); */
    /* { */
    /*     char* output = UTIL_DUMP_LIST(sa, f->dataflow, next, */
    /*                                   dump_data_declaration, sa2, "", "", "", ""); */
    /*     if( 0 != strlen(output) ) */
    /*         coutput("  /\** Declare the variables that will hold the data, and all the accounting for each *\/\n" */
    /*                 "%s\n", */
    /*                 output); */
    /* } */

    /* coutput("  /\** Lookup the input data, and store them in the context if any *\/\n"); */
    /* for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) { */

    /*     if(fl->access_type == JDF_VAR_TYPE_CTL) continue;  /\* control flow, nothing to store *\/ */

    /*     jdf_generate_code_flow_initialization(jdf, f->fname, fl, di); */
    /*     coutput("#if defined(DAGUE_SIM)\n" */
    /*             "  if( (NULL != e%s) && (e%s->sim_exec_date > __dague_simulation_date) )\n" */
    /*             "    __dague_simulation_date =  e%s->sim_exec_date;\n" */
    /*             "#endif\n", */
    /*             fl->varname, */
    /*             fl->varname, */
    /*             fl->varname); */
    /* } */
    /* coutput("#if defined(DAGUE_SIM)\n" */
    /*         "  if( this_task->function->sim_cost_fct != NULL ) {\n" */
    /*         "    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);\n" */
    /*         "  } else {\n" */
    /*         "    this_task->sim_exec_date = __dague_simulation_date;\n" */
    /*         "  }\n" */
    /*         "  if( context->largest_simulation_date < this_task->sim_exec_date )\n" */
    /*         "    context->largest_simulation_date = this_task->sim_exec_date;\n" */
    /*         "#endif\n"); */

    /* jdf_generate_code_papi_events_before(jdf, f); */
    /* jdf_generate_code_cache_awareness_update(jdf, f); */

    /* jdf_generate_code_dry_run_before(jdf, f); */

    
    /* coutput("    return 0;\n" */
    /* 	    "}\n"); */
    



    // EXPRESSION DU BODY
    /* jdf_coutput_prettycomment('-', "%s BODY", f->fname); */

    /* if( profile_on ) { */
    /*     sa3 = string_arena_new(64); */
    /*     linfo.prefix = ""; */
    /*     linfo.sa = sa2; */
    /*     linfo.assignments = "this_task->locals"; */

    /*     coutput("  TAKE_TIME(context, 2*this_task->function->function_id, %s_hash( __dague_handle, this_task->locals), __dague_handle->super.%s, ((dague_ddesc_t*)(__dague_handle->super.%s))->data_key((dague_ddesc_t*)__dague_handle->super.%s, %s) );\n", */
    /*             f->fname, */
    /*             f->predicate->func_or_mem, f->predicate->func_or_mem, f->predicate->func_or_mem, */
    /*             UTIL_DUMP_LIST(sa3, f->predicate->parameters, next, */
    /*                            dump_expr, (void**)&linfo, */
    /*                            "", "", ", ", "") ); */
    /*     string_arena_free(sa3); */
    /* } */
    /* coutput("%s\n", f->body); */
    /* if( !JDF_COMPILER_GLOBAL_ARGS.noline ) { */
    /*     coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename); */
    /* } */
    /* jdf_coutput_prettycomment('-', "END OF %s BODY", f->fname); */
    /* jdf_generate_code_dry_run_after(jdf, f); */

    /* ai.idx = 0; */
    /* coutput("  return 0;\n" */
    /*         "}\n" */



    coutput("static int complete_%s(dague_execution_unit_t *context, dague_execution_context_t *this_task)\n"
            "{\n"
	    "   // fprintf(stderr, \"début de la fonction COMPLETE %s\\n\");\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (__dague_%s_internal_handle_t *)this_task->dague_handle;\n"
            "  (void)context; (void)__dague_handle;\n"
            "%s",
            name, name, jdf_basename, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name, 
                                 dump_assignments, &ai, "", "  int ", ";\n", ";\n"));

    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa, f->definitions, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    /* if( profile_on ) { */
    /*     coutput("  TAKE_TIME(context,2*this_task->function->function_id+1, %s_hash( __dague_handle, this_task->locals ), NULL, 0);\n", */
    /*             f->fname); */
    /* } */
    jdf_generate_code_papi_events_after(jdf, f);

    /* coutput("#if defined(DISTRIBUTED)\n" */
    /*         "  /\** If not working on distributed, there is no risk that data is not in place *\/\n"); */
    /* for( di = 0, fl = f->dataflow; fl != NULL; fl = fl->next, di++ ) { */
    /*     jdf_generate_code_flow_final_writes(jdf, fl, di); */
    /* } */
    /* coutput("#endif /\* DISTRIBUTED *\/\n"); */

    jdf_generate_code_grapher_task_done(jdf, f, "this_task");

    jdf_generate_code_call_release_dependencies(jdf, f, "this_task");

    coutput("   // fprintf(stderr, \"Fin de la fonction COMPLETE %s\\n\");\n"
	    "    return 0;\n"
            "}\n\n", name);
    string_arena_free(sa);
    string_arena_free(sa2);
}

static void jdf_generate_code_free_hash_table_entry(const jdf_t *jdf, const jdf_function_entry_t *f)
{
    jdf_dataflow_t *dl;
    jdf_dep_t *dep;
    expr_info_t info;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);
    int i, cond_index;
    char* condition[] = {"    if( %s ) {\n", "    else if( %s ) {\n"};
    assignment_info_t ai;

    ai.sa = sa;
    ai.idx = 0;
    ai.holder = "context->locals";
    ai.expr = NULL;

    coutput("  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {\n"
            "%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name, 
                                 dump_assignments, &ai, "", "    int ", ";\n", ";\n"));
    /* Quiet the unused variable warnings */
    coutput("%s\n",
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name,
                                 dump_string, NULL, "   ", " (void)", ";", ";\n"));

    for( dl = f->dataflow; dl != NULL; dl = dl->next ) {
        if(dl->access_type == JDF_VAR_TYPE_CTL) continue;
        cond_index = 0;
        for( dep = dl->deps; dep != NULL; dep = dep->next ) {
            if( dep->type & JDF_DEP_TYPE_IN ) {
                i = jdf_data_input_index(f, dl->varname);

                switch( dep->guard->guard_type ) {
                case JDF_GUARD_UNCONDITIONAL:
                    if( NULL != dep->guard->calltrue->var ) {
                        if( 0 != cond_index ) coutput("    else {\n");
                        coutput("    data_repo_entry_used_once( eu, %s_repo, context->data[%d].data_repo->key );\n"
                                "    (void)AUNREF(context->data[%d].data);\n",
                                dep->guard->calltrue->func_or_mem, i, i);
                        if( 0 != cond_index ) coutput("    }\n");
                    }
                    goto next_dependency;
                case JDF_GUARD_BINARY:
                    if( NULL != dep->guard->calltrue->var ) {
                        info.prefix = "";
                        info.sa = sa1;
                        info.assignments = "context->locals";

                        coutput((0 == cond_index ? condition[0] : condition[1]),
                                dump_expr((void**)dep->guard->guard, &info));
                        coutput("      data_repo_entry_used_once( eu, %s_repo, context->data[%d].data_repo->key );\n"
                                "      (void)AUNREF(context->data[%d].data);\n"
                                "    }\n",
                                dep->guard->calltrue->func_or_mem, i, i);
                        cond_index++;
                    }
                    break;
                case JDF_GUARD_TERNARY:
                    if( NULL != dep->guard->calltrue->var ) {
                        info.prefix = "";
                        info.sa = sa1;
                        info.assignments = "context->locals";
                        coutput((0 == cond_index ? condition[0] : condition[1]),
                                dump_expr((void**)dep->guard->guard, &info));
                        coutput("      data_repo_entry_used_once( eu, %s_repo, context->data[%d].data_repo->key );\n"
                                "      (void)AUNREF(context->data[%d].data);\n",
                                dep->guard->calltrue->func_or_mem, i, i);
                        if( NULL != dep->guard->callfalse->var ) {
                            coutput("    } else {\n"
                                    "      data_repo_entry_used_once( eu, %s_repo, context->data[%d].data_repo->key );\n"
                                    "      (void)AUNREF(context->data[%d].data);\n",
                                    dep->guard->callfalse->func_or_mem, i, i);
                        }
                    } else if( NULL != dep->guard->callfalse->var ) {
                        info.prefix = "";
                        info.sa = sa1;
                        info.assignments = "context->locals";
                        coutput("    if( !(%s) ) {\n"
                                "      data_repo_entry_used_once( eu, %s_repo, context->data[%d].data_repo->key );\n"
                                "      (void)AUNREF(context->data[%d].data);\n",
                                dump_expr((void**)dep->guard->guard, &info),
                                dep->guard->callfalse->func_or_mem, i, i);
                    }
                    coutput("    }\n");
                    goto next_dependency;
                }
            }
        }
    next_dependency:
        (void)jdf;  /* just to keep the compilers happy regarding the goto to an empty statement */
    }
    coutput("  }\n");

    string_arena_free(sa);
    string_arena_free(sa1);
}

static void jdf_generate_code_release_deps(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    int has_output_data = function_has_data_output(f);

    coutput("static int %s(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t *)context->dague_handle;\n"
            "  dague_release_dep_fct_arg_t arg;\n"
            "  arg.nb_released = 0;\n"
            "  arg.output_usage = 0;\n"
            "  arg.action_mask = action_mask;\n"
            "  arg.deps = deps;\n"
            "  arg.ready_list = NULL;\n"
            "  (void)__dague_handle;\n",
            name, jdf_basename, jdf_basename);

    if( 0 != has_output_data )
        coutput("  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {\n"
                "    arg.output_entry = data_repo_lookup_entry_and_create( eu, %s_repo, %s_hash(__dague_handle, context->locals) );\n"
                "#if defined(DAGUE_SIM)\n"
                "    assert(arg.output_entry->sim_exec_date == 0);\n"
                "    arg.output_entry->sim_exec_date = context->sim_exec_date;\n"
                "#endif\n"
                "  }\n",
                f->fname, f->fname);
    else
        coutput("  arg.output_entry = NULL;\n");

    /* coutput("#if defined(DISTRIBUTED)\n" */
    /*         "  arg.remote_deps_count = 0;\n" */
    /*         "  arg.remote_deps = NULL;\n" */
    /*         "#endif\n" */

    coutput("  iterate_successors_of_%s_%s(eu, context, action_mask, dague_release_dep_fct, &arg);\n"
            "\n",
            jdf_basename, f->fname);

    coutput("  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {\n");
    if( 0 != has_output_data ) {
        coutput("    data_repo_entry_addto_usage_limit(%s_repo, arg.output_entry->key, arg.output_usage);\n",
                f->fname);
    }
    coutput("    if( NULL != arg.ready_list ) {\n"
            "      // Previous : __dague_schedule(eu, arg.ready_list);\n"
	    "      //hook_of_%s_%s_task_create(eu, arg.ready_list);\n"
            "      generic_scheduling_func(eu, (dague_list_item_t*)arg.ready_list);"
            "      arg.ready_list = NULL;\n"
            "    }\n"
            "  }\n",
            /* "#if defined(DISTRIBUTED)\n" */
            /* "  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {\n" */
            /* "    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);\n" */
            /* "  }\n" */
            /* "#endif\n", */
	    jdf_basename, f->fname);

    if( 0 != has_output_data )
        jdf_generate_code_free_hash_table_entry(jdf, f);

    coutput("  assert( NULL == arg.ready_list );\n"
            "  return arg.nb_released;\n"
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
    expr_info_t info, linfo;
    string_arena_t *sa2, *sa1, *sa_close;
    int i, nbopen;
    char *p;

    string_arena_init(sa_open);

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
    
    p = (char*)malloc(strlen(targetf->fname) + 2);
    sprintf(p, "%s_", targetf->fname);

    linfo.prefix = p;
    linfo.sa = sa1;
    linfo.assignments = "nc.locals";

    info.sa = sa2;
    info.prefix = "";
    info.assignments = "assignments";

    sa_close = string_arena_new(64);

    nbopen = 0;

    string_arena_add_string(sa_open, "%s%s%s.function = (const dague_function_t*)&%s_%s;\n",
                            prefix, indent(nbopen), var, jdf_basename, targetf->fname);
    for(el = call->parameters, nl = targetf->parameters, i = 0, def = targetf->definitions; 
        el != NULL && nl != NULL; 
        el = el->next, nl = nl->next, i++, def = def->next) {
        
        if( strcmp(nl->name, def->name) ) {
            jdf_fatal(lineno, 
                      "During code generation: parameter %s of function %s has no matching definition (found definition of internal %s)\n",
                      nl->name, targetf->fname, def->name);
            exit(1);
        }

        string_arena_add_string(sa_close,
                                "%s%s}\n", prefix, indent(nbopen));
        if( el->op == JDF_RANGE ) {
            string_arena_add_string(sa_open, 
                                    "%s%s{\n"
                                    "%s%s  int %s_%s;\n"
                                    "%s%s  for( %s_%s = %s;",
                                    prefix, indent(nbopen),
                                    prefix, indent(nbopen), targetf->fname, nl->name,
                                    prefix, indent(nbopen), targetf->fname, nl->name, dump_expr((void**)el->jdf_ba1, &info));
            string_arena_add_string(sa_open, "%s_%s <= %s; %s_%s++ ) {\n",
                                    targetf->fname, nl->name, dump_expr((void**)el->jdf_ba2, &info), targetf->fname, nl->name);
            string_arena_add_string(sa_close,
                                    "%s%s  }\n", prefix, indent(nbopen));
            nbopen++;
        } else {
            string_arena_add_string(sa_open, 
                                    "%s%s{\n"
                                    "%s%s  const int %s_%s = %s;\n",
                                    prefix, indent(nbopen),
                                    prefix, indent(nbopen), targetf->fname, nl->name, dump_expr((void**)el, &info));
        }
        
        if( def->expr->op == JDF_RANGE ) {
            string_arena_add_string(sa_open, 
                                    "%s%s  if( (%s_%s >= (%s))", 
                                    prefix, indent(nbopen), targetf->fname, nl->name, 
                                    dump_expr((void**)def->expr->jdf_ba1, &linfo));
            string_arena_add_string(sa_open, " && (%s_%s <= (%s)) ) {\n",
                                    targetf->fname, nl->name, 
                                    dump_expr((void**)def->expr->jdf_ba2, &linfo));
            string_arena_add_string(sa_close, "%s%s  }\n",
                                    prefix, indent(nbopen));
            nbopen++;
        } else {
            string_arena_add_string(sa_open, 
                                    "%s%s  if( (%s_%s == (%s)) ) {\n", 
                                    prefix, indent(nbopen), targetf->fname, nl->name, 
                                    dump_expr((void**)def->expr, &linfo));
            string_arena_add_string(sa_close, "%s%s  }\n", prefix, indent(nbopen));
            nbopen++;
        }
        
        string_arena_add_string(sa_open, 
                                "%s%s  %s.locals[%d].value = %s_%s;\n", 
                                prefix, indent(nbopen), var, i, 
                                targetf->fname, nl->name);
        
        nbopen++;
    }

    asprintf(&linfo.assignments, "%s.locals", var);
    for(; NULL != def; def = def->next, i++) {
        string_arena_add_string(sa_open, "%s%s  const int %s_%s = %s;\n",
                                prefix, indent(nbopen-1),
                                targetf->fname, def->name,
                                dump_expr((void**)def->expr, &linfo));
        string_arena_add_string(sa_open, "%s%s  %s.locals[%d].value = %s_%s;\n", 
                                prefix, indent(nbopen-1), var, i, 
                                targetf->fname, def->name);
    }

    /**
     * If we have to execute code possibly comming from the user then we need to instantiate
     * the entire stack of the target function, including the local definitions.
     */
    /* string_arena_add_string(sa_open,  */
    /*                         "#if defined(DISTRIBUTED)\n" */
    /*                         "%s%s  rank_dst = ((dague_ddesc_t*)__dague_handle->super.%s)->rank_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n" */
    /*                         "#endif\n", */
    /*                         prefix, indent(nbopen), targetf->predicate->func_or_mem, targetf->predicate->func_or_mem, */
    /*                         UTIL_DUMP_LIST(sa2, targetf->predicate->parameters, next, */
    /*                                        dump_expr, (void**)&linfo, */
    /*                                        "", "", ", ", "")); */

    string_arena_add_string(sa_open,
                            "#if defined(DAGUE_DEBUG)\n"
                            "%s%sif( NULL != eu ) {\n"
                            "%s%s  char tmp[128], tmp1[128];\n"
                            "%s%s  DEBUG((\"thread %%d release deps of %s:%%s to %s:%%s (from node %%d to %%d)\\n\", eu->eu_id,\n"
                            "%s%s         dague_service_to_string(this_task, tmp, 128),\n"
                            "%s%s         dague_service_to_string(&%s, tmp1, 128), rank_src, rank_dst));\n"
                            "%s%s}\n"
                            "#endif\n",
                            prefix, indent(nbopen),
                            prefix, indent(nbopen),
                            prefix, indent(nbopen), flow->varname, call->var,
                            prefix, indent(nbopen),
                            prefix, indent(nbopen), var,
                            prefix, indent(nbopen));
    free(linfo.assignments);
    linfo.assignments = NULL;
    free(p);
    linfo.prefix = NULL;

    if( NULL != targetf->priority ) {
        string_arena_add_string(sa_open,
                                "%s%s  %s.priority = priority_of_%s_%s_as_expr_fct(this_task->dague_handle, nc.locals);\n",
                                prefix, indent(nbopen), var, jdf_basename, targetf->fname);
    } else {
        string_arena_add_string(sa_open, "%s%s  %s.priority = 0;\n",
                                prefix, indent(nbopen), var);
    }

    string_arena_add_string(sa_open, 
                            "%s%s  if( DAGUE_ITERATE_STOP == %s )\n"
                            "%s%s    return;\n",
                            prefix, indent(nbopen), calltext,
                            prefix, indent(nbopen));

    string_arena_add_string(sa_open, "%s", string_arena_get_string(sa_close));

    string_arena_free(sa_close);
    string_arena_free(sa2);

    if( (void*)el != (void*)nl) {
        jdf_fatal(lineno,
                  "During code generation: call to %s at this line has not the same number of parameters as the function definition.\n",
                  call->func_or_mem);
        exit(1);        
    }

    return string_arena_get_string(sa_open);
}

static void jdf_generate_code_iterate_successors(const jdf_t *jdf, const jdf_function_entry_t *f, const char *name)
{
    jdf_dataflow_t *fl;
    jdf_dep_t *dl;
    int flowempty, flowtomem;
    string_arena_t *sa = string_arena_new(64);
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    string_arena_t *sa_coutput = string_arena_new(1024);
    int flownb, depnb;
    assignment_info_t ai;
    expr_info_t info;

    info.sa = sa2;
    info.prefix = "";
    info.assignments = "this_task->locals";

    ai.sa = sa2;
    ai.idx = 0;
    ai.holder = "this_task->locals";
    ai.expr = NULL;
    coutput("static void\n"
            "%s(dague_execution_unit_t *eu, dague_execution_context_t *this_task,\n"
            "               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)this_task->dague_handle;\n"
            "  dague_execution_context_t nc;\n"
            "  dague_arena_t* arena = NULL;\n"
            "  int rank_src = 0, rank_dst = 0;\n"
            "%s"
            "  (void)rank_src; (void)rank_dst; (void)__dague_handle;\n",
            name,
            jdf_basename, jdf_basename,
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name, 
                                 dump_assignments, &ai, "", "  int ", ";\n", ";\n"));
    coutput("%s",
            UTIL_DUMP_LIST_FIELD(sa1, f->definitions, next, name,
                                 dump_string, NULL, "", "  (void)", ";", ";\n"));

    coutput("  nc.dague_handle = this_task->dague_handle;\n");
    /* coutput("#if defined(DISTRIBUTED)\n" */
    /*         "  rank_src = ((dague_ddesc_t*)__dague_handle->super.%s)->rank_of((dague_ddesc_t*)__dague_handle->super.%s, %s);\n" */
    /*         "#endif\n", */
    /*         f->predicate->func_or_mem, f->predicate->func_or_mem, */
    /*         UTIL_DUMP_LIST(sa, f->predicate->parameters, next, */
    /*                        dump_expr, (void**)&info, */
    /*                        "", "", ", ", "")); */

    flownb = 0;
    for(fl = f->dataflow; fl != NULL; fl = fl->next) {
        flowempty = 1;
        flowtomem = 0;
        depnb = 0;
        string_arena_init(sa_coutput);
        for(dl = fl->deps; dl != NULL; dl = dl->next) {
            if( dl->type & JDF_DEP_TYPE_OUT )  {
                string_arena_init(sa);
                if( JDF_VAR_TYPE_CTL == fl->access_type ) {
                    string_arena_add_string(sa, "NULL");
                } else {
                    string_arena_add_string(sa, "__dague_handle->super.arenas[DAGUE_%s_%s_ARENA]",
                                            jdf_basename, dl->datatype_name);
                }
                /* string_arena_add_string(sa_coutput, */
                /*                         "#if defined(DISTRIBUTED)\n" */
                /*                         "    arena = %s;\n" */
                /*                         "#endif  /\* defined(DISTRIBUTED) *\/\n", string_arena_get_string(sa)); */
                string_arena_init(sa);
                string_arena_add_string(sa, "ontask(eu, &nc, this_task, %d, %d, rank_src, rank_dst, arena, ontask_arg)",
                                        flownb, depnb);

                switch( dl->guard->guard_type ) {
                case JDF_GUARD_UNCONDITIONAL:
                    if( NULL != dl->guard->calltrue->var) {
                        flowempty = 0;
                        
                        string_arena_add_string(sa_coutput,
                                                "%s",
                                                jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa), dl->guard->calltrue, dl->lineno, 
                                                                            "    ", "nc") );
                    } else {
                        flowtomem = 1;
                    }
                    break;
                case JDF_GUARD_BINARY:
                    if( NULL != dl->guard->calltrue->var ) {
                        flowempty = 0;
                        string_arena_add_string(sa_coutput,
                                                "    if( %s ) {\n"
                                                "%s"
                                                "    }\n",
                                                dump_expr((void**)dl->guard->guard, &info),
                                                jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa), dl->guard->calltrue, dl->lineno, 
                                                                            "      ", "nc") );
                    } else {
                        flowtomem = 1;
                    }
                    break;
                case JDF_GUARD_TERNARY:
                    if( NULL != dl->guard->calltrue->var ) {
                        flowempty = 0;
                        string_arena_add_string(sa_coutput,
                                                "    if( %s ) {\n"
                                                "%s"
                                                "    }",
                                                dump_expr((void**)dl->guard->guard, &info),
                                                jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa), dl->guard->calltrue, dl->lineno, 
                                                                            "      ", "nc"));

                        depnb++;
                        string_arena_init(sa);
                        string_arena_add_string(sa, "ontask(eu, &nc, this_task, %d, %d, rank_src, rank_dst, arena, ontask_arg)",
                                                flownb, depnb);

                        if( NULL != dl->guard->callfalse->var ) {
                            string_arena_add_string(sa_coutput,
                                                    " else {\n"
                                                    "%s"
                                                    "    }\n",
                                                    jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa), dl->guard->callfalse, dl->lineno, 
                                                                                "      ", "nc") );
                        } else {
                            string_arena_add_string(sa_coutput,
                                                    "\n");
                        }
                    } else {
                        depnb++;
                        string_arena_init(sa);
                        string_arena_add_string(sa, "ontask(eu, &nc, this_task, %d, %d, rank_src, rank_dst, arena, ontask_arg)",
                                                flownb, depnb);

                        if( NULL != dl->guard->callfalse->var ) {
                            flowempty = 0;
                            string_arena_add_string(sa_coutput,
                                                    "    if( !(%s) ) {\n"
                                                    "%s"
                                                    "    }\n",
                                                    dump_expr((void**)dl->guard->guard, &info),
                                                    jdf_dump_context_assignment(sa1, jdf, fl, string_arena_get_string(sa), dl->guard->callfalse, dl->lineno, 
                                                                                "      ", "nc") );
                        } else {
                            flowtomem = 1;
                        }
                    }
                    break;
                }
                depnb++;
            }
        }
        if( (1 == flowempty) && (0 == flowtomem) ) {
            coutput("  /* Flow of data %s has only IN dependencies */\n", fl->varname);
        } else if( 1 == flowempty ) {
            coutput("  /* Flow of data %s has only OUTPUT dependencies to Memory */\n", fl->varname);
            flownb++;
        } else {
            coutput("  /* Flow of Data %s */\n"
                    "  if( action_mask & (1 << %d) ) {\n"
                    "%s"
                    "  }\n",
                    fl->varname, flownb, string_arena_get_string(sa_coutput));
            flownb++;
        }
    }
    coutput("  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;\n");
    coutput("}\n\n");

    string_arena_free(sa);
    string_arena_free(sa1);
    string_arena_free(sa2);
    string_arena_free(sa_coutput);
}

static void jdf_generate_inline_c_function(jdf_expr_t *expr)
{
    static int inline_c_functions = 0;
    string_arena_t *sa1 = string_arena_new(64);
    string_arena_t *sa2 = string_arena_new(64);
    assignment_info_t ai;

    assert(JDF_OP_IS_C_CODE(expr->op));
    asprintf(&expr->jdf_c_code.fname, "%s_inline_c_expr%d_line_%d", 
             jdf_basename, ++inline_c_functions, expr->jdf_c_code.lineno);
    coutput("static inline int %s(const dague_handle_t *__dague_handle_parent, const assignment_t *assignments)\n"
            "{\n"
            "  const __dague_%s_internal_handle_t *__dague_handle = (const __dague_%s_internal_handle_t*)__dague_handle_parent;\n"
            "  (void)__dague_handle;\n",
            expr->jdf_c_code.fname, jdf_basename, jdf_basename);

    if( NULL != expr->jdf_c_code.function_context ) {
        coutput("  /* This inline C function was declared in the context of the task %s */\n",
                expr->jdf_c_code.function_context->fname);

        ai.sa = sa1;
        ai.idx = 0;
        ai.holder = "assignments";
        ai.expr = NULL;
        coutput("%s\n",
                UTIL_DUMP_LIST_FIELD(sa2, expr->jdf_c_code.function_context->definitions, next, name, 
                                     dump_assignments, &ai, "", "  int ", ";\n", ";\n"));
        coutput("%s\n",
                UTIL_DUMP_LIST_FIELD(sa2, expr->jdf_c_code.function_context->definitions, next, name, 
                                     dump_string, NULL, "", "  (void)", ";", ";\n"));
    } else {
        coutput("  /* This inline C function was declared in the global context: no variables */\n"
                "  (void)assignments;\n");
    }

    string_arena_free(sa1);
    string_arena_free(sa2);

    coutput("%s\n", expr->jdf_c_code.code);
    if( !JDF_COMPILER_GLOBAL_ARGS.noline )
        coutput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    coutput("}\n"
            "\n");
}

static void jdf_generate_inline_c_functions(const jdf_t *jdf)
{
    jdf_expr_t *le;
    for( le = jdf->inline_c_functions; NULL != le; le = le->next ) {
        jdf_generate_inline_c_function(le);
    }
}


/**
 * Analyze the code to optimize the output
 */
int jdf_optimize( jdf_t* jdf )
{
    jdf_function_entry_t *f;
    string_arena_t *sa;
    int i, can_be_startup, high_priority;
    
    sa = string_arena_new(64);
    for(i = 0, f = jdf->functions; NULL != f; f = f->next, i++) {
        /* Check if the function has the HIGH_PRIORITY property on */
        high_priority = jdf_property_get_int(f->properties, "high_priority", 0);
        if( high_priority ) {
            f->flags |= JDF_FUNCTION_FLAG_HIGH_PRIORITY;
        }

        can_be_startup = 1;
        UTIL_DUMP_LIST(sa, f->dataflow, next, has_ready_input_dependency, &can_be_startup, NULL, NULL, NULL, NULL);
        if( can_be_startup )
            f->flags |= JDF_FUNCTION_FLAG_CAN_BE_STARTUP;
    }
    string_arena_free(sa);
    return 0;
}

/** Main Function */

int jdf2c(const char *output_c, const char *output_h, const char *_jdf_basename, const jdf_t *jdf)
{
    int ret = 0;

#if 0
    /* TODO: Thomas needs to see if this is old junk or WIP */
    init_unique_rgb_color();
#endif
    jdf_cudafilename = malloc((strlen(_jdf_basename)+4) * sizeof(char));
    sprintf(jdf_cudafilename, "%s.cu", _jdf_basename);


    jdf_cfilename = output_c;
    jdf_basename = _jdf_basename;
    cfile = NULL;
    hfile = NULL;
    cudafile = NULL;

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

    cudafile = fopen(jdf_cudafilename, "w");
    if(cudafile == NULL)
    {
	fprintf(stderr, "unable to create %s: %s\n", jdf_cudafilename, strerror(errno));
	ret = -1;
	goto err;
    }

    cudafile_lineno = 1;
    cfile_lineno = 1;
    hfile_lineno = 1;
    

    /**
     * Now generate the code.
     */
    /**
     * Dump all the prologue sections
     */
    if( NULL != jdf->prologue ) {
        houtput("%s\n", jdf->prologue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            houtput("#line %d \"%s\"\n", cfile_lineno+1, jdf_cfilename);
    }

    jdf_generate_header_file(jdf);



    jdf_generate_structure(jdf);
    jdf_generate_inline_c_functions(jdf);
    jdf_generate_hashfunctions(jdf);
    jdf_generate_predeclarations( jdf );
    jdf_generate_code_generic_scheduling_func(jdf);
    jdf_generate_functions_statics(jdf);
    jdf_generate_startup_hook( jdf );

    /**
     * Generate the externally visible function.
     */
    jdf_generate_constructor(jdf);

    jdf_generate_destructor( jdf );

    /**
     * Dump all the epilogue sections
     */
    if( NULL != jdf->epilogue ) {
        coutput("%s\n", jdf->epilogue->external_code);
        if( !JDF_COMPILER_GLOBAL_ARGS.noline )
            coutput("#line %d \"%s\"\n",cfile_lineno+1, jdf_cfilename);
    }

    houtput("#endif /* _%s_h_ */ \n",
            jdf_basename);

 err:
    free(jdf_cudafilename);
    if( NULL != cudafile)
	fclose(cudafile);

    if( NULL != cfile ) 
        fclose(cfile);

    if( NULL != hfile )
        fclose(hfile);

    return ret;
}

