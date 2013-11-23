/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#include <stdio.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#include <stdlib.h>
#include <assert.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"

jdf_t current_jdf;
int current_lineno;

extern const char *yyfilename;

void jdf_warn(int lineno, const char *format, ...)
{
    char msg[512];
    va_list ap;

    va_start(ap, format);
    vsnprintf(msg, 512, format, ap);
    va_end(ap);

    fprintf(stderr, "Warning on %s:%d: %s", yyfilename, lineno, msg);
}

void jdf_fatal(int lineno, const char *format, ...)
{
    char msg[512];
    va_list ap;

    va_start(ap, format);
    vsnprintf(msg, 512, format, ap);
    va_end(ap);

    fprintf(stderr, "Fatal Error on %s:%d: %s", yyfilename, lineno, msg);
}

void jdf_prepare_parsing(void)
{
    current_jdf.prologue  = NULL;
    current_jdf.epilogue  = NULL;
    current_jdf.globals   = NULL;
    current_jdf.functions = NULL;
    current_jdf.global_properties = NULL;
    current_lineno = 1;
}

static int jdf_sanity_check_global_redefinitions(void)
{
    jdf_global_entry_t *g1, *g2;
    int rc = 0;

    for(g1 = current_jdf.globals; g1 != NULL; g1 = g1->next) {
        for(g2 = g1->next; g2 != NULL; g2 = g2->next) {
            if( !strcmp(g1->name, g2->name) ) {
                jdf_fatal(JDF_OBJECT_LINENO(g2), "Global %s is redefined here (previous definition was on line %d)\n",
                          g1->name, JDF_OBJECT_LINENO(g1));
                rc = -1;
            }
        }
    }
    return rc;
}

static int jdf_sanity_check_global_masked(void)
{
    jdf_global_entry_t *g;
    jdf_function_entry_t *f;
    jdf_name_list_t *n;
    jdf_def_list_t *d;
    int rc = 0;

    for(g = current_jdf.globals; g != NULL; g = g->next) {
        for(f = current_jdf.functions; f != NULL; f = f->next) {
            for(n = f->parameters; n != NULL; n = n->next) {
                if( !strcmp(n->name, g->name) ) {
                    jdf_warn(JDF_OBJECT_LINENO(f), "Global %s defined line %d is masked by the local parameter %s of function %s\n",
                             g->name, JDF_OBJECT_LINENO(g), n->name, f->fname);
                    rc++;
                }
            }
            for(d = f->locals; d != NULL; d = d->next) {
                if( !strcmp(d->name, g->name) ) {
                    jdf_warn(JDF_OBJECT_LINENO(d), "Global %s defined line %d is masked by the local definition of %s in function %s\n",
                             g->name, JDF_OBJECT_LINENO(g), d->name, f->fname);
                    rc++;
                }
            }
        }
    }
    return rc;
}

static int jdf_sanity_check_expr_bound_before_global(jdf_expr_t *e, jdf_global_entry_t *g1)
{
    jdf_global_entry_t *g2;
    char *vc, *dot;
    int rc = 0;
    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        dot = strchr(vc, '.');
        if( NULL != dot )
            *dot = '\0';
        for(g2 = current_jdf.globals; g2 != g1; g2 = g2->next) {
            if( !strcmp( vc, g2->name ) ) {
                break;
            }
        }
        if( g2 == g1 ) {
            jdf_fatal(JDF_OBJECT_LINENO(g1), "Global %s is defined using variable %s (in %s) which is unbound at this time\n",
                      g1->name, vc, e->jdf_var);
            rc = -1;
        }
        free(vc);
        return rc;
    case JDF_CST:
        return 0;
    case JDF_TERNARY:
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_tat, g1) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_ta1, g1) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_ta2, g1) < 0 )
            rc = -1;
        return rc;
    case JDF_NOT:
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_ua, g1) < 0 )
            rc = -1;
        return rc;
    case JDF_C_CODE:
        return 0;
    default:
        if( (NULL != e->jdf_ba1) && (jdf_sanity_check_expr_bound_before_global(e->jdf_ba1, g1) < 0) )
            rc = -1;
        if( (NULL != e->jdf_ba2) && (jdf_sanity_check_expr_bound_before_global(e->jdf_ba2, g1) < 0) )
            rc = -1;
        return rc;
    }
}

static int jdf_sanity_check_global_unbound(void)
{
    int rc = 0;
    jdf_global_entry_t *g;
    for(g = current_jdf.globals; g != NULL; g = g->next) {
        if( NULL != g->expression ) {
            if( jdf_sanity_check_expr_bound_before_global(g->expression, g) < 0 )
                rc = -1;
        }
    }
    return rc;
}

static int jdf_sanity_check_function_redefinitions(void)
{
    jdf_function_entry_t *f1, *f2;
    int rc = 0;

    for(f1 = current_jdf.functions; f1 != NULL; f1 = f1->next) {
        for(f2 = f1->next; f2 != NULL; f2 = f2->next) {
            if( !strcmp(f1->fname, f2->fname) ) {
                jdf_fatal(JDF_OBJECT_LINENO(f2), "Function %s is redefined here (previous definition was on line %d)\n",
                          f1->fname, JDF_OBJECT_LINENO(f1));
                rc = -1;
            }
        }
    }
    return rc;
}

static int jdf_sanity_check_parameters_are_consistent_with_definitions(void)
{
    jdf_function_entry_t *f;
    jdf_name_list_t *p;
    jdf_def_list_t *d, *d2;
    int rc = 0;
    int pi, found_def;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        pi = 1;
        for(p = f->parameters; p != NULL; p = p->next, pi++) {
            found_def = 0;
            for(d = f->locals; d != NULL; d = d->next) {
                if( 0 == strcmp(d->name, p->name) ) {
                    if( found_def ) {
                        jdf_fatal(JDF_OBJECT_LINENO(f), "The definition of %s (%dth parameter of function %s) appears more than once.\n",
                                  p->name, pi, f->fname);
                        rc = -1;
                    } else {
                        found_def = 1;
                    }
                }
            }
            if( !found_def ) {
                jdf_fatal(JDF_OBJECT_LINENO(f), "Parameter %s of function %s is declared but no range is associated to it\n",
                          p->name, f->fname);
                rc = -1;
            }
        }

        pi = 1;
        for(d = f->locals; d!= NULL; d = d->next, pi++) {
            found_def = 0;
            for(p = f->parameters; p != NULL; p = p->next) {
                if( strcmp(d->name, p->name) == 0 ) {
                    found_def = 1;
                    break;
                }
            }
            if( found_def == 0 ) {
                if( d->expr->op == JDF_RANGE ) {
                    jdf_warn(JDF_OBJECT_LINENO(f), "Definition %d of function %s for %s is a range, but not a parameter of the function.\n"
                             "  If this range allows for more than one value, that would make multiple functions %s with the same name.\n",
                             pi, f->fname, d->name, f->fname);
                }
                for(d2 = d->next; d2!=NULL; d2=d2->next) {
                    if( !strcmp(d->name, d2->name) ) {
                        jdf_fatal(JDF_OBJECT_LINENO(f), "The definition of %s in function %s appears more than once.\n",
                                  d->name, f->fname);
                        rc = -1;
                    }
                }
            }
        }
    }
    return rc;
}

static int jdf_sanity_check_expr_bound_before_definition(jdf_expr_t *e, jdf_function_entry_t *f, jdf_def_list_t *d)
{
    jdf_global_entry_t *g;
    jdf_def_list_t *d2;
    char *vc, *dot;
    int rc = 0;

    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        dot = strchr(vc, '.');
        if( NULL != dot )
            *dot = '\0';
        for(g = current_jdf.globals; g != NULL; g = g->next) {
            if( !strcmp( vc, g->name ) ) {
                break;
            }
        }
        if( g == NULL ) {
            for(d2 = f->locals; d2 != d; d2 = d2->next) {
                if( !strcmp( vc, d2->name ) ) {
                    break;
                }
            }
            if( d2 == d ) {
                jdf_fatal(JDF_OBJECT_LINENO(d), "Local %s is defined using variable %s (in %s) which is unbound at this time\n",
                          d->name,  vc, e->jdf_var);
                rc = -1;
            }
            free(vc);
        }
        return rc;
    case JDF_CST:
        return 0;
    case JDF_TERNARY:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_tat, f, d) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ta1, f, d) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ta2, f, d) < 0 )
            rc = -1;
        return rc;
    case JDF_NOT:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ua, f, d) < 0 )
            rc = -1;
        return rc;
    case JDF_C_CODE:
        return 0;
    default:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ba1, f, d) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ba2, f, d) < 0 )
            rc = -1;
        return rc;
    }
}

static int jdf_sanity_check_definition_unbound(void)
{
    int rc = 0;
    jdf_function_entry_t *f;
    jdf_def_list_t *d;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        for(d = f->locals; d != NULL; d = d->next) {
            if( jdf_sanity_check_expr_bound_before_definition(d->expr, f, d) < 0 )
                rc = -1;
        }
    }
    return rc;
}

static int jdf_sanity_check_expr_bound(jdf_expr_t *e, const char *kind, jdf_function_entry_t *f)
{
    jdf_global_entry_t *g;
    jdf_def_list_t *d;
    char *vc, *dot;
    int rc = 0;

    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        dot = strchr(vc, '.');
        if( NULL != dot )
            *dot = '\0';
        for(g = current_jdf.globals; g != NULL; g = g->next) {
            if( !strcmp( vc, g->name ) ) {
                break;
            }
        }
        if( g == NULL ) {
            for(d = f->locals; d != NULL; d = d->next) {
                if( !strcmp( vc, d->name ) ) {
                    break;
                }
            }
            if( d == NULL ) {
                jdf_fatal(JDF_OBJECT_LINENO(f), "%s of function %s is defined using variable %s (in %s) which is unbound at this time\n",
                          kind, f->fname, vc, e->jdf_var);
                rc = -1;
            }
            free(vc);
        }
        return rc;
    case JDF_CST:
        return 0;
    case JDF_TERNARY:
        if( jdf_sanity_check_expr_bound(e->jdf_tat, kind, f) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound(e->jdf_ta1, kind, f) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound(e->jdf_ta2, kind, f) < 0 )
            rc = -1;
        return rc;
    case JDF_NOT:
        if( jdf_sanity_check_expr_bound(e->jdf_ua, kind, f) < 0 )
            rc = -1;
        return rc;
    case JDF_C_CODE:
        return 0;
    default:
        if( jdf_sanity_check_expr_bound(e->jdf_ba1, kind, f) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound(e->jdf_ba2, kind, f) < 0 )
            rc = -1;
        return rc;
    }
}

static int jdf_sanity_check_predicates_unbound(void)
{
    int rc = 0, i;
    jdf_function_entry_t *f;
    jdf_expr_t *e;
    char kind[64];

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        i = 0;
        for(e = f->predicate->parameters; e != NULL; e = e->next) {
            snprintf(kind, 64, "Parameter number %d of predicate", i);
            if( jdf_sanity_check_expr_bound(e, kind, f) < 0 )
                rc = -1;
            i++;
        }
    }
    return rc;
}

static int jdf_sanity_check_dataflow_expressions_unbound(void)
{
    int rc = 0;
    jdf_function_entry_t *f;
    jdf_dataflow_t *flow;
    jdf_expr_t *e;
    jdf_dep_t *dep;
    int i, j, k;
    char kind[128];

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        i = 1;
        for(flow = f->dataflow; flow != NULL; flow = flow->next) {
            j =  1;
            for(dep = flow->deps; dep != NULL; dep = dep->next) {
                snprintf(kind, 128,
                         "Guard of dependency %d\n"
                         "  of dataflow number %d (variable %s) at line %d",
                         j, i,  flow->varname, JDF_OBJECT_LINENO(flow));
                if( (dep->guard->guard_type != JDF_GUARD_UNCONDITIONAL) &&
                    (jdf_sanity_check_expr_bound(dep->guard->guard, kind, f) < 0) )
                    rc = -1;
                k = 1;
                for(e = dep->guard->calltrue->parameters; e != NULL; e = e->next) {
                    snprintf(kind, 128,
                             "Parameter %d of dependency %d\n"
                             "  of dataflow number %d (variable %s) at line %d",
                             k, j, i, flow->varname, JDF_OBJECT_LINENO(flow));
                    if( jdf_sanity_check_expr_bound(e, kind, f) < 0 )
                        rc = -1;
                    k++;
                }
                if( dep->guard->guard_type == JDF_GUARD_TERNARY ) {
                    k = 1;
                    for(e = dep->guard->callfalse->parameters; e != NULL; e = e->next) {
                        snprintf(kind, 128,
                                 "Parameter %d of dependency %d (when guard false)\n"
                                 "  of dataflow number %d (variable %s) at line %d",
                                 k, j, i,  flow->varname, JDF_OBJECT_LINENO(flow));
                        if( jdf_sanity_check_expr_bound(e, kind, f) < 0 )
                            rc = -1;
                        k++;
                    }
                }
                j++;
            }
            i++;
        }
    }
    return rc;
}

static int jdf_sanity_check_dataflow_naming_collisions(void)
{
    int rc = 0;
    jdf_function_entry_t *f1, *f2;
    jdf_dataflow_t *flow;
    jdf_dep_t *dep;

    for(f1 = current_jdf.functions; f1 != NULL; f1 = f1->next) {
        for(f2 = current_jdf.functions; f2 != NULL; f2 = f2->next) {
            for(flow = f2->dataflow; flow != NULL; flow = flow->next) {
                for(dep = flow->deps; dep != NULL; dep = dep->next) {
                    if( !strcmp(dep->guard->calltrue->func_or_mem, f1->fname) &&
                        (dep->guard->calltrue->var == NULL) ) {
                        jdf_fatal(JDF_OBJECT_LINENO(dep),
                                  "%s is the name of a function (defined line %d):\n"
                                  "  it cannot be also used as a memory reference in function %s\n",
                                  f1->fname, JDF_OBJECT_LINENO(f1), f2->fname);
                        rc = -1;
                    }
                    if( dep->guard->guard_type == JDF_GUARD_TERNARY &&
                        !strcmp(dep->guard->callfalse->func_or_mem, f1->fname) &&
                        (dep->guard->callfalse->var == NULL) ) {
                        jdf_fatal(JDF_OBJECT_LINENO(dep),
                                  "%s is the name of a function (defined line %d):\n"
                                  "  it cannot be also used as a memory reference in function %s\n",
                                  f1->fname, JDF_OBJECT_LINENO(f1), f2->fname);
                        rc = -1;
                    }
                }
            }
        }
    }
    return rc;
}

static int jdf_sanity_check_in_out_flow_match( jdf_function_entry_t* fout,
                                               jdf_dataflow_t* flowout,
                                               jdf_call_t* callout)
{
    jdf_function_entry_t* fin;
    jdf_dataflow_t*  flowin;
    jdf_dep_t *depin;
    int matched = 0;

    /*printf("Investigate flow %s:%s -> %s:%s\n", fout->fname, flowout->varname,
      callout->var, callout->func_or_mem);*/
    for(fin = current_jdf.functions; fin != NULL; fin = fin->next) {
        if( strcmp(fin->fname, callout->func_or_mem) ) continue;

        /* found the function, let's find the data */
        for(flowin = fin->dataflow; flowin != NULL; flowin = flowin->next) {
            if( !strcmp(flowin->varname, callout->var) ) {
                break;
            }
        }

        /* Did we found the right out dependency? */
        if( NULL == flowin ) {
            jdf_fatal(JDF_OBJECT_LINENO(flowout),
                      "Function %s has no data named %s,\n"
                      "  but dependency %s:%s (line %d) references it\n",
                      fin->fname, callout->var, fout->fname, flowout->varname, JDF_OBJECT_LINENO(flowout));
            return -1;
        }

        for( depin = flowin->deps; depin != NULL; depin = depin->next ) {
            if( (depin->guard->calltrue->var != NULL) &&
                (0 == strcmp(depin->guard->calltrue->func_or_mem, fout->fname)) ) {
                matched = 1;
                break;
            }
            if( (depin->guard->guard_type == JDF_GUARD_TERNARY) &&
                (depin->guard->callfalse->var != NULL) &&
                (0 == strcmp(depin->guard->callfalse->func_or_mem, fout->fname)) ) {
                matched = 1;
                break;
            }
        }
        if( matched ) return 0;  /* we found it */
        jdf_fatal(JDF_OBJECT_LINENO(flowout),
                  "Function %s dependency %s toward %s:%s not matched on the %s side\n",
                  fout->fname, flowout->varname, fin->fname, callout->var, fin->fname);
        return -1;
    }
    jdf_fatal(JDF_OBJECT_LINENO(flowout),
              "There is no function named %s,\n"
              "  but dependency %s:%s (lineno %d) references it\n",
              callout->func_or_mem, fout->fname, flowout->varname, JDF_OBJECT_LINENO(flowout));
    return -1;
}

static int jdf_sanity_check_dataflow_unexisting_data(void)
{
    int rc = 0, matched;
    jdf_function_entry_t *f1;
    jdf_dataflow_t *flow1;
    jdf_dep_t *dep;
    jdf_call_t* call;
    int i, j;

    for(f1 = current_jdf.functions; f1 != NULL; f1 = f1->next) {
        i = 1;
        for(flow1 = f1->dataflow; flow1 != NULL; flow1 = flow1->next) {
            j = 1;
            matched = 0;
            for( dep = flow1->deps; dep != NULL; dep = dep->next ) {
                if( (dep->guard->calltrue->var != NULL) ) {
                    call = dep->guard->calltrue;
                    matched = jdf_sanity_check_in_out_flow_match( f1, flow1, call );
                    if( matched )
                        break;
                }

                if( (dep->guard->guard_type == JDF_GUARD_TERNARY) &&
                    (dep->guard->callfalse->var != NULL) ) {
                    call = dep->guard->callfalse;
                    matched = jdf_sanity_check_in_out_flow_match( f1, flow1, call );
                    if( matched )
                        break;
                }
                j++;
            }
            if( matched ) return -1;
            i++;
        }
    }

    return rc;
}

static int jdf_sanity_check_control(void)
{
    jdf_function_entry_t *func;
    jdf_dataflow_t *flow;
    jdf_dep_t *dep;
    int rc = 0, i, j;

    /* For all the functions */
    for(func = current_jdf.functions; func != NULL; func = func->next) {
        i = 1;
        /* For each flow of data */
        for(flow = func->dataflow; flow != NULL; flow = flow->next, i++) {
            if( JDF_FLOW_TYPE_CTL != flow->flow_flags ) continue;
            j = 1;
            /* For each CONTROL dependency */
            for( dep = flow->deps; dep != NULL; dep = dep->next, j++ ) {
                if( (dep->guard->calltrue->var == NULL) ||
                    ((dep->guard->guard_type == JDF_GUARD_TERNARY) &&
                     (dep->guard->callfalse->var == NULL)) ) {
                    rc++;
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "In function %s:%d the control of dependency #%d of flow %s(#%d) cannot refer to data\n",
                              func->fname, JDF_OBJECT_LINENO(flow), j, flow->varname, i );
                }
            }
        }
    }

    return rc;
}

static int compute_canonical_data_location(const char *name, const jdf_expr_t *p, char **_str, char **_canon)
{
    jdf_global_entry_t *g;
    char *str;
    char *canon;
    char *params;
    char *canon_base;
    jdf_expr_t *align;
    int ret;
    jdf_expr_t pseudo;

    params = malloc_and_dump_jdf_expr_list(p);

    str = (char*)malloc(strlen(name) + strlen(params) + 4);
    sprintf(str, "%s(%s)", name, params);

    /* Find if this variable is in the globals list */
    for( g = current_jdf.globals; g != NULL; g = g->next ) {
        if( !strcmp( g->name, name ) )
            break;
    }

    ret = 1;
    if( NULL != g ) {
        /* Find if it has a "aligned" property */
        align = jdf_find_property( g->properties, "aligned", NULL );
        if( align != NULL ) {
            /* Canonical representation exists */
            /* Check it is well formed */
            if ( align->op != JDF_VAR ) {
                jdf_warn(JDF_OBJECT_LINENO(g), "Attribute Aligned on variable %s is malformed: expected an identifier, got something else. Attribute ignored.\n",
                         name);
            } else {
                ret = 0;
                pseudo = *align;
                pseudo.next = NULL;
                canon_base = malloc_and_dump_jdf_expr_list( &pseudo );
                canon = (char*)malloc(strlen(canon_base) + strlen(params) + 4);
                sprintf(canon, "%s(%s)", canon_base, params );
                free(canon_base);
            }
        }
    }

    if( ret == 1 ) {
        /* There is no canonical representation for this data */
        /* Use the data itself */
        canon = strdup( str );
    }

    free( params );
    *_str = str;
    *_canon = canon;

    return ret;
}

static int jdf_sanity_check_call_compatible(const jdf_call_t *c,
                                            const jdf_dep_t *dep,
                                            const jdf_call_t *d,
                                            const jdf_expr_t *cond,
                                            const jdf_function_entry_t *f)
{
    int ret;
    char *cstr, *dstr;
    char *ccanon, *dcanon;
    jdf_expr_t plist;
    int ciscanon, discanon;
    char *condstr;

    /* Simple case: d is a call to another kernel, not a memory reference */
    if( NULL != d->var )
        return 0;

    ciscanon = compute_canonical_data_location( c->func_or_mem, c->parameters, &cstr, &ccanon );
    discanon = compute_canonical_data_location( d->func_or_mem, d->parameters, &dstr, &dcanon );

    if( strcmp(ccanon, dcanon) ) {
        /* d does not have the same representation as c..
         * There is a risk: depends on the data distribution...,
         *  on expression evaluations, etc...
         */
        if( cond ) {
            plist = *(jdf_expr_t *)cond;
            plist.next = NULL;
            condstr = malloc_and_dump_jdf_expr_list(&plist);

            jdf_warn(JDF_OBJECT_LINENO(dep),
                     "Function %s runs on a node depending on data %s%s%s%s, but refers directly (as %s) to data %s%s%s%s, if %s is true.\n"
                     "  This is a potential direct remote memory reference.\n"
                     "  To remove this warning, %s should be syntaxically equal to %s, or marked as aligned to %s\n"
                     "  If this is not possible, and data are located on different nodes at runtime, this will result in a fault.\n",
                     f->fname,
                     cstr, ciscanon ? "" : " (aligned with ", ciscanon ? "" : ccanon, ciscanon ? "" : ")",
                     (dep->dep_flags & JDF_DEP_FLOW_IN & JDF_DEP_FLOW_OUT) ? "INOUT" : ( (dep->dep_flags & JDF_DEP_FLOW_IN) ? "IN" : "OUT" ),
                     dstr, discanon ? "" : " (aligned with ", discanon ? "" : dcanon, discanon ? "" : ")",
                     condstr,
                     dstr, cstr, ccanon);

            free(condstr);
        } else {
            jdf_warn(JDF_OBJECT_LINENO(dep),
                     "Function %s runs on a node depending on data %s%s%s%s, but refers directly (as %s) to data %s%s%s%s.\n"
                     "  This is a potential direct remote memory reference.\n"
                     "  To remove this warning, %s should be syntaxically equal to %s, or marked as aligned to %s\n"
                     "  If this is not possible, and data are located on different nodes at runtime, this will result in a fault.\n",
                     f->fname,
                     cstr, ciscanon ? "" : " (aligned with ", ciscanon ? "" : ccanon, ciscanon ? "" : ")",
                     (dep->dep_flags & JDF_DEP_FLOW_IN & JDF_DEP_FLOW_OUT) ? "INOUT" : ( (dep->dep_flags & JDF_DEP_FLOW_IN) ? "IN" : "OUT" ),
                     dstr, discanon ? "" : " (aligned with ", discanon ? "" : dcanon, discanon ? "" : ")",
                     dstr, cstr, ccanon);
        }
        ret = 1;
    } else {
        ret = 0;
    }

    free(cstr);
    free(ccanon);
    free(dstr);
    free(dcanon);

    return ret;
}

static int jdf_sanity_check_remote_memory_references(void)
{
    int rc = 0;
    jdf_function_entry_t *f;
    jdf_call_t *c;
    jdf_dataflow_t *dl;
    jdf_dep_t *dep;
    jdf_guarded_call_t *g;
    jdf_expr_t not;

    not.op = JDF_NOT;

    for( f = current_jdf.functions; f != NULL; f = f->next) {
        c = f->predicate;

        /* Now, iterate on each of the dependencies of f,
         * and each of the calls of these dependencies,
         * and try to assert whether c is compatible...
         */
        for(dl = f->dataflow; dl != NULL; dl = dl->next) {
            for(dep = dl->deps; dep != NULL; dep = dep->next) {
                g = dep->guard;
                switch( g->guard_type ) {
                case JDF_GUARD_UNCONDITIONAL:
                case JDF_GUARD_BINARY:
                    if( jdf_sanity_check_call_compatible(c, dep, g->calltrue, g->guard, f) )
                        rc++;
                    break;
                case JDF_GUARD_TERNARY:
                    if( jdf_sanity_check_call_compatible(c, dep, g->calltrue, g->guard, f) )
                        rc++;
                    not.jdf_ua = g->guard;
                    if( jdf_sanity_check_call_compatible(c, dep, g->callfalse, &not, f) )
                        rc++;
                    break;
                }
            }
        }
    }
    return rc;
}

int jdf_sanity_checks( jdf_warning_mask_t mask )
{
    int rc = 0;
    int fatal = 0;
    int rcsum = 0;

#define DO_CHECK( call )                        \
    do {                                        \
        rc = (call);                            \
        if(rc < 0)                              \
            fatal = 1;                          \
        else                                    \
            rcsum += rc;                        \
    } while(0)

    DO_CHECK( jdf_sanity_check_global_redefinitions() );
    DO_CHECK( jdf_sanity_check_global_unbound() );
    if( mask & JDF_WARN_MASKED_GLOBALS ) {
        DO_CHECK( jdf_sanity_check_global_masked() );
    }

    DO_CHECK( jdf_sanity_check_function_redefinitions() );
    DO_CHECK( jdf_sanity_check_parameters_are_consistent_with_definitions() );
    DO_CHECK( jdf_sanity_check_definition_unbound() );

    DO_CHECK( jdf_sanity_check_predicates_unbound() );
    DO_CHECK( jdf_sanity_check_dataflow_expressions_unbound() );

    DO_CHECK( jdf_sanity_check_dataflow_naming_collisions() );
    DO_CHECK( jdf_sanity_check_dataflow_unexisting_data() );

    if( mask & JDF_WARN_REMOTE_MEM_REFERENCE ) {
        DO_CHECK( jdf_sanity_check_remote_memory_references() );
    }
    /* Check the control validity */
    DO_CHECK( jdf_sanity_check_control() );

#undef DO_CHECK

    if(fatal)
        return -1;
    return rcsum;
}

/**
 * Compare two expressions and return 0 if they are identical.
 */
static int jdf_compare_expr(const jdf_expr_t* ex1, const jdf_expr_t* ex2)
{
    int ret;

    if( ex1 == ex2 ) {ret = 0; goto print_and_return;}
    if( ex1->op != ex2->op ) {ret = 1; goto print_and_return;}
    if( JDF_OP_IS_CST(ex1->op) )
    {ret = !(ex1->jdf_cst == ex2->jdf_cst); goto print_and_return;}
    if( JDF_OP_IS_STRING(ex1->op) || JDF_OP_IS_VAR(ex1->op) )
    {ret = strcmp(ex1->jdf_var, ex2->jdf_var); goto print_and_return;}
    if( JDF_OP_IS_C_CODE(ex1->op) )
    {ret = strcmp(ex1->jdf_c_code.code, ex2->jdf_c_code.code); goto print_and_return;}
    if( JDF_OP_IS_UNARY(ex1->op) )
    {ret = jdf_compare_expr(ex1->jdf_ua, ex2->jdf_ua); goto print_and_return;}
    if( JDF_OP_IS_BINARY(ex1->op) )
    {ret = (jdf_compare_expr(ex1->jdf_ba1, ex2->jdf_ba1) &
            jdf_compare_expr(ex1->jdf_ba2, ex2->jdf_ba2)); goto print_and_return;}
    assert(JDF_OP_IS_TERNARY(ex1->op));
    ret = (jdf_compare_expr(ex1->jdf_ta1, ex2->jdf_ta1) &
           jdf_compare_expr(ex1->jdf_ta2, ex2->jdf_ta2) &
           jdf_compare_expr(ex1->jdf_ta3, ex2->jdf_ta3));
  print_and_return:
#if 0
    {
        string_arena_t* sa  = string_arena_new(64);
        string_arena_t* sa1 = string_arena_new(64);
        string_arena_t* sa2 = string_arena_new(64);
        expr_info_t linfo, rinfo;

        linfo.sa = sa1;
        linfo.prefix = "";
        linfo.assignments = "ex1";
        rinfo.sa = sa2;
        rinfo.prefix = "";
        rinfo.assignments = "ex1";
        string_arena_add_string(sa, "%s ex1:%s\n%s ex2:%s\n",
                                (0 == ret ? "==" : "!="), dump_expr((void**)ex1, &linfo),
                                (0 == ret ? "==" : "!="), dump_expr((void**)ex2, &rinfo));
        printf("%s\n", string_arena_get_string(sa));
        string_arena_free(sa);
        string_arena_free(sa1);
        string_arena_free(sa2);
    }
#endif
    return ret;
}

/**
 * Reorder the output dependencies to group together the ones using
 * identical datatypes. They will share an identical dep_datatype_index
 * which is the index of the smallest one. Compute the number of
 * different datatypes.
 */
static void jdf_reorder_dep_list_by_type(jdf_dataflow_t* flow,
                                         uint32_t* dep_index)
{
    uint32_t i, j, in_index, dep_count, swap_with;
    jdf_dep_t *dep, *sdep, **dep_array = NULL;
    jdf_datatransfer_type_t *ddt, *sddt;
    jdf_dep_t *deps_location = flow->deps;

    /**
     * Step 1: Transform the list of dependencies into an array, to facilitate
     *         the massaging.
     */
    for( dep_count = 0, dep = deps_location; NULL != dep; dep_count++, dep = dep->next );
    if( dep_count < 2 ) {
        if( 1 == dep_count ) {
            dep = deps_location;
            if( dep->dep_flags & JDF_DEP_FLOW_OUT ) {
                dep->dep_index = *dep_index;
                dep->dep_datatype_index = *dep_index;
                flow->flow_dep_mask |= (1 << dep->dep_index);
                (*dep_index)++;
            }
        }
        return;  /* nothing to reorder */
    }
    dep_array = (jdf_dep_t**)malloc(dep_count * sizeof(jdf_dep_t*));

    for( i = 0, dep = deps_location;
         NULL != dep;
         dep_array[i++] = dep, dep = dep->next);

    /**
     * Step 2: Rearrange the entries to bring all those using the same datatype
     *         together. First put all inputs at the begining followed by all the
     *         outputs. Then order the outputs based on the output type (including
     *         the CTL), so we can handle them easier when generating the code for
     *         releasing the output dependencies.
     */
    for( i = 0; i < dep_count; i++ ) {
        dep = dep_array[i];
        dep->dep_index          = *dep_index;  /* meaningless */
        dep->dep_datatype_index = *dep_index;  /* meaningless */
        if( dep->dep_flags & JDF_DEP_FLOW_IN ) continue;
        for( j = i+1; j < dep_count; j++ ) {
            sdep = dep_array[j];
            if( sdep->dep_flags & JDF_DEP_FLOW_IN ) {
                dep_array[j] = dep_array[i];
                dep_array[i] = sdep;
                /* Leave the outter loop at the next loop index so the field get initialized */
                break;
            }
        }
    }
    for( in_index = i = 0; i < dep_count; i++ ) {
        dep = dep_array[i];
        if( dep->dep_flags & JDF_DEP_FLOW_IN ) {
            dep->dep_index = in_index++;
            continue;  /* skip all the input dependencies */
        }
        dep->dep_index = *dep_index;
        dep->dep_datatype_index = *dep_index;
        flow->flow_dep_mask |= (1 << dep->dep_index);
        ddt = &dep->datatype;
        swap_with = i + 1;
        for( j = swap_with; j < dep_count; j++ ) {
            sdep = dep_array[j];
            sddt = &sdep->datatype;
            if( jdf_compare_expr(ddt->type,   sddt->type) )   continue;
            if( jdf_compare_expr(ddt->layout, sddt->layout) ) continue;
            if( jdf_compare_expr(ddt->count,  sddt->count) )  continue;
            if( jdf_compare_expr(ddt->displ,  sddt->displ) )  continue;
            /* Same output datatype and count, good to swap */
            if( swap_with != j ) {
                dep_array[j] = dep_array[swap_with];
                dep_array[swap_with] = sdep;
            }
            sdep->dep_datatype_index = *dep_index;
            sdep->dep_index = *dep_index;
            swap_with++;
        }
        i = swap_with - 1;  /* jump after the current group of dependencies (sharing the datatype) */
        (*dep_index)++;
    }
    /* Step 3: Rebuild the list */
    for( flow->deps = dep = dep_array[0], i = 1; i < dep_count; i++) {
        dep->next = dep_array[i];
        dep = dep->next;
    }
    dep->next = NULL;
    free(dep_array);
}


/**
 * Flatten all the flows of data for the specified function, by creating the indexes
 * and masks used to describe the flow of data and the index of the output dependency
 * in the context of the flow. For a flow multiple output dependencies sharing the
 * same caracteristics will be merged together. Each one of them will have it's own
 * flow, but they will all have the index of the smallest flow sharing the same
 * caracteristics.
 */
int jdf_flatten_function(jdf_function_entry_t* function)
{
    uint32_t flow_index = 0, dep_index = 0;
    jdf_dataflow_t* flow;
    jdf_dep_t *dep;

    for( flow = function->dataflow; NULL != flow; flow = flow->next, flow_index++ ) {

        jdf_reorder_dep_list_by_type(flow, &dep_index);
        if( (1U << dep_index) > 0x00FFFFFF /* should be DAGUE_ACTION_DEPS_MASK */) {
            jdf_fatal(JDF_OBJECT_LINENO(function),
                      "Function %s has too many output flow with different datatypes (up to 24 supported)\n",
                      function->fname);
            return -1;
        }

        flow->flow_index = (uint8_t)flow_index;
#if 1
        {
            expr_info_t linfo;
            string_arena_t* sa = string_arena_new(64);
            linfo.sa = sa;
            linfo.prefix = ":";
            linfo.assignments = "";
            for(dep = flow->deps; NULL != dep; dep = dep->next) {
                string_arena_init(sa);
                dump_expr((void**)dep->datatype.type, &linfo);

                printf("%s: %6s[idx %d, mask 0x%x] %2s %8d %8d <%s %s>\n", function->fname,
                       flow->varname, flow->flow_index, flow->flow_dep_mask,
                       (JDF_DEP_FLOW_OUT & dep->dep_flags ? "->" : "<-"),
                       dep->dep_index, dep->dep_datatype_index,
                       dep->guard->calltrue->func_or_mem, string_arena_get_string(sa));
            }
            string_arena_free(sa);
        }
#endif
    }
    return 0;
}
