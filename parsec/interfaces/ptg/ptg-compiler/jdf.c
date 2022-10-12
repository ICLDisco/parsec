/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <stdio.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <stdlib.h>
#include <assert.h>

#include "jdf.h"
#include "string_arena.h"
#include "jdf2c_utils.h"

jdf_t current_jdf;
int current_lineno;
int verbose_level = 0;

static char *FUNCTION_PROPERTY_KEYWORDS[] = {
    "profile",
    "high_priority",
    JDF_PROP_UD_MAKE_KEY_FN_NAME,
    JDF_PROP_UD_HASH_STRUCT_NAME,
    JDF_PROP_UD_NB_LOCAL_TASKS_FN_NAME,
    JDF_PROP_UD_STARTUP_TASKS_FN_NAME,
    JDF_PROP_UD_FIND_DEPS_FN_NAME,
    JDF_PROP_UD_ALLOC_DEPS_FN_NAME,
    JDF_PROP_UD_FREE_DEPS_FN_NAME,
    NULL
};

extern const char *yyfilename;

#if (defined(PARSEC_DEBUG_NOISIER))
#define DO_DEBUG_VERBOSE( VAL, ARG ) \
    if( verbose_level >= (VAL) ) { ARG; }
#else
#define DO_DEBUG_VERBOSE( VAL, ARG )
#endif

void jdf_dump_function_flows(jdf_function_entry_t* function, int expanded);

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
    jdf_function_entry_t *func;
    jdf_param_list_t *param;
    jdf_variable_list_t *d;
    int rc = 0;

    for(g = current_jdf.globals; g != NULL; g = g->next) {
        for(func = current_jdf.functions; func != NULL; func = func->next) {
            for(param = func->parameters; param != NULL; param = param->next) {
                if( !strcmp(param->name, g->name) ) {
                    jdf_warn(JDF_OBJECT_LINENO(func), "Global %s defined line %d is masked by the local parameter %s of function %s\n",
                             g->name, JDF_OBJECT_LINENO(g), param->name, func->fname);
                    rc++;
                }
            }
            for(d = func->locals; d != NULL; d = d->next) {
                if( !strcmp(d->name, g->name) ) {
                    jdf_warn(JDF_OBJECT_LINENO(d), "Global %s defined line %d is masked by the local definition of %s in function %s\n",
                             g->name, JDF_OBJECT_LINENO(g), d->name, func->fname);
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
    jdf_expr_t *nr;
    int rc = 0;
    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        if( NULL != (dot = strchr(vc, '.')) )
            *dot = '\0';
        /* also check for -> */
        if( NULL != (dot = strstr(vc, "->")) )
            *dot = '\0';
        for(g2 = current_jdf.globals; g2 != g1; g2 = g2->next) {
            if( !strcmp( vc, g2->name ) ) {
                break;
            }
        }
        if( g2 == g1 ) {
            for(nr = e->local_variables; NULL != nr; nr = nr->next) {
                if( !strcmp(nr->alias, vc) )
                    break;
            }
            if(NULL == nr) {
                jdf_fatal(JDF_OBJECT_LINENO(g1), "Global %s is defined using variable %s (in %s) which is unbound at this time\n",
                          g1->name, vc, e->jdf_var);
                rc = -1;
            }
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

/*
 * Link each parameter with the corresponding local for a function. Don't check for
 * correctness, just link each param with the first matching local.
 */
int jdf_link_params_and_locals(jdf_function_entry_t* f)
{
    jdf_param_list_t *param;
    jdf_variable_list_t *local;

    for(param = f->parameters; param != NULL; param = param->next) {
        for(local = f->locals; local != NULL; local = local->next) {
            if( 0 != strcmp(local->name, param->name) )
                continue;
            param->local = local;
            local->param = param;
            break;
        }
    }
    return 0;
}

static int jdf_sanity_check_parameters_are_consistent_with_definitions(void)
{
    jdf_function_entry_t *f;
    jdf_param_list_t *param;
    jdf_variable_list_t *local, *local2;
    int pi, rc = 0;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        pi = 1;
        for(param = f->parameters; param != NULL; param = param->next, pi++) {
            for(local = f->locals; local != NULL; local = local->next) {
                if( 0 != strcmp(local->name, param->name) ) {
                    continue;
                }

                if( param->local != local ) {
                    jdf_fatal(JDF_OBJECT_LINENO(f), "The definition of %s (%dth parameter of the %s function) appears more than once.\n",
                              param->name, pi, f->fname);
                    rc = -1;
                } else if( local->param != param ) {
                    jdf_fatal(JDF_OBJECT_LINENO(f), "The local %s:%d point to a different parameter than the %dth %s:%d.\n",
                              local->name, JDF_OBJECT_LINENO(local), pi, param->name, JDF_OBJECT_LINENO(param));
                    rc = -1;
                } else {
                    for(local2 = local->next; local2 != NULL; local2 = local2->next) {
                        if( !strcmp(local->name, local2->name) ) {
                            jdf_fatal(JDF_OBJECT_LINENO(f), "The definition of %s in function %s appears more than once (line %d).\n",
                                      local->name, f->fname, JDF_OBJECT_LINENO(local2));
                            rc = -1;
                        }
                    }
                }
            }
            if( NULL == param->local ) {
                jdf_fatal(JDF_OBJECT_LINENO(f), "Parameter %s of function %s is declared but no range is associated to it\n",
                          param->name, f->fname);
                rc = -1;
            }
        }

        pi = 1;
        for(local = f->locals; local != NULL; local = local->next, pi++) {
            if( NULL == local->param ) {
                if( local->expr->op == JDF_RANGE ) {
                    jdf_warn(JDF_OBJECT_LINENO(f), "Definition %d of function %s for %s is a range, but not a parameter of the function.\n"
                             "  If this range allows for more than one value, that would make multiple functions %s with the same name.\n",
                             pi, f->fname, local->name, f->fname);
                }
                continue;  /* this local is not linked to a parameter */
            }
            param = local->param;
        }
    }
    return rc;
}

static int jdf_sanity_check_expr_bound_before_definition(jdf_expr_t *e, jdf_function_entry_t *f, jdf_variable_list_t *v)
{
    jdf_global_entry_t *g;
    jdf_variable_list_t *v2;
    jdf_expr_t *nr;
    char *vc, *dot;
    int rc = 0;

    if(NULL != e->local_variables) {
        /* Use the opportunity to update the number of local definitions
         * that we may need to define this expression */
        int nb = 0;
        for(nr = e->local_variables; nr != NULL; nr = nr->next)
            if(NULL != nr->alias)
                nb++;
    }
    
    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        if( NULL != (dot = strchr(vc, '.')) )
            *dot = '\0';
        if( NULL != (dot = strstr(vc, "->")) )
            *dot = '\0';
        for(g = current_jdf.globals; g != NULL; g = g->next) {
            if( !strcmp( vc, g->name ) ) {
                break;
            }
        }
        if( g == NULL ) {
            for(nr = e->local_variables; NULL != nr; nr = nr->next) {
                if( !strcmp(nr->alias, vc) )
                    break;
            }
            if(NULL == nr) {
                for(v2 = f->locals; v2 != v; v2 = v2->next) {
                    if( !strcmp( vc, v2->name ) ) {
                        break;
                    }
                }
                if( v2 == v ) {
                    jdf_fatal(JDF_OBJECT_LINENO(v), "Local %s is defined using variable %s (in %s) which is unbound at this time\n",
                              v->name,  vc, e->jdf_var);
                    rc = -1;
                }
            }
        }
        free(vc);
        return rc;
    case JDF_CST:
        return 0;
    case JDF_TERNARY:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_tat, f, v) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ta1, f, v) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ta2, f, v) < 0 )
            rc = -1;
        return rc;
    case JDF_NOT:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ua, f, v) < 0 )
            rc = -1;
        return rc;
    case JDF_C_CODE:
        return 0;
    default:
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ba1, f, v) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_definition(e->jdf_ba2, f, v) < 0 )
            rc = -1;
        return rc;
    }
}

static int jdf_sanity_check_definition_unbound(void)
{
    int rc = 0;
    jdf_function_entry_t *f;
    jdf_variable_list_t *v;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        for(v = f->locals; v != NULL; v = v->next) {
            if( jdf_sanity_check_expr_bound_before_definition(v->expr, f, v) < 0 )
                rc = -1;
        }
    }
    return rc;
}

static int jdf_sanity_check_expr_bound(jdf_expr_t *e, const char *kind, jdf_function_entry_t *f)
{
    jdf_global_entry_t *g;
    jdf_variable_list_t *v;
    jdf_expr_t *nr;
    char *vc, *dot;
    int rc = 0;
    
    if(NULL != e->local_variables) {
        /* Use the opportunity to update the number of local definitions
         * that we may need to define this expression */
        int nb = 0;
        for(nr = e->local_variables; nr != NULL; nr = nr->next)
            if(NULL != nr->alias)
                nb++;
    }
    
    switch( e->op ) {
    case JDF_VAR:
        vc = strdup(e->jdf_var);
        if( NULL != (dot = strchr(vc, '.')) )
            *dot = '\0';
        if( NULL != (dot = strstr(vc, "->")) )
            *dot = '\0';
        for(g = current_jdf.globals; g != NULL; g = g->next) {
            if( !strcmp( vc, g->name ) ) {
                break;
            }
        }
        if( g == NULL ) {
            for(v = f->locals; v != NULL; v = v->next) {
                if( !strcmp( vc, v->name ) ) {
                    break;
                }
            }
            if( NULL == v ) {
                for(nr = e->local_variables; NULL != nr; nr = nr->next) {
                    if( !strcmp(nr->alias, vc) )
                        break;
                }
                if(NULL == nr) {
                    jdf_fatal(JDF_OBJECT_LINENO(f), "%s of function %s is defined using variable %s (in %s) which is unbound at this time\n",
                              kind, f->fname, vc, e->jdf_var);
                    rc = -1;
                }
            }
        }
        free(vc);
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
                if( NULL != dep->guard->calltrue ) {
                    for(e = dep->guard->calltrue->parameters; e != NULL; e = e->next) {
                        snprintf(kind, 128,
                                 "Parameter %d of dependency %d\n"
                                 "  of dataflow number %d (variable %s) at line %d",
                                 k, j, i, flow->varname, JDF_OBJECT_LINENO(flow));
                        if( jdf_sanity_check_expr_bound(e, kind, f) < 0 )
                            rc = -1;
                        k++;
                    }
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
    jdf_guarded_call_t *guard;

    for(f1 = current_jdf.functions; f1 != NULL; f1 = f1->next) {
        for(f2 = current_jdf.functions; f2 != NULL; f2 = f2->next) {
            for(flow = f2->dataflow; flow != NULL; flow = flow->next) {
                for(dep = flow->deps; dep != NULL; dep = dep->next) {
                    guard = dep->guard;
                    /* Special case for the arena definition for WRITE-only flows */
                    if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dep) ) {
                        if( JDF_GUARD_UNCONDITIONAL != guard->guard_type ) {
                            jdf_fatal(JDF_OBJECT_LINENO(dep),
                                      "expected WRITE-only expression with wrong type (internal error)\n");
                            rc = -1;
                        }
                        if( (JDF_FLOW_TYPE_CTL | JDF_FLOW_TYPE_READ) & flow->flow_flags ) {
                            jdf_fatal(JDF_OBJECT_LINENO(dep),
                                      "Incorrect dependency (CTL or READ) in a WRITE-only flow (internal error)\n");
                            rc = -1;
                        }
                        if( !(JDF_FLOW_TYPE_WRITE & flow->flow_flags) ) {
                            jdf_fatal(JDF_OBJECT_LINENO(dep),
                                      "Lack of dependency in a not WRITE-only flow (internal error)\n");
                            rc = -1;
                        }
                        continue;
                    }
                    if( !strcmp(guard->calltrue->func_or_mem, f1->fname) &&
                        (guard->calltrue->var == NULL) &&
                        (guard->calltrue->parameters != NULL)) {
                        jdf_fatal(JDF_OBJECT_LINENO(dep),
                                  "%s is the name of a function (defined line %d):\n"
                                  "  it cannot be also used as a memory reference in function %s\n",
                                  f1->fname, JDF_OBJECT_LINENO(f1), f2->fname);
                        rc = -1;
                    }
                    if( guard->guard_type == JDF_GUARD_TERNARY &&
                        !strcmp(guard->callfalse->func_or_mem, f1->fname) &&
                        (guard->callfalse->var == NULL) &&
                        (guard->callfalse->parameters != NULL)) {
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

static int jdf_sanity_check_flows_and_deps_number(void)
{
    jdf_function_entry_t *f;
    jdf_dataflow_t *flow;
    jdf_dep_t *dep;
    int rc = 0, flows_in, flows_out, deps_in, deps_out;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        flows_in = flows_out = 0;
        for(flow = f->dataflow; flow != NULL; flow = flow->next) {

            flows_out += !!(JDF_FLOW_TYPE_WRITE & flow->flow_flags);
            flows_in  += !!(JDF_FLOW_TYPE_READ & flow->flow_flags);

            deps_in = deps_out = 0;
            for(dep = flow->deps; dep != NULL; dep = dep->next) {
                deps_in += !!(JDF_DEP_FLOW_IN & dep->dep_flags);
                deps_out += !!(JDF_DEP_FLOW_OUT & dep->dep_flags);
            }
            if( MAX_DEP_IN_COUNT < deps_in ) {
                jdf_warn(JDF_OBJECT_LINENO(flow),
                         "Function %s: flow %s has too many (%d) input dependencies (max allowed by this PaRSEC build is %d)\n",
                         f->fname, flow->varname, deps_in, MAX_DEP_IN_COUNT);
                rc--;
            }
            if( MAX_DEP_OUT_COUNT < deps_out ) {
                jdf_warn(JDF_OBJECT_LINENO(flow),
                         "Function %s: flow %s has too many (%d) output dependencies (max allowed by this PaRSEC build is %d)\n",
                         f->fname, flow->varname, deps_out, MAX_DEP_OUT_COUNT);
                rc--;
            }
        }
        if( MAX_PARAM_COUNT < flows_in ) {
            jdf_warn(JDF_OBJECT_LINENO(f),
                     "Function %s: has too many (%d) input or READ flows (max allowed by this PaRSEC build is %d)\n",
                     f->fname, flows_in, MAX_PARAM_COUNT);
            rc--;
        }
        if( MAX_PARAM_COUNT < flows_out ) {
            jdf_warn(JDF_OBJECT_LINENO(f),
                     "Function %s: has too many (%d) output or WRITE flows (max allowed by this PaRSEC build is %d)\n",
                     f->fname, flows_out, MAX_PARAM_COUNT);
            rc--;
        }
    }
    return rc;
}

static int jdf_sanity_check_dataflow_type_consistency(void)
{
    int rc = 0, output_deps, input_deps, type_deps, met_complete_in_dep;
    jdf_function_entry_t *f;
    jdf_dataflow_t *flow;
    jdf_dep_t *dep;
    jdf_expr_t *disable_warning = NULL;

    for(f = current_jdf.functions; f != NULL; f = f->next) {
        for(flow = f->dataflow; flow != NULL; flow = flow->next) {

            if( JDF_FLOW_TYPE_CTL & flow->flow_flags ) {
                continue;  /* not much we can say about */
            }
            input_deps = output_deps = type_deps = 0;
            met_complete_in_dep = 0;
            for(dep = flow->deps; dep != NULL; dep = dep->next) {
                /* Check for datatype definition concistency: if a type and a layout are equal
                 * then the count must be 1 and the displacement must be zero. Generate a warning
                 * and replace the default if it's not the case.
                 */
                if( (disable_warning = jdf_find_property( dep->guard->properties, DISABLE_DEP_WARNING_PROPERTY_NAME, NULL)) != NULL ) {
                    if( (JDF_VAR == disable_warning->op) && (0 == strcasecmp("disable", disable_warning->jdf_var)) ) {
                        continue;
                    }
                }
                if( dep->datatype_remote.type == dep->datatype_remote.layout ) {
                    if( (JDF_CST != dep->datatype_remote.count->op) ||
                        ((JDF_CST == dep->datatype_remote.count->op) && (1 != dep->datatype_remote.count->jdf_cst))) {
                        jdf_warn(JDF_OBJECT_LINENO(dep),
                                 "Function %s: flow %s has the same layout and type but the count is not the"
                                 " expected constant 1. The generated code will abide by the input code.\n",
                                 f->fname, flow->varname);
                    }
                    if( (JDF_CST != dep->datatype_remote.displ->op) ||
                        ((JDF_CST == dep->datatype_remote.displ->op) && (0 != dep->datatype_remote.displ->jdf_cst))) {
                        jdf_warn(JDF_OBJECT_LINENO(dep),
                                 "Function %s: flow %s has the same layout and type but the displacement is not the"
                                 " expected constant 0. The generated code will abide by the input code.\n",
                                 f->fname, flow->varname);
                    }
                }
                if( dep->datatype_local.type == dep->datatype_local.layout ) {
                    if( (JDF_CST != dep->datatype_local.count->op) ||
                        ((JDF_CST == dep->datatype_local.count->op) && (1 != dep->datatype_local.count->jdf_cst))) {
                        jdf_warn(JDF_OBJECT_LINENO(dep),
                                 "Function %s: flow %s has the same layout and type but the count is not the"
                                 " expected constant 1. The generated code will abide by the input code.\n",
                                 f->fname, flow->varname);
                    }
                    if( (JDF_CST != dep->datatype_local.displ->op) ||
                        ((JDF_CST == dep->datatype_local.displ->op) && (0 != dep->datatype_local.displ->jdf_cst))) {
                        jdf_warn(JDF_OBJECT_LINENO(dep),
                                 "Function %s: flow %s has the same layout and type but the displacement is not the"
                                 " expected constant 0. The generated code will abide by the input code.\n",
                                 f->fname, flow->varname);
                    }
                }

                /* Special case for the arena definition for WRITE-only flows */
                if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dep) ) {
                    type_deps++;
                    continue;
                }
                if( JDF_DEP_FLOW_OUT & dep->dep_flags ) {
                    output_deps++;
                    continue;
                }
                if( JDF_DEP_FLOW_IN & dep->dep_flags ) {
                    input_deps++;
                    if( met_complete_in_dep ) {
                        jdf_warn(JDF_OBJECT_LINENO(dep),
                                 "Function %s: flow %s has a totally satisfiable input dependency at line %d."
                                 " All other input dependencies will be discarded in the generated code,"
                                 " including the dependency at line %d\n",
                                 f->fname, flow->varname, met_complete_in_dep, JDF_OBJECT_LINENO(dep));
                    }
                    switch( dep->guard->guard_type ) {
                    case JDF_GUARD_UNCONDITIONAL:
                    case JDF_GUARD_TERNARY:
                        met_complete_in_dep = JDF_OBJECT_LINENO(dep);
                        break;
                    default:  /* just to force some compilers to stop complaining */
                        break;
                    }
                    continue;
                }
            }
            if( (NULL != disable_warning) &&
                (JDF_VAR == disable_warning->op) &&
                (0 == strcasecmp("disable", disable_warning->jdf_var)) ) {
                continue;
            }
            if( JDF_FLOW_TYPE_WRITE & flow->flow_flags ) {
                /* We should have no IN dependencies, except for the arena assignment
                 * and at least one OUT dep */
                if( 0 == output_deps ) {
                    jdf_warn(JDF_OBJECT_LINENO(flow),
                             "Function %s: WRITE flow %s is missing an output deps.\n",
                             f->fname, flow->varname);
                    continue;
                }
                if( (JDF_FLOW_TYPE_READ & flow->flow_flags) && (0 == input_deps) ) {
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "Function %s: READ-WRITE flow %s without one input deps.\n",
                              f->fname, flow->varname);
                    rc--;
                }
            }
            if( JDF_FLOW_TYPE_READ & flow->flow_flags ) {
                /* We should not have any OUT dependencies but we should have at least one IN */
                if( 0 != type_deps ) {
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "Function %s: READ flow %s cannot have a type definition.\n",
                              f->fname, flow->varname);
                    rc--;
                }
                if( (JDF_FLOW_TYPE_WRITE & flow->flow_flags) && (0 == output_deps) ) {
                    jdf_warn(JDF_OBJECT_LINENO(flow),
                             "Function %s: Mismatch between the WRITE flow %s and its output dependencies (%s != output %d)\n",
                             f->fname, flow->varname, (JDF_FLOW_TYPE_WRITE & flow->flow_flags) ? "write":"read", output_deps);
                }
                if( 0 == input_deps ) {
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "Function %s: READ flow %s without one input deps.\n",
                              f->fname, flow->varname);
                    rc--;
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
    jdf_dep_t *dep;
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

        for( dep = flowin->deps; dep != NULL; dep = dep->next ) {
            /* Skip the default type declaration for WRITE-only dependencies */
            if( JDF_IS_DEP_WRITE_ONLY_INPUT_TYPE(dep) )
                continue;

            if( (dep->guard->calltrue->var != NULL) &&
                (dep->guard->calltrue->parameters != NULL) &&
                (0 == strcmp(dep->guard->calltrue->func_or_mem, fout->fname)) ) {
                matched = 1;
                break;
            }
            if( (dep->guard->guard_type == JDF_GUARD_TERNARY) &&
                (dep->guard->callfalse->var != NULL) &&
                (dep->guard->callfalse->parameters != NULL) &&
                (0 == strcmp(dep->guard->callfalse->func_or_mem, fout->fname)) ) {
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

                /* Skip the default type declaration for WRITE-only dependencies */
                if( (NULL == dep->guard->guard) && (NULL == dep->guard->calltrue) && (NULL == dep->guard->callfalse) )
                    continue;
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
            if( !(JDF_FLOW_TYPE_CTL & flow->flow_flags) ) continue;
            j = 1;
            /* For each CONTROL dependency */
            for( dep = flow->deps; dep != NULL; dep = dep->next, j++ ) {
                if( (dep->guard->calltrue->var == NULL) ||
                    ((dep->guard->guard_type == JDF_GUARD_TERNARY) &&
                     (dep->guard->callfalse->var == NULL)) ) {
                    jdf_fatal(JDF_OBJECT_LINENO(flow),
                              "In function %s:%d the control of dependency #%d of flow %s(#%d) cannot refer to data\n",
                              func->fname, JDF_OBJECT_LINENO(flow), j, flow->varname, i );
                    rc--;
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

    if( strcmp(ccanon, dcanon)
        && strcmp(dcanon, PARSEC_WRITE_MAGIC_NAME"()")
        && strcmp(dcanon, PARSEC_NULL_MAGIC_NAME"()") ) {
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

    memset(&not, 0, sizeof(jdf_expr_t));
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

    DO_CHECK( jdf_sanity_check_flows_and_deps_number() );
    DO_CHECK( jdf_sanity_check_dataflow_naming_collisions() );
    DO_CHECK( jdf_sanity_check_dataflow_type_consistency() );
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
    if( (NULL == ex1) || (NULL == ex2) ) {ret = 1; goto print_and_return;}
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
 * Compare two datatypes, and if any of their components are identical, return
 * 0.
 *
 * @return 0 if the datatypes are identical, 1 otherwise.
 */
static int
jdf_compare_datatype(const jdf_datatransfer_type_t* src,
                     const jdf_datatransfer_type_t* dst)
{
    if( jdf_compare_expr(src->type,   dst->type) )   return 1;
    if( jdf_compare_expr(src->layout, dst->layout) ) return 1;
    if( jdf_compare_expr(src->count,  dst->count) )  return 1;
    if( jdf_compare_expr(src->displ,  dst->displ) )  return 1;
    return 0;
}

#define COMPARE_EXPR(TYPE, SA, SET_IF_EQUAL)                            \
    do {                                                                \
        if( src->TYPE != dst->TYPE ) {  /* if they were replaced previously */ \
            DO_DEBUG_VERBOSE(3, ({                                      \
                        dump_expr((void**)src->TYPE, &linfo);           \
                        if( strlen(string_arena_get_string(linfo.sa)) ) \
                            string_arena_add_string((SA), "%s", string_arena_get_string(linfo.sa)); \
                        string_arena_add_string((SA), "<< AND >>");     \
                        dump_expr((void**)dst->TYPE, &linfo);           \
                        if( strlen(string_arena_get_string(linfo.sa)) ) \
                            string_arena_add_string((SA), "%s", string_arena_get_string(linfo.sa)); \
                    }));                                                \
                                                                        \
            if( 0 == jdf_compare_expr(src->TYPE, dst->TYPE) ) {         \
                if( JDF_OP_IS_C_CODE(dst->TYPE->op) && (NULL == dst->TYPE->jdf_c_code.fname) ) { \
                    int rc = asprintf(&dst->TYPE->jdf_c_code.fname, "same_type_as_line_%d", JDF_OBJECT_LINENO(dst->TYPE)); \
                    (void)rc;                                           \
                }                                                       \
                JDF_OBJECT_RELEASE(dst->TYPE);                          \
                dst->TYPE = src->TYPE;                                  \
                JDF_OBJECT_RETAIN(src->TYPE);                           \
            } else { (SET_IF_EQUAL) = 1; }                              \
            DO_DEBUG_VERBOSE(1, ({                                      \
                        printf( "%s at line %d is %s to %s at line %d (is%s inline_c, fname %s)%s\n", # TYPE, \
                                JDF_OBJECT_LINENO(src->TYPE), (0 == (SET_IF_EQUAL) ? "equal" : "different"), # TYPE, JDF_OBJECT_LINENO(dst->TYPE), \
                                (JDF_OP_IS_C_CODE(dst->TYPE->op) ? "" : " not"), \
                                dst->TYPE->jdf_c_code.fname, string_arena_get_string(sa)); \
                    }));                                                \
        }                                                               \
    } while (0)

/**
 * Compare two datatype and for each component that is identical release the one
 * from the destination and replace it with the one from the source. This will
 * allow to generate less code, as we can now reuse the same generated code
 * multiple times. This optimization can be done outside the boundaries of a
 * single flow, but it remains scoped to a single function (due to the order of
 * the locals in the task description).
 *
 * @return 0 if they are identical.
 */
static int
jdf_datatype_remove_redundancy(const jdf_datatransfer_type_t* src,
                               jdf_datatransfer_type_t* dst)
{
    int are_types_equal = 0, are_layout_equal = 0, are_count_equal = 0, are_displ_equal = 0;
    string_arena_t* sa = string_arena_new(64);
    string_arena_t* sa1 = string_arena_new(64);
    expr_info_t linfo;

    linfo.sa = sa1;
    linfo.prefix = ":";
    linfo.assignments = "";
    COMPARE_EXPR(type, sa, are_types_equal);
    if( src->layout == dst->layout ) {
        are_layout_equal = are_types_equal;  /* layouts depend on the type */
    } else if( (NULL != src->layout) && (NULL != dst->layout) ) {
        COMPARE_EXPR(layout, sa, are_layout_equal);
    }  /* otherwise we default to the initial 0 */
    COMPARE_EXPR(count, sa, are_count_equal);
    COMPARE_EXPR(displ, sa, are_displ_equal);

    string_arena_free(sa);
    string_arena_free(sa1);

    (void)linfo; /* Used with verbose=3 */

    /* the return is similar to strcmp: 0 stands for equality */
    return (are_types_equal | are_layout_equal | are_count_equal | are_displ_equal);
}

#define SAVE_AND_UPDATE_INDEX(DEP, IDX1, IDX2, UPDATE)                  \
    do {                                                                \
        if( 0xff == ((DEP)->dep_index) ) (DEP)->dep_index = (IDX1)++;   \
        if( 0xff == ((DEP)->dep_datatype_index) ) {                     \
            (DEP)->dep_datatype_index = (IDX2);                         \
            if(UPDATE) (IDX2)++;                                        \
        }                                                               \
    } while (0)

#define MARK_FLOW_DEP_AND_UPDATE_INDEX(FLOW, DEP, UPDATE)               \
    do {                                                                \
        if( (DEP)->dep_flags & JDF_DEP_FLOW_OUT ) {                     \
            SAVE_AND_UPDATE_INDEX((DEP), global_out_index, *dep_out_index, UPDATE); \
            (FLOW)->flow_dep_mask_out |= (1 << (DEP)->dep_index);       \
            (FLOW)->flow_flags |= JDF_FLOW_IS_OUT;                      \
        }                                                               \
        if( (DEP)->dep_flags & JDF_DEP_FLOW_IN ) {                      \
            SAVE_AND_UPDATE_INDEX((DEP), global_in_index, *dep_in_index, UPDATE); \
            (FLOW)->flow_dep_mask_in |= (1 << (DEP)->dep_index);        \
            (FLOW)->flow_flags |= JDF_FLOW_IS_IN;                       \
        }                                                               \
    } while(0)
#define UPDATE_INDEX(DEP)                                               \
    do {                                                                \
        if( (DEP)->dep_flags & JDF_DEP_FLOW_OUT ) {                     \
            (*dep_out_index)++;                                         \
        } else {                                                        \
            (*dep_in_index)++;                                          \
        }                                                               \
    } while(0)
#define COPY_INDEX(DST, SRC)                                    \
    (DST)->dep_datatype_index = (SRC)->dep_datatype_index


/* Function to check the datatype specified on the dependency.
 * Returns:
 * - DEP_UNDEFINED_DATATYPE if no datatype has been set up by the user.
 * - DEP_CUSTOM_DATATYPE otherwise. */
int jdf_dep_undefined_type(jdf_datatransfer_type_t datatype ){
    if( JDF_STRING == datatype.type->op ){
        return DEP_UNDEFINED_DATATYPE;
    }
    //Otherwise datatype can be JDF_VAR or also JDF_C_CODE
    return DEP_CUSTOM_DATATYPE;
}

#define APPEND(DEP)\
    do {                                                                \
        if(DEP->dep_flags & JDF_DEP_FLOW_IN){\
            if(input_begin == NULL){  input_begin = input_end = DEP; }else{ input_end->next = DEP; input_end = DEP;}\
        }else{\
            if(output_begin == NULL){ output_begin = output_end = DEP; }else{ output_end->next = DEP; output_end = DEP; }\
        }\
    } while(0)

/**
 * Reorder the output dependencies to group together the ones using
 * identical datatypes. They will share an identical dep_datatype_index
 * which is the index of the smallest one. Compute the number of
 * different datatypes.
 */
static void jdf_reorder_dep_list_by_type(jdf_dataflow_t* flow,
                                         uint32_t* dep_in_index,
                                         uint32_t* dep_out_index)
{
    uint32_t i, j, dep_count, saved_out_index;
    uint32_t global_in_index, global_out_index;
    jdf_dep_t *dep, *sdep, **dep_array = NULL;

    global_in_index  = *dep_in_index;
    global_out_index = saved_out_index = *dep_out_index;

    jdf_dep_t *input_end = NULL;
    jdf_dep_t *input_begin = NULL;
    jdf_dep_t *output_end = NULL;
    jdf_dep_t *output_begin = NULL;

    /**
     * Step 1: Transform the list of dependencies into an array, to facilitate
     *         the manipulation.
     */
    for( dep_count = 0, dep = flow->deps; NULL != dep; dep_count++, dep = dep->next ) {
        dep->dep_index          = 0xff;
        dep->dep_datatype_index = 0xff;
    }
    if( dep_count < 2 ) {
        if( 1 == dep_count ) {
            dep = flow->deps;
            MARK_FLOW_DEP_AND_UPDATE_INDEX(flow, dep, 1);
        }
        return;  /* nothing to reorder */
    }
    dep_array = (jdf_dep_t**)malloc(dep_count * sizeof(jdf_dep_t*));
    for( i = 0, dep = flow->deps; NULL != dep; dep_array[i++] = dep, dep = dep->next );

    /* Step 2: Rearrange by local type. Order all those dependencies using the same datatype
     *         together.
     *         Build two sublists (one for input dependencies and another one for output
     *         dependencies) ordered by local type.
     */
    for( i = 0; i < dep_count; i++ ) {
        dep = dep_array[i];
        if(dep == NULL) continue;
        APPEND(dep);
        int count = 0;
        for( j = i+1; j < dep_count; j++ ) {
            sdep = dep_array[j];
            if(sdep == NULL) continue;
            if( !((dep->dep_flags & sdep->dep_flags) & (JDF_DEP_FLOW_IN|JDF_DEP_FLOW_OUT)) ) break;
            if( jdf_compare_datatype(&dep->datatype_local, &sdep->datatype_local) ) continue;
            APPEND(sdep);
            dep_array[j] = NULL;
            count++;
        }
    }


    /* Step 3: Move to the beginning of the output dependency list the group of
     *         non-typed dependencies (local type).
     * Note /!\ type = DEFAULT is different than dependency without explicit type,
     * they will be two different groups of local type.
     */

    /* Terminate with NULL output dep list. */
    if(output_end != NULL) {
        output_end->next = NULL;
    }

    /* Locate non-typed group and put it at the begining. */
    if( (NULL != output_begin) && (DEP_UNDEFINED_DATATYPE != jdf_dep_undefined_type(output_begin->datatype_local))) {
        dep = output_begin;
        jdf_dep_t *pre_first_default = dep;
        jdf_dep_t *pre_last_default = NULL;
        while(dep != NULL){
            if( DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dep->datatype_local) ){/* No explicit type on dependency */
                pre_last_default = dep;
                while(pre_last_default->next != NULL){
                    if( DEP_UNDEFINED_DATATYPE != jdf_dep_undefined_type(pre_last_default->next->datatype_local) ){/* Explicit type on dependency */
                        break;
                    }
                    pre_last_default = pre_last_default->next;
                }
                sdep = output_begin;
                output_begin = pre_first_default->next;
                pre_first_default->next = pre_last_default->next;
                pre_last_default->next = sdep;
                break;
            }
            pre_first_default = dep;
            dep = dep->next;
        }
    }
    /* Build deps list. First the input dependencies, and then output ones.
     * First put all inputs at the beginning followed by all the
     * outputs.
     */
    if(input_begin != NULL) {
        flow->deps = input_begin;
        input_end->next = output_begin;
    } else {
        flow->deps = output_begin;
    }

    /* set up again array of */
    for( i = 0, dep = flow->deps; NULL != dep; dep_array[i++] = dep, dep = dep->next );

    /**
     * Step 4: For each group of local type, order the outputs based on the output type (including
     *         the CTL).
     *         Set up the dep_index and dep_datatype_index by remote type grouped by local type.
     *         As a result, those dependencies with same local and remote type will shared the
     *         same dep_datatype_index.
     *
     * Note, we are not actually reordering by remote type. The only implication of this is messier JDF
     * generated code.
     * The important thing is that we are rearranged by reshape type (we maximize reuse
     * of futures on output dependencies, thus, minimal number of reshapings) and that
     * dep_datatype_index is different when varying the combination <local type, remote type>
     * (thus, enough space on output to pass to the communication engine when sending data).
     */
    for( i = 0; i < dep_count; i++ ) {
        dep = dep_array[i];
        MARK_FLOW_DEP_AND_UPDATE_INDEX(flow, dep, 0);
        for( j = i+1; j < dep_count; j++ ) {
            sdep = dep_array[j];
            if(sdep == NULL) continue;
            if( !((dep->dep_flags & sdep->dep_flags) & (JDF_DEP_FLOW_IN|JDF_DEP_FLOW_OUT)) ) break;
            if( jdf_compare_datatype(&dep->datatype_local, &sdep->datatype_local) ) continue;
            if( jdf_compare_datatype(&dep->datatype_remote, &sdep->datatype_remote) ) continue;
            COPY_INDEX(sdep, dep);
        }
        UPDATE_INDEX(dep);
    }

    free(dep_array);
}

/**
 * Helper function to dump all the flows (input, output and control) of a function,
 * as seen by the compiler. If expanded is not 0 then each flow will be detailed,
 * otherwise only the point to the expr will be shown.
 *
 * This function is not used when DEBUG is not enabled, so it might generate a
 * compilation warning.
 */
void jdf_dump_function_flows(jdf_function_entry_t* function, int expanded)
{
    jdf_dataflow_t* flow;

    for( flow = function->dataflow; NULL != flow; flow = flow->next) {
        string_arena_t* sa1 = string_arena_new(64);
        string_arena_t* sa2 = string_arena_new(64);
        expr_info_t linfo;
        jdf_dep_t *dep;

        linfo.sa = sa1;
        linfo.prefix = ":";
        linfo.assignments = "";
        linfo.suffix = "";
        for(dep = flow->deps; NULL != dep; dep = dep->next) {
            string_arena_init(sa2);

            /* Local type */
            string_arena_add_string(sa2, "local_type = %p ", dep->datatype_local.type);
            if( expanded ) dump_expr((void**)dep->datatype_local.type, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s%s>", string_arena_get_string(sa1), (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dep->datatype_local))?"-UNDEF":"");

            if( dep->datatype_local.layout != dep->datatype_local.type ) {
                string_arena_add_string(sa2, " layout = %p ", dep->datatype_local.layout);
                if( expanded ) dump_expr((void**)dep->datatype_local.layout, &linfo);
                if( strlen(string_arena_get_string(sa1)) )
                    string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));
            }

            string_arena_add_string(sa2, " count = %p ", dep->datatype_local.count);
            if( expanded ) dump_expr((void**)dep->datatype_local.count, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));

            string_arena_add_string(sa2, " displ = %p ", dep->datatype_local.displ);
            if( expanded ) dump_expr((void**)dep->datatype_local.displ, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));

            /* Remote type */
            string_arena_add_string(sa2, "remote_type = %p ", dep->datatype_remote.type);
            if( expanded ) dump_expr((void**)dep->datatype_remote.type, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s%s>", string_arena_get_string(sa1), (DEP_UNDEFINED_DATATYPE == jdf_dep_undefined_type(dep->datatype_remote))?"-UNDEF":"");

            if( dep->datatype_remote.layout != dep->datatype_remote.type ) {
                string_arena_add_string(sa2, " layout = %p ", dep->datatype_remote.layout);
                if( expanded ) dump_expr((void**)dep->datatype_remote.layout, &linfo);
                if( strlen(string_arena_get_string(sa1)) )
                    string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));
            }

            string_arena_add_string(sa2, " count = %p ", dep->datatype_remote.count);
            if( expanded ) dump_expr((void**)dep->datatype_remote.count, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));

            string_arena_add_string(sa2, " displ = %p ", dep->datatype_remote.displ);
            if( expanded ) dump_expr((void**)dep->datatype_remote.displ, &linfo);
            if( strlen(string_arena_get_string(sa1)) )
                string_arena_add_string(sa2, "<%s>", string_arena_get_string(sa1));

            printf("%s: %6s[%1s%1s idx %d, mask 0x%x/0x%x] %2s dep_index %8d dep_dt_index %8d %p <%s %s>\n", function->fname,
                   flow->varname, (flow->flow_flags & JDF_FLOW_IS_IN ? "R" : " "),
                   (flow->flow_flags & JDF_FLOW_IS_OUT ? "W" : " "),
                   flow->flow_index, flow->flow_dep_mask_in, flow->flow_dep_mask_out,
                   (JDF_DEP_FLOW_OUT & dep->dep_flags ? "->" : "<-"),
                   dep->dep_index, dep->dep_datatype_index, dep,
                   dep->guard->calltrue->func_or_mem,
                   string_arena_get_string(sa2));
        }
        string_arena_free(sa1);
        string_arena_free(sa2);
    }
    printf("\n");
}
/**
 * Flatten all the flows of data for the specified function, by creating the
 * indexes and masks used to describe the flow of data and the index of the
 * output dependencies.  For all flows multiple output dependencies sharing the
 * same caracteristics will be merged together. Each one of them will have it's
 * own virtual flow, but they will share the index of the smallest flow with
 * the same caracteristics, index which correspond to the datatype location.
 *
 * In same time order the flows to push all flows with an output at the begining
 * of the list of flows. This enables the engine to use a single index for
 * managing the activation mask and the data output location.
 */
int jdf_flatten_function(jdf_function_entry_t* function)
{
    uint32_t flow_index, dep_in_index = 0, dep_out_index = 0;
    jdf_dataflow_t* flow;

    for( flow = function->dataflow; NULL != flow; flow = flow->next) {
        flow->flow_index  = 0xFF;
        jdf_reorder_dep_list_by_type(flow, &dep_in_index, &dep_out_index);
        if( ((1U << dep_in_index) > 0x1FFFFFFF /* should be ~PARSEC_DEPENDENCIES_BITMASK */) ||
            ((1U << dep_out_index) > 0x00FFFFFF /* should be PARSEC_ACTION_DEPS_MASK*/)) {
            jdf_fatal(JDF_OBJECT_LINENO(function),
                      "Function %s has too many input or output flow with different datatypes (up to 24 supported)\n",
                      function->fname);
            return -1;
        }
    }
    /* First name all the OUTPUT flows */
    for( flow_index = 0, flow = function->dataflow; NULL != flow; flow = flow->next )
        if( (flow->flow_flags & JDF_FLOW_IS_OUT) && !(flow->flow_flags & JDF_FLOW_TYPE_CTL) )
            flow->flow_index = flow_index++;
    /* And now name all the others (pure INPUT flows) */
    for( flow = function->dataflow; NULL != flow; flow = flow->next )
        if( (0xFF == flow->flow_index) && !(flow->flow_flags & JDF_FLOW_TYPE_CTL) )
            flow->flow_index = flow_index++;
    /* And last the CONTROL flows */
    for( flow = function->dataflow; NULL != flow; flow = flow->next )
        if( 0xFF == flow->flow_index ) {
            assert(flow->flow_flags & JDF_FLOW_TYPE_CTL);
            flow->flow_index = flow_index++;
        }
    /* Let's reorder the dataflow list based on the flow_index field */
    jdf_dataflow_t *parent, *reverse_order = NULL;
    flow_index = 0;
    while( NULL != (flow = function->dataflow) ) {
        parent = NULL;
        /* Find the right index */
        while( NULL != flow ) {
            if( flow_index == flow->flow_index )
                break;
            parent = flow;
            flow = flow->next;
        }
        assert(NULL != flow);
        /* Remove current (flow) from the previous chain */
        if( NULL !=  parent )
            parent->next = flow->next;
        else
            function->dataflow = flow->next;
        /* And add it into the new list that is ordered in reverse */
        flow->next = reverse_order;
        reverse_order = flow;
        flow_index++;
    }
    /* Reorder the reverse_order list to get the expected ordering */
    while( NULL != reverse_order ) {
        parent = reverse_order->next;
        reverse_order->next = function->dataflow;
        function->dataflow = reverse_order;
        reverse_order = parent;
    }

    DO_DEBUG_VERBOSE(3, jdf_dump_function_flows(function, 1));

    for( flow = function->dataflow; NULL != flow; flow = flow->next ) {
        jdf_dataflow_t* sflow;
        jdf_dep_t *dep, *dep2;
        for( dep = flow->deps; NULL != dep; dep = dep->next ) {
            for( sflow = flow; NULL != sflow; sflow = sflow->next ) {
                if( sflow == flow ) dep2 = dep->next;
                else dep2 = sflow->deps;
                for( ; NULL != dep2; dep2 = dep2->next) {
                    jdf_datatype_remove_redundancy(&dep->datatype_remote, &dep2->datatype_remote);
                }
            }
        }
    }

    DO_DEBUG_VERBOSE(1, jdf_dump_function_flows(function, 0));

    return 0;
}

/**
 * Accessors to get typed properties (int and string).
 */
int jdf_property_get_int( const jdf_def_list_t* properties,
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

/**
 * Do not return a copy of the variable name, instead return directly the pointer.
 */
const char*jdf_property_get_string( const jdf_def_list_t* properties,
                                    const char* prop_name,
                                    const char* ret_if_not_found )
{
    jdf_def_list_t* property;
    jdf_expr_t* expr = jdf_find_property(properties, prop_name, &property);

    if( NULL != expr ) {
        if( JDF_OP_IS_VAR(expr->op) || JDF_OP_IS_STRING(expr->op) )
            return expr->jdf_var;
        printf("Warning: property %s defined at line %d only support ON/OFF\n",
               prop_name, JDF_OBJECT_LINENO(property));
    }
    return ret_if_not_found;  /* the expected default */
}

const char* jdf_property_get_function( const jdf_def_list_t* properties,
                                       const char* prop_name,
                                       const char* ret_if_not_found )
{
    jdf_def_list_t* property;
    jdf_expr_t* expr = jdf_find_property(properties, prop_name, &property);

    if( NULL != expr ) {
        if( JDF_OP_IS_C_CODE(expr->op) )
            return expr->jdf_c_code.fname;
        printf("Warning: property %s defined at line %d is not defined as a function.\n",
               prop_name, JDF_OBJECT_LINENO(property));
    }
    return ret_if_not_found;  /* the expected default */
}

jdf_def_list_t *jdf_add_function_property(jdf_def_list_t **properties, const char *prop_name, const char *prop_value)
{
    jdf_def_list_t* assign    = calloc(1, sizeof(jdf_def_list_t));
    assign->properties        = NULL;
    assign->name              = strdup(prop_name);
    assign->expr              = calloc(1, sizeof(jdf_expr_t));
    assign->expr->op = JDF_C_CODE;
    assign->expr->jdf_c_code.code = NULL;
    assign->expr->jdf_c_code.lineno = -1;
    /* This will be set by the upper level parsing if necessary */
    assign->expr->jdf_c_code.function_context = NULL;
    assign->expr->jdf_c_code.fname = strdup(prop_value);
    JDF_OBJECT_ONAME(assign->expr) = strdup(prop_value);
    JDF_OBJECT_LINENO(assign) = -1;
    assign->next = *properties;
    *properties = assign;
    return assign;
}

jdf_def_list_t *jdf_add_string_property(jdf_def_list_t **properties, const char *prop_name, const char *prop_value)
{
    jdf_def_list_t* assign    = calloc(1, sizeof(jdf_def_list_t));
    assign->properties        = NULL;
    assign->name              = strdup(prop_name);
    assign->expr              = calloc(1, sizeof(jdf_expr_t));
    assign->expr->op          = JDF_VAR;
    assign->expr->jdf_var     = strdup(prop_value);
    /* JDF_OBJECT_ONAME(assign->expr) = strdup(prop_value); */
    JDF_OBJECT_LINENO(assign) = -1;
    assign->next = *properties;
    *properties = assign;
    return assign;
}

int jdf_function_property_is_keyword(const char *name)
{
    char **p = FUNCTION_PROPERTY_KEYWORDS;
    while(*p != NULL) {
        if (strcmp( *p, name) == 0)
            return 1;
        p++;
    }
    return 0;
}

int jdf_assign_ldef_index(jdf_function_entry_t *f)
{
    int nb_ldef_for_locals, nb_ldef_for_deps, nb_ldef_for_calls = 0;
    jdf_expr_t *ld;
    jdf_variable_list_t *vl;
    jdf_dataflow_t *fl;
    jdf_dep_t *dep;

    f->nb_max_local_def = 0;
    
    /* Local definitions can appear either in the locals or in the deps/calls.
     *  If they appear in the locals, they need to have a unique position
     *  If they appear in the dataflow, each dep can re-use the locals of another dep
     *                                  each call can re-use the locals of another call
     */

    DO_DEBUG_VERBOSE(2, ({fprintf(stderr, "Indexing task class %s\n", f->fname);}) );
    
    for(vl = f->locals; NULL != vl; vl = vl->next) {
        for( ld = vl->expr->local_variables; ld != NULL; ld = ld->next ) {
            assert(NULL != ld->alias);
            if( ld->ldef_index == -1 ) {
                ld->ldef_index = f->nb_max_local_def;
                f->nb_max_local_def++;
                DO_DEBUG_VERBOSE(2, ({ fprintf(stderr, "  local %s, ldef %s is at %d\n", vl->name, ld->alias, ld->ldef_index); }) );
            }
        }
    }

    nb_ldef_for_locals = f->nb_max_local_def;
    for(fl = f->dataflow; NULL != fl; fl = fl->next) {
        int depi = 0;
        for(dep = fl->deps; NULL != dep; dep = dep->next, depi++) {
            nb_ldef_for_deps = nb_ldef_for_locals;
            for(ld = dep->local_defs; NULL != ld; ld = ld->next) {
                assert(NULL != ld->alias);
                if( ld->ldef_index == -1 ) {
                    ld->ldef_index = nb_ldef_for_deps;
                    nb_ldef_for_deps++;
                    DO_DEBUG_VERBOSE(2, ({ fprintf(stderr, "  Flow for %s, dep %d: ldef %s is at %d\n", fl->varname, depi, ld->alias, ld->ldef_index); }) );
                }
            }
            switch( dep->guard->guard_type ) {
            case JDF_GUARD_UNCONDITIONAL:
            case JDF_GUARD_BINARY:
                nb_ldef_for_calls = nb_ldef_for_deps;
                for(ld = dep->guard->calltrue->local_defs; NULL != ld; ld = ld->next) {
                    assert(NULL != ld->alias);
                    if( ld->ldef_index == -1 ) {
                        ld->ldef_index = nb_ldef_for_calls;
                        nb_ldef_for_calls++;
                        DO_DEBUG_VERBOSE(2, ({ fprintf(stderr, "  Flow for %s, dep %d, calltrue: ldef %s is at %d\n", fl->varname, depi, ld->alias, ld->ldef_index); }) );
                    }
                }
                break;
            case JDF_GUARD_TERNARY:
                nb_ldef_for_calls = nb_ldef_for_deps;
                for(ld = dep->guard->calltrue->local_defs; NULL != ld; ld = ld->next) {
                    assert(NULL != ld->alias);
                    if( ld->ldef_index == -1 ) {
                        ld->ldef_index = nb_ldef_for_calls;
                        nb_ldef_for_calls++;
                        DO_DEBUG_VERBOSE(2, ({ fprintf(stderr, "  Flow for %s, dep %d, calltrue: ldef %s is at %d\n", fl->varname, depi, ld->alias, ld->ldef_index); }) );
                    }
                }
                nb_ldef_for_calls = nb_ldef_for_deps;
                for(ld = dep->guard->callfalse->local_defs; NULL != ld; ld = ld->next) {
                    assert(NULL != ld->alias);
                    if( ld->ldef_index == -1 ) {
                        ld->ldef_index = nb_ldef_for_calls;
                        nb_ldef_for_calls++;
                        DO_DEBUG_VERBOSE(2, ({ fprintf(stderr, "  Flow for %s, dep %d, callfalse: ldef %s is at %d\n", fl->varname, depi, ld->alias, ld->ldef_index); }) );
                    }
                }
                break;
            }
            if( nb_ldef_for_deps > f->nb_max_local_def )
                f->nb_max_local_def = nb_ldef_for_deps;
            if( nb_ldef_for_calls > f->nb_max_local_def )
                f->nb_max_local_def = nb_ldef_for_calls;
        }
    }
    return 0;
}
