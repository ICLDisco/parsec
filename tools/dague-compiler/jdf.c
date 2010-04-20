#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "jdf.h"

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
    current_jdf.preambles = NULL;
    current_jdf.globals   = NULL;
    current_jdf.functions = NULL;
    current_lineno = 1;
}

static int jdf_sanity_check_global_redefinitions(void)
{
    jdf_global_entry_t *g1, *g2;
    int rc = 0;

    for(g1 = current_jdf.globals; g1 != NULL; g1 = g1->next) {
        for(g2 = g1->next; g2 != NULL; g2 = g2->next) {
            if( !strcmp(g1->name, g2->name) ) {
                jdf_fatal(g2->lineno, "Global %s is redefined here (previous definition was on line %d)\n",
                          g1->name, g1->lineno);
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
                    jdf_warn(f->lineno, "Global %s defined line %d is masked by the local parameter %s of function %s\n",
                             g->name, g->lineno, n->name, f->fname);
                    rc++;
                }
            }
            for(d = f->definitions; d != NULL; d = d->next) {
                if( !strcmp(d->name, g->name) ) {
                    jdf_warn(d->lineno, "Global %s defined line %d is masked by the local definition of %s in function %s\n",
                             g->name, g->lineno, d->name, f->fname);
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
    int rc = 0;
    switch( e->op ) {
    case JDF_VAR:
        for(g2 = current_jdf.globals; g2 != g1; g2 = g2->next) {
            if( !strcmp( e->jdf_var, g2->name ) ) {
                break;
            }
        }
        if( g2 == g1 ) {
            jdf_fatal(g1->lineno, "Global %s is defined using variable %s which is unbound at this time\n",
                      g1->name, e->jdf_var);
            rc = -1;
        }
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
    default:
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_ba1, g1) < 0 )
            rc = -1;
        if( jdf_sanity_check_expr_bound_before_global(e->jdf_ba2, g1) < 0 )
            rc = 1;
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

int jdf_sanity_checks( jdf_warning_mask_t mask )
{
    int rc = 0;
    int fatal = 0;
    int rcsum = 0;

#define DO_CHECK( call ) do {                       \
    rc = call;                                      \
    if(rc < 0)                                      \
        fatal = 1;                                  \
    else                                            \
        rcsum += rc;                                \
    } while(0)

    DO_CHECK( jdf_sanity_check_global_redefinitions() );
    DO_CHECK( jdf_sanity_check_global_unbound() );
    if( mask & JDF_WARN_MASKED_GLOBALS )
        DO_CHECK( jdf_sanity_check_global_masked() );

#undef DO_CHECK

    if(fatal)
        return -1;
    return rc;
}
