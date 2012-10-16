/*
 * Copyright (c) 2012     The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "jdf.h"

static int jdf_expr_complete_unparse( const jdf_expr_t *e, FILE *out );

static int jdf_expr_unparse_bop(const jdf_expr_t *a1, const char *op, const jdf_expr_t *a2, FILE *out)
{
    int err = 0;
    fprintf(out, "(");
    err = jdf_expr_complete_unparse(a1, out);
    if( err < 0 )
        return err;
    fprintf(out, ") %s (", op);
    err = jdf_expr_complete_unparse(a2, out);
    fprintf(out, ")");
    return err;
}

static int jdf_expr_complete_unparse( const jdf_expr_t *e, FILE *out )
{
    int err = 0;

    switch( e->op ) {
    case JDF_EQUAL:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "==", e->jdf_ba2, out);
        break;

    case JDF_NOTEQUAL:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "!=", e->jdf_ba2, out);
        break;

    case JDF_AND:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "&&", e->jdf_ba2, out);
        break;

    case JDF_OR:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "||", e->jdf_ba2, out);
        break;

    case JDF_XOR:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "^", e->jdf_ba2, out);
        break;

    case JDF_LESS:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "<", e->jdf_ba2, out);
        break;

    case JDF_LEQ:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "<=", e->jdf_ba2, out);
        break;

    case JDF_MORE:
        err = jdf_expr_unparse_bop(e->jdf_ba1, ">", e->jdf_ba2, out);
        break;

    case JDF_MEQ:
        err = jdf_expr_unparse_bop(e->jdf_ba1, ">=", e->jdf_ba2, out);
        break;

    case JDF_NOT:
        fprintf(out, "!(");
        err = jdf_expr_complete_unparse(e->jdf_ua, out);
        fprintf(out, ")");
        break;

    case JDF_PLUS:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "+", e->jdf_ba2, out);
        break;

    case JDF_MINUS:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "-", e->jdf_ba2, out);
        break;

    case JDF_TIMES:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "*", e->jdf_ba2, out);
        break;

    case JDF_DIV:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "/", e->jdf_ba2, out);
        break;

    case JDF_MODULO:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "%", e->jdf_ba2, out);
        break;

    case JDF_SHL:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "<<", e->jdf_ba2, out);
        break;

    case JDF_SHR:
        err = jdf_expr_unparse_bop(e->jdf_ba1, ">>", e->jdf_ba2, out);
        break;

    case JDF_RANGE:
        err = jdf_expr_unparse_bop(e->jdf_ba1, "..", e->jdf_ba2, out);
        break;

    case JDF_TERNARY:
        fprintf(out, "(");
        err = jdf_expr_complete_unparse(e->jdf_ta1, out);
        fprintf(out, ") ? (");
        err |= jdf_expr_complete_unparse(e->jdf_ta2, out);
        fprintf(out, ")");
        if( e->jdf_ta3 ) {
            fprintf(out, ":(");
            err |= jdf_expr_complete_unparse(e->jdf_ta3, out);
            fprintf(out, ")");
        }
        break;

    case JDF_VAR:
        fprintf(out, "%s", e->jdf_var);
        break;

    case JDF_STRING:
        fprintf(out, "\"%s\"", e->jdf_var);
        break;

    case JDF_CST:
        fprintf(out, "%d", e->jdf_cst);
        break;

    case JDF_C_CODE:
        fprintf(out, "inline_c %%{ %s %%}", e->jdf_c_code.code);
        break;
    }

    if( (e->next != NULL) && (err >= 0) ) {
        fprintf(out, ", ");
        return jdf_expr_complete_unparse(e->next, out);
    }

    return err;
}

static int jdf_def_list_unparse( const jdf_def_list_t *defs, FILE *out, const char *sep )
{
    const jdf_def_list_t *dl;
    int err = 0;

    for(dl = defs; dl != NULL; dl = dl->next) {
        fprintf(out, "%s = ", dl->name);
        err = jdf_expr_complete_unparse(dl->expr, out);
        if( err < 0 )
            return err;
        if( dl->next != NULL )
            fprintf(out, "%s", sep);
    }
    return err;
}

static int jdf_properties_unparse( const jdf_def_list_t *defs, FILE *out )
{
    int err = 0;
    if( defs == NULL )
        return err;
    fprintf(out, "[");
    err = jdf_def_list_unparse(defs, out, " ");
    fprintf(out, "]");
    return err;
}

static int jdf_global_entry_unparse( const jdf_global_entry_t *e, FILE *out )
{
    int err = 0;

    if( NULL == e )
        return err;

    if( e->expression != NULL ) {
        fprintf(out, "%s = ", e->name);
        err = jdf_expr_complete_unparse( e->expression, out );
        if (err < 0 )
            return err;
    } else {
        fprintf(out, "%-9s ", e->name);
    }
    err = jdf_properties_unparse( e->properties, out );
    fprintf(out, "\n");

    if( err >= 0 )
        err = jdf_global_entry_unparse( e->next, out );
    return err;
}

static int jdf_name_list_unparse(const jdf_name_list_t *nl, FILE *out)
{
    int err = 0;
    const jdf_name_list_t *e;
    for(e = nl; e != NULL; e = e->next) {
        fprintf(out, "%s%s", e->name, e->next != NULL ? ", " : "");
    }
    return err;
}

static int jdf_call_unparse(const jdf_call_t *call, FILE *out)
{
    int err = 0;

    if( call->var ) {
        fprintf(out, "%s ", call->var);
    }
    fprintf(out, "%s(", call->func_or_mem);
    err = jdf_expr_complete_unparse(call->parameters, out);
    fprintf(out, ")");

    return err;
}

static int jdf_datatransfer_type_unparse(jdf_datatransfer_type_t dt, FILE *out)
{
    int err = 0;
    if( dt.simple ) {
        if( dt.u.simple_name != NULL && strcmp(dt.u.simple_name, "DEFAULT") ) {
            fprintf(out, "[type = %s", dt.u.simple_name);
            if( dt.nb_elt != NULL &&
                dt.nb_elt->op != JDF_CST &&
                dt.nb_elt->jdf_cst != 1 ) {
                fprintf(out, " nb_elt = ");
                err = jdf_expr_complete_unparse(dt.nb_elt, out);
            }
            fprintf(out, "]");
        } else {
            if( dt.nb_elt != NULL &&
                dt.nb_elt->op != JDF_CST &&
                dt.nb_elt->jdf_cst != 1 ) {
                fprintf(out, "[nb_elt = ");
                err = jdf_expr_complete_unparse(dt.nb_elt, out);
                fprintf(out, "]");
            }
        }
    } else {
        if( dt.u.complex_expr != NULL ) {
            fprintf(out, "[type_index = ");
            err = jdf_expr_complete_unparse(dt.u.complex_expr, out);
            if( err >= 0 &&
                dt.nb_elt != NULL &&
                dt.nb_elt->op != JDF_CST &&
                dt.nb_elt->jdf_cst != 1 ) {
                fprintf(out, " nb_elt = ");
                err = jdf_expr_complete_unparse(dt.nb_elt, out);
            }
            fprintf(out, "]");
        } else {
            if( dt.nb_elt != NULL &&
                dt.nb_elt->op != JDF_CST &&
                dt.nb_elt->jdf_cst != 1 ) {
                fprintf(out, "[nb_elt = ");
                err = jdf_expr_complete_unparse(dt.nb_elt, out);
                fprintf(out, "]");
            }
        }
    }
    return err;
}

static int jdf_guarded_call_unparse(const jdf_guarded_call_t *g, FILE *out)
{
    int err = 0;

    assert( NULL == g->properties );

    switch( g->guard_type ) {
    case JDF_GUARD_UNCONDITIONAL:
        return jdf_call_unparse(g->calltrue, out);
    case JDF_GUARD_BINARY:
        fprintf(out, "(");
        err = jdf_expr_complete_unparse(g->guard, out);
        fprintf(out, ") ? ");
        err |= jdf_call_unparse(g->calltrue, out);
        return err;
    case JDF_GUARD_TERNARY:
        fprintf(out, "(");
        err = jdf_expr_complete_unparse(g->guard, out);
        fprintf(out, ") ? ");
        err = jdf_call_unparse(g->calltrue, out);
        fprintf(out, " : ");
        err |= jdf_call_unparse(g->callfalse, out);
        return err;
        break;
    default:
        fprintf(stderr, "Improbable guard type %d (neither a unconditional, binary or ternary)\n", (int)(g->guard_type));
        return -1;
    }
}

static int jdf_deps_unparse( const jdf_dep_t *deps, FILE *out )
{
    int err = 0;

    if( NULL == deps )
        return err;

    if( deps->type == JDF_DEP_TYPE_IN ) {
        fprintf(out, "<- ");
    } else if( deps->type == JDF_DEP_TYPE_OUT ) {
        fprintf(out, "-> ");
    } else {
        fprintf(stderr, "Improbable dependency type %x is not IN xor OUT\n",
                deps->type);
        return -1;
    }

    err = jdf_guarded_call_unparse( deps->guard, out );
    if( err < 0 )
        return err;

    err = jdf_datatransfer_type_unparse( deps->datatype, out );
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    if( deps->next == NULL )
        return err;

    fprintf(out, "             ");
    return jdf_deps_unparse( deps->next, out );
}

static int jdf_dataflow_unparse( const jdf_dataflow_t *dataflow, FILE *out )
{
    int err = 0;

    if( NULL == dataflow )
        return err;

    if( dataflow->access_type == JDF_VAR_TYPE_CTL ) {
        fprintf(out, "  CTL   ");
    } else if( dataflow->access_type == JDF_VAR_TYPE_READ ) {
        fprintf(out, "  READ  ");
    } else if( dataflow->access_type == JDF_VAR_TYPE_WRITE ) {
        fprintf(out, "  WRITE ");
    } else if( dataflow->access_type == (JDF_VAR_TYPE_READ | JDF_VAR_TYPE_WRITE) ) {
        fprintf(out, "  RW    ");
    } else {
        fprintf(stderr, "Improbable flow access type %x is not CTL, READ, WRITE or RW\n", dataflow->access_type);
        return -1;
    }

    fprintf(out, "%-4s ", dataflow->varname);

    err = jdf_deps_unparse( dataflow->deps, out );
    if( err < 0 )
        return err;

    return jdf_dataflow_unparse( dataflow->next, out);
}

static int jdf_function_entry_unparse( const jdf_function_entry_t *f, FILE *out )
{
    int err = 0;
    if( NULL == f )
        return err;

    fprintf(out, "%s(", f->fname);
    err = jdf_name_list_unparse(f->parameters, out);
    fprintf(out, ")");
    if( err < 0 )
        return err;
    fprintf(out, " ");
    err = jdf_properties_unparse(f->properties, out);
    if(err < 0)
        return err;
    fprintf(out, "\n");

    fprintf(out, "  /* Execution Space */\n  ");
    err = jdf_def_list_unparse(f->locals, out, "\n  ");
    fprintf(out, "\n");
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    if( f->simcost ) {
        fprintf(out, "SIMCOST ");
        err = jdf_expr_complete_unparse( f->simcost, out );
        fprintf(out, "\n");
        if( err < 0 )
            return err;
        fprintf(out, "\n");
    }

    fprintf(out, "  /* Locality */\n");
    fprintf(out, "  : ");
    err = jdf_call_unparse(f->predicate, out);
    fprintf(out, "\n");
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    err = jdf_dataflow_unparse( f->dataflow, out );
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    if( f->priority ) {
        fprintf(out, "; ");
        err = jdf_expr_complete_unparse( f->priority, out );
        fprintf(out, "\n");
        if( err < 0 )
            return err;
        fprintf(out, "\n");
    }

    fprintf(out, "BODY\n");
    fprintf(out, "{\n");
    fprintf(out, "%s\n", f->body);
    fprintf(out, "}\n");
    fprintf(out, "END\n\n");

    return jdf_function_entry_unparse( f->next, out );
}

int jdf_unparse( const jdf_t *jdf, FILE *out )
{
    int err = 0;

    if( jdf->prologue && jdf->prologue->external_code )
        fprintf(out, "extern \"C\" %%{\n%s\n%%}\n", jdf->prologue->external_code );
    else {
        fprintf(stderr,
                "**Warning** Malformed JDF structure: a prologue is mandatory in the grammar...\n");
        err = 1;
    }
    fprintf(out, "\n");

    err = jdf_global_entry_unparse( jdf->globals, out );
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    err = jdf_properties_unparse( jdf->global_properties, out );
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    err = jdf_function_entry_unparse( jdf->functions, out );
    if( err < 0 )
        return err;
    fprintf(out, "\n");

    if( jdf->epilogue && jdf->epilogue->external_code )
        fprintf(out, "extern \"C\" {\n%s\n}\n", jdf->epilogue->external_code );

    return err;
}
