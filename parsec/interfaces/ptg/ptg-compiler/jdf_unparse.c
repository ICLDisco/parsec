/*
 * Copyright (c) 2012-2021 The University of Tennessee and The University
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

static int jdf_variable_list_unparse( const jdf_variable_list_t *locals, FILE *out, const char *sep )
{
    const jdf_variable_list_t *local;
    int err = 0;

    for(local = locals; local != NULL; local = local->next) {
        fprintf(out, "%s = ", local->name);
        err = jdf_expr_complete_unparse(local->expr, out);
        if( err < 0 )
            return err;
        if( local->next != NULL )
            fprintf(out, "%s", sep);
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

static int jdf_param_list_unparse(const jdf_param_list_t *pl, FILE *out)
{
    for(; NULL != pl; pl = pl->next) {
        fprintf(out, "%s%s", pl->name, NULL != pl->next ? ", " : "");
    }
    return 0;
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
    char start[1] = "[";

    if( (JDF_STRING == dt.type->op) || (JDF_VAR == dt.type->op) ) {
        if( strcmp(dt.type->jdf_var, "DEFAULT") ) {
            fprintf(out, "%s type = %s", start, dt.type->jdf_var);
            start[0] = '\0';
        }
    } else {
        fprintf(out, "%s type = ", start); start[0] = '\0';
        err = jdf_expr_complete_unparse(dt.type, out);
        if( 0 != err ) goto recover_and_exit;
    }

    if( dt.type != dt.layout ) {
        fprintf(out, "%s layout = ", start); start[0] = '\0';
        err = jdf_expr_complete_unparse(dt.layout, out);
        if( 0 != err ) goto recover_and_exit;

        if( !((JDF_CST == dt.count->op) && (1 == dt.count->jdf_cst)) ) {
            fprintf(out, "%s count = ", start); start[0] = '\0';
            err =  jdf_expr_complete_unparse(dt.count, out);
            if( 0 != err ) goto recover_and_exit;
        }

        if( !((JDF_CST == dt.displ->op) && (0 == dt.displ->jdf_cst)) ) {
            fprintf(out, "%s displ = ", start); start[0] = '\0';
            err =  jdf_expr_complete_unparse(dt.displ, out);
            if( 0 != err ) goto recover_and_exit;
        }
    }
recover_and_exit:
    if( '\0'== start[0] )
        fprintf(out, "]");
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

    if( deps->dep_flags & JDF_DEP_FLOW_IN ) {
        fprintf(out, "<- ");
    } else if( deps->dep_flags & JDF_DEP_FLOW_OUT ) {
        fprintf(out, "-> ");
    } else {
        fprintf(stderr, "Improbable dependency type %x is not IN xor OUT\n",
                deps->dep_flags);
        return -1;
    }

    err = jdf_guarded_call_unparse( deps->guard, out );
    if( err < 0 )
        return err;

    err = jdf_datatransfer_type_unparse( deps->datatype_remote, out );
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

    if( dataflow->flow_flags & JDF_FLOW_TYPE_CTL ) {
        fprintf(out, "  CTL   ");
    } else if( dataflow->flow_flags & JDF_FLOW_TYPE_READ ) {
        if( dataflow->flow_flags & JDF_FLOW_TYPE_WRITE ) {
            fprintf(out, "  RW    ");
        } else {
            fprintf(out, "  READ  ");
        }
    } else if( dataflow->flow_flags & JDF_FLOW_TYPE_WRITE ) {
        fprintf(out, "  WRITE ");
    } else {
        fprintf(stderr, "Improbable flow access type %x is not CTL, READ, WRITE or RW\n", dataflow->flow_flags);
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

    if( NULL != f->parameters &&
        NULL != f->locals     &&
        NULL != f->predicate  &&
        NULL != f->dataflow   &&
        NULL != f->bodies          ){

        fprintf(out, "%s(", f->fname);
        err = jdf_param_list_unparse(f->parameters, out);
        fprintf(out, ")");
        if( err < 0 )
            return err;

        if (f->properties != NULL) {
            fprintf(out, " ");
            err = jdf_properties_unparse(f->properties, out);
            if(err < 0)
                return err;
        }
        fprintf(out, "\n");

        fprintf(out, "  /* Execution Space */\n  ");
        err = jdf_variable_list_unparse(f->locals, out, "\n  ");
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

        {
            jdf_body_t* body = f->bodies;
            do {  /* There must be at least one */
                fprintf(out, "BODY\n");
                jdf_properties_unparse(body->properties, out);
                fprintf(out, "{\n");
                fprintf(out, "%s\n", body->external_code);
                fprintf(out, "}\n");
                fprintf(out, "END\n\n");
                body = body->next;
            } while (NULL != body);
        }
    }

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
