/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "expr.h"
#include "symbol.h"

#define EXPR_EVAL_ERROR_SIZE   512
static char expr_eval_error[EXPR_EVAL_ERROR_SIZE];

/* This function should negate an expression  */
expr_t *negate_expr(expr_t *e){
    expr_t *n = (expr_t*)calloc(1, sizeof(expr_t));
    n->uop1 = e;
    n->op   = EXPR_OP_UNARY_NOT;
    return n;
}


static int expr_eval_unary(unsigned char op, const expr_t *op1,
                           const assignment_t *assignments, unsigned int nbassignments,
                           int *v)
{
    int rc;
    int v1;
    
    assert( EXPR_IS_UNARY(op) );

    rc = expr_eval(op1, assignments, nbassignments, &v1);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    switch(op) {
    case EXPR_OP_UNARY_NOT:
        *v = !v1;
        break;
    }

    return EXPR_SUCCESS;
}

static int expr_eval_binary(unsigned char op, const expr_t *op1, const expr_t *op2,
                            const assignment_t *assignments, unsigned int nbassignments,
                            int *v)
{
    int rc;
    int v1, v2;

    assert( EXPR_IS_BINARY(op) );

    rc = expr_eval(op1, assignments, nbassignments, &v1);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = expr_eval(op2, assignments, nbassignments, &v2);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    switch(op) {
    case EXPR_OP_BINARY_MOD:
        *v = (v1 % v2);
        break;
    case EXPR_OP_BINARY_EQUAL:
        *v = (v1 == v2);
        break;
    case EXPR_OP_BINARY_PLUS:
        *v = (v1 + v2);
        break;
    case EXPR_OP_BINARY_MINUS:
        *v = (v1 - v2);
        break;
    case EXPR_OP_BINARY_TIMES:
        *v = (v1 * v2);
        break;
    case EXPR_OP_BINARY_RANGE:
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Cannot evaluate range");
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    }

    return EXPR_SUCCESS;
}

static int expr_eval_symbol(const symbol_t *sym, const assignment_t *assignments, unsigned int nbassignments, int *res)
{
    assignment_t* assignment;

    /* look at the global symbols first */
    const symbol_t *gsym = dplasma_search_global_symbol( sym->name );
    if( gsym != NULL ){
        int int_res;
        if( EXPR_SUCCESS == expr_eval((expr_t *)gsym->min, NULL, 0, &int_res) ){
            *res = int_res;
            return EXPR_SUCCESS;
        }
    }

    if( EXPR_SUCCESS == dplasma_find_assignment(sym->name, assignments, nbassignments, &assignment) ) {
        *res = assignment->value;
        return EXPR_SUCCESS;
    }

    snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Symbol not found in assignment: %s", sym->name);
    return EXPR_FAILURE_SYMBOL_NOT_FOUND;
}

int expr_eval(const expr_t *expr,
              const assignment_t *assignments, unsigned int nbassignments,
              int *res)
{
    if( EXPR_OP_SYMB == expr->op ) {
        int int_res;
        int ret_val = expr_eval_symbol(expr->var, assignments, nbassignments, &int_res);
        *res = int_res;
        return ret_val;
    }
    if ( EXPR_OP_CONST_INT == expr->op ) {
        *res = expr->const_int;
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Success");
        return EXPR_SUCCESS;
    }
    if ( EXPR_IS_UNARY(expr->op) ) {
        return expr_eval_unary(expr->op, expr->uop1, assignments, nbassignments, res);
    }
    if ( EXPR_IS_BINARY(expr->op) ) {
        return expr_eval_binary(expr->op, expr->bop1, expr->bop2, assignments, nbassignments, res);
    }
    snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Unkown operand %d in expression", expr->op);
    return EXPR_FAILURE_UNKNOWN_OP;
}

int expr_parse_symbols( const expr_t* expr,
                        expr_symbol_check_callback_t* callback,
                        void* data )
{
    int rc;

    if( EXPR_OP_SYMB == expr->op ) {
        return callback(expr->var, data);
    }
    if( EXPR_OP_CONST_INT == expr->op ) {
        return EXPR_SUCCESS;
    }
    if( EXPR_IS_UNARY(expr->op) ) {
        return expr_parse_symbols( expr->uop1, callback, data );
    }
    rc = expr_parse_symbols( expr->bop1, callback, data );
    /* if we got an error don't check for the second expression */
    if( EXPR_SUCCESS == rc ) {
        return expr_parse_symbols( expr->bop2, callback, data );
    }
    return rc;
}

int expr_depend_on_symbol( const expr_t* expr,
                           const symbol_t* symbol )
{
    int rc;

    if( EXPR_OP_SYMB == expr->op ) {
        if( expr->var == symbol ) {
            return EXPR_SUCCESS;
        }
        return EXPR_FAILURE_SYMBOL_NOT_FOUND;
    }
    if( EXPR_OP_CONST_INT == expr->op ) {
        return EXPR_FAILURE_SYMBOL_NOT_FOUND;
    }
    if( EXPR_IS_UNARY(expr->op) ) {
        return expr_depend_on_symbol( expr->uop1, symbol );
    }
    rc = expr_depend_on_symbol( expr->bop1, symbol );
    if( EXPR_FAILURE_SYMBOL_NOT_FOUND == rc ) { /* not yet check for the second expression */
        return expr_depend_on_symbol( expr->bop2, symbol );
    }
    return rc;
}

#define EXPR_ABSOLUTE_RANGE_MIN 1
#define EXPR_ABSOLUTE_RANGE_MAX 2

int __expr_absolute_range_recursive( const expr_t* expr, int direction,
                                     int* pmin, int* pmax )
{
    int rc, *storage, lmin, lmax, rmin, rmax;

    if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
        storage = pmin;
    } else {
        assert( EXPR_ABSOLUTE_RANGE_MAX == direction );
        storage = pmax;
    }

    if( EXPR_OP_SYMB == expr->op ) {
        const symbol_t* symbol = expr->var;
        const symbol_t* gsym = dplasma_search_global_symbol( symbol->name );
        if( gsym != NULL ) {
            if( EXPR_SUCCESS == expr_eval((expr_t *)gsym->min, NULL, 0, storage) ) {
                return EXPR_SUCCESS;
            }
            return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
        }
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            rc = __expr_absolute_range_recursive( symbol->min, direction, pmin, pmax );
        } else {
            rc = __expr_absolute_range_recursive( symbol->max, direction, pmin, pmax );
        }
        return rc;
    }

    if( EXPR_OP_CONST_INT == expr->op ) {
        *storage = expr->const_int;
        return EXPR_SUCCESS;
    }

    if( EXPR_IS_UNARY(expr->op) ) {
        /* there is no range for boolean values */
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    }

    assert( EXPR_IS_BINARY(expr->op) );
    switch(expr->op) {
    case EXPR_OP_BINARY_MOD:
        printf( "No idea how to compute a min or max of a %%\n" );
        return EXPR_FAILURE_UNKNOWN;
    case EXPR_OP_BINARY_EQUAL:
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    case EXPR_OP_BINARY_PLUS:
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin + rmin;
        } else {
            *pmax = lmax + rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_MINUS:
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin + rmin;
        } else {
            *pmax = lmax + rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_TIMES:
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( expr->bop1, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin * rmin;
        } else {
            *pmax = lmax * rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_RANGE:
        /* should we continue down the expressions of the range ? */
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    }
    return rc;
}

int expr_absolute_range(const expr_t* expr,
                        int* pmin, int* pmax)
{
    int rc, unused;

    assert( expr->op == EXPR_OP_BINARY_RANGE );

    rc = __expr_absolute_range_recursive( expr->bop1, EXPR_ABSOLUTE_RANGE_MIN, pmin, &unused );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = __expr_absolute_range_recursive( expr->bop2, EXPR_ABSOLUTE_RANGE_MIN, pmin, &unused );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    return EXPR_SUCCESS;
}

int expr_range_to_min_max(const expr_t *expr,
                          const assignment_t *assignments, unsigned int nbassignments,
                          int *min, int *max)
{
    int rc;

    assert( expr->op == EXPR_OP_BINARY_RANGE );

    rc = expr_eval(expr->bop1, assignments, nbassignments, min);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = expr_eval(expr->bop2, assignments, nbassignments, max);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    return EXPR_SUCCESS;
}

expr_t *expr_new_var(const symbol_t *symb)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_SYMB;
    r->var = (symbol_t*)symb;
    return r;
}

expr_t *expr_new_int(int v)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_CONST_INT;
    r->const_int = v;
    return r;
}

expr_t *expr_new_binary(const expr_t *op1, char op, const expr_t *op2)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));

    r->bop1 = (expr_t*)op1;
    r->bop2 = (expr_t*)op2;
    
    switch( op ) {
    case '+':
        r->op = EXPR_OP_BINARY_PLUS;
        return r;
    case '-':
        r->op = EXPR_OP_BINARY_MINUS;
        return r;
    case '*':
        r->op = EXPR_OP_BINARY_TIMES;
        return r;
    case '%':
        r->op = EXPR_OP_BINARY_MOD;
        return r;
    case '=':
        r->op = EXPR_OP_BINARY_EQUAL;
        return r;
    case '.':
        r->op = EXPR_OP_BINARY_RANGE;
        return r;
    default:
        free(r);
        return NULL;
    }

    return r;
}

char *expr_error(void)
{
    return expr_eval_error;
}

static void expr_dump_unary(unsigned char op, const expr_t *op1)
{
    switch(op) {
    case EXPR_OP_UNARY_NOT:
        printf("!(");
        break;
    }

    if( NULL == op1 ) {
        printf("NULL");
    } else {
        expr_dump(op1);
        if( op == EXPR_OP_UNARY_NOT ){
            printf(")");
        }
    }
}

static void expr_dump_binary(unsigned char op, const expr_t *op1, const expr_t *op2)
{
    expr_dump(op1);

    switch( op ) {
    case EXPR_OP_BINARY_PLUS:
        printf(" + ");
        break;
    case EXPR_OP_BINARY_MINUS:
        printf(" - ");
        break;
    case EXPR_OP_BINARY_TIMES:
        printf(" * ");
        break;
    case EXPR_OP_BINARY_MOD:
        printf(" %% ");
        break;
    case EXPR_OP_BINARY_EQUAL:
        printf(" == ");
        break;
    case EXPR_OP_BINARY_RANGE:
        printf(" .. ");
        break;
    }

    expr_dump(op2);
}

void expr_dump(const expr_t *e)
{
    if(NULL == e) {
        printf("NULL");
    }

    if( EXPR_OP_SYMB == e->op ) {
        int res;
        if( EXPR_SUCCESS == expr_eval_symbol(e->var, NULL, 0, &res)){
            printf("%d", res);
        }else{
            printf("%s", e->var->name);
        }
    } else if ( EXPR_OP_CONST_INT == e->op ) {
        printf("%d", e->const_int);
    } else if ( EXPR_IS_UNARY(e->op) ) {
        expr_dump_unary(e->op, e->uop1);
    } else if ( EXPR_IS_BINARY(e->op) ) {
        expr_dump_binary(e->op, e->bop1, e->bop2);
    } else {
        fprintf(stderr, "Unkown operand %d in expression", e->op);
    }    
}
