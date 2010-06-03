/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "expr.h"
#include "symbol.h"

#ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
#define likely(x)       __builtin_expect((x),1)
#else
#define likely(x)       x
#endif

#define EXPR_EVAL_ERROR_SIZE   512
static char expr_eval_error[EXPR_EVAL_ERROR_SIZE];

static int expr_eval_unary(const struct dague_object *parent, unsigned char op, const expr_t *op1,
                           const assignment_t *assignments, unsigned int nbassignments,
                           int *v)
{
    int rc;
    int v1;
    
    assert( EXPR_IS_UNARY(op) );

    rc = expr_eval(parent, op1, assignments, nbassignments, &v1);
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

static int expr_eval_binary(const struct dague_object *parent, 
                            unsigned char op, const expr_t *op1, const expr_t *op2,
                            const assignment_t *assignments, unsigned int nbassignments,
                            int *v)
{
    int rc;
    int v1, v2;

    assert( EXPR_IS_BINARY(op) );

    rc = expr_eval(parent, op1, assignments, nbassignments, &v1);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = expr_eval(parent, op2, assignments, nbassignments, &v2);
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
    case EXPR_OP_BINARY_NOT_EQUAL:
        *v = (v1 != v2);
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
    case EXPR_OP_BINARY_DIV:
        if( 0 == v2 ) {
            *v = INT_MAX;
        } else {
            *v = v1 / v2;
        }
        break;
    case EXPR_OP_BINARY_AND:
        if( (0 != v1) && (0 != v2) ) {
            *v = 1;
        } else {
            *v = 0;
        }
        break;
    case EXPR_OP_BINARY_OR:
        if( (0 != v1) || (0 != v2) ) {
            *v = 1;
        } else {
            *v = 0;
        }
        break;
    case EXPR_OP_BINARY_XOR:
        if( ((0 != v1) || (0 != v2)) && !((0 != v1) && (0 != v2)) ) {
            *v = 1;
        } else {
            *v = 0;
        }
        break;
    case EXPR_OP_BINARY_LESS:
        *v = (v1 < v2);
        break;
    case EXPR_OP_BINARY_LESS_OR_EQUAL:
        *v = (v1 <= v2);
        break;
    case EXPR_OP_BINARY_MORE:
        *v = (v1 > v2);
        break;
    case EXPR_OP_BINARY_MORE_OR_EQUAL:
        *v = (v1 >= v2);
        break;
    case EXPR_OP_BINARY_SHL:
        *v = (v1 << v2);
        break;
    case EXPR_OP_BINARY_RANGE:
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Cannot evaluate range");
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    }

    return EXPR_SUCCESS;
}

static int expr_eval_symbol(const struct dague_object *parent, const symbol_t *sym, const assignment_t *assignments, unsigned int nbassignments, int *res)
{
    assignment_t* assignment;

    /* look at the global symbols first */
    const symbol_t *gsym = dague_search_global_symbol( sym->name );
    if( gsym != NULL ){
        int int_res;
        if( EXPR_SUCCESS == expr_eval(parent, (expr_t *)gsym->min, NULL, 0, &int_res) ){
            *res = int_res;
            return EXPR_SUCCESS;
        }
    }

    if( EXPR_SUCCESS == dague_find_assignment(sym->name, assignments, nbassignments, &assignment) ) {
        *res = assignment->value;
        return EXPR_SUCCESS;
    }

    snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Symbol not found in assignment: %s", sym->name);
    return EXPR_FAILURE_SYMBOL_NOT_FOUND;
}

int expr_eval(const struct dague_object *parent,
              const expr_t *expr,
              const assignment_t *assignments, unsigned int nbassignments,
              int *res)
{
    /* If the expression was already evaluated to a constant, just return the
     * precalculated value.
     */
    if( EXPR_FLAG_CONSTANT & expr->flags ) {
        *res = expr->value;
        return EXPR_SUCCESS;
    }
    assert( EXPR_OP_CONST_INT != expr->op );

    if( likely( EXPR_IS_INLINE(expr->op) ) ) {
        *res = expr->inline_func(parent, assignments);
        return EXPR_SUCCESS;
    }

    if( EXPR_OP_SYMB == expr->op ) {
        int ret_val = expr_eval_symbol(parent, expr->variable, assignments, nbassignments, res);
        return ret_val;
    }
    if ( EXPR_IS_UNARY(expr->op) ) {
        return expr_eval_unary(parent, expr->op, expr->uop1, assignments, nbassignments, res);
    }
    if ( EXPR_IS_BINARY(expr->op) ) {
        return expr_eval_binary(parent, expr->op, expr->bop1, expr->bop2, assignments, nbassignments, res);
    }
    if ( EXPR_IS_TERTIAR(expr->op) ) {
        int ret_val = expr_eval(parent, expr->tcond, assignments, nbassignments, res);
        if( EXPR_SUCCESS != ret_val )
            return ret_val;

        if( 0 != res ) {
            return expr_eval(parent, expr->top1, assignments, nbassignments, res);
        }
        return expr_eval(parent, expr->top2, assignments, nbassignments, res);
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
        return callback(expr->variable, data);
    }
    if( EXPR_OP_CONST_INT == expr->op ) {
        return EXPR_SUCCESS;
    }
    if( EXPR_IS_UNARY(expr->op) ) {
        return expr_parse_symbols( expr->uop1, callback, data );
    }
    if( EXPR_IS_BINARY(expr->op) ) {
        rc = expr_parse_symbols( expr->bop1, callback, data );
        /* if we got an error don't check for the second expression */
        if( EXPR_SUCCESS == rc ) {
            return expr_parse_symbols( expr->bop2, callback, data );
        }
        return rc;
    }
    assert( EXPR_IS_TERTIAR(expr->op) );
    rc = expr_parse_symbols( expr->tcond, callback, data );
    if( EXPR_SUCCESS == rc ) {
        rc = expr_parse_symbols( expr->top1, callback, data );
        /* if we got an error don't check for the second expression */
        if( EXPR_SUCCESS == rc ) {
            return expr_parse_symbols( expr->top2, callback, data );
        }
    }
    return rc;
}

int expr_depend_on_symbol( const expr_t* expr,
                           const symbol_t* symbol )
{
    int rc;

    if( EXPR_OP_SYMB == expr->op ) {
        if( !strcmp(expr->variable->name, symbol->name) ) {
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

    if( EXPR_IS_BINARY(expr->op) ) {
        rc = expr_depend_on_symbol( expr->bop1, symbol );
        if( EXPR_FAILURE_SYMBOL_NOT_FOUND == rc ) { /* not yet check for the second expression */
            return expr_depend_on_symbol( expr->bop2, symbol );
        }
        return rc;
    }

    assert( EXPR_IS_TERTIAR(expr->op) );

    rc = expr_depend_on_symbol( expr->tcond, symbol );
    if( EXPR_FAILURE_SYMBOL_NOT_FOUND == rc ) {
        rc = expr_depend_on_symbol( expr->top1, symbol );
        if( EXPR_FAILURE_SYMBOL_NOT_FOUND == rc ) {
            return expr_depend_on_symbol( expr->top2, symbol );
        }
    }

    return rc;
}

#define EXPR_ABSOLUTE_RANGE_MIN 1
#define EXPR_ABSOLUTE_RANGE_MAX 2

static int __expr_absolute_range_recursive( const struct dague_object *dague_object,
                                            const expr_t* expr, int direction,
                                            int* pmin, int* pmax )
{
    int rc, *storage, lmin, lmax, rmin, rmax;

    if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
        storage = pmin;
    } else {
        assert( EXPR_ABSOLUTE_RANGE_MAX == direction );
        storage = pmax;
    }

    /* If the expression was already evaluated to a constant, just return the
     * precalculated value.
     */
    if( EXPR_FLAG_CONSTANT & expr->flags ) {
        *storage = expr->value;
        return EXPR_SUCCESS;
    }
    assert( EXPR_OP_CONST_INT != expr->op );

    if( EXPR_OP_SYMB == expr->op ) {
        const symbol_t* symbol = expr->variable;
        const symbol_t* gsym = dague_search_global_symbol( symbol->name );
        if( gsym != NULL ) {
            if( EXPR_SUCCESS == expr_eval(dague_object, (expr_t *)gsym->min, NULL, 0, storage) ) {
                return EXPR_SUCCESS;
            }
            return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
        }
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            rc = __expr_absolute_range_recursive( dague_object, symbol->min, direction, pmin, pmax );
        } else {
            rc = __expr_absolute_range_recursive( dague_object, symbol->max, direction, pmin, pmax );
        }
        return rc;
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
    case EXPR_OP_BINARY_NOT_EQUAL:
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    case EXPR_OP_BINARY_PLUS:
        rc = __expr_absolute_range_recursive( dague_object, expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( dague_object, expr->bop2, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin + rmin;
        } else {
            *pmax = lmax + rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_MINUS:
        rc = __expr_absolute_range_recursive( dague_object, expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( dague_object, expr->bop2, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin - rmin;
        } else {
            *pmax = lmax - rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_TIMES:
        rc = __expr_absolute_range_recursive( dague_object, expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( dague_object, expr->bop2, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin * rmin;
        } else {
            *pmax = lmax * rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_SHL:
        rc = __expr_absolute_range_recursive( dague_object, expr->bop1, direction, &lmin, &lmax );
        rc = __expr_absolute_range_recursive( dague_object, expr->bop2, direction, &rmin, &rmax );
        if( EXPR_ABSOLUTE_RANGE_MIN == direction ) {
            *pmin = lmin << rmin;
        } else {
            *pmax = lmax << rmax;
        }
        return EXPR_SUCCESS;
    case EXPR_OP_BINARY_LESS:
    case EXPR_OP_BINARY_LESS_OR_EQUAL:
    case EXPR_OP_BINARY_MORE:
    case EXPR_OP_BINARY_MORE_OR_EQUAL:
    case EXPR_OP_BINARY_DIV:
    case EXPR_OP_BINARY_AND:
    case EXPR_OP_BINARY_OR:
    case EXPR_OP_BINARY_XOR:
    case EXPR_OP_BINARY_RANGE:
        /* should we continue down the expressions of the range ? */
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
    }
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int expr_absolute_range(const struct dague_object *dague_object,
                        const expr_t* expr,
                        int* pmin, int* pmax)
{
    int rc, unused;

    rc = __expr_absolute_range_recursive( dague_object, expr, EXPR_ABSOLUTE_RANGE_MIN, pmin, &unused );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = __expr_absolute_range_recursive( dague_object, expr, EXPR_ABSOLUTE_RANGE_MAX, &unused, pmax );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    return EXPR_SUCCESS;
}

int expr_range_to_min_max(const struct dague_object *dague_object,
                          const expr_t *expr,
                          const assignment_t *assignments, unsigned int nbassignments,
                          int *min, int *max)
{
    int rc;

    assert( expr->op == EXPR_OP_BINARY_RANGE );

    rc = expr_eval(dague_object, expr->bop1, assignments, nbassignments, min);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = expr_eval(dague_object, expr->bop2, assignments, nbassignments, max);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    return EXPR_SUCCESS;
}

expr_t *expr_new_var(const symbol_t *symb)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_SYMB;
    r->variable = (symbol_t*)symb;
    if( dague_symbol_is_global(symb) &&
        ((NULL != symb->min) && (symb->min->flags & EXPR_FLAG_CONSTANT)) ) {
        r->flags = EXPR_FLAG_CONSTANT;
        r->value = symb->min->value;
    } else {
        r->flags = 0;  /* no flags */
    }
    return r;
}

expr_t *expr_new_int(int v)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op    = EXPR_OP_CONST_INT;
    r->flags = EXPR_FLAG_CONSTANT;
    r->value = v;
    return r;
}

/* This function should negate an expression */
expr_t *expr_new_unary(char op, expr_t *e)
{
    expr_t *n = (expr_t*)calloc(1, sizeof(expr_t));
    n->uop1 = e;
    n->flags = 0;  /* unknown yet */
    if( op == '!' ) {
        n->op   = EXPR_OP_UNARY_NOT;
        if( e->flags & EXPR_FLAG_CONSTANT ) {
            n->flags = EXPR_FLAG_CONSTANT;
            n->value = !(n->value);
        }
    }
    return n;
}

expr_t *expr_new_binary(const expr_t *op1, char op, const expr_t *op2)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    int is_constant;

    r->bop1  = (expr_t*)op1;
    r->bop2  = (expr_t*)op2;
    r->flags = 0;  /* unknown yet */
    is_constant = op1->flags & op2->flags & EXPR_FLAG_CONSTANT;

    switch( op ) {
    case '+':
        r->op = EXPR_OP_BINARY_PLUS;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = op1->value + op2->value;
        }
        return r;
    case '-':
        r->op = EXPR_OP_BINARY_MINUS;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = op1->value - op2->value;
        }
        return r;
    case '*':
        r->op = EXPR_OP_BINARY_TIMES;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = op1->value * op2->value;
        }
        return r;
    case '%':
        r->op = EXPR_OP_BINARY_MOD;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = op1->value % op2->value;
        }
        return r;
    case '=':
        r->op = EXPR_OP_BINARY_EQUAL;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = op1->value && op2->value;
        }
        return r;
    case '!':
        r->op = EXPR_OP_BINARY_NOT_EQUAL;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = !(op1->value && op2->value);
        }
        return r;
    case '.':
        r->op = EXPR_OP_BINARY_RANGE;
        return r;
    case '/':
        r->op = EXPR_OP_BINARY_DIV;
        return r;
    case '|':
        r->op = EXPR_OP_BINARY_OR;
        return r;
    case '&':
        r->op = EXPR_OP_BINARY_AND;
        return r;
    case '^':
        r->op = EXPR_OP_BINARY_XOR;
        return r;
    case '<':
        r->op = EXPR_OP_BINARY_LESS;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = (op1->value < op2->value);
        }
        return r;
    case '{':
        r->op = EXPR_OP_BINARY_LESS_OR_EQUAL;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = (op1->value <= op2->value);
        }
        return r;
    case '>':
        r->op = EXPR_OP_BINARY_MORE;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = (op1->value > op2->value);
        }
        return r;
    case '}':
        r->op = EXPR_OP_BINARY_MORE_OR_EQUAL;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = (op1->value >= op2->value);
        }
        return r;
    case 'L':
        r->op = EXPR_OP_BINARY_SHL;
        if( is_constant ) {
            r->flags = EXPR_FLAG_CONSTANT;
            r->value = (op1->value << op2->value);
        }
        return r;
    }

    free(r);
    fprintf(stderr, "[%s:%d] Unknown operand %c. Return NULL expression\n", __FILE__, __LINE__, op );
    return NULL;
}

expr_t *expr_new_tertiar(const expr_t *cond, const expr_t *op1, const expr_t *op2)
{
    expr_t *r;

    if( cond->flags & EXPR_FLAG_CONSTANT ) {
        /* the condition is constant therefore we can safely translate the tertiar
         * expression into a single expression depending on the cond value.
         */
        if( cond->value ) {
            return (expr_t*)op1;
        }
        return (expr_t*)op2;
    }
    r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_CONDITIONAL;
    r->flags = 0;  /* unknown yet */
    r->tcond = (expr_t*)cond;
    r->top1 = (expr_t*)op1;
    r->top2 = (expr_t*)op2;
    return r;
}

char *expr_error(void)
{
    return expr_eval_error;
}

static void expr_dump_unary(FILE *out, const struct dague_object *dague_object, unsigned char op, const expr_t *op1)
{
    switch(op) {
    case EXPR_OP_UNARY_NOT:
        fprintf(out, "!(");
        break;
    }

    if( NULL == op1 ) {
        fprintf(out, "NULL");
    } else {
        expr_dump(out, dague_object, op1);
        if( op == EXPR_OP_UNARY_NOT ){
            fprintf(out, ")");
        }
    }
}

static void expr_dump_binary(FILE *out, const struct dague_object *dague_object, unsigned char op, const expr_t *op1, const expr_t *op2)
{
    if( EXPR_OP_BINARY_RANGE == op ) {
        fprintf(out,  " [" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " .. " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  "] " );
        return;
    }

    if( EXPR_OP_BINARY_EQUAL == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " == " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_NOT_EQUAL == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " != " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_LESS == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " < " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_LESS_OR_EQUAL == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " <= " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_MORE == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " > " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_MORE_OR_EQUAL == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " >= " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    if( EXPR_OP_BINARY_SHL == op ) {
        fprintf(out,  " (" );
        expr_dump(out, dague_object, op1);
        fprintf(out,  " << " );
        expr_dump(out, dague_object, op2);
        fprintf(out,  ") " );
        return;
    }

    expr_dump(out, dague_object, op1);

    switch( op ) {
    case EXPR_OP_BINARY_PLUS:
        fprintf(out, " + ");
        break;
    case EXPR_OP_BINARY_MINUS:
        fprintf(out, " - ");
        break;
    case EXPR_OP_BINARY_TIMES:
        fprintf(out, " * ");
        break;
    case EXPR_OP_BINARY_MOD:
        fprintf(out, " %% ");
        break;
    case EXPR_OP_BINARY_DIV:
        fprintf(out, " / ");
        break;
    case EXPR_OP_BINARY_AND:
        fprintf(out, " & ");
        break;
    case EXPR_OP_BINARY_OR:
        fprintf(out, " | ");
        break;
    case EXPR_OP_BINARY_XOR:
        fprintf(out, " ^ ");
        break;
    }

    expr_dump(out, dague_object, op2);
}

void expr_dump(FILE *out, const struct dague_object *dague_object, const expr_t *e)
{
    if( NULL == e ) {
        fprintf(out, "NULL");
        return;
    }
    if( EXPR_FLAG_CONSTANT & e->flags ) {
        if( EXPR_OP_CONST_INT == e->op ) {
            fprintf(out,  "%d", e->value );
            return;
        }
        fprintf(out,  "{%d:", e->value );
    }
    if( EXPR_OP_SYMB == e->op ) {
        if( dague_symbol_is_global(e->variable) ) {
            fprintf(out, "%s", e->variable->name);
        } else {
            int res;
            if( EXPR_SUCCESS == expr_eval_symbol(dague_object, e->variable, NULL, 0, &res)){
                fprintf(out, "%d", res);
            }else{
                fprintf(out, "%s", e->variable->name);
            }
        }
    } else if( EXPR_OP_CONST_INT == e->op ) {
        fprintf(out, "%d", e->value);
    } else if( EXPR_IS_UNARY(e->op) ) {
        expr_dump_unary(out, dague_object, e->op, e->uop1);
    } else if( EXPR_IS_BINARY(e->op) ) {
        expr_dump_binary(out, dague_object, e->op, e->bop1, e->bop2);
    } else if( EXPR_IS_TERTIAR(e->op) ) {
        fprintf( out, "(");
        expr_dump(out, dague_object, e->tcond);
        fprintf( out, " ? " );
        expr_dump(out, dague_object, e->top1);
        fprintf( out, " : ");
        expr_dump(out, dague_object, e->top2);
        fprintf( out, ")");
    } else {
        fprintf(stderr, "[%s:%d] Unkown operand %d in expression\n", __FILE__, __LINE__, e->op);
    }
    if( EXPR_FLAG_CONSTANT & e->flags ) {
        fprintf(out,  "}" );
    }
}
