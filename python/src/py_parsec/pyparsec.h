/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * Small C helpers for the Cython bindings.
 *
 * Cython cannot build a C variadic call whose length is only known at run
 * time, and the DTD insertion and unpacking functions are variadic. Both stop
 * reading their arguments at a terminator (PARSEC_DTD_ARG_END for insertion, a
 * NULL output pointer for unpacking), so the helpers below always pass
 * PARSEC_DTD_MAX_PARAMS argument slots, put the terminator right after the
 * last real argument, and let the callee ignore the unused tail. A single call
 * site then covers every parameter count the DTD interface accepts.
 */

#ifndef PYPARSEC_H_HAS_BEEN_INCLUDED
#define PYPARSEC_H_HAS_BEEN_INCLUDED

#include <limits.h>
#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/remote_dep.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/interfaces/dtd/insert_function.h"

#if PARSEC_DTD_MAX_PARAMS != 64
#error "PYPARSEC_SLOTS64 below must expand to PARSEC_DTD_MAX_PARAMS slots"
#endif

#define PYPARSEC_SLOT4(S, i)  S(i), S(i+1), S(i+2), S(i+3)
#define PYPARSEC_SLOT16(S, i) PYPARSEC_SLOT4(S, i), PYPARSEC_SLOT4(S, i+4), \
                              PYPARSEC_SLOT4(S, i+8), PYPARSEC_SLOT4(S, i+12)
#define PYPARSEC_SLOTS64(S)   PYPARSEC_SLOT16(S, 0), PYPARSEC_SLOT16(S, 16), \
                              PYPARSEC_SLOT16(S, 32), PYPARSEC_SLOT16(S, 48)

#define PYPARSEC_INSERT_PAIR(i) flags[i], ptrs[i]
#define PYPARSEC_UNPACK_OUT(i)  outs[i]

/* Insert one task of an explicit DTD task class. Each argument is a
 * (flags, pointer) pair, exactly as for parsec_dtd_insert_task_with_task_class(). */
static inline void
pyparsec_dtd_insert(parsec_taskpool_t *tp, parsec_task_class_t *tc,
                    int priority, int device_type,
                    int nb_args, const int *args_flags, void *const *args_ptrs)
{
    int flags[PARSEC_DTD_MAX_PARAMS];
    void *ptrs[PARSEC_DTD_MAX_PARAMS];

    for( int i = 0; i < PARSEC_DTD_MAX_PARAMS; i++ ) {
        flags[i] = (i < nb_args) ? args_flags[i] : PARSEC_DTD_ARG_END;
        ptrs[i]  = (i < nb_args) ? args_ptrs[i]  : NULL;
    }
    parsec_dtd_insert_task_with_task_class(tp, tc, priority, device_type,
                                           PYPARSEC_SLOTS64(PYPARSEC_INSERT_PAIR),
                                           PARSEC_DTD_ARG_END);
}

/* Unpack the first nb_args parameters of a DTD task into args_out. */
static inline void
pyparsec_dtd_unpack(parsec_task_t *task, int nb_args, void *const *args_out)
{
    void *outs[PARSEC_DTD_MAX_PARAMS];

    for( int i = 0; i < PARSEC_DTD_MAX_PARAMS; i++ ) {
        outs[i] = (i < nb_args) ? args_out[i] : NULL;
    }
    parsec_dtd_unpack_args(task, PYPARSEC_SLOTS64(PYPARSEC_UNPACK_OUT));
}

/* Create a DTD task class from parallel arrays of parameter sizes and
 * operation flags (the pairs parsec_dtd_create_task_class() reads). */
static inline parsec_task_class_t *
pyparsec_dtd_create_task_class(const char *name, int nb_params,
                               const int *sizes, const int *ops)
{
    parsec_dtd_param_t params[PARSEC_DTD_MAX_PARAMS];

    if( nb_params < 0 || nb_params > PARSEC_DTD_MAX_PARAMS ) return NULL;
    for( int i = 0; i < nb_params; i++ ) {
        params[i].op = (parsec_dtd_op_t)ops[i];
        params[i].size = sizes[i];
        params[i].profile_info = NULL;
    }
    /* The DTD task class starts with its parsec_task_class_t. */
    return (parsec_task_class_t *)parsec_dtd_create_task_classv(name, nb_params, params);
}

static inline const parsec_task_class_t *
pyparsec_task_class_of(const parsec_task_t *task)
{
    return task->task_class;
}

/* Largest number of data flows a DTD task class may declare (the limit
 * parsec_dtd_create_task_classv() enforces with parsec_fatal()). */
static inline int
pyparsec_dtd_max_flows(void)
{
    int bits = 0;
    while( (bits < (int)(sizeof(uint32_t) * CHAR_BIT)) &&
           (((uint32_t)1 << bits) < PARSEC_ACTION_RELEASE_LOCAL_DEPS) ) {
        bits++;
    }
    return (MAX_PARAM_COUNT < bits) ? MAX_PARAM_COUNT : bits;
}

/* True between parsec_context_start() and the end of parsec_context_wait(). */
static inline int
pyparsec_context_is_active(const parsec_context_t *context)
{
    return 0 != (context->flags & PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE);
}

/* The data collection accessors are variadic function pointers. */
static inline parsec_data_key_t
pyparsec_data_key(parsec_data_collection_t *dc, int m, int n)
{
    return dc->data_key(dc, m, n);
}

static inline int
pyparsec_rank_of(parsec_data_collection_t *dc, int m, int n)
{
    return (int)dc->rank_of(dc, m, n);
}

/* Host copy of a local tile, NULL if the tile is not stored on this rank. */
static inline void *
pyparsec_local_tile(parsec_data_collection_t *dc, int m, int n)
{
    parsec_data_t *data;
    parsec_data_copy_t *copy;

    if( dc->myrank != dc->rank_of(dc, m, n) ) return NULL;
    data = dc->data_of(dc, m, n);
    if( NULL == data ) return NULL;
    copy = parsec_data_get_copy(data, 0);
    return (NULL == copy) ? NULL : parsec_data_copy_get_ptr(copy);
}

static inline size_t
pyparsec_sizeof_matrix_type(int mtype)
{
    int size = parsec_datadist_getsizeoftype((parsec_matrix_type_t)mtype);
    return (size < 0) ? 0 : (size_t)size;
}

/* Arena datatype describing a full mb x nb tile with leading dimension ld,
 * registered with the context for DTD; returns the DTD arena id or -1. */
static inline int
pyparsec_dtd_tile_arena_new(parsec_context_t *ctx, int mtype, int mb, int nb, int ld)
{
    parsec_datatype_t dtt;
    parsec_arena_datatype_t *adt;
    int id = -1;

    if( PARSEC_SUCCESS != parsec_translate_matrix_type((parsec_matrix_type_t)mtype, &dtt) ) return -1;
    adt = parsec_matrix_adt_new_rect(dtt, mb, nb, ld);
    if( NULL == adt ) return -1;
    if( PARSEC_SUCCESS != parsec_dtd_attach_arena_datatype(ctx, adt, &id) ) {
        parsec_matrix_adt_free(&adt);
        return -1;
    }
    return id;
}

#endif  /* PYPARSEC_H_HAS_BEEN_INCLUDED */
