/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * PROTOTYPE -- single-declaration DTD task classes.
 *
 * A DTD task currently describes its parameter list three times: once in
 * parsec_dtd_create_task_class(), once at every parsec_dtd_insert_task*()
 * call site, and once in the body's parsec_dtd_unpack_args(). The three
 * lists must agree on order, count and size, and nothing checks that they
 * do -- a mismatch is a silent memcpy of the wrong number of bytes into a
 * caller-supplied void*.
 *
 * This header lets the parameter list be written once, as an X-macro, and
 * generates all three uses from it. Drift between them becomes impossible,
 * and every scalar is type-checked by the compiler at both ends.
 *
 * Usage:
 *
 *   #define ZGEMM_PARAMS(_)                                      \
 *       _(VALUE, int,                 transA, 0              )   \
 *       _(VALUE, dplasma_complex64_t, alpha,  0              )   \
 *       _(INPUT, dplasma_complex64_t, A,      0              )   \
 *       _(INOUT, dplasma_complex64_t, C,      PARSEC_AFFINITY)
 *
 *   PARSEC_DTD_DECLARE_TASK_CLASS(zgemm, ZGEMM_PARAMS);
 *
 * Each entry is (KIND, TYPE, NAME, CLASS_FLAGS):
 *   KIND        one of VALUE, REF, SCRATCH, INPUT, INOUT, OUTPUT
 *   TYPE        the C type of the scalar, or the element type of the tile
 *   NAME        the field name, used identically at insertion and in the body
 *   CLASS_FLAGS flags fixed for every task of the class, e.g. PARSEC_AFFINITY
 *               or PARSEC_DONT_TRACK; 0 for none
 *
 *
 * READING THE PARAMETERS BACK
 *
 * Three styles are generated from the same list, and a body picks whichever
 * suits it. They differ only in how much copying they do; all three see the
 * same values.
 *
 *   1. In place, no copy. The by-value parameters are read where the runtime
 *      already put them, and only the data flows are fetched:
 *
 *          const zgemm_values_t *v = zgemm_values(this_task);
 *          zgemm_flows_t         f = zgemm_flows(this_task);
 *          CORE_zgemm(v->transA, ..., f.A, ..., f.C, v->ldc);
 *
 *   2. One struct, copied. Gives a private, mutable copy of everything:
 *
 *          zgemm_body_t task;
 *          zgemm_unpack(this_task, &task);
 *          CORE_zgemm(task.transA, ..., task.A, ...);
 *
 *   3. Bare locals, copied. Reads like the parameter list itself:
 *
 *          PARSEC_DTD_UNPACK_LOCALS(zgemm, ZGEMM_PARAMS, this_task);
 *          CORE_zgemm(transA, transB, m, n, k, alpha, A, lda, ...);
 *
 * Style 1 costs a pointer addition plus one load per flow. Styles 2 and 3 add
 * one load and one store per parameter on top of that; the compiler removes
 * both for any parameter the body does not read.
 *
 *
 * HOW THE IN-PLACE READ IS POSSIBLE
 *
 * The runtime stores the by-value parameters of a task back to back, in
 * declaration order, each taking exactly as many bytes as the task class
 * declared for it. That block is reachable with parsec_dtd_task_values().
 *
 * So the layout is already a struct; it is simply one the compiler has never
 * been told about. This header tells it: NAME##_values_t is declared with the
 * same fields in the same order, and each field is padded up to a multiple of
 * sizeof(void *) by declaring that same padded size to the task class. Every
 * field therefore lands on a sizeof(void *) boundary, no field is misaligned,
 * and the compiler inserts no padding of its own -- which is checked, not
 * assumed, by a _Static_assert on sizeof(NAME##_values_t).
 *
 * This is the reason the parameter list has to be written once: the struct the
 * body reads and the sizes the runtime was given come out of the same list, so
 * they cannot disagree.
 *
 *
 * Known prototype limitations:
 *   - one arena/region index is applied to every flow of the class;
 *   - SCRATCH is fixed-size (sizeof(TYPE)). Runtime-sized scratch needs a
 *     count expression that parsec_dtd_create_task_class() cannot take;
 *   - PARSEC_PROFILE_INFO is not exposed. It infers a parameter's type from
 *     its declared size, which the padding above would mislead;
 *   - a body of a generated task class must use the generated accessors. A
 *     hand-written parsec_dtd_unpack_args() would copy the padded size into
 *     an unpadded variable.
 */

#ifndef PARSEC_DTD_TASK_CLASS_DECL_H_HAS_BEEN_INCLUDED
#define PARSEC_DTD_TASK_CLASS_DECL_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"
#include "parsec/data.h"
#include "parsec/interfaces/dtd/insert_function.h"

BEGIN_C_DECLS

/* Every by-value parameter occupies a whole number of pointer-sized slots, so
 * that each one starts aligned however the preceding ones are sized. */
#define PARSEC_DTD__SLOT_ALIGN     (sizeof(void *))
#define PARSEC_DTD__SLOT_SIZE(TYPE)                                             \
    ((((sizeof(TYPE)) + PARSEC_DTD__SLOT_ALIGN - 1) / PARSEC_DTD__SLOT_ALIGN)   \
     * PARSEC_DTD__SLOT_ALIGN)

/* A field of the value block, named so the body can read it, and widened to
 * the slot size so the struct matches what the runtime laid out. */
#define PARSEC_DTD__SLOT(TYPE, NAME)                                            \
    union { TYPE NAME; char NAME##__slot[PARSEC_DTD__SLOT_SIZE(TYPE)]; };

/*
 * Per-kind dispatch. Each generated construct pastes the kind onto a macro
 * name, so an unsupported kind is a compile error naming the kind rather than
 * a runtime surprise.
 *
 * REF is carried as a by-value pointer: the body wants the pointer, and a
 * pointer copied into the value block is readable in place, whereas one held
 * in the runtime's parameter descriptor is not.
 */

/* Field in the insertion argument struct. By-value fields are widened here
 * too, because the runtime copies the declared size out of this struct. */
#define PARSEC_DTD__ARG_FIELD_VALUE(TYPE, NAME)   PARSEC_DTD__SLOT(TYPE, NAME)
#define PARSEC_DTD__ARG_FIELD_REF(TYPE, NAME)     PARSEC_DTD__SLOT(TYPE *, NAME)
#define PARSEC_DTD__ARG_FIELD_SCRATCH(TYPE, NAME) /* runtime allocated, nothing to pass */
#define PARSEC_DTD__ARG_FIELD_INPUT(TYPE, NAME)   parsec_dtd_tile_t *NAME; int NAME##_flags;
#define PARSEC_DTD__ARG_FIELD_INOUT(TYPE, NAME)   parsec_dtd_tile_t *NAME; int NAME##_flags;
#define PARSEC_DTD__ARG_FIELD_OUTPUT(TYPE, NAME)  parsec_dtd_tile_t *NAME; int NAME##_flags;

/* Field in the value block. Flows are not in it: their address is not known
 * until the task is scheduled. */
#define PARSEC_DTD__VAL_FIELD_VALUE(TYPE, NAME)   PARSEC_DTD__SLOT(TYPE, NAME)
#define PARSEC_DTD__VAL_FIELD_REF(TYPE, NAME)     PARSEC_DTD__SLOT(TYPE *, NAME)
#define PARSEC_DTD__VAL_FIELD_SCRATCH(TYPE, NAME) PARSEC_DTD__SLOT(TYPE, NAME)
#define PARSEC_DTD__VAL_FIELD_INPUT(TYPE, NAME)
#define PARSEC_DTD__VAL_FIELD_INOUT(TYPE, NAME)
#define PARSEC_DTD__VAL_FIELD_OUTPUT(TYPE, NAME)

/* Bytes the kind takes in the value block, and hence the size declared to the
 * task class. The two are the same expression on purpose. */
#define PARSEC_DTD__VAL_BYTES_VALUE(TYPE)   PARSEC_DTD__SLOT_SIZE(TYPE)
#define PARSEC_DTD__VAL_BYTES_REF(TYPE)     PARSEC_DTD__SLOT_SIZE(TYPE *)
#define PARSEC_DTD__VAL_BYTES_SCRATCH(TYPE) PARSEC_DTD__SLOT_SIZE(TYPE)
#define PARSEC_DTD__VAL_BYTES_INPUT(TYPE)   0
#define PARSEC_DTD__VAL_BYTES_INOUT(TYPE)   0
#define PARSEC_DTD__VAL_BYTES_OUTPUT(TYPE)  0

/* Field in the flows struct, and the fetch that fills it. */
#define PARSEC_DTD__FLOW_FIELD_VALUE(TYPE, NAME)
#define PARSEC_DTD__FLOW_FIELD_REF(TYPE, NAME)
#define PARSEC_DTD__FLOW_FIELD_SCRATCH(TYPE, NAME)
#define PARSEC_DTD__FLOW_FIELD_INPUT(TYPE, NAME)   TYPE *NAME;
#define PARSEC_DTD__FLOW_FIELD_INOUT(TYPE, NAME)   TYPE *NAME;
#define PARSEC_DTD__FLOW_FIELD_OUTPUT(TYPE, NAME)  TYPE *NAME;

#define PARSEC_DTD__FLOW_FETCH_VALUE(TYPE, NAME, WHICH)
#define PARSEC_DTD__FLOW_FETCH_REF(TYPE, NAME, WHICH)
#define PARSEC_DTD__FLOW_FETCH_SCRATCH(TYPE, NAME, WHICH)
#define PARSEC_DTD__FLOW_FETCH_INPUT(TYPE, NAME, WHICH)                         \
    flows__.NAME = (TYPE *)PARSEC_DATA_COPY_GET_PTR(this_task->data[flow__++].WHICH);
#define PARSEC_DTD__FLOW_FETCH_INOUT(TYPE, NAME, WHICH)                         \
    PARSEC_DTD__FLOW_FETCH_INPUT(TYPE, NAME, WHICH)
#define PARSEC_DTD__FLOW_FETCH_OUTPUT(TYPE, NAME, WHICH)                        \
    PARSEC_DTD__FLOW_FETCH_INPUT(TYPE, NAME, WHICH)

/* Field in the body struct. A flow is handed to the body as a pointer to the
 * copy that lives on the device the task was scheduled on; a scratch as a
 * pointer to its slot in the task's own value block. */
#define PARSEC_DTD__BODY_FIELD_VALUE(TYPE, NAME)   TYPE NAME;
#define PARSEC_DTD__BODY_FIELD_REF(TYPE, NAME)     TYPE *NAME;
#define PARSEC_DTD__BODY_FIELD_SCRATCH(TYPE, NAME) TYPE *NAME;
#define PARSEC_DTD__BODY_FIELD_INPUT(TYPE, NAME)   TYPE *NAME;
#define PARSEC_DTD__BODY_FIELD_INOUT(TYPE, NAME)   TYPE *NAME;
#define PARSEC_DTD__BODY_FIELD_OUTPUT(TYPE, NAME)  TYPE *NAME;

/* Same, read out of the value block and the flows. */
#define PARSEC_DTD__BODY_READ_VALUE(TYPE, NAME)   values__->NAME
#define PARSEC_DTD__BODY_READ_REF(TYPE, NAME)     values__->NAME
#define PARSEC_DTD__BODY_READ_SCRATCH(TYPE, NAME) &values__->NAME
#define PARSEC_DTD__BODY_READ_INPUT(TYPE, NAME)   flows__.NAME
#define PARSEC_DTD__BODY_READ_INOUT(TYPE, NAME)   flows__.NAME
#define PARSEC_DTD__BODY_READ_OUTPUT(TYPE, NAME)  flows__.NAME

/* Operation type recorded in the task class. */
#define PARSEC_DTD__OP_VALUE    PARSEC_VALUE
#define PARSEC_DTD__OP_REF      PARSEC_VALUE
#define PARSEC_DTD__OP_SCRATCH  PARSEC_SCRATCH
#define PARSEC_DTD__OP_INPUT    PARSEC_INPUT
#define PARSEC_DTD__OP_INOUT    PARSEC_INOUT
#define PARSEC_DTD__OP_OUTPUT   PARSEC_OUTPUT

/* Only flows carry an arena/region index, and only flows are passed by
 * reference rather than occupying a slot in the value block. */
#define PARSEC_DTD__ARENA_VALUE(A)   0
#define PARSEC_DTD__ARENA_REF(A)     0
#define PARSEC_DTD__ARENA_SCRATCH(A) 0
#define PARSEC_DTD__ARENA_INPUT(A)   (A)
#define PARSEC_DTD__ARENA_INOUT(A)   (A)
#define PARSEC_DTD__ARENA_OUTPUT(A)  (A)

#define PARSEC_DTD__CLASS_SIZE_VALUE(TYPE)   ((int)PARSEC_DTD__VAL_BYTES_VALUE(TYPE))
#define PARSEC_DTD__CLASS_SIZE_REF(TYPE)     ((int)PARSEC_DTD__VAL_BYTES_REF(TYPE))
#define PARSEC_DTD__CLASS_SIZE_SCRATCH(TYPE) ((int)PARSEC_DTD__VAL_BYTES_SCRATCH(TYPE))
#define PARSEC_DTD__CLASS_SIZE_INPUT(TYPE)   ((int)PASSED_BY_REF)
#define PARSEC_DTD__CLASS_SIZE_INOUT(TYPE)   ((int)PASSED_BY_REF)
#define PARSEC_DTD__CLASS_SIZE_OUTPUT(TYPE)  ((int)PASSED_BY_REF)

/* The (flags, pointer) pair the insertion vararg list expects. Per-insertion
 * flags exist only for flows; the class-level flags live in the task class. */
#define PARSEC_DTD__INSERT_ARG_VALUE(TYPE, NAME)   PARSEC_DTD_EMPTY_FLAG, &args->NAME,
#define PARSEC_DTD__INSERT_ARG_REF(TYPE, NAME)     PARSEC_DTD_EMPTY_FLAG, &args->NAME,
#define PARSEC_DTD__INSERT_ARG_SCRATCH(TYPE, NAME) PARSEC_DTD_EMPTY_FLAG, NULL,
#define PARSEC_DTD__INSERT_ARG_INPUT(TYPE, NAME)   args->NAME##_flags, args->NAME,
#define PARSEC_DTD__INSERT_ARG_INOUT(TYPE, NAME)   args->NAME##_flags, args->NAME,
#define PARSEC_DTD__INSERT_ARG_OUTPUT(TYPE, NAME)  args->NAME##_flags, args->NAME,

/*
 * Per-entry expansions used by PARSEC_DTD_DECLARE_TASK_CLASS.
 */
#define PARSEC_DTD__EMIT_ARG_FIELD(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__ARG_FIELD_##KIND(TYPE, NAME)

#define PARSEC_DTD__EMIT_VAL_FIELD(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__VAL_FIELD_##KIND(TYPE, NAME)

#define PARSEC_DTD__EMIT_VAL_BYTES(KIND, TYPE, NAME, FLAGS) \
    + PARSEC_DTD__VAL_BYTES_##KIND(TYPE)

#define PARSEC_DTD__EMIT_FLOW_FIELD(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__FLOW_FIELD_##KIND(TYPE, NAME)

#define PARSEC_DTD__EMIT_FLOW_FETCH_HOST(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__FLOW_FETCH_##KIND(TYPE, NAME, data_in)

#define PARSEC_DTD__EMIT_FLOW_FETCH_DEVICE(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__FLOW_FETCH_##KIND(TYPE, NAME, data_out)

#define PARSEC_DTD__EMIT_BODY_FIELD(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__BODY_FIELD_##KIND(TYPE, NAME)

#define PARSEC_DTD__EMIT_BODY_ASSIGN(KIND, TYPE, NAME, FLAGS) \
    body->NAME = PARSEC_DTD__BODY_READ_##KIND(TYPE, NAME);

/* parsec_dtd_create_task_class() reads (int size, int op) pairs. */
#define PARSEC_DTD__EMIT_CLASS_ARG(KIND, TYPE, NAME, FLAGS)                \
    PARSEC_DTD__CLASS_SIZE_##KIND(TYPE),                                   \
    PARSEC_DTD__OP_##KIND | (FLAGS) | PARSEC_DTD__ARENA_##KIND(arena_index),

#define PARSEC_DTD__EMIT_INSERT_ARG(KIND, TYPE, NAME, FLAGS) \
    PARSEC_DTD__INSERT_ARG_##KIND(TYPE, NAME)

#define PARSEC_DTD__EMIT_COUNT(KIND, TYPE, NAME, FLAGS) +1

/* One typed local per parameter. The cast to void keeps a parameter the body
 * happens not to read from drawing an unused-variable warning. */
#define PARSEC_DTD__EMIT_LOCAL(KIND, TYPE, NAME, FLAGS)                         \
    PARSEC_DTD__BODY_FIELD_##KIND(TYPE, NAME)                                   \
    NAME = PARSEC_DTD__BODY_READ_##KIND(TYPE, NAME);                            \
    (void)NAME;

/**
 * Declare one local per parameter, with the type taken from the parameter
 * list, and read them out of the task. For bodies that would rather say
 * `alpha` than `task.alpha`; NAME##_unpack() with an explicit NAME##_body_t
 * is otherwise equivalent.
 *
 *   int zgemm_body(parsec_execution_stream_t *es, parsec_task_t *this_task)
 *   {
 *       PARSEC_DTD_UNPACK_LOCALS(zgemm, ZGEMM_PARAMS, this_task);
 *       CORE_zgemm(transA, transB, m, n, k, alpha, A, lda, ...);
 *   }
 */
#define PARSEC_DTD_UNPACK_LOCALS(NAME, PARAMS, THIS_TASK)                       \
    NAME##_values_t *values__ = NAME##_values(THIS_TASK);                       \
    NAME##_flows_t   flows__  = NAME##_flows(THIS_TASK);                        \
    (void)values__; (void)flows__;                                              \
    PARAMS(PARSEC_DTD__EMIT_LOCAL)

/**
 * Generate the argument structs, the task-class constructor, the typed
 * insertion function and the three readers for task class NAME.
 */
#define PARSEC_DTD_DECLARE_TASK_CLASS(NAME, PARAMS)                             \
                                                                                \
enum { NAME##_NB_PARAMS = (0 PARAMS(PARSEC_DTD__EMIT_COUNT)) };                 \
                                                                                \
typedef struct NAME##_args_s {                                                  \
    PARAMS(PARSEC_DTD__EMIT_ARG_FIELD)                                          \
} NAME##_args_t;                                                                \
                                                                                \
typedef struct NAME##_values_s {                                                \
    PARAMS(PARSEC_DTD__EMIT_VAL_FIELD)                                          \
} NAME##_values_t;                                                              \
                                                                                \
typedef struct NAME##_flows_s {                                                 \
    PARAMS(PARSEC_DTD__EMIT_FLOW_FIELD)                                         \
    char NAME##__at_least_one_member;                                           \
} NAME##_flows_t;                                                               \
                                                                                \
typedef struct NAME##_body_s {                                                  \
    PARAMS(PARSEC_DTD__EMIT_BODY_FIELD)                                         \
} NAME##_body_t;                                                                \
                                                                                \
static inline parsec_task_class_t *                                             \
NAME##_task_class_new(parsec_taskpool_t *tp, int arena_index)                    \
{                                                                               \
    (void)arena_index; /* a task class with no data flow has no use for it */    \
    return parsec_dtd_create_task_class(tp, #NAME,                              \
                                        PARAMS(PARSEC_DTD__EMIT_CLASS_ARG)      \
                                        PARSEC_DTD_ARG_END);                    \
}                                                                               \
                                                                                \
static inline void                                                              \
NAME##_insert(parsec_taskpool_t *tp, parsec_task_class_t *tc,                   \
              int priority, int device_type, const NAME##_args_t *args)         \
{                                                                               \
    parsec_dtd_insert_task_with_task_class(tp, tc, priority, device_type,       \
                                          PARAMS(PARSEC_DTD__EMIT_INSERT_ARG)   \
                                          PARSEC_DTD_ARG_END);                  \
}                                                                               \
                                                                                \
/* The by-value parameters, where the runtime already put them. */              \
static inline NAME##_values_t *                                                 \
NAME##_values(parsec_task_t *this_task)                                         \
{                                                                               \
    return (NAME##_values_t *)parsec_dtd_task_values(this_task);                \
}                                                                               \
                                                                                \
/* The data copies on the device the task was scheduled on. */                  \
static inline NAME##_flows_t                                                    \
NAME##_flows(const parsec_task_t *this_task)                                    \
{                                                                               \
    NAME##_flows_t flows__ = { 0 };                                             \
    int flow__ = 0; (void)flow__; (void)this_task;                              \
    PARAMS(PARSEC_DTD__EMIT_FLOW_FETCH_HOST)                                    \
    return flows__;                                                             \
}                                                                               \
                                                                                \
/* The same flows as parsec_dtd_get_dev_ptr() reports, for a device chore. */   \
static inline NAME##_flows_t                                                    \
NAME##_device_flows(const parsec_task_t *this_task)                             \
{                                                                               \
    NAME##_flows_t flows__ = { 0 };                                             \
    int flow__ = 0; (void)flow__; (void)this_task;                              \
    PARAMS(PARSEC_DTD__EMIT_FLOW_FETCH_DEVICE)                                  \
    return flows__;                                                             \
}                                                                               \
                                                                                \
static inline void                                                              \
NAME##_unpack(parsec_task_t *this_task, NAME##_body_t *body)                    \
{                                                                               \
    NAME##_values_t *values__ = NAME##_values(this_task);                       \
    NAME##_flows_t   flows__  = NAME##_flows(this_task);                        \
    (void)values__; (void)flows__;                                              \
    PARAMS(PARSEC_DTD__EMIT_BODY_ASSIGN)                                        \
}                                                                               \
                                                                                \
/* If the compiler laid NAME##_values_t out with any padding of its own, it no  \
 * longer describes the value block and every field past the padding would be   \
 * read from the wrong offset. */                                               \
_Static_assert(sizeof(NAME##_values_t) ==                                       \
               (0 PARAMS(PARSEC_DTD__EMIT_VAL_BYTES)),                          \
               "DTD task class " #NAME ": the generated value struct does not " \
               "match the parameter sizes declared to the runtime");            \
                                                                                \
_Static_assert(NAME##_NB_PARAMS <= PARSEC_DTD_MAX_PARAMS,                       \
               "DTD task class " #NAME " declares too many parameters")

END_C_DECLS

#endif /* PARSEC_DTD_TASK_CLASS_DECL_H_HAS_BEEN_INCLUDED */
