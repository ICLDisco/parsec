/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Coverage for parsec_arena_datatype_set_type() when the arena cannot be
 * constructed (an elem_size of 0, or an alignment that is not a power of two
 * larger than 1). The call must report PARSEC_ERR_BAD_PARAM, and the adt
 * must still release cleanly: in a Debug build the arena destructor asserts
 * on its counters, so an arena left with uninitialized counters aborts here.
 * An arena that never went through parsec_arena_construct() must be
 * releasable as well.
 *
 * Before each allocation, a block of the arena's size is filled with
 * nonzero bytes and freed, so that the allocator is likely to hand the same
 * memory to the arena and stale bytes, not zeroes, sit in any field the
 * arena leaves unset.
 */

#include "parsec/runtime.h"
#undef NDEBUG
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsec/parsec_internal.h"  /* struct parsec_arena_datatype_s */
#include "parsec/arena.h"
#include "parsec/data.h"
#include "parsec/datatype.h"

static int failures = 0;

static void check(int condition, const char *what)
{
    if( !condition ) {
        fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void dirty_heap(void)
{
    unsigned char *p = malloc(sizeof(parsec_arena_t));
    if( NULL == p ) return;
    /* text-like bytes that differ from one field to the next, so that stale
     * counters neither compare equal nor read as 0 or INT32_MAX */
    for( size_t i = 0; i < sizeof(parsec_arena_t); i++ )
        p[i] = (unsigned char)('A' + i % 26);
    free(p);
}

/* set_type() on a heap adt with parameters the arena rejects */
static void test_rejected(size_t elem_size, size_t alignment, const char *what)
{
    parsec_arena_datatype_t *adt;
    int rc;

    dirty_heap();
    adt = parsec_arena_datatype_new();
    rc = parsec_arena_datatype_set_type(adt, elem_size, alignment, PARSEC_DATATYPE_NULL);
    check(PARSEC_ERR_BAD_PARAM == rc, what);
    check(NULL != adt->arena, "the arena stays attached after a failed set_type");
    if( NULL != adt->arena ) {
        check(0 == adt->arena->elem_size, "a rejected arena is marked as not constructed");
        check(0 == adt->arena->used && 0 == adt->arena->released,
              "a rejected arena has zeroed counters");
    }
    parsec_arena_datatype_release(&adt);
}

/* the same on an adt embedded in another structure, as in a PTG taskpool */
static void test_rejected_embedded(void)
{
    parsec_arena_datatype_t adt;
    int rc;

    PARSEC_OBJ_CONSTRUCT(&adt, parsec_arena_datatype_t);
    dirty_heap();
    rc = parsec_arena_datatype_set_type(&adt, 0, PARSEC_ARENA_ALIGNMENT_SSE, PARSEC_DATATYPE_NULL);
    check(PARSEC_ERR_BAD_PARAM == rc, "embedded adt: elem_size 0 is rejected");
    PARSEC_OBJ_DESTRUCT(&adt);
}

static void test_accepted(void)
{
    parsec_arena_datatype_t *adt;
    int rc;

    dirty_heap();
    adt = parsec_arena_datatype_new();
    rc = parsec_arena_datatype_set_type(adt, 64, PARSEC_ARENA_ALIGNMENT_SSE, PARSEC_DATATYPE_NULL);
    check(PARSEC_SUCCESS == rc, "elem_size 64, SSE alignment is accepted");
    check(NULL != adt->arena && 64 == adt->arena->elem_size,
          "an accepted arena records its element size");
    parsec_arena_datatype_release(&adt);
}

static void test_never_constructed(void)
{
    parsec_arena_t *arena;

    dirty_heap();
    arena = PARSEC_OBJ_NEW(parsec_arena_t);
    check(0 == arena->elem_size, "a new arena is marked as not constructed");
    check(0 == arena->used && 0 == arena->released &&
          0 == arena->max_used && 0 == arena->max_released,
          "a new arena has zeroed counters");
    PARSEC_OBJ_RELEASE(arena);
}

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    test_rejected(0, PARSEC_ARENA_ALIGNMENT_SSE, "elem_size 0 is rejected");
    test_rejected(64, 0, "alignment 0 is rejected");
    test_rejected(64, 1, "alignment 1 is rejected");
    test_rejected(64, 24, "alignment 24 is rejected");
    test_rejected_embedded();
    test_accepted();
    test_never_constructed();

    if( 0 != failures ) {
        fprintf(stderr, "arena_datatype: %d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    printf("arena_datatype: all tests passed\n");
    return EXIT_SUCCESS;
}
