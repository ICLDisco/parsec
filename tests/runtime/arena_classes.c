/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/**
 * parsec_arena_set_count_classes(): an arena caches the chunks of several
 * elements it hands out by power-of-two class. Checks, on an arena of 8-byte
 * elements:
 *  - a request for n > 1 elements gets a chunk of the smallest power of two
 *    >= n, and a copy spanning exactly n elements;
 *  - a released chunk is handed out again to the next request of its class,
 *    and not to a request of another class;
 *  - above the largest class the request is served as is, and not cached;
 *  - the arena is destructed with chunks still cached, its max_used bounded,
 *    which checks that cached chunks stay counted as used.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_internal.h"
#include "tests/tests_runtime.h"

#define ELEM sizeof(double)
#define MAX_COUNT 4096

static int errors = 0;

#define EXPECT(cond, ...)                                       \
    do {                                                        \
        if( !(cond) ) {                                         \
            errors++;                                           \
            fprintf(stderr, "arena_classes: " __VA_ARGS__);     \
            fputc('\n', stderr);                                \
        }                                                       \
    } while(0)

static size_t class_of(size_t count)
{
    size_t c = 2;
    while( c < count ) c <<= 1;
    return c;
}

/* Allocate count elements, check the chunk and the span, release; returns the buffer */
static void *request(parsec_arena_t *arena, size_t count)
{
    parsec_data_copy_t *copy = parsec_arena_get_new_copy(arena, count, 0, parsec_datatype_double_t);
    size_t expected = ((count > 1) && (count <= MAX_COUNT)) ? class_of(count) : count;
    void *ptr;

    EXPECT(NULL != copy, "no copy for %zu elements", count);
    if( NULL == copy ) return NULL;
    EXPECT(count * ELEM == copy->original->span, "%zu elements span %zu bytes", count, copy->original->span);
    EXPECT(expected == copy->arena_chunk->count, "%zu elements served as a chunk of %u, expected %zu",
           count, copy->arena_chunk->count, expected);
    memset(copy->device_private, 0x5a, count * ELEM);  /* all of the request is writable */
    ptr = copy->device_private;
    PARSEC_OBJ_RELEASE(copy->original);  /* a self-contained copy, as a receive buffer is */
    PARSEC_DATA_COPY_RELEASE(copy);
    return ptr;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec = NULL;
    parsec_arena_t *arena;
    size_t edges[] = { 2, 3, 4, 5, 7, 8, 9, 100, 1023, 1024, 1025, MAX_COUNT - 1, MAX_COUNT };
    int rank, world, rc;
    void *p;

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED, &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    /* bounded max_used, so that the destructor checks used == released */
    arena = PARSEC_OBJ_NEW(parsec_arena_t);
    rc = parsec_arena_construct_ex(arena, ELEM, PARSEC_ARENA_ALIGNMENT_SSE,
                                   (size_t)64 * MAX_COUNT * ELEM, (size_t)16 * MAX_COUNT * ELEM);
    PARSEC_CHECK_ERROR(rc, "parsec_arena_construct_ex");
    rc = parsec_arena_set_count_classes(arena, MAX_COUNT);
    PARSEC_CHECK_ERROR(rc, "parsec_arena_set_count_classes");

    for( size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); i++ ) {
        size_t n = edges[i], c = class_of(n);
        p = request(arena, n);
        EXPECT(p == request(arena, c), "%zu elements: chunk not reused by a request of %zu", n, c);
        if( 2 * c <= MAX_COUNT ) {
            EXPECT(p != request(arena, 2 * c), "%zu elements: chunk of class %zu handed to a request of %zu",
                   n, c, 2 * c);
        }
    }
    /* single elements keep their own cache */
    p = request(arena, 1);
    EXPECT(p == request(arena, 1), "a single element was not reused");
    /* above the largest class: served as is */
    (void)request(arena, MAX_COUNT + 1);
    (void)request(arena, 3 * MAX_COUNT);

    /* several chunks of one class alive at once, then all cached */
    {
        parsec_data_copy_t *copies[8];
        for( int i = 0; i < 8; i++ ) {
            copies[i] = parsec_arena_get_new_copy(arena, 300, 0, parsec_datatype_double_t);
            EXPECT(NULL != copies[i], "no copy %d of 300 elements", i);
        }
        for( int i = 0; i < 8; i++ ) {
            if( NULL == copies[i] ) continue;
            PARSEC_OBJ_RELEASE(copies[i]->original);
            PARSEC_DATA_COPY_RELEASE(copies[i]);
        }
    }

    PARSEC_OBJ_RELEASE(arena);  /* with chunks cached in several classes */

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    if( 0 != errors ) {
        fprintf(stderr, "arena_classes: FAILED (%d errors)\n", errors);
        return EXIT_FAILURE;
    }
    printf("arena_classes: PASS\n");
    return EXIT_SUCCESS;
}
