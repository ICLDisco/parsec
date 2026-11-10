/*
 * Copyright (c) 2026
 */

/**
 * PARSEC_ARENA_AUTO picks the arena from the size of the request: up to
 * parsec_arena_auto_max_bucket bytes come as one element of the smallest
 * power-of-two bucket that holds them (64 bytes at least), which keeps them
 * for the next request of that bucket; anything larger is allocated on demand,
 * in grains of 64 bytes, and freed on release. Whatever serves it, the data
 * spans exactly the request.
 *
 * Check this at the edges of the buckets and of the largest one, with a
 * datatype whose extent differs from its packed size, and that a request a
 * size_t cannot measure is refused instead of wrapping around. The arenas are
 * only observed through what they hand out: which bucket served a request
 * shows in the buffer the next request of that bucket gets back.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_internal.h"
#include "parsec/datatype.h"
#include "tests/tests_runtime.h"

static int errors = 0;

#define EXPECT(cond, ...)                                   \
    do {                                                    \
        if( !(cond) ) {                                     \
            errors++;                                       \
            fprintf(stderr, "arena_auto: " __VA_ARGS__);    \
            fputc('\n', stderr);                            \
        }                                                   \
    } while(0)

/* The bucket serving a request of bytes */
static size_t bucket_of(size_t bytes)
{
    size_t bucket = 64;
    while( bucket < bytes ) bucket <<= 1;
    return bucket;
}

/* Request count elements of dtt, each spanning extent bytes: check the span,
 * the datatype, that all of it is writable and, above the largest bucket, the
 * number of grains; release it and return where it was. */
static void *request(const char *what, size_t count, parsec_datatype_t dtt, size_t extent)
{
    size_t bytes = count * extent;
    parsec_data_copy_t *copy = parsec_arena_get_new_copy(PARSEC_ARENA_AUTO, count, 0, dtt);
    void *ptr;

    EXPECT(NULL != copy, "%s: no copy for %zu elements of %zu bytes", what, count, extent);
    if( NULL == copy ) return NULL;
    EXPECT(bytes == copy->original->span, "%s: %zu bytes span %zu", what, bytes, copy->original->span);
    EXPECT(dtt == copy->dtt, "%s: %zu bytes: the datatype was not kept", what, bytes);
    if( bytes <= parsec_arena_auto_max_bucket ) {
        EXPECT(1 == copy->arena_chunk->count,
               "%s: %zu bytes served as %u elements, expected one of a bucket",
               what, bytes, copy->arena_chunk->count);
    } else {
        size_t grains = (bytes + 63) / 64;
        EXPECT(grains == copy->arena_chunk->count,
               "%s: %zu bytes served as %u elements, expected %zu grains of 64 bytes",
               what, bytes, copy->arena_chunk->count, grains);
    }
    memset(copy->device_private, 0x5a, bytes);  /* all of the request is writable */
    ptr = copy->device_private;
    PARSEC_DATA_COPY_RELEASE(copy);
    return ptr;
}

/* A request of bytes is served by the bucket of bucket_of(bytes): once
 * released, a request of exactly that size gets the same buffer back, and a
 * request of the next bucket does not, since the first bucket keeps it. */
static void edge(size_t bytes)
{
    size_t bucket = bucket_of(bytes);
    void *p = request("bytes", bytes, parsec_datatype_int8_t, 1);

    if( NULL == p ) return;
    EXPECT(p == request("bytes", bucket, parsec_datatype_int8_t, 1),
           "%zu bytes: not served by the bucket of %zu", bytes, bucket);
    if( 2 * bucket <= parsec_arena_auto_max_bucket ) {
        EXPECT(p != request("bytes", 2 * bucket, parsec_datatype_int8_t, 1),
               "%zu bytes: the bucket of %zu handed out what the bucket of %zu keeps",
               bytes, 2 * bucket, bucket);
    }
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec = NULL;
    parsec_datatype_t dstrided = PARSEC_DATATYPE_NULL;
    parsec_data_copy_t *copy;
    size_t edges[] = { 1, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097, 65536, 65537 };
    size_t max_bucket;
    void *p;
    int rank, world, rc;

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED, &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");
    max_bucket = parsec_arena_auto_max_bucket;

    /* bytes, at the edges of the first buckets and of the largest one */
    for(size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); i++) {
        edge(edges[i]);
    }
    edge(max_bucket - 1);
    edge(max_bucket);
    /* just above it, the request is served by itself, in grains */
    (void)request("bytes", max_bucket + 1, parsec_datatype_int8_t, 1);
    (void)request("bytes", 3 * max_bucket, parsec_datatype_int8_t, 1);

    /* a smaller largest bucket moves the edge */
    parsec_arena_auto_max_bucket = 4096;
    edge(4096);
    (void)request("bytes", 4097, parsec_datatype_int8_t, 1);
    parsec_arena_auto_max_bucket = max_bucket;

    /* a double every 16 bytes: packed size 8, extent 16. The request is
     * measured by the extent: 4 of them take the 64-byte bucket, 5 take 80
     * bytes of the 128-byte one. Without MPI there are no derived datatypes. */
    rc = parsec_type_create_resized(parsec_datatype_double_t, 0, 2 * sizeof(double), &dstrided);
    PARSEC_CHECK_ERROR(rc, "parsec_type_create_resized");
    if( PARSEC_DATATYPE_NULL != dstrided ) {
        p = request("resized", 4, dstrided, 16);
        EXPECT(p == request("bytes", 64, parsec_datatype_int8_t, 1),
               "4 resized doubles (64 bytes) were not served by the bucket of 64");
        p = request("resized", 5, dstrided, 16);
        EXPECT(p == request("bytes", 128, parsec_datatype_int8_t, 1),
               "5 resized doubles (80 bytes) were not served by the bucket of 128");
        (void)request("resized", 1, dstrided, 16);
        (void)request("resized", max_bucket / 16, dstrided, 16);
        (void)request("resized", max_bucket / 16 + 1, dstrided, 16);
    }
    p = request("doubles", 8, parsec_datatype_double_t, 8);
    EXPECT(p == request("bytes", 64, parsec_datatype_int8_t, 1),
           "8 doubles were not served by the bucket of 64");
    (void)request("doubles", 9, parsec_datatype_double_t, 8);

    /* nothing, and more than a size_t holds: refused, not wrapped around */
    copy = parsec_arena_get_new_copy(PARSEC_ARENA_AUTO, 0, 0, parsec_datatype_int8_t);
    EXPECT(NULL == copy, "a request of 0 bytes was served");
    if( NULL != copy ) PARSEC_DATA_COPY_RELEASE(copy);
    copy = parsec_arena_get_new_copy(PARSEC_ARENA_AUTO, SIZE_MAX / 8 + 2, 0, parsec_datatype_double_t);
    EXPECT(NULL == copy, "a request of %zu doubles, more than a size_t holds, was served", SIZE_MAX / 8 + 2);
    if( NULL != copy ) PARSEC_DATA_COPY_RELEASE(copy);
    if( PARSEC_DATATYPE_NULL != dstrided ) {
        copy = parsec_arena_get_new_copy(PARSEC_ARENA_AUTO, SIZE_MAX / 16 + 2, 0, dstrided);
        EXPECT(NULL == copy, "a request of %zu resized doubles, more than a size_t holds, was served", SIZE_MAX / 16 + 2);
        if( NULL != copy ) PARSEC_DATA_COPY_RELEASE(copy);
        parsec_type_free(&dstrided);
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( errors ) {
        fprintf(stderr, "arena_auto: FAILED (%d errors)\n", errors);
        return EXIT_FAILURE;
    }
    printf("arena_auto: PASS\n");
    return EXIT_SUCCESS;
}
