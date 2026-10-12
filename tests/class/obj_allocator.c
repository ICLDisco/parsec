/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#undef NDEBUG
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>
#include <unistd.h>

#include "parsec/class/parsec_object.h"
#include "parsec/class/parsec_obj_freelist.h"
#if defined(BUILDING_PARSEC)
#include "parsec/mempool.h"
#endif  /* defined(BUILDING_PARSEC) */

/* a test class that counts constructions and destructions */
typedef struct {
    parsec_object_t super;
    int value;
    char payload[200];
} counted_t;

static int32_t nb_constructed = 0;
static int32_t nb_destructed = 0;

static void counted_construct(counted_t *obj)
{
    obj->value = 42;
    __sync_fetch_and_add(&nb_constructed, 1);
}

static void counted_destruct(counted_t *obj)
{
    assert(42 == obj->value);
    obj->value = -1;
    __sync_fetch_and_add(&nb_destructed, 1);
}

PARSEC_OBJ_CLASS_DECLARATION(counted_t);
PARSEC_OBJ_CLASS_INSTANCE(counted_t, parsec_object_t, counted_construct, counted_destruct);

/* a custom allocator with its own state */
typedef struct {
    parsec_obj_allocator_t super;
    int nb_alloc;
    int nb_release;
} counting_allocator_t;

static void *counting_alloc(parsec_obj_allocator_t *allocator, parsec_class_t *cls)
{
    ((counting_allocator_t*)allocator)->nb_alloc++;
    return malloc(cls->cls_sizeof);
}

static void counting_release(parsec_obj_allocator_t *allocator, parsec_object_t *obj)
{
    ((counting_allocator_t*)allocator)->nb_release++;
    parsec_obj_destruct(obj);
    free(obj);
}

static void test_custom_allocator(void)
{
    counting_allocator_t allocator = { { counting_alloc, counting_release }, 0, 0 };
    counted_t *obj = PARSEC_OBJ_NEW_WALLOCATOR(counted_t, &allocator.super);
    assert(NULL != obj && 42 == obj->value);
    assert(1 == allocator.nb_alloc && 0 == allocator.nb_release);
    PARSEC_OBJ_RETAIN(obj);
    PARSEC_OBJ_RELEASE(obj);
    assert(NULL != obj && 0 == allocator.nb_release);
    PARSEC_OBJ_RELEASE(obj);
    assert(NULL == obj && 1 == allocator.nb_release);

    /* construct in place, release through the allocator */
    obj = malloc(sizeof(counted_t));
    PARSEC_OBJ_CONSTRUCT_WALLOCATOR(obj, counted_t, &allocator.super);
    PARSEC_OBJ_RELEASE(obj);
    assert(NULL == obj && 2 == allocator.nb_release);
    assert(nb_constructed == nb_destructed);
}

static void test_default_allocators(void)
{
    counted_t stack_obj;
    counted_t *obj = PARSEC_OBJ_NEW(counted_t);
    PARSEC_OBJ_RELEASE(obj);
    assert(NULL == obj);

    PARSEC_OBJ_CONSTRUCT(&stack_obj, counted_t);
    PARSEC_OBJ_DESTRUCT(&stack_obj);
    assert(nb_constructed == nb_destructed);
}

static void test_freelist_serial(void)
{
    parsec_obj_freelist_t fl;
    counted_t *a, *b;
    void *addr;

    PARSEC_OBJ_FREELIST_CONSTRUCT(&fl, counted_t);
    a = PARSEC_OBJ_FREELIST_NEW(&fl, counted_t);
    assert(42 == a->value);
    addr = a;
    PARSEC_OBJ_RELEASE(a);
    assert(nb_constructed == nb_destructed);
    /* storage is recycled and constructors are run again */
    b = PARSEC_OBJ_FREELIST_NEW(&fl, counted_t);
    assert(addr == (void*)b && 42 == b->value);
    assert(1 == fl.nb_elt);
    PARSEC_OBJ_RETAIN(b);
    PARSEC_OBJ_RELEASE(b);
    PARSEC_OBJ_RELEASE(b);
    assert(0 == parsec_obj_freelist_destruct(&fl));
    assert(nb_constructed == nb_destructed);
}

#define NB_THREADS 4
#define NB_ITER    100000
#define NB_HELD    16
static parsec_obj_freelist_t shared_fl;

static void *freelist_thread(void *arg)
{
    counted_t *held[NB_HELD];
    (void)arg;
    for(int i = 0; i < NB_ITER; i++) {
        for(int j = 0; j < NB_HELD; j++) {
            held[j] = PARSEC_OBJ_FREELIST_NEW(&shared_fl, counted_t);
            assert(42 == held[j]->value);
        }
        for(int j = 0; j < NB_HELD; j++) {
            PARSEC_OBJ_RELEASE(held[j]);
        }
    }
    return NULL;
}

static void test_freelist_threaded(void)
{
    pthread_t threads[NB_THREADS];
    PARSEC_OBJ_FREELIST_CONSTRUCT(&shared_fl, counted_t);
    for(int t = 0; t < NB_THREADS; t++)
        pthread_create(&threads[t], NULL, freelist_thread, NULL);
    for(int t = 0; t < NB_THREADS; t++)
        pthread_join(threads[t], NULL);
    assert(shared_fl.nb_elt <= NB_THREADS * NB_HELD);
    printf("threaded free-list: %d elements for %d allocations\n",
           shared_fl.nb_elt, NB_THREADS * NB_ITER * NB_HELD);
    assert(0 == parsec_obj_freelist_destruct(&shared_fl));
    assert(nb_constructed == nb_destructed);
}

#if defined(BUILDING_PARSEC)
/* mempool elements are list items with a back pointer to their thread mempool */
typedef struct {
    parsec_list_item_t super;
    parsec_thread_mempool_t *mempool_owner;
    int value;
} pooled_t;

static int32_t nb_pooled_constructed = 0;
static int32_t nb_pooled_destructed = 0;

static void pooled_construct(pooled_t *obj)
{
    obj->value = 7;
    __sync_fetch_and_add(&nb_pooled_constructed, 1);
}

static void pooled_destruct(pooled_t *obj)
{
    assert(7 == obj->value);
    obj->value = -1;
    __sync_fetch_and_add(&nb_pooled_destructed, 1);
}

PARSEC_OBJ_CLASS_DECLARATION(pooled_t);
PARSEC_OBJ_CLASS_INSTANCE(pooled_t, parsec_list_item_t, pooled_construct, pooled_destruct);

static parsec_mempool_t shared_mempool;

static void *mempool_thread(void *arg)
{
    parsec_thread_mempool_t *tm = &shared_mempool.thread_mempools[(intptr_t)arg];
    pooled_t *held[NB_HELD];
    for(int i = 0; i < NB_ITER; i++) {
        for(int j = 0; j < NB_HELD; j++) {
            if( j % 2 ) {
                held[j] = PARSEC_THREAD_MEMPOOL_OBJ_NEW(pooled_t, tm);
            } else {
                held[j] = parsec_thread_mempool_allocate(tm);
            }
            assert(7 == held[j]->value && tm == held[j]->mempool_owner);
        }
        for(int j = 0; j < NB_HELD; j++) {
            if( j % 3 ) {
                PARSEC_OBJ_RETAIN(held[j]);
                PARSEC_OBJ_RELEASE(held[j]);
                PARSEC_OBJ_RELEASE(held[j]);
            } else {
                parsec_thread_mempool_free(tm, held[j]);
            }
        }
    }
    return NULL;
}

static void test_mempool(void)
{
    parsec_mempool_t mempool;
    parsec_thread_mempool_t *tm;
    pooled_t *a, *b;
    pthread_t threads[NB_THREADS];
    uint64_t nb_elt;

    /* serial: object model and legacy interface on the same mempool */
    parsec_mempool_construct(&mempool, PARSEC_OBJ_CLASS(pooled_t), sizeof(pooled_t),
                             offsetof(pooled_t, mempool_owner), 1);
    tm = &mempool.thread_mempools[0];
    a = PARSEC_THREAD_MEMPOOL_OBJ_NEW(pooled_t, tm);
    assert(7 == a->value && tm == a->mempool_owner);
    PARSEC_OBJ_RELEASE(a);
    assert(NULL == a && nb_pooled_constructed == nb_pooled_destructed);
    b = parsec_thread_mempool_allocate(tm);
    assert(7 == b->value && 1 == b->super.super.obj_reference_count);
    parsec_mempool_free(&mempool, b);
    assert(nb_pooled_constructed == nb_pooled_destructed);
    nb_elt = parsec_mempool_destruct(&mempool);
    assert(1 == nb_elt);

    /* threaded: every thread uses its own thread mempool */
    parsec_mempool_construct(&shared_mempool, PARSEC_OBJ_CLASS(pooled_t), sizeof(pooled_t),
                             offsetof(pooled_t, mempool_owner), NB_THREADS);
    for(intptr_t t = 0; t < NB_THREADS; t++)
        pthread_create(&threads[t], NULL, mempool_thread, (void*)t);
    for(int t = 0; t < NB_THREADS; t++)
        pthread_join(threads[t], NULL);
    nb_elt = parsec_mempool_destruct(&shared_mempool);
    assert(nb_elt <= NB_THREADS * NB_HELD);
    assert(nb_pooled_constructed == nb_pooled_destructed);
    printf("threaded mempool: %"PRIu64" elements for %d allocations\n",
           nb_elt, NB_THREADS * NB_ITER * NB_HELD);
}
#endif  /* defined(BUILDING_PARSEC) */

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;
    test_default_allocators();
    test_custom_allocator();
    test_freelist_serial();
    test_freelist_threaded();
#if defined(BUILDING_PARSEC)
    test_mempool();
#endif  /* defined(BUILDING_PARSEC) */
    printf("obj_allocator: all tests passed\n");
    return 0;
}
