/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/*
 * Class initialization:
 *  - many threads create the first object of a class at the same time, and
 *    must all see fully built constructor arrays (release/acquire publication
 *    of cls_initialized);
 *  - parsec_class_finalize() returns the classes to their static state so the
 *    class system can be used again.
 */

#include "parsec/runtime.h"
#undef NDEBUG
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_object.h"

#define NB_THREADS 16
#define NB_ROUNDS  200

typedef struct {
    parsec_object_t super;
    int base_value;
    int derived_value;
} race_t;

static void race_base_construct(race_t *obj) { obj->base_value = 1; }
static void race_derived_construct(race_t *obj)
{
    /* the parent constructor must have run first */
    obj->derived_value = obj->base_value + 1;
}

/* Two levels of inheritance, so the constructor array has two entries. The
 * descriptors are rebuilt by hand between the tests to get them uninitialized. */
static parsec_class_t race_base_class;
static parsec_class_t race_derived_class;

static void reset_classes(void)
{
    race_base_class = (parsec_class_t){ "race_base", PARSEC_OBJ_CLASS(parsec_object_t),
        (parsec_construct_t)race_base_construct, NULL, 0, 0, NULL, NULL, sizeof(race_t) };
    race_derived_class = (parsec_class_t){ "race_derived", &race_base_class,
        (parsec_construct_t)race_derived_construct, NULL, 0, 0, NULL, NULL, sizeof(race_t) };
}

static volatile int32_t nb_ready = 0;
static volatile int32_t go = 0;
static volatile int32_t nb_errors = 0;

static void *first_use(void *arg)
{
    race_t *obj;
    (void)arg;
    /* spin barrier, to make all threads hit the first use at the same time */
    parsec_atomic_fetch_inc_int32(&nb_ready);
    while( !go ) { parsec_atomic_rmb(); }

    obj = (race_t*)parsec_obj_new(&race_derived_class);
    if( 1 != obj->base_value || 2 != obj->derived_value ) {
        parsec_atomic_fetch_inc_int32(&nb_errors);
    }
    PARSEC_OBJ_RELEASE(obj);
    return NULL;
}

static void race_first_use(void)
{
    pthread_t threads[NB_THREADS];
    int t;

    reset_classes();
    nb_ready = 0;
    go = 0;
    for( t = 0; t < NB_THREADS; t++ )
        pthread_create(&threads[t], NULL, first_use, NULL);
    while( NB_THREADS != nb_ready ) { parsec_atomic_rmb(); }
    go = 1;
    parsec_atomic_wmb();
    for( t = 0; t < NB_THREADS; t++ )
        pthread_join(threads[t], NULL);
    assert(PARSEC_CLASS_IS_INITIALIZED(&race_derived_class));
    parsec_class_finalize();
}

/* The classes are usable again after the class system has been finalized */
static void finalize_and_reuse(void)
{
    int round;

    reset_classes();
    for( round = 0; round < 3; round++ ) {
        race_t *obj;
        assert(!PARSEC_CLASS_IS_INITIALIZED(&race_derived_class));
        obj = (race_t*)parsec_obj_new(&race_derived_class);
        assert(PARSEC_CLASS_IS_INITIALIZED(&race_derived_class));
        assert(3 == race_derived_class.cls_depth);
        assert(1 == obj->base_value && 2 == obj->derived_value);
        PARSEC_OBJ_RELEASE(obj);
        parsec_class_finalize();
        assert(!PARSEC_CLASS_IS_INITIALIZED(&race_derived_class));
        assert(!PARSEC_CLASS_IS_INITIALIZED(&race_base_class));
        assert(NULL == race_derived_class.cls_construct_array);
    }
}

int main(void)
{
    int round;
    for( round = 0; round < NB_ROUNDS; round++ ) {
        race_first_use();
    }
    assert(0 == nb_errors);
    finalize_and_reuse();
    printf("class_init: success\n");
    return 0;
}
