/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* Coverage for parsec_taskpool_trim_chores() and the primitive underneath
 * it, on a taskpool assembled by hand so that the incarnations of every task
 * class are known exactly. What is checked is that the right incarnations go
 * away, that the ones left keep their order and their terminator, that a task
 * class is never stripped of all of them, and that a specification which does
 * not parse changes nothing.
 */

#include "parsec/runtime.h"
#undef NDEBUG
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsec/parsec_internal.h"
#include "parsec/mca/device/device.h"

static int failures = 0;

static void check(int condition, const char *what)
{
    if( !condition ) {
        fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

/* Each task class of the fixture starts from this set, in this order. */
static const uint8_t initial_types[] = {
    PARSEC_DEV_CUDA, PARSEC_DEV_HIP, PARSEC_DEV_LEVEL_ZERO, PARSEC_DEV_CPU
};
#define NB_INITIAL_TYPES ((int)(sizeof(initial_types) / sizeof(initial_types[0])))

static const char *class_names[] = { "gemm", "potrf", "trsm" };
#define NB_CLASSES ((int)(sizeof(class_names) / sizeof(class_names[0])))

/* A hook per incarnation, so that a chore that moved can still be told from
 * the one that used to sit at its index. */
static parsec_hook_t *hook_for(int class_index, int type_index)
{
    return (parsec_hook_t*)(uintptr_t)(1 + class_index * NB_INITIAL_TYPES + type_index);
}

static parsec_taskpool_t *fixture_new(void)
{
    parsec_taskpool_t *tp = (parsec_taskpool_t*)calloc(1, sizeof(parsec_taskpool_t));

    tp->nb_task_classes = NB_CLASSES;
    tp->taskpool_name = strdup("fixture");
    tp->task_classes_array = (const parsec_task_class_t**)
        calloc(NB_CLASSES, sizeof(parsec_task_class_t*));

    for( int i = 0; i < NB_CLASSES; i++ ) {
        parsec_task_class_t *tc = (parsec_task_class_t*)calloc(1, sizeof(parsec_task_class_t));
        __parsec_chore_t *chores = (__parsec_chore_t*)
            calloc(NB_INITIAL_TYPES + 1, sizeof(__parsec_chore_t));

        for( int j = 0; j < NB_INITIAL_TYPES; j++ ) {
            chores[j].type = initial_types[j];
            chores[j].hook = hook_for(i, j);
        }
        chores[NB_INITIAL_TYPES].type = PARSEC_DEV_NONE;

        tc->name = class_names[i];
        tc->incarnations = chores;
        tp->task_classes_array[i] = tc;
    }
    return tp;
}

static void fixture_free(parsec_taskpool_t *tp)
{
    for( int i = 0; i < NB_CLASSES; i++ ) {
        free((void*)tp->task_classes_array[i]->incarnations);
        free((void*)tp->task_classes_array[i]);
    }
    free(tp->task_classes_array);
    free(tp->taskpool_name);
    free(tp);
}

/* The types a task class is left with, in order, and how many there are. */
static int types_of(parsec_taskpool_t *tp, const char *name, uint8_t *types)
{
    for( int i = 0; i < NB_CLASSES; i++ ) {
        const parsec_task_class_t *tc = tp->task_classes_array[i];
        if( 0 != strcmp(tc->name, name) ) continue;
        int count;
        for( count = 0; PARSEC_DEV_NONE != tc->incarnations[count].type; count++ ) {
            types[count] = (uint8_t)tc->incarnations[count].type;
        }
        return count;
    }
    return -1;
}

/* Whether a task class kept the hooks of the types it still declares. Catches
 * a compaction that shifts the types without carrying the rest along. */
static int hooks_are_consistent(parsec_taskpool_t *tp)
{
    for( int i = 0; i < NB_CLASSES; i++ ) {
        const parsec_task_class_t *tc = tp->task_classes_array[i];
        for( int k = 0; PARSEC_DEV_NONE != tc->incarnations[k].type; k++ ) {
            int expected = -1;
            for( int j = 0; j < NB_INITIAL_TYPES; j++ ) {
                if( initial_types[j] == tc->incarnations[k].type ) expected = j;
            }
            if( (expected < 0) || (tc->incarnations[k].hook != hook_for(i, expected)) ) {
                return 0;
            }
        }
    }
    return 1;
}

static void test_one_class_one_type(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];
    int rc = parsec_taskpool_trim_chores(tp, "trsm:cuda");

    check(1 == rc, "removing one type from one class reports one removal");
    check(3 == types_of(tp, "trsm", types), "trsm keeps its three other incarnations");
    check((PARSEC_DEV_HIP == types[0]) && (PARSEC_DEV_LEVEL_ZERO == types[1]) &&
          (PARSEC_DEV_CPU == types[2]), "the incarnations left keep their order");
    check(4 == types_of(tp, "gemm", types), "a class that was not named is untouched");
    check(hooks_are_consistent(tp), "every incarnation left kept its own hook");
    fixture_free(tp);
}

static void test_several_types_and_classes(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];
    int rc = parsec_taskpool_trim_chores(tp, "gemm:cuda,hip;potrf:level_zero");

    check(3 == rc, "two entries report every removal between them");
    check(2 == types_of(tp, "gemm", types), "gemm gave up the two types it was asked for");
    check((PARSEC_DEV_LEVEL_ZERO == types[0]) && (PARSEC_DEV_CPU == types[1]),
          "gemm kept the rest in order");
    check(3 == types_of(tp, "potrf", types), "potrf gave up the one type it was asked for");
    check(4 == types_of(tp, "trsm", types), "trsm was named by neither entry");
    fixture_free(tp);
}

static void test_groups_and_wildcard(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];
    int rc = parsec_taskpool_trim_chores(tp, "*:gpu");

    check(9 == rc, "gpu stands for every accelerator type of every class");
    for( int i = 0; i < NB_CLASSES; i++ ) {
        check(1 == types_of(tp, class_names[i], types), "each class is left with one incarnation");
        check(PARSEC_DEV_CPU == types[0], "the one left is the CPU incarnation");
    }
    fixture_free(tp);
}

static void test_case_is_ignored(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];

    check(1 == parsec_taskpool_trim_chores(tp, "POTRF:CUDA"),
          "the class and the type are both matched without regard to case");
    check(3 == types_of(tp, "potrf", types), "the matching class is the one that gave up a type");
    fixture_free(tp);
}

static void test_unknown_class_is_not_an_error(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];

    check(0 == parsec_taskpool_trim_chores(tp, "nowhere:cuda"),
          "a class the taskpool does not have removes nothing and is not an error");
    check(4 == types_of(tp, "gemm", types), "and leaves the classes it does have alone");
    fixture_free(tp);
}

static void test_emptying_a_class_is_refused(void)
{
    parsec_taskpool_t *tp = fixture_new();
    uint8_t types[NB_INITIAL_TYPES];

    check(PARSEC_ERR_BAD_PARAM == parsec_taskpool_trim_chores(tp, "trsm:all"),
          "a class cannot be stripped of every incarnation it has");
    check(4 == types_of(tp, "trsm", types),
          "the class that could not comply keeps all of them");

    /* One class that cannot comply must not stop the others from being
     * looked at, nor turn their outcome into a silent success. */
    check(PARSEC_ERR_BAD_PARAM == parsec_taskpool_trim_chores(tp, "*:cpu,cuda,hip,level_zero"),
          "the refusal is reported even when other classes could comply");
    fixture_free(tp);
}

static void test_bad_specifications(void)
{
    static const char *bad[] = {
        "trsm",             /* no colon, so no device type at all */
        ":cuda",            /* no task class */
        "trsm:",            /* no device type */
        "trsm:banana",      /* a device type that does not exist */
        "trsm:cuda;potrf",  /* the second entry is the malformed one */
        NULL
    };

    for( int i = 0; NULL != bad[i]; i++ ) {
        parsec_taskpool_t *tp = fixture_new();
        uint8_t types[NB_INITIAL_TYPES];

        check(PARSEC_ERR_BAD_PARAM == parsec_taskpool_trim_chores(tp, bad[i]),
              "a specification that does not parse is refused");
        check(4 == types_of(tp, "trsm", types),
              "a specification that does not parse leaves the taskpool alone");
        fixture_free(tp);
    }
    check(PARSEC_ERR_BAD_PARAM == parsec_taskpool_trim_chores(NULL, "trsm:cuda"),
          "a taskpool that is not there is refused");
}

static void test_device_type_names(void)
{
    check(PARSEC_DEV_CUDA == parsec_device_type_from_name("cuda", 4), "cuda");
    check(PARSEC_DEV_CPU == parsec_device_type_from_name("CPU", 3), "CPU, whatever the case");
    check(PARSEC_DEV_LEVEL_ZERO == parsec_device_type_from_name("ze", 2), "ze, for level_zero");
    check(PARSEC_DEV_GPU_MASK == parsec_device_type_from_name("gpu", 3), "gpu, for all of them");
    check(PARSEC_DEV_NONE == parsec_device_type_from_name("cud", 3), "a name cut short is not cuda");
    check(PARSEC_DEV_NONE == parsec_device_type_from_name("cudax", 5), "nor is a longer one");
}

int main(int argc, char *argv[])
{
    (void)argc; (void)argv;

    test_device_type_names();
    test_one_class_one_type();
    test_several_types_and_classes();
    test_groups_and_wildcard();
    test_case_is_ignored();
    test_unknown_class_is_not_an_error();
    test_emptying_a_class_is_refused();
    test_bad_specifications();

    if( 0 != failures ) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("All checks passed\n");
    return 0;
}
