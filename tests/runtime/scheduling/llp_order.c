/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * The llp scheduler keeps the ready tasks of each execution stream in a
 * LIFO sorted by priority. A ring of new tasks scheduled at distance d must
 * land after at least d tasks and after all the tasks of higher priority,
 * before the tasks of the same priority already queued, and in the order of
 * the ring. A task that returned PARSEC_HOOK_RETURN_AGAIN is rescheduled
 * with distance + 1, so it must not come back first.
 *
 * This test drives the llp module of the only execution stream directly:
 * it schedules rings of tasks of known priorities at known distances, then
 * selects every task back and checks the order.
 */
#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STEPS 12
#define MAX_TASKS 64

typedef struct {
    const char *labels;     /* one task per character, in the order of the ring */
    int priorities[8];
    int distance;
} sched_step_t;

typedef struct {
    const char *what;
    sched_step_t steps[MAX_STEPS];  /* ends with a NULL labels */
    const char *expected;           /* the order in which the tasks are selected */
} sched_case_t;

static const sched_case_t cases[] = {
    /* The first step fills the empty queue, the second one is the test. */
    { "lower task goes below the higher ones",
      { {"abc", {10, 8, 5}, 0}, {"X", {7}, 0}, {NULL} }, "abXc" },
    { "ring interleaves with the queue",
      { {"abc", {10, 8, 5}, 0}, {"XY", {9, 6}, 0}, {NULL} }, "aXbYc" },
    { "every task of a ring finds its place",
      { {"abcd", {9, 7, 5, 3}, 0}, {"WXYZ", {8, 6, 4, 2}, 0}, {NULL} }, "aWbXcYdZ" },
    { "new tasks go before old ones of the same priority, in ring order",
      { {"abc", {10, 8, 5}, 0}, {"XY", {8, 8}, 0}, {NULL} }, "aXYbc" },
    { "higher tasks go first",
      { {"abc", {10, 8, 5}, 0}, {"XY", {12, 11}, 0}, {NULL} }, "XYabc" },
    { "distance 2 skips two higher tasks",
      { {"abc", {10, 8, 5}, 0}, {"X", {7}, 2}, {NULL} }, "abXc" },
    { "distance 2 holds back a higher task",
      { {"abc", {10, 8, 5}, 0}, {"X", {20}, 2}, {NULL} }, "abXc" },
    { "distance past the end of the queue",
      { {"ab", {10, 8}, 0}, {"XY", {20, 1}, 5}, {NULL} }, "abXY" },
    { "rescheduled task (distance 1) lets an equal one run first",
      { {"ab", {3, 3}, 0}, {"X", {3}, 1}, {NULL} }, "aXb" },
    { "rescheduled task demoted to the lowest priority goes last",
      { {"ab", {0, 0}, 0}, {"X", {-1}, 1}, {NULL} }, "abX" },
    { "tasks scheduled one by one come out by priority, newest first",
      { {"a", {3}, 0}, {"b", {7}, 0}, {"c", {1}, 0}, {"d", {7}, 0}, {"e", {5}, 0},
        {"f", {0}, 0}, {"g", {9}, 0}, {"h", {3}, 0}, {"i", {5}, 0}, {"j", {2}, 0},
        {NULL} }, "gdbiehajcf" },
};

static int run_case(parsec_execution_stream_t *es, const sched_case_t *tc)
{
    parsec_task_t *tasks[MAX_TASKS], *task;
    char labels[MAX_TASKS], got[MAX_TASKS + 2];
    int nb_tasks = 0, nb_got = 0, i;
    int32_t distance;

    for( const sched_step_t *step = tc->steps; NULL != step->labels; step++ ) {
        parsec_task_t *ring = NULL;
        for( i = 0; '\0' != step->labels[i]; i++ ) {
            task = PARSEC_OBJ_NEW(parsec_task_t);
            task->priority = step->priorities[i];
            PARSEC_LIST_ITEM_SINGLETON(task);
            if( NULL == ring ) {
                ring = task;
            } else {
                parsec_list_item_ring_push(&ring->super, &task->super);
            }
            tasks[nb_tasks] = task;
            labels[nb_tasks] = step->labels[i];
            nb_tasks++;
        }
        parsec_current_scheduler->module.schedule(es, ring, step->distance);
    }

    /* select everything back, and a bit more to catch a corrupted queue */
    while( nb_got <= nb_tasks &&
           NULL != (task = parsec_current_scheduler->module.select(es, &distance)) ) {
        got[nb_got] = '?';
        for( i = 0; i < nb_tasks; i++ ) {
            if( tasks[i] == task ) {
                got[nb_got] = labels[i];
                break;
            }
        }
        nb_got++;
    }
    got[nb_got] = '\0';

    for( i = 0; i < nb_tasks; i++ ) {
        PARSEC_OBJ_RELEASE(tasks[i]);
    }

    if( 0 != strcmp(got, tc->expected) ) {
        fprintf(stderr, "FAIL %s: selected \"%s\", expected \"%s\"\n",
                tc->what, got, tc->expected);
        return 1;
    }
    printf("ok   %s: \"%s\"\n", tc->what, got);
    return 0;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    const char *sched_name;
    int rc, failed = 0;

    /* this test is about llp, whatever the environment asks for */
    setenv("PARSEC_MCA_mca_sched", "llp", 1);

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    sched_name = parsec_current_scheduler->component->base_version.mca_component_name;
    if( 0 != strcmp(sched_name, "llp") ) {
        fprintf(stderr, "FAIL the selected scheduler is %s, not llp\n", sched_name);
        failed = 1;
    } else {
        for( size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++ ) {
            failed += run_case(parsec_my_execution_stream(), &cases[c]);
        }
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0 == failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
