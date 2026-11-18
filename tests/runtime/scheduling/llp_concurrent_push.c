/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * To merge a ring of new tasks into the sorted LIFO of an execution stream,
 * the llp scheduler takes the whole LIFO out, merges, and puts the result
 * back. Other threads may push into the LIFO of stream 0 in the meantime
 * (the communication thread schedules there): the tasks they pushed are
 * taken out in turn and merged back.
 *
 * This test makes that happen: another thread schedules high priority tasks
 * on stream 0 while stream 0 merges low priority tasks behind a long queue.
 * Then stream 0 selects every task back and checks that each one comes out
 * exactly once, in priority order. In paranoid builds the LIFO also checks
 * that every task it hands out had been attached to it.
 *
 * The race cannot be forced, only made likely: the test reports how many
 * pushes happened while stream 0 was scheduling, and passes without proving
 * anything if none did (e.g., on a single core).
 */
#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/sys/atomic.h"
#include "tests/tests_runtime.h"

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_ROUNDS   10
#define NB_QUEUED   2000    /* queued before the race: each merge walks them */
#define NB_PUSHED  20000    /* scheduled on stream 0 by the other thread */
#define NB_MERGED   2000    /* at most, scheduled by stream 0 itself */
#define NB_TASKS   (NB_QUEUED + NB_PUSHED + NB_MERGED)

/* pushed tasks go in front, merged tasks behind the queued ones */
#define PRIO_PUSHED 1000
#define PRIO_QUEUED  100
#define PRIO_MERGED    0

typedef struct {
    parsec_execution_stream_t *es;
    parsec_task_t **tasks;      /* the NB_PUSHED tasks to schedule */
    volatile int32_t go;
    volatile int32_t done;
    volatile int32_t merging;   /* odd while stream 0 is in schedule() */
    int nb_overlaps;            /* pushes done within one schedule() of stream 0 */
} race_t;

static void *pusher(void *arg)
{
    race_t *race = (race_t*)arg;
    int32_t before;

    while( 0 == race->go ) parsec_atomic_rmb();

    for( int i = 0; i < NB_PUSHED; i++ ) {
        before = race->merging;
        parsec_atomic_rmb();
        parsec_current_scheduler->module.schedule(race->es, race->tasks[i], 0);
        parsec_atomic_rmb();
        if( (before & 1) && before == race->merging ) {
            race->nb_overlaps++;
        }
    }
    parsec_atomic_wmb();
    race->done = 1;
    return NULL;
}

static parsec_task_t *new_task(int id, int priority)
{
    parsec_task_t *task = PARSEC_OBJ_NEW(parsec_task_t);
    task->priority = priority;
    task->locals[0].value = id;
    PARSEC_LIST_ITEM_SINGLETON(task);
    return task;
}

static int run_round(parsec_execution_stream_t *es, int round, int *nb_overlaps)
{
    parsec_task_t **tasks = malloc(NB_TASKS * sizeof(parsec_task_t*));
    char *selected = calloc(NB_TASKS, 1);
    race_t race = { .es = es, .tasks = tasks + NB_QUEUED };
    parsec_task_t *task;
    pthread_t thread;
    int i, nb_merged = 0, nb_expected, nb_got = 0, last_priority = INT_MAX;
    int32_t distance;
    const char *error = NULL;

    for( i = 0; i < NB_TASKS; i++ ) {
        tasks[i] = new_task(i, i < NB_QUEUED ? PRIO_QUEUED :
                               i < NB_QUEUED + NB_PUSHED ? PRIO_PUSHED : PRIO_MERGED);
    }
    for( i = 1; i < NB_QUEUED; i++ ) {
        parsec_list_item_ring_push(&tasks[0]->super, &tasks[i]->super);
    }
    parsec_current_scheduler->module.schedule(es, tasks[0], 0);

    if( 0 != pthread_create(&thread, NULL, pusher, &race) ) {
        fprintf(stderr, "FAIL round %d: cannot create the pushing thread\n", round);
        return 1;
    }
    parsec_atomic_wmb();
    race.go = 1;
    /* each of these walks the whole LIFO, giving the other thread time to push */
    while( nb_merged < NB_MERGED && 0 == race.done ) {
        parsec_atomic_fetch_inc_int32(&race.merging);
        parsec_current_scheduler->module.schedule(es, tasks[NB_QUEUED + NB_PUSHED + nb_merged], 0);
        parsec_atomic_fetch_inc_int32(&race.merging);
        nb_merged++;
    }
    pthread_join(thread, NULL);
    nb_expected = NB_QUEUED + NB_PUSHED + nb_merged;

    /* select everything back, and a bit more to catch a corrupted queue */
    while( nb_got <= nb_expected &&
           NULL != (task = parsec_current_scheduler->module.select(es, &distance)) ) {
        int id = task->locals[0].value;
        if( id < 0 || id >= NB_TASKS || tasks[id] != task ) {
            error = "selected an unknown task";
        } else if( selected[id] ) {
            error = "selected a task twice";
        } else if( task->priority > last_priority ) {
            error = "selected a task out of priority order";
        }
        if( NULL != error ) break;
        selected[id] = 1;
        last_priority = task->priority;
        nb_got++;
    }
    if( NULL == error && nb_got != nb_expected ) {
        error = "lost tasks";
    }
    if( NULL != error ) {
        /* leak the tasks: some may still be queued */
        fprintf(stderr, "FAIL round %d: %s (%d of %d selected, %d pushes during a merge)\n",
                round, error, nb_got, nb_expected, race.nb_overlaps);
        return 1;
    }

    for( i = 0; i < NB_TASKS; i++ ) {
        PARSEC_OBJ_RELEASE(tasks[i]);
    }
    free(tasks);
    free(selected);
    printf("ok   round %d: %d tasks, %d merged by stream 0, %d of the %d pushes during one of these\n",
           round, nb_got, nb_merged, race.nb_overlaps, NB_PUSHED);
    *nb_overlaps += race.nb_overlaps;
    return 0;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    const char *sched_name;
    int rc, failed = 0, nb_overlaps = 0;

    /* this test is about llp, whatever the environment asks for */
    setenv("PARSEC_MCA_mca_sched", "llp", 1);
    /* the other thread inherits the binding of the main thread: keep both
     * free to run on different cores */
    setenv("PARSEC_MCA_runtime_bind_main_thread", "0", 1);

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    sched_name = parsec_current_scheduler->component->base_version.mca_component_name;
    if( 0 != strcmp(sched_name, "llp") ) {
        fprintf(stderr, "FAIL the selected scheduler is %s, not llp\n", sched_name);
        failed = 1;
    } else {
        for( int r = 0; r < NB_ROUNDS && 0 == failed; r++ ) {
            failed += run_round(parsec_my_execution_stream(), r, &nb_overlaps);
        }
        if( 0 == failed && 0 == nb_overlaps ) {
            printf("note no push happened while stream 0 was scheduling: "
                   "the concurrent path was not exercised\n");
        }
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0 == failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
