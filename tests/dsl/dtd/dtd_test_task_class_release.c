/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * A DTD task class gets a task_class_id, and a slot in the taskpool's
 * task_classes_array, only when its first chore is added. Releasing a class
 * that never got one (no chore added, or the first add_chore failed) must
 * leave task_classes_array alone, and the taskpool must stay usable.
 */

#include "parsec.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "tests/tests_runtime.h"

#include <stdint.h>

static int32_t observed_sum = 0;

static parsec_hook_return_t
add_value(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int value;

    parsec_dtd_unpack_args(this_task, &value);
    (void)parsec_atomic_fetch_add_int32(&observed_sum, value);

    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

static int
release_unregistered(parsec_taskpool_t *tp, parsec_task_class_t *tc)
{
    uint32_t nb_task_classes = tp->nb_task_classes;

    if( UINT8_MAX != tc->task_class_id ) {
        parsec_warning("Task class '%s' was registered with id %d before any chore was added\n",
                       tc->name, (int)tc->task_class_id);
        return 1;
    }
    parsec_dtd_task_class_release(tp, tc);
    if( nb_task_classes != tp->nb_task_classes ) {
        parsec_warning("Releasing an unregistered task class changed nb_task_classes from %u to %u\n",
                       nb_task_classes, tp->nb_task_classes);
        return 1;
    }
    return 0;
}

int
main(int argc, char **argv)
{
    parsec_context_t *parsec;
    parsec_taskpool_t *dtd_tp;
    parsec_task_class_t *tc;
    parsec_dtd_task_class_t *dtd_tc;
    parsec_dtd_param_t params[] = {
        { .op = PARSEC_VALUE, .size = sizeof(int), .profile_info = NULL },
        { .op = PARSEC_INOUT, .size = PASSED_BY_REF, .profile_info = NULL },
    };
    int rc, ntasks = 8, expected = 0, ret = 0;
    uint8_t id;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    dtd_tp = parsec_dtd_taskpool_new();

    /* Before the taskpool has a context: no task mempools are built yet. */
    tc = parsec_dtd_create_task_class(dtd_tp, "no_chore",
                                      sizeof(int), PARSEC_VALUE,
                                      PASSED_BY_REF, PARSEC_INOUT,
                                      PARSEC_DTD_ARG_END);
    ret |= release_unregistered(dtd_tp, tc);

    dtd_tc = parsec_dtd_create_task_classv("no_chore_v", 2, params);
    ret |= release_unregistered(dtd_tp, &dtd_tc->super);

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    /* With a context the task mempools are built at creation. */
    tc = parsec_dtd_create_task_class(dtd_tp, "no_chore_with_context",
                                      sizeof(int), PARSEC_VALUE,
                                      PASSED_BY_REF, PARSEC_INOUT,
                                      PARSEC_DTD_ARG_END);
    ret |= release_unregistered(dtd_tp, tc);

    /* A first add_chore that fails leaves the class unregistered too. */
    tc = parsec_dtd_create_task_class(dtd_tp, "failed_chore",
                                      sizeof(int), PARSEC_VALUE,
                                      PARSEC_DTD_ARG_END);
    rc = parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_NONE, add_value);
    if( PARSEC_SUCCESS == rc ) {
        parsec_warning("add_chore accepted PARSEC_DEV_NONE\n");
        ret = 1;
    }
    ret |= release_unregistered(dtd_tp, tc);

    /* The taskpool still registers and runs a class that does get a chore. */
    tc = parsec_dtd_create_task_class(dtd_tp, "with_chore",
                                      sizeof(int), PARSEC_VALUE,
                                      PARSEC_DTD_ARG_END);
    rc = parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_CPU, add_value);
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_task_class_add_chore");
    id = tc->task_class_id;
    if( (UINT8_MAX == id) || (tc != dtd_tp->task_classes_array[id]) ) {
        parsec_warning("Task class 'with_chore' is not registered after add_chore (id %d)\n", (int)id);
        ret = 1;
    }

    for( int i = 0; i < ntasks; i++ ) {
        int value = i + 1;
        expected += value;
        parsec_dtd_insert_task_with_task_class(dtd_tp, tc, 0, PARSEC_DEV_CPU,
                                               PARSEC_DTD_EMPTY_FLAG, &value,
                                               PARSEC_DTD_ARG_END);
    }

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( expected != observed_sum ) {
        parsec_warning("Observed %d, expected %d\n", observed_sum, expected);
        ret = 1;
    }

    parsec_dtd_task_class_release(dtd_tp, tc);
    if( (UINT8_MAX != id) && (NULL != dtd_tp->task_classes_array[id]) ) {
        parsec_warning("Releasing task class 'with_chore' left its slot %d set\n", (int)id);
        ret = 1;
    }

    parsec_taskpool_free(dtd_tp);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return ret;
}
