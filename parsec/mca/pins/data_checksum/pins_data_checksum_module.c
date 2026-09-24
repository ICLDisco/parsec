/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/**
 * Fingerprints every tile a task reads but does not write, once before the
 * body runs and once after, and reports any that differ. Nothing is allowed
 * to modify a tile while a task holds it for reading, so a mismatch names a
 * writer that the runtime does not know about: a message delivered into a
 * buffer that is still live, a copy handed to two tasks at once, a device
 * staging area written back over a tile in use.
 *
 * The report identifies the task and the flow that observed the damage,
 * which is the last moment the data was known good, rather than leaving the
 * corruption to surface thousands of tasks later as a wrong answer.
 *
 * Deliberately limited to what can be checked without guessing:
 *
 * - Only host copies. A device copy lives in memory this module cannot read.
 * - Only contiguous datatypes. A strided type has gaps that belong to nobody
 *   and would change for legitimate reasons.
 * - Only within one task. Comparing across tasks would need a stable identity
 *   for a copy, and there is none: an arena chunk that is recycled reappears
 *   at the same address with its version reset, so the same key would name
 *   two unrelated tiles.
 *
 * --mca pins_data_checksum_trace 2 extends the window past the body to the
 * point the task completes, which is where the runtime prepares its output
 * and releases the copies, and where a device task does its real work.
 *
 * Everything here is compiled away unless PARSEC_DEBUG_PARANOID is set, and
 * is inert until the module is selected with --mca mca_pins data_checksum.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/mca/pins/data_checksum/pins_data_checksum.h"
#include "parsec/execution_stream.h"
#include "parsec/data_internal.h"
#include "parsec/parsec_internal.h"
#include "parsec/datatype.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(PARSEC_DEBUG_PARANOID)

#define TASK_STR_LEN 256

/**
 * Per-thread state. The two callback slots must be distinct objects because
 * registering one stores the rest of the chain into the slot it is given,
 * but both events work on the same task, so they share one allocation.
 */
typedef struct {
    parsec_pins_next_callback_t begin_cb;
    parsec_pins_next_callback_t end_cb;
    parsec_pins_next_callback_t complete_cb;
    const parsec_task_t        *task;
    uint64_t                    checksum[MAX_PARAM_COUNT];
    int                         verify[MAX_PARAM_COUNT];
} checksum_thread_data_t;

#define BEGIN_SELF(cb) \
    ((checksum_thread_data_t*)((char*)(cb) - offsetof(checksum_thread_data_t, begin_cb)))
#define END_SELF(cb) \
    ((checksum_thread_data_t*)((char*)(cb) - offsetof(checksum_thread_data_t, end_cb)))
#define COMPLETE_SELF(cb) \
    ((checksum_thread_data_t*)((char*)(cb) - offsetof(checksum_thread_data_t, complete_cb)))

/**
 * A run that reports nothing is only good news if something was actually
 * looked at. Every copy this module declines to fingerprint is counted
 * under the reason it was declined, and the totals are printed at the end,
 * so "silent" can be told apart from "checked nothing".
 */
static volatile int64_t nb_mismatches    = 0;
static volatile int64_t nb_tasks         = 0;
static volatile int64_t nb_checked       = 0;
static volatile int64_t nb_skip_no_copy  = 0;
static volatile int64_t nb_skip_device   = 0;
static volatile int64_t nb_skip_no_dtt   = 0;
static volatile int64_t nb_skip_strided  = 0;
static volatile int64_t nb_flows_written = 0;
static volatile int64_t nb_skip_late     = 0;

/**
 * FNV-1a. Not a strong hash, but the failures we are after rewrite whole
 * doubles, and this costs a few nanoseconds per tile.
 */
static uint64_t data_checksum_fnv1a(const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t*)buf;
    uint64_t h = 14695981039346656037ULL;
    size_t i;
    for( i = 0; i < len; i++ ) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/** Fingerprint a host copy. Returns 0 when the copy cannot be fingerprinted. */
static int data_copy_checksum(const parsec_data_copy_t *copy, uint64_t *checksum)
{
    int size;
    ptrdiff_t lb, extent;

    if( NULL == copy || NULL == copy->device_private ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_no_copy);
        return 0;
    }
    if( 0 != copy->device_index ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_device);
        return 0;
    }
    if( PARSEC_DATATYPE_NULL == copy->dtt ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_no_dtt);
        return 0;
    }
    if( 0 != parsec_type_size(copy->dtt, &size) || size <= 0 ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_no_dtt);
        return 0;
    }
    if( 0 != parsec_type_extent(copy->dtt, &lb, &extent) || extent != (ptrdiff_t)size ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_strided);
        return 0;
    }

    *checksum = data_checksum_fnv1a(copy->device_private, (size_t)size);
    parsec_atomic_fetch_inc_int64(&nb_checked);
    return 1;
}

static const parsec_flow_t *task_flow(const parsec_task_t *task, int i)
{
    /* Both arrays are indexed by flow; an output-only flow is absent from in[]. */
    return NULL != task->task_class->in[i] ? task->task_class->in[i]
                                           : task->task_class->out[i];
}

/**
 * Name the tile a copy belongs to, in the coordinates the algorithm uses.
 * Those mean the same thing in every run, which is what lets two runs of a
 * deterministic program be compared. key_base holds the name the
 * application gave the collection but is only filled in when profiling is
 * compiled in, so say so rather than letting several matrices share a name.
 */
static const char *tile_name(const parsec_data_copy_t *copy, char *str, size_t len)
{
    parsec_data_t *data = (NULL == copy) ? NULL : copy->original;
    char coords[64];
    const char *owner;

    if( NULL == data || NULL == data->dc ) {
        snprintf(str, len, "anonymous");
        return str;
    }
    owner = (NULL != data->dc->key_base) ? data->dc->key_base : "unnamed";
    if( NULL != data->dc->key_to_string &&
        0 < data->dc->key_to_string(data->dc, data->key, coords, sizeof(coords)) )
        snprintf(str, len, "%s%s", owner, coords);   /* already parenthesised */
    else
        snprintf(str, len, "%s#%" PRIu64, owner, (uint64_t)data->key);
    return str;
}

/**
 * Report what a task read or wrote. A transfer log only shows data that
 * crossed a rank boundary; a tile that is updated in place by local tasks
 * and ends up wrong leaves no trace there at all. Fingerprinting both ends
 * of every task covers those, and since the program is deterministic the
 * same task reads the same bytes in every run, so the first task whose
 * input differs between a good run and a bad one is where to look.
 */
/**
 * Just the class and its parameters, as the transfer log prints them.
 * parsec_task_snprintf would do, but it appends the priority and the
 * dependency keys, which say nothing here and do not repeat between runs.
 */
static const char *task_name(const parsec_task_t *task, char *str, size_t len)
{
    const parsec_task_class_t *tc = task->task_class;
    size_t index = snprintf(str, len, "%s(", tc->name);
    unsigned int ip;

    for( ip = 0; ip < tc->nb_parameters && index < len; ip++ )
        index += snprintf(str + index, len - index, "%s%d", (0 == ip) ? "" : ", ",
                          task->locals[tc->params[ip]->context_index].value);
    if( index < len ) snprintf(str + index, len - index, ")");
    return str;
}

static void trace_flow(const parsec_task_t *task, const char *direction,
                       int i, const parsec_data_copy_t *copy)
{
    char tile[64], name[MAX_TASK_STRLEN];
    uint64_t checksum;

    /* A copy nothing has written yet holds whatever the allocator left
     * there, which differs between runs for no interesting reason. The
     * tasks that fill a matrix in read their tile before writing it, and
     * reporting those would bury the real differences. */
    if( NULL != copy && 0 == copy->version ) return;

    if( !data_copy_checksum(copy, &checksum) ) return;
    parsec_inform("TILE %s tile=%s flow=%d hash=%016" PRIx64 " tp=%u task=%s",
                  direction, tile_name(copy, tile, sizeof(tile)), i, checksum,
                  task->taskpool->taskpool_id, task_name(task, name, MAX_TASK_STRLEN));
}

static void data_checksum_exec_begin(parsec_execution_stream_t *es,
                                     parsec_task_t *task,
                                     parsec_pins_next_callback_t *cb_data)
{
    checksum_thread_data_t *self = BEGIN_SELF(cb_data);
    int i;

    self->task = task;
    parsec_atomic_fetch_inc_int64(&nb_tasks);
    for( i = 0; i < task->task_class->nb_flows; i++ ) {
        const parsec_flow_t *flow = task_flow(task, i);

        self->verify[i] = 0;
        if( NULL == flow ) continue;
        if( PARSEC_FLOW_ACCESS_READ != (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags) ) {
            /* the body may legitimately write anything else */
            if( PARSEC_FLOW_ACCESS_NONE != (PARSEC_FLOW_ACCESS_MASK & flow->flow_flags) )
                parsec_atomic_fetch_inc_int64(&nb_flows_written);
            continue;
        }

        self->verify[i] = data_copy_checksum(task->data[i].data_in, &self->checksum[i]);
    }

    if( parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_TILES )
        for( i = 0; i < task->task_class->nb_flows; i++ )
            trace_flow(task, "IN ", i, task->data[i].data_in);
    (void)es;
}

/**
 * Compare every read-only flow against what it held last time we looked and
 * report the ones that moved. The stored value is refreshed as we go, so a
 * flow reported once is not reported again: each window is judged on its
 * own and the report names the one the change happened in.
 */
static void verify_read_only(const parsec_task_t *task,
                             checksum_thread_data_t *self,
                             const char *window)
{
    int i;

    for( i = 0; i < task->task_class->nb_flows; i++ ) {
        const parsec_data_copy_t *copy;
        uint64_t checksum;

        if( !self->verify[i] ) continue;

        copy = task->data[i].data_in;
        if( !data_copy_checksum(copy, &checksum) ) continue;
        if( checksum == self->checksum[i] ) continue;

        char str[TASK_STR_LEN];
        parsec_task_snprintf(str, TASK_STR_LEN, task);
        parsec_atomic_fetch_inc_int64(&nb_mismatches);
        parsec_warning("PINS DATA CHECKSUM: %s read-only flow %s (%d): copy %p of data %p "
                       "was modified %s (before %016"PRIx64", after %016"PRIx64")",
                       str, task_flow(task, i)->name, i,
                       (const void*)copy, (const void*)copy->original,
                       window, self->checksum[i], checksum);
        self->checksum[i] = checksum;
    }
}

static void data_checksum_exec_end(parsec_execution_stream_t *es,
                                   parsec_task_t *task,
                                   parsec_pins_next_callback_t *cb_data)
{
    checksum_thread_data_t *self = END_SELF(cb_data);
    int i;

    if( parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_TILES )
        for( i = 0; i < task->task_class->nb_flows; i++ )
            trace_flow(task, "OUT", i, task->data[i].data_out);

    /* Only the body we fingerprinted can be judged. A task that arrives here
     * by another route leaves the scratch describing a different one. */
    if( self->task != task ) return;

    verify_read_only(task, self, "while the body was running");

    /* Keep the scratch when completion is also checked: that runs later and
     * needs the values this pass just refreshed. */
    if( !(parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_COMPLETION) )
        self->task = NULL;
    (void)es;
}

/**
 * The same flows again once the task is done, which catches a writer that
 * strikes after the body returned but while the task still holds the data.
 * For a task that ran on a device that is where the asynchronous execution
 * and the stage-out happen, and a staging buffer written back over a tile
 * still in use is one of the failures this module exists to name.
 *
 * COMPLETE_EXEC_BEGIN rather than COMPLETE_EXEC_END: the latter fires after
 * release_task, by which point the copies may have been handed back and
 * reading them would be a use-after-free.
 */
static void data_checksum_complete_begin(parsec_execution_stream_t *es,
                                         parsec_task_t *task,
                                         parsec_pins_next_callback_t *cb_data)
{
    checksum_thread_data_t *self = COMPLETE_SELF(cb_data);

    /* The scratch is per thread, so a task that completes on another thread,
     * or after this one has moved on, has nothing to compare against. That
     * is the common case for a device task; count it rather than let the
     * absence of a report pass for a clean result. */
    if( self->task != task ) {
        parsec_atomic_fetch_inc_int64(&nb_skip_late);
        return;
    }
    self->task = NULL;

    verify_read_only(task, self, "after the body returned, before the task completed");
    (void)es;
}

static void pins_fini_data_checksum(parsec_context_t *master)
{
    parsec_inform("PINS DATA CHECKSUM: %"PRId64" mismatches over %"PRId64" read-only tiles "
                  "fingerprinted in %"PRId64" tasks\n"
                  "\tnot fingerprinted: %"PRId64" writable flows (by design), "
                  "%"PRId64" without a host copy, %"PRId64" on a device, "
                  "%"PRId64" without a datatype, %"PRId64" strided",
                  nb_mismatches, nb_checked, nb_tasks,
                  nb_flows_written, nb_skip_no_copy, nb_skip_device,
                  nb_skip_no_dtt, nb_skip_strided);
    if( parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_COMPLETION )
        parsec_inform("PINS DATA CHECKSUM: %"PRId64" tasks completed away from the thread "
                      "that ran them and were not re-checked at completion",
                      nb_skip_late);
    (void)master;
}

static void pins_thread_init_data_checksum(parsec_execution_stream_t *es)
{
    checksum_thread_data_t *self =
        (checksum_thread_data_t*)calloc(1, sizeof(checksum_thread_data_t));
    PARSEC_PINS_REGISTER(es, EXEC_BEGIN, data_checksum_exec_begin, &self->begin_cb);
    PARSEC_PINS_REGISTER(es, EXEC_END, data_checksum_exec_end, &self->end_cb);
    if( parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_COMPLETION )
        PARSEC_PINS_REGISTER(es, COMPLETE_EXEC_BEGIN, data_checksum_complete_begin,
                             &self->complete_cb);
}

static void pins_thread_fini_data_checksum(parsec_execution_stream_t *es)
{
    parsec_pins_next_callback_t *begin_cb, *end_cb, *complete_cb;
    if( parsec_pins_data_checksum_trace & PARSEC_PINS_DATA_CHECKSUM_COMPLETION )
        PARSEC_PINS_UNREGISTER(es, COMPLETE_EXEC_BEGIN, data_checksum_complete_begin,
                               &complete_cb);
    PARSEC_PINS_UNREGISTER(es, EXEC_BEGIN, data_checksum_exec_begin, &begin_cb);
    PARSEC_PINS_UNREGISTER(es, EXEC_END, data_checksum_exec_end, &end_cb);
    /* All three slots are members of the one allocation made at thread init. */
    free(BEGIN_SELF(begin_cb));
}

const parsec_pins_module_t parsec_pins_data_checksum_module = {
    &parsec_pins_data_checksum_component,
    {
        NULL,
        pins_fini_data_checksum,
        NULL,
        NULL,
        pins_thread_init_data_checksum,
        pins_thread_fini_data_checksum
    },
    { NULL }
};

#else /* !defined(PARSEC_DEBUG_PARANOID) */

const parsec_pins_module_t parsec_pins_data_checksum_module = {
    &parsec_pins_data_checksum_component,
    { NULL, NULL, NULL, NULL, NULL, NULL },
    { NULL }
};

#endif /* defined(PARSEC_DEBUG_PARANOID) */
