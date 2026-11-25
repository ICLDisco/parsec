/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/taskpool_merge.h"
#include "parsec/scheduling.h"
#include "parsec/execution_stream.h"
#include "parsec/data_internal.h"
#include "parsec/data_distribution.h"
#include "parsec/class/parsec_hash_table.h"
#include "parsec/interfaces/interface.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/utils/mca_param.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* What a block does to its tile */
#define MERGE_READ   ((uint8_t)0x1)
#define MERGE_WRITE  ((uint8_t)0x2)

/* The most memory references one task can have: one input and every output
 * dependency of each flow. */
#define MERGE_MAX_ACCESS (MAX_PARAM_COUNT * (1 + MAX_DEP_OUT_COUNT))

typedef struct parsec_merge_block_s parsec_merge_block_t;
typedef struct parsec_merge_tile_s  parsec_merge_tile_t;
typedef struct parsec_merge_taskpool_s parsec_merge_taskpool_t;
typedef struct parsec_merge_tracker_s  parsec_merge_tracker_t;

/**
 * The accesses of one taskpool to one tile. Every field but data, owner and
 * mode is protected by the lock of the hash table bucket of the tile.
 */
struct parsec_merge_block_s {
    parsec_merge_block_t    *next;     /**< next younger block on the tile */
    parsec_merge_tile_t     *tile;     /**< the tile, while the block is linked */
    parsec_merge_taskpool_t *owner;
    parsec_data_t           *data;
    int64_t                  pending;  /**< memory references and holds left */
    uint8_t                  mode;     /**< MERGE_READ or MERGE_WRITE */
    uint8_t                  active;   /**< the tasks of owner may touch the tile */
    uint8_t                  linked;   /**< still queued on the tile */
    parsec_task_t           *parked;   /**< ring of tasks waiting for active */
};

/** The queue of blocks of one tile, oldest first. */
struct parsec_merge_tile_s {
    parsec_hash_table_item_t ht_item;  /**< key: the parsec_data_t pointer */
    parsec_merge_block_t    *head;
    parsec_merge_block_t    *tail;
};

/** What auto merge keeps about one taskpool. */
struct parsec_merge_taskpool_s {
    parsec_merge_tracker_t *tracker;
    parsec_taskpool_t      *tp;
    uint64_t                seq;        /**< order in which it was given to the context */
    size_t                  nb_blocks;
    parsec_merge_block_t   *blocks;
};

/** What auto merge keeps about one context. */
struct parsec_merge_tracker_s {
    parsec_context_t   *context;
    parsec_hash_table_t tiles;          /**< parsec_merge_tile_t by parsec_data_t */
    pthread_mutex_t     register_lock;  /**< queues the blocks of one taskpool at a time */
    uint64_t            next_seq;
};

/* One memory reference of a task */
typedef struct {
    parsec_data_t        *data;
    const parsec_flow_t  *flow;
    uint8_t               is_output;
} merge_access_t;

static int parsec_merge_verbose = 0;

/******************************************************************************
 * Reading the task class description
 ******************************************************************************/

static inline int32_t
merge_eval(const parsec_expr_t *expr, const parsec_taskpool_t *tp,
           const parsec_assignment_t *locals)
{
    assert(PARSEC_EXPR_OP_INLINE == expr->op);
    return expr->u_expr.v_func.func.inline_func_int32(tp, locals);
}

static inline int
merge_dep_applies(const parsec_taskpool_t *tp, const parsec_dep_t *dep,
                  const parsec_assignment_t *locals)
{
    if( NULL == dep->cond ) return 1;
    return 0 != merge_eval(dep->cond, tp, locals);
}

/* Only the task classes of the JDF itself have memory references; the startup
 * task classes jdf2c appends after them, and the tasks devices create for
 * their own needs, are not ordered. */
static inline int
merge_tracks_class(const parsec_taskpool_t *tp, const parsec_task_class_t *tc)
{
    return (NULL != tc) &&
           (PARSEC_TASK_CLASS_TYPE_PTG == tc->task_class_type) &&
           (tc->task_class_id < tp->nb_task_classes) &&
           (tp->task_classes_array[tc->task_class_id] == tc);
}

/**
 * The memory references of the task of class tc at locals. The input of a
 * flow is its first dependency whose guard holds, the outputs are all those
 * whose guard holds. from_memory gets one bit per flow index whose input
 * comes from memory (a collection, NEW or NULL) rather than from a task.
 */
static int
merge_static_accesses(const parsec_taskpool_t *tp, const parsec_task_class_t *tc,
                      const parsec_assignment_t *locals,
                      merge_access_t *acc, uint32_t *from_memory)
{
    int n = 0;

    *from_memory = 0;
    for( int i = 0; (i < MAX_PARAM_COUNT) && (NULL != tc->in[i]); i++ ) {
        const parsec_flow_t *flow = tc->in[i];
        if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) continue;
        for( int j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
            const parsec_dep_t *dep = flow->dep_in[j];
            if( !merge_dep_applies(tp, dep, locals) ) continue;
            if( PARSEC_LOCAL_DATA_TASK_CLASS_ID == dep->task_class_id ) {
                *from_memory |= (1U << flow->flow_index);
                if( NULL != dep->direct_data ) {
                    parsec_data_t *data = dep->direct_data(tp, locals);
                    if( NULL != data ) {
                        acc[n].data = data; acc[n].flow = flow; acc[n].is_output = 0;
                        n++;
                    }
                }
            }
            break;
        }
    }
    for( int i = 0; (i < MAX_PARAM_COUNT) && (NULL != tc->out[i]); i++ ) {
        const parsec_flow_t *flow = tc->out[i];
        if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) continue;
        for( int j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != flow->dep_out[j]); j++ ) {
            const parsec_dep_t *dep = flow->dep_out[j];
            if( PARSEC_LOCAL_DATA_TASK_CLASS_ID != dep->task_class_id ) continue;
            if( NULL == dep->direct_data ) continue;
            if( !merge_dep_applies(tp, dep, locals) ) continue;
            parsec_data_t *data = dep->direct_data(tp, locals);
            if( NULL != data ) {
                assert(n < MERGE_MAX_ACCESS);
                acc[n].data = data; acc[n].flow = flow; acc[n].is_output = 1;
                n++;
            }
        }
    }
    return n;
}

/**
 * Can the data entering a task through flow end up modified in place? It can
 * if some task-to-task edge leaving flow, or leaving a read-only flow it is
 * forwarded to, reaches a flow that writes. Guards are not evaluated, so the
 * answer errs on the side of writing.
 */
static int
merge_flow_reaches_writer(const parsec_taskpool_t *tp, const parsec_flow_t *flow,
                          uint32_t *visiting)
{
    for( int j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != flow->dep_out[j]); j++ ) {
        const parsec_dep_t *dep = flow->dep_out[j];
        const parsec_flow_t *target = dep->flow;
        if( PARSEC_LOCAL_DATA_TASK_CLASS_ID == dep->task_class_id ) continue;
        if( NULL == target ) continue;
        uint8_t access = target->flow_flags & PARSEC_FLOW_ACCESS_MASK;
        if( PARSEC_FLOW_ACCESS_NONE == access ) continue;
        if( access & PARSEC_FLOW_ACCESS_WRITE ) return 1;
        if( dep->task_class_id >= tp->nb_task_classes ) return 1;
        uint32_t bit = 1U << target->flow_index;
        if( visiting[dep->task_class_id] & bit ) continue;
        visiting[dep->task_class_id] |= bit;
        if( merge_flow_reaches_writer(tp, target, visiting) ) return 1;
    }
    return 0;
}

/******************************************************************************
 * Counting the accesses of a taskpool
 ******************************************************************************/

typedef struct {
    parsec_data_t *data;
    int64_t        count;
    uint8_t        mode;
} merge_count_t;

/* An open addressing table of merge_count_t, private to one registration */
typedef struct {
    merge_count_t *slots;
    size_t         capacity;  /* a power of two */
    size_t         used;
} merge_counts_t;

static inline size_t merge_hash_ptr(const void *ptr)
{
    uint64_t x = (uint64_t)(uintptr_t)ptr;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (size_t)x;
}

static int merge_counts_grow(merge_counts_t *counts)
{
    size_t capacity = (0 == counts->capacity) ? 1024 : 2 * counts->capacity;
    merge_count_t *slots = (merge_count_t*)calloc(capacity, sizeof(merge_count_t));
    if( NULL == slots ) return PARSEC_ERR_OUT_OF_RESOURCE;
    for( size_t i = 0; i < counts->capacity; i++ ) {
        if( NULL == counts->slots[i].data ) continue;
        size_t h = merge_hash_ptr(counts->slots[i].data) & (capacity - 1);
        while( NULL != slots[h].data ) h = (h + 1) & (capacity - 1);
        slots[h] = counts->slots[i];
    }
    free(counts->slots);
    counts->slots = slots;
    counts->capacity = capacity;
    return PARSEC_SUCCESS;
}

static int merge_counts_add(merge_counts_t *counts, parsec_data_t *data, uint8_t mode)
{
    if( 2 * (counts->used + 1) > counts->capacity ) {
        int rc = merge_counts_grow(counts);
        if( PARSEC_SUCCESS != rc ) return rc;
    }
    size_t h = merge_hash_ptr(data) & (counts->capacity - 1);
    while( (NULL != counts->slots[h].data) && (data != counts->slots[h].data) )
        h = (h + 1) & (counts->capacity - 1);
    if( NULL == counts->slots[h].data ) {
        counts->slots[h].data = data;
        counts->used++;
    }
    counts->slots[h].count++;
    counts->slots[h].mode |= mode;
    return PARSEC_SUCCESS;
}

typedef struct {
    parsec_taskpool_t         *tp;
    const parsec_task_class_t *tc;
    parsec_task_t             *task;
    merge_counts_t            *counts;
    uint32_t                   writer_known;  /* flows of tc whose answer is cached */
    uint32_t                   writer;        /* flows of tc that reach a writer */
    uint32_t                  *visiting;
    int                        my_rank;
    size_t                     nb_tasks;
    int                        rc;
} merge_enum_t;

static uint8_t merge_input_mode(merge_enum_t *ctx, const parsec_flow_t *flow)
{
    uint32_t bit = 1U << flow->flow_index;

    if( flow->flow_flags & PARSEC_FLOW_ACCESS_WRITE ) return MERGE_WRITE;
    if( !(ctx->writer_known & bit) ) {
        memset(ctx->visiting, 0, ctx->tp->nb_task_classes * sizeof(uint32_t));
        if( merge_flow_reaches_writer(ctx->tp, flow, ctx->visiting) )
            ctx->writer |= bit;
        ctx->writer_known |= bit;
    }
    return (ctx->writer & bit) ? MERGE_WRITE : MERGE_READ;
}

static void merge_visit_task(merge_enum_t *ctx)
{
    const parsec_task_class_t *tc = ctx->tc;
    merge_access_t acc[MERGE_MAX_ACCESS];
    uint32_t from_memory;
    int n;

    if( NULL != tc->data_affinity ) {
        parsec_data_ref_t ref;
        tc->data_affinity(ctx->task, &ref);
        if( ref.dc->rank_of_key(ref.dc, ref.key) != (uint32_t)ctx->my_rank )
            return;  /* runs elsewhere */
    }
    ctx->nb_tasks++;
    n = merge_static_accesses(ctx->tp, tc, ctx->task->locals, acc, &from_memory);
    for( int i = 0; (i < n) && (PARSEC_SUCCESS == ctx->rc); i++ ) {
        uint8_t mode = acc[i].is_output ? MERGE_WRITE : merge_input_mode(ctx, acc[i].flow);
        ctx->rc = merge_counts_add(ctx->counts, acc[i].data, mode);
    }
}

/* Walk the execution space of ctx->tc, one local at a time */
static void merge_enum_locals(merge_enum_t *ctx, int li)
{
    const parsec_task_class_t *tc = ctx->tc;
    parsec_assignment_t *locals = ctx->task->locals;

    if( PARSEC_SUCCESS != ctx->rc ) return;
    if( li == tc->nb_locals ) {
        merge_visit_task(ctx);
        return;
    }
    const parsec_symbol_t *s = tc->locals[li];
    assert(s->context_index == li);
    int32_t min = merge_eval(s->min, ctx->tp, locals);
    int32_t max = merge_eval(s->max, ctx->tp, locals);
    for( int64_t v = min; v <= max; ) {
        locals[li].value = (int)v;
        merge_enum_locals(ctx, li + 1);
        int32_t inc = (NULL != s->expr_inc) ? merge_eval(s->expr_inc, ctx->tp, locals) : s->cst_inc;
        if( inc <= 0 ) break;  /* a definition rather than a range: one value */
        v += inc;
    }
}

/******************************************************************************
 * Queues of blocks
 ******************************************************************************/

static inline parsec_key_t merge_key(const parsec_data_t *data)
{
    return (parsec_key_t)(uintptr_t)data;
}

static inline parsec_task_t *merge_ring_append(parsec_task_t *ring, parsec_task_t *more)
{
    if( NULL == more ) return ring;
    if( NULL == ring ) return more;
    return (parsec_task_t*)parsec_list_item_ring_merge(&ring->super, &more->super);
}

/* With the bucket of tile locked: activate the blocks at the front of the
 * queue that may run, and return the tasks that waited for them. The oldest
 * block always runs; the ones behind it only if they and everything before
 * them only read. */
static parsec_task_t *merge_tile_activate(parsec_merge_tile_t *tile)
{
    parsec_task_t *ring = NULL;

    for( parsec_merge_block_t *b = tile->head; NULL != b; b = b->next ) {
        if( (b != tile->head) && (b->mode & MERGE_WRITE) ) break;
        if( !b->active ) {
            b->active = 1;
            ring = merge_ring_append(ring, b->parked);
            b->parked = NULL;
        }
        if( b->mode & MERGE_WRITE ) break;
    }
    return ring;
}

/* With the bucket of block->data locked: take a block out of its queue, and
 * return the tasks this lets go. Drops the tile once its queue is empty. */
static parsec_task_t *merge_block_retire(parsec_merge_tracker_t *tracker,
                                         parsec_merge_block_t *block)
{
    parsec_merge_tile_t *tile = block->tile;
    parsec_merge_block_t *prev = NULL;
    parsec_task_t *ring;

    assert(block->linked);
    assert(NULL == block->parked);  /* nothing waits on a block that ran */
    for( parsec_merge_block_t *b = tile->head; b != block; b = b->next ) {
        assert(NULL != b);
        prev = b;
    }
    if( NULL == prev ) tile->head = block->next;
    else prev->next = block->next;
    if( tile->tail == block ) tile->tail = prev;
    block->next = NULL;
    block->tile = NULL;
    block->linked = 0;

    ring = merge_tile_activate(tile);
    if( NULL == tile->head ) {
        parsec_hash_table_nolock_remove(&tracker->tiles, tile->ht_item.key);
        free(tile);
    }
    return ring;
}

/* With the bucket of data locked: the block of owner on data, or NULL */
static parsec_merge_block_t *merge_find_block(parsec_merge_tracker_t *tracker,
                                              parsec_merge_taskpool_t *owner,
                                              const parsec_data_t *data)
{
    parsec_merge_tile_t *tile = parsec_hash_table_nolock_find(&tracker->tiles, merge_key(data));
    if( NULL == tile ) return NULL;
    for( parsec_merge_block_t *b = tile->head; NULL != b; b = b->next ) {
        if( b->owner == owner ) return b;
    }
    return NULL;
}

/* Give tasks let go by a block back to the scheduler, from whichever thread
 * let them go: a worker, the communication thread or the main thread. */
static void merge_schedule(parsec_merge_tracker_t *tracker, parsec_task_t *ring)
{
    parsec_execution_stream_t *es = parsec_my_execution_stream();

    if( NULL == ring ) return;
    if( (NULL == es) || (NULL == es->scheduler_object) ||
        (es->virtual_process->parsec_context != tracker->context) ) {
        es = tracker->context->virtual_processes[0]->execution_streams[0];
    }
    __parsec_schedule(es, ring, 0);
}

/* Take count off the block of owner on data, if there is one */
static void merge_release(parsec_merge_taskpool_t *owner, parsec_data_t *data, int64_t count)
{
    parsec_merge_tracker_t *tracker = owner->tracker;
    parsec_key_t key = merge_key(data);
    parsec_task_t *ring = NULL;
    parsec_merge_block_t *block;

    parsec_hash_table_lock_bucket(&tracker->tiles, key);
    block = merge_find_block(tracker, owner, data);
    if( NULL != block ) {
        assert(block->active);
        assert(block->pending >= count);
        block->pending -= count;
        if( 0 == block->pending ) {
            ring = merge_block_retire(tracker, block);
        }
    }
    parsec_hash_table_unlock_bucket(&tracker->tiles, key);
    merge_schedule(tracker, ring);
}

/******************************************************************************
 * Public interface
 ******************************************************************************/

int parsec_context_set_auto_merge(parsec_context_t *context, int enable)
{
    if( (NULL == context) ||
        ((PARSEC_AUTO_MERGE_ON != enable) && (PARSEC_AUTO_MERGE_OFF != enable)) )
        return PARSEC_ERR_BAD_PARAM;
    if( (PARSEC_AUTO_MERGE_ON == enable) && (NULL == context->merge_tracker) )
        return PARSEC_ERR_NOT_SUPPORTED;
    context->auto_merge = enable;
    return PARSEC_SUCCESS;
}

int parsec_taskpool_set_auto_merge(parsec_taskpool_t *tp, int enable)
{
    if( (NULL == tp) || (NULL != tp->merge) ||
        ((PARSEC_AUTO_MERGE_ON != enable) && (PARSEC_AUTO_MERGE_OFF != enable) &&
         (PARSEC_AUTO_MERGE_DEFAULT != enable)) )
        return PARSEC_ERR_BAD_PARAM;
    tp->auto_merge = enable;
    return PARSEC_SUCCESS;
}

/******************************************************************************
 * Interface with the runtime
 ******************************************************************************/

int parsec_merge_context_init(parsec_context_t *context)
{
    parsec_merge_tracker_t *tracker;
    int auto_merge = 0;

    parsec_mca_param_reg_int_name("runtime", "auto_merge",
                                  "Order the PTG taskpools of a context as if their DAGs were merged in the "
                                  "order they are given to the context: per tile, a task waits until the "
                                  "earlier taskpools are done with the tiles it reads or writes from memory "
                                  "(0: off, 1: on). parsec_context_set_auto_merge() overrides it.",
                                  false, false, auto_merge, &auto_merge);
    parsec_mca_param_reg_int_name("runtime", "auto_merge_verbose",
                                  "Report what automatic taskpool merging counts for each taskpool",
                                  false, false, parsec_merge_verbose, &parsec_merge_verbose);
    context->auto_merge = (0 != auto_merge) ? PARSEC_AUTO_MERGE_ON : PARSEC_AUTO_MERGE_OFF;

    tracker = (parsec_merge_tracker_t*)calloc(1, sizeof(parsec_merge_tracker_t));
    if( NULL == tracker ) return PARSEC_ERR_OUT_OF_RESOURCE;
    tracker->context = context;
    parsec_hash_table_init(&tracker->tiles, offsetof(parsec_merge_tile_t, ht_item),
                           12, parsec_hash_table_generic_key_fn, NULL);
    pthread_mutex_init(&tracker->register_lock, NULL);
    tracker->next_seq = 0;
    context->merge_tracker = tracker;
    return PARSEC_SUCCESS;
}

static void merge_warn_leftover_tile(void *item, void *cb_data)
{
    parsec_merge_tile_t *tile = (parsec_merge_tile_t*)item;
    int *nb = (int*)cb_data;
    (*nb)++;
    (void)tile;
}

void parsec_merge_context_fini(parsec_context_t *context)
{
    parsec_merge_tracker_t *tracker = context->merge_tracker;
    int nb = 0;

    if( NULL == tracker ) return;
    parsec_hash_table_for_all(&tracker->tiles, merge_warn_leftover_tile, &nb);
    if( 0 != nb ) {
        parsec_warning("Automatic taskpool merging: %d tiles are still claimed by taskpools "
                       "that never terminated", nb);
    }
    parsec_hash_table_fini(&tracker->tiles);
    pthread_mutex_destroy(&tracker->register_lock);
    free(tracker);
    context->merge_tracker = NULL;
}

int parsec_merge_taskpool_register(parsec_context_t *context, parsec_taskpool_t *tp)
{
    parsec_merge_tracker_t *tracker = context->merge_tracker;
    merge_counts_t counts = { NULL, 0, 0 };
    parsec_merge_taskpool_t *rec;
    parsec_task_t task;
    uint32_t *visiting = NULL;
    size_t nb_tasks = 0, nb_waiting = 0;
    int want, rc = PARSEC_SUCCESS;

    want = (PARSEC_AUTO_MERGE_DEFAULT == tp->auto_merge) ? context->auto_merge : tp->auto_merge;
    if( PARSEC_AUTO_MERGE_OFF == want ) return PARSEC_SUCCESS;
    if( NULL == tracker ) return PARSEC_SUCCESS;
    assert(NULL == tp->merge);
    if( PARSEC_TASKPOOL_TYPE_PTG != tp->taskpool_type ) {
        /* Only a PTG taskpool can say up front which tiles it touches */
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                             "Automatic taskpool merging ignores taskpool %s (%u): not a PTG taskpool",
                             (NULL != tp->taskpool_name) ? tp->taskpool_name : "", tp->taskpool_id);
        return PARSEC_SUCCESS;
    }

    rec = (parsec_merge_taskpool_t*)calloc(1, sizeof(parsec_merge_taskpool_t));
    visiting = (uint32_t*)calloc((tp->nb_task_classes > 0 ? tp->nb_task_classes : 1), sizeof(uint32_t));
    if( (NULL == rec) || (NULL == visiting) ) {
        rc = PARSEC_ERR_OUT_OF_RESOURCE;
        goto fail;
    }
    rec->tracker = tracker;
    rec->tp = tp;

    /* Count the memory references of every local task */
    memset(&task, 0, sizeof(parsec_task_t));
    task.taskpool = tp;
    for( uint32_t i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t *tc = tp->task_classes_array[i];
        if( !merge_tracks_class(tp, tc) ) continue;
        merge_enum_t ctx = { .tp = tp, .tc = tc, .task = &task, .counts = &counts,
                             .writer_known = 0, .writer = 0, .visiting = visiting,
                             .my_rank = context->my_rank, .nb_tasks = 0, .rc = PARSEC_SUCCESS };
        task.task_class = tc;
        for( int li = 0; li < MAX_LOCAL_COUNT; li++ ) task.locals[li].value = 0;
        merge_enum_locals(&ctx, 0);
        if( PARSEC_SUCCESS != ctx.rc ) {
            rc = ctx.rc;
            goto fail;
        }
        nb_tasks += ctx.nb_tasks;
    }

    rec->blocks = (parsec_merge_block_t*)calloc((counts.used > 0 ? counts.used : 1), sizeof(parsec_merge_block_t));
    if( NULL == rec->blocks ) {
        rc = PARSEC_ERR_OUT_OF_RESOURCE;
        goto fail;
    }

    /* Queue one block per tile behind the taskpools that came before. All the
     * blocks of a taskpool go in under one lock, so that two taskpools given
     * to the context at the same time are in the same order on every tile. */
    pthread_mutex_lock(&tracker->register_lock);
    rec->seq = ++tracker->next_seq;
    for( size_t s = 0; s < counts.capacity; s++ ) {
        merge_count_t *c = &counts.slots[s];
        if( NULL == c->data ) continue;
        parsec_merge_block_t *block = &rec->blocks[rec->nb_blocks++];
        parsec_key_t key = merge_key(c->data);

        block->owner   = rec;
        block->data    = c->data;
        block->pending = c->count;
        block->mode    = (c->mode & MERGE_WRITE) ? MERGE_WRITE : MERGE_READ;
        block->parked  = NULL;

        parsec_hash_table_lock_bucket(&tracker->tiles, key);
        parsec_merge_tile_t *tile = parsec_hash_table_nolock_find(&tracker->tiles, key);
        if( NULL == tile ) {
            tile = (parsec_merge_tile_t*)calloc(1, sizeof(parsec_merge_tile_t));
            tile->ht_item.key = key;
            parsec_hash_table_nolock_insert(&tracker->tiles, &tile->ht_item);
        }
        block->tile = tile;
        block->linked = 1;
        if( NULL == tile->tail ) tile->head = block;
        else tile->tail->next = block;
        tile->tail = block;
        /* Nothing waits on a block yet, the tasks of tp are not created */
        (void)merge_tile_activate(tile);
        if( !block->active ) nb_waiting++;
        parsec_hash_table_unlock_bucket(&tracker->tiles, key);
    }
    tp->merge = rec;
    pthread_mutex_unlock(&tracker->register_lock);

    if( parsec_merge_verbose ) {
        parsec_inform("Automatic taskpool merging: taskpool %s (%u) is #%llu on rank %d: "
                      "%zu local tasks touch %zu tiles, %zu of them still claimed by earlier taskpools",
                      (NULL != tp->taskpool_name) ? tp->taskpool_name : "", tp->taskpool_id,
                      (unsigned long long)rec->seq, context->my_rank, nb_tasks, rec->nb_blocks, nb_waiting);
    }
    free(counts.slots);
    free(visiting);
    return PARSEC_SUCCESS;

  fail:
    parsec_warning("Automatic taskpool merging could not count the accesses of taskpool %s (%u): "
                   "it will not be ordered (error %d)",
                   (NULL != tp->taskpool_name) ? tp->taskpool_name : "", tp->taskpool_id, rc);
    free(counts.slots);
    free(visiting);
    if( NULL != rec ) {
        free(rec->blocks);
        free(rec);
    }
    return rc;
}

void parsec_merge_taskpool_terminated(parsec_taskpool_t *tp)
{
    parsec_merge_taskpool_t *rec = tp->merge;
    size_t leftover = 0;

    if( NULL == rec ) return;
    for( size_t i = 0; i < rec->nb_blocks; i++ ) {
        parsec_merge_block_t *block = &rec->blocks[i];
        parsec_key_t key = merge_key(block->data);
        parsec_task_t *ring = NULL;

        parsec_hash_table_lock_bucket(&rec->tracker->tiles, key);
        if( block->linked ) {
            /* Something was counted that never came back, typically a copy
             * reshaped on its way to a successor. Nothing of tp is running any
             * more, so the tile is free all the same. */
            leftover++;
            block->pending = 0;
            ring = merge_block_retire(rec->tracker, block);
        }
        parsec_hash_table_unlock_bucket(&rec->tracker->tiles, key);
        merge_schedule(rec->tracker, ring);
    }
    if( leftover && parsec_merge_verbose ) {
        parsec_inform("Automatic taskpool merging: taskpool %s (%u) released %zu tiles at termination",
                      (NULL != tp->taskpool_name) ? tp->taskpool_name : "", tp->taskpool_id, leftover);
    }
    tp->merge = NULL;
    free(rec->blocks);
    free(rec);
}

int parsec_merge_task_gate(parsec_execution_stream_t *es, parsec_task_t *task)
{
    parsec_taskpool_t *tp = task->taskpool;
    parsec_merge_taskpool_t *rec = tp->merge;
    const parsec_task_class_t *tc = task->task_class;
    merge_access_t acc[MERGE_MAX_ACCESS];
    uint32_t from_memory;
    int n;

    (void)es;
    if( !merge_tracks_class(tp, tc) ) return 1;
    n = merge_static_accesses(tp, tc, task->locals, acc, &from_memory);
    for( int i = 0; i < n; i++ ) {
        parsec_key_t key = merge_key(acc[i].data);
        parsec_merge_block_t *block;

        parsec_hash_table_lock_bucket(&rec->tracker->tiles, key);
        block = merge_find_block(rec->tracker, rec, acc[i].data);
        if( (NULL != block) && !block->active ) {
#if defined(PARSEC_DEBUG_NOISIER)
            /* Once parked, the task belongs to whoever lets it go */
            char tmp[MAX_TASK_STRLEN];
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                                 "Automatic taskpool merging parks %s until earlier taskpools are done with data %p",
                                 parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task), acc[i].data);
#endif
            PARSEC_LIST_ITEM_SINGLETON(task);
            if( NULL == block->parked ) block->parked = task;
            else parsec_list_item_ring_push(&block->parked->super, &task->super);
            parsec_hash_table_unlock_bucket(&rec->tracker->tiles, key);
            return 0;
        }
        parsec_hash_table_unlock_bucket(&rec->tracker->tiles, key);
    }
    return 1;
}

int parsec_merge_complete_execution(parsec_execution_stream_t *es, parsec_task_t *task)
{
    parsec_taskpool_t *tp = task->taskpool;
    parsec_merge_taskpool_t *rec = tp->merge;
    const parsec_task_class_t *tc = task->task_class;
    parsec_data_t *done[MERGE_MAX_ACCESS + MAX_PARAM_COUNT];
    int nb = 0, rc = 0;

    if( merge_tracks_class(tp, tc) ) {
        merge_access_t acc[MERGE_MAX_ACCESS];
        uint32_t from_memory;
        int n = merge_static_accesses(tp, tc, task->locals, acc, &from_memory);

        for( int i = 0; i < n; i++ ) done[nb++] = acc[i].data;
        /* A flow fed by another task may carry a copy of a tile: the
         * predecessor held the tile for us when it handed the copy over. The
         * copies are looked at now, the completion may release them. */
        for( int i = 0; (i < MAX_PARAM_COUNT) && (NULL != tc->in[i]); i++ ) {
            const parsec_flow_t *flow = tc->in[i];
            if( PARSEC_FLOW_ACCESS_NONE == (flow->flow_flags & PARSEC_FLOW_ACCESS_MASK) ) continue;
            if( from_memory & (1U << flow->flow_index) ) continue;
            parsec_data_copy_t *copy = task->data[flow->flow_index].data_in;
            if( (NULL == copy) || (NULL == copy->original) ) continue;
            done[nb++] = copy->original;
        }
    }

    if( NULL != tc->complete_execution )
        rc = tc->complete_execution(es, task);

    /* The successors were given their holds by the completion above; only now
     * may what this task held be released. */
    for( int i = 0; i < nb; i++ ) {
        merge_release(rec, done[i], 1);
    }
    return rc;
}

parsec_data_t *parsec_merge_hold(parsec_taskpool_t *tp, const parsec_data_copy_t *copy)
{
    parsec_merge_taskpool_t *rec = tp->merge;
    parsec_data_t *data;
    parsec_merge_block_t *block;
    parsec_key_t key;

    if( (NULL == rec) || (NULL == copy) || (NULL == (data = copy->original)) ) return NULL;
    key = merge_key(data);
    parsec_hash_table_lock_bucket(&rec->tracker->tiles, key);
    block = merge_find_block(rec->tracker, rec, data);
    if( NULL != block ) {
        assert(block->active);
        assert(block->pending > 0);  /* whoever hands the copy on holds it already */
        block->pending++;
    }
    parsec_hash_table_unlock_bucket(&rec->tracker->tiles, key);
    return (NULL != block) ? data : NULL;
}

void parsec_merge_unhold(parsec_taskpool_t *tp, parsec_data_t *data)
{
    if( (NULL == tp->merge) || (NULL == data) ) return;
    merge_release(tp->merge, data, 1);
}
