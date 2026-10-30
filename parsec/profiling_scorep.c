/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

/*
 * Profiling substrate mirroring the PaRSEC profiling events as Score-P
 * user regions.
 */

#include "parsec/parsec_config.h"

#if defined(PARSEC_PROF_TRACE_SCOREP)

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* Only the function declarations are needed: SCOREP_USER_ENABLE is not
 * defined, the user adapter macros are never used. */
#include <scorep/SCOREP_User_Functions.h>

#include "parsec/profiling.h"
#include "parsec/profiling_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/sys/tls.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/mca_param.h"

/* The user adapter symbols are resolved only if the application is linked
 * with Score-P (scorep --user). Otherwise they stay NULL and the substrate
 * deactivates itself. */
#pragma weak SCOREP_User_RegionInit
#pragma weak SCOREP_User_RegionEnter
#pragma weak SCOREP_User_RegionEnd
#pragma weak SCOREP_User_RegionSetGroup

#define PARSEC_SCOREP_GROUP_NAME "PaRSEC"

/* Region handles indexed by the dictionary key. The table is two-level and
 * chunks are never reallocated, so that tracing threads can read it while
 * new keys are registered. */
#define PARSEC_SCOREP_CHUNK_SIZE 256
#define PARSEC_SCOREP_MAX_CHUNKS 256

typedef struct parsec_scorep_chunk_s {
    SCOREP_User_RegionHandle handles[PARSEC_SCOREP_CHUNK_SIZE];
} parsec_scorep_chunk_t;

static parsec_scorep_chunk_t *parsec_scorep_chunks[PARSEC_SCOREP_MAX_CHUNKS];

/* Score-P definitions cannot be removed, so keep the region of each name
 * across dictionary flushes and reuse it when the name is registered again. */
typedef struct parsec_scorep_region_s {
    struct parsec_scorep_region_s *next;
    char                          *name;
    SCOREP_User_RegionHandle       handle;
} parsec_scorep_region_t;

static parsec_scorep_region_t *parsec_scorep_regions = NULL;
static pthread_mutex_t parsec_scorep_regions_lock = PTHREAD_MUTEX_INITIALIZER;

/* Per thread stack of the regions entered through Score-P. */
typedef struct parsec_scorep_frame_s {
    SCOREP_User_RegionHandle handle;
    int                      key;
    uint32_t                 taskpool_id;
    uint64_t                 event_id;
} parsec_scorep_frame_t;

typedef struct parsec_scorep_stack_s {
    int                    depth;
    int                    size;
    parsec_scorep_frame_t *frames;
} parsec_scorep_stack_t;

static PARSEC_TLS_DECLARE(parsec_scorep_tls_stack);

static int parsec_scorep_enabled = 0;
static int parsec_scorep_all_streams = 0;
static int parsec_scorep_mca_registered = 0;
static int parsec_scorep_active = 0;
static int32_t parsec_scorep_repair_warned = 0;

static int scorep_register_mca(void)
{
    if( !parsec_scorep_mca_registered ) {
        parsec_mca_param_reg_int_name("profile", "scorep",
                                      "Enable the Score-P profiling substrate and mirror PaRSEC profiling events as Score-P user regions"
                                      " (the application must be linked with Score-P, e.g., scorep --user)",
                                      false, false,
                                      parsec_scorep_enabled,
                                      &parsec_scorep_enabled);
        parsec_mca_param_reg_int_name("profile", "scorep_all_streams",
                                      "Also mirror into Score-P the events traced into a profiling stream that is not the default"
                                      " stream of the calling thread (e.g., GPU streams). Such events may not nest properly and are"
                                      " truncated to keep the Score-P regions balanced",
                                      false, false,
                                      parsec_scorep_all_streams,
                                      &parsec_scorep_all_streams);
        parsec_scorep_mca_registered = 1;
    }
    return parsec_scorep_enabled;
}

static int scorep_init(int process_id)
{
    (void)process_id;

    if( NULL == SCOREP_User_RegionInit ||
        NULL == SCOREP_User_RegionEnter ||
        NULL == SCOREP_User_RegionEnd ) {
        parsec_warning("PaRSEC Profiling -- Score-P: profile_scorep is set but the application is not linked with the"
                       " Score-P user adapter (scorep --user). Score-P profiling is disabled.");
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    PARSEC_TLS_KEY_CREATE(parsec_scorep_tls_stack);
    parsec_scorep_active = 1;
    return 0;
}

static void scorep_dictionary_flush(void)
{
    /* The region of each name is kept in parsec_scorep_regions, only forget
     * the mapping from keys to regions. */
    for(int i = 0; i < PARSEC_SCOREP_MAX_CHUNKS; i++) {
        if( NULL != parsec_scorep_chunks[i] ) {
            memset(parsec_scorep_chunks[i], 0, sizeof(parsec_scorep_chunk_t));
        }
    }
}

static SCOREP_User_RegionHandle parsec_scorep_region_lookup(const char *name)
{
    parsec_scorep_region_t *region;
    SCOREP_User_RegionHandle handle = SCOREP_USER_INVALID_REGION;

    pthread_mutex_lock(&parsec_scorep_regions_lock);
    for(region = parsec_scorep_regions; NULL != region; region = region->next) {
        if( 0 == strcmp(region->name, name) ) {
            handle = region->handle;
            goto unlock_and_return;
        }
    }
    region = (parsec_scorep_region_t*)malloc(sizeof(parsec_scorep_region_t));
    if( NULL == region ) {
        goto unlock_and_return;
    }
    region->name = strdup(name);
    if( NULL == region->name ) {
        free(region);
        goto unlock_and_return;
    }
    region->handle = SCOREP_USER_INVALID_REGION;
    /* The regions are PaRSEC events, they have no source location */
    SCOREP_User_RegionInit(&region->handle,
                           &SCOREP_User_LastFileName, &SCOREP_User_LastFileHandle,
                           region->name, SCOREP_USER_REGION_TYPE_COMMON,
                           "", 0);
    if( NULL != SCOREP_User_RegionSetGroup ) {
        SCOREP_User_RegionSetGroup(region->handle, PARSEC_SCOREP_GROUP_NAME);
    }
    region->next = parsec_scorep_regions;
    parsec_scorep_regions = region;
    handle = region->handle;
  unlock_and_return:
    pthread_mutex_unlock(&parsec_scorep_regions_lock);
    return handle;
}

static void scorep_register_key(int key, const parsec_profiling_key_t *entry)
{
    parsec_scorep_chunk_t *chunk;
    int c = key / PARSEC_SCOREP_CHUNK_SIZE;

    if( !parsec_scorep_active || key < 0 || c >= PARSEC_SCOREP_MAX_CHUNKS ) {
        return;
    }

    chunk = parsec_scorep_chunks[c];
    if( NULL == chunk ) {
        chunk = (parsec_scorep_chunk_t*)calloc(1, sizeof(parsec_scorep_chunk_t));
        if( NULL == chunk ) {
            return;
        }
        if( !parsec_atomic_cas_ptr(&parsec_scorep_chunks[c], NULL, chunk) ) {
            free(chunk);
            chunk = parsec_scorep_chunks[c];
        }
    }
    chunk->handles[key % PARSEC_SCOREP_CHUNK_SIZE] = parsec_scorep_region_lookup(entry->name);
}

static SCOREP_User_RegionHandle parsec_scorep_get_handle(int key)
{
    parsec_scorep_chunk_t *chunk;
    int c = key / PARSEC_SCOREP_CHUNK_SIZE;

    if( key < 0 || c >= PARSEC_SCOREP_MAX_CHUNKS ) {
        return SCOREP_USER_INVALID_REGION;
    }
    chunk = parsec_scorep_chunks[c];
    if( NULL == chunk ) {
        return SCOREP_USER_INVALID_REGION;
    }
    return chunk->handles[key % PARSEC_SCOREP_CHUNK_SIZE];
}

static parsec_scorep_stack_t *parsec_scorep_get_stack(void)
{
    parsec_scorep_stack_t *stack = PARSEC_TLS_GET_SPECIFIC(parsec_scorep_tls_stack);
    if( NULL == stack ) {
        stack = (parsec_scorep_stack_t*)calloc(1, sizeof(parsec_scorep_stack_t));
        if( NULL == stack ) {
            return NULL;
        }
        PARSEC_TLS_SET_SPECIFIC(parsec_scorep_tls_stack, stack);
    }
    return stack;
}

static int parsec_scorep_same_frame(const parsec_scorep_frame_t *frame,
                                    int key, uint64_t event_id, uint32_t taskpool_id)
{
    if( frame->key != key || frame->event_id != event_id ) {
        return 0;
    }
    return (frame->taskpool_id == taskpool_id) ||
           (frame->taskpool_id == PROFILE_OBJECT_ID_NULL) ||
           (taskpool_id == PROFILE_OBJECT_ID_NULL);
}

/* Exit the regions of the frames above (and including) depth */
static void parsec_scorep_unwind(parsec_scorep_stack_t *stack, int depth)
{
    while( stack->depth > depth ) {
        stack->depth--;
        SCOREP_User_RegionEnd(stack->frames[stack->depth].handle);
    }
}

static int scorep_trace(parsec_profiling_stream_t *stream, void *state,
                        const parsec_profiling_trace_event_t *event)
{
    parsec_scorep_stack_t *stack;
    SCOREP_User_RegionHandle handle;
    int depth;

    (void)stream; (void)state;
    if( !parsec_scorep_active ||
        (event->flags & PARSEC_PROFILING_EVENT_COUNTER) ||
        (!event->is_default_stream && !parsec_scorep_all_streams) ) {
        return 0;
    }
    stack = parsec_scorep_get_stack();
    if( NULL == stack ) {
        return 0;
    }

    if( event->is_start ) {
        handle = parsec_scorep_get_handle(event->key);
        if( SCOREP_USER_INVALID_REGION == handle ) {
            return 0;
        }
        if( stack->depth == stack->size ) {
            int new_size = (0 == stack->size) ? 16 : 2 * stack->size;
            parsec_scorep_frame_t *frames =
                (parsec_scorep_frame_t*)realloc(stack->frames, new_size * sizeof(parsec_scorep_frame_t));
            if( NULL == frames ) {
                return 0;
            }
            stack->frames = frames;
            stack->size = new_size;
        }
        stack->frames[stack->depth].handle = handle;
        stack->frames[stack->depth].key = event->key;
        stack->frames[stack->depth].event_id = event->event_id;
        stack->frames[stack->depth].taskpool_id = event->taskpool_id;
        stack->depth++;
        SCOREP_User_RegionEnter(handle);
        return 0;
    }

    for( depth = stack->depth - 1; depth >= 0; depth-- ) {
        if( parsec_scorep_same_frame(&stack->frames[depth], event->key, event->event_id, event->taskpool_id) ) {
            break;
        }
    }
    if( depth < 0 ) {
        /* Either never entered, or already closed to keep the regions nested */
        return 0;
    }
    if( depth != stack->depth - 1 &&
        parsec_atomic_cas_int32(&parsec_scorep_repair_warned, 0, 1) ) {
        parsec_warning("PaRSEC Profiling -- Score-P: profiling events are not properly nested on a thread,"
                       " the enclosed Score-P regions are closed early to keep the measurement consistent.");
    }
    parsec_scorep_unwind(stack, depth);
    return 0;
}

static void scorep_thread_fini(void)
{
    parsec_scorep_stack_t *stack;

    if( !parsec_scorep_active ) {
        return;
    }
    stack = PARSEC_TLS_GET_SPECIFIC(parsec_scorep_tls_stack);
    if( NULL == stack ) {
        return;
    }
    parsec_scorep_unwind(stack, 0);
    PARSEC_TLS_SET_SPECIFIC(parsec_scorep_tls_stack, NULL);
    free(stack->frames);
    free(stack);
}

static void scorep_fini(void)
{
    parsec_scorep_region_t *region;

    scorep_thread_fini();
    parsec_scorep_active = 0;
    for(int i = 0; i < PARSEC_SCOREP_MAX_CHUNKS; i++) {
        free(parsec_scorep_chunks[i]);
        parsec_scorep_chunks[i] = NULL;
    }
    pthread_mutex_lock(&parsec_scorep_regions_lock);
    while( NULL != (region = parsec_scorep_regions) ) {
        parsec_scorep_regions = region->next;
        free(region->name);
        free(region);
    }
    pthread_mutex_unlock(&parsec_scorep_regions_lock);
}

parsec_profiling_substrate_t parsec_profiling_substrate_scorep = {
    .name             = "scorep",
    .kind             = PARSEC_PROFILING_SUBSTRATE_MIRROR,
    .register_mca     = scorep_register_mca,
    .init             = scorep_init,
    .fini             = scorep_fini,
    .register_key     = scorep_register_key,
    .dictionary_flush = scorep_dictionary_flush,
    .trace            = scorep_trace,
    .thread_fini      = scorep_thread_fini,
};

#endif /* defined(PARSEC_PROF_TRACE_SCOREP) */
