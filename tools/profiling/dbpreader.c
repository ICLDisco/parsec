/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <errno.h>
#if defined(PARSEC_PROFILING_USE_MMAP)
#include <sys/mman.h>
#endif
#include <sys/types.h>
#include <pthread.h>
#include <fcntl.h>
#include <stdarg.h>

#include "parsec/profiling.h"
#include "parsec/parsec_binary_profile.h"
#include "dbpreader.h"

#ifdef DEBUG
#undef DEBUG
#endif

#if defined(PARSEC_DEBUG_NOISIER)
#define DEBUG(...) output(__VA_ARGS__)
#else
#define DEBUG(toto) do {} while(0)
#endif

#ifdef WARNING
#undef WARNING
#endif
#define WARNING(...) output(__VA_ARGS__)

static void output(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}

/* Buffer constants */
static int event_buffer_size = 0;
static int event_avail_space = 0;

typedef enum {
    SUCCESS,
    UNABLE_TO_OPEN,
    TOO_SMALL,
    NO_MAGICK,
    WRONG_BYTE_ORDER,
    DIFF_HR_ID,
    DIFF_BUFFER_SIZE,
    DICT_IGNORED,
    DICT_BROKEN,
    THREADS_BROKEN,
    TRACE_TRUNCATED,
    TRACE_OVERFLOW,
    DUPLICATE_RANK,
} OPEN_ERROR;

struct dbp_file {
    struct dbp_multifile_reader *parent;
    char  *hr_id;
    char  *filename;
    int    fd;
    int    rank;
    int    nb_infos;
    int    nb_threads;
    int    nb_dico_map;
    int    error;
    int   *dico_map;
    struct dbp_info  **infos;
    struct dbp_thread *threads;
};

struct dbp_event {
    parsec_profiling_output_t *native;
};

void dbp_file_print(dbp_file_t * file) {
    printf("parent %p\n", file->parent);
    printf("hr_id %s\n", file->hr_id);
    printf("fd %d\n", file->fd);
    printf("filename %s\n", file->filename);
    printf("rank %d\n", file->rank);
    printf("nb_infos %d\n", file->nb_infos);
    printf("nb_threads %d\n", file->nb_threads);
    printf("infos %p\n", file->infos);
}

int dbp_event_get_key(const dbp_event_t *e)
{
    return e->native->event.key;
}

int dbp_event_get_flags(const dbp_event_t *e)
{
    return e->native->event.flags;
}

uint64_t dbp_event_get_event_id(const dbp_event_t *e)
{
    return e->native->event.event_id;
}

uint32_t dbp_event_get_taskpool_id(const dbp_event_t *e)
{
    return e->native->event.taskpool_id;
}

uint64_t dbp_event_get_timestamp(const dbp_event_t *e)
{
    return e->native->event.timestamp;
}

void *dbp_event_get_info(const dbp_event_t *e)
{
    if( EVENT_HAS_INFO( e->native ) ) {
        return e->native->info;
    }
    return NULL;
}

int dbp_event_info_len(const dbp_event_t *e, const dbp_file_t *file)
{
    if( e->native->event.flags & PARSEC_PROFILING_EVENT_HAS_INFO ) {
        return dbp_dictionary_keylen(dbp_file_get_dictionary(file, BASE_KEY(dbp_event_get_key(e))));
    }
    return 0;
}

struct dbp_event_iterator {
    const dbp_thread_t              *thread;
    dbp_event_t                      current_event;
    parsec_profiling_buffer_t        *current_events_buffer;
    int64_t                          current_event_position;
    int64_t                          current_event_index;
    int64_t                          current_buffer_position;
#ifndef _NDEBUG
    uint64_t                         last_event_date;
#endif
};

struct dbp_info {
    char *key;
    char *value;
};

// file here
struct dbp_dictionary {
    int keylen;
    char *name;
    char *attributes;
    char *convertor;
};

char *dbp_dictionary_name(const dbp_dictionary_t *dico)
{
    return dico->name;
}

char *dbp_dictionary_attributes(const dbp_dictionary_t *dico)
{
    return dico->attributes;
}

char *dbp_dictionary_convertor(const dbp_dictionary_t *dico)
{
    return dico->convertor;
}

int dbp_dictionary_keylen(const dbp_dictionary_t *dico)
{
    return dico->keylen;
}

struct dbp_multifile_reader {
    parsec_profiling_binary_file_header_t header;
    int nb_files;
    int dico_size;
    int dico_allocated;
    int nb_infos;
    int last_error;
    dbp_info_t *infos;
    dbp_dictionary_t *dico_keys;
    dbp_file_t *files;
};

dbp_dictionary_t *dbp_file_get_dictionary(const dbp_file_t *file, int did)
{
    int global_dico_id;
    assert( did >= 0 && did < file->nb_dico_map );
    global_dico_id = file->dico_map[did];
    assert( global_dico_id >= 0 && global_dico_id < file->parent->dico_size );
    return &(file->parent->dico_keys[global_dico_id]);
}

int dbp_file_translate_local_dico_to_global(const dbp_file_t *file, int lid)
{
    assert( lid >= 0 && lid < file->nb_dico_map );
    return file->dico_map[lid];
}

#define DBP_EVENT_LENGTH(dbp_event, dbp_object)             \
  (sizeof(parsec_profiling_output_base_event_t) +           \
   (EVENT_HAS_INFO((dbp_event)->native) ?                   \
    (dbp_object)->parent->dico_keys[(dbp_object)->dico_map[BASE_KEY((dbp_event)->native->event.key)]].keylen : 0))

typedef struct {
    uint64_t timestamp;
    off_t    offset;
    int64_t  event_idx;
} event_cache_item_t;

typedef struct {
    event_cache_item_t *items;
    size_t              len;
    size_t              size;
} event_cache_key_t;

typedef struct {
    pthread_mutex_t    mtx;
    event_cache_key_t *keys;
    int                done;
} event_cache_t;

struct dbp_thread {
    const parsec_profiling_stream_t *profile;
    dbp_file_t                      *file;
    dbp_info_t                      *infos;
    event_cache_t                    cache;
    int                              nb_infos;
};

#if defined(PARSEC_PROFILING_USE_MMAP)
static void release_events_buffer(parsec_profiling_buffer_t *buffer)
{
    if( NULL == buffer )
        return;
    if( munmap(buffer, event_buffer_size) == -1 ) {
        WARNING("Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                 buffer, strerror(errno));
    }
}

static parsec_profiling_buffer_t *refer_events_buffer( const dbp_file_t *file, int64_t offset )
{
    parsec_profiling_buffer_t *res;
    res = mmap(NULL, event_buffer_size, PROT_READ, MAP_SHARED, file->fd, offset);
    if( MAP_FAILED == res )
        return NULL;
    return res;
}
#else
static void release_events_buffer(parsec_profiling_buffer_t *buffer)
{
    if( NULL == buffer )
        return;
    free(buffer);
}

static parsec_profiling_buffer_t *refer_events_buffer( const dbp_file_t *file, int64_t offset )
{
    off_t pos = lseek(file->fd, offset, SEEK_SET);
    if( -1 == pos ) {
        return NULL;
    }
    parsec_profiling_buffer_t *res = (parsec_profiling_buffer_t*)malloc(event_buffer_size);
    pos = read(file->fd, res, event_buffer_size);
    if( pos <= 0 ) {
        free(res);
        res = NULL;
    }
    return res;
}

#endif  /* defined(PARSEC_PROFILING_USE_MMAP) */

dbp_event_iterator_t *dbp_iterator_new_from_thread(const dbp_thread_t *th)
{
    dbp_event_iterator_t *res = (dbp_event_iterator_t*)malloc(sizeof(dbp_event_iterator_t));
    res->thread = th;
    res->current_event.native = NULL;
    res->current_event_position = 0;
    res->current_event_index = 0;
    res->current_buffer_position = (off_t)-1;
    res->current_events_buffer  = NULL;
#ifndef _NDEBUG
    res->last_event_date = 0;
#endif
    (void)dbp_iterator_first(res);
    return res;
}

dbp_event_iterator_t *dbp_iterator_new_from_iterator(const dbp_event_iterator_t *it)
{
    dbp_event_iterator_t *res = (dbp_event_iterator_t*)malloc(sizeof(dbp_event_iterator_t));
    res->thread = it->thread;
    res->current_event.native = it->current_event.native;
    res->current_event_position = it->current_event_position;
    res->current_event_index = it->current_event_index;
    res->current_buffer_position = it->current_buffer_position;
    res->current_events_buffer = refer_events_buffer( it->thread->file, res->current_buffer_position );
#ifndef _NDEBUG
    res->last_event_date = it->last_event_date;
#endif
    return res;
}

const dbp_event_t *dbp_iterator_current(dbp_event_iterator_t *it)
{
    if( it->current_events_buffer == NULL ||
        it->current_event.native == NULL )
        return NULL;
#ifndef _NDEBUG
    assert(it->current_event.native->event.timestamp >= it->last_event_date);
    it->last_event_date = it->current_event.native->event.timestamp;
#endif
    return &it->current_event;
}

/* move iterator to position event_pos in current buffer and index event_idx */
static inline const dbp_event_t *
dbp_iterator_move_to_event(dbp_event_iterator_t *it,
                           int64_t event_pos, int64_t event_idx)
{
    assert( event_pos >= 0 && event_pos < event_avail_space );
    assert( event_idx >= 0 );

    it->current_event_position = event_pos;
    it->current_event_index = event_idx;

    if( it->current_events_buffer != NULL ) {
        assert( it->current_event_index < it->current_events_buffer->this_buffer.nb_events );
        it->current_event.native = (parsec_profiling_output_t*)&(it->current_events_buffer->buffer[it->current_event_position]);
    } else {
        it->current_event.native = NULL;
    }

    assert((it->current_event.native == NULL) ||
           (it->current_event.native->event.timestamp != 0));
    return dbp_iterator_current(it);
}

/* move iterator to first event in buffer with offset */
static inline const dbp_event_t *
dbp_iterator_set_offset(dbp_event_iterator_t *it, off_t offset)
{
    if( it->current_events_buffer != NULL ) {
        release_events_buffer( it->current_events_buffer );
        it->current_events_buffer = NULL;
        it->current_event.native = NULL;
    }

    it->current_events_buffer = refer_events_buffer( it->thread->file, offset );
    it->current_buffer_position = offset;

    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    return dbp_iterator_move_to_event(it, 0, 0);
}

const dbp_event_t *dbp_iterator_first(dbp_event_iterator_t *it)
{
    return dbp_iterator_set_offset(it, it->thread->profile->first_events_buffer_offset);
}

static const dbp_event_t *dbp_iterator_next_buffer(dbp_event_iterator_t *it)
{
    off_t next_off;

    if( NULL == it->current_event.native )
        return NULL;
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );

    next_off = it->current_events_buffer->next_buffer_file_offset;
    return dbp_iterator_set_offset(it, next_off);
}

static const dbp_event_t *dbp_iterator_next_in_buffer(dbp_event_iterator_t *it)
{
    size_t elen;

    if( NULL == it->current_event.native )
        return NULL;
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );

    if( it->current_event_index+1 >= it->current_events_buffer->this_buffer.nb_events ) {
        it->current_event.native = NULL;
        return NULL;
    }

    elen = DBP_EVENT_LENGTH(&it->current_event, it->thread->file);
    return dbp_iterator_move_to_event(it, it->current_event_position + elen,
                                          it->current_event_index + 1);
}

const dbp_event_t *dbp_iterator_next(dbp_event_iterator_t *it)
{
    if( NULL == it->current_event.native )
        return NULL;
    assert( it->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );

    if( it->current_event_index+1 >= it->current_events_buffer->this_buffer.nb_events ) {
        return dbp_iterator_next_buffer(it);
    }

    return dbp_iterator_next_in_buffer(it);
}

const dbp_thread_t *dbp_iterator_thread(const dbp_event_iterator_t *it)
{
    return it->thread;
}

void dbp_iterator_delete(dbp_event_iterator_t *it)
{
    if( NULL != it->current_events_buffer )
        release_events_buffer(it->current_events_buffer);
    free(it);
}

static inline int dbp_events_match(const dbp_event_t *s, const dbp_event_t *e)
{
    int s_key = dbp_event_get_key(s);
    int e_key = dbp_event_get_key(e  );
    return ( (KEY_IS_START(s_key)          && KEY_IS_END(e_key))            &&
             (BASE_KEY(    s_key)          == BASE_KEY(  e_key))            &&
             (dbp_event_get_event_id(   s) == dbp_event_get_event_id(   e)) &&
             (dbp_event_get_taskpool_id(s) == dbp_event_get_taskpool_id(e)) &&
             (dbp_event_get_timestamp(  s) <= dbp_event_get_timestamp(  e)) );
}

/* minimum allocation count for cache */
#define EVENT_CACHE_MIN_ALLOC 64
/* build a "cache" of events where the end event
 * does not immediately follow the start event */
static void build_unmatched_events_in_thread(dbp_thread_t *thr)
{
    const dbp_event_t      *e1,  *e2;
    dbp_event_iterator_t   *i1,  *i2;
    int key2;
    uint64_t timestamp2;

    event_cache_key_t  *cache_key;
    event_cache_item_t *cache_item;
    size_t              cache_index;
    off_t               offset;

    /* lock cache; we're modifying volatile state! */
    pthread_mutex_lock(&thr->cache.mtx);
    if( thr->cache.done ) {
        /* cache already built, we don't need to do anything */
        goto build_events_done;
    }

    /* iterator 1 points to current event */
    i1 = dbp_iterator_new_from_thread( thr );
    e1 = dbp_iterator_current( i1 );

    /* iterator 2 points to next event */
    i2 = dbp_iterator_new_from_thread( thr );
    e2 = dbp_iterator_next( i2 );

    while( NULL != e2 ) {
        key2 = dbp_event_get_key(e2);

        /* if e2 is end event, but e1 doesn't match, e2 not in expected order
         * store e2 position in cache to lookup later for potential match */
        if( KEY_IS_END(key2) && !dbp_events_match(e1, e2) ) {
            cache_key = &thr->cache.keys[BASE_KEY(key2)];

            cache_index = cache_key->len++;
            /* if index == size, we need to grow the array */
            if( cache_index == cache_key->size ) {
                /* if size == 0, this is the first mismatched event with this key */
                cache_key->size = cache_key->size ? cache_key->size * 2 :
                                                    EVENT_CACHE_MIN_ALLOC;
                cache_key->items = realloc(cache_key->items, sizeof(event_cache_item_t[cache_key->size]));
            }

            /* cache timestamp and buffer offset for this event
             * note that we don't store the exact index of the event,
             * so a consumer should make sure to search through the buffer */
            cache_item = &cache_key->items[cache_index];

            /* event pos it always less than event_avail_space
             * and buffer offset is always a multiple of event_buffer_size
             * so these can be combined together and recovered */
            assert( i2->current_event_position >= 0 );
            assert( i2->current_event_position < event_avail_space );
            assert( (i2->current_buffer_position % event_buffer_size) == 0 );

            offset = i2->current_buffer_position + i2->current_event_position;
            cache_item->timestamp = dbp_event_get_timestamp(e2);
            cache_item->offset    = offset;
            cache_item->event_idx = i2->current_event_index;

            assert( (offset % event_buffer_size) == i2->current_event_position);
        }

        /* advance both iterators */
        e1 = dbp_iterator_next( i1 );
        e2 = dbp_iterator_next( i2 );
    }

    dbp_iterator_delete( i1 );
    dbp_iterator_delete( i2 );

    /* set cache to done - it doesn't need to be rebuilt */
    thr->cache.done = 1;

build_events_done:
    pthread_mutex_unlock(&thr->cache.mtx);
}

typedef struct {
    const dbp_event_t        *ref;
    const event_cache_item_t *last;
} bsearch_key_t;

static int bsearch_compare(const void *key, const void *el)
{
    /* technically does Bad Thing (shouldn't modify key), but probably works */
    bsearch_key_t      *bsearch_key = (bsearch_key_t *)key;
    const event_cache_item_t *cache_item  = (const event_cache_item_t *)el;
    bsearch_key->last = cache_item;
    if( dbp_event_get_timestamp(bsearch_key->ref) < cache_item->timestamp )
        return -1;
    if( dbp_event_get_timestamp(bsearch_key->ref) > cache_item->timestamp )
        return 1;
    return 0;
}

static const event_cache_item_t*
dbp_event_find_in_cache(const dbp_thread_t *thr,
                        const dbp_event_t *ref)
{
    event_cache_key_t *cache_key;
    bsearch_key_t      bsearch_key = { ref, NULL };

    /* ensure we have an unmatched event cache
     * casts away const because build_unmatched_events_in_thread modifies thr
     * thr should be memory we allocated anyway though, so this should be safe
     * ... right? */
    build_unmatched_events_in_thread((dbp_thread_t*)thr);

    /* do binary search in cache of key for events at ref timestamp
     * we throw away the results of the search, because it's very unlikely to
     * find this EXACT timestamp; however, we use a side-effect of the search
     * in the comparison function (bsearch_compare) to store the last item in
     * the cache array that was considered; this is the "insertion point" for
     * ref's timestamp and is the timestamp closest to ref's we could find, so
     * we return it as the starting point for the subsequent search */
    cache_key = &thr->cache.keys[BASE_KEY(dbp_event_get_key(ref))];
    bsearch(&bsearch_key, cache_key->items, cache_key->len,
            sizeof(event_cache_item_t), bsearch_compare);

    return bsearch_key.last;
}

int dbp_iterator_move_to_matching_event(dbp_event_iterator_t *pos,
                                        const dbp_event_t *ref)
{
    const event_cache_item_t *cache_item;
    const event_cache_key_t  *cache_key;
    const dbp_event_t        *e;
    const dbp_thread_t       *thr = pos->thread;
    off_t                     offset;
    int64_t                   event_pos;

    cache_item = dbp_event_find_in_cache( thr, ref );
    cache_key  = &thr->cache.keys[BASE_KEY(dbp_event_get_key(ref))];

    /* dbp_event_find_in_cache can return NULL
     * if the thread doesn't have a matching event */
    if( NULL == cache_item ) {
        /* set iterator to past-the-end */
        dbp_iterator_set_offset(pos, (off_t)-1);
        return 0;
    }

    assert(&cache_key->items[0]              <= cache_item);
    assert(&cache_key->items[cache_key->len] >  cache_item);

    /* iterate over all cached events containing possible matches */
    while( cache_item < &cache_key->items[cache_key->len] ) {
        /* we computed cache_item->offset as buffer_position + event_position,
         * so we must recover these values */
        event_pos = cache_item->offset % event_buffer_size;
        offset = cache_item->offset - event_pos;
        /* change buffer if necessary */
        if( pos->current_buffer_position != offset )
            dbp_iterator_set_offset(pos, offset);
        /* set iterator to current cached event */
        e = dbp_iterator_move_to_event(pos, event_pos, cache_item->event_idx);
        /* check if event matches */
        if( (NULL != e) && dbp_events_match(ref, e) )
            return 1;
        cache_item++;
    }

    /* set iterator to past-the-end */
    dbp_iterator_set_offset(pos, (off_t)-1);
    return 0;
}

dbp_event_iterator_t *dbp_iterator_find_matching_event_all_threads(const dbp_event_iterator_t *pos)
{
    dbp_event_iterator_t *it;
    dbp_thread_t *thr;
    const dbp_event_t *ref;
    const dbp_event_t *e;
    dbp_file_t *dbp_file;
    int tid;

    dbp_file = pos->thread->file;
    ref = dbp_iterator_current((dbp_event_iterator_t *)pos);

    /* most start events are immediately followed by their end event */
    it = dbp_iterator_new_from_iterator(pos);
    e = dbp_iterator_next(it);
    /* e can be NULL if pos is last event in stream; there is no next event */
    if( (NULL != e) && dbp_events_match(ref, e) )
        return it;
    dbp_iterator_delete(it);

    /* search through possibly matching events in this thread */
    it = dbp_iterator_new_from_thread( pos->thread );
    if( dbp_iterator_move_to_matching_event(it, ref) )
        return it;
    dbp_iterator_delete(it);

    /* try other threads */
    for( tid = 0; tid < dbp_file_nb_threads(dbp_file); tid++) {
        thr = dbp_file_get_thread(dbp_file, tid);
        /* skip same thread */
        if( pos->thread == thr )
            continue;
        /* same logic as above */
        it = dbp_iterator_new_from_thread( thr );
        if( dbp_iterator_move_to_matching_event(it, ref) )
            return it;
        dbp_iterator_delete(it);
    }

    return NULL;
}

char *dbp_info_get_key(const dbp_info_t *info)
{
    return info->key;
}

char *dbp_info_get_value(const dbp_info_t *info)
{
    return info->value;
}

int dbp_thread_nb_events(const dbp_thread_t *th)
{
    return th->profile->nb_events;
}

int dbp_thread_nb_infos(const dbp_thread_t *th)
{
    return th->nb_infos;
}

dbp_info_t *dbp_thread_get_info(const dbp_thread_t *th, int iid)
{
    assert( iid >= 0 && iid < th->nb_infos );
    return &th->infos[iid];
}

char *dbp_thread_get_hr_id(const dbp_thread_t *th)
{
    return th->profile->hr_id;
}

dbp_thread_t *dbp_file_get_thread(const dbp_file_t *file, int tid)
{
    assert( tid >= 0 && tid < file->nb_threads );
    return &file->threads[tid];
}

char *dbp_file_hr_id(const dbp_file_t *file)
{
    return file->hr_id;
}

int dbp_file_get_rank(const dbp_file_t *file)
{
    return file->rank;
}


int dbp_file_nb_threads(const dbp_file_t *file)
{
    return file->nb_threads;
}

char * dbp_file_get_name(const dbp_file_t *file)
{
    return file->filename;
}

int dbp_file_nb_infos(const dbp_file_t *file)
{
    return file->nb_infos;
}

int dbp_file_nb_dictionary_entries(const dbp_file_t *file)
{
    return file->nb_dico_map;
}

int dbp_file_error(const dbp_file_t *file)
{
    return file->error;
}

dbp_info_t *dbp_file_get_info(const dbp_file_t *file, int iid)
{
    assert( iid >= 0 && iid < file->nb_infos && file->infos != NULL);
    return file->infos[iid];
}

dbp_file_t *dbp_reader_get_file(const dbp_multifile_reader_t *dbp, int fid)
{
    assert(fid >= 0 && fid < dbp->nb_files );
    return &dbp->files[fid];
}

int dbp_reader_nb_files(const dbp_multifile_reader_t *dbp)
{
    return dbp->nb_files;
}

int dbp_reader_nb_dictionary_entries(const dbp_multifile_reader_t *dbp)
{
    return dbp->dico_size;
}

dbp_dictionary_t *dbp_reader_get_dictionary(const dbp_multifile_reader_t *dbp, int i)
{
    assert(i >= 0);
    assert(i < dbp->dico_size);
    return &dbp->dico_keys[i];
}

int dbp_reader_last_error(const dbp_multifile_reader_t *dbp)
{
    return dbp->last_error;
}

void dbp_reader_close_files(dbp_multifile_reader_t *dbp)
{
    (void)dbp;
}

static void read_infos(dbp_file_t *dbp, parsec_profiling_binary_file_header_t *head)
{
    parsec_profiling_buffer_t *info, *next;
    parsec_profiling_info_buffer_t *ib;
    dbp_info_t *id;
    char *value;
    int nb, nbthis, pos, vpos, tr, vs;

    dbp->nb_infos = head->info_size;
    if( dbp->nb_infos == 0 ) {
        dbp->infos = NULL;
        return;
    }

    dbp->infos = (dbp_info_t**)malloc(sizeof(dbp_info_t*) * dbp->nb_infos);

    info = refer_events_buffer(dbp, head->info_offset );
    if( NULL == info ) {
        fprintf(stderr, "Unable to read first info at offset %"PRId64": %d general file info in '%s' lost\n",
                head->info_offset, dbp->nb_infos, dbp->filename);
        dbp->nb_infos = 0;
        free( dbp->infos );
        dbp->infos = NULL;
        return;
    }
    assert( PROFILING_BUFFER_TYPE_GLOBAL_INFO == info->buffer_type );

    nb = 0;
    nbthis = 0;
    pos = 0;
    while( nb < dbp->nb_infos ) {
        assert( PROFILING_BUFFER_TYPE_GLOBAL_INFO == info->buffer_type );

        ib = (parsec_profiling_info_buffer_t*)&info->buffer[pos];
        id = (dbp_info_t *)malloc(sizeof(dbp_info_t));
        id->key = (char*)malloc(ib->info_size+1);
        id->value = (char*)malloc(ib->value_size+1);
        memcpy(id->key, ib->info_and_value, ib->info_size);
        id->key[ib->info_size] = '\0';

        nbthis++;

        pos += sizeof(parsec_profiling_info_buffer_t) + ib->info_size - 1;

        value = ib->info_and_value + ib->info_size;
        vpos = 0;
        vs = ib->value_size;
        while( vpos < vs ) {
            tr = (event_avail_space - pos) < (vs-vpos) ? (event_avail_space - pos) : (vs-vpos);
            memcpy(id->value + vpos, value, tr);
            pos += tr;
            vpos += tr;
            if( pos == event_avail_space ) {
                next = refer_events_buffer( dbp, info->next_buffer_file_offset );
                if( NULL == next ) {
                    fprintf(stderr, "Info entry %d is broken. Only %d entries read from '%s'\n",
                            dbp->nb_infos - nb, nb, dbp->filename);
                    release_events_buffer( info );
                    dbp->nb_infos = nb;
                    free(id);
                    return;
                }
                assert( PROFILING_BUFFER_TYPE_GLOBAL_INFO == next->buffer_type );
                release_events_buffer( info );
                info = next;

                pos = 0;
                nbthis = 0;
                value = (char*)&(info->buffer[pos]);
            }
        }
        id->value[vs] = '\0';

        dbp->infos[nb] = id;
        nb++;

        if( (nb < dbp->nb_infos) && (nbthis == info->this_buffer.nb_infos) ) {
            next = refer_events_buffer( dbp, info->next_buffer_file_offset );
            if( NULL == next ) {
                fprintf(stderr, "Info entry %d is broken. Only %d entries read from '%s'\n",
                        dbp->nb_infos - nb, nb, dbp->filename);
                release_events_buffer( info );
                dbp->nb_infos = nb;
                return;
            }
            assert( PROFILING_BUFFER_TYPE_GLOBAL_INFO == next->buffer_type );
            release_events_buffer( info );
            info = next;

            pos = 0;
            nbthis = 0;
        }
    }
    release_events_buffer( info );
}

static int read_dictionary(dbp_file_t *file, const parsec_profiling_binary_file_header_t *head)
{
    parsec_profiling_buffer_t *dico, *next;
    parsec_profiling_key_buffer_t *a;
    int nb, nbthis, pos, i;
    dbp_multifile_reader_t *dbp = file->parent;

    /* Dictionaries match: take the first in memory */
    dico = refer_events_buffer( file, head->dictionary_offset );
    if( NULL == dico ) {
        fprintf(stderr, "Unable to read entire dictionary entry at offset %"PRId64"\n",
                head->dictionary_offset);
        return -1;
    }
    assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );

    file->nb_dico_map = head->dictionary_size;
    file->dico_map = (int*)malloc(sizeof(int) * file->nb_dico_map);
    memset(file->dico_map, -1, sizeof(int) * file->nb_dico_map);

    nb = 0;
    nbthis = dico->this_buffer.nb_dictionary_entries;
    pos = 0;
    while( nb < file->nb_dico_map ) {
        a = (parsec_profiling_key_buffer_t*)&dico->buffer[pos];

        for(i = 0; i < dbp->dico_size; i++) {
            if( a->keyinfo_length == dbp->dico_keys[i].keylen &&
                strcmp(a->name, dbp->dico_keys[i].name) == 0 &&
                strcmp(a->convertor, dbp->dico_keys[i].convertor) == 0 )
                break;
        }
        file->dico_map[nb] = i;
        if(i == dbp->dico_size) {
            if (dbp->dico_size == dbp->dico_allocated) {
                dbp->dico_allocated += 16;
                dbp->dico_keys = realloc(dbp->dico_keys, dbp->dico_allocated * sizeof(dbp_dictionary_t));
            }
            dbp->dico_keys[i].name = malloc(64);
            strncpy(dbp->dico_keys[i].name, a->name, 64);
            assert(strlen(a->attributes) > 6);
            dbp->dico_keys[i].attributes = malloc(128);
            strncpy(dbp->dico_keys[i].attributes, ((char *) a->attributes) + strlen(a->attributes) - 6, 128);
            dbp->dico_keys[i].convertor = (char *) malloc(a->keyinfo_convertor_length + 1);
            memcpy(dbp->dico_keys[i].convertor,
                   a->convertor,
                   a->keyinfo_convertor_length);
            dbp->dico_keys[i].convertor[a->keyinfo_convertor_length] = '\0';
            dbp->dico_keys[i].keylen = a->keyinfo_length;
            dbp->dico_size++;
        }
        pos += a->keyinfo_convertor_length - 1 + sizeof(parsec_profiling_key_buffer_t);
        nb++;
        nbthis--;

        if( nb < file->nb_dico_map && nbthis == 0 ) {
            next = refer_events_buffer( file, dico->next_buffer_file_offset );
            if( NULL == next ) {
                fprintf(stderr, "Dictionary entry %d is broken. Dictionary broken.\n", nb);
                release_events_buffer( dico );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_DICTIONARY == dico->buffer_type );
            release_events_buffer( dico );
            dico = next;
            nbthis = dico->this_buffer.nb_dictionary_entries;

            pos = 0;
        }
    }
    release_events_buffer( dico );
    return 0;
}

static size_t read_thread_infos(parsec_profiling_stream_t* res,
                                dbp_thread_t *th,
                                int nb_infos, const char *br)
{
    parsec_profiling_info_buffer_t *ib;
    dbp_info_t *nfo;
    int i, pos = 0;
    pos = 0;
    res->infos = NULL;
    th->infos = (dbp_info_t *)calloc(nb_infos, sizeof(dbp_info_t));
    for(i = 0; i < nb_infos; i++) {
        ib = (parsec_profiling_info_buffer_t*)(&br[pos]);
        pos += ib->info_size + ib->value_size + sizeof(parsec_profiling_info_buffer_t) - 1;
        nfo = (dbp_info_t*)&th->infos[i];
        nfo->key = (char*)malloc(ib->info_size+1);
        memcpy(nfo->key, ib->info_and_value, ib->info_size);
        nfo->key[ib->info_size] = '\0';
        nfo->value = (char*)malloc(ib->value_size+1);
        memcpy(nfo->value, ib->info_and_value + ib->info_size, ib->value_size);
        nfo->value[ib->value_size] = '\0';
    }
    th->nb_infos = nb_infos;
    return pos;
}

static int read_threads(dbp_file_t *dbp, const parsec_profiling_binary_file_header_t *head)
{
    parsec_profiling_stream_t *res;
    parsec_profiling_stream_buffer_t *br;
    parsec_profiling_buffer_t *b, *next;
    dbp_thread_t *thr;
    int nb, nbthis, pos;

    dbp->nb_threads = head->nb_threads;
    dbp->threads = (dbp_thread_t*)calloc(head->nb_threads, sizeof(dbp_thread_t));

    pos = 0;
    nb = head->nb_threads;
    b = refer_events_buffer(dbp, head->thread_offset);
    nbthis = b->this_buffer.nb_threads;
    while( nb > 0 ) {
        assert(PROFILING_BUFFER_TYPE_THREAD == b->buffer_type);
        assert(nbthis > 0);

        br = (parsec_profiling_stream_buffer_t*)&(b->buffer[pos]);
        res = (parsec_profiling_stream_t*)malloc( sizeof(parsec_profiling_stream_t) );
        res->next_event_position = -1; /* No need for a next event position */
        res->nb_events = br->nb_events;
        res->hr_id = (char*)malloc(128);
        strncpy(res->hr_id, br->hr_id, 128);
        res->first_events_buffer_offset = br->first_events_buffer_offset;
        res->current_events_buffer = refer_events_buffer(dbp, br->first_events_buffer_offset);

        PARSEC_OBJ_CONSTRUCT( res, parsec_list_item_t );

        thr = &dbp->threads[head->nb_threads - nb];
        thr->file        = dbp;
        thr->profile     = res;
        pthread_mutex_init(&thr->cache.mtx, NULL);
        thr->cache.keys = (event_cache_key_t*)calloc(
               dbp_file_nb_dictionary_entries(dbp), sizeof(event_cache_key_t));
        thr->cache.done  = 0;

        pos += sizeof(parsec_profiling_stream_buffer_t) - sizeof(parsec_profiling_info_buffer_t);
        pos += read_thread_infos( res, thr, br->nb_infos, (char*)br->infos );

        nbthis--;
        nb--;
        assert(nb >= 0);

        if( nbthis == 0 && nb > 0 ) {
            assert( b->next_buffer_file_offset != -1 );
            next = refer_events_buffer(dbp, b->next_buffer_file_offset);
            if( NULL == next ) {
                fprintf(stderr, "Unable to read thread entry %d/%d at offset %lx: Profile file broken\n",
                        head->nb_threads-nb, head->nb_threads, (unsigned long)b->next_buffer_file_offset);
                release_events_buffer( b );
                return -1;
            }
            assert( PROFILING_BUFFER_TYPE_THREAD == next->buffer_type );
            release_events_buffer( b );
            b = next;

            nbthis = b->this_buffer.nb_threads;

            pos = 0;
        }
    }

    release_events_buffer( b );
    return 0;
}

static dbp_multifile_reader_t *open_files(int nbfiles, char **filenames)
{
    int fd, i, p, n;
    parsec_profiling_buffer_t dummy_events_buffer;
    parsec_profiling_binary_file_header_t head;
    dbp_multifile_reader_t *dbp;

    dbp = (dbp_multifile_reader_t*)malloc(sizeof(dbp_multifile_reader_t));
    dbp->files = (dbp_file_t*)malloc(nbfiles * sizeof(dbp_file_t));
    dbp->last_error = SUCCESS;
    dbp->dico_size = 0;
    dbp->dico_allocated = 8;
    dbp->dico_keys = calloc(sizeof(dbp_dictionary_t), dbp->dico_allocated);

    n = 0;
    for(i = 0; i < nbfiles; i++) {
        dbp->files[n].error = SUCCESS;
        if (i > 0 && dbp->files[n - 1].error != SUCCESS)
            dbp->last_error = dbp->files[n - 1].error;

        fd = open(filenames[i], O_RDONLY);
        if( fd == -1 ) {
            fprintf(stderr, "Unable to open %s: %s -- skipped\n", filenames[i], strerror(errno));
            dbp->files[n].error = -UNABLE_TO_OPEN;
            continue;
        }

        dbp->files[n].filename = strdup(filenames[i]);
        dbp->files[n].parent = dbp;
        dbp->files[n].fd = fd;
        dbp->files[n].nb_infos = 0;

        if( (p = read( fd, &head, sizeof(parsec_profiling_binary_file_header_t) )) != sizeof(parsec_profiling_binary_file_header_t) ) {
            fprintf(stderr, "read %d bytes\n", p);
            fprintf(stderr, "File %s does not seem to be a correct PARSEC Binary Profile, ignored\n",
                    filenames[i]);
            dbp->files[n].error = -TOO_SMALL;
            goto close_and_continue;
        }

        if( strncmp( head.magick, PARSEC_PROFILING_MAGICK, 24 ) ) {
            fprintf(stderr, "read %d bytes found '%s', expected '%s'\n", p, head.magick, PARSEC_PROFILING_MAGICK);
            fprintf(stderr, "File %s does not seem to be a correct PARSEC Binary Profile, ignored\n",
                    filenames[i]);
            dbp->files[n].error = -NO_MAGICK;
            goto close_and_continue;
        }
        if( head.byte_order != 0x0123456789ABCDEF ) {
            fprintf(stderr, "The profile in file %s has been generated with a different byte ordering. File ignored\n",
                    filenames[i]);
            dbp->files[n].error = -WRONG_BYTE_ORDER;
            goto close_and_continue;
        }

        if(n == 0) {
            event_buffer_size = head.profile_buffer_size;
            event_avail_space = event_buffer_size -
                ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);
        } else {
            if( strncmp(head.hr_id, dbp->files[0].hr_id, 128) ) {
                fprintf(stderr, "The profile in file %s has unique id %s, which is not compatible with id %s of file %s. File ignored.\n",
                        dbp->files[n].filename, head.hr_id,
                        dbp->files[0].hr_id, dbp->files[0].filename);
                dbp->files[n].error = -DIFF_HR_ID;
                goto close_and_continue;
            }

            if( head.profile_buffer_size != event_buffer_size ) {
                fprintf(stderr, "The profile in file %s has a buffer size of %d, which is not compatible with the buffer size %d of file %s. File ignored.\n",
                        dbp->files[n].filename, head.profile_buffer_size,
                        event_buffer_size, dbp->files[0].filename);
                dbp->files[n].error = -DIFF_BUFFER_SIZE;
                goto close_and_continue;
            }
        }

        dbp->files[n].hr_id = strdup(head.hr_id);
        dbp->files[n].rank = head.rank;

        read_infos(&dbp->files[n], &head /*dbp->header*/);

        if( read_dictionary(&dbp->files[n], &head) != 0 ) {
            fprintf(stderr, "The profile in file %s has a broken dictionary. Trying to use the dictionary of next file. Ignoring the file.\n",
                    dbp->files[n].filename);
            dbp->files[n].error = -DICT_BROKEN;
            goto close_and_continue;
        }

        if( read_threads(&dbp->files[n], &head) != 0 ) {
            fprintf(stderr, "unable to read all threads of profile %d in file %s. File ignored.\n",
                    n, dbp->files[n].filename);
            dbp->files[n].error = -THREADS_BROKEN;
            goto close_and_continue;
        }

      close_and_continue:
        if( SUCCESS != dbp->files[n].error ) {
            close(fd);
            dbp->files[n].fd = -1;  /* not opened anymore */
            dbp->last_error = dbp->files[n].error;  /* record last error */
        }
        n++;
    }

    dbp->nb_files = n;
    event_avail_space = event_buffer_size -
        ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);

    return dbp;
}

dbp_multifile_reader_t *dbp_reader_open_files(int nbfiles, char *files[])
{
    dbp_multifile_reader_t *dbp;

    (void)event_buffer_size;
    (void)event_avail_space;

    dbp = open_files(nbfiles, files);

    return dbp;
}

void dbp_reader_destruct(dbp_multifile_reader_t *dbp)
{
    free( dbp->files );
    free( dbp );
}
