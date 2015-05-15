/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#if defined(DAGUE_PROFILING_USE_MMAP)
#include <sys/mman.h>
#endif
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>

#include "profiling.h"
#include "dbp.h"
#include "data_distribution.h"
#include "debug.h"
#include "dague/class/fifo.h"
#include "dague/dague_hwloc.h"
#include "dague/os-spec-timing.h"

#define min(a, b) ((a)<(b)?(a):(b))

#define MINIMAL_EVENT_BUFFER_SIZE          (10*sysconf(_SC_PAGESIZE))
#ifndef HOST_NAME_MAX
#if defined(MAC_OS_X)
#define HOST_NAME_MAX _SC_HOST_NAME_MAX
#else
#define HOST_NAME_MAX 1024
#endif  /* defined(MAC_OS_X) */
#endif /* defined(HOST_NAME_MAX) */

#ifndef HOST_NAME_MAX
#if defined(MAC_OS_X)
#define HOST_NAME_MAX _SC_HOST_NAME_MAX
#else
#define HOST_NAME_MAX 1024
#endif  /* defined(MAC_OS_X) */
#endif /* defined(HOST_NAME_MAX) */

/**
 * Externally visible on/off switch for the profiling of new events. It
 * only protects the macros, a direct call to the dague_profiling_trace
 * will always succeed. It is automatically turned on by the init call.
 */
int dague_profile_enabled = 0;
static int __profile_initialized = 0;  /* not initialized */

static dague_profiling_buffer_t *allocate_empty_buffer(off_t *offset, char type);

/* Process-global dictionary */
static unsigned int dague_prof_keys_count, dague_prof_keys_number;
static dague_profiling_key_t* dague_prof_keys;

static int __already_called = 0;
static dague_time_t dague_start_time;
static int          start_called = 0;

/* Process-global profiling list */
static dague_list_t threads;
static char *hr_id = NULL;
static dague_profiling_info_t *dague_profiling_infos = NULL;

static char *dague_profiling_last_error = NULL;
static int   dague_profiling_raise_error = 0;

/* File backend globals. */
static pthread_mutex_t file_backend_lock = PTHREAD_MUTEX_INITIALIZER;
static off_t file_backend_next_offset = 0;
static int   file_backend_fd = -1;

/* File backend constants, computed at init time */
static size_t event_buffer_size = 0;
static size_t event_avail_space = 0;
static int file_backend_extendable;

static dague_profiling_binary_file_header_t *profile_head = NULL;
static char *bpf_filename = NULL;
static pthread_key_t thread_specific_profiling_key;

static void set_last_error(const char *format, ...)
{
    va_list ap;
    if( dague_profiling_last_error )
        free(dague_profiling_last_error);
    va_start(ap, format);
    vasprintf(&dague_profiling_last_error, format, ap);
    va_end(ap);
    dague_profiling_raise_error = 1;
}
static int switch_event_buffer(dague_thread_profiling_t *context);

char *dague_profiling_strerror(void)
{
    return dague_profiling_last_error;
}

void dague_profiling_add_information( const char *key, const char *value )
{
    dague_profiling_info_t *n;
    n = (dague_profiling_info_t *)calloc(1, sizeof(dague_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = dague_profiling_infos;
    dague_profiling_infos = n;
}

void dague_profiling_thread_add_information(dague_thread_profiling_t * thread,
                                            const char *key, const char *value )
{
    dague_profiling_info_t *n;
    n = (dague_profiling_info_t *)calloc(1, sizeof(dague_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = thread->infos;
    thread->infos = n;
}

int dague_profiling_init( void )
{
    dague_profiling_buffer_t dummy_events_buffer;
    long ps;

    if( __profile_initialized ) return -1;

    pthread_key_create(&thread_specific_profiling_key, NULL);

    OBJ_CONSTRUCT( &threads, dague_list_t );

    dague_prof_keys = (dague_profiling_key_t*)calloc(128, sizeof(dague_profiling_key_t));
    dague_prof_keys_count = 0;
    dague_prof_keys_number = 128;

    file_backend_extendable = 1;
    ps = sysconf(_SC_PAGESIZE);
    event_buffer_size = ps * ((MINIMAL_EVENT_BUFFER_SIZE + ps) / ps);
    event_avail_space = event_buffer_size -
        ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);

    assert( sizeof(dague_profiling_binary_file_header_t) < event_buffer_size );

    /* default start time is time of call of profiling init.
     * Can be reset once explicitly by the user. */
    dague_profiling_start();
    /**
     * As we called the _start function automatically, the timing will be
     * based on this moment. By forcing back the __already_called to 0, we
     * allow the caller to decide when to rebase the timing in case there
     * is a need.
     */
    __already_called = 0;
    dague_profile_enabled = 1;  /* turn on the profiling */

    /* add the hostname, for the sake of explicit profiling */
    char buf[HOST_NAME_MAX];
    if (0 == gethostname(buf, HOST_NAME_MAX))
        dague_profiling_add_information("hostname", buf);
    else
        dague_profiling_add_information("hostname", "");

    /* the current working directory may also be helpful */
    char * newcwd = NULL;
    int bufsize = HOST_NAME_MAX;
    errno = 0;
    char * cwd = getcwd(buf, bufsize);
    while (cwd == NULL && errno == ERANGE) {
        bufsize *= 2;
        cwd = realloc(cwd, bufsize);
        if (cwd == NULL)            /* failed  - just give up */
            break;
        errno = 0;
        newcwd = getcwd(cwd, bufsize);
        if (newcwd == NULL) {
            free(cwd);
            cwd = NULL;
        }
    }
    if (cwd != NULL) {
        dague_profiling_add_information("cwd", cwd);
        if (cwd != buf)
            free(cwd);
    } else
        dague_profiling_add_information("cwd", "");

    __profile_initialized = 1; //* confirmed */
    return 0;
}

void dague_profiling_start(void)
{
    if(start_called)
        return;

#if defined(HAVE_MPI)
    {
        int flag;
        (void)MPI_Initialized(&flag);
        if(flag) MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    start_called = 1;
    dague_start_time = take_time();
}

dague_thread_profiling_t *dague_profiling_thread_init( size_t length, const char *format, ...)
{
    va_list ap;
    dague_thread_profiling_t *res;
    int rc; (void)rc;

    if( !__profile_initialized ) return NULL;
    if( -1 == file_backend_fd ) {
        set_last_error("Profiling system: dague_profiling_thread_init: call before dague_profiling_dbp_start");
        return NULL;
    }
    /** Remark: maybe calloc would be less perturbing for the measurements,
     *  if we consider that we don't care about the _init phase, but only
     *  about the measurement phase that happens later.
     */
    res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) + length );
    if( NULL == res ) {
        set_last_error("Profiling system: dague_profiling_thread_init: unable to allocate %u bytes", length);
        fprintf(stderr, "*** %s\n", dague_profiling_strerror());
        return NULL;
    }
    pthread_setspecific(thread_specific_profiling_key, res);

    OBJ_CONSTRUCT(res, dague_list_item_t);
    va_start(ap, format);
    rc = vasprintf(&res->hr_id, format, ap); assert(rc!=-1);
    va_end(ap);

    assert( event_buffer_size != 0 );
    /* To trigger a buffer allocation at first creation of an event */
    res->next_event_position = event_buffer_size;
    res->nb_events = 0;

    res->infos = NULL;

    res->first_events_buffer_offset = (off_t)-1;
    res->current_events_buffer = NULL;

    dague_list_fifo_push( &threads, (dague_list_item_t*)res );

    /* Allocate the first page to save time on the first event tracing */
    switch_event_buffer(res);

    return res;
}

int dague_profiling_fini( void )
{
    dague_thread_profiling_t *t;

    if( !__profile_initialized ) return -1;

    if( bpf_filename ) {
        if( 0 != dague_profiling_dbp_dump() ) {
            return -1;
        }
    }

    while( (t = (dague_thread_profiling_t*)dague_ulist_fifo_pop(&threads)) ) {
        free(t->hr_id);
        free(t);
    }
    free(hr_id);
    OBJ_DESTRUCT(&threads);

    dague_profiling_dictionary_flush();
    free(dague_prof_keys);
    dague_prof_keys_number = 0;
    start_called = 0;  /* Allow the profiling to be reinitialized */
    dague_profile_enabled = 0;  /* turn off the profiling */
    __profile_initialized = 0;  /* not initialized */
    return 0;
}

int dague_profiling_reset( void )
{
    dague_thread_profiling_t *t;

    DAGUE_LIST_ITERATOR(&threads, it, {
        t = (dague_thread_profiling_t*)it;
        t->next_event_position = 0;
        /* TODO: should reset the backend file / recreate it */
    });

    return 0;
}

int dague_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end )
{
    unsigned int i;
    int pos = -1;

    if( !__profile_initialized ) return 0;
    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL == dague_prof_keys[i].name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(dague_prof_keys[i].name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        if( dague_prof_keys_count == dague_prof_keys_number ) {
            set_last_error("Profiling system: error: dague_profiling_add_dictionary_keyword: Number of keyword limits reached");
            return -1;
        }
        pos = dague_prof_keys_count;
        dague_prof_keys_count++;
    }

    dague_prof_keys[pos].name = strdup(key_name);
    dague_prof_keys[pos].attributes = strdup(attributes);
    dague_prof_keys[pos].info_length = info_length;
    if( NULL != convertor_code )
        dague_prof_keys[pos].convertor = strdup(convertor_code);
    else
        dague_prof_keys[pos].convertor = NULL;

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    return 0;
}


int dague_profiling_dictionary_flush( void )
{
    unsigned int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL != dague_prof_keys[i].name ) {
            free(dague_prof_keys[i].name);
            free(dague_prof_keys[i].attributes);
        }
    }
    dague_prof_keys_count = 0;

    return 0;
}

static dague_profiling_buffer_t *allocate_empty_buffer(off_t *offset, char type)
{
    dague_profiling_buffer_t *res;

    if( !file_backend_extendable ) {
        *offset = -1;
        return NULL;
    }

    if( ftruncate(file_backend_fd, file_backend_next_offset+event_buffer_size) == -1 ) {
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }

#if defined(DAGUE_PROFILING_USE_MMAP)
    res = mmap(NULL, event_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_backend_fd, file_backend_next_offset);

    if( MAP_FAILED == res ) {
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }
#else
    res = (dague_profiling_buffer_t*)malloc(event_buffer_size);
#if !defined(NDEBUG)
    memset(res, 0, event_buffer_size);
#endif
    if( NULL == res ) {
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }
#endif

    res->this_buffer_file_offset = file_backend_next_offset;

    *offset = file_backend_next_offset;
    file_backend_next_offset += event_buffer_size;

    if(PROFILING_BUFFER_TYPE_HEADER != type ) {
        res->next_buffer_file_offset = (off_t)-1;

        res->buffer_type = type;
        switch( type ) {
        case PROFILING_BUFFER_TYPE_EVENTS:
            res->this_buffer.nb_events = 0;
            break;
        case PROFILING_BUFFER_TYPE_DICTIONARY:
            res->this_buffer.nb_dictionary_entries = 0;
            break;
        case PROFILING_BUFFER_TYPE_THREAD:
            res->this_buffer.nb_threads = 0;
            break;
        case PROFILING_BUFFER_TYPE_GLOBAL_INFO:
            res->this_buffer.nb_infos = 0;
            break;
        }
    } else {
        assert( *offset == 0 );
    }

    return res;
}

static void write_down_existing_buffer(dague_profiling_buffer_t *buffer,
                                       size_t count)
{
    (void)count;
    if( NULL == buffer )
        return;
    assert( count > 0 );
    memset( &(buffer->buffer[count]), 0, event_avail_space - count );
#if defined(DAGUE_PROFILING_USE_MMAP)
    if( munmap(buffer, event_buffer_size) == -1 ) {
        fprintf(stderr, "Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                buffer, strerror(errno));
    }
#else
    if( lseek(file_backend_fd, buffer->this_buffer_file_offset, SEEK_SET) == (off_t)-1 ) {
        fprintf(stderr, "Warning profiling system: seek in the events backend file at %ld failed: %s. Events trace will be truncated.\n",
                (long)buffer->this_buffer_file_offset, strerror(errno));
    } else {
        if( (size_t)(write(file_backend_fd, buffer, event_buffer_size)) != event_buffer_size ) {
            fprintf(stderr, "Warning profiling system: write in the events backend file at %ld failed: %s. Events trace will be truncated.\n",
                     (long)buffer->this_buffer_file_offset, strerror(errno));
        }
    }
    free(buffer);
#endif
}

static int switch_event_buffer( dague_thread_profiling_t *context )
{
    dague_profiling_buffer_t *new_buffer;
    dague_profiling_buffer_t *old_buffer;
    off_t off;

    pthread_mutex_lock( &file_backend_lock );

    new_buffer = allocate_empty_buffer(&off, PROFILING_BUFFER_TYPE_EVENTS);

    if( NULL == new_buffer ) {
        pthread_mutex_unlock( &file_backend_lock );
        return -1;
    }

    old_buffer = context->current_events_buffer;
    if( NULL == old_buffer ) {
        context->first_events_buffer_offset = off;
    } else {
        old_buffer->next_buffer_file_offset = off;
    }
    write_down_existing_buffer( old_buffer, context->next_event_position );

    context->current_events_buffer = new_buffer;
    context->current_events_buffer_offset = off;
    context->next_event_position = 0;

    pthread_mutex_unlock( &file_backend_lock );

    return 0;
}

int
dague_profiling_trace_flags(dague_thread_profiling_t* context, int key,
                            uint64_t event_id, uint32_t handle_id,
                            void *info, uint16_t flags)
{
    dague_profiling_output_t *this_event;
    size_t this_event_length;
    dague_time_t now;

    if( -1 == file_backend_fd ) {
        return -1;
    }

    this_event_length = EVENT_LENGTH( key, (NULL != info) );
    assert( this_event_length < event_avail_space );
    if( context->next_event_position + this_event_length > event_avail_space ) {
        if( switch_event_buffer(context) == -1 ) {
            return -2;
        }
    }
    this_event = (dague_profiling_output_t *)&context->current_events_buffer->buffer[context->next_event_position];
    assert( context->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    context->current_events_buffer->this_buffer.nb_events++;

    context->next_event_position += this_event_length;
    context->nb_events++;

    this_event->event.key = (uint16_t)key;
    this_event->event.event_id = event_id;
    this_event->event.handle_id = handle_id;
    this_event->event.flags = 0;

    if( NULL != info ) {
        memcpy(this_event->info, info, dague_prof_keys[ BASE_KEY(key) ].info_length);
        this_event->event.flags = DAGUE_PROFILING_EVENT_HAS_INFO;
    }
    this_event->event.flags |= flags;
    now = take_time();
    this_event->event.timestamp = diff_time(dague_start_time, now);

    return 0;
}

int
dague_profiling_ts_trace_flags(int key,
                               uint64_t event_id, uint32_t object_id,
                               void *info, uint16_t flags)
{
    dague_thread_profiling_t* context = (dague_thread_profiling_t*)
        pthread_getspecific(thread_specific_profiling_key);
    return dague_profiling_trace_flags(context, key, event_id, object_id,
                                       info, flags);
}

int dague_profiling_ts_trace(int key, uint64_t event_id, uint32_t object_id, void *info)
{
    dague_thread_profiling_t* context = (dague_thread_profiling_t*)
        pthread_getspecific(thread_specific_profiling_key);
    return dague_profiling_trace_flags(context, key, event_id, object_id,
                                       info, 0);
}

static int64_t dump_global_infos(int *nbinfos)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_info_buffer_t *ib;
    dague_profiling_info_t *i;
    int nb, nbthis, is, vs;
    int pos, tc, vpos;
    off_t first_off;
    char *value;

    if( NULL == dague_profiling_infos ) {
        *nbinfos = 0;
        return -1;
    }

    b = allocate_empty_buffer(&first_off, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
    if( NULL == b ) {
        set_last_error("Profiling system: error: Unable to dump the global infos -- buffer allocation error\n");
        *nbinfos = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;
    for(i = dague_profiling_infos; i != NULL; i = i->next) {
        is = strlen(i->key);
        vs = strlen(i->value);

        if( pos + sizeof(dague_profiling_info_buffer_t) + is - 1 >= event_avail_space ) {
            b->this_buffer.nb_infos = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
            if( NULL == n ) {
                set_last_error("Profiling System: error: Global Infos will be truncated to %d infos only -- buffer allocation error\n", nb);
                *nbinfos = nb;
                return first_off;
            }

            write_down_existing_buffer(b, pos);

            b = n;
            pos = 0;
            nbthis = 0;
        }

        /* The key must fit in event_avail_space */
        if( sizeof(dague_profiling_info_buffer_t) + is - 1 > event_avail_space ) {
            set_last_error("Profiling System: error: Key of size %d does not fit in an entire file segment of %d bytes\n",
                           is, event_avail_space);
            nb++;
            continue;
        }

        nbthis++;

        ib = (dague_profiling_info_buffer_t *)&(b->buffer[pos]);
        ib->info_size = is;
        ib->value_size = vs;
        memcpy(ib->info_and_value + 0,  i->key,   is);

        pos += sizeof(dague_profiling_info_buffer_t) + is - 1;

        vpos = 0;
        value = ib->info_and_value + is;
        while( vpos < vs ) {
            tc = (int)(event_avail_space - pos) < (vs-vpos) ? (int)(event_avail_space - pos) : (vs-vpos);
            memcpy(value, i->value + vpos, tc);
            vpos += tc;
            pos += tc;
            if( pos == (int)event_avail_space ) {
                b->this_buffer.nb_infos = nbthis;
                n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
                if( NULL == n ) {
                    set_last_error("Profiling System: error: Global Infos will be truncated to %d infos only -- buffer allocation error\n", nb);
                    *nbinfos = nb;
                    return first_off;
                }

                write_down_existing_buffer(b, pos);

                b = n;
                pos = 0;
                nbthis = 0;
                value = (char*)&(b->buffer[pos]);
            }
        }
        nb++;
    }

    b->this_buffer.nb_infos = nbthis;
    write_down_existing_buffer(b, pos);

    *nbinfos = nb;
    return first_off;
}

static int64_t dump_dictionary(int *nbdico)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_key_buffer_t *kb;
    dague_profiling_key_t *k;
    unsigned int i;
    int nb, nbthis, cs, pos;
    off_t first_off;

    if( 0 == dague_prof_keys_count ) {
        *nbdico = 0;
        return -1;
    }

    b = allocate_empty_buffer(&first_off, PROFILING_BUFFER_TYPE_DICTIONARY);
    if( NULL == b ) {
        set_last_error("Profiling System: error: Unable to dump the dictionary -- buffer allocation error\n");
        *nbdico = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;
    for(i = 0; i < dague_prof_keys_count; i++) {
        k = &dague_prof_keys[i];
        if(NULL == k->convertor )
            cs = 0;
        else
            cs = strlen(k->convertor);

        if( pos + sizeof(dague_profiling_key_buffer_t) + cs - 1 >= event_avail_space ) {
            b->this_buffer.nb_dictionary_entries = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_DICTIONARY);
            if( NULL == n ) {
                set_last_error("Profiling system: error: Dictionary will be truncated to %d entries only -- buffer allocation error\n", nb);
                *nbdico = nb;
                return first_off;
            }

            write_down_existing_buffer(b, pos);

            b = n;
            pos = 0;
            nbthis = 0;

        }
        kb = (dague_profiling_key_buffer_t *)&(b->buffer[pos]);
        strncpy(kb->name, k->name, 64);
        strncpy(kb->attributes, k->attributes, 128);
        kb->keyinfo_length = k->info_length;
        kb->keyinfo_convertor_length = cs;
        if( cs > 0 ) {
            memcpy(kb->convertor, k->convertor, cs);
        }
        nb++;
        nbthis++;
        pos += sizeof(dague_profiling_key_buffer_t) + cs - 1;
    }

    b->this_buffer.nb_dictionary_entries = nbthis;
    write_down_existing_buffer(b, pos);

    *nbdico = nb;
    return first_off;
}

static size_t thread_size(dague_thread_profiling_t *thread)
{
    size_t s = 0;
    dague_profiling_info_t *i;
    int ks, vs;

    s += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
    for(i = thread->infos; NULL!=i; i = i->next) {
        ks = strlen(i->key);
        vs = strlen(i->value);
        if( s + ks + vs + sizeof(dague_profiling_info_buffer_t) - 1 > event_avail_space ) {
            set_last_error("Profiling system: warning: unable to save info %s of thread %s, info ignored\n",
                           i->key, thread->hr_id);
            continue;
        }
        s += ks + vs + sizeof(dague_profiling_info_buffer_t) - 1;
    }
    return s;
}

static int64_t dump_thread(int *nbth)
{
    dague_profiling_buffer_t *b, *n;
    dague_profiling_thread_buffer_t *tb;
    int nb, nbthis, nbinfos, ks, vs, pos;
    dague_profiling_info_t *i;
    dague_profiling_info_buffer_t *ib;
    off_t off;
    size_t th_size;
    dague_list_item_t *it;
    dague_thread_profiling_t* thread;

    if( dague_list_is_empty(&threads) ) {
        *nbth = 0;
        return -1;
    }

    b = allocate_empty_buffer(&off, PROFILING_BUFFER_TYPE_THREAD);
    if( NULL == b ) {
        set_last_error("Profiling system: error: Unable to dump some thread profiles -- buffer allocation error\n");
        *nbth = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;

    for(it = DAGUE_LIST_ITERATOR_FIRST( &threads );
        it != DAGUE_LIST_ITERATOR_END( &threads );
        it = DAGUE_LIST_ITERATOR_NEXT( it ) ) {
        thread = (dague_thread_profiling_t*)it;

        if(thread->nb_events == 0)
            continue; /** We don't store threads with no events at all */

        th_size = thread_size(thread);

        if( pos + th_size >= event_avail_space ) {
            b->this_buffer.nb_threads = nbthis;
            n = allocate_empty_buffer(&b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_THREAD);
            if( NULL == n ) {
                set_last_error("Profiling system: error: Threads will be truncated to %d threads only -- buffer allocation error\n", nb);
                *nbth = nb;
                return off;
            }

            write_down_existing_buffer(b, pos);

            b = n;
            pos = 0;
            nbthis = 0;
        }

        tb = (dague_profiling_thread_buffer_t *)&(b->buffer[pos]);
        tb->nb_events = thread->nb_events;
        strncpy(tb->hr_id, thread->hr_id, 128);
        tb->first_events_buffer_offset = thread->first_events_buffer_offset;

        nb++;
        nbthis++;

        nbinfos = 0;
        i = thread->infos;
        pos += sizeof(dague_profiling_thread_buffer_t) - sizeof(dague_profiling_info_buffer_t);
        while( NULL != i ) {
            ks = strlen(i->key);
            vs = strlen(i->value);
            if( pos + ks + vs + sizeof(dague_profiling_info_buffer_t) - 1 >= event_avail_space ) {
                continue;
            }
            ib = (dague_profiling_info_buffer_t*)&(b->buffer[pos]);
            ib->info_size = ks;
            ib->value_size = vs;
            memcpy(ib->info_and_value, i->key, ks);
            memcpy(ib->info_and_value + ks, i->value, vs);
            pos += ks + vs + sizeof(dague_profiling_info_buffer_t) - 1;
            i = i->next;
            nbinfos++;
        }
        tb->nb_infos = nbinfos;
    }

    b->this_buffer.nb_threads = nbthis;
    write_down_existing_buffer(b, pos);

    *nbth = nb;
    return off;
}

int dague_profiling_dbp_dump( void )
{
    int nb_threads = 0;
    dague_thread_profiling_t *t;
    int nb_infos, nb_dico;

    if( !__profile_initialized ) return 0;

    if( NULL == bpf_filename ) {
        set_last_error("Profiling system: User Error: dague_profiling_dbp_dump before dague_profiling_dbp_start()");
        return -1;
    }
    if( NULL == profile_head ) {
        set_last_error("Profiling system: User Error: dague_profiling_dbp_dump before dague_profiling_dbp_start()");
        return -1;
    }

    /* Flush existing events buffer, unconditionally */
    DAGUE_LIST_ITERATOR(&threads, it, {
        t = (dague_thread_profiling_t*)it;
        if( NULL != t->current_events_buffer && t->next_event_position != 0 ) {
            write_down_existing_buffer(t->current_events_buffer, t->next_event_position);
            t->current_events_buffer = NULL;
        }
    });

    profile_head->dictionary_offset = dump_dictionary(&nb_dico);
    profile_head->dictionary_size = nb_dico;

    profile_head->info_offset = dump_global_infos(&nb_infos);
    profile_head->info_size = nb_infos;

    profile_head->thread_offset = dump_thread(&nb_threads);
    profile_head->nb_threads = nb_threads;

    /* The head is now complete. Last flush. */
    write_down_existing_buffer((dague_profiling_buffer_t *)profile_head,
                               sizeof(dague_profiling_binary_file_header_t));

    /* Close the backend file */
    pthread_mutex_lock(&file_backend_lock);
    close(file_backend_fd);
    file_backend_fd = -1;
    file_backend_extendable = 0;
    free(bpf_filename);
    bpf_filename = NULL;
    pthread_mutex_unlock(&file_backend_lock);

    if( dague_profiling_raise_error )
        return -1;

    return 0;
}

/**
 * Globally decide on a filename for the profiling file based on the requested
 * basefile, followed by the rank and then by a 6 letter unique key (generated
 * by mkstemp). The 6 letter key is used by all participants to create profiling
 * files that can be matched together.
 *
 * The basename is always respected, even in the case where it points to another
 * directory.
 */
int dague_profiling_dbp_start( const char *basefile, const char *hr_info )
{
    int64_t zero;
    char *xmlbuffer;
    int rank = 0, worldsize = 1, buflen;
    int  min_fd;
#if defined(HAVE_MPI)
    char *unique_str;

    int MPI_ready;
    (void)MPI_Initialized(&MPI_ready);
    if(MPI_ready) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    }
#endif

    if( !__profile_initialized ) return -1;

    asprintf(&bpf_filename, "%s-%d.prof-XXXXXX", basefile, rank);

    if( rank == 0 ) {
        /**
         * The first process create the unique locally unique filename, and then
         * share it with every other participants. If such a file cannot be
         * created broacast an empty key to all other processes.
         */
        min_fd = file_backend_fd = mkstemp(bpf_filename);
        if( -1 == file_backend_fd ) {
            set_last_error("Profiling system: error: Unable to create backend file %s: %s. Events not logged.\n",
                           bpf_filename, strerror(errno));
            file_backend_extendable = 0;
            memset(bpf_filename, 0, strlen(bpf_filename));
        }
    }

#if defined(HAVE_MPI)
    if( worldsize > 1) {
        unique_str = bpf_filename + (strlen(bpf_filename) - 6);  /* pinpoint directly into the bpf_filename */

        MPI_Bcast(unique_str, 7, MPI_CHAR, 0, MPI_COMM_WORLD);
        if( 0 != rank ) {
            if( *unique_str != '\0') {
                file_backend_fd = open(bpf_filename, O_WRONLY | O_CREAT | O_TRUNC, 00600);
            }  /* else we are in the error propagation from the rank 0 */
        }
        MPI_Allreduce(&file_backend_fd, &min_fd, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
#endif
    if( -1 == min_fd ) {
        set_last_error("Profiling system: error: one (or more) process could not create the backend file. Events not logged.\n");
        if( -1 != file_backend_fd ) {
            close(file_backend_fd);
            unlink(bpf_filename);
        }
        free(bpf_filename);
        bpf_filename = NULL;
        file_backend_extendable = 0;
        file_backend_fd = -1;
        return -1;
    }

    /* Create the header of the profiling file */
    profile_head = (dague_profiling_binary_file_header_t*)allocate_empty_buffer(&zero, PROFILING_BUFFER_TYPE_HEADER);
    if( NULL == profile_head )
        return -1;

    memcpy(profile_head->magick, DAGUE_PROFILING_MAGICK, strlen(DAGUE_PROFILING_MAGICK) + 1);
    profile_head->byte_order = 0x0123456789ABCDEF;
    profile_head->profile_buffer_size = event_buffer_size;
    strncpy(profile_head->hr_id, hr_info, 128);
    profile_head->rank = rank;
    profile_head->worldsize = worldsize;

    /* Reset the error system */
    set_last_error("Profiling system: success");
    dague_profiling_raise_error = 0;

    /* It's fine to re-reset the event date: we're back with a zero-length event set */
    start_called = 0;

    if( dague_hwloc_export_topology(&buflen, &xmlbuffer) != -1 &&
        buflen > 0 ) {
        dague_profiling_add_information("HWLOC-XML", xmlbuffer);
        dague_hwloc_free_xml_buffer(xmlbuffer);
    }
    return 0;
}

uint64_t dague_profiling_get_time(void) {
    return diff_time(dague_start_time, take_time());
}

void dague_profiling_enable(void)
{
    dague_profile_enabled = 1;
}
void dague_profiling_disable(void)
{
    dague_profile_enabled = 0;
}

char *dague_profile_ddesc_key_to_string = "";
