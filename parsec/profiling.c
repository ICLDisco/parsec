/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <pthread.h>
#include <errno.h>
#if defined(PARSEC_PROFILING_USE_MMAP)
#include <sys/mman.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include "parsec/profiling.h"
#include "parsec/parsec_binary_profile.h"
#include "parsec/data_distribution.h"
#include "parsec/utils/debug.h"
#include "parsec/class/list.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"
#include "parsec/sys/tls.h"

#define min(a, b) ((a)<(b)?(a):(b))

#ifndef HOST_NAME_MAX
#if defined(PARSEC_OSX)
#define HOST_NAME_MAX _SC_HOST_NAME_MAX
#else
#define HOST_NAME_MAX 1024
#endif  /* defined(PARSEC_OSX) */
#endif /* defined(HOST_NAME_MAX) */

#ifndef HOST_NAME_MAX
#if defined(PARSEC_OSX)
#define HOST_NAME_MAX _SC_HOST_NAME_MAX
#else
#define HOST_NAME_MAX 1024
#endif  /* defined(PARSEC_OSX) */
#endif /* defined(HOST_NAME_MAX) */

int parsec_profile_enabled = 0;
static int __profile_initialized = 0;  /* not initialized */

/**
 * A freelist of buffers links buffers one after the other.
 *
 * Only the field next is used in the freelist;
 * However, buffers are larger than a single pointer: they
 * are of size event_buffer_size. Fields in the header may
 * be set at allocation / free time, so we set the field
 * next inside the data part (which is never used as long
 * as the buffer stays in the free list)
 */
typedef struct tl_freelist_buffer_s {
    parsec_profiling_buffer_t    buffer; /**< buffer head */
    struct tl_freelist_buffer_s *next;   /**< Next buffer */
} tl_freelist_buffer_t;

typedef struct tl_freelist_s {
    tl_freelist_buffer_t *first;
    pthread_mutex_t       lock;
    int                   nb_allocated;
} tl_freelist_t;
static int parsec_profiling_per_thread_buffer_freelist_min = 2;

static tl_freelist_t *default_freelist = NULL;
#define TLS_STORAGE(t)  ( (tl_freelist_t*)((t)->tls_storage) )

static parsec_profiling_buffer_t *allocate_empty_buffer(tl_freelist_t *fl, off_t *offset, char type);

/* Process-global dictionary */
static unsigned int parsec_prof_keys_count, parsec_prof_keys_number;
static parsec_profiling_key_t* parsec_prof_keys;

static int __already_called = 0;
static parsec_time_t parsec_start_time;
static int          start_called = 0;

/* Process-global profiling list */
static parsec_list_t threads;
static char *hr_id = NULL;
static parsec_profiling_info_t *parsec_profiling_infos = NULL;

#define MAX_PROFILING_ERROR_STRING_LEN 1024
static char  parsec_profiling_last_error[MAX_PROFILING_ERROR_STRING_LEN+1] = { '\0', };
static int   parsec_profiling_raise_error = 0;

/* File backend globals. */
static pthread_mutex_t file_backend_lock = PTHREAD_MUTEX_INITIALIZER;
static off_t  file_backend_next_offset = 0;
static size_t file_backend_size = 0;
static int    file_backend_fd = -1;

/* File backend constants, computed at init time */
static size_t event_buffer_size = 0;
static int    parsec_profiling_file_multiplier = 1;
static size_t event_avail_space = 0;
static int file_backend_extendable;

static parsec_profiling_binary_file_header_t *profile_head = NULL;
static char *bpf_filename = NULL;
PARSEC_TLS_DECLARE(tls_profiling);

static int parsec_profiling_show_profiling_performance = 0;
static parsec_profiling_perf_t parsec_profiling_global_perf[PERF_MAX];

#define do_and_measure_perf( perf_counter, code ) do {                  \
        parsec_time_t start, end;                                       \
        parsec_profiling_perf_t *pa;                                    \
        parsec_thread_profiling_t* tp;                                  \
        tp = PARSEC_TLS_GET_SPECIFIC(tls_profiling);                    \
        if( NULL == tp )                                                \
            pa = &parsec_profiling_global_perf[perf_counter];           \
        else                                                            \
            pa = &tp->thread_perf[perf_counter];                        \
        start = take_time();                                            \
        code;                                                           \
        end = take_time();                                              \
        pa->perf_time_spent += diff_time(start, end);                   \
        pa->perf_number_calls++;                                        \
    } while(0)

static off_t find_free_segment(void)
{
    off_t my_offset;
    do_and_measure_perf(PERF_WAITING,
      pthread_mutex_lock( &file_backend_lock ));
    if( file_backend_next_offset + event_buffer_size > file_backend_size ) {
        file_backend_size += parsec_profiling_file_multiplier * event_buffer_size;
        do_and_measure_perf(PERF_RESIZE,
          if( ftruncate(file_backend_fd, file_backend_size) == -1 ) {
              fprintf(stderr, "### Profiling: unable to resize backend file to %"PRIu64" bytes: %s\n",
                      (uint64_t)file_backend_size, strerror(errno));
              file_backend_extendable = 0;
              pthread_mutex_unlock(&file_backend_lock);
              assert(0);
              return (off_t)-1;
          });
    }
    my_offset = file_backend_next_offset;
    file_backend_next_offset += event_buffer_size;
    pthread_mutex_unlock(&file_backend_lock);
    return my_offset;
}

static parsec_profiling_buffer_t *do_allocate_new_buffer(void)
{
    parsec_profiling_buffer_t *res = NULL;
    off_t my_offset;

    my_offset = find_free_segment();
    if( -1 == my_offset ) {
        close(file_backend_fd);
        file_backend_fd = -1;
        fprintf(stderr, "### Profiling: Unable to find a free segment in backend file\n");
        file_backend_extendable = 0;
        return NULL;
    }

#if defined(PARSEC_PROFILING_USE_MMAP)
    do_and_measure_perf(PERF_MMAP,
      res = mmap(NULL, event_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_backend_fd, my_offset));
    res = (res == MAP_FAILED) ? NULL : res;
#else
    do_and_measure_perf(PERF_MALLOC,
      res = (parsec_profiling_buffer_t*)malloc(event_buffer_size));
#endif

    if( NULL == res ) {
        file_backend_extendable = 0;
        close(file_backend_fd);
        file_backend_fd = -1;
        fprintf(stderr, "### Profiling: Unable to allocate / map segment in backend file (%s)n", strerror(errno));
        return NULL;
    }

#if 0 && !defined(NDEBUG)
    do_and_measure_perf(PERF_MEMSET,
      memset(res, 0, event_buffer_size));
#endif

    res->this_buffer_file_offset = my_offset;
    res->next_buffer_file_offset = (off_t)-1;

    return res;
}

#if !defined(PARSEC_PROFILING_USE_MMAP)
static void do_assign_buffer_to_free_segment(parsec_profiling_buffer_t *res)
{
    off_t my_offset = find_free_segment();
#if 0 && !defined(NDEBUG)
    do_and_measure_perf(PERF_MEMSET,
      memset(res, 0, event_buffer_size));
#endif
    res->this_buffer_file_offset = my_offset;
    res->next_buffer_file_offset = (off_t)-1;
}
#endif

static parsec_profiling_buffer_t *allocate_from_freelist(tl_freelist_t *fl)
{
    tl_freelist_buffer_t *head;
    parsec_profiling_buffer_t *res = NULL;

    if( !file_backend_extendable ) {
        return NULL;
    }

    pthread_mutex_lock(&fl->lock);
    head = fl->first;
    if(NULL != head) {
        fl->first = head->next;
        pthread_mutex_unlock(&fl->lock);
        res = (parsec_profiling_buffer_t*)head;
    } else {
        fl->nb_allocated++;
        pthread_mutex_unlock(&fl->lock);
        res = do_allocate_new_buffer();
    }
    return res;
}

static void free_to_freelist(tl_freelist_t *fl, parsec_profiling_buffer_t *b)
{
#if !defined(PARSEC_PROFILING_USE_MMAP)
    int ret;
#endif
    if( NULL == b )
        return;

#if defined(PARSEC_PROFILING_USE_MMAP)
    do_and_measure_perf(PERF_MUNMAP,
      if( munmap(b, event_buffer_size) == -1 ) {
          fprintf(stderr, "Warning profiling system: unmap of the events backend file at %p failed: %s\n",
                  b, strerror(errno));
      });
    b = do_allocate_new_buffer();
    if( NULL == b )
        return;
#else
    do_and_measure_perf(PERF_WAITING,
      pthread_mutex_lock( &file_backend_lock ));
    do_and_measure_perf(PERF_LSEEK,
      ret = lseek(file_backend_fd, b->this_buffer_file_offset, SEEK_SET));
    if(ret == (off_t)-1 ) {
        fprintf(stderr, "Warning profiling system: seek in the events backend file at %ld failed: %s. Events trace will be truncated.\n",
                (long)b->this_buffer_file_offset, strerror(errno));
    } else {
        do_and_measure_perf(PERF_WRITE,
           ret = write(file_backend_fd, b, event_buffer_size));
        if( (size_t)(ret) != event_buffer_size ) {
            fprintf(stderr, "Warning profiling system: write in the events backend file at %ld failed: %s. Events trace will be truncated.\n",
                     (long)b->this_buffer_file_offset, strerror(errno));
        }
    }
    pthread_mutex_unlock( &file_backend_lock );
    do_assign_buffer_to_free_segment(b);
#endif

    pthread_mutex_lock(&fl->lock);
    ((tl_freelist_buffer_t *)b)->next = fl->first;
    fl->first = (tl_freelist_buffer_t*)b;
    pthread_mutex_unlock(&fl->lock);
}


#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
/**
 * Profiling I/O command structure
 *
 * Command is coded in the buffer address:
 *   if buffer == IO_CMD_STOP, the helper thread is requested to leave.
 *       No command is supposed to be enqueued after this command.
 *   if buffer == IO_CMD_FLUSH, the helper thread will increment
 *       the io_cmd_flush_counter, and signal this on the
 *       io_cmd_flush_cond. This means that all I/O operations
 *       enqueued prior to that command have been exectued.
 *   For all other values of buffer:
 *       buffer should be unmapped
 *       a new buffer should be mapped into the backend file
 *       that new buffer should be added to the buffers freelist of fl
 */
typedef struct io_cmd_s {
    struct io_cmd_s           *next;
    parsec_profiling_buffer_t *buffer;
    tl_freelist_t             *fl;
} io_cmd_t;
#define IO_CMD_STOP   ((void*)-1)
#define IO_CMD_FLUSH  ((void*) 0)

typedef struct io_cmd_queue_s {
    io_cmd_t *next;
    io_cmd_t *last;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} io_cmd_queue_t;
static io_cmd_queue_t cmd_queue;
static io_cmd_queue_t free_cmd_queue;
static pthread_t io_helper_thread_id;

static int             io_cmd_flush_counter;
static pthread_mutex_t io_cmd_flush_mutex;
static pthread_cond_t  io_cmd_flush_cond;

static io_cmd_t *io_cmd_allocate(void)
{
    io_cmd_t *cmd;
    pthread_mutex_lock(&free_cmd_queue.lock);
    if( free_cmd_queue.next == NULL ) {
        pthread_mutex_unlock(&free_cmd_queue.lock);
        cmd = (io_cmd_t*)malloc(sizeof(io_cmd_t));
        return cmd;
    } else {
        cmd = free_cmd_queue.next;
        free_cmd_queue.next = cmd->next;
        pthread_mutex_unlock(&free_cmd_queue.lock);
        return cmd;
    }
}

static void io_cmd_free(io_cmd_t *cmd)
{
    pthread_mutex_lock(&free_cmd_queue.lock);
    cmd->next = free_cmd_queue.next;
    free_cmd_queue.next = cmd;
    pthread_mutex_unlock(&free_cmd_queue.lock);
}

static void io_cmd_queue_init(io_cmd_queue_t *queue)
{
    queue->next = NULL;
    queue->last = NULL;
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->cond, NULL);

}

static void io_cmd_queue_destroy(io_cmd_queue_t *queue)
{
    io_cmd_t *cmd;
    pthread_mutex_lock(&queue->lock);
    while( NULL != queue->next ) {
        cmd = queue->next;
        queue->next = cmd->next;
        free(cmd);
    }
    queue->last = NULL;
    pthread_mutex_unlock(&queue->lock);
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->cond);
}

static void *io_helper_thread_fct(void *_)
{
    int stop = 0;
    io_cmd_t *cmd;
    (void)_;

    while( stop == 0 ) {
        pthread_mutex_lock(&cmd_queue.lock);
        while( NULL == cmd_queue.next ) {
            pthread_cond_wait(&cmd_queue.cond, &cmd_queue.lock);
        }
        cmd = cmd_queue.next;
        if( cmd_queue.next == cmd_queue.last )
            cmd_queue.last = NULL;
        cmd_queue.next = cmd->next;
        pthread_mutex_unlock(&cmd_queue.lock);

        if( IO_CMD_FLUSH == cmd->buffer ) {
            pthread_mutex_lock(&io_cmd_flush_mutex);
            io_cmd_flush_counter++;
            pthread_cond_signal(&io_cmd_flush_cond);
            pthread_mutex_unlock(&io_cmd_flush_mutex);
        } else if( IO_CMD_STOP == cmd->buffer ) {
            stop = 1;
        } else {
            free_to_freelist(cmd->fl, cmd->buffer);
        }

        io_cmd_free(cmd);
    }

    io_cmd_queue_destroy(&cmd_queue);
    io_cmd_queue_destroy(&free_cmd_queue);

    return NULL;
}

static void io_helper_thread_init(void)
{
    io_cmd_queue_init(&cmd_queue);
    io_cmd_queue_init(&free_cmd_queue);
    io_cmd_flush_counter = 0;
    pthread_mutex_init(&io_cmd_flush_mutex, NULL);
    pthread_cond_init(&io_cmd_flush_cond, NULL);

    pthread_create(&io_helper_thread_id, NULL, io_helper_thread_fct, NULL);
}
#endif /* PARSEC_PROFILING_USE_HELPER_THREAD */


static void set_last_error(const char *format, ...)
{
    va_list ap;
    int rc;
    va_start(ap, format);
    rc = vsnprintf(parsec_profiling_last_error, MAX_PROFILING_ERROR_STRING_LEN, format, ap);
    va_end(ap);
    fprintf(stderr, "-- %s", parsec_profiling_last_error);
    parsec_profiling_raise_error = 1;
    (void)rc;
}
static int switch_event_buffer(parsec_thread_profiling_t *context);

char *parsec_profiling_strerror(void)
{
    return parsec_profiling_last_error;
}

void parsec_profiling_add_information( const char *key, const char *value )
{
    parsec_profiling_info_t *n;
    n = (parsec_profiling_info_t *)calloc(1, sizeof(parsec_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = parsec_profiling_infos;
    parsec_profiling_infos = n;
}

void parsec_profiling_thread_add_information(parsec_thread_profiling_t * thread,
                                            const char *key, const char *value )
{
    parsec_profiling_info_t *n;
    n = (parsec_profiling_info_t *)calloc(1, sizeof(parsec_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = thread->infos;
    thread->infos = n;
}

int parsec_profiling_init( void )
{
    parsec_profiling_buffer_t dummy_events_buffer;
    long ps;
    int parsec_profiling_minimal_ebs;

    if( __profile_initialized ) return -1;

    PARSEC_TLS_KEY_CREATE(tls_profiling);

    OBJ_CONSTRUCT( &threads, parsec_list_t );

    parsec_prof_keys_count = 0;
    parsec_prof_keys_number = 128;
    parsec_prof_keys = (parsec_profiling_key_t*)calloc(parsec_prof_keys_number, sizeof(parsec_profiling_key_t));

    file_backend_extendable = 1;
    ps = sysconf(_SC_PAGESIZE);

    parsec_profiling_minimal_ebs = 1;
    parsec_mca_param_reg_int_name("profile", "buffer_pages", "Number of pages per profiling buffer"
                                 "(default is 1, must be at least large enough to hold the binary file header)",
                                 false, false, parsec_profiling_minimal_ebs, &parsec_profiling_minimal_ebs);
    parsec_mca_param_reg_int_name("profile", "file_resize", "Number of buffers per file resize"
                                 "(default is 1)",
                                 false, false, parsec_profiling_file_multiplier, &parsec_profiling_file_multiplier);
    parsec_mca_param_reg_int_name("profile", "show_profiling_performance", "Print profiling performance at the end of the execution"
                                      "(default is no/0)",
                                      false, false, parsec_profiling_show_profiling_performance, &parsec_profiling_show_profiling_performance);
    if( parsec_profiling_minimal_ebs <= 0 )
        parsec_profiling_minimal_ebs = 10;
    if( parsec_profiling_file_multiplier <= 0 )
        parsec_profiling_file_multiplier = 1;

    event_buffer_size = parsec_profiling_minimal_ebs*ps;
    while( event_buffer_size < sizeof(parsec_profiling_binary_file_header_t) ){
        parsec_profiling_minimal_ebs++;
        event_buffer_size = parsec_profiling_minimal_ebs*ps;
    }

    event_avail_space = event_buffer_size -
        ( (char*)&dummy_events_buffer.buffer[0] - (char*)&dummy_events_buffer);

    assert( sizeof(parsec_profiling_binary_file_header_t) < event_buffer_size );

    /* As we called the _start function automatically, the timing will be
     * based on this moment. By forcing back the __already_called to 0, we
     * allow the caller to decide when to rebase the timing in case there
     * is a need.
     */
    __already_called = 0;
    parsec_profile_enabled = 1;  /* turn on the profiling */

    /* add the hostname, for the sake of explicit profiling */
    char buf[HOST_NAME_MAX];
    if (0 == gethostname(buf, HOST_NAME_MAX))
        parsec_profiling_add_information("hostname", buf);
    else
        parsec_profiling_add_information("hostname", "");

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
        parsec_profiling_add_information("cwd", cwd);
        if (cwd != buf)
            free(cwd);
    } else
        parsec_profiling_add_information("cwd", "");

#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
    io_helper_thread_init();
#endif

    __profile_initialized = 1; //* confirmed */
    return 0;
}

void parsec_profiling_start(void)
{
    if(start_called)
        return;

#if defined(PARSEC_HAVE_MPI)
    {
        int flag;
        (void)MPI_Initialized(&flag);
        if(flag) MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    start_called = 1;
    parsec_start_time = take_time();
}

parsec_thread_profiling_t *parsec_profiling_thread_init( size_t length, const char *format, ...)
{
    va_list ap;
    parsec_thread_profiling_t *res;
    int rc;

    if( !__profile_initialized ) return NULL;
    if( -1 == file_backend_fd ) {
        set_last_error("Profiling system: parsec_profiling_thread_init: call before parsec_profiling_dbp_start");
        return NULL;
    }
    /*  Remark: maybe calloc would be less perturbing for the measurements,
     *  if we consider that we don't care about the _init phase, but only
     *  about the measurement phase that happens later.
     */
    res = (parsec_thread_profiling_t*)malloc( sizeof(parsec_thread_profiling_t) + length );
    if( NULL == res ) {
        set_last_error("Profiling system: parsec_profiling_thread_init: unable to allocate %u bytes", length);
        fprintf(stderr, "*** %s\n", parsec_profiling_strerror());
        return NULL;
    }
    PARSEC_TLS_SET_SPECIFIC(tls_profiling, res);

    res->tls_storage = malloc(sizeof(tl_freelist_t));
    tl_freelist_t *t_fl = (tl_freelist_t*)res->tls_storage;
    tl_freelist_buffer_t *e;
    pthread_mutex_init(&t_fl->lock, NULL);
    e = (tl_freelist_buffer_t*)do_allocate_new_buffer();
    if( NULL == e ) {
        free(res->tls_storage);
        free(res);
        return NULL;
    }
    e->next = NULL;
    t_fl->first = e;
    t_fl->nb_allocated = 1;
    for(rc = 1; rc < parsec_profiling_per_thread_buffer_freelist_min; rc++) {
        e->next = (tl_freelist_buffer_t*)do_allocate_new_buffer();
        assert(NULL != e->next);
        e = e->next;
        e->next = NULL;
        t_fl->nb_allocated++;
    }

    OBJ_CONSTRUCT(res, parsec_list_item_t);
    va_start(ap, format);
    rc = vasprintf(&res->hr_id, format, ap); assert(rc!=-1); (void)rc;
    va_end(ap);

    assert( event_buffer_size != 0 );
    /* To trigger a buffer allocation at first creation of an event */
    res->next_event_position = event_buffer_size;
    res->nb_events = 0;

    res->infos = NULL;

    res->first_events_buffer_offset = (off_t)-1;
    res->current_events_buffer = NULL;

    parsec_list_push_back( &threads, (parsec_list_item_t*)res );

    /* Allocate the first page to save time on the first event tracing */
    switch_event_buffer(res);

    memset(res->thread_perf, 0, sizeof(parsec_profiling_perf_t)*PERF_MAX);

    return res;
}

int parsec_profiling_fini( void )
{
    parsec_thread_profiling_t *t;
    int i;
        
    if( !__profile_initialized ) return -1;

    if( bpf_filename ) {
        if( 0 != parsec_profiling_dbp_dump() ) {
            return -1;
        }
    }

    while( (t = (parsec_thread_profiling_t*)parsec_list_nolock_pop_front(&threads)) ) {
        tl_freelist_t *fl = TLS_STORAGE(t);
        tl_freelist_buffer_t *b;
        while(fl->first != NULL) {
            b = fl->first;
            fl->first = b->next;
            free(b);
        }
        if( parsec_profiling_show_profiling_performance ) {
            for(i = 0; i < PERF_MAX; i++) {
                parsec_profiling_global_perf[i].perf_time_spent += t->thread_perf[i].perf_time_spent;
                parsec_profiling_global_perf[i].perf_number_calls += t->thread_perf[i].perf_number_calls;
            }
        }

        pthread_mutex_destroy(&fl->lock);
        free(fl);
        free(t->hr_id);
        free(t);
    }
    free(hr_id);
    OBJ_DESTRUCT(&threads);

#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
    io_cmd_t *cmd = io_cmd_allocate();
    cmd->buffer = IO_CMD_STOP;
    cmd->fl = NULL;
    cmd->next = NULL;
    pthread_mutex_lock(&cmd_queue.lock);
    if( NULL == cmd_queue.last ) {
        cmd_queue.last = cmd_queue.next = cmd;
    } else {
        cmd_queue.last->next = cmd;
        cmd_queue.last = cmd;
    }
    pthread_cond_signal(&cmd_queue.cond);
    pthread_mutex_unlock(&cmd_queue.lock);
    pthread_join(io_helper_thread_id, NULL);
#endif
    
    if( parsec_profiling_show_profiling_performance ) {
        int rank = 0;
#if defined(PARSEC_HAVE_MPI)
        int MPI_ready;
        (void)MPI_Initialized(&MPI_ready);
        if(MPI_ready) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
#endif
        parsec_profiling_perf_t *pa = parsec_profiling_global_perf;
        char *ti;
#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
        ti = "Helper Thread:";
#else
        ti = "";
#endif
        fprintf(stderr,
                "### Profiling Performance on rank %d\n"
                "#   Buffer Size: %lu bytes\n"
                "#   File Resize Size: %lu bytes (%d buffers)\n"
#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
                "#   User Thread: Time spent waiting to append command to a queue: %"PRIu64" %s. Number of calls: %u\n"
#endif
                "#   %sTime Spent Resizing the backend file: %"PRIu64" %s. Number of resize: %u\n"
#if defined(PARSEC_PROFILING_USE_MMAP)
                "#   %sTime Spent Mapping the backend file: %"PRIu64" %s. Number of mmap: %u\n"
                "#   %sTime Spent Unmapping mapped buffers: %"PRIu64" %s. Number of munmap: %u\n"
#else
                "#   %sTime Spent Allocating Buffers: %"PRIu64" %s. Number of malloc: %u\n"
                "#   %sTime Spent Freeing Buffers: %"PRIu64" %s. Number of free: %u\n"
                "#   %sTime Spent Seeking in file: %"PRIu64" %s. Number of lseeks: %u\n"
                "#   %sTime Spent Writing (synchronously) Buffers: %"PRIu64" %s. Number of writes: %u\n"
#endif
                "#   %sTime Spent Resetting Buffers to 0: %"PRIu64" %s. Number of memset: %u\n"
                "#   %sTime spent waiting for Exclusive Access to Buffer Management: %"PRIu64" %s. Number of calls: %u\n",
                rank,
                event_buffer_size,
                event_buffer_size * parsec_profiling_file_multiplier, parsec_profiling_file_multiplier,
#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
                pa[PERF_USER_WAITING].perf_time_spent, TIMER_UNIT, pa[PERF_USER_WAITING].perf_number_calls,
#endif
                ti, pa[PERF_RESIZE].perf_time_spent, TIMER_UNIT, pa[PERF_RESIZE].perf_number_calls,
#if defined(PARSEC_PROFILING_USE_MMAP)
                ti, pa[PERF_MMAP].perf_time_spent, TIMER_UNIT, pa[PERF_MMAP].perf_number_calls,
                ti, pa[PERF_MUNMAP].perf_time_spent, TIMER_UNIT, pa[PERF_MUNMAP].perf_number_calls,
#else
                ti, pa[PERF_MALLOC].perf_time_spent, TIMER_UNIT, pa[PERF_MALLOC].perf_number_calls,
                ti, pa[PERF_FREE].perf_time_spent, TIMER_UNIT, pa[PERF_FREE].perf_number_calls,
                ti, pa[PERF_LSEEK].perf_time_spent, TIMER_UNIT, pa[PERF_LSEEK].perf_number_calls,
                ti, pa[PERF_WRITE].perf_time_spent, TIMER_UNIT, pa[PERF_WRITE].perf_number_calls,
#endif
                ti, pa[PERF_MEMSET].perf_time_spent, TIMER_UNIT, pa[PERF_MEMSET].perf_number_calls,
                ti, pa[PERF_WAITING].perf_time_spent, TIMER_UNIT, pa[PERF_WAITING].perf_number_calls);
    }
    memset(parsec_profiling_global_perf, 0, sizeof(parsec_profiling_perf_t)*PERF_MAX);

    while(default_freelist->first != NULL) {
        tl_freelist_buffer_t *b = default_freelist->first;
        default_freelist->first = b->next;
        free(b);
    }
    pthread_mutex_destroy(&default_freelist->lock);
    free(default_freelist);

    parsec_profiling_dictionary_flush();
    free(parsec_prof_keys);
    parsec_prof_keys_number = 0;
    start_called = 0;  /* Allow the profiling to be reinitialized */
    parsec_profile_enabled = 0;  /* turn off the profiling */
    __profile_initialized = 0;  /* not initialized */

    return 0;
}

int parsec_profiling_reset( void )
{
    parsec_thread_profiling_t *t;

    PARSEC_LIST_ITERATOR(&threads, it, {
        t = (parsec_thread_profiling_t*)it;
        t->next_event_position = 0;
        /* TODO: should reset the backend file / recreate it */
    });

    return 0;
}

int parsec_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end )
{
    unsigned int i;
    int pos = -1;

    if( !__profile_initialized ) return 0;
    for( i = 0; i < parsec_prof_keys_count; i++ ) {
        if( NULL == parsec_prof_keys[i].name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(parsec_prof_keys[i].name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        if( parsec_prof_keys_count == parsec_prof_keys_number ) {
            set_last_error("Profiling system: error: parsec_profiling_add_dictionary_keyword: Number of keyword limits reached");
            return -1;
        }
        pos = parsec_prof_keys_count;
        parsec_prof_keys_count++;
    }

    parsec_prof_keys[pos].name = strdup(key_name);
    parsec_prof_keys[pos].attributes = strdup(attributes);
    parsec_prof_keys[pos].info_length = info_length;
    if( NULL != convertor_code )
        parsec_prof_keys[pos].convertor = strdup(convertor_code);
    else
        parsec_prof_keys[pos].convertor = NULL;

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    return 0;
}


int parsec_profiling_dictionary_flush( void )
{
    unsigned int i;

    for( i = 0; i < parsec_prof_keys_count; i++ ) {
        if( NULL != parsec_prof_keys[i].name ) {
            free(parsec_prof_keys[i].name);
            free(parsec_prof_keys[i].attributes);
            if( NULL != parsec_prof_keys[i].convertor ) {
                free(parsec_prof_keys[i].convertor);
            }
        }
    }
    parsec_prof_keys_count = 0;

    return 0;
}

static parsec_profiling_buffer_t *allocate_empty_buffer(tl_freelist_t *fl, off_t *offset, char type)
{
    parsec_profiling_buffer_t *res;

    res = allocate_from_freelist(fl);
    if( NULL == res ) {
        file_backend_extendable = 0;
        *offset = -1;
        return NULL;
    }

    *offset = res->this_buffer_file_offset;

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

static void write_down_existing_buffer(tl_freelist_t *fl,
                                       parsec_profiling_buffer_t *buffer,
                                       size_t count)
{
    if( NULL == buffer )
        return;

    do_and_measure_perf(PERF_MEMSET,
      memset( &(buffer->buffer[count]), 0, event_avail_space - count ));

#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
    io_cmd_t *cmd = io_cmd_allocate();
    cmd->buffer = buffer;
    cmd->fl = fl;
    cmd->next = NULL;
    do_and_measure_perf(PERF_USER_WAITING,
       pthread_mutex_lock(&cmd_queue.lock));
    if( NULL == cmd_queue.last ) {
        cmd_queue.last = cmd_queue.next = cmd;
    } else {
        cmd_queue.last->next = cmd;
        cmd_queue.last = cmd;
    }
    pthread_cond_signal(&cmd_queue.cond);
    pthread_mutex_unlock(&cmd_queue.lock);
#else
    free_to_freelist(fl, buffer);
#endif
}

static int switch_event_buffer( parsec_thread_profiling_t *context )
{
    parsec_profiling_buffer_t *new_buffer;
    parsec_profiling_buffer_t *old_buffer;
    off_t off;

    new_buffer = allocate_empty_buffer((tl_freelist_t*)context->tls_storage, &off, PROFILING_BUFFER_TYPE_EVENTS);

    old_buffer = context->current_events_buffer;
    if( NULL == old_buffer ) {
        context->first_events_buffer_offset = off;
    } else {
        old_buffer->next_buffer_file_offset = off;
    }
    write_down_existing_buffer((tl_freelist_t*)context->tls_storage, old_buffer, context->next_event_position );

    context->current_events_buffer = new_buffer;
    context->current_events_buffer_offset = off;
    context->next_event_position = 0;

    return 0;
}

int parsec_profiling_ts_trace_flags(int key, uint64_t event_id, uint32_t taskpool_id,
                                    void *info, uint16_t flags )
{
    parsec_thread_profiling_t* ctx;

    if( (-1 == file_backend_fd) || (!start_called) ) {
        return -1;
    }

    ctx = PARSEC_TLS_GET_SPECIFIC(tls_profiling);
    if( NULL != ctx )
        return parsec_profiling_trace_flags(ctx, key, event_id, taskpool_id, info, flags);

    set_last_error("Profiling system: error: called parsec_profiling_ts_trace_flags"
                   " from a thread that did not call parsec_profiling_thread_init\n");
    return -1;
}

int
parsec_profiling_trace_flags(parsec_thread_profiling_t* context, int key,
                            uint64_t event_id, uint32_t taskpool_id,
                            void *info, uint16_t flags)
{
    parsec_profiling_output_t *this_event;
    size_t this_event_length;
    parsec_time_t now;

    if( (-1 == file_backend_fd) || (!start_called) ) {
        return -1;
    }

    this_event_length = EVENT_LENGTH( key, (NULL != info) );
    assert( this_event_length < event_avail_space );
    if( context->next_event_position + this_event_length > event_avail_space ) {
        if( switch_event_buffer(context) == -1 ) {
            return -2;
        }
    }
    this_event = (parsec_profiling_output_t *)&context->current_events_buffer->buffer[context->next_event_position];
    assert( context->current_events_buffer->buffer_type == PROFILING_BUFFER_TYPE_EVENTS );
    context->current_events_buffer->this_buffer.nb_events++;

    context->next_event_position += this_event_length;
    context->nb_events++;

    this_event->event.key = (uint16_t)key;
    this_event->event.event_id = event_id;
    this_event->event.taskpool_id = taskpool_id;
    this_event->event.flags = 0;

    if( NULL != info ) {
        memcpy(this_event->info, info, parsec_prof_keys[ BASE_KEY(key) ].info_length);
        this_event->event.flags = PARSEC_PROFILING_EVENT_HAS_INFO;
    }
    this_event->event.flags |= flags;
    now = take_time();
    this_event->event.timestamp = diff_time(parsec_start_time, now);

    return 0;
}

static int64_t dump_global_infos(int *nbinfos)
{
    parsec_profiling_buffer_t *b, *n;
    parsec_profiling_info_buffer_t *ib;
    parsec_profiling_info_t *i;
    int nb, nbthis, is, vs;
    int pos, tc, vpos;
    off_t first_off;
    char *value;

    if( NULL == parsec_profiling_infos ) {
        *nbinfos = 0;
        return -1;
    }

    b = allocate_empty_buffer(default_freelist, &first_off, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
    if( NULL == b ) {
        set_last_error("Profiling system: error: Unable to dump the global infos -- buffer allocation error\n");
        *nbinfos = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;
    for(i = parsec_profiling_infos; i != NULL; i = i->next) {
        is = strlen(i->key);
        vs = strlen(i->value);

        if( pos + sizeof(parsec_profiling_info_buffer_t) + is - 1 >= event_avail_space ) {
            b->this_buffer.nb_infos = nbthis;
            n = allocate_empty_buffer(default_freelist, &b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
            write_down_existing_buffer(default_freelist, b, pos);

            if( NULL == n ) {
                set_last_error("Profiling System: error: Global Infos will be truncated to %d infos only -- buffer allocation error\n", nb);
                *nbinfos = nb;
                return first_off;
            }

            b = n;
            pos = 0;
            nbthis = 0;
        }

        /* The key must fit in event_avail_space */
        if( sizeof(parsec_profiling_info_buffer_t) + is - 1 > event_avail_space ) {
            set_last_error("Profiling System: error: Key of size %d does not fit in an entire file segment of %d bytes\n",
                           is, event_avail_space);
            nb++;
            continue;
        }

        nbthis++;

        ib = (parsec_profiling_info_buffer_t *)&(b->buffer[pos]);
        ib->info_size = is;
        ib->value_size = vs;
        memcpy(ib->info_and_value + 0,  i->key,   is);

        pos += sizeof(parsec_profiling_info_buffer_t) + is - 1;

        vpos = 0;
        value = ib->info_and_value + is;
        while( vpos < vs ) {
            tc = (int)(event_avail_space - pos) < (vs-vpos) ? (int)(event_avail_space - pos) : (vs-vpos);
            memcpy(value, i->value + vpos, tc);
            vpos += tc;
            pos += tc;
            if( pos == (int)event_avail_space ) {
                b->this_buffer.nb_infos = nbthis;
                n = allocate_empty_buffer(default_freelist, &b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_GLOBAL_INFO);
                write_down_existing_buffer(default_freelist, b, pos);

                if( NULL == n ) {
                    set_last_error("Profiling System: error: Global Infos will be truncated to %d infos only -- buffer allocation error\n", nb);
                    *nbinfos = nb;
                    return first_off;
                }

                b = n;
                pos = 0;
                nbthis = 0;
                value = (char*)&(b->buffer[pos]);
            }
        }
        nb++;
    }

    b->this_buffer.nb_infos = nbthis;
    write_down_existing_buffer(default_freelist, b, pos);

    *nbinfos = nb;
    return first_off;
}

static int64_t dump_dictionary(int *nbdico)
{
    parsec_profiling_buffer_t *b, *n;
    parsec_profiling_key_buffer_t *kb;
    parsec_profiling_key_t *k;
    unsigned int i;
    int nb, nbthis, cs, pos;
    off_t first_off;

    if( 0 == parsec_prof_keys_count ) {
        *nbdico = 0;
        return -1;
    }

    b = allocate_empty_buffer(default_freelist, &first_off, PROFILING_BUFFER_TYPE_DICTIONARY);
    if( NULL == b ) {
        set_last_error("Profiling System: error: Unable to dump the dictionary -- buffer allocation error\n");
        *nbdico = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;
    for(i = 0; i < parsec_prof_keys_count; i++) {
        k = &parsec_prof_keys[i];
        if(NULL == k->convertor )
            cs = 0;
        else
            cs = strlen(k->convertor);

        if( pos + sizeof(parsec_profiling_key_buffer_t) + cs - 1 >= event_avail_space ) {
            b->this_buffer.nb_dictionary_entries = nbthis;
            n = allocate_empty_buffer(default_freelist, &b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_DICTIONARY);

            write_down_existing_buffer(default_freelist, b, pos);

            b = n;
            pos = 0;
            nbthis = 0;

            if( NULL == b ) {
                set_last_error("Profiling system: error: Dictionary will be truncated to %d entries only -- buffer allocation error\n", nb);
                *nbdico = nb;
                return first_off;
            }
        }
        kb = (parsec_profiling_key_buffer_t *)&(b->buffer[pos]);
        strncpy(kb->name, k->name, 63); /* We copy only up to 63 bytes to leave room for the '\0' */
        strncpy(kb->attributes, k->attributes, 127); /* We copy only up to 127 bytes to leave room for the '\0' */
        kb->keyinfo_length = k->info_length;
        kb->keyinfo_convertor_length = cs;
        if( cs > 0 ) {
            memcpy(kb->convertor, k->convertor, cs);
        }
        nb++;
        nbthis++;
        pos += sizeof(parsec_profiling_key_buffer_t) + cs - 1;
    }

    b->this_buffer.nb_dictionary_entries = nbthis;
    write_down_existing_buffer(default_freelist, b, pos);

    *nbdico = nb;
    return first_off;
}

static size_t thread_size(parsec_thread_profiling_t *thread)
{
    size_t s = 0;
    parsec_profiling_info_t *i;
    int ks, vs;

    s += sizeof(parsec_profiling_thread_buffer_t) - sizeof(parsec_profiling_info_buffer_t);
    for(i = thread->infos; NULL!=i; i = i->next) {
        ks = strlen(i->key);
        vs = strlen(i->value);
        if( s + ks + vs + sizeof(parsec_profiling_info_buffer_t) - 1 > event_avail_space ) {
            set_last_error("Profiling system: warning: unable to save info %s of thread %s, info ignored\n",
                           i->key, thread->hr_id);
            continue;
        }
        s += ks + vs + sizeof(parsec_profiling_info_buffer_t) - 1;
    }
    return s;
}

static int64_t dump_thread(int *nbth)
{
    parsec_profiling_buffer_t *b, *n;
    parsec_profiling_thread_buffer_t *tb;
    int nb, nbthis, nbinfos, ks, vs, pos;
    parsec_profiling_info_t *i;
    parsec_profiling_info_buffer_t *ib;
    off_t off;
    size_t th_size;
    parsec_list_item_t *it;
    parsec_thread_profiling_t* thread;

    if( parsec_list_is_empty(&threads) ) {
        *nbth = 0;
        return -1;
    }

    b = allocate_empty_buffer(default_freelist, &off, PROFILING_BUFFER_TYPE_THREAD);
    if( NULL == b ) {
        set_last_error("Profiling system: error: Unable to dump some thread profiles -- buffer allocation error\n");
        *nbth = 0;
        return -1;
    }

    pos = 0;
    nb = 0;
    nbthis = 0;

    for(it = PARSEC_LIST_ITERATOR_FIRST( &threads );
        it != PARSEC_LIST_ITERATOR_END( &threads );
        it = PARSEC_LIST_ITERATOR_NEXT( it ) ) {
        thread = (parsec_thread_profiling_t*)it;

        if(thread->nb_events == 0)
            continue; /* We don't store threads with no events at all */

        th_size = thread_size(thread);

        if( pos + th_size >= event_avail_space ) {
            b->this_buffer.nb_threads = nbthis;
            n = allocate_empty_buffer((tl_freelist_t*)thread->tls_storage, &b->next_buffer_file_offset, PROFILING_BUFFER_TYPE_THREAD);
            write_down_existing_buffer((tl_freelist_t*)thread->tls_storage, b, pos);

            if( NULL == n ) {
                set_last_error("Profiling system: error: Threads will be truncated to %d threads only -- buffer allocation error\n", nb);
                *nbth = nb;
                return off;
            }

            b = n;
            pos = 0;
            nbthis = 0;
        }

        tb = (parsec_profiling_thread_buffer_t *)&(b->buffer[pos]);
        tb->nb_events = thread->nb_events;
        strncpy(tb->hr_id, thread->hr_id, 127); /* We copy only up to 127 bytes to leave room for the '\0' */
        tb->first_events_buffer_offset = thread->first_events_buffer_offset;

        nb++;
        nbthis++;

        nbinfos = 0;
        i = thread->infos;
        pos += sizeof(parsec_profiling_thread_buffer_t) - sizeof(parsec_profiling_info_buffer_t);
        while( NULL != i ) {
            ks = strlen(i->key);
            vs = strlen(i->value);
            if( pos + ks + vs + sizeof(parsec_profiling_info_buffer_t) - 1 >= event_avail_space ) {
                continue;
            }
            ib = (parsec_profiling_info_buffer_t*)&(b->buffer[pos]);
            ib->info_size = ks;
            ib->value_size = vs;
            memcpy(ib->info_and_value, i->key, ks);
            memcpy(ib->info_and_value + ks, i->value, vs);
            pos += ks + vs + sizeof(parsec_profiling_info_buffer_t) - 1;
            i = i->next;
            nbinfos++;
        }
        tb->nb_infos = nbinfos;
    }

    b->this_buffer.nb_threads = nbthis;
    write_down_existing_buffer(default_freelist, b, pos);

    *nbth = nb;
    return off;
}

int parsec_profiling_dbp_dump( void )
{
    int nb_threads = 0;
    parsec_thread_profiling_t *t;
    int nb_infos, nb_dico;
    parsec_list_item_t *it;

    if( !__profile_initialized ) return 0;

    if( NULL == bpf_filename ) {
        set_last_error("Profiling system: User Error: parsec_profiling_dbp_dump before parsec_profiling_dbp_start()");
        return -1;
    }
    if( NULL == profile_head ) {
        set_last_error("Profiling system: User Error: parsec_profiling_dbp_dump before parsec_profiling_dbp_start()");
        return -1;
    }

    /* Flush existing events buffer, unconditionally */
    for(it = PARSEC_LIST_ITERATOR_FIRST( &threads );
        it != PARSEC_LIST_ITERATOR_END( &threads );
        it = PARSEC_LIST_ITERATOR_NEXT( it ) ) {
        t = (parsec_thread_profiling_t*)it;
        if( NULL != t->current_events_buffer && t->next_event_position != 0 ) {
            write_down_existing_buffer((tl_freelist_t*)t->tls_storage, t->current_events_buffer, t->next_event_position);
            t->current_events_buffer = NULL;
        }
    }

    profile_head->dictionary_offset = dump_dictionary(&nb_dico);
    profile_head->dictionary_size = nb_dico;

    profile_head->info_offset = dump_global_infos(&nb_infos);
    profile_head->info_size = nb_infos;

    profile_head->thread_offset = dump_thread(&nb_threads);
    profile_head->nb_threads = nb_threads;

    /* The head is now complete. Last flush. */
    write_down_existing_buffer(default_freelist,
                               (parsec_profiling_buffer_t *)profile_head,
                               sizeof(parsec_profiling_binary_file_header_t));

#if defined(PARSEC_PROFILING_USE_HELPER_THREAD)
    int my_flush_ticket;
    pthread_mutex_lock(&io_cmd_flush_mutex);
    my_flush_ticket = io_cmd_flush_counter + 1;
    pthread_mutex_unlock(&io_cmd_flush_mutex);

    io_cmd_t *cmd = io_cmd_allocate();
    cmd->buffer = IO_CMD_FLUSH;
    cmd->fl = NULL;
    cmd->next = NULL;
    do_and_measure_perf(PERF_USER_WAITING,
       pthread_mutex_lock(&cmd_queue.lock));
    if( NULL == cmd_queue.last ) {
        cmd_queue.last = cmd_queue.next = cmd;
    } else {
        cmd_queue.last->next = cmd;
        cmd_queue.last = cmd;
    }
    pthread_cond_signal(&cmd_queue.cond);
    pthread_mutex_unlock(&cmd_queue.lock);

    pthread_mutex_lock(&io_cmd_flush_mutex);
    while( io_cmd_flush_counter != my_flush_ticket ) {
        pthread_cond_wait(&io_cmd_flush_cond, &io_cmd_flush_mutex);
    }
    pthread_mutex_unlock(&io_cmd_flush_mutex);
#endif

#if defined(PARSEC_PROFILING_USE_MMAP)
    tl_freelist_buffer_t *b;
    /* Buffers that were unecessarily pre-maped need to be released */
    for(it = PARSEC_LIST_ITERATOR_FIRST( &threads );
        it != PARSEC_LIST_ITERATOR_END( &threads );
        it = PARSEC_LIST_ITERATOR_NEXT( it ) ) {
        t = (parsec_thread_profiling_t*)it;
        tl_freelist_t *fl = TLS_STORAGE(t);
        while(fl->first != NULL) {
            b = fl->first;
            fl->first = b->next;
            fl->nb_allocated--;
            munmap(b, event_buffer_size);
        }
    }
    while(default_freelist->first != NULL) {
        b = default_freelist->first;
        default_freelist->first = b->next;
        default_freelist->nb_allocated--;
        munmap(b, event_buffer_size);
    }
#endif

    /* Close the backend file */
    pthread_mutex_lock(&file_backend_lock);
    close(file_backend_fd);
    file_backend_fd = -1;
    file_backend_extendable = 0;
    free(bpf_filename);
    bpf_filename = NULL;
    pthread_mutex_unlock(&file_backend_lock);

    if( parsec_profiling_raise_error )
        return -1;

    return 0;
}

int parsec_profiling_dbp_start( const char *basefile, const char *hr_info )
{
    int64_t zero;
    char *xmlbuffer;
    int rank = 0, worldsize = 1, buflen;
    int  min_fd = -1, rc;
#if defined(PARSEC_HAVE_MPI)
    char *unique_str;

    int MPI_ready;
    (void)MPI_Initialized(&MPI_ready);
    if(MPI_ready) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    }
#endif

    if( !__profile_initialized ) return -1;

    rc = asprintf(&bpf_filename, "%s-%d.prof-XXXXXX", basefile, rank);
    if (rc == -1) {
        set_last_error("Profiling system: error: one (or more) process could not create the backend file name (out of ressource).\n");
        return -1;
    }
    if( rank == 0 ) {
        /* The first process create the unique locally unique filename, and then
         * share it with every other participants. If such a file cannot be
         * created broacast an empty key to all other processes.
         */
        mode_t old_mask = umask(S_IWGRP | S_IWOTH);
        min_fd = file_backend_fd = mkstemp(bpf_filename);
        (void)umask(old_mask);
        if( -1 == file_backend_fd ) {
            set_last_error("Profiling system: error: Unable to create backend file %s: %s. Events not logged.\n",
                           bpf_filename, strerror(errno));
            file_backend_extendable = 0;
            memset(bpf_filename, 0, strlen(bpf_filename));
        }
    }

#if defined(PARSEC_HAVE_MPI)
    if( worldsize > 1) {
        unique_str = bpf_filename + (strlen(bpf_filename) - 6);  /* pinpoint directly into the bpf_filename */

        MPI_Bcast(unique_str, 7, MPI_CHAR, 0, MPI_COMM_WORLD);
        if( 0 != rank ) {
            if( *unique_str != '\0') {
                file_backend_fd = open(bpf_filename, O_RDWR | O_CREAT | O_TRUNC, 00600);
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

    default_freelist = malloc(sizeof(tl_freelist_t));
    tl_freelist_buffer_t *e;
    pthread_mutex_init(&default_freelist->lock, NULL);
    e = (tl_freelist_buffer_t*)do_allocate_new_buffer();
    if( NULL == e ) {
        return -1;
    }
    e->next = NULL;
    default_freelist->first = e;
    default_freelist->nb_allocated = 1;
    for(rc = 1; rc < parsec_profiling_per_thread_buffer_freelist_min; rc++) {
        e->next = (tl_freelist_buffer_t*)do_allocate_new_buffer();
        if( NULL == e->next ) {
            return -1;
        }
        e = e->next;
        e->next = NULL;
        default_freelist->nb_allocated++;
    }

    /* Create the header of the profiling file */
    profile_head = (parsec_profiling_binary_file_header_t*)allocate_empty_buffer(default_freelist, &zero, PROFILING_BUFFER_TYPE_HEADER);
    if( NULL == profile_head )
        return -1;

    memcpy(profile_head->magick, PARSEC_PROFILING_MAGICK, strlen(PARSEC_PROFILING_MAGICK) + 1);
    profile_head->byte_order = 0x0123456789ABCDEF;
    profile_head->profile_buffer_size = event_buffer_size;
    strncpy(profile_head->hr_id, hr_info, 127); /* We copy only up to 127 bytes to leave room for the '\0' */
    profile_head->rank = rank;
    profile_head->worldsize = worldsize;

    /* Reset the error system */
    set_last_error("Profiling system: success");
    parsec_profiling_raise_error = 0;

    /* It's fine to re-reset the event date: we're back with a zero-length event set */
    start_called = 0;

    if( parsec_hwloc_export_topology(&buflen, &xmlbuffer) != -1 &&
        buflen > 0 ) {
        parsec_profiling_add_information("HWLOC-XML", xmlbuffer);
        parsec_hwloc_free_xml_buffer(xmlbuffer);
    }
    return 0;
}

uint64_t parsec_profiling_get_time(void) {
    return diff_time(parsec_start_time, take_time());
}

void parsec_profiling_enable(void)
{
    parsec_profile_enabled = 1;
}
void parsec_profiling_disable(void)
{
    parsec_profile_enabled = 0;
}

void profiling_save_dinfo(const char *key, double value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_iinfo(const char *key, int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_uint64info(const char *key, unsigned long long int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_sinfo(const char *key, char* svalue)
{
    parsec_profiling_add_information(key, svalue);
}

void profiling_thread_save_dinfo(parsec_thread_profiling_t * thread,
                                 const char *key, double value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_thread_add_information(thread, key, svalue);
    free(svalue);
}

void profiling_thread_save_iinfo(parsec_thread_profiling_t * thread,
                                 const char *key, int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_thread_add_information(thread, key, svalue);
    free(svalue);
}

void profiling_thread_save_uint64info(parsec_thread_profiling_t * thread,
                                      const char *key, unsigned long long int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_thread_add_information(thread, key, svalue);
    free(svalue);
}

void profiling_thread_save_sinfo(parsec_thread_profiling_t * thread,
                                 const char *key, char* svalue)
{
    parsec_profiling_thread_add_information(thread, key, svalue);
}
