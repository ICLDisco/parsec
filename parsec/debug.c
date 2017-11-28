/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/debug.h"
#include "parsec/utils/output.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"
#include "parsec/os-spec-timing.h"

#include <stdio.h>
#include <string.h>
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* PARSEC_HAVE_UNISTD_H */
#include <pthread.h>

/* globals for use in macros from debug.h */
char parsec_debug_hostname[32]   = "unknownhost";
int parsec_debug_rank            = -1;
int parsec_debug_output          = 0;
int parsec_debug_verbose         = 1;
int parsec_debug_history_verbose = 1;
int parsec_debug_colorize        = 10; /* 10 is the size of the format string for colors */
int parsec_debug_coredump_on_fatal = 0;
int parsec_debug_history_on_fatal = 0;
void (*parsec_weaksym_exit)(int status) = _Exit;

/* debug backtrace circular buffer */
static int bt_output    = -1;
static int ST_SIZE      = 128;
static int ST_ASIZE     = 64;
static uint32_t st_idx  = 0;
static void **stack     = NULL;
static int* stack_size  = NULL;


#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
#include <mpi.h>
static void parsec_mpi_exit(int status) {
    MPI_Abort(MPI_COMM_WORLD, status);
}
#endif

void parsec_debug_init(void) {
#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
    int mpi_is_up;
    MPI_Initialized(&mpi_is_up);
    if( mpi_is_up ) {
        MPI_Comm_rank(MPI_COMM_WORLD, &parsec_debug_rank);
        parsec_weaksym_exit = parsec_mpi_exit;
    }
#endif
    gethostname(parsec_debug_hostname, sizeof(parsec_debug_hostname));

    parsec_debug_output = parsec_output_open(NULL);

    parsec_mca_param_reg_int_name("debug", "verbose",
        "Set the output level for debug messages"
        ", 0: Errors only"
        ", 1: Warnings (minimum recommended)"
        ", 2: Info (default)"
        ", 3-4: User Debug"
        ", 5-9: Devel Debug"
        ", >=10: Chatterbox Debug"
#if !defined(PARSEC_DEBUG_PARANOID) || !defined(PARSEC_DEBUG_NOISIER) || !defined(PARSEC_DEBUG_HISTORY)
        " (heaviest debug output available only when compiling with PARSEC_DEBUG_PARANOID, PARSEC_DEBUG_NOISIER and/or PARSEC_DEBUG_HISTORY in ccmake)"
#endif
        , false, false, 1, &parsec_debug_verbose);
    parsec_output_set_verbosity(parsec_debug_output, parsec_debug_verbose);
    parsec_output_set_verbosity(0, parsec_debug_verbose);

#if defined(PARSEC_DEBUG_HISTORY)
    parsec_debug_history_init();
#endif

    parsec_mca_param_reg_int_name("debug", "color",
        "Toggle on/off color output for debug messages",
        false, false, 1, &parsec_debug_colorize);
    parsec_debug_colorize = parsec_debug_colorize? 10: 0;

    parsec_mca_param_reg_int_name("debug", "coredump_on_fatal",
        "Toggle on/off raise sigabort on internal engine error",
        false, false, 0, &parsec_debug_coredump_on_fatal);
    parsec_debug_coredump_on_fatal = parsec_debug_coredump_on_fatal ? 1: 0;

    parsec_mca_param_reg_int_name("debug", "history_on_fatal",
        "Toggle on/off dump the debug history on internal engine error",
        false, false, 0, &parsec_debug_history_on_fatal);
    parsec_debug_history_on_fatal = parsec_debug_history_on_fatal ? 1: 0;

    /* We do not want backtraces in the syslog, so, we do not
     * inherit the defaults... */
    char* opt;
    parsec_output_stream_t lds;
    parsec_mca_param_reg_string_name("debug", "backtrace_output",
        "Define the output for the backtrace dumps (none, stderr, file)",
        false, false, "file", &opt);
    if( 0 == strcasecmp(opt, "none") ) {
        bt_output = -1;
    }
    else if( 0 == strcasecmp(opt, "stderr") ) {
        OBJ_CONSTRUCT(&lds, parsec_output_stream_t);
        lds.lds_want_stderr = true;
        lds.lds_want_syslog = false;
        bt_output = parsec_output_open(&lds);
        OBJ_DESTRUCT(&lds);
    }
    else if( 0 == strcasecmp(opt, "file") ) {
        OBJ_CONSTRUCT(&lds, parsec_output_stream_t);
        lds.lds_want_file = true;
        lds.lds_want_syslog = false;
        lds.lds_file_suffix = "backtraces";
        bt_output = parsec_output_open(&lds);
        OBJ_DESTRUCT(&lds);
    }
    else {
        parsec_warning("Invalid value %s for parameter debug_backtrace_output", opt);
    }
    free(opt);

    parsec_mca_param_reg_int_name("debug", "backtrace_keep",
        "Maximum number of backtrace to keep in backtrace circular buffer",
        false, false, ST_ASIZE, &ST_ASIZE);
    parsec_mca_param_reg_int_name("debug", "backtrace_size",
        "Maximum size for each backtrace",
        false, false, ST_SIZE, &ST_SIZE);
    if( -1 != bt_output ) {
        stack = malloc(ST_ASIZE*ST_SIZE*sizeof(void*));
        stack_size = malloc(ST_ASIZE*sizeof(int));
        if( (NULL == stack_size) || (NULL == stack) ) {
            parsec_warning("Backtrace debug framework DISABLED: could not allocate the backtrace circular buffer with backtrace_keep=%d and backtrace_size=%d", ST_ASIZE, ST_SIZE);
            if( NULL != stack_size ) { free(stack_size); stack_size = NULL; }
            if( NULL != stack ) { free(stack); stack = NULL; }
            if( bt_output > 0 ) {
                parsec_output_close(bt_output);
                bt_output = -1;
            }
            return;
        }
        memset(stack_size, 0, ST_ASIZE*sizeof(int));
        memset(stack, 0, ST_ASIZE*ST_SIZE*sizeof(int));
    }
}

void parsec_debug_fini(void)
{
    if( 0 < parsec_debug_output ) {
        parsec_output_close(parsec_debug_output);
    }

    if( 0 < bt_output ) {
        parsec_output_close(bt_output);
        bt_output = -1;
    }
    if( NULL != stack_size ) { free(stack_size); stack_size = NULL; }
    if( NULL != stack ) { free(stack); stack = NULL; }

    parsec_debug_history_fini();
}


/* STACKTRACES circular buffer */
#include <execinfo.h>

void parsec_debug_backtrace_save(void) {
    uint32_t my_idx = parsec_atomic_inc_32b(&st_idx) % ST_ASIZE;
    stack_size[my_idx] = backtrace(&stack[my_idx*ST_SIZE], ST_SIZE);
}

void parsec_debug_backtrace_dump(void) {
    int i, my, r = parsec_debug_rank, t;
    char **s;

    for(i = 0; i < ST_ASIZE; i++) {
        my = (st_idx + i) % ST_ASIZE;
        if( NULL == stack[my*ST_SIZE] ) continue;
        parsec_output(bt_output, "[%d] --- %u ---\n", r, st_idx + i);
        s = backtrace_symbols(&stack[my*ST_SIZE], stack_size[my]);
        for(t = 0; t < stack_size[my]; t++) {
            parsec_output(bt_output, "[%d]  %s\n", r, s[t]);
        }
        free(s);
        parsec_output(bt_output, "[%d]\n", r);
    }
}

/* DEBUG HISTORY circular buffer */

#if defined(PARSEC_DEBUG_HISTORY)

/**
 * Maximal size of the history length per thread.  This value is
 * settable using parsec_debug_max_history_length MCA parameter.
 */
static unsigned int parsec_debug_max_history_length_per_thread = 96;
static pthread_key_t thread_specific_debug_key;

typedef struct {
    parsec_time_t timestamp;
    int           allocated_size;
    char         *mark;
} mark_t;

typedef struct {
    uint32_t nextmark;
    mark_t   marks[1]; /* Actual size parsec_debug_max_history_length */
} mark_buffer_t;

/**
 * Each thread maintains a double buffer of marks
 * so that during the dump operation, threads add new marks to
 * a buffer that is not read.
 * This structure also holds some variables to help the writing of
 * the dump function.
 */
typedef struct mark_double_buffer_s {
    mark_buffer_t               *buffers[2];         /** Double buffers: one to write, the other to read */
    pthread_t                    thread_id;          /** Used by reading function to display information */
    uint32_t                     cur_mark;           /** Used by reading function to keep track of progress */
    struct mark_double_buffer_s *next;               /** Pointer to the double buffers of next thread */
} mark_double_buffer_t;

static volatile mark_double_buffer_t *mark_buffers = NULL;
static volatile uint32_t writing_buffer = 0; /** Can be 0 or 1 */
static parsec_time_t debug_start;

static void parsec_debug_history_init_thread(void)
{
    mark_double_buffer_t *my_buffers;
    if( parsec_debug_max_history_length_per_thread > 0 ) {
        my_buffers = (mark_double_buffer_t*)malloc(sizeof(mark_double_buffer_t));
        /* Calloc does all the initialization job:
         *   nextmark is set to 0
         *   For each mark :
         *     timestamp is set to arbitrary 0
         *     allocated_size is set to 0
         *     mark is set to NULL for each mark
         */
        my_buffers->buffers[0] = (mark_buffer_t *)calloc(1, sizeof(mark_buffer_t) + (parsec_debug_max_history_length_per_thread-1) * sizeof(mark_t));
        my_buffers->buffers[1] = (mark_buffer_t *)calloc(1, sizeof(mark_buffer_t) + (parsec_debug_max_history_length_per_thread-1) * sizeof(mark_t));
        my_buffers->thread_id  = pthread_self();
        pthread_setspecific(thread_specific_debug_key, my_buffers);
        /* Just need to chain this thread buffers to the global list for dumping and cleaning */
        do {
            my_buffers->next = (mark_double_buffer_t*)mark_buffers;
        } while( !parsec_atomic_cas_ptr(&mark_buffers, my_buffers->next, my_buffers) );
    }
}

static inline mark_t *get_my_mark(void) {
    uint32_t mymark_idx;
    mark_double_buffer_t *my_buffers;
    mark_buffer_t *marks;

    assert( parsec_debug_max_history_length_per_thread > 0 );
    
    my_buffers = (mark_double_buffer_t*)pthread_getspecific(thread_specific_debug_key);
    if( NULL == my_buffers ) {
        parsec_debug_history_init_thread();
        my_buffers = (mark_double_buffer_t*)pthread_getspecific(thread_specific_debug_key);
    }    
    marks = my_buffers->buffers[writing_buffer];
    mymark_idx = marks->nextmark++;
    mymark_idx %= parsec_debug_max_history_length_per_thread;
    marks->marks[mymark_idx].timestamp = take_time();
    return (mark_t*)&marks->marks[mymark_idx];
}

void parsec_debug_history_add(const char *format, ...) {
    va_list args;
    int actual_size;
    mark_t *my_mark;

    if( parsec_debug_max_history_length_per_thread == 0 )
        return;
    
    my_mark = get_my_mark();

    va_start(args, format);
    if( my_mark->allocated_size == 0 ) {
        my_mark->allocated_size = vasprintf(&my_mark->mark, format, args) + 1;
    } else {
        actual_size = vsnprintf(my_mark->mark, my_mark->allocated_size, format, args);
        if( actual_size >= my_mark->allocated_size ) {
            free(my_mark->mark);
            my_mark->allocated_size = vasprintf(&my_mark->mark, format, args) + 1;
        }
    }
    va_end(args);
}

void parsec_debug_history_dump(void) {
    int printing_buffer = writing_buffer;
    parsec_time_t min_ts;
    mark_double_buffer_t *db, *min_db;

    if( parsec_debug_max_history_length_per_thread == 0 )
        return;
    
    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     * This CAS can only fail if parsec_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    parsec_atomic_cas_32b(&writing_buffer, printing_buffer, printing_buffer == 0 ? 1 : 0);

    /* Set the starting point for all threads */
    for(db = (mark_double_buffer_t*)mark_buffers; db != NULL; db = db->next) {
        if( db->buffers[printing_buffer]->nextmark >= parsec_debug_max_history_length_per_thread ) {
            db->cur_mark = db->buffers[printing_buffer]->nextmark - parsec_debug_max_history_length_per_thread;
        } else {
            db->cur_mark = 0;
        }
    }
    
    /* As long as there is a mark to display for one thread */
    parsec_inform("== Begin debug history =====================================================");
    while(1) {
        /* Find the thread with the lowest timestamp */
        min_db  = NULL;
        for(db = (mark_double_buffer_t*)mark_buffers; db != NULL; db = db->next) {
            if( db->cur_mark != (db->buffers[printing_buffer]->nextmark ) ) {
                if(min_db == NULL || time_less(db->buffers[printing_buffer]->marks[db->cur_mark % parsec_debug_max_history_length_per_thread].timestamp, min_ts)) {
                    min_db = db;
                    min_ts = db->buffers[printing_buffer]->marks[db->cur_mark % parsec_debug_max_history_length_per_thread].timestamp;
                }
            }
        }
        if( min_db == NULL )
            break;
        if( NULL != min_db->buffers[printing_buffer]->marks[min_db->cur_mark % parsec_debug_max_history_length_per_thread].mark ) {
            parsec_output(parsec_debug_output, " %p/%lu (%6.03g s) -- %s",
                          (void*)min_db->thread_id,
                          min_db->cur_mark,
                          (double)diff_time(debug_start, min_ts) / 1e9,
                          min_db->buffers[printing_buffer]->marks[min_db->cur_mark % parsec_debug_max_history_length_per_thread].mark);
            min_db->cur_mark++;
        } else {
            parsec_output(parsec_debug_output, " (empty mark)");
        }
    }
    parsec_inform("== End debug history =====================================================");
}

static void debug_history_purge_one(void)
{
    mark_double_buffer_t *db;
    uint32_t ii, purging_buffer = writing_buffer;
    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     * This CAS can only fail if parsec_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    parsec_atomic_cas_32b(&writing_buffer, purging_buffer, purging_buffer == 0 ? 1 : 0);

    for(db = (mark_double_buffer_t*)mark_buffers; NULL != db; db = db->next) {
        for(ii = 0; ii < parsec_debug_max_history_length_per_thread; ii++) {
            if( NULL != db->buffers[purging_buffer]->marks[ii].mark ) {
                db->buffers[purging_buffer]->marks[ii].mark[0] = '\0';
            }
        }
    }
}

void parsec_debug_history_purge(void) {
    debug_history_purge_one();
    debug_history_purge_one();
}

void parsec_debug_history_init(void) {
    int default_history_length = parsec_debug_max_history_length_per_thread;
    int chosen_history_length;
    
    pthread_key_create(&thread_specific_debug_key, NULL);

    parsec_mca_param_reg_int_name("debug", "history_verbose",
                                  "Set the output level for debug history ring buffer; same values as debug_verbose",
                                  false, false, parsec_debug_verbose, &parsec_debug_history_verbose);
    parsec_mca_param_reg_int_name("parsec", "debug_max_history_length_per_thread",
                                  "How many debug line to keep in the history",
                                  false, false, default_history_length, &chosen_history_length);

    debug_start = take_time();
    if( chosen_history_length > 0 ) {
        parsec_debug_max_history_length_per_thread = chosen_history_length;
        parsec_debug_history_init_thread();
    } else {
        parsec_debug_max_history_length_per_thread = 0;
    }
}

void parsec_debug_history_fini(void) {
    uint32_t b, ii;
    mark_double_buffer_t *db, *next;

    for(db = (mark_double_buffer_t*)mark_buffers; NULL != db; db = next) {
        next = db->next;
        for( b = 0; b < 2; b++) {
            for(ii = 0; ii < parsec_debug_max_history_length_per_thread; ii++) {
                if( NULL != db->buffers[b]->marks[ii].mark )
                    free( db->buffers[b]->marks[ii].mark);
            }
            free(db->buffers[b]);
        }
        free(db);
    }
    mark_buffers = NULL;
}

#endif /* defined(PARSEC_DEBUG_HISTORY) */
