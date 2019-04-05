/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"
#include "parsec/os-spec-timing.h"
#include "parsec/sys/tls.h"

#include <stdio.h>
#include <string.h>
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* PARSEC_HAVE_UNISTD_H */
#include <pthread.h>

/* globals for use in macros from debug.h */
int parsec_debug_rank            = -1;
int parsec_debug_output          = 0;
int parsec_debug_verbose         = 1;
int parsec_debug_history_verbose = 1;
int parsec_debug_colorize        = 10; /* 10 is the size of the format string for colors */
int parsec_debug_coredump_on_fatal = 0;
int parsec_debug_history_on_fatal = 0;

/* debug backtrace circular buffer */
static int bt_output    = -1;
static int ST_SIZE      = 128;
static int ST_ASIZE     = 64;
static int32_t st_idx   = 0;
static void **stack     = NULL;
static int* stack_size  = NULL;


void parsec_debug_init(void)
{
    /* The caller is supposed to set parsec_debug_rank if she expects
     * nicer output. */
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
        PARSEC_OBJ_CONSTRUCT(&lds, parsec_output_stream_t);
        lds.lds_want_stderr = true;
        lds.lds_want_syslog = false;
        bt_output = parsec_output_open(&lds);
        PARSEC_OBJ_DESTRUCT(&lds);
    }
    else if( 0 == strcasecmp(opt, "file") ) {
        PARSEC_OBJ_CONSTRUCT(&lds, parsec_output_stream_t);
        lds.lds_want_file = true;
        lds.lds_want_syslog = false;
        lds.lds_file_suffix = "backtraces";
        bt_output = parsec_output_open(&lds);
        PARSEC_OBJ_DESTRUCT(&lds);
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
    uint32_t my_idx = (parsec_atomic_fetch_inc_int32(&st_idx) + 1) % ST_ASIZE;
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
 * Maximal size of the history length per thread in bytes. This value
 * is settable using parsec_debug_max_history_length MCA parameter.
 */
static size_t parsec_debug_max_history_length_per_thread = (1024*1024);

/**
 * Main header structure for each debug mark
 */
typedef struct {
    parsec_time_t timestamp;  /** Timestamp of the mark (to re-order in case of multithread use */
    uint32_t      mark_index; /** Index of the mark (to decide if we cycled / if we skipped some marks */
    uint32_t      next_mark;  /** Counter in MARK_SIZE bytes of the address of the next mark */
    char          mark[1];    /** Bytes of the mark, '\0'-ended, ends before buffer start + next_mark*MARK_SIZE */
} mark_t;

/** Align the marks buffer by elements of size MARK_SIZE */
#define MARK_SIZE sizeof(mark_t)

/** UNDEFINED MARK is used to signal that a mark buffer is empty (the start mark is UNDEFINED) */
#define UNDEFINED_MARK (0xFFFFFFFF)

/**
 * A mark buffer is an allocated memory segment of size parsec_debug_max_history_length_per_thread (upgraded to 
 * a multiple of MARK_SIZE, and a set of pointers / counters to track marks inside that buffer. */
typedef struct {
    unsigned char *buffer;       /** Pointer to a pre-allocated buffer of parsec_debug_max_history_length_per_thread bytes */
    uint32_t current_start_mark; /** Counter of the first mark that is still complete (mark is at address buffer + current_start_mark * MARK_SIZE) */
    uint32_t current_end_mark;   /** Counter of the next mark to write (mark is at address buffer + current_end_mark * MARK_SIZE) */
    uint32_t current_index;      /** Index of the next mark to write */
} mark_buffer_t;

/**
 * Each thread maintains a double buffer of marks
 * so that during the dump operation, threads add new marks to
 * a buffer that is not read.
 * This structure also holds some variables to help the writing of
 * the dump function.
 */
typedef struct mark_double_buffer_s {
    mark_buffer_t                buffers[2];         /** Double buffers: one to write, the other to read */
    pthread_t                    thread_id;          /** Used by reading function to display information */
    struct mark_double_buffer_s *next;               /** Pointer to the double buffers of next thread */
} mark_double_buffer_t;

/**
 * Thread Local Storage key / variable so each thread finds its own buffers
 */
PARSEC_TLS_DECLARE(tls_debug);

/**
 * This global stores a linked list of all mark buffers (one per thread that
 * called a parsec_debug function)
 */
static volatile mark_double_buffer_t *mark_buffers = NULL;

/**
 * This global stores which buffer of the double buffer is currently used for writing
 */
static volatile int32_t writing_buffer = 0; /** Can be 0 or 1 */

/**
 * In order to print reasonable times, each timing before being printed is negatively offset by
 * the date at init call. */
static parsec_time_t debug_start;

/**
 * @brief Initializes thread-specific storage for the debug history of the calling thread
 *
 * @details
 *   This assumes that the calling thread has no buffers initializes yet, allocates
 *   the memory for the calling thread an initializes it. It also touches the entire
 *   memory allocated in order to guarantee that long page faults management have a
 *   lower probability to happen during a subsequent call to parsec_debug_history_add
 *   The allocated memory is both chained to the global mark_buffers, and assigned
 *   to the thread local storage for later lookup by parsec_debug_history_add.
 */
static void parsec_debug_history_init_thread(void)
{
    mark_double_buffer_t *my_buffers;
    if( parsec_debug_max_history_length_per_thread > 0 ) {
        /* The calloc here also sets to 0 the current_end_mark, and the current_index
         * for both buffers. This initializes the structure appropriately */
        my_buffers = (mark_double_buffer_t*)calloc(1, sizeof(mark_double_buffer_t));
        my_buffers->buffers[0].current_start_mark = UNDEFINED_MARK;
        my_buffers->buffers[1].current_start_mark = UNDEFINED_MARK;
        /* The calloc here is used to ensure that no page fault will happen after this step */
        my_buffers->buffers[0].buffer = (unsigned char*)calloc(1, parsec_debug_max_history_length_per_thread);
        my_buffers->buffers[1].buffer = (unsigned char*)calloc(1, parsec_debug_max_history_length_per_thread);
        my_buffers->thread_id  = pthread_self();
        PARSEC_TLS_SET_SPECIFIC(tls_debug, my_buffers);
        /* Just need to chain this thread buffers to the global list for dumping and cleaning */
        do {
            my_buffers->next = (mark_double_buffer_t*)mark_buffers;
        } while( !parsec_atomic_cas_ptr(&mark_buffers, my_buffers->next, my_buffers) );
    }
}

/**
 * @brief Returns a unique mark for the calling thread
 *
 * @details
 *   This function finds the current writing buffer for the calling thread
 *   or create a double buffer for the thread if there are none.
 *   It then create / overwrite, in the history buffer, a new mark
 *   capable of storing actual_size bytes.
 *   If the buffer is too small, it returns in actual_space how many bytes
 *   are usable.
 *   If actual_size is too big to be stored at the current position in the
 *   buffer, it will cycle, and start overwriting the previous part of the
 *   buffer.
 *
 *  @param[IN] actual_size: number of bytes to write to the mark
 *  @param[OUT] actual_space: number of bytes available
 *  @return the pointer to the new mark.
 */
static inline mark_t *get_my_mark(int actual_size, int *actual_space)
{
    mark_double_buffer_t *my_double_buffers;
    mark_buffer_t *my_buffer;
    mark_t *my_mark;
    int bytes_left;
    uint32_t actual_slots;

    my_double_buffers = (mark_double_buffer_t*)PARSEC_TLS_GET_SPECIFIC(tls_debug);
    if( NULL == my_double_buffers ) {
        parsec_debug_history_init_thread();
        my_double_buffers = (mark_double_buffer_t*)PARSEC_TLS_GET_SPECIFIC(tls_debug);
        assert(NULL != my_double_buffers);
    }
    my_buffer = &my_double_buffers->buffers[writing_buffer];

    /* Find where we can put actual_size bytes + MARK_SIZE */
    my_mark = (mark_t*)&my_buffer->buffer[my_buffer->current_end_mark * MARK_SIZE];
    bytes_left = ( ((char *)(my_buffer->buffer + parsec_debug_max_history_length_per_thread) - my_mark->mark) );
    if( bytes_left > actual_size ) {
        /* We have room to store everything at my_mark */
        *actual_space = actual_size;
    } else {
        if( my_buffer->current_end_mark == 0 ) {
            /* Special case: the entire buffer is not long enough to store */
            *actual_space = bytes_left;
            my_buffer->current_start_mark = 0;
        } else {
            /* Skip this spot, it's too short */

            /* If the start mark is between my_mark and the end, roll it
             * back to the first mark after the beginning */
            if( my_buffer->current_start_mark >= my_buffer->current_end_mark ) {
                my_buffer->current_start_mark = ((mark_t*)my_buffer->buffer)->next_mark;
            }

            /* And move the writing mark to the beginning */
            my_buffer->current_end_mark = 0;

            /* Write an empty mark at my_mark, that points to the actual
             * chosen position for my_mark */
            my_mark->next_mark = my_buffer->current_end_mark;
            my_mark->mark[0] = '\0';
            my_mark->timestamp = take_time();
            my_mark->mark_index = my_buffer->current_index;

            /* Set the same variables as in the if case */
            my_mark = (mark_t*)my_buffer->buffer;
            bytes_left = parsec_debug_max_history_length_per_thread - MARK_SIZE;
            if( bytes_left - (int)MARK_SIZE < actual_size )
                *actual_space = bytes_left;
            else
                *actual_space = actual_size;
        }
    }

    /* Compute how many slots we will use */
    actual_slots = 1 + (*actual_space + MARK_SIZE - 3) / MARK_SIZE;

    /* If the start mark is still undefined, define it now */
    if( my_buffer->current_start_mark == UNDEFINED_MARK ) {
        my_buffer->current_start_mark = my_buffer->current_end_mark;
    } else {
        /* Otherwise, while the start mark is inside the new mark, move it following the
         * previous chaining. */
        while( my_buffer->current_start_mark >= my_buffer->current_end_mark &&
               my_buffer->current_start_mark < my_buffer->current_end_mark + actual_slots ) {
            /* In case the log is empty: the start mark would then become the same as the current end mark */
            if( ((mark_t*)(my_buffer->buffer + MARK_SIZE*my_buffer->current_start_mark))->next_mark == my_buffer->current_end_mark)
                break;
            /* Otherwise move on */
            assert(my_buffer->current_start_mark != ((mark_t*)(my_buffer->buffer + MARK_SIZE*my_buffer->current_start_mark))->next_mark);
            my_buffer->current_start_mark = ((mark_t*)(my_buffer->buffer + MARK_SIZE*my_buffer->current_start_mark))->next_mark;
        }
    }

    /* Now prepare my_mark: make it point to its end, set the time and index, and reset the
     * string (in case another thread tries to display it before it is actually vsprintfed */
    if( my_buffer->current_end_mark + actual_slots >= (parsec_debug_max_history_length_per_thread / MARK_SIZE) )
        my_mark->next_mark = 0;
    else {
        assert( my_buffer->current_end_mark + actual_slots < parsec_debug_max_history_length_per_thread / MARK_SIZE );
        my_mark->next_mark = my_buffer->current_end_mark + actual_slots;
    }
    my_mark->timestamp = take_time();
    my_mark->mark[0] = '\0';
    my_mark->mark_index = my_buffer->current_index++;
    /* And update the writing mark pointer */
    my_buffer->current_end_mark = my_mark->next_mark;

    return my_mark;
}

/**
 * @brief Appends a debug message to the history
 * 
 * @details
 *    This function is similar to printf: it takes the
 *    same arguments. It is not necessary to complete each print
 *    with a carriage return or a line feed, as each history line
 *    will appear independently anyway.
 *
 *  @param[IN] format a printf format
 *  @param[IN] ... the corresponding parameters
 */
void parsec_debug_history_add(const char *format, ...) {
    va_list args;
    mark_t *my_mark;
    int actual_size, actual_space;

    if( parsec_debug_max_history_length_per_thread == 0 )
        return;

    va_start(args, format);
    actual_size = vsnprintf(NULL, 0, format, args);
    va_end(args);

    va_start(args, format);
    my_mark = get_my_mark(actual_size, &actual_space);
    if( actual_space < actual_size && actual_space > 4 ) {
        vsnprintf(my_mark->mark, actual_space-4, format, args);
        snprintf(my_mark->mark + actual_space-4, 4, "...");
    } else {
        vsnprintf(my_mark->mark, actual_space, format, args);
    }
    va_end(args);
}

/**
 * @brief
 *   Outputs on the debug output each line still available in the
 *   debug history.
 *
 * @details
 *   This function is intendend to be called from a debugger or
 *   when the programmer sees fit. It will atomically swap the double
 *   buffers so other threads can continue adding history information
 *   while the printing happens. The debug history is reordered between
 *   all the threads and output.
 *
 *   This has a side effect: the history dumped is forgotten as it
 *   is dumped.
 */
void parsec_debug_history_dump(void) {
    int printing_buffer = writing_buffer;
    parsec_time_t min_ts;
    mark_double_buffer_t *db, *min_db;
    mark_t *min_mark;

    if( parsec_debug_max_history_length_per_thread == 0 )
        return;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     * This CAS can only fail if parsec_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    parsec_atomic_cas_int32(&writing_buffer, printing_buffer, printing_buffer == 0 ? 1 : 0);

    /* As long as there is a mark to display for one thread */
    parsec_inform("== Begin debug history =====================================================");
    while(1) {
        /* Find the thread with the lowest timestamp */
        min_db  = NULL;
        min_mark = NULL;
        for(db = (mark_double_buffer_t*)mark_buffers; db != NULL; db = db->next) {
            if( db->buffers[printing_buffer].current_start_mark != UNDEFINED_MARK ) {
                if(min_db == NULL || time_less( ((mark_t*)&db->buffers[printing_buffer].buffer[db->buffers[printing_buffer].current_start_mark*MARK_SIZE])->timestamp, min_ts)) {
                    min_db = db;
                    min_mark = (mark_t*)&db->buffers[printing_buffer].buffer[db->buffers[printing_buffer].current_start_mark*MARK_SIZE];
                    min_ts = min_mark->timestamp;
                }
            }
        }
        if( min_db == NULL )
            break;
        if( min_mark->mark[0] != '\0') {
            parsec_output(parsec_debug_output, " %p/%lu (%6.03g s) -- %s",
                          (void*)min_db->thread_id,
                          min_mark->mark_index,
                          (double)diff_time(debug_start, min_ts) / 1e9,
                          min_mark->mark);
        } else {
            parsec_output(parsec_debug_output, " %p/%lu (%6.03g s) -- (empty mark)",
                          (void*)min_db->thread_id,
                          min_mark->mark_index,
                          (double)diff_time(debug_start, min_ts) / 1e9);
        }
        min_db->buffers[printing_buffer].current_start_mark = min_mark->next_mark;
        if( min_db->buffers[printing_buffer].current_start_mark == min_db->buffers[printing_buffer].current_end_mark ) {
            min_db->buffers[printing_buffer].current_start_mark = UNDEFINED_MARK;
            min_db->buffers[printing_buffer].current_end_mark = 0;
            ((mark_t*)min_db->buffers[printing_buffer].buffer)->mark[0]='\0';
            ((mark_t*)min_db->buffers[printing_buffer].buffer)->next_mark=0;
        }
    }
    parsec_inform("== End debug history =====================================================");
}

/**
 * @brief cleans the debug history
 *
 * @details
 *   Intended to be called by a debugger or by the developer when
 *   they see fit, this empties the history of the debug buffers of
 *   all threads
 */
void parsec_debug_history_purge(void) {
    mark_double_buffer_t *db;
    int purging_buffer;
    for(db = (mark_double_buffer_t*)mark_buffers; NULL != db; db = db->next) {
        for(purging_buffer = 0; purging_buffer < 2; purging_buffer++) {
            db->buffers[purging_buffer].current_start_mark = UNDEFINED_MARK;
            db->buffers[purging_buffer].current_end_mark = 0;
            ((mark_t*)db->buffers[purging_buffer].buffer)->mark[0]='\0';
            ((mark_t*)db->buffers[purging_buffer].buffer)->next_mark=0;
        }
    }
}

/**
 * @brief initializes the debug history
 *
 * @details
 *   This function in particular reads the level of verbosity required
 *   for the debugging history, and the size of the history buffers.
 *   It initializes the history buffers for the calling thread.
 */
void parsec_debug_history_init(void) {
    int default_history_length = parsec_debug_max_history_length_per_thread;
    int chosen_history_length;

    PARSEC_TLS_KEY_CREATE(tls_debug);

    parsec_mca_param_reg_int_name("debug", "history_verbose",
                                  "Set the output level for debug history ring buffer; same values as debug_verbose",
                                  false, false, parsec_debug_verbose, &parsec_debug_history_verbose);
    parsec_mca_param_reg_int_name("parsec", "debug_max_history_length_per_thread",
                                  "How many bytes of history to keep per thread",
                                  false, false, default_history_length, &chosen_history_length);

    debug_start = take_time();
    if( chosen_history_length > 0 ) {
        parsec_debug_max_history_length_per_thread = MARK_SIZE * ( (chosen_history_length + MARK_SIZE - 1)/ MARK_SIZE);
        parsec_debug_history_init_thread();
    } else {
        parsec_debug_max_history_length_per_thread = 0;
    }
}

/**
 * @brief
 *   Finalizes the debug history
 *
 * @details
 *   Frees all memory allocated by any thread during the execution
 *   and used by the debug history.
 */
void parsec_debug_history_fini(void) {
    uint32_t b;
    mark_double_buffer_t *db, *next;

    for(db = (mark_double_buffer_t*)mark_buffers; NULL != db; db = next) {
        next = db->next;
        for( b = 0; b < 2; b++) {
            free( db->buffers[b].buffer );
        }
        free(db);
    }
    mark_buffers = NULL;
}

#endif /* defined(PARSEC_DEBUG_HISTORY) */
