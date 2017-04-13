/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/debug.h"
#include "parsec/utils/output.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"

#include <stdio.h>
#include <string.h>
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* PARSEC_HAVE_UNISTD_H */

/* globals for use in macros from debug.h */
char parsec_debug_hostname[32]   = "unknownhost";
int parsec_debug_rank            = -1;
int parsec_debug_output          = 0;
int parsec_debug_verbose         = 1;
int parsec_debug_history_verbose = 4;
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
        , false, false, 2, &parsec_debug_verbose);
    parsec_output_set_verbosity(parsec_debug_output, parsec_debug_verbose);
    parsec_output_set_verbosity(0, parsec_debug_verbose);

#if defined(PARSEC_DEBUG_HISTORY)
    parsec_mca_param_reg_int_name("debug", "history_verbose",
        "Set the output level for debug history ring buffer; same values as debug_verbose"
        , false, false, 5, &parsec_debug_history_verbose);
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

    parsec_debug_history_purge();
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

#define MAX_MARKS 96

typedef struct {
    volatile uint32_t nextmark;
    char *marks[MAX_MARKS];
} mark_buffer_t;

static mark_buffer_t marks_A = {.nextmark = 0},
                     marks_B = {.nextmark = 0};
static mark_buffer_t  *marks = &marks_A;

static inline void set_my_mark(const char *newm) {
    uint32_t mymark_idx = parsec_atomic_inc_32b(&marks->nextmark) - 1;
    char *oldm;
    mymark_idx %= MAX_MARKS;

    do {
        oldm = marks->marks[mymark_idx];
    } while( !parsec_atomic_cas_ptr( &marks->marks[mymark_idx], oldm, newm ) );
    if( oldm != NULL )
        free(oldm);
}

void parsec_debug_history_add(const char *format, ...) {
    char* debug_str;
    va_list args;

    va_start(args, format);
    vasprintf(&debug_str, format, args);
    va_end(args);

    set_my_mark(debug_str);
}

void parsec_debug_history_dump(void) {
    int current_mark, ii;
    char *gm;
    mark_buffer_t *cmark, *nmark;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     */
    cmark = marks;
    nmark = (marks == &marks_A ? &marks_B : &marks_A );
    nmark->nextmark = 0;
    /* This CAS can only fail if parsec_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    parsec_atomic_cas_ptr( &marks, cmark, nmark );

    current_mark = cmark->nextmark > MAX_MARKS ? MAX_MARKS : cmark->nextmark;
    parsec_inform("== Displaying debug history of the last %d of %u events pushed since last dump", current_mark, cmark->nextmark);
    for(ii = 0; ii < MAX_MARKS; ii++) {
        int i = ((int)cmark->nextmark + ii) % MAX_MARKS;
        do {
            gm = cmark->marks[i];
        } while( !parsec_atomic_cas_ptr( &cmark->marks[i], gm, NULL ) );
        if( gm != NULL ) {
            parsec_output(parsec_debug_output, " %s", gm);
            free(gm);
        } else {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "A mark has not been stored at this position since the last dump");
        }
    }
    parsec_inform("== End debug history =====================================================");
}

static void debug_history_purge_one(void) {
    int ii;
    char *gm;
    mark_buffer_t *cmark, *nmark;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     */
    cmark = marks;
    nmark = (marks == &marks_A ? &marks_B : &marks_A );
    nmark->nextmark = 0;
    /* This CAS can only fail if parsec_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    parsec_atomic_cas_ptr( &marks, cmark, nmark );

    for(ii = 0; ii < MAX_MARKS; ii++) {
        int i = ((int)cmark->nextmark + ii) % MAX_MARKS;
        do {
            gm = cmark->marks[i];
        } while( !parsec_atomic_cas_ptr( &cmark->marks[i], gm, NULL ) );
        if( gm != NULL ) {
            free(gm);
        }
    }
}

void parsec_debug_history_purge(void) {
    debug_history_purge_one();
    debug_history_purge_one();
}

#endif /* defined(PARSEC_DEBUG_HISTORY) */
