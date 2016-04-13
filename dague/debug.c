/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/debug.h"
#include "dague/utils/output.h"
#include "dague/sys/atomic.h"
#include "dague/utils/mca_param.h"

#include <stdio.h>
#include <string.h>
#if defined(DAGUE_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* DAGUE_HAVE_UNISTD_H */

/* globals for use in macros from debug.h */
char dague_debug_hostname[32]   = "unknownhost";
int dague_debug_rank            = -1;
int dague_debug_output          = 0;
int dague_debug_verbose         = 1;
int dague_debug_colorize        = 10; /* 10 is the size of the format string for colors */
int dague_debug_coredump_on_abort = 0;

/* debug backtrace circular buffer */
static int bt_output    = -1;
static int ST_SIZE      = 128;
static int ST_ASIZE     = 64;
static uint32_t st_idx  = 0;
static void **stack     = NULL;
static int* stack_size  = NULL;


#if defined(DISTRIBUTED) && defined(DAGUE_HAVE_MPI)
#include <mpi.h>
#endif

void dague_debug_init(void) {
#if defined(DISTRIBUTED) && defined(DAGUE_HAVE_MPI)
    int is_mpi_up;
    MPI_Initialized(&is_mpi_up);
    if( 0 == is_mpi_up ) {
        return ;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &dague_debug_rank);
#endif
    gethostname(dague_debug_hostname, sizeof(dague_debug_hostname));

    dague_debug_output = dague_output_open(NULL);

    dague_mca_param_reg_int_name("debug", "verbose",
        "Set the output level for debug messages"
        ", 0: Errors only"
        ", 1: Warnings (minimum recommended)"
        ", 2: Info (default)"
        ", 3-4: User Debug"
        ", 5-9: Devel Debug"
        ", >=10: Chatterbox Debug"
#if !defined(DAGUE_DEBUG_PARANOID) || !defined(DAGUE_DEBUG_NOISIER) || !defined(DAGUE_DEBUG_HISTORY)
        " (heaviest debug output available only when compiling with DAGUE_DEBUG_PARANOID, DAGUE_DEBUG_NOISIER and/or DAGUE_DEBUG_HISTORY in ccmake)"
#endif
        , false, false, 2, &dague_debug_verbose);
    dague_output_set_verbosity(dague_debug_output, dague_debug_verbose);
    dague_output_set_verbosity(0, dague_debug_verbose);

    dague_mca_param_reg_int_name("debug", "color",
        "Toggle on/off color output for debug messages",
        false, false, 1, &dague_debug_colorize);
    dague_debug_colorize = dague_debug_colorize? 10: 0;

    dague_mca_param_reg_int_name("debug", "coredump_on_abort",
        "Toggle on/off raise sigabort on internal engine error",
        false, false, 0, &dague_debug_coredump_on_abort);
    dague_debug_coredump_on_abort = dague_debug_coredump_on_abort ? 1: 0;

    /* We do not want backtraces in the syslog, so, we do not
     * inherit the defaults... */
    char* opt;
    dague_output_stream_t lds;
    dague_mca_param_reg_string_name("debug", "backtrace_output",
        "Define the output for the backtrace dumps (none, stderr, file)",
        false, false, "file", &opt);
    if( 0 == strcasecmp(opt, "none") ) {
        bt_output = -1;
    }
    else if( 0 == strcasecmp(opt, "stderr") ) {
        OBJ_CONSTRUCT(&lds, dague_output_stream_t);
        lds.lds_want_stderr = true;
        lds.lds_want_syslog = false;
        bt_output = dague_output_open(&lds);
        OBJ_DESTRUCT(&lds);
    }
    else if( 0 == strcasecmp(opt, "file") ) {
        OBJ_CONSTRUCT(&lds, dague_output_stream_t);
        lds.lds_want_file = true;
        lds.lds_want_syslog = false;
        lds.lds_file_suffix = "backtraces";
        bt_output = dague_output_open(&lds);
        OBJ_DESTRUCT(&lds);
    }
    else {
        dague_warning("Invalid value %s for parameter debug_backtrace_output", opt);
    }
    free(opt);

    dague_mca_param_reg_int_name("debug", "backtrace_keep",
        "Maximum number of backtrace to keep in backtrace circular buffer",
        false, false, ST_ASIZE, &ST_ASIZE);
    dague_mca_param_reg_int_name("debug", "backtrace_size",
        "Maximum size for each backtrace",
        false, false, ST_SIZE, &ST_SIZE);
    if( -1 != bt_output ) {
        stack = malloc(ST_ASIZE*ST_SIZE*sizeof(void*));
        stack_size = malloc(ST_ASIZE*sizeof(int));
        if( NULL == stack_size
         || NULL == stack ) {
             dague_warning("Backtrace debug framework DISABLED: could not allocate the backtrace circular buffer with backtrace_keep=%d and backtrace_size=%d", ST_ASIZE, ST_SIZE);
             if( NULL != stack_size ) free(stack_size);
             if( NULL != stack ) free(stack);
             if( bt_output > 0 ) {
                 dague_output_close(bt_output);
                 bt_output = -1;
                 return;
             }
         }
         memset(stack_size, 0, ST_ASIZE*sizeof(int));
         memset(stack, 0, ST_ASIZE*ST_SIZE*sizeof(int));
    }
}

void dague_debug_fini(void)
{
    if( 0 < dague_debug_output ) {
        dague_output_close(dague_debug_output);
    }

    if( 0 < bt_output ) {
        dague_output_close(bt_output);
    }
    if( NULL != stack_size ) free(stack_size);
    if( NULL != stack ) free(stack);

    dague_debug_history_purge();
    if( 0 < dague_debug_output ) {
        dague_output_close(dague_debug_output);
    }
}


/* STACKTRACES circular buffer */
#include <execinfo.h>

void dague_debug_backtrace_save(void) {
    uint32_t my_idx = dague_atomic_inc_32b(&st_idx) % ST_ASIZE;
    stack_size[my_idx] = backtrace(&stack[my_idx*ST_SIZE], ST_SIZE);
}

void dague_debug_backtrace_dump(void) {
    int i, my, r = dague_debug_rank, t;
    char **s;

    for(i = 0; i < ST_ASIZE; i++) {
        my = (st_idx + i) % ST_ASIZE;
        if( NULL == stack[my*ST_SIZE] ) continue;
        dague_output(bt_output, "[%d] --- %u ---\n", r, st_idx + i);
        s = backtrace_symbols(&stack[my*ST_SIZE], stack_size[my]);
        for(t = 0; t < stack_size[my]; t++) {
            dague_output(bt_output, "[%d]  %s\n", r, s[t]);
        }
        free(s);
        dague_output(bt_output, "[%d]\n", r);
    }
}

/* DEBUG HISTORY circular buffer */

#if defined(DAGUE_DEBUG_HISTORY)

#define MAX_MARKS 96

typedef struct {
    volatile uint32_t nextmark;
    char *marks[MAX_MARKS];
} mark_buffer_t;

static mark_buffer_t marks_A = {.nextmark = 0},
                     marks_B = {.nextmark = 0};
static mark_buffer_t  *marks = &marks_A;

static inline void set_my_mark(const char *newm) {
    uint32_t mymark_idx = dague_atomic_inc_32b(&marks->nextmark) - 1;
    char *oldm;
    mymark_idx %= MAX_MARKS;

    do {
        oldm = marks->marks[mymark_idx];
    } while( !dague_atomic_cas( &marks->marks[mymark_idx], oldm, newm ) );
    if( oldm != NULL )
        free(oldm);
}

void dague_debug_history_add(const char *format, ...) {
    char* debug_str;
    va_list args;

    va_start(args, format);
    vasprintf(&debug_str, format, args);
    va_end(args);

    set_my_mark(debug_str);
}

void dague_debug_history_dump(void) {
    int current_mark, ii;
    char *gm;
    mark_buffer_t *cmark, *nmark;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     */
    cmark = marks;
    nmark = (marks == &marks_A ? &marks_B : &marks_A );
    nmark->nextmark = 0;
    /* This CAS can only fail if dague_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    dague_atomic_cas( &marks, cmark, nmark );

    current_mark = cmark->nextmark > MAX_MARKS ? MAX_MARKS : cmark->nextmark;
    dague_inform("== Displaying debug history of the last %d of %u events pushed since last dump", current_mark, cmark->nextmark);
    for(ii = 0; ii < MAX_MARKS; ii++) {
        int i = ((int)cmark->nextmark + ii) % MAX_MARKS;
        do {
            gm = cmark->marks[i];
        } while( !dague_atomic_cas( &cmark->marks[i], gm, NULL ) );
        if( gm != NULL ) {
            dague_output(dague_debug_output, " %s", gm);
            free(gm);
        } else {
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "A mark has not been stored at this position since the last dump");
        }
    }
    dague_inform("== End debug history =====================================================");
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
    /* This CAS can only fail if dague_debug_history_dump is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    dague_atomic_cas( &marks, cmark, nmark );

    for(ii = 0; ii < MAX_MARKS; ii++) {
        int i = ((int)cmark->nextmark + ii) % MAX_MARKS;
        do {
            gm = cmark->marks[i];
        } while( !dague_atomic_cas( &cmark->marks[i], gm, NULL ) );
        if( gm != NULL ) {
            free(gm);
        }
    }
}

void dague_debug_history_purge(void) {
    debug_history_purge_one();
    debug_history_purge_one();
}

#endif /* defined(DAGUE_DEBUG_HISTORY) */
