/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "remote_dep.h"
#include <dague/sys/atomic.h>

#if defined(HAVE_ERRNO_H)
#include <errno.h>
#endif  /* defined(HAVE_ERRNO_H) */
#include <stdio.h>

#include <execinfo.h>

int dague_verbose = 0;
#define ST_SIZE 128
#define ST_ASIZE 64
static uint32_t st_idx = 0;
static void *stack[ST_ASIZE][ST_SIZE];
static int   stack_size[ST_ASIZE];

void debug_save_stack_trace(void)
{
    uint32_t my_idx = dague_atomic_inc_32b( &st_idx ) % ST_ASIZE;
    stack_size[my_idx] = backtrace( stack[my_idx], ST_SIZE );
}

void debug_dump_stack_traces(void)
{
    int i, my, r = 0, t;
    char **s;
#if defined(HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
#endif

    for(i = 0; i < ST_ASIZE; i++) {
        my = (st_idx + i) % ST_ASIZE;
        fprintf(stderr, "[%d] --- %u ---\n", r, st_idx + i);
        s = backtrace_symbols(stack[my], stack_size[my]);
        for(t = 0; t < stack_size[my]; t++) {
            fprintf(stderr, "[%d]  %s\n", r, s[t]);
        }
        free(s);
        fprintf(stderr, "[%d]\n", r);
    }
}

#if !defined(HAVE_ASPRINTF)
int asprintf(char **ptr, const char *fmt, ...)
{
    int length;
    va_list ap;

    va_start(ap, fmt);
    length = vasprintf(ptr, fmt, ap);
    va_end(ap);

    return length;
}
#endif  /* !defined(HAVE_ASPRINTF) */

#if !defined(HAVE_VASPRINTF)
int vasprintf(char **ptr, const char *fmt, va_list ap)
{
    int length;
    va_list ap2;
    char* temp = (char*)malloc(64);

    /* va_list might have pointer to internal state and using
       it twice is a bad idea.  So make a copy for the second
       use.  Copy order taken from Autoconf docs. */
#if defined(HAVE_VA_COPY)
    va_copy(ap2, ap);
#elif defined(HAVE_UNDERSCORE_VA_COPY)
    __va_copy(ap2, ap);
#else
    memcpy (&ap2, &ap, sizeof(va_list));
#endif

    /* guess the size using a nice feature of snprintf and friends:
     *
     *  The functions snprintf() and vsnprintf() do not write more than size bytes (including
     *  the  trailing  '\0').  If the output was truncated due to this limit then the return
     *  value is the number of characters (not including the trailing '\0') which  would
     *  have  been written  to  the  final  string  if enough space had been available.
     */
    length = vsnprintf(temp, 64, fmt, ap);
    free(temp);

    /* allocate a buffer */
    *ptr = (char *) malloc((size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        va_end(ap2);
        return -1;
    }

    /* fill the buffer */
    length = vsprintf(*ptr, fmt, ap2);
#if defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY)
    va_end(ap2);
#endif  /* defined(HAVE_VA_COPY) || defined(HAVE_UNDERSCORE_VA_COPY) */

    /* realloc */
    *ptr = (char*) realloc(*ptr, (size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        return -1;
    }

    return length;
}
#endif  /* !defined(HAVE_VASPRINTF) */

#if defined(DAGUE_DEBUG_HISTORY)

#define MAX_MARKS 96

typedef struct {
    volatile uint32_t nextmark;
    char *marks[MAX_MARKS];
} mark_buffer_t;

static mark_buffer_t marks_A = {.nextmark = 0},
                     marks_B = {.nextmark = 0};
static mark_buffer_t  *marks = &marks_A;

static inline void set_my_mark(const char *newm)
{
    uint32_t mymark_idx = dague_atomic_inc_32b(&marks->nextmark) - 1;
    char *oldm;
    mymark_idx %= MAX_MARKS;
    
    do {
        oldm = marks->marks[mymark_idx];
    } while( !dague_atomic_cas( &marks->marks[mymark_idx], oldm, newm ) );
    if( oldm != NULL )
        free(oldm);
}

void dague_debug_history_add(const char *format, ...)
{
    char* debug_str;
    va_list args;
    
    va_start(args, format);
    vasprintf(&debug_str, format, args);
    va_end(args);
    
    set_my_mark( debug_str );
}

void debug_mark_exe(int th, int vp, const struct dague_execution_context_s *ctx)
{
    int j;
    char msg[512];
    int pos = 0;
    int len = 512;

    pos += snprintf(msg+pos, len-pos, "%s(", ctx->function->name);
    for(j = 0; j < ctx->function->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s",
                        j, ctx->function->locals[j]->name, ctx->locals[j].value,
                        (j == ctx->function->nb_parameters-1) ? ")\n" : ", ");
    }

    dague_debug_history_add("Mark: execution on thread %d of VP %d:\t%s",
                            th, vp, msg);
}

void debug_mark_ctl_msg_activate_sent(int to, const void *b, const struct remote_dep_wire_activate_s *m)
{
    int j;
    char msg[512];
    int pos = 0;
    int len = 512;
    dague_handle_t *object;
    const dague_function_t *f;

    pos += snprintf(msg+pos, len-pos, "Mark: emission of an activate message to %d\n", to);
    pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emision\n", b);
    object = dague_handle_lookup( m->handle_id );
    f = object->functions_array[m->function_id];
    pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
    for(j = 0; j < f->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s", 
                        j,
                        f->locals[j]->name, m->locals[j].value,
                        (j == f->nb_parameters - 1) ? ")\n" : ", ");
    }
    pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                    (uint32_t)m->which);

    /* Do not use set_my_mark: msg is a stack-allocated buffer */
    dague_debug_history_add("%s", msg);
}

void debug_mark_ctl_msg_activate_recv(int from, const void *b, const struct remote_dep_wire_activate_s *m)
{
    int j;
    char msg[512];
    int pos = 0;
    int len = 512;
    dague_handle_t *object;
    const dague_function_t *f;

    pos += snprintf(msg+pos, len-pos, "Mark: reception of an activate message from %d\n", from);
    pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n", b);
    object = dague_handle_lookup( m->handle_id );
    f = object->functions_array[m->function_id];
    pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
    for(j = 0; j < f->nb_parameters; j++) {
        pos += snprintf(msg+pos, len-pos, "locals[%d](%s)=%d%s", 
                        j,
                        f->locals[j]->name, m->locals[j].value,
                        (j == f->nb_parameters - 1) ? ")\n" : ", ");
    }
    pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                    (uint32_t)m->which);
    pos += snprintf(msg+pos, len-pos, "\t      deps = 0x%X\n",
                    (uint32_t)m->deps);

    /* Do not use set_my_mark: msg is a stack-allocated buffer */
    dague_debug_history_add("%s", msg);
}

void debug_mark_ctl_msg_get_sent(int to, const void *b, const struct remote_dep_wire_get_s *m)
{
    dague_debug_history_add("Mark: emission of a Get control message to %d\n"
                            "\t      Using buffer %p for emission\n"
                            "\t      deps requested = 0x%X\n"
                            "\t      which requested = 0x%08x\n"
                            "\t      tag for the reception of data = %d\n", 
                            to, b, m->deps, m->which, m->tag);
}

void debug_mark_ctl_msg_get_recv(int from, const void *b, const struct remote_dep_wire_get_s *m)
{
    dague_debug_history_add("Mark: reception of a Get control message from %d\n"
                            "\t      Using buffer %p for reception\n"
                            "\t      deps requested = 0x%X\n"
                            "\t      which requested = 0x%08x\n"
                            "\t      tag for the reception of data = %d\n", 
                            from, b, m->deps, m->which, m->tag);
}

void debug_mark_dta_msg_start_send(int to, const void *b, int tag)
{
    dague_debug_history_add("Mark: Start emitting data to %d\n"
                            "\t      Using buffer %p for emission\n"
                            "\t      tag for the emission of data = %d\n", 
                            to, b, tag);
}

void debug_mark_dta_msg_end_send(int tag)
{
    dague_debug_history_add("Mark: Done sending data of tag %d\n", tag);
}

void debug_mark_dta_msg_start_recv(int from, const void *b, int tag)
{
    dague_debug_history_add("Mark: Start receiving data from %d\n"
                            "\t      Using buffer %p for reception\n"
                            "\t      tag for the reception of data = %d\n",
                            from, b, tag);
}

void debug_mark_dta_msg_end_recv(int tag)
{
    dague_debug_history_add("Mark: Done receiving data with tag %d\n", tag);
}

void debug_mark_display_history(void)
{
    int current_mark, ii;
    char *gm;
    mark_buffer_t *cmark, *nmark;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     */
    cmark = marks;
    nmark = (marks == &marks_A ? &marks_B : &marks_A );
    nmark->nextmark = 0;
    /* This CAS can only fail if debug_mark_display_history is called
     * in parallel by two threads. The atomic swap is not wanted for that,
     * it is wanted to avoid reading from the buffer that is being used to
     * push new marks.
     */
    dague_atomic_cas( &marks, cmark, nmark );

    current_mark = cmark->nextmark > MAX_MARKS ? MAX_MARKS : cmark->nextmark;
    for(ii = 0; ii < MAX_MARKS; ii++) {
        int i = ((int)cmark->nextmark + ii) % MAX_MARKS;
        do {
            gm = cmark->marks[i];
        } while( !dague_atomic_cas( &cmark->marks[i], gm, NULL ) );
        if( gm != NULL ) {
            _DAGUE_OUTPUT("..", ("%s", gm));
            free(gm);
        } else {
            if(dague_verbose) _DAGUE_OUTPUT("^.", ("A mark here was already displayed, or has not been pushed yet\n"));
        }
    }
    if(dague_verbose) _DAGUE_OUTPUT("^.", ("DISPLAYED last %d of %u events pushed since last display\n", current_mark, cmark->nextmark));
}

void debug_mark_purge_history(void)
{
    int ii;
    char *gm;
    mark_buffer_t *cmark, *nmark;

    /* Atomically swap the current marks buffer, to avoid the case when we read
     * something that is changing
     */
    cmark = marks;
    nmark = (marks == &marks_A ? &marks_B : &marks_A );
    nmark->nextmark = 0;
    /* This CAS can only fail if debug_mark_display_history is called
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

void debug_mark_purge_all_history(void) {
    debug_mark_purge_history();
    debug_mark_purge_history();
}

#endif
