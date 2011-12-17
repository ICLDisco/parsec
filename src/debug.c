/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "debug.h"

#include "dague.h"
#include "remote_dep.h"
#include "atomic.h"

#if defined(HAVE_ERRNO_H)
#include <errno.h>
#endif  /* defined(HAVE_ERRNO_H) */

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

typedef struct {
    const dague_function_t *function;
    assignment_t            locals[MAX_LOCAL_COUNT];
} execution_mark_t;

#define TYPE_SEND_ACTIVATE        1
#define TYPE_RECV_ACTIVATE        2
#define TYPE_SEND_GET             3
#define TYPE_RECV_GET             4
#define TYPE_SEND_START_DTA       5
#define TYPE_SEND_END_DTA         6
#define TYPE_RECV_START_DTA       7
#define TYPE_RECV_END_DTA         8
#define TYPE_GENERAL_MESSAGE      9

typedef struct {
    uint32_t fromto;
    uint32_t type;
    const void *buffer;
    union {
        remote_dep_wire_activate_t activate;
        remote_dep_wire_get_t      get;
        uint32_t tag;
    } msg;
} communication_mark_t;

typedef struct {
    int core;  /**< if core == -1, this is the MPI core, and a message mark. 
                  < if core == -2, this is a general message, without core information.
                  <  Otherwise it's an execution mark */
    union {
        execution_mark_t     exe;
        communication_mark_t comm;
        char                *general_message;
    } u;
} mark_t;

#define MAX_MARKS 96

typedef struct {
    volatile uint32_t nextmark;
    mark_t marks[MAX_MARKS];
} mark_buffer_t;

static mark_buffer_t marks_A = {.nextmark = 0},
                     marks_B = {.nextmark = 0};
static mark_buffer_t  *marks = &marks_A;

static inline mark_t *get_my_mark(void)
{
    uint32_t mymark_idx = dague_atomic_inc_32b(&marks->nextmark) - 1;
    char *m;
    mymark_idx %= MAX_MARKS;
    
    if( marks->marks[mymark_idx].core == -2 ) {
        /* This was a general message, let's free it before we get this mark back */
        do {
            m = marks->marks[mymark_idx].u.general_message;
        } while( !dague_atomic_cas( &marks->marks[mymark_idx].u.general_message, m, NULL ) );
        /* This could happen if somebody is printing at the same time as I'm getting this */
        if( m != NULL )
            free(m);
    }

    return (&marks->marks[mymark_idx]);
}

void dague_debug_history_add(const char *format, ...)
{
    mark_t *mark = get_my_mark();
    char* debug_str;
    va_list args;
    
    va_start(args, format);
    vasprintf(&debug_str, format, args);
    va_end(args);
    
    mark->core = -2;
    mark->u.general_message = debug_str;
}

void debug_mark_exe(int core, const struct dague_execution_context_t *ctx)
{
    int i;
    mark_t  *mymark = get_my_mark();

    if( mymark == NULL )
        return;

    assert(ctx->function->nb_parameters < MAX_LOCAL_COUNT);
    mymark->core = core;
    mymark->u.exe.function = ctx->function;
    for(i = 0; i < ctx->function->nb_parameters; i++)
        mymark->u.exe.locals[i] = ctx->locals[i];
}

void debug_mark_ctl_msg_activate_sent(int to, const void *b, const struct remote_dep_wire_activate_t *m)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto = to;
    mymark->u.comm.type   = TYPE_SEND_ACTIVATE;
    mymark->u.comm.buffer = b;
    memcpy(&mymark->u.comm.msg.activate, m, sizeof(remote_dep_wire_activate_t));
}

void debug_mark_ctl_msg_activate_recv(int from, const void *b, const struct remote_dep_wire_activate_t *m)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto = from;
    mymark->u.comm.type   = TYPE_RECV_ACTIVATE;
    mymark->u.comm.buffer = b;
    memcpy(&mymark->u.comm.msg.activate, m, sizeof(remote_dep_wire_activate_t));
}

void debug_mark_ctl_msg_get_sent(int to, const void *b, const struct remote_dep_wire_get_t *m)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto = to;
    mymark->u.comm.type   = TYPE_SEND_GET;
    mymark->u.comm.buffer = b;
    memcpy(&mymark->u.comm.msg.get, m, sizeof(remote_dep_wire_get_t));
}

void debug_mark_ctl_msg_get_recv(int from, const void *b, const remote_dep_wire_get_t *m)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto = from;
    mymark->u.comm.type   = TYPE_RECV_GET;
    mymark->u.comm.buffer = b;
    memcpy(&mymark->u.comm.msg.get, m, sizeof(remote_dep_wire_get_t));
}

void debug_mark_dta_msg_start_send(int to, const void *b, int tag)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto  = to;
    mymark->u.comm.type    = TYPE_SEND_START_DTA;
    mymark->u.comm.buffer  = b;
    mymark->u.comm.msg.tag = tag;
}

void debug_mark_dta_msg_end_send(int tag)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.type    = TYPE_SEND_END_DTA;
    mymark->u.comm.msg.tag = tag;
}

void debug_mark_dta_msg_start_recv(int from, const void *b, int tag)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.fromto  = from;
    mymark->u.comm.type    = TYPE_RECV_START_DTA;
    mymark->u.comm.buffer  = b;
    mymark->u.comm.msg.tag = tag;
}

void debug_mark_dta_msg_end_recv(int tag)
{
    mark_t *mymark = get_my_mark();
    if( mymark == NULL )
        return;

    mymark->core = -1;
    mymark->u.comm.type    = TYPE_RECV_END_DTA;
    mymark->u.comm.msg.tag = tag;
}

#define reali(m, i) (int)(i+m->nextmark)

void debug_mark_display_history(void)
{
    int current_mark;
    int i, j;
    mark_t  *m;
    char msg[512];
    int pos, len = 512;
    const dague_function_t *f;
    const dague_object_t* object;
    char *gm;
    mark_buffer_t *cmark, *nmark;
    int rank;
#if defined(HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

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
    for(i = ( (int)cmark->nextmark % MAX_MARKS); i != ( (int)cmark->nextmark + MAX_MARKS - 1) % MAX_MARKS; i = (i + 1) % MAX_MARKS) {
        m = &cmark->marks[i];
        pos = 0;

        if( m->core == -1 ) {
            /** This is a communication mark */
            switch( m->u.comm.type ) {
            case TYPE_SEND_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of an activate message to %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emision\n",
                                m->u.comm.buffer);
                object = dague_object_lookup( m->u.comm.msg.activate.object_id );
                if( object == NULL || object->functions_array[m->u.comm.msg.activate.function_id] == NULL ) {
                    pos += snprintf(msg+pos, len-pos, "\t    Message changed during reading.\n");
                    break;
                }
                f = object->functions_array[m->u.comm.msg.activate.function_id];
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(j = 0; j < f->nb_parameters; j++) {
                    pos += snprintf(msg+pos, len-pos, "locals[%u]=%d%s", 
                                    j,
                                    m->u.comm.msg.activate.locals[j].value,
                                    (j == f->nb_parameters - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                (uint32_t)m->u.comm.msg.activate.which);
                break;

            case TYPE_RECV_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: reception of an activate message from %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                object = dague_object_lookup( m->u.comm.msg.activate.object_id );
                if( object == NULL || object->functions_array[m->u.comm.msg.activate.function_id] == NULL ) {
                    pos += snprintf(msg+pos, len-pos, "\t    Message changed during reading.\n");
                    break;
                }
                f = object->functions_array[m->u.comm.msg.activate.function_id];
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(j = 0; j < f->nb_parameters; j++) {
                    pos += snprintf(msg+pos, len-pos, "locals[%u]=%d%s", 
                                    j,
                                    m->u.comm.msg.activate.locals[j].value,
                                    (j == f->nb_parameters - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                (uint32_t)m->u.comm.msg.activate.which);
                pos += snprintf(msg+pos, len-pos, "\t      deps = 0x%X\n",
                                (uint32_t)m->u.comm.msg.activate.deps);
                break;

            case TYPE_SEND_GET:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of a Get control message to %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emission\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      deps requested = 0x%X\n",
                                (uint32_t)m->u.comm.msg.get.deps);
                pos += snprintf(msg+pos, len-pos, "\t      which requested = 0x%08x\n",
                                (uint32_t)m->u.comm.msg.get.which);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                (int)m->u.comm.msg.get.tag);
                break;

            case TYPE_RECV_GET:
                pos += snprintf(msg+pos, len-pos, "mark %d: reception of a Get control message from %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      deps requested = 0x%X\n",
                                (uint32_t)m->u.comm.msg.get.deps);
                pos += snprintf(msg+pos, len-pos, "\t      which requested = 0x%08x\n",
                                (uint32_t)m->u.comm.msg.get.which);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                (int)m->u.comm.msg.get.tag);
                break;

            case TYPE_SEND_START_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Start emitting data to %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emission\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the emission of data = %d\n",
                                (int)m->u.comm.msg.tag);
                break;

            case TYPE_SEND_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done sending data of tag %d\n", 
                                reali(cmark, i), (int)m->u.comm.msg.tag);
                break;

            case TYPE_RECV_START_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Start receiving data from %d\n", 
                                reali(cmark, i), (int)m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                (int)m->u.comm.msg.tag);
                 break;

            case TYPE_RECV_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done receiving data with tag %d\n", 
                                reali(cmark, i), (int)m->u.comm.msg.tag);
                break;
            default: 
                pos += snprintf(msg+pos, len-pos, "mark %d: WAT? type %d\n", 
                                reali(cmark, i), (int)m->u.comm.type);
                break;
            }
            fprintf(stderr, "[%d]: %s", rank, msg);
        } else if( m->core >= 0 ) {
            pos += snprintf(msg+pos, len-pos, "mark %d: execution on core %d\n", reali(cmark, i), (int)m->core);
            if( m->u.exe.function == NULL ) {
                pos += snprintf(msg+pos, len-pos, "\t      Message changed while reading it, ignored\n");
            } else {
                pos += snprintf(msg+pos, len-pos, "\t      %s(", m->u.exe.function->name);
                for(j = 0; j < m->u.exe.function->nb_parameters; j++) {
                    pos += snprintf(msg+pos, len-pos, "locals[%u]=%d%s",
                                    j, m->u.exe.locals[j].value,
                                    (j == m->u.exe.function->nb_parameters-1) ? ")\n" : ", ");
                }
            }
            fprintf(stderr, "[%d]: %s", rank, msg);
        } else if( m->core == -2) {
            do {
                gm = m->u.general_message;
            } while( !dague_atomic_cas( &m->u.general_message, gm, NULL ) );
            if( gm != NULL ) {
                fprintf(stderr, "[%d]: %s", rank, gm);
                free(gm);
            } else {
                fprintf(stderr, "[%d]: -- This mark was already displayed, or has not been pushed yet\n", rank);
            }
        } else {
            fprintf(stderr, "Unknown mark type %d\n", m->core);
        }
    }
    fprintf(stderr, "DISPLAYED last %u of %u events pushed since last display\n", current_mark, cmark->nextmark);
}

#endif
