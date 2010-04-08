/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "debug.h"

#include "dplasma.h"
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
/*
 * Make a good guess about how long a printf-style varargs formatted
 * string will be once all the % escapes are filled in.  We don't
 * handle every % escape here, but we handle enough, and then add a
 * fudge factor in at the end.
 */
static int guess_strlen(const char *fmt, va_list ap)
{
    char *sarg, carg;
    double darg;
    float farg;
    size_t i;
    int iarg;
    int len;
    long larg;

    /* Start off with a fudge factor of 128 to handle the % escapes that
       we aren't calculating here */

    len = (int)strlen(fmt) + 128;
    for (i = 0; i < strlen(fmt); ++i) {
        if ('%' == fmt[i] && i + 1 < strlen(fmt)
            && '%' != fmt[i + 1]) {
            ++i;
            switch (fmt[i]) {
            case 'c':
                carg = va_arg(ap, int);
                len += 1;  /* let's suppose it's a printable char */
                break;
            case 's':
                sarg = va_arg(ap, char *);

                /* If there's an arg, get the strlen, otherwise we'll
                 * use (null) */

                if (NULL != sarg) {
                    len += (int)strlen(sarg);
                } else {
#if defined(DPLASMA_DEBUG)
                    printf("DEBUG WARNING: Got a NULL argument to vasprintf!\n");
#endif  /* defined(DPLASMA_DEBUG) */
                    len += 6;  /* for "(null)" */
                }
                break;

            case 'd':
            case 'i':
                iarg = va_arg(ap, int);
                /* Alloc for minus sign */
                if (iarg < 0)
                    ++len;
                /* Now get the log10 */
                do {
                    ++len;
                    iarg /= 10;
                } while (0 != iarg);
                break;

            case 'x':
            case 'X':
                iarg = va_arg(ap, int);
                /* Now get the log16 */
                do {
                    ++len;
                    iarg /= 16;
                } while (0 != iarg);
                break;

            case 'f':
                farg = (float)va_arg(ap, int);
                /* Alloc for minus sign */
                if (farg < 0) {
                    ++len;
                    farg = -farg;
                }
                /* Alloc for 3 decimal places + '.' */
                len += 4;
                /* Now get the log10 */
                do {
                    ++len;
                    farg /= 10.0;
                } while (0 != farg);
                break;

            case 'g':
                darg = va_arg(ap, int);
                /* Alloc for minus sign */
                if (darg < 0) {
                    ++len;
                    darg = -darg;
                }
                /* Alloc for 3 decimal places + '.' */
                len += 4;
                /* Now get the log10 */
                do {
                    ++len;
                    darg /= 10.0;
                } while (0 != darg);
                break;

            case 'l':
                /* Get %ld %lx %lX %lf */
                if (i + 1 < strlen(fmt)) {
                    ++i;
                    switch (fmt[i]) {
                    case 'x':
                    case 'X':
                        larg = va_arg(ap, int);
                        /* Now get the log16 */
                        do {
                            ++len;
                            larg /= 16;
                        } while (0 != larg);
                        break;

                    case 'f':
                        darg = va_arg(ap, int);
                        /* Alloc for minus sign */
                        if (darg < 0) {
                            ++len;
                            darg = -darg;
                        }
                        /* Alloc for 3 decimal places + '.' */
                        len += 4;
                        /* Now get the log10 */
                        do {
                            ++len;
                            darg /= 10.0;
                        } while (0 != darg);
                        break;

                    case 'd':
                    default:
                        larg = va_arg(ap, int);
                        /* Now get the log10 */
                        do {
                            ++len;
                            larg /= 10;
                        } while (0 != larg);
                        break;
                    }
                }

            default:
                break;
            }
        }
    }

    return len;
}

int vasprintf(char **ptr, const char *fmt, va_list ap)
{
    int length;
    va_list ap2;

    /* va_list might have pointer to internal state and using
       it twice is a bad idea.  So make a copy for the second
       use.  Copy order taken from Autoconf docs. */
#if HAVE_VA_COPY
    va_copy(ap2, ap);
#elif HAVE_UNDERSCORE_VA_COPY
    __va_copy(ap2, ap);
#else
    memcpy (&ap2, &ap, sizeof(va_list));
#endif

    /* guess the size */
    length = guess_strlen(fmt, ap);

    /* allocate a buffer */
    *ptr = (char *) malloc((size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        va_end(ap2);
        return -1;
    }

    /* fill the buffer */
    length = vsprintf(*ptr, fmt, ap2);
#if HAVE_VA_COPY || HAVE_UNDERSCORE_VA_COPY
    va_end(ap2);
#endif  /* HAVE_VA_COPY || HAVE_UNDERSCORE_VA_COPY */

    /* realloc */
    *ptr = (char*) realloc(*ptr, (size_t) length + 1);
    if (NULL == *ptr) {
        errno = ENOMEM;
        return -1;
    }

    return length;
}
#endif  /* !defined(HAVE_VASPRINTF) */

#if defined(DPLASMA_DEBUG_HISTORY)

typedef struct {
    dplasma_t *function;
    assignment_t locals[MAX_LOCAL_COUNT];
} execution_mark_t;

#define TYPE_SEND_ACTIVATE        1
#define TYPE_RECV_ACTIVATE        2
#define TYPE_SEND_GET             3
#define TYPE_RECV_GET             4
#define TYPE_SEND_START_DTA       5
#define TYPE_SEND_END_DTA         6
#define TYPE_RECV_START_DTA       7
#define TYPE_RECV_END_DTA         8

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
    int core;  /**< if core < 0, this is the MPI core, and a message mark. Otherwise it's an execution mark */
    union {
        execution_mark_t     exe;
        communication_mark_t comm;
    } u;
} mark_t;

#define MAX_MARKS 96
static volatile uint32_t nextmark = 0;
static mark_t   marks[MAX_MARKS];

static inline mark_t *get_my_mark(void)
{
    uint32_t mymark_idx = dplasma_atomic_inc_32b(&nextmark) - 1;
    mymark_idx %= MAX_MARKS;
    
    return (&marks[mymark_idx]);
}

void debug_mark_exe(int core, const struct dplasma_execution_context_t *ctx)
{
    int i;
    mark_t  *mymark = get_my_mark();

    if( mymark == NULL )
        return;

    assert(ctx->function->nb_locals < MAX_LOCAL_COUNT);
    mymark->core = core;
    mymark->u.exe.function = ctx->function;
    for(i = 0; i < ctx->function->nb_locals; i++)
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

#define reali(i) (i+nextmark)

void debug_mark_display_history(void)
{
    uint32_t current_mark = nextmark;
    uint32_t i, j;
    mark_t  *m;
    char msg[512];
    int pos, len = 512;
    dplasma_t *f;

    current_mark = current_mark > MAX_MARKS ? MAX_MARKS : current_mark;
    for(i = nextmark % MAX_MARKS; i != (nextmark + MAX_MARKS - 1) % MAX_MARKS; i = (i + 1) % MAX_MARKS) {
        m = &marks[i];
        pos = 0;

        if( m->core < 0 ) {
            /** This is a communication mark */
            switch( m->u.comm.type ) {
            case TYPE_SEND_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of an activate message to %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emision\n",
                                m->u.comm.buffer);
                f = (dplasma_t*)m->u.comm.msg.activate.function;
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(j = 0; j < f->nb_locals; j++) {
                    pos += snprintf(msg+pos, len-pos, "%s=%d%s", 
                                    m->u.comm.msg.activate.locals[j].sym->name,
                                    m->u.comm.msg.activate.locals[j].value,
                                    (j == f->nb_locals - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                m->u.comm.msg.activate.which);
                break;

            case TYPE_RECV_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: reception of an activate message from %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                f = (dplasma_t*)m->u.comm.msg.activate.function;
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(j = 0; j < f->nb_locals; j++) {
                    pos += snprintf(msg+pos, len-pos, "%s=%d%s", 
                                    m->u.comm.msg.activate.locals[j].sym->name,
                                    m->u.comm.msg.activate.locals[j].value,
                                    (j == f->nb_locals - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                m->u.comm.msg.activate.which);
                pos += snprintf(msg+pos, len-pos, "\t      deps = 0x%X\n",
                                m->u.comm.msg.activate.deps);
                break;

            case TYPE_SEND_GET:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of a Get control message to %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emission\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      deps requested = 0x%X\n",
                                m->u.comm.msg.get.deps);
                pos += snprintf(msg+pos, len-pos, "\t      which requested = 0x%08x\n",
                                m->u.comm.msg.get.which);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                m->u.comm.msg.get.tag);
                break;

            case TYPE_RECV_GET:
                pos += snprintf(msg+pos, len-pos, "mark %d: reception of a Get control message from %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      deps requested = 0x%X\n",
                                m->u.comm.msg.get.deps);
                pos += snprintf(msg+pos, len-pos, "\t      which requested = 0x%08x\n",
                                m->u.comm.msg.get.which);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                m->u.comm.msg.get.tag);
                break;

            case TYPE_SEND_START_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Start emitting data to %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emission\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the emission of data = %d\n",
                                m->u.comm.msg.tag);
                break;

            case TYPE_SEND_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done sending data of tag %d\n", 
                                reali(i), m->u.comm.msg.tag);
                break;

            case TYPE_RECV_START_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Start receiving data from %d\n", 
                                reali(i), m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                m->u.comm.msg.tag);
                 break;

            case TYPE_RECV_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done receiving data with tag %d\n", 
                                reali(i), m->u.comm.msg.tag);
                break;
            default: 
                pos += snprintf(msg+pos, len-pos, "mark %d: WAT? type %d\n", reali(i), m->u.comm.type);
                break;
            }
            fprintf(stderr, "%s", msg);
        } else {
            pos += snprintf(msg+pos, len-pos, "mark %d: execution on core %d\n", reali(i), m->core);
            pos += snprintf(msg+pos, len-pos, "\t      %s(", m->u.exe.function->name);
            for(j = 0; j < m->u.exe.function->nb_locals; j++) {
                pos += snprintf(msg+pos, len-pos, "%s=%d%s",
                                m->u.exe.locals[j].sym->name, m->u.exe.locals[j].value,
                                (j == m->u.exe.function->nb_locals-1) ? ")\n" : ", ");
            }
            fprintf(stderr, "%s", msg);
        }
    }
    fprintf(stderr, "DISPLAYING last %u of %u events\n", current_mark, nextmark);
}

#endif
