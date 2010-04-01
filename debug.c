#include "debug.h"

#include "dplasma.h"
#include "remote_dep.h"
#include "atomic.h"

#if defined(DPLASMA_DEBUG)

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
    uint32_t fromto:24;
    uint32_t type  :8;
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

#define MAX_MARKS 8192
static volatile uint32_t nextmark = 0;
static mark_t   marks[MAX_MARKS];

static inline mark_t *get_my_mark(void)
{
    uint32_t mymark_idx = dplasma_atomic_inc_32b(&nextmark) - 1;
 
    if( mymark_idx >= MAX_MARKS ) {
        DEBUG(("Unable to mark more events\n"));
        return NULL;
    }
    return (&marks[mymark_idx]);
}

void debug_mark_exe(int core, const struct dplasma_execution_context_t *ctx)
{
    int i;
    mark_t  *mymark = get_my_mark();

    if( mymark == NULL )
        return;

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

void debug_mark_display_history(void)
{
    uint32_t current_mark = nextmark;
    uint32_t i, j;
    mark_t  *m;
    char msg[512];
    int pos, len = 512;
    dplasma_t *f;

    current_mark = current_mark > MAX_MARKS ? MAX_MARKS : current_mark;
    for(i = 0; i < current_mark; i++) {
        m = &marks[i];
        pos = 0;

        if( m->core < 0 ) {
            /** This is a communication mark */
            switch( m->u.comm.type ) {
            case TYPE_SEND_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of an activate message to %d\n", 
                                i, m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emision\n",
                                m->u.comm.buffer);
                f = (dplasma_t*)m->u.comm.msg.activate.function;
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(i = 0; i < f->nb_locals; i++) {
                    pos += snprintf(msg+pos, len-pos, "%s=%d%s", 
                                    m->u.comm.msg.activate.locals[i].sym->name,
                                    m->u.comm.msg.activate.locals[i].value,
                                    (i == f->nb_locals - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                m->u.comm.msg.activate.which);
                break;

            case TYPE_RECV_ACTIVATE:
                pos += snprintf(msg+pos, len-pos, "mark %d: reception of an activate message from %d\n", 
                                i, m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                f = (dplasma_t*)m->u.comm.msg.activate.function;
                pos += snprintf(msg+pos, len-pos, "\t      Activation passed=%s(", f->name);
                for(i = 0; i < f->nb_locals; i++) {
                    pos += snprintf(msg+pos, len-pos, "%s=%d%s", 
                                    m->u.comm.msg.activate.locals[i].sym->name,
                                    m->u.comm.msg.activate.locals[i].value,
                                    (i == f->nb_locals - 1) ? ")\n" : ", ");
                }
                pos += snprintf(msg+pos, len-pos, "\t      which = 0x%08x\n", 
                                m->u.comm.msg.activate.which);
                pos += snprintf(msg+pos, len-pos, "\t      deps = 0x%X\n",
                                m->u.comm.msg.activate.deps);
                break;

            case TYPE_SEND_GET:
                pos += snprintf(msg+pos, len-pos, "mark %d: emission of a Get control message to %d\n", 
                                i, m->u.comm.fromto);
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
                                i, m->u.comm.fromto);
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
                                i, m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for emission\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the emission of data = %d\n",
                                m->u.comm.msg.tag);
                break;

            case TYPE_SEND_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done sending data of tag %d\n", 
                                i, m->u.comm.msg.tag);
                break;

            case TYPE_RECV_START_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Start receiving data from %d\n", 
                                i, m->u.comm.fromto);
                pos += snprintf(msg+pos, len-pos, "\t      Using buffer %p for reception\n",
                                m->u.comm.buffer);
                pos += snprintf(msg+pos, len-pos, "\t      tag for the reception of data = %d\n",
                                m->u.comm.msg.tag);
                 break;

            case TYPE_RECV_END_DTA:
                pos += snprintf(msg+pos, len-pos, "mark %d: Done receiving data with tag %d\n", 
                                i, m->u.comm.msg.tag);
                break;
            }
            DEBUG(("%s", msg));
        } else {
            pos += snprintf(msg+pos, len-pos, "mark %d: execution on core %d\n", i, m->core);
            pos += snprintf(msg+pos, len-pos, "\t      %s(", m->u.exe.function->name);
            for(j = 0; j < m->u.exe.function->nb_locals; j++) {
                pos += snprintf(msg+pos, len-pos, "%s=%d%s",
                                m->u.exe.locals[j].sym->name, m->u.exe.locals[j].value,
                                (j == m->u.exe.function->nb_locals-1) ? ")\n" : ", ");
            }
            DEBUG(("%s", msg));
        }
    }
}

#endif
