/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>

#include "parsec_config.h"
#include "parsec_internal.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/profiling.h"
#include "parsec/datarepo.h"
#include "parsec/bindthread.h"
#include "parsec/execution_unit.h"
#include "parsec/vpmap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/os-spec-timing.h"
#include "parsec/remote_dep.h"

#include "parsec/ayudame.h"
#include "parsec/constants.h"
#include "parsec/thread/thread.h"

#define SEND_BUF_LEN    65 /*np 64 + '\0'*/
#define RECV_BUF_LEN    65
#define PORT0           34242

static inline char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp) {
        if (a_delim == *tmp) {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);
    count++;

    result = malloc(sizeof(char*) * count);

    if (result) {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token) {
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        *(result + idx) = 0;
    }
    return result;
}

static inline void handle_error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

static inline size_t safe_write(int fd, const char* buf)
{
    size_t c = 0, size = strlen(buf);
    ssize_t i;

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

    i = write(fd, (char*)&size, sizeof(size_t));
    if ( i < 0 ) handle_error("Error: write size");

    while ( c < size ) {
        i = write(fd, buf+c, size-c);
        if ( i < 0 ) handle_error("Error: write");
        c += i;
    }

    flag = 0;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

    return c;
}

static inline size_t safe_read(int fd, char* buf)
{
    size_t c = 0, size;
    ssize_t i;

    i = read(fd, (char*)&size, sizeof(size_t));
    if ( i < 0 ) handle_error("Error: read size");

    while( c < size ) {
        i = read(fd, buf+c, size-c);
        if ( i < 0 )
            handle_error("Error: read");
        else if ( i == 0 )
            return c;
        else
            c += i;
    }
    return c;
}


#if ( DEMO_SC == 1 )
void parsec_thread_check_status(void *arg)
{
    parsec_execution_unit_t *eu = (parsec_execution_unit_t*)arg;
    parsec_context_t* parsec_context = eu->virtual_process->parsec_context;
    int th_id = eu->th_id;

    if ( parsec_context->parsec_threads[th_id].status == STREAM_STOPPING ) {
        printf("[P%d:%d] I'll be back\n", parsec_context->my_rank, th_id);
        /*empty task queue?*/
        ABT_mutex_lock(parsec_context->parsec_threads[ eu->th_id ].mutex);
        /*l'ult de calcul rentre dans check_status, et realise qu'il doit s'areter:*/
        /*- premiere instruction, il migre vers le stream de comm, il perd son quantum et attend*/
        ABT_thread me;
        ABT_thread_self(&me);
        ABT_thread_set_migratable(me, ABT_TRUE);
        extern parsec_thread_t dep_thread;
        parsec_context->parsec_threads[th_id].status = STREAM_GOING_TO_BED;

        ABT_thread_migrate_to_pool(me, parsec_context->monitoring_steering_threads[0].pool); /*I lost my quantum*/
        /*l'ult de calcul qui a migre recoit du temps d'execution:*/
        parsec_context->parsec_threads[th_id].status = STREAM_SLEEPING;
        ABT_cond_wait(parsec_context->parsec_threads[ eu->th_id ].cond,
                      parsec_context->parsec_threads[ eu->th_id ].mutex);
        /*I move to my new pool*/
        if (ABT_POOL_NULL != parsec_context->parsec_threads[ eu->th_id ].pool) {
            ABT_thread_migrate_to_pool(me, parsec_context->parsec_threads[ eu->th_id ].pool);
            /*I bind my stream to this core*/
            int core = parsec_bindthread(eu->core_id, -1);
            if ( core == -1 )
                printf("[P%d:%d] this doesn't feel like home (%d:%d).\n",
                       parsec_context->my_rank, th_id, core, eu->core_id);
            else
                printf("[P%d:%d] home sweet home (core %d)!\n", parsec_context->my_rank, th_id, eu->core_id);
        }

        parsec_context->parsec_threads[th_id].status = STREAM_RUNNING;
        ABT_mutex_unlock(parsec_context->parsec_threads[ eu->th_id ].mutex);
        printf("[P%d:%d] I'm back!\n", parsec_context->my_rank, th_id);
    }
}
#endif

static inline int in_list(int *list, int sz, int e)
{
    int i;
    for (i = 0; i < sz; ++i)
        if (list[i] == e)
            return 1;
    return 0;
}


static inline void put_to_sleep(parsec_context_t* context, struct pollfd *abt_pfd, char *recv_buf)
{
    char send_buf[SEND_BUF_LEN];
    char **core_tokens = str_split(recv_buf, ' ');
    int nb_cores = 0;
    while (core_tokens[nb_cores])
        nb_cores++;
    int *cores = (int*)malloc(sizeof(int)*nb_cores);
    int k;
    for (k = 0; k < nb_cores; ++k)
        sscanf(core_tokens[k], "%d", cores+k);

    int i, j;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    for(i=0; i<context->nb_vp; ++i) {
        vp = context->virtual_processes[i];
        for (j = 0; j < vp->nb_cores; ++j) {
            eu = vp->execution_units[j];
            if ( in_list(cores, nb_cores, eu->core_id) ) {
                if ( context->parsec_threads[ eu->th_id ].status == STREAM_RUNNING ) {
                    context->parsec_threads[ eu->th_id ].status = STREAM_STOPPING;
                    printf("[P%d:Listener] th_id=%d on core %d is going to sleep!\n",
                           context->my_rank, eu->th_id, eu->core_id);
                    /*waiting for the guy to go to bed*/
                    while ( context->parsec_threads[ eu->th_id ].status != STREAM_SLEEPING )
                        ABT_thread_yield(); /* give him time to migrate and cond_wait */
                    PARSEC_THREAD_STREAM_FREE( context->parsec_threads[ eu->th_id ] );
                }
                else {
                    printf("[P%d:Listener] th_id=%d on core %d is not running!\n",
                           context->my_rank, eu->th_id, eu->core_id);
                    memset(send_buf, 0, SEND_BUF_LEN);
                    sprintf(send_buf, "nope.");
                    safe_write(abt_pfd->fd, send_buf);
                    return;
                }
            }
        }
    }
    memset(send_buf, 0, SEND_BUF_LEN);
    sprintf(send_buf, "done.");
    safe_write(abt_pfd->fd, send_buf);
}

static inline void wake_core(parsec_thread_t* thread)
{
    /*I create a new stream*/

    PARSEC_THREAD_STREAM_RECREATE(*thread);
    /*I release the ULT*/
    ABT_mutex_lock(thread->mutex);
    ABT_cond_signal(thread->cond);
    ABT_mutex_unlock(thread->mutex);
}

static inline void wake_up(parsec_context_t* context, struct pollfd *abt_pfd, char *recv_buf)
{
    char send_buf[SEND_BUF_LEN];
    char **core_tokens = str_split(recv_buf, ' ');
    int nb_cores = 0;
    while (core_tokens[nb_cores])
        nb_cores++;
    int *cores = (int*)malloc(sizeof(int)*nb_cores);
    int k;
    for (k = 0; k < nb_cores; ++k)
        sscanf(core_tokens[k], "%d", cores+k);

    int i, j;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    for(i=0; i<context->nb_vp; ++i) {
        vp = context->virtual_processes[i];
        for (j = 0; j < vp->nb_cores; ++j) {
            eu = vp->execution_units[j];
            if ( in_list(cores, nb_cores, eu->core_id ) ) {
                if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING ) {
                    printf("[P%d:Listener] th_id %d on core %d is waking up\n",
                           context->my_rank, eu->th_id, eu->core_id);
                    wake_core(context->parsec_threads+eu->th_id);
                    context->parsec_threads[ eu->th_id ].status = STREAM_READY;
                }
                else {
                    printf("[P%d:Listener] th_id %d on core %d is not sleeping!\n",
                           context->my_rank, eu->th_id, eu->core_id);
                    memset(send_buf, 0, SEND_BUF_LEN);
                    sprintf(send_buf, "nope.");
                    safe_write(abt_pfd->fd, send_buf);
                    return;
                }
            }
        }
    }
    memset(send_buf, 0, SEND_BUF_LEN);
    sprintf(send_buf, "done.");
    safe_write(abt_pfd->fd, send_buf);
}

/*'a'=ready, 'b'=running, 'c'=stopping, 'd'=sleeping*/
static inline void send_stream_status(parsec_context_t* context, struct pollfd* abt_pfd )
{
    char send_buf[SEND_BUF_LEN];
    int i, j, c = 0;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    memset(send_buf, 0, SEND_BUF_LEN);
    for(i=0; i<context->nb_vp; ++i) {
        vp = context->virtual_processes[i];
        for (j = 0; j < vp->nb_cores; ++j) {
            eu = vp->execution_units[j];
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_READY )
                send_buf[ eu->core_id ] = 'a';
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_RUNNING )
                send_buf[ eu->core_id ] = 'c';
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_STOPPING )
                send_buf[ eu->core_id ] = 'd';
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_GOING_TO_BED )
                send_buf[ eu->core_id ] = 'e';
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING )
                send_buf[ eu->core_id ] = 'f';
            c++;
        }
    }

    safe_write( abt_pfd->fd, send_buf);
}

static inline int connect_demo(parsec_context_t *context, char *host_str, int port, struct pollfd* abt_pfd)
{
    int sockfd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int ret, rank = context->my_rank;
    char send_buf[SEND_BUF_LEN];

    if ( !(server = gethostbyname(host_str)) ) {
        if ( context->my_rank == 0 )
            printf("[P%d:Listener] Warning: host not found (%s), turning listening thread OFF.\n", context->my_rank, host_str);
        return -1;
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd<0) {
        printf("[P%d:Listener] ", context->my_rank);
        handle_error("Error: socket");
    }
    memset( &serv_addr, 0, sizeof(serv_addr) );
    serv_addr.sin_family = AF_INET;
    memcpy( &serv_addr.sin_addr.s_addr, server->h_addr, server->h_length );
    serv_addr.sin_port = htons(port);

    if (0>(ret = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)))) {
        if ( context->my_rank == 0 )
            printf("[P%d:Listener] Warning: connect failed, turning listening thread OFF.\n", context->my_rank);
        return ret;
    }

    abt_pfd->fd = sockfd;
    abt_pfd->events = POLLIN | POLLRDHUP;

    sprintf(send_buf, "%d", context->my_rank);
    safe_write(abt_pfd->fd, send_buf);
    printf("[P%d:Listener] connected to server (%s:%d) and identified as proc %d!\n",
           rank, host_str, serv_addr.sin_port, context->my_rank);
    return 0;
}

static inline void disconnect_demo(parsec_context_t *context, struct pollfd* abt_pfd)
{
    close(abt_pfd->fd);
    printf("[P%d:Listener] disconnected from server!\n", context->my_rank);
}

static inline void get_command(char ** dst, char *src)
{
    int pos = strcspn(src, " ")+1; /* one space between dest and command, please*/
    *dst = src+pos;
}

static inline void check_events_demo(parsec_context_t* context, struct pollfd* abt_pfd, int* quit)
{
    char recv_buf[RECV_BUF_LEN], *cmd;
    int ret, rank;
    rank = context->my_rank;

    if(*quit) return;

    ret = poll(abt_pfd, 1, 1);
    if ( ret == -1 ) {
        printf("[P%d:Listener] ", context->my_rank);
        handle_error("Error: poll");
    } else if ( ret != 0 ) {
        if ( abt_pfd->revents & POLLIN ) {
            memset(recv_buf, 0, RECV_BUF_LEN);
            safe_read(abt_pfd->fd, recv_buf);
            /* printf("[P%d] received message (%s)\n", rank, recv_buf); */
            get_command(&cmd, recv_buf);
            switch(cmd[0]) {
            case 's': put_to_sleep(context, abt_pfd, cmd+2); break;
            case 'w': wake_up(context, abt_pfd, cmd+2); break;
            case 'n': send_stream_status(context, abt_pfd); break;
            case 'q': disconnect_demo(context, abt_pfd); break;
            case 'd': /*shrink_streams(context, abt_pfd);*/ break;
            case 'i': /*expand_streams(context, abt_pfd);*/ break;
            default:
                printf("[P%d:Listener] Unknown command: (%s)\n", rank, cmd);
            break;
            }
        }

        if ( abt_pfd->revents & POLLRDHUP ) {
            *quit = 1;
            printf("[P%d:Listener] Unexpected disconnection!\n", rank);
        }
        abt_pfd->revents = 0;
    }
}

void wake_up_everybody(parsec_context_t* context)
{
    int i, j;
    parsec_execution_unit_t* eu;
    parsec_vp_t* vp;

    for(i=0; i<context->nb_vp; ++i) {
        vp = context->virtual_processes[i];
        for (j = 0; j < vp->nb_cores; ++j) {
            eu = vp->execution_units[j];
            while ( context->parsec_threads[ eu->th_id ].status != STREAM_SLEEPING &&
                    context->parsec_threads[ eu->th_id ].status != STREAM_RUNNING ) {
                ABT_thread_yield(); /*give him some time to change to a stable state*/
            }
            if ( context->parsec_threads[ eu->th_id ].status == STREAM_SLEEPING ) {
                context->parsec_threads[ eu->th_id ].status = STREAM_READY;
                printf("[P%d:Listener] th_id %d on core %d is waking up\n",
                       context->my_rank, eu->th_id, eu->core_id);
#if ( DEMO_SC == 1 )
                wake_core(context->parsec_threads+eu->th_id);
#endif
            }
        }
    }
}

static inline void check_end(parsec_context_t* context, int* quit)
{
    if ( context->active_objects == 0 ) {
        printf("context->active_objects = %d\n", context->active_objects);
        wake_up_everybody(context);
        *quit = 1;
    }
}

void* __listener_thread(void* arguments)
{
    struct pollfd abt_pfd;
    parsec_context_t *context = (parsec_context_t*)arguments;

    char *host_str = getenv("SERVER_HOSTNAME");
    char *port_str = getenv("SERVER_PORT");
    if ( !host_str || !port_str ) {
        if ( context->my_rank == 0 )
            printf("[P%d:Listening] Warning: env variables: SERVER_HOSTNAME, SERVER_PORT required, turning listening thread OFF.\n", context->my_rank);
        return 0;
    }

    int port = atoi(port_str);
    int quit = 0, ret;

    if ( 0 > (ret = connect_demo(context, host_str, port, &abt_pfd) ) )
        return 0;

    ABT_xstream m;
    ABT_thread me;
    int rank;
    ABT_xstream_self(&m);
    ABT_thread_self(&me);
    ABT_xstream_get_rank(m, &rank);

    while(!quit) {
        check_events_demo(context, &abt_pfd, &quit);
        /*I yield to the communication thread*/
        ABT_thread_yield();
    }

    disconnect_demo(context, &abt_pfd);
    return 0;
}









