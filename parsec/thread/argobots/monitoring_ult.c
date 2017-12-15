/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

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
#include "parsec/mca/pins/alperf/pins_alperf.h"

#define SEND_BUF_LEN    65 /*np 64 + '\0'*/
#define RECV_BUF_LEN    65

static inline void handle_error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

static inline size_t safe_write(int fd, const char* buf, size_t size)
{
    size_t c = 0;
    ssize_t i;

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

    /* i = write(fd, (char*)&size, sizeof(size_t)); */
    /* if ( i < 0 ) handle_error("Error: write size"); */

    while ( c < size ) {
        i = write(fd, buf+c, size-c);
        if ( i < 0 ) handle_error("Error: write");
        c += i;
    }

    flag = 0;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

    return c;
}

#define BUFF_SIZE 4096

static inline size_t check_alperf_contrib(char *message, size_t size, int n) {
    (void) n;
    if ( pins_alperf_counter_store.nb_counters > 0 ) {
        uint64_t t;
        double ts;
        size_t blen;

        t = parsec_profiling_get_time();
#if defined(HAVE_CLOCK_GETTIME)
        ts = (double)t / 1000000000.0;
#elif defined(__IA64) || defined(__X86) || defined(__bgp__)
        /*time scale in GTics*/
        ts = (double)t / 1000000000.0;
#else
        ts = (double)t / 1000000.0;
#endif

        *PINS_ALPERF_DATE = ts;
        assert( pins_alperf_counter_store_size() <= size );
        blen = size < pins_alperf_counter_store_size() ? size : pins_alperf_counter_store_size();
        memcpy(message, pins_alperf_counter_store.counters, blen);

        return blen; /*change the value returned if you pack more than one key*/
    }
    return 0;
}

static inline void check_stats(parsec_context_t *context, parsec_time_t *last_update, struct pollfd *agr_pfd, int *quit)
{
    parsec_time_t now = take_time();
    uint64_t interval = diff_time( *last_update, now );
    uint64_t threshold = 1;
    (void)threshold;
    (void)quit;
    char time_unit;
    time_unit = 's';

#if defined(HAVE_CLOCK_GETTIME)
    threshold *= 1000000000;
#elif defined(__IA64) || defined(__X86) || defined(__bgp__)
    time_unit = 'c';
    threshold *= 1000000000;
#else
    threshold *= 1000000;
#endif

    threshold *= 2;

    if( threshold < interval ) {
        int i, n = 0;
        for(i=0; i<context->nb_vp; ++i)
            n += context->virtual_processes[i]->nb_cores;

        int k = 0;
        char *message = (char*)malloc(BUFF_SIZE);
        memset(message, 0, BUFF_SIZE);

        k += check_alperf_contrib(message+k, BUFF_SIZE-k, n);
        /* printf("Sending %d bytes\n", (int)k); */
        safe_write(agr_pfd->fd, message, k);

        *last_update = now;
    }
}

static inline int connect_monitoring(parsec_context_t *context, char *host_str, int port, struct pollfd* agr_pfd)
{
    int sockfd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int ret, rank = context->my_rank;
    char send_buf[SEND_BUF_LEN];

    if ( !(server = gethostbyname(host_str)) ) {
        if ( context->my_rank == 0 )
            printf("[P%d:Monitoring] Warning: host not found (%s), turning monitoring thread OFF.\n", context->my_rank, host_str);
        return -1;
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd<0) {
        printf("[P%d:Monitoring] ", context->my_rank);
        handle_error("Error: socket" );
    }
    memset( &serv_addr, 0, sizeof(serv_addr) );
    serv_addr.sin_family = AF_INET;
    memcpy( &serv_addr.sin_addr.s_addr, server->h_addr, server->h_length );
    serv_addr.sin_port = htons(port);


    if (0>(ret = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)))) {
        if ( context->my_rank == 0 )
            printf("[P%d:Monitoring] Warning: connect failed, turning monitoring thread OFF.\n", context->my_rank);
        return ret;
    }

    agr_pfd->fd = sockfd;
    agr_pfd->events = POLLIN | POLLRDHUP;
    memset(send_buf, 0, SEND_BUF_LEN);

    int i, M, vp, P = -1, Q = -1;
    for(vp = M = 0; vp < context->nb_vp; ++vp)
        M += context->virtual_processes[vp]->nb_cores;

    sprintf(send_buf, "1;%s;%d;%d;%d;%d;%d;",
            "dpotrf", context->my_rank, context->nb_nodes, M, P, Q);

    for (i = 0; i < pins_alperf_counter_store.nb_counters; ++i)
        sprintf(send_buf+strlen(send_buf), "%s;", PINS_ALPERF_COUNTER(i)->name);

    safe_write(agr_pfd->fd, send_buf, strlen(send_buf));
    printf("[P%d:Monitoring] connected to agregator (%s:%d) and identified as proc %d!\n",
           rank, host_str, serv_addr.sin_port, context->my_rank);
    return 0;
}


static inline void disconnect_monitoring(parsec_context_t *context, struct pollfd* agr_pfd)
{
    close(agr_pfd->fd);
    printf("[P%d:Monitoring] disconnected from agregator!\n", context->my_rank);
}


void* __monitoring_thread(void* arguments)
{
    (void)arguments;
#ifdef DEMO_SC
    struct pollfd agr_pfd;
    parsec_context_t *context = (parsec_context_t*)arguments;

    char *host_str = getenv("AGGREGATOR_HOSTNAME");
    char *port_str = getenv("AGGREGATOR_PORT");
    if ( !host_str || !port_str ) {
        if ( context->my_rank == 0 )
            printf("[P%d:Monitoring] Warning: env variables AGGREGATOR_HOSTNAME, AGGREGATOR_PORT required, turning monitoring thread OFF.\n", context->my_rank);
        return 0;
    }

    int port = atoi(port_str);
    int quit = 0, ret;

    if ( 0 > (ret = connect_monitoring(context, host_str, port, &agr_pfd) ) )
        return 0;

    parsec_time_t last_update = take_time();
    while(!quit) {
        check_stats(context, &last_update, &agr_pfd, &quit);
        ABT_thread_yield();
    }

    disconnect_monitoring(context, &agr_pfd);
#endif
    return 0;
}
