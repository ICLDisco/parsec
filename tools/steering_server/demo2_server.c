/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <arpa/inet.h>

#define MAX_PROC        64
#define SEND_BUF_LEN    MAX_PROC+1
#define RECV_BUF_LEN    MAX_PROC+1
#define PORT0           34243

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

static inline void connect_demo(int sockfd, int np, struct pollfd* abt_pfds, int *active)
{
    char recv_buf[RECV_BUF_LEN];
    struct sockaddr_in abt_addr;
    socklen_t addrlen;
    struct pollfd tmp_fd;
    int c = 0;

    listen(sockfd, np);

    while ( c < np ) {
        addrlen = sizeof(abt_addr);
        if ( 0 > ( tmp_fd.fd = accept(sockfd, (struct sockaddr*)&abt_addr, &addrlen) ) )
            handle_error("Error: accept");

        /*read proc rank*/
        int rank;
        memset(recv_buf, 0, RECV_BUF_LEN);
        safe_read(tmp_fd.fd, recv_buf);

        rank = atoi(recv_buf);
        if ( rank < 0 || rank > np ) {
            printf("Error: rank %d unknown\n", rank);
            exit(EXIT_FAILURE);
        }
        printf("Process %d connected!\n", rank);
        abt_pfds[rank].fd = tmp_fd.fd;
        abt_pfds[rank].events = POLLIN | POLLRDHUP;
        active[rank] = 1;
        c++;
    }
    printf("%d processes are now connected!\n", c);
}

static inline void open_socket(int *sockfd, char *port_str)
{
    int ret;
    struct sockaddr_in my_addr;

    if ( 0 > ( *sockfd = socket(AF_INET, SOCK_STREAM, 0) ) )
        handle_error("Error: socket");

    int port = atoi(port_str);
    memset(&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    my_addr.sin_addr.s_addr = INADDR_ANY;
    my_addr.sin_port = htons(port);
    if ( 0 > ( ret = bind(*sockfd, (struct sockaddr*)&my_addr, sizeof(my_addr)) ) )
        handle_error("Error: bind");

    struct addrinfo hints, *info, *p;
    int gai_result;

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; /*either IPV4 or IPV6*/
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_CANONNAME;

    if ((gai_result = getaddrinfo(hostname, "http", &hints, &info)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(gai_result));
        exit(1);
    }

    /* for(p = info; p != NULL; p = p->ai_next) { */
    /*     printf("server running on %s:%d\n", p->ai_canonname, port); */
    /* } */
    if (info)
        printf("server running on %s:%d\n", info->ai_canonname, port);
    freeaddrinfo(info);
}

static inline void print_commands()
{
    printf("<prefix> <one-char-command> <core_id>\n");
    printf("prefix=the mpi rank\n");
    printf("one-char-command list:\n");
    printf(" - s: put a core to sleep\n");
    printf(" - w: wake a core up\n");
    printf(" - n: get cores statuses\n");
    printf(" - q: disconnect from the application\n");
}

static inline void send_command(char *send_buf, struct pollfd *abt_pfds, int *active, int dest)
{
    if ( dest >= 0 && active[dest] )
        safe_write(abt_pfds[dest].fd, send_buf);
    else if ( dest == -1 ) {/*broadcast*/
        int i;
        for ( i = 0; i < MAX_PROC; ++i ) {
            if ( active[i] )
                safe_write(abt_pfds[i].fd, send_buf);
        }
    }
}

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

static inline int check_command(char *send_buf, int *d, char *c)
{
    char **tokens;

    char *tmp = (char*)malloc((strlen(send_buf)+1)*sizeof(char));
    memset(tmp, 0, strlen(send_buf)+1);
    memcpy(tmp, send_buf, strlen(send_buf));
    tokens = str_split(tmp, ' ');

    if ( tokens[0] ) { /*token has at least one command*/
        if ( tokens[0][0] == 'q' ) {
            *d = -1;
            *c = 'q';
            goto happy_ending;
        }
        /*here, the token should be an integer*/
        int ret = sscanf(tokens[0], "%d", d);
        if ( ret == 0 ) /*I read something else*/
            goto sad_ending;
    }
    else
        goto sad_ending;

    if ( tokens[1] ) { /*time to get this command*/
        if ( strlen(tokens[1]) > 1 )
            printf("Ok, you tried a weird command, I'll stick to the first character\n");
        if ( !( tokens[1][0] == 's' || tokens[1][0] == 'w' || tokens[1][0] == 'n' ) )
            goto sad_ending;
        *c = tokens[1][0];
        if ( tokens[1][0] == 'n' )
            goto happy_ending;
    }
    else
        goto sad_ending;

    if ( tokens[2] ) { /*core target*/
        int core;
        int ret = sscanf(tokens[2], "%d", &core);
        if ( ret == 0 ) /*obviously not an integer*/
            goto sad_ending;
        goto happy_ending;
    }

  sad_ending:
    printf("Ok, I don't understand what you want (%s)\n", send_buf);
    free(tokens);
    return 1;

  happy_ending:
    free(tokens);
    return 0;
}

static inline void disconnect_demo(int np, struct pollfd *abt_pfds, int *active)
{
    int i;
    for ( i = 0; i < np; ++i )
        if ( active[i] )
            close(abt_pfds[i].fd);

}

static inline void close_socket(int *sockfd)
{
    close(*sockfd);
}

static inline void print_statuses(char *buf)
{
    int size = strlen(buf);
    int i;
    printf("Cores statuses:\n");
    for ( i = 0; i < size; ++i ) {
        if ( buf[i] == 'a' ) printf("core_%d > READY;\t ", i);
        if ( buf[i] == 'b' ) printf("core_%d > LEAVING BED;\t ", i);
        if ( buf[i] == 'c' ) printf("core_%d > RUNNING;\t ", i);
        if ( buf[i] == 'd' ) printf("core_%d > STOPPING;\t ", i);
        if ( buf[i] == 'e' ) printf("core_%d > GOING_TO_BED;\t ", i);
        if ( buf[i] == 'f' ) printf("core_%d > SLEEPING;\t ", i);
    }
    printf("\n");
}


static inline void wait_response(int *abt_alive, char c, int *active, struct pollfd *abt_pfds, int src)
{
    int ret;
    char recv_buf[RECV_BUF_LEN];

    if ( src != -1 ) {/*P2P*/
        memset(recv_buf, 0, RECV_BUF_LEN);
        while(1) {
            ret = poll(abt_pfds+src, 1, 10);
            if ( -1 == ret ) handle_error("Error: poll");
            else if ( 0 != ret ) {
                if ( abt_pfds[src].revents & POLLIN ) {
                    safe_read(abt_pfds[src].fd, recv_buf);
                    if ( c != 'n' ) /*the app sent back an ACK 'done.'*/
                        printf("Answer: %s\n\n", recv_buf);
                    else
                        print_statuses(recv_buf);
                }
                if ( abt_pfds[src].revents & POLLRDHUP ) {
                    abt_alive = 0;
                    printf("Application disconnected...\n");
                    break;
                }
                abt_pfds[src].revents = 0;
                break;
            }
        }
    }
    else { /*dest = -1 > broadcast*/
        if ( c == 'q' ) {
            int i;
            for ( i = 0; i < MAX_PROC; ++i ) {
                if ( active[i] ) {
                    memset(recv_buf, 0, RECV_BUF_LEN);
                    while(1) {
                        ret = poll(abt_pfds+src, 1, 10);
                        if ( -1 == ret ) handle_error("Error: poll");
                        else if ( 0 != ret ) {
                            if ( abt_pfds[src].revents & POLLIN ) {
                                safe_read(abt_pfds[src].fd, recv_buf);
                                if ( c != 'n' ) /*the app sent back an ACK 'done.'*/
                                    printf("Answer: %s\n\n", recv_buf);
                                else
                                    print_statuses(recv_buf);
                            }
                            if ( abt_pfds[src].revents & POLLRDHUP ) {
                                abt_alive = 0;
                                printf("Application disconnected...\n");
                                break;
                            }
                            abt_pfds[src].revents = 0;
                            break;
                        }
                    }
                }
            }
        }
    }
}

static inline void read_command_from_keyboard(char *buf)
{
    char *pos;
    fgets(buf, SEND_BUF_LEN, stdin);
    if ((pos=strchr(buf, '\n')) != NULL)
        *pos = '\0';
}

int
main(int argc, char *argv[])
{
    char send_buf[SEND_BUF_LEN];
    int quit = 0, np = 1, sockfd, abt_alive = 1, dest, ret;
    char c = '\0';

    if ( argc < 3 ) {
        printf("Usage: %s <number_of_mpi_processes> <port>\n", argv[0]);
        exit(1);
    }
    if ( 64 < (np = atoi(argv[1])) ) {
        printf("Warning: too much processes %d, will only control the first %d to connect\n",
               np, MAX_PROC);
        np = 64;
    }
    int *active = (int*)malloc(MAX_PROC*sizeof(int));
    struct pollfd *abt_pfds = (struct pollfd*)malloc(np*sizeof(struct pollfd));

    open_socket(&sockfd, argv[2]);

    while(!quit) {
        memset(active, 0, MAX_PROC*sizeof(int));
        connect_demo(sockfd, np, abt_pfds, active);

        while(abt_alive) {

            print_commands();

            memset(send_buf, 0, SEND_BUF_LEN);
            read_command_from_keyboard(send_buf);
            ret = check_command(send_buf, &dest, &c);

            if ( 0 == ret && ( dest == -1 || active[dest] ) ) {
                send_command(send_buf, abt_pfds, active, dest);
                wait_response(&abt_alive, c, active, abt_pfds, dest);
                if ( c == 'q' ) {
                    quit = 1;
                    disconnect_demo(np, abt_pfds, active);
                }
            }
            else {
                if ( dest > -1 )
                    printf("Error: cmd=(%s) is not valid, dest=%d, active[%d]=%d\n",
                           send_buf, dest, dest, active[dest]);
                else
                    printf("Error: cmd=(%s) is not valid, dest=ALL\n",
                           send_buf);
            }
        }
    }

    close_socket(&sockfd);

    free(active);
    free(abt_pfds);

    return 0;
}

