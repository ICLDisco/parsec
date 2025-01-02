#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct node {
    char         *tname;
    char         *accesses;
    int           nbsucc;
    struct node **succ;
} node_t;

typedef struct {
    node_t  **node;
    int       size;
    int       allocated;
} nl_t;

static node_t ta_start = {
    .tname = "S#A",
    .accesses = "M0x1"};
static node_t ta_end = {
    .tname = "E#A",
    .accesses = "M0x1"};
static node_t tb_start = {
    .tname = "S#B",
    .accesses = "R0x1,W0x2"};
static node_t tb_end = {
    .tname = "E#B",
    .accesses = "R0x1,W0x2"};
static node_t tc_start = {
    .tname = "S#C",
    .accesses = "R0x1,W0x3"};
static node_t tc_end = {
    .tname = "E#C",
    .accesses = "R0x1,W0x3"};
static node_t td_start = {
    .tname = "S#D",
    .accesses = "R0x3,W0x1"};
static node_t td_end = {
    .tname = "E#D",
    .accesses = "R0x3,W0x1"};
static node_t te_start = {
    .tname = "S#E",
    .accesses = "R0x1,W0x3"};
static node_t te_end = {
    .tname = "E#E",
    .accesses = "R0x1,W0x3"};
static node_t tf_start = {
    .tname = "S#F",
    .accesses = "M0x3"};
static node_t tf_end = {
    .tname = "E#F",
    .accesses = "M0x3"};

#define NBNODES 12
#define MAXSUCC  2

node_t **load_dummy_graph(void)
{
    node_t **r = (node_t**)malloc(NBNODES*sizeof(node_t*));
    r[0] = &ta_start;
    r[1] = &ta_end;
    r[2] = &tb_start;
    r[3] = &tb_end;
    r[4] = &tc_start;
    r[5] = &tc_end;
    r[6] = &td_start;
    r[7] = &td_end;
    r[8] = &te_start;
    r[9] = &te_end;
    r[10]= &tf_start;
    r[11]= &tf_end;

    ta_start.nbsucc = 1;
    ta_start.succ = (node_t**)malloc(1*sizeof(node_t));
    ta_start.succ[0] = &ta_end;

    tb_start.nbsucc = 1;
    tb_start.succ = (node_t**)malloc(1*sizeof(node_t));
    tb_start.succ[0] = &tb_end;

    tc_start.nbsucc = 1;
    tc_start.succ = (node_t**)malloc(1*sizeof(node_t));
    tc_start.succ[0] = &tc_end;

    td_start.nbsucc = 1;
    td_start.succ = (node_t**)malloc(1*sizeof(node_t));
    td_start.succ[0] = &td_end;

    te_start.nbsucc = 1;
    te_start.succ = (node_t**)malloc(1*sizeof(node_t));
    te_start.succ[0] = &te_end;

    tf_start.nbsucc = 1;
    tf_start.succ = (node_t**)malloc(1*sizeof(node_t));
    tf_start.succ[0] = &tf_end;

    ta_end.nbsucc = 2;
    ta_end.succ = (node_t**)malloc(2*sizeof(node_t));
    ta_end.succ[0] = &tb_start;
    ta_end.succ[1] = &tc_start;

    tb_end.nbsucc = 2;
    tb_end.succ = (node_t**)malloc(1*sizeof(node_t));
    tb_end.succ[0] = &te_start;
    tb_end.succ[1] = &tf_start;

    tc_end.nbsucc = 1;
    tc_end.succ = (node_t**)malloc(1*sizeof(node_t));
    tc_end.succ[0] = &td_start;

    td_end.nbsucc = 1;
    td_end.succ = (node_t**)malloc(1*sizeof(node_t));
    td_end.succ[0] = &te_start;

    tf_end.nbsucc = 1;
    tf_end.succ = (node_t**)malloc(1*sizeof(node_t));
    tf_end.succ[0] = &td_start;

    return r;
}

typedef struct {
    off_t         tname;
    off_t         accesses;
    int           nbsucc;
    int           succ[MAXSUCC];
} filenode_t;

typedef struct {
    int           nbnodes;
    off_t         nodes[NBNODES];
} filenode_header_t;

int node_index_of(node_t *a, node_t **r, int n)
{
    int i;
    for(i = 0; i < n; i++)
        if( r[i] == a )
            return i;
    return -1;
}

int main(int argc, char *argv[])
{
    int i, j;
    char zero = 0;
/*     filenode_header_t h; */
/*     filenode_t t; */
    node_t **nodes;
    off_t next_string;
    int fd = open("dummy.grp", O_CREAT|O_WRONLY|O_TRUNC, 00644);
    int nbnodes = NBNODES;

    if( fd == -1 ) {
        perror("unable to create dummy.grp:");
        exit(1);
    }

    nodes = load_dummy_graph();

    write(fd, &nbnodes, sizeof(int));

    for(i = 0; i < NBNODES; i++) {
        write( fd, nodes[i]->tname,    strlen(nodes[i]->tname)    + 1 );
        write( fd, nodes[i]->accesses, strlen(nodes[i]->accesses) + 1 );
        write( fd, &(nodes[i]->nbsucc), sizeof(int) );

        for(j = 0; j < nodes[i]->nbsucc; j++) {
            assert( j < MAXSUCC );
            int succ = node_index_of(nodes[i]->succ[j], nodes, NBNODES);
            write( fd, &succ, sizeof(int) );
            assert( succ != -1 );
        }
    }

    next_string = lseek( fd, 0, SEEK_CUR );
    for(i = 0; (i + next_string) % getpagesize() != 0; i++)
        write(fd, &zero, sizeof(char));

    close(fd);
    return 0;
}
