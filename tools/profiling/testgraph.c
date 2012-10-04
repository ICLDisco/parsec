#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    char *g;
    unsigned int i, n, e, l;
    FILE *f;

    graphInit();

    n = 0;
    for(i = 1; i < (unsigned int)argc; i++) 
        n += add_nodes_from_dotfile( argv[i], i-1, NULL, 0 );
    e = 0;
    for(i = 1; i < (unsigned int)argc; i++) 
        e += add_edges_from_dotfile( argv[i] );
    fprintf(stderr, "%d nodes, %d edges\n", n, e);

    srand( getpid() );
    for(i = 0; i < 5; i++) {
        set_node_status( (unsigned int) ((double)rand() * (double)n/(double)RAND_MAX), STATUS_READY );
    }

    graphRenderStatusAtDistance(STATUS_READY, 3, &g, &l);
    f = fopen("test.png", "w");
    fwrite(g, 1, l, f);
    fclose(f);
    free(g);

    return graphFini();
}
