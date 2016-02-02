#include <dague.h>
#include <data_dist/matrix/matrix.h>
#include <data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <dague/arena.h>

#include "tree_dist.h"
#include "project.h"
#include "walk_utils.h" /** Must be included before walk.h */
#include "walk.h"

#include <unistd.h>
extern char *optarg;
extern int optind;
extern int optopt;
extern int opterr;
extern int optreset;

#define SUM_VALUE 0xbdae8a4ea45fc32eLL

typedef struct {
    int n;
    int l;
    double s;
} leaf_t;

static void cksum_node_fn(tree_dist_t *tree, node_t *node, int n, int l, void *param)
{
    uint64_t *cksum = (uint64_t*)param;
    uint64_t up = 0;
    uint64_t nv, ov;
    up ^= *(uint64_t*)&node->s;
    up ^= *(uint64_t*)&node->d;
    up ^= (((uint64_t)l)<<32) | (uint64_t)n;
    do {
        ov = *cksum;
        nv = ov ^ up;
    } while(!dague_atomic_cas(cksum, ov, nv));
    (void)tree;
}

typedef struct {
    int32_t size;
    char *value;
    int32_t cur_pos;
    pthread_rwlock_t resize_lock;
} redim_string_t;

static void rs_add(redim_string_t *rs, const char *format, ...)
{
    int sz;
    int32_t my_pos;
    va_list ap;

    /** Compute size needed */
    va_start(ap, format);
    sz = vsnprintf(NULL, 0, format, ap) + 1;
    va_end(ap);

    my_pos = dague_atomic_add_32b(&rs->cur_pos, sz) - sz;
    assert(my_pos>=0);
    for(;;) {
        if( my_pos + sz < rs->size ) {
            pthread_rwlock_rdlock(&rs->resize_lock);
            va_start(ap, format);
            vsnprintf(rs->value + my_pos, sz, format, ap);
            va_end(ap);
            pthread_rwlock_unlock(&rs->resize_lock);
            return;
        }

        pthread_rwlock_wrlock(&rs->resize_lock);
        if( my_pos + sz > rs->size ) {
            for(rs->size = 2*rs->size;
                rs->size < my_pos + sz;
                rs->size *= 2) /* nothing */;
            rs->value = (char*)realloc(rs->value, rs->size);
        }
        pthread_rwlock_unlock(&rs->resize_lock);
    }
}

redim_string_t *rs_new(void)
{
    redim_string_t *rs = (redim_string_t*)malloc(sizeof(redim_string_t));
    rs->size = 1;
    rs->value = malloc(rs->size);
    rs->value[0] = '\0';
    rs->cur_pos = 0;
    pthread_rwlock_init(&rs->resize_lock, NULL);
    return rs;
}

void rs_free(redim_string_t *rs)
{
    pthread_rwlock_destroy(&rs->resize_lock);
    free(rs->value);
    free(rs);
}

char *rs_string(redim_string_t *rs)
{
    int i;
    for(i = 0; i < rs->cur_pos-1; i++) {
        if( rs->value[i] == '\0' ) rs->value[i] = '\n';
    }
    return rs->value;
}

static void print_node_fn(tree_dist_t *tree, node_t *node, int n, int l, void *param)
{
    redim_string_t *rs = (redim_string_t*)param;
    if( NULL != node )
        rs_add(rs, "N_%d_%d [label=\"[%d,%d](%g, %g)\"];", n, l, n, l, node->s, node->d);
    else
        rs_add(rs, "N_%d_%d [label=\"[%d,%d](-, -)\"];", n, l, n, l);
    (void)param;
    (void)tree;
}

static void print_link_fn(tree_dist_t *tree, node_t *node, int n, int l, void *param)
{
    redim_string_t *rs = (redim_string_t*)param;
    if(n > 0)
        rs_add(rs, "N_%d_%d -> N_%d_%d;", n, l, n-1, l/2);
    (void)param;
    (void)tree;
    (void)node;
}

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world;
    tree_dist_t *treeA;
    two_dim_block_cyclic_t fakeDesc;
    dague_project_handle_t *project;
    dague_walk_handle_t *walker;
    dague_arena_t arena;
    int do_checks = 0, be_verbose = 0;
    int pargc = 0, i, dashdash = -1;
    char **pargv;
    int ret, ch;
    uint64_t cksum = 0;
    redim_string_t *rs;

#if defined(DAGUE_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            dashdash = i;
            pargc = 0;
        } else if( dashdash != -1 ) {
            pargc++;
        }
    }
    pargv = malloc( (pargc+1) * sizeof(char*));
    if( dashdash != -1 ) {
        for(i = dashdash+1; i < argc; i++) {
            pargv[i-dashdash-1] = strdup(argv[i]);
        }
        pargv[i-dashdash-1] = NULL;
    } else {
        pargv[0] = NULL;
    }
    dague = dague_init(-1, &pargc, &pargv);


    while ((ch = getopt(argc, argv, "xvd:")) != -1) {
        switch (ch) {
        case 'x':
            do_checks = 1;
            break;
        case 'v':
            be_verbose = 1;
            break;
        case 'd':
            if( rank == atoi(optarg) ) {
                char hostname[256];
                gethostname(hostname, 256);
                fprintf(stderr, "Rank %d is pid %d on host %s -- Waiting for debugger to attach\n",
                        rank, getpid(), hostname);
                ret = 1;
                while(ret != 0) {
                    sleep(1);
                }
            }
            break;
        case '?':
        default:
            fprintf(stderr,
                    "Usage: %s [-x] [-v] [-d rank -d rank -d rank] -- <parsec arguments>\n"
                    "   Implement the Project operation to build a Haar tree using PaRSEC JDFs\n"
                    "   if -x, create a haar-tree, and check that the tree correspond to a pre-computed checksum\n"
                    "   otherwise, output A.dot, a DOT file of the created tree\n"
                    "   if -v, print some information on what task is executed by what rank\n"
                    "   -d rank will make rank d wait for a debugger to attach\n",
                    argv[0]);
            exit(1);
        }
    }

    treeA = tree_dist_create_empty(rank, world);

    two_dim_block_cyclic_init(&fakeDesc, matrix_RealFloat, matrix_Tile,
                              world, rank, 1, 1, world, world, 0, 0, world, world, 1, 1, 1);

    dague_arena_construct( &arena,
                           2 * dague_datadist_getsizeoftype(matrix_RealFloat),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_float_t
                         );
    project = dague_project_new(treeA, &treeA->super, world, (dague_ddesc_t*)&fakeDesc, 1e-3, 0, be_verbose);
    project->arenas[DAGUE_project_DEFAULT_ARENA] = &arena;
    dague_enqueue(dague, &project->super);
    dague_context_wait(dague);

    if( do_checks ) {
        walker = dague_walk_new(treeA, &treeA->super, world, (dague_ddesc_t*)&fakeDesc,
                                &cksum, cksum_node_fn, NULL,
                                0, be_verbose);
    } else {
        rs = rs_new();
        walker = dague_walk_new(treeA, &treeA->super, world, (dague_ddesc_t*)&fakeDesc,
                                rs, print_node_fn, print_link_fn,
                                0, be_verbose);
    }
    walker->arenas[DAGUE_walk_DEFAULT_ARENA] = &arena;
    dague_enqueue(dague, &walker->super);
    dague_context_wait(dague);

    ret = 0;
    if( do_checks ) {
        uint64_t sum = 0;
        printf("Rank %d contributes with %llx\n", rank, cksum);
        MPI_Reduce(&cksum, &sum, 1, MPI_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);
        if( rank == 0 ) {
            if( sum != SUM_VALUE ) {
                printf("****  Sum = %llx instead of %llx\n", sum, SUM_VALUE);
                ret = 1;
            }
        }
    } else {
        if( rank == 0 ) {
            int r;
            printf("digraph G {\n");
            printf("%s", rs_string(rs));
            rs_free(rs);
            for(r = 1; r < world; r++) {
                char *buff;
                int size;
                MPI_Recv(&size, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buff = (char *)malloc(size);
                MPI_Recv(buff, size, MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("%s", buff);
                free(buff);
            }
            printf("}\n");
        } else {
            char *buff = rs_string(rs);
            int size = strlen(buff)+1;
            MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(buff, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            rs_free(rs);
        }
    }

    project->arenas[DAGUE_project_DEFAULT_ARENA] = NULL;
    dague_handle_free(&project->super);
    walker->arenas[DAGUE_walk_DEFAULT_ARENA] = NULL;
    dague_handle_free(&walker->super);

    dague_fini(&dague);

#ifdef DAGUE_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
