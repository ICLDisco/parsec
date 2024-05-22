#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/arena.h"

#include "tree_dist.h"
#include "project.h"
#include "walk_utils.h" /** Must be included before walk.h */
#include "walk.h"

#include <unistd.h>
#include <string.h>

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
    union {
        double   d;
        uint64_t u;
    } aliasing;
    int64_t *cksum = (int64_t*)param;
    int64_t up = 0;
    int64_t nv, ov;
    aliasing.d = node->s;
    up ^= aliasing.u;
    aliasing.d = node->d;
    up ^= aliasing.u;
    up ^= (((int64_t)l)<<32) | (int64_t)n;
    do {
        ov = *cksum;
        nv = ov ^ up;
    } while(!parsec_atomic_cas_int64(cksum, ov, nv));
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

    my_pos = parsec_atomic_fetch_add_int32(&rs->cur_pos, sz);
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
    parsec_context_t* parsec;
    int rc;
    int rank, world;
    tree_dist_t *treeA;
    parsec_matrix_block_cyclic_t fakeDesc;
    parsec_project_taskpool_t *project;
    parsec_walk_taskpool_t *walker;
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype_t);
    int do_checks = 0, be_verbose = 0;
    int pargc = 0, i;
    char **pargv;
    int ret, ch;
    uint64_t cksum = 0;
    redim_string_t *rs;

#if defined(PARSEC_HAVE_MPI)
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

    pargc = 0; pargv = NULL;
    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }
    parsec = parsec_init(1, &pargc, &pargv);

    while ((ch = getopt(argc, argv, "xvd:m:M:f:")) != -1) {
        switch (ch) {
        case 'x':
            do_checks = 1;
            break;
        case 'v':
            be_verbose = 1;
            break;
        case 'd':
            if( rank == atoi(optarg) ) {
                volatile int loop;
                char hostname[256];
                gethostname(hostname, 256);
                fprintf(stderr, "Rank %d is pid %d on host %s -- Waiting for debugger to attach\n",
                        rank, getpid(), hostname);
                loop = 1;
                while(loop != 0) {
                    sleep(1);
                }
            }
            break;
        case '?':
        default:
            fprintf(stderr,
                    "Usage: %s [-x] [-v] [-d rank -d rank -d rank] -- <parsec arguments>\n"
                    "   Implement the Project operation to build a Hartree-Fock function using PaRSEC JDFs\n"
                    "   if -x, create a function, and check that the tree correspond to a pre-computed checksum\n"
                    "   otherwise, output A.dot, a DOT file of the created tree\n"
                    "   if -v, print some information on what task is executed by what rank\n"
                    "   -d rank will make rank d wait for a debugger to attach\n",
                    argv[0]);
            exit(1);
        }
    }

    treeA = tree_dist_create_empty(rank, world);

    parsec_matrix_block_cyclic_init(&fakeDesc, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
                              rank,
                              1, 1, world, world,
                              0, 0, world, world,
                              1, world, 1, 1, 0, 0);

    parsec_matrix_adt_define_rect( adt,
             parsec_datatype_float_t, 2, 1, 2);

#if defined(HAVE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    project = parsec_project_new(treeA, world, (parsec_data_collection_t*)&fakeDesc, 1e-3, be_verbose, 1.0);
    project->arenas_datatypes[PARSEC_project_DEFAULT_ADT_IDX] = *adt;
    rc = parsec_context_add_taskpool(parsec, &project->super);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(&project->super);
    ret = 0;

    if( do_checks ) {
        walker = parsec_walk_new(treeA, world, (parsec_data_collection_t*)&fakeDesc,
                                &cksum, cksum_node_fn, NULL,
                                be_verbose);
    } else {
        rs = rs_new();
        walker = parsec_walk_new(treeA, world, (parsec_data_collection_t*)&fakeDesc,
                                rs, print_node_fn, print_link_fn,
                                be_verbose);
    }
    walker->arenas_datatypes[PARSEC_walk_DEFAULT_ADT_IDX] = *adt;
    rc = parsec_context_add_taskpool(parsec, &walker->super);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

#if defined(HAVE_MPI)
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
#endif  /* defined(HAVE_MPI) */
    parsec_matrix_adt_free( &adt );

    parsec_taskpool_free(&walker->super);

    tree_dist_free(treeA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
