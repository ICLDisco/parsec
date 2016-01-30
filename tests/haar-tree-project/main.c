#include "dague.h"
#include "data_dist/matrix/matrix.h"
#include "dague/arena.h"

#include "tree_dist.h"
#include "project.h"

typedef struct {
    int n;
    int l;
    double s;
} leaf_t;

static void walker_checksum_node(tree_dist_t *tree, int nid, int l, int n, double s, double d, void *param)
{
    double *sum = (double*)param;
    *sum += s + d;
    *sum += l ^ n;
    (void)tree;
    (void)nid;
}

static void walker_checksum_child(tree_dist_t *tree, int nid, int pl, int pn, int cl, int cn, void *param)
{
    double *sum = (double*)param;
    *sum += pl ^ cl;
    *sum += pn ^ cn;
    (void)tree;
    (void)nid;
}

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world;
    tree_dist_t *treeA;
    dague_project_handle_t *project;
    dague_arena_t arena;
    node_t node;
    int do_checks = 0;
    int pargc = 0, i, dashdash = -1;
    char **pargv;
    int ret;

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

    if( argc > 1 && !strcmp(argv[1], "-x" )) {
        do_checks = 1;
    }

    treeA = tree_dist_create_empty(rank, world);
    node.s = 0.0;
    node.d = 0.0;
    tree_dist_insert_node(treeA, &node, 0, 0);

    dague_arena_construct( &arena,
                           2 * dague_datadist_getsizeoftype(matrix_RealFloat),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_float_t
                         );

    project = dague_project_new(treeA, &treeA->super, 1e-3);
    project->arenas[DAGUE_project_DEFAULT_ARENA] = &arena;
    dague_enqueue(dague, &project->super);
    dague_context_wait(dague);

    if( do_checks ) {
        double sum = 0.0;
        walk_tree(&walker_checksum_node, &walker_checksum_child, &sum, treeA);
        ret = (23589.2 == sum);
    } else {
        tree_dist_to_dotfile(treeA, "A.dot");
        ret = 0;
    }

    project->arenas[DAGUE_project_DEFAULT_ARENA] = NULL;
    dague_handle_free(&project->super);

    dague_fini(&dague);

#ifdef DAGUE_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
