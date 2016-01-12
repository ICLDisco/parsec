#include <dague.h>
#include <data_dist/matrix/matrix.h>
#include <dague/arena.h>

#include "tree_dist.h"
#include "project.h"

typedef struct {
    int n;
    int l;
    double s;
} leaf_t;

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world;
    tree_dist_t *treeA;
    dague_project_handle_t *project;
    dague_arena_t arena;
    node_t node;

#if defined(HAVE_MPI)
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
    dague = dague_init(-1, &argc, &argv);

    treeA = tree_dist_create_empty(rank, world);
    node.s = 0.0;
    node.d = 0.0;
    tree_dist_insert_node(treeA, &node, 0, 0);


    dague_arena_construct( &arena,
                           2 * dague_datadist_getsizeoftype(matrix_RealFloat),
                           DAGUE_ARENA_ALIGNMENT_SSE,
#ifdef DISTRIBUTED
                           MPI_FLOAT
#else
                           NULL
#endif  /* DISTRIBUTED */
                         );

    project = dague_project_new(treeA, &treeA->super, 1e-3);
    project->arenas[DAGUE_project_DEFAULT_ARENA] = &arena;
    dague_enqueue(dague, &project->super);
    dague_context_wait(dague);

    tree_dist_to_dotfile(treeA, "A.dot");

    project->arenas[DAGUE_project_DEFAULT_ARENA] = NULL;
    dague_handle_free(&project->super);

    dague_fini(&dague);

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
