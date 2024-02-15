#include "parsec.h"
#include <mpi.h>

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank = 0, world = 1;

#if defined(PARSEC_HAVE_MPI)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    parsec = parsec_init( -1, &argc, &argv );
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    parsec_taskpool_free( dtd_tp );
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
