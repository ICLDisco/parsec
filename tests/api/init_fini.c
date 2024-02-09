#include <mpi.h>
#include "parsec.h"

int main(int argc, char *argv[])
{
    int mpith = MPI_THREAD_SINGLE;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &mpith);
    assert(mpith >= MPI_THREAD_SERIALIZED); // parsec will do the complaining in NDEBUG
    parsec_context_t *parsec = parsec_init(-1, &argc, &argv);
    parsec_fini(&parsec);
    MPI_Finalize();
}
