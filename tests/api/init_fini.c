#include <mpi.h>
#include "parsec.h"

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);
    parsec_context_t *parsec = parsec_init(-1, &argc, &argv);
    parsec_fini(&parsec);
    MPI_Finalize();
}
