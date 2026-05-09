/*
 * Copyright (c) 2021-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif
#include "parsec.h"

int main(int argc, char *argv[])
{
#if defined(PARSEC_HAVE_MPI)
    int mpith = MPI_THREAD_SINGLE;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &mpith);
    assert(mpith >= MPI_THREAD_SERIALIZED); // parsec will do the complaining in NDEBUG
#endif
    parsec_context_t *parsec = parsec_init(-1, &argc, &argv);
    parsec_fini(&parsec);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
}
