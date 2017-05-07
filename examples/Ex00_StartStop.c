/**
 * This first example shows how to initialize, and terminate PaRSEC, in an mpi,
 * or non mpi environment.
 *    parsec_init()
 *    parsec_context_wait()
 *    parsec_fini()
 *
 * [mpicc|cc] -o Ex00_StartStop Ex00_StartStop.c `pkg-config --cflags --libs parsec` -lpthread -lm -lrt [-ldl -lcudart]
 *
 */
#include <parsec.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world;
    int rc;

    /**
     * Let's initialize MPI *before* parsec
     */
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

    /**
     * Then, we can initialize the engine by giving the number of threads on the
     * current process, and the arguments.
     * If -1, is given for the number of threads, the number of physical core on
     * the architecture will be detected and used as the default for the number
     * of threads.
     */
    parsec = parsec_init(-1, &argc, &argv);

    /**
     * Let's do some computation with the runtime, and wait for the end.
     * Here, no computation is performed.
     */
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    /**
     * We now finalize the engine, and then MPI if needed
     */
    rc = parsec_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_fini");
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return 0;
}
