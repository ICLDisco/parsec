#include <stdio.h>
#include <mpi.h>
#include <parsec.h>

int main(int argc, char **argv) {
    parsec_context_t* parsec_context;
    int parsec_argc = argc;
    char** parsec_argv = argv;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    parsec_context = parsec_init(-1, &parsec_argc, &parsec_argv);
    if(NULL == parsec_context) {
        fprintf(stderr, "parsec_init failed\n");
        MPI_Finalize();
        return -1;
    }

    printf("PaRSEC OK\n");

    parsec_fini(&parsec_context);
    MPI_Finalize();
    return 0;
}
