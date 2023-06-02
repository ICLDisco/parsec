#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "stage_custom.h"
parsec_taskpool_t* testing_stage_custom_New( parsec_context_t *ctx, int M, int N, int MB, int NB, int P, int *ret);

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
    int pargc = 0;
    char **pargv = NULL;
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int i;
    int size = 1;
    int rank = 0;
    int M;
    int N;
    int MB;
    int NB;
    int P = 1;
    int ret = 0;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* DISTRIBUTED */

    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }
    /* Initialize PaRSEC */
    parsec = parsec_init(-1, &pargc, &pargv);
    if( NULL == parsec ) {
        /* Failed to correctly initialize. In a correct scenario report*/
         /* upstream, but in this particular case bail out.*/
        exit(-1);
    }

    assert(size == 1);

    /* Test: comparing results when:
        - tile matrix transfered to GPU with default stage_in/stage_out
        - lapack matrix transfered to GPU with custum stage_in/stage_out */

    MB = NB = 1;
    M = N = 1;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
    }

    MB = NB = 1;
    M = N = 10;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
    }

    MB = NB = 4;
    M = N = 20;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
    }

    MB = NB = 40;
    M = N = 240;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
    }

    if(ret!= 0){
        printf("TEST FAILED\n");
    }else{
        printf("TEST PASSED\n");
    }

    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif /* DISTRIBUTED */

    return ret;
}
