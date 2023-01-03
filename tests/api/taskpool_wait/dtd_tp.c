#include "parsec/parsec_config.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "parsec/parsec_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

int task( parsec_execution_stream_t    *es, parsec_task_t *this_task ) {
    (void)es;
    int delta, m, n, *A;

    parsec_dtd_unpack_args(this_task, &A, &m, &n, &delta);

    printf( "Rank: %d, core %d: entering DTD task (%d, %d) of TP %s (0x%p), sleeping for %g seconds\n",
            this_task->taskpool->context->my_rank, es->core_id,
            m, n, this_task->taskpool->taskpool_name, this_task->taskpool, delta / 1.0e6 );
    usleep(delta);
    printf( "Rank: %d, core %d: leaving DTD task (%d, %d) of TP %s (0x%p) (slept for %g seconds)\n",
            this_task->taskpool->context->my_rank, es->core_id,
            m, n, this_task->taskpool->taskpool_name, this_task->taskpool, delta / 1.0e6 );

    return PARSEC_HOOK_RETURN_DONE;
}

void new_dtd_taskpool(parsec_taskpool_t *dtd_tp, int TILE_FULL, parsec_matrix_block_cyclic_t *A, int deltamin, int deltamax)
{
    for(int m = 0; m < A->super.mt; m++) {
        for(int n = 0; n < A->super.nt; n++) {
            double r = (double)rand() / (double)RAND_MAX;
            int delta = deltamin + (deltamax-deltamin)*r;
            parsec_data_key_t key = A->super.super.data_key(&A->super.super, m, n);
            parsec_dtd_insert_task(dtd_tp, task,    0, PARSEC_DEV_CPU, "task (DTD)",
                                    PASSED_BY_REF,    PARSEC_DTD_TILE_OF_KEY(&A->super.super, key), PARSEC_INPUT | TILE_FULL | PARSEC_AFFINITY,
                                    sizeof(int),      &m, PARSEC_VALUE,
                                    sizeof(int),      &n, PARSEC_VALUE,
                                    sizeof(int),      &delta, PARSEC_VALUE,
                                    PARSEC_DTD_ARG_END);
        }
    }
    parsec_dtd_data_flush_all( dtd_tp, &A->super.super );
}
