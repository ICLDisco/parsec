/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#if !defined(_TESTS_RUNTIME_H_)
#define _TESTS_RUNTIME_H_

#include "parsec.h"
#include "parsec/datatype.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#define PARSEC_TEST_THREAD_SINGLE     MPI_THREAD_SINGLE
#define PARSEC_TEST_THREAD_FUNNELED   MPI_THREAD_FUNNELED
#define PARSEC_TEST_THREAD_SERIALIZED MPI_THREAD_SERIALIZED
#define PARSEC_TEST_THREAD_MULTIPLE   MPI_THREAD_MULTIPLE
#else
#define PARSEC_TEST_THREAD_SINGLE     0
#define PARSEC_TEST_THREAD_FUNNELED   1
#define PARSEC_TEST_THREAD_SERIALIZED 2
#define PARSEC_TEST_THREAD_MULTIPLE   3
#endif

typedef enum parsec_tests_reduce_op_e {
    PARSEC_TESTS_REDUCE_SUM,
    PARSEC_TESTS_REDUCE_BXOR,
    PARSEC_TESTS_REDUCE_MAXLOC_INT
} parsec_tests_reduce_op_t;

/**
 * Initialize the process launcher/runtime pair used by PaRSEC tests.
 *
 * Tests should call this helper instead of directly initializing MPI or PMIx.
 * parsec_init() initializes the selected communication backend as needed, and
 * this helper retrieves rank/size from the PaRSEC context afterwards.
 *
 * @param[in] nb_cores Number of cores to pass to parsec_init().
 * @param[in] required_thread Minimum MPI thread level, using
 *            PARSEC_TEST_THREAD_*.
 * @param[inout] pargc PaRSEC argc, passed to parsec_init().
 * @param[inout] pargv PaRSEC argv, passed to parsec_init().
 * @param[out] parsec Initialized PaRSEC context.
 * @param[out] rank Current process rank in the selected communication backend.
 * @param[out] world Number of processes in the selected communication backend.
 */
int parsec_tests_context_init(int nb_cores, int required_thread,
                              int *pargc, char ***pargv,
                              parsec_context_t **parsec,
                              int *rank, int *world);

/**
 * Finalize the PaRSEC context and any process launcher initialized by
 * parsec_tests_context_init().
 */
int parsec_tests_context_fini(parsec_context_t **parsec);

/**
 * Synchronize all processes participating in the selected test runtime.
 *
 * This is intentionally a test helper, not a public runtime API.  It accepts
 * the PaRSEC context so future communication backends can implement the same
 * operation without exposing their transport details to tests.  For now, only
 * MPI-backed runs have a useful implementation; non-MPI backends return
 * PARSEC_ERR_NOT_IMPLEMENTED.
 */
int parsec_tests_barrier(parsec_context_t *parsec);

/**
 * Abort all processes participating in the selected test runtime.
 *
 * MPI-backed tests call MPI_Abort on MPI_COMM_WORLD.  Other backends terminate
 * the local process until they grow a distributed abort primitive.
 */
void parsec_tests_abort(parsec_context_t *parsec, int errorcode);

/**
 * Reduce values across all processes participating in the selected test runtime.
 *
 * A NULL send buffer means in-place reduction into recvbuf.  MPI-backed tests
 * call MPI_Allreduce.  Single-process non-MPI runs copy sendbuf into recvbuf
 * and return success; multi-process non-MPI backends return
 * PARSEC_ERR_NOT_IMPLEMENTED until their collective support is added.  The
 * PARSEC_TESTS_REDUCE_MAXLOC_INT operation expects count int pairs laid out as
 * {value, rank} and uses MPI_2INT/MPI_MAXLOC when MPI backs the test runtime.
 */
int parsec_tests_allreduce(parsec_context_t *parsec,
                           const void *sendbuf,
                           void *recvbuf,
                           int count,
                           parsec_datatype_t datatype,
                           parsec_tests_reduce_op_t op);

#endif /* _TESTS_RUNTIME_H_ */
