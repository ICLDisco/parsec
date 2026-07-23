/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
parsec_tests_context_init(int nb_cores, int required_thread,
                          int *pargc, char ***pargv,
                          parsec_context_t **parsec,
                          int *rank, int *world)
{
    int rc;

    if( NULL == parsec ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    /*
     * From this point on, rank and world size come from the PaRSEC context.
     * This keeps the tests independent from the selected communication backend:
     * MPI builds discover them from MPI, UCX builds discover them from PMIx.
     */
    *parsec = parsec_init(nb_cores, pargc, pargv);
    if( NULL == *parsec ) {
        return PARSEC_ERROR;
    }

#if defined(PARSEC_HAVE_MPI)
    {
        int mpi_initialized = 0, provided = PARSEC_TEST_THREAD_SINGLE;

        MPI_Initialized(&mpi_initialized);
        if( mpi_initialized ) {
            MPI_Query_thread(&provided);
            if( provided < required_thread ) {
                fprintf(stderr, "MPI thread support is insufficient: requested %d, provided %d\n",
                        required_thread, provided);
                (void)parsec_tests_context_fini(parsec);
                return PARSEC_ERR_NOT_SUPPORTED;
            }
        }
    }
#else
    (void)required_thread;
#endif

    if( NULL != rank ) {
        rc = parsec_context_query(*parsec, PARSEC_CONTEXT_QUERY_RANK);
        if( rc < 0 ) {
            (void)parsec_tests_context_fini(parsec);
            return rc;
        }
        *rank = rc;
    }
    if( NULL != world ) {
        rc = parsec_context_query(*parsec, PARSEC_CONTEXT_QUERY_NODES);
        if( rc < 0 ) {
            (void)parsec_tests_context_fini(parsec);
            return rc;
        }
        /*
         * A build without a communication engine reports 0 nodes to indicate
         * that no distributed runtime is active.  Tests still expect a usable
         * local world size, so expose that case as a single-process run.
         */
        if( 0 == rc ) {
            rc = 1;
        }
        *world = rc;
    }

    return PARSEC_SUCCESS;
}

int
parsec_tests_context_fini(parsec_context_t **parsec)
{
    int rc = PARSEC_SUCCESS;

    if( (NULL != parsec) && (NULL != *parsec) ) {
        rc = parsec_fini(parsec);
    }

    return rc;
}

int
parsec_tests_barrier(parsec_context_t *parsec)
{
    (void)parsec;

#if defined(PARSEC_HAVE_MPI)
    {
        int mpi_initialized = 0;
        int rc;

        rc = MPI_Initialized(&mpi_initialized);
        if( (MPI_SUCCESS == rc) && mpi_initialized ) {
            rc = MPI_Barrier(MPI_COMM_WORLD);
            return (MPI_SUCCESS == rc) ? PARSEC_SUCCESS : PARSEC_ERROR;
        }
    }
#endif

    return PARSEC_ERR_NOT_IMPLEMENTED;
}

void
parsec_tests_abort(parsec_context_t *parsec, int errorcode)
{
    (void)parsec;

#if defined(PARSEC_HAVE_MPI)
    {
        int mpi_initialized = 0;
        int rc = MPI_Initialized(&mpi_initialized);
        if( (MPI_SUCCESS == rc) && mpi_initialized ) {
            MPI_Abort(MPI_COMM_WORLD, errorcode);
        }
    }
#endif

    exit(errorcode);
}

int
parsec_tests_allreduce(parsec_context_t *parsec,
                       const void *sendbuf,
                       void *recvbuf,
                       int count,
                       parsec_datatype_t datatype,
                       parsec_tests_reduce_op_t op)
{
    if( (NULL == recvbuf) || (count < 0) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (PARSEC_TESTS_REDUCE_SUM != op) &&
        (PARSEC_TESTS_REDUCE_BXOR != op) &&
        (PARSEC_TESTS_REDUCE_MAXLOC_INT != op) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    if( (PARSEC_TESTS_REDUCE_MAXLOC_INT == op) &&
        (parsec_datatype_int_t != datatype) ) {
        return PARSEC_ERR_BAD_PARAM;
    }

#if defined(PARSEC_HAVE_MPI)
    {
        MPI_Op mpi_op;
        MPI_Datatype mpi_datatype = datatype;
        int mpi_initialized = 0;
        int rc;

        switch(op) {
        case PARSEC_TESTS_REDUCE_SUM:
            mpi_op = MPI_SUM;
            break;
        case PARSEC_TESTS_REDUCE_BXOR:
            mpi_op = MPI_BXOR;
            break;
        case PARSEC_TESTS_REDUCE_MAXLOC_INT:
            mpi_op = MPI_MAXLOC;
            mpi_datatype = MPI_2INT;
            break;
        default:
            return PARSEC_ERR_BAD_PARAM;
        }

        rc = MPI_Initialized(&mpi_initialized);
        if( (MPI_SUCCESS == rc) && mpi_initialized ) {
            rc = MPI_Allreduce((NULL == sendbuf || sendbuf == recvbuf) ? MPI_IN_PLACE : (void *)sendbuf,
                               recvbuf, count, mpi_datatype, mpi_op, MPI_COMM_WORLD);
            return (MPI_SUCCESS == rc) ? PARSEC_SUCCESS : PARSEC_ERROR;
        }
    }
#endif

    {
        int nodes = (NULL == parsec) ? 1 : parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_NODES);

        if( nodes < 0 ) {
            return nodes;
        }
        if( nodes > 1 ) {
            return PARSEC_ERR_NOT_IMPLEMENTED;
        }
        if( (NULL != sendbuf) && (sendbuf != recvbuf) && (0 < count) ) {
            if( PARSEC_TESTS_REDUCE_MAXLOC_INT == op ) {
                memcpy(recvbuf, sendbuf, 2 * (size_t)count * sizeof(int));
                return PARSEC_SUCCESS;
            }

            int size, rc;

            rc = parsec_type_size(datatype, &size);
            if( PARSEC_SUCCESS != rc ) {
                return rc;
            }
            memcpy(recvbuf, sendbuf, (size_t)count * (size_t)size);
        }
    }

    (void)op;
    return PARSEC_SUCCESS;
}
