/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/runtime.h"
#include "tests/tests_runtime.h"
#include "rtt_wrapper.h"
#include "rtt_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "parsec/utils/debug.h"

static int next_message_size(int current, int upper)
{
    int base = 1, next;

    if( current >= upper ) {
        return upper;
    }
    if( current < 4 ) {
        next = current + 1;
    } else {
        while( base <= current / 2 ) {
            base <<= 1;
        }
        next = (current == base) ? current + base / 2 : base << 1;
    }
    if( next <= current ) {
        next = current + 1;
    }
    return (next > upper) ? upper : next;
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, ch, i;
    int size, nb, rc, idx;
    int pargc = 0;
    char **pargv = NULL;
    int loops = 4;
    int start_length = 256, end_length = 256;
    struct timeval tstart, tend;
    double t, bw;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *rtt;

    while ((ch = getopt(argc, argv, "n:l:u:h")) != -1) {
        switch (ch) {
            case 'n': loops = atoi(optarg); break;
            case 'l': start_length = atoi(optarg); break;
            case 'u': end_length = atoi(optarg); break;
            case '?': case 'h': default:
                fprintf(stderr,
                        "-n : round trips across all ranks (default: 4)\n"
                        "-l : lower message size in bytes (default: 256)\n"
                        "-u : upper message size in bytes (default: lower size)\n"
                        "\n");
                 exit(1);
        }
    }
    if( loops < 1 ) {
        loops = 1;
    }
    if( start_length < 1 ) {
        start_length = 1;
    }
    if( end_length < start_length ) {
        end_length = start_length;
    }

    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &pargc, &pargv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    nb = loops * world;
    for(idx = 0, size = start_length; ; idx++) {
        /* Match the collection datatype to the message currently under test.
         * Reusing an upper-bound collection works but makes the distributed
         * receive path reshape every smaller short message back to bytes.
         */
        dcA = create_and_distribute_data(rank, world, size);
        parsec_data_collection_set_key(dcA, "A");

        rtt = rtt_new(dcA, size, nb);
        rc = parsec_context_add_taskpool(parsec, rtt);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        rc = parsec_tests_barrier(parsec);
        if( (PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_IMPLEMENTED != rc) ) {
            PARSEC_CHECK_ERROR(rc, "parsec_tests_barrier");
        }
        gettimeofday(&tstart, NULL);

        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        rc = parsec_tests_barrier(parsec);
        if( (PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_IMPLEMENTED != rc) ) {
            PARSEC_CHECK_ERROR(rc, "parsec_tests_barrier");
        }
        gettimeofday(&tend, NULL);

        if( 0 == rank ) {
            t = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_usec - tstart.tv_usec);
            bw = (0.0 == t) ? 0.0 : (double)nb * (double)size * 8.0 * 1000000.0 / (t * 1024.0 * 1024.0);
            printf("%3d: %8d bytes %d hops --> %10.2f Mib/s, %10.2f usec/hop, %10.2f usec/round-trip\n",
                   idx, size, nb, bw, t / (double)nb, t / (double)loops);
        }

        parsec_taskpool_free((parsec_taskpool_t*)rtt);
        free_data(dcA);
        if( size == end_length ) {
            break;
        }
        size = next_message_size(size, end_length);
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
