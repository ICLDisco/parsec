/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * Several PTG taskpools working on the same tiles of one block-cyclic matrix,
 * all given to one context before a single parsec_context_wait():
 *
 *     M  am_map:   RW A <- descA(m, n) -> descA(m, n)          (CPU and CUDA)
 *     C  am_chain: K in-place updates per tile, read from and written back
 *                  to descA(m, n) by the first and last              (CPU)
 *     B  am_bcast: descA(m, n) read once, forwarded read only to J users,
 *                  some of them on other ranks                       (CPU)
 *
 * The sequence of taskpools (-s, "MCBM" by default) is meant to run in that
 * order on every tile. Without automatic merging (-o 0) nothing orders them,
 * and the bodies that see a tile out of order are counted as violations. With
 * it (-o 1, the default) there must be none, and every tile must end up with
 * the value the sequence gives it.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/mca/device/device.h"
#include "tests/tests_runtime.h"

#include "auto_merge_ops.h"
#include "am_map.h"
#include "am_chain.h"
#include "am_bcast.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double am_wtime(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static void usage(const char *name)
{
    fprintf(stderr,
            "Usage: %s [options] [-- parsec options]\n"
            "  -M mt    tiles per column (default 8)\n"
            "  -N nt    tiles per row (default 8)\n"
            "  -b nb    tile size, nb x nb integers (default 16)\n"
            "  -P p     rows of the process grid (default: number of ranks)\n"
            "  -s seq   taskpools to run, in order: M (map), C (chain), B (broadcast) (default MCBM)\n"
            "  -k K     tasks per tile of a chain (default 3)\n"
            "  -j J     users per tile of a broadcast (default 3)\n"
            "  -d us    delay of the slow bodies, in microseconds (default 2000)\n"
            "  -r R     run the sequence R times (default 1)\n"
            "  -o 0|1   automatic merging off or on (default 1)\n"
            "  -g 0|1   let the map run on GPUs (default 1)\n"
            "  -a       add the taskpools before starting the context (default: after)\n"
            "  -w       wait for each taskpool before adding the next (the barrier baseline)\n"
            "  -v       report the counts of automatic merging\n",
            name);
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_matrix_block_cyclic_t dcA;
    parsec_arena_datatype_t adt;
    parsec_datatype_t dt;
    int rank = 0, world = 1, rc, ch;
    int mt = 8, nt = 8, nb = 16, P = -1, K = 3, J = 3, delay = 2000, repeat = 1;
    int merge = 1, use_gpu = 1, add_first = 0, barrier = 0, verbose = 0;
    const char *seq = "MCBM";
    int32_t nb_viol = 0;

    while( -1 != (ch = getopt(argc, argv, "M:N:b:P:s:k:j:d:r:o:g:awvh")) ) {
        switch(ch) {
        case 'M': mt = atoi(optarg); break;
        case 'N': nt = atoi(optarg); break;
        case 'b': nb = atoi(optarg); break;
        case 'P': P = atoi(optarg); break;
        case 's': seq = optarg; break;
        case 'k': K = atoi(optarg); break;
        case 'j': J = atoi(optarg); break;
        case 'd': delay = atoi(optarg); break;
        case 'r': repeat = atoi(optarg); break;
        case 'o': merge = atoi(optarg); break;
        case 'g': use_gpu = atoi(optarg); break;
        case 'a': add_first = 1; break;
        case 'w': barrier = 1; break;
        case 'v': verbose = 1; break;
        case 'h': default: usage(argv[0]); exit(1);
        }
    }
    for( int i = 1; i <= argc - optind; i++ ) argv[i] = argv[optind + i - 1];
    argc = argc - optind + 1;
    if( (mt < 1) || (nt < 1) || (nb < 1) || (K < 1) || (J < 1) || (repeat < 1) ) {
        usage(argv[0]);
        exit(1);
    }
    for( const char *c = seq; '\0' != *c; c++ ) {
        if( (*c != 'M') && (*c != 'C') && (*c != 'B') ) {
            fprintf(stderr, "Unknown taskpool '%c' in -s %s\n", *c, seq);
            exit(1);
        }
    }

    if( verbose ) setenv("PARSEC_MCA_runtime_auto_merge_verbose", "1", 1);
    /* A CPU-only run has no business reserving device memory: several ranks
     * of it on one node would share, and exhaust, the same GPU */
    if( !use_gpu || !am_have_cuda_kernels ) {
        setenv("PARSEC_MCA_device_cuda_enabled", "0", 0);
        setenv("PARSEC_MCA_device_hip_enabled", "0", 0);
    }
    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    rc = parsec_context_set_auto_merge(parsec, merge ? PARSEC_AUTO_MERGE_ON : PARSEC_AUTO_MERGE_OFF);
    PARSEC_CHECK_ERROR(rc, "parsec_context_set_auto_merge");

    int nb_gpus = 0;
    for( unsigned int i = 0; i < parsec_nb_devices; i++ ) {
        parsec_device_module_t *dev = parsec_mca_device_get(i);
        if( PARSEC_DEV_IS_GPU(dev->type) ) nb_gpus++;
    }
    /* Without kernels, a CUDA body must never be picked */
    if( !am_have_cuda_kernels ) use_gpu = 0;

    /* One column of ranks: the users of a broadcast live on other ranks */
    if( P < 1 ) P = world;
    if( world % P ) {
        if( 0 == rank ) fprintf(stderr, "-P %d does not divide %d ranks\n", P, world);
        parsec_tests_context_fini(&parsec);
        exit(1);
    }
    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                                    rank, nb, nb, mt * nb, nt * nb, 0, 0, mt * nb, nt * nb,
                                    P, world / P, 1, 1, 0, 0);
    size_t bytes = (size_t)dcA.super.nb_local_tiles * (size_t)dcA.super.bsiz *
                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype);
    dcA.mat = parsec_data_allocate(bytes > 0 ? bytes : 1);
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "A");
    for( int m = 0; m < mt; m++ ) {
        for( int n = 0; n < nt; n++ ) {
            if( dcA.super.super.rank_of(&dcA.super.super, m, n) != (uint32_t)rank ) continue;
            parsec_data_t *data = dcA.super.super.data_of(&dcA.super.super, m, n);
            int *tile = (int*)parsec_data_copy_get_ptr(parsec_data_get_copy(data, 0));
            for( int i = 0; i < nb * nb; i++ ) tile[i] = (int)am_init_value(m, n, i);
        }
    }
    parsec_translate_matrix_type(PARSEC_MATRIX_INTEGER, &dt);
    PARSEC_OBJ_CONSTRUCT(&adt, parsec_arena_datatype_t);
    parsec_matrix_adt_define_rect(&adt, dt, nb, nb, nb);

    /* Build every taskpool, numbering the operations they apply in order */
    int nb_tp = (int)strlen(seq) * repeat;
    parsec_taskpool_t **tps = (parsec_taskpool_t**)calloc(nb_tp, sizeof(parsec_taskpool_t*));
    int nb_ops = 0;
    for( int t = 0; t < nb_tp; t++ ) {
        switch( seq[t % strlen(seq)] ) {
        case 'M': {
            parsec_am_map_taskpool_t *tp = parsec_am_map_new((parsec_tiled_matrix_t*)&dcA, nb_ops, delay, &nb_viol);
            tp->arenas_datatypes[PARSEC_am_map_DEFAULT_ADT_IDX] = adt;
            tps[t] = &tp->super;
            nb_ops += 1;
            break;
        }
        case 'C': {
            parsec_am_chain_taskpool_t *tp = parsec_am_chain_new((parsec_tiled_matrix_t*)&dcA, nb_ops, K, &nb_viol);
            tp->arenas_datatypes[PARSEC_am_chain_DEFAULT_ADT_IDX] = adt;
            tps[t] = &tp->super;
            nb_ops += K;
            break;
        }
        case 'B': {
            parsec_am_bcast_taskpool_t *tp = parsec_am_bcast_new((parsec_tiled_matrix_t*)&dcA, nb_ops, J, delay, &nb_viol);
            tp->arenas_datatypes[PARSEC_am_bcast_DEFAULT_ADT_IDX] = adt;
            tps[t] = &tp->super;
            break;
        }
        }
        PARSEC_OBJ_RETAIN(adt.arena);
        if( !use_gpu ) parsec_mca_device_taskpool_restrict(tps[t], PARSEC_DEV_CPU);
    }

    parsec_tests_barrier(parsec);
    double start = am_wtime();
    if( barrier ) {
        /* One context_wait per taskpool: ordered, but nothing overlaps */
        for( int t = 0; t < nb_tp; t++ ) {
            rc = parsec_context_add_taskpool(parsec, tps[t]);
            PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
            rc = parsec_context_start(parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_context_start");
            rc = parsec_context_wait(parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        }
    } else {
        if( !add_first ) {
            rc = parsec_context_start(parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        }
        for( int t = 0; t < nb_tp; t++ ) {
            rc = parsec_context_add_taskpool(parsec, tps[t]);
            PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        }
        if( add_first ) {
            rc = parsec_context_start(parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        }
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    }
    double elapsed = am_wtime() - start;

    /* Every tile must have gone through every operation, in order */
    int32_t nb_wrong = 0;
    for( int m = 0; m < mt; m++ ) {
        for( int n = 0; n < nt; n++ ) {
            if( dcA.super.super.rank_of(&dcA.super.super, m, n) != (uint32_t)rank ) continue;
            parsec_data_t *data = dcA.super.super.data_of(&dcA.super.super, m, n);
            const int *tile = (const int*)parsec_data_copy_get_ptr(parsec_data_get_copy(data, 0));
            if( am_check_tile(tile, nb * nb, m, n, nb_ops) ) {
                if( nb_wrong < 4 ) {
                    fprintf(stderr, "[%d] tile (%d, %d) holds %u, expected %u after %d operations\n",
                            rank, m, n, (uint32_t)tile[0], am_expected(m, n, 0, nb_ops), nb_ops);
                }
                nb_wrong++;
            }
        }
    }

    int32_t counts[2] = { nb_viol, nb_wrong };
    rc = parsec_tests_allreduce(parsec, NULL, counts, 2, parsec_datatype_int32_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");
    if( 0 == rank ) {
        printf("auto_merge %s%s: sequence %s x%d (%d taskpools, %d operations) on %dx%d tiles of %dx%d, "
               "%d rank(s), %d GPU(s) %s: %d violations, %d wrong tiles, %.3f s\n",
               merge ? "on" : "off", barrier ? " (one wait per taskpool)" : "",
               seq, repeat, nb_tp, nb_ops, mt, nt, nb, nb, world,
               nb_gpus, use_gpu ? "used" : "unused", counts[0], counts[1], elapsed);
    }

    for( int t = 0; t < nb_tp; t++ ) parsec_taskpool_free(tps[t]);
    free(tps);
    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);
    for( int t = 0; t < nb_tp; t++ ) PARSEC_OBJ_RELEASE(adt.arena);
    parsec_matrix_arena_datatype_destruct_free_type(&adt);
    parsec_tests_context_fini(&parsec);

    /* Without merging, the violations are the point of the run, not a failure */
    if( merge && (counts[0] || counts[1]) ) return 1;
    return 0;
}
