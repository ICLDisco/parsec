/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 *
 * Benchmark for the zone_malloc allocator with rbtree free-list management.
 *
 * The rbtree organises free chunks by size so that find_or_larger() runs in
 * O(log k) time, where k is the number of distinct free-chunk sizes present.
 * Each scenario deliberately varies k:
 *
 *   sequential        k=1 always              — baseline, minimal tree work
 *   stride-2          k=1 (after coalescing)  — exercises the coalesce path
 *   power-of-2        k=log2(N) ≈ 12          — realistic GPU buffer pool
 *   many distinct     k≈sqrt(2N) ≈ 88         — stress find() / insert / remove
 *   find_or_larger    k≈sqrt(N) ≈ 63          — every alloc must walk to a
 *                                                larger node in the tree
 *   random mixed      k varies, steady state  — workload simulation
 *
 * Output columns:
 *   total time   — wall time for all iterations of the scenario
 *   throughput   — (alloc+free) operations per second
 *   k            — number of distinct free-chunk sizes in the rbtree
 *                  during the timed section (-1 = varies)
 *
 * Usage: zonemalloc_benchmark [-n num_segments] [-N iterations] [-- parsec-opts]
 */

#include "parsec.h"
#include "parsec/utils/zone_malloc.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define UNIT_SIZE  512u   /* bytes per unit (typical GPU page bucket) */

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static zone_malloc_t *make_zone(char **base_out, int num_segments)
{
    char *b = (char *)malloc((size_t)num_segments * UNIT_SIZE);
    assert(b);
    *base_out = b;
    zone_malloc_t *z = zone_malloc_init(b, num_segments, UNIT_SIZE);
    assert(z);
    return z;
}

static void destroy_zone(zone_malloc_t *z, char *base)
{
    zone_malloc_fini(&z);
    free(base);
}

/* ------------------------------------------------------------------ */
/* Scenario 1 – sequential same-size                                   */
/* k = 1 throughout; baseline overhead of the rbtree.                  */
/* ------------------------------------------------------------------ */

static double bench_sequential(int num_segments, int iterations)
{
    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **ptrs = (void **)malloc((size_t)num_segments * sizeof(void *));

    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < num_segments; i++) {
            ptrs[i] = zone_malloc(z, UNIT_SIZE);
			if (!ptrs[i]) {
				fprintf(stderr, "Failed to allocate sequential segment %d\n", i);
				return 0.0;
			}
        }
        for (int i = 0; i < num_segments; i++)
            zone_free(z, ptrs[i]);
    }
    double elapsed = wall_time() - t0;

    free(ptrs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 2 – stride-2 fragmentation then coalescing                 */
/* Alloc all, free even slots (k=1 isolated holes), then free odd      */
/* slots (each free merges left+right, k collapses back to 1).         */
/* ------------------------------------------------------------------ */

static double bench_stride2(int num_segments, int iterations)
{
    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **ptrs = (void **)malloc((size_t)num_segments * sizeof(void *));

    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < num_segments; i++)
            ptrs[i] = zone_malloc(z, UNIT_SIZE);
        /* free even: creates N/2 isolated 1-unit holes → k=1 */
        for (int i = 0; i < num_segments; i += 2)
            zone_free(z, ptrs[i]);
        /* free odd: each triggers two-way coalesce → k=1 */
        for (int i = 1; i < num_segments; i += 2)
            zone_free(z, ptrs[i]);
    }
    double elapsed = wall_time() - t0;

    free(ptrs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 3 – power-of-2 sizes                                       */
/* Sizes 1, 2, 4, 8, ... units → k = floor(log2(num_segments)).       */
/* Typical GPU scratch-buffer pool allocation pattern.                  */
/* ------------------------------------------------------------------ */

static int pow2_L(int num_segments)
{
    int L = 0;
    unsigned s = 1, total = 0;
    while (total + s <= (unsigned)num_segments) { total += s; s <<= 1; L++; }
    return L;
}

static double bench_pow2(int num_segments, int iterations, long *actual_ops_out)
{
    int L = pow2_L(num_segments);
    unsigned *sizes = (unsigned *)malloc((size_t)L * sizeof(unsigned));
    { unsigned s = 1; for (int i = 0; i < L; i++) { sizes[i] = s; s <<= 1; } }

    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **ptrs = (void **)malloc((size_t)L * sizeof(void *));

    long total_ops = 0;
    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        int n_alloc = 0;
        for (int i = 0; i < L; i++) {
            ptrs[i] = zone_malloc(z, (size_t)sizes[i] * UNIT_SIZE);
            if (ptrs[i]) n_alloc++;
        }
        /* largest first: coalescing unwinds back to one big free chunk */
        for (int i = L - 1; i >= 0; i--)
            if (ptrs[i]) zone_free(z, ptrs[i]);
        total_ops += 2L * n_alloc;
    }
    double elapsed = wall_time() - t0;
    *actual_ops_out = total_ops;

    free(sizes); free(ptrs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 4 – many distinct free-chunk sizes (exact matches)         */
/*                                                                      */
/* Setup: interleave 1-unit spacers with i-unit data blocks (i=1..K).  */
/* Free all data blocks → rbtree holds K distinct sizes simultaneously, */
/* none can coalesce (spacers keep them apart).                         */
/* Each alloc in the timed loop calls find() for an exact size → k=K.  */
/*                                                                      */
/* Slot layout: [spacer:1][chunk:i] for i=1..K                         */
/* Units needed: sum_{i=1}^{K} (1+i) = K + K(K+1)/2 = K(K+3)/2       */
/* ------------------------------------------------------------------ */

static int distinct_K(int num_segments)
{
    int K = 0, total = 0;
    while (total + 1 + (K + 1) <= num_segments)
        { K++; total += 1 + K; }
    return K;
}

static double bench_many_distinct(int num_segments, int iterations, long *actual_ops_out)
{
    int K = distinct_K(num_segments);

    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **spacers = (void **)malloc((size_t)K * sizeof(void *));
    void **chunks  = (void **)malloc((size_t)K * sizeof(void *));
    void **allocs  = (void **)malloc((size_t)K * sizeof(void *));

    /* build fragmented heap: spacer prevents coalescing between chunks */
    for (int i = 0; i < K; i++) {
        spacers[i] = zone_malloc(z, UNIT_SIZE);
        assert(spacers[i]);
        chunks[i]  = zone_malloc(z, (size_t)(i + 1) * UNIT_SIZE);
        assert(chunks[i]);
    }
    for (int i = 0; i < K; i++)
        zone_free(z, chunks[i]);
    /* rbtree now has K distinct nodes: sizes 1, 2, 3, ..., K */

    long total_ops = 0;
    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        /* each alloc calls find(size) — exact match exists at depth O(log K) */
        int n_alloc = 0;
        for (int i = 0; i < K; i++) {
            allocs[i] = zone_malloc(z, (size_t)(i + 1) * UNIT_SIZE);
            if (allocs[i]) n_alloc++;
        }
        /* reverse free: restores the K distinct nodes for next iteration */
        for (int i = K - 1; i >= 0; i--)
            if (allocs[i]) zone_free(z, allocs[i]);
        total_ops += 2L * n_alloc;
    }
    double elapsed = wall_time() - t0;
    *actual_ops_out = total_ops;

    for (int i = 0; i < K; i++)
        zone_free(z, spacers[i]);
    free(spacers); free(chunks); free(allocs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 5 – find_or_larger with deliberate gaps                    */
/*                                                                      */
/* Free chunks exist only for even sizes: 2, 4, 6, ..., 2K.           */
/* Every allocation requests an odd size (1, 3, 5, ..., 2K-1) so      */
/* find_or_larger() always has to advance to the next larger node.     */
/* This exercises the "or larger" branch on every single allocation.   */
/*                                                                      */
/* Slot layout: [spacer:1][chunk:2i] for i=1..K                        */
/* Units needed: sum_{i=1}^{K} (1+2i) = K + K(K+1) = K(K+2)          */
/*                                                                      */
/* Allocation of 2i-1 units from a 2i-unit chunk leaves a 1-unit      */
/* leftover; freeing in reverse order coalesces it back, restoring     */
/* the original even-sized chunks for the next iteration.              */
/* ------------------------------------------------------------------ */

static int gaps_K(int num_segments)
{
    int K = 0;
    while ((K + 1) * (K + 3) <= num_segments)
        K++;
    return K;
}

static double bench_find_or_larger(int num_segments, int iterations, long *actual_ops_out)
{
    int K = gaps_K(num_segments);

    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **spacers = (void **)malloc((size_t)K * sizeof(void *));
    void **chunks  = (void **)malloc((size_t)K * sizeof(void *));
    void **allocs  = (void **)malloc((size_t)K * sizeof(void *));

    /* build heap: even-sized free chunks, isolated by spacers */
    for (int i = 0; i < K; i++) {
        spacers[i] = zone_malloc(z, UNIT_SIZE);
        assert(spacers[i]);
        chunks[i]  = zone_malloc(z, (size_t)(2 * (i + 1)) * UNIT_SIZE);
        assert(chunks[i]);
    }
    for (int i = 0; i < K; i++)
        zone_free(z, chunks[i]);
    /* rbtree has K nodes: sizes 2, 4, 6, ..., 2K */

    long total_ops = 0;
    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        /* request odd sizes → find_or_larger must skip to the next even node */
        int n_alloc = 0;
        for (int i = 0; i < K; i++) {
            allocs[i] = zone_malloc(z, (size_t)(2 * i + 1) * UNIT_SIZE);
            if (allocs[i]) n_alloc++;
        }
        /* reverse free: leftover 1-unit fragments coalesce with the freed
         * block to restore the original even-sized free chunks */
        for (int i = K - 1; i >= 0; i--)
            if (allocs[i]) zone_free(z, allocs[i]);
        total_ops += 2L * n_alloc;
    }
    double elapsed = wall_time() - t0;
    *actual_ops_out = total_ops;

    for (int i = 0; i < K; i++)
        zone_free(z, spacers[i]);
    free(spacers); free(chunks); free(allocs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 6 – random mixed alloc/free (steady-state churn)           */
/* Maintains ~pool_size live allocations; randomly replaces entries.   */
/* Simulates a real workload; k fluctuates in a realistic range.       */
/* ------------------------------------------------------------------ */

static double bench_random_mixed(int num_segments, int iterations)
{
    int pool_size = num_segments / 8;
    if (pool_size < 1) pool_size = 1;

    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **pool = (void **)calloc((size_t)pool_size, sizeof(void *));

    /* deterministic LCG — no stdlib rand() state interference */
    uint64_t rng = 0xdeadbeef12345678ULL;
#define LCG_STEP() (rng = rng * 6364136223846793005ULL + 1442695040888963407ULL)
#define LCG_BITS(n) ((int)((LCG_STEP() >> 33) & (((uint64_t)1 << (n)) - 1)))

    for (int i = 0; i < pool_size; i++) {
        int sz = LCG_BITS(2) + 1;
        pool[i] = zone_malloc(z, (size_t)sz * UNIT_SIZE);
        if (!pool[i]) pool[i] = zone_malloc(z, UNIT_SIZE);
    }

    double t0 = wall_time();
    long n_ops = (long)pool_size * iterations;
    for (long op = 0; op < n_ops; op++) {
        int slot = (int)((LCG_STEP() >> 1) % (uint64_t)pool_size);
        if (pool[slot]) { zone_free(z, pool[slot]); pool[slot] = NULL; }
        int sz = LCG_BITS(2) + 1;
        pool[slot] = zone_malloc(z, (size_t)sz * UNIT_SIZE);
    }
    double elapsed = wall_time() - t0;

    for (int i = 0; i < pool_size; i++)
        if (pool[i]) zone_free(z, pool[i]);

#undef LCG_STEP
#undef LCG_BITS

    free(pool);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 7 – full pool then OOM attempt                             */
/* Allocates every last segment, verifies that a further allocation    */
/* returns NULL (rbtree is empty; find_or_larger returns NULL in O(1)),*/
/* then frees all segments. k = 0 while the pool is full.             */
/* ------------------------------------------------------------------ */

static double bench_full_oom(int num_segments, int iterations)
{
    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **ptrs = (void **)malloc((size_t)num_segments * sizeof(void *));
        for (int i = 0; i < num_segments; i++) {
            ptrs[i] = zone_malloc(z, UNIT_SIZE);
            assert(ptrs[i]);
        }

    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        /* No free chunks remain in the rbtree; must return NULL. */
        assert(zone_malloc(z, UNIT_SIZE) == NULL);
    }
    double elapsed = wall_time() - t0;
        for (int i = 0; i < num_segments; i++)
            zone_free(z, ptrs[i]);

    free(ptrs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Scenario 8 – fragmentation impact on memory availability            */
/*                                                                      */
/* Creates a stride-2 fragmented heap: fills with 1-unit blocks, then */
/* frees every other one.  50 % of memory is nominally free but spread */
/* across isolated 1-unit holes — only size-1 requests can succeed.   */
/*                                                                      */
/* Timed section: repeatedly probe all power-of-2 request sizes from   */
/* 1 up to num_segments/2 units.  For each size the benchmark greedily */
/* allocates until OOM, then frees to restore the fragmented state.    */
/* For sizes > 1 this is a fast-fail path: find_or_larger finds the    */
/* single rbtree node (key=1) is smaller than the request and returns  */
/* NULL without scanning the N/2 individual free chunks.               */
/*                                                                      */
/* After timing, a per-size table is printed showing how much of the   */
/* nominally-free memory can actually be satisfied.                    */
/* ------------------------------------------------------------------ */

static double bench_fragmentation(int num_segments, int iterations)
{
    char *base;
    zone_malloc_t *z = make_zone(&base, num_segments);
    void **held   = (void **)malloc((size_t)num_segments * sizeof(void *));
    void **allocs = (void **)malloc((size_t)num_segments * sizeof(void *));

    /* ---- build fragmented state once (outside timed section) ---- */
    for (int i = 0; i < num_segments; i++) {
        held[i] = zone_malloc(z, UNIT_SIZE);
        assert(held[i]);
    }
    for (int i = 0; i < num_segments; i += 2) {
        zone_free(z, held[i]);
        held[i] = NULL;
    }
    /* rbtree: single node, key=1, list holds num_segments/2 free chunks */

    /* ---- timed probe loop ---- */
    double t0 = wall_time();
    for (int it = 0; it < iterations; it++) {
        for (int s = 1; s <= num_segments / 2; s <<= 1) {
            int n = 0;
            void *p;
            while ((p = zone_malloc(z, (size_t)s * UNIT_SIZE)) != NULL)
                allocs[n++] = p;
            for (int j = 0; j < n; j++)
                zone_free(z, allocs[j]);
        }
    }
    double elapsed = wall_time() - t0;

    /* ---- fragmentation report (one pass, outside timing) ---- */
    size_t pool_bytes = (size_t)num_segments * UNIT_SIZE;
    size_t free_bytes = pool_bytes - zone_in_use(z);
    printf("\n  fragmentation detail (stride-2, %.0f%% of pool nominally free):\n",
           100.0 * (double)free_bytes / (double)pool_bytes);
    printf("  %-14s  %10s  %14s  %8s\n",
           "request size", "succeeded", "bytes usable", "usable%");
    printf("  %-14s  %10s  %14s  %8s\n",
           "------------", "---------", "------------", "-------");
    for (int s = 1; s <= num_segments / 2; s <<= 1) {
        int n = 0;
        void *p;
        while ((p = zone_malloc(z, (size_t)s * UNIT_SIZE)) != NULL)
            allocs[n++] = p;
        size_t usable = (size_t)n * (size_t)s * UNIT_SIZE;
        printf("  %3d unit(s)%4s  %10d  %14zu  %7.0f%%\n",
               s, "", n, usable,
               free_bytes > 0 ? 100.0 * (double)usable / (double)free_bytes : 0.0);
        for (int j = 0; j < n; j++)
            zone_free(z, allocs[j]);
    }

    /* ---- teardown ---- */
    for (int i = 1; i < num_segments; i += 2)
        zone_free(z, held[i]);
    free(held); free(allocs);
    destroy_zone(z, base);
    return elapsed;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                            */
/* ------------------------------------------------------------------ */

static void print_row(const char *name, double elapsed, long ops, int k)
{
    const char *k_str = (k >= 0) ? "" : "varies";
    if (elapsed < 1e-9 || ops == 0) {
        /* Timer resolution too coarse or no successful operations. */
        if (k >= 0)
            printf("  %-34s  %8.3f s  %8s ns/op  %11s ops/s   k=%-4d\n",
                   name, elapsed, "<res", "<res", k);
        else
            printf("  %-34s  %8.3f s  %8s ns/op  %11s ops/s   k=%s\n",
                   name, elapsed, "<res", "<res", k_str);
        return;
    }
    double ns_per_op = elapsed * 1e9 / (double)ops;
    if (k >= 0)
        printf("  %-34s  %8.3f s  %8.1f ns/op  %11.0f ops/s   k=%-4d\n",
               name, elapsed, ns_per_op, (double)ops / elapsed, k);
    else
        printf("  %-34s  %8.3f s  %8.1f ns/op  %11.0f ops/s   k=varies\n",
               name, elapsed, ns_per_op, (double)ops / elapsed);
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-n num_segments] [-N iterations] [-- parsec-opts]\n"
            "  -n : total units in the managed pool (default: 4096)\n"
            "  -N : outer repetitions per scenario  (default: 200)\n"
            "  -h : print this message\n",
            prog);
}

int main(int argc, char **argv)
{
    int num_segments = 4096;
    int iterations   = 200;
    int ch;
    char *endptr;

    while ((ch = getopt(argc, argv, "n:N:h")) != -1) {
        switch (ch) {
        case 'n':
            num_segments = (int)strtol(optarg, &endptr, 10);
            if (endptr == optarg || *endptr != '\0' || num_segments <= 0) {
                fprintf(stderr, "invalid -n value '%s'\n", optarg);
                return 1;
            }
            break;
        case 'N':
            iterations = (int)strtol(optarg, &endptr, 10);
            if (endptr == optarg || *endptr != '\0' || iterations <= 0) {
                fprintf(stderr, "invalid -N value '%s'\n", optarg);
                return 1;
            }
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
#endif

    /* Pass only the portion after '--' to parsec_init so it does not
     * choke on our own options. */
    int pargc = 0;
    char **pargv = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }

    parsec_context_t *parsec = parsec_init(1, &pargc, &pargv);
    if (!parsec) {
        fprintf(stderr, "parsec_init failed\n");
        return 1;
    }

    printf("zone_malloc rbtree benchmark\n");
    printf("  pool: %d segments x %u bytes = %u KiB\n",
           num_segments, UNIT_SIZE, (unsigned)num_segments * UNIT_SIZE / 1024);
    printf("  %d iterations per scenario\n\n", iterations);
    printf("  %-34s  %10s  %14s  %13s   %s\n",
           "scenario", "total time", "ns/op", "throughput", "k (rbtree nodes)");
    printf("  %s\n",
           "------------------------------------------------------------------------------------"
           "----------");

    double t;

    t = bench_sequential(num_segments, iterations);
    print_row("sequential same-size", t,
              2L * iterations * (long)num_segments, 1);

    t = bench_stride2(num_segments, iterations);
    print_row("stride-2 coalescing", t,
              3L * iterations * (long)num_segments, 1);

    {
        int L = pow2_L(num_segments);
        long actual_ops;
        t = bench_pow2(num_segments, iterations, &actual_ops);
        print_row("power-of-2 sizes", t, actual_ops, L);
    }

    {
        int K = distinct_K(num_segments);
        long actual_ops;
        t = bench_many_distinct(num_segments, iterations, &actual_ops);
        print_row("many distinct sizes (exact)", t, actual_ops, K);
    }

    {
        int K = gaps_K(num_segments);
        long actual_ops;
        t = bench_find_or_larger(num_segments, iterations, &actual_ops);
        print_row("find_or_larger with gaps", t, actual_ops, K);
    }

    {
        int pool_size = num_segments / 8;
        if (pool_size < 1) pool_size = 1;
        t = bench_random_mixed(num_segments, iterations);
        print_row("random mixed alloc/free", t,
                  2L * iterations * (long)pool_size, -1);
    }

    t = bench_full_oom(num_segments, iterations);
    /* ops: num_segments allocs + 1 failed alloc + num_segments frees per iter */
    print_row("full pool + OOM attempt", t,
              (long)iterations * (2L * num_segments + 1), 0);

    {
        /* count the probe sizes: 1, 2, 4, ..., num_segments/2 */
        int n_probes = 0;
        for (int s = 1; s <= num_segments / 2; s <<= 1) n_probes++;
        /* bench_fragmentation prints the detail table before returning */
        t = bench_fragmentation(num_segments, iterations);
        /* ops per iter: num_segments alloc+free (size-1 succeeds) + (n_probes-1) failed finds */
        print_row("fragmentation probes", t,
                  (long)iterations * ((long)num_segments + n_probes - 1), 1);
    }

    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif
    return 0;
}
