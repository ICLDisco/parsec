/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * Cost of DTD argument passing, old interface versus the generated one.
 *
 * Empty tasks carrying 1 to 10 PARSEC_VALUE int arguments and no data flows,
 * inserted and run on a single core. No flows means no dependency tracking,
 * so what is left is the cost of describing, packing and unpacking the
 * argument list.
 *
 * Four variants per arity:
 *
 *   direct     parsec_dtd_insert_task() with (size, pointer, op) triplets,
 *              body unpacks with parsec_dtd_unpack_args()
 *   taskclass  parsec_dtd_insert_task_with_task_class() with (flags, pointer)
 *              pairs, body unpacks with parsec_dtd_unpack_args()
 *   generated  PARSEC_DTD_DECLARE_TASK_CLASS(), insertion through the typed
 *              NAME##_insert(), body copies out with PARSEC_DTD_UNPACK_LOCALS()
 *   inplace    the same class and the same insertion as generated, with a body
 *              that reads the values where they are instead of copying them
 *
 * generated and inplace differ only in the body, so the gap between their
 * execution times is the cost of the copy and nothing else.
 *
 * The argument lists of all four variants are emitted from one X-macro per
 * arity. The tokens the compiler sees are exactly what a person would have
 * typed; the preprocessor is used only to avoid writing forty near-identical
 * functions by hand.
 *
 * Insertion and execution are timed separately: the context is not started
 * until the insertion loop is over, and the DTD window is raised beyond the
 * task count so the inserting thread never stops to execute.
 */

#include "parsec/runtime.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/interfaces/dtd/dtd_task_class_decl.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* Argument lists, built up one entry at a time. */
#define BENCH_PARAMS_1(_)  _(VALUE, int, a0, 0)
#define BENCH_PARAMS_2(_)  BENCH_PARAMS_1(_) _(VALUE, int, a1, 0)
#define BENCH_PARAMS_3(_)  BENCH_PARAMS_2(_) _(VALUE, int, a2, 0)
#define BENCH_PARAMS_4(_)  BENCH_PARAMS_3(_) _(VALUE, int, a3, 0)
#define BENCH_PARAMS_5(_)  BENCH_PARAMS_4(_) _(VALUE, int, a4, 0)
#define BENCH_PARAMS_6(_)  BENCH_PARAMS_5(_) _(VALUE, int, a5, 0)
#define BENCH_PARAMS_7(_)  BENCH_PARAMS_6(_) _(VALUE, int, a6, 0)
#define BENCH_PARAMS_8(_)  BENCH_PARAMS_7(_) _(VALUE, int, a7, 0)
#define BENCH_PARAMS_9(_)  BENCH_PARAMS_8(_) _(VALUE, int, a8, 0)
#define BENCH_PARAMS_10(_) BENCH_PARAMS_9(_) _(VALUE, int, a9, 0)

/* Emitters for the hand-written styles. */
#define B_CLASS_ARG(KIND, TYPE, NAME, FLAGS)     ((int)sizeof(TYPE)), PARSEC_VALUE,
#define B_TC_INSERT_ARG(KIND, TYPE, NAME, FLAGS) PARSEC_DTD_EMPTY_FLAG, &NAME,
#define B_DIRECT_INSERT_ARG(KIND, TYPE, NAME, FLAGS) ((int)sizeof(TYPE)), &NAME, PARSEC_VALUE,
#define B_DECL(KIND, TYPE, NAME, FLAGS)          TYPE NAME;
#define B_DECL_INIT(KIND, TYPE, NAME, FLAGS)     TYPE NAME = (TYPE)(seed++);
#define B_UNPACK_ARG(KIND, TYPE, NAME, FLAGS)    &NAME,
#define B_SUM(KIND, TYPE, NAME, FLAGS)           + NAME
#define B_SUM_INPLACE(KIND, TYPE, NAME, FLAGS)   + v->NAME
#define B_DESIG(KIND, TYPE, NAME, FLAGS)         .NAME = NAME,

static int64_t bench_sink = 0;

/*
 * Every body adds up the arguments it was given, and every run hands out the
 * consecutive integers 1..ntasks*nb_args, so the total is known in advance.
 * A variant that reads its arguments from the wrong place does not merely run
 * at a different speed, it fails here.
 */
static void
bench_check_sink(const char *what, int nb_args, int ntasks)
{
    int64_t last = (int64_t)ntasks * nb_args;
    int64_t expected = last * (last + 1) / 2;

    if( bench_sink != expected ) {
        fprintf(stderr, "%s: wrong argument values: summed %lld, expected %lld\n",
                what, (long long)bench_sink, (long long)expected);
        exit(1);
    }
}

static double
now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/*
 * One taskpool per measurement: a DTD taskpool caps the number of task
 * classes it can hold, and a fresh one also keeps each variant from seeing
 * another variant's task-class hash entries.
 */
static parsec_taskpool_t *
bench_tp_new(parsec_context_t *ctx)
{
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();
    int rc = parsec_context_add_taskpool(ctx, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    return tp;
}

static void
bench_tp_run_and_free(parsec_context_t *ctx, parsec_taskpool_t *tp,
                      parsec_task_class_t *tc, double *exec_sec)
{
    double t0;
    int rc;

    t0 = now_sec();
    rc = parsec_context_start(ctx);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    *exec_sec = now_sec() - t0;

    /* The task class belongs to the taskpool: release it first. */
    if( NULL != tc ) parsec_dtd_task_class_release(tp, tc);
    parsec_taskpool_free(tp);
}

#define BENCH_DEFINE(N)                                                          \
                                                                                 \
/* Body shared by the two hand-written variants. The parenthesized call is        \
 * token-for-token what parsec_dtd_unpack_args(t, &a0, ...) expands to. */        \
static int                                                                       \
bench_body_old_##N(parsec_execution_stream_t *es, parsec_task_t *this_task)       \
{                                                                                \
    (void)es;                                                                    \
    BENCH_PARAMS_##N(B_DECL)                                                     \
    (parsec_dtd_unpack_args)(this_task,                                          \
                             BENCH_PARAMS_##N(B_UNPACK_ARG)                      \
                             (void *)0);                                         \
    bench_sink += (0 BENCH_PARAMS_##N(B_SUM));                                   \
    return PARSEC_HOOK_RETURN_DONE;                                              \
}                                                                                \
                                                                                 \
PARSEC_DTD_DECLARE_TASK_CLASS(bench##N, BENCH_PARAMS_##N);                       \
                                                                                 \
static int                                                                       \
bench_body_new_##N(parsec_execution_stream_t *es, parsec_task_t *this_task)       \
{                                                                                \
    (void)es;                                                                    \
    PARSEC_DTD_UNPACK_LOCALS(bench##N, BENCH_PARAMS_##N, this_task);              \
    bench_sink += (0 BENCH_PARAMS_##N(B_SUM));                                   \
    return PARSEC_HOOK_RETURN_DONE;                                              \
}                                                                                \
                                                                                 \
/* Same task class and same insertion as bench_body_new_##N; the only            \
 * difference is that the body reads the values where they already are. */       \
static int                                                                       \
bench_body_inplace_##N(parsec_execution_stream_t *es, parsec_task_t *this_task)   \
{                                                                                \
    (void)es;                                                                    \
    const bench##N##_values_t *v = bench##N##_values(this_task);                  \
    bench_sink += (0 BENCH_PARAMS_##N(B_SUM_INPLACE));                           \
    return PARSEC_HOOK_RETURN_DONE;                                              \
}                                                                                \
                                                                                 \
static void                                                                      \
bench_direct_##N(parsec_context_t *ctx, int ntasks,                              \
                 double *insert_sec, double *exec_sec)                           \
{                                                                                \
    parsec_taskpool_t *tp = bench_tp_new(ctx);                                    \
    int seed = 1;                                                                \
    bench_sink = 0;                                                              \
    double t0 = now_sec();                                                       \
    for( int i = 0; i < ntasks; i++ ) {                                          \
        BENCH_PARAMS_##N(B_DECL_INIT)                                            \
        parsec_dtd_insert_task(tp, bench_body_old_##N, 0, PARSEC_DEV_CPU,         \
                               "direct" #N,                                      \
                               BENCH_PARAMS_##N(B_DIRECT_INSERT_ARG)             \
                               PARSEC_DTD_ARG_END);                              \
    }                                                                            \
    *insert_sec = now_sec() - t0;                                                \
    bench_tp_run_and_free(ctx, tp, NULL, exec_sec);                              \
    bench_check_sink("direct" #N, N, ntasks);                                    \
}                                                                                \
                                                                                 \
static void                                                                      \
bench_taskclass_##N(parsec_context_t *ctx, int ntasks,                            \
                    double *insert_sec, double *exec_sec)                        \
{                                                                                \
    parsec_taskpool_t *tp = bench_tp_new(ctx);                                    \
    parsec_task_class_t *tc =                                                    \
        parsec_dtd_create_task_class(tp, "taskclass" #N,                          \
                                     BENCH_PARAMS_##N(B_CLASS_ARG)               \
                                     PARSEC_DTD_ARG_END);                        \
    parsec_dtd_task_class_add_chore(tp, tc, PARSEC_DEV_CPU, bench_body_old_##N);  \
    int seed = 1;                                                                \
    bench_sink = 0;                                                              \
    double t0 = now_sec();                                                       \
    for( int i = 0; i < ntasks; i++ ) {                                          \
        BENCH_PARAMS_##N(B_DECL_INIT)                                            \
        parsec_dtd_insert_task_with_task_class(tp, tc, 0, PARSEC_DEV_CPU,         \
                                               BENCH_PARAMS_##N(B_TC_INSERT_ARG) \
                                               PARSEC_DTD_ARG_END);              \
    }                                                                            \
    *insert_sec = now_sec() - t0;                                                \
    bench_tp_run_and_free(ctx, tp, tc, exec_sec);                                \
    bench_check_sink("taskclass" #N, N, ntasks);                                 \
}                                                                                \
                                                                                 \
static void                                                                      \
bench_generated_##N(parsec_context_t *ctx, int ntasks,                            \
                    double *insert_sec, double *exec_sec)                        \
{                                                                                \
    parsec_taskpool_t *tp = bench_tp_new(ctx);                                    \
    parsec_task_class_t *tc = bench##N##_task_class_new(tp, 0);                   \
    parsec_dtd_task_class_add_chore(tp, tc, PARSEC_DEV_CPU, bench_body_new_##N);  \
    int seed = 1;                                                                \
    bench_sink = 0;                                                              \
    double t0 = now_sec();                                                       \
    for( int i = 0; i < ntasks; i++ ) {                                          \
        BENCH_PARAMS_##N(B_DECL_INIT)                                            \
        bench##N##_insert(tp, tc, 0, PARSEC_DEV_CPU, &(bench##N##_args_t){        \
            BENCH_PARAMS_##N(B_DESIG)                                            \
        });                                                                      \
    }                                                                            \
    *insert_sec = now_sec() - t0;                                                \
    bench_tp_run_and_free(ctx, tp, tc, exec_sec);                                \
    bench_check_sink("generated" #N, N, ntasks);                                 \
}                                                                                \
                                                                                 \
static void                                                                      \
bench_inplace_##N(parsec_context_t *ctx, int ntasks,                              \
                  double *insert_sec, double *exec_sec)                          \
{                                                                                \
    parsec_taskpool_t *tp = bench_tp_new(ctx);                                    \
    parsec_task_class_t *tc = bench##N##_task_class_new(tp, 0);                   \
    parsec_dtd_task_class_add_chore(tp, tc, PARSEC_DEV_CPU,                       \
                                    bench_body_inplace_##N);                     \
    int seed = 1;                                                                \
    bench_sink = 0;                                                              \
    double t0 = now_sec();                                                       \
    for( int i = 0; i < ntasks; i++ ) {                                          \
        BENCH_PARAMS_##N(B_DECL_INIT)                                            \
        bench##N##_insert(tp, tc, 0, PARSEC_DEV_CPU, &(bench##N##_args_t){        \
            BENCH_PARAMS_##N(B_DESIG)                                            \
        });                                                                      \
    }                                                                            \
    *insert_sec = now_sec() - t0;                                                \
    bench_tp_run_and_free(ctx, tp, tc, exec_sec);                                \
    bench_check_sink("inplace" #N, N, ntasks);                                   \
}

BENCH_DEFINE(1)
BENCH_DEFINE(2)
BENCH_DEFINE(3)
BENCH_DEFINE(4)
BENCH_DEFINE(5)
BENCH_DEFINE(6)
BENCH_DEFINE(7)
BENCH_DEFINE(8)
BENCH_DEFINE(9)
BENCH_DEFINE(10)

typedef void (*bench_fn_t)(parsec_context_t *, int, double *, double *);

#define BENCH_ROW(N) { N, bench_direct_##N, bench_taskclass_##N, \
                          bench_generated_##N, bench_inplace_##N }

static const struct {
    int        nb_args;
    bench_fn_t direct;
    bench_fn_t taskclass;
    bench_fn_t generated;
    bench_fn_t inplace;
} bench_table[] = {
    BENCH_ROW(1), BENCH_ROW(2), BENCH_ROW(3),  BENCH_ROW(4), BENCH_ROW(5),
    BENCH_ROW(6), BENCH_ROW(7), BENCH_ROW(8),  BENCH_ROW(9), BENCH_ROW(10),
};

#define NB_ARITIES ((int)(sizeof(bench_table)/sizeof(bench_table[0])))
#define NB_VARIANTS 4

#define MAX_REPS 256

static double
mean_of(const double *values, int n)
{
    double sum = 0.0;
    for( int i = 0; i < n; i++ ) sum += values[i];
    return sum / n;
}

/* Sample standard deviation (Bessel-corrected). */
static double
stddev_of(const double *values, int n, double mean)
{
    double sum_sq = 0.0;
    if( n < 2 ) return 0.0;
    for( int i = 0; i < n; i++ ) {
        double d = values[i] - mean;
        sum_sq += d * d;
    }
    return sqrt(sum_sq / (n - 1));
}

static double
min_of(const double *values, int n)
{
    double best = values[0];
    for( int i = 1; i < n; i++ ) if( values[i] < best ) best = values[i];
    return best;
}

int
main(int argc, char **argv)
{
    parsec_context_t *ctx;
    int rank = 0, world = 1;
    int ntasks = 200000, nreps = 25;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if( MPI_THREAD_MULTIPLE > provided ) {
            parsec_fatal("PaRSEC needs MPI_THREAD_MULTIPLE\n");
        }
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if( 1 != world ) {
        parsec_fatal("This benchmark is single process; run without mpirun or with -np 1\n");
    }

    if( argc > 1 && atoi(argv[1]) > 0 ) ntasks = atoi(argv[1]);
    if( argc > 2 && atoi(argv[2]) > 0 ) nreps  = atoi(argv[2]);

    /* One core, and no accelerators: we are timing the interface, not a kernel. */
    ctx = parsec_init(1, &argc, &argv);

    /*
     * Raise the window past the task count so the inserting thread never
     * breaks off to execute. Insertion and execution then measure separately.
     */
    parsec_dtd_window_size = 1 << 30;
    parsec_dtd_threshold_size = 1 << 30;

    if( nreps > MAX_REPS ) nreps = MAX_REPS;

    static const char *variant_name[NB_VARIANTS] = { "direct", "taskclass",
                                                     "generated", "inplace" };
    static double ins[NB_ARITIES][NB_VARIANTS][MAX_REPS];
    static double exe[NB_ARITIES][NB_VARIANTS][MAX_REPS];

    /*
     * Rep-major order. Sweeping every (arity, variant) inside each rep means
     * a slow stretch of the machine, or the CPU still ramping its frequency,
     * lands on all of them rather than on whichever happened to run first.
     */
    for( int r = -1; r < nreps; r++ ) {
        for( int a = 0; a < NB_ARITIES; a++ ) {
            bench_fn_t fns[NB_VARIANTS] = { bench_table[a].direct,
                                            bench_table[a].taskclass,
                                            bench_table[a].generated,
                                            bench_table[a].inplace };
            for( int vi = 0; vi < NB_VARIANTS; vi++ ) {
                /*
                 * Rotate which variant goes first from one rep to the next.
                 * The first measurement at each arity is a little slower than
                 * the rest whatever it is, so without rotating, that penalty
                 * would land on the same variant every time and read as a
                 * property of the interface. Rotating spreads it evenly and
                 * lets the standard deviation account for it.
                 */
                int v = (vi + (r + NB_VARIANTS)) % NB_VARIANTS;
                double i_sec, e_sec;

                /*
                 * Every measurement gets its own discarded run first. Task
                 * size grows with the arity, so the first run at a new arity
                 * pays to fault in differently sized mempool blocks; without
                 * this, whichever variant happened to be measured first at
                 * each arity looked ~20 ns/task slower than the other two.
                 */
                fns[v](ctx, ntasks, &i_sec, &e_sec);

                fns[v](ctx, ntasks, &i_sec, &e_sec);
                if( r < 0 ) continue;  /* whole-sweep warm-up sweep, discarded */
                ins[a][v][r] = i_sec;
                exe[a][v][r] = e_sec;
            }
        }
    }

    printf("# DTD argument passing cost, %d tasks per measurement, %d timed reps "
           "(each preceded by a discarded run, plus one discarded warm-up sweep), 1 core\n",
           ntasks, nreps);
    printf("# ns per task; sd is the sample standard deviation over the %d reps\n", nreps);
    printf("args,variant,insert_mean,insert_sd,insert_min,exec_mean,exec_sd,exec_min\n");
    for( int a = 0; a < NB_ARITIES; a++ ) {
        for( int v = 0; v < NB_VARIANTS; v++ ) {
            double scale = 1.0e9 / ntasks;
            double im = mean_of(ins[a][v], nreps) * scale;
            double em = mean_of(exe[a][v], nreps) * scale;
            double isd = stddev_of(ins[a][v], nreps, im / scale) * scale;
            double esd = stddev_of(exe[a][v], nreps, em / scale) * scale;
            printf("%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                   bench_table[a].nb_args, variant_name[v],
                   im, isd, min_of(ins[a][v], nreps) * scale,
                   em, esd, min_of(exe[a][v], nreps) * scale);
        }
    }

    /* Keep the accumulator observable so no body can be optimized away. */
    printf("# checksum %lld\n", (long long)bench_sink);

    parsec_fini(&ctx);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    (void)rank;
    return 0;
}
