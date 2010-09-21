#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#if defined(HAVE_SCHED_SETAFFINITY) && defined (__USE_GNU) || 1
#include <linux/unistd.h>
#define gettid() syscall(__NR_gettid)
#else
#warning Not using sched_setaffinity
#endif  /* HAVE_SCHED_SETAFFINITY */

#include <sched.h>

#include <cblas.h>

#define MAX_THREADS 128

static int NB = 120;
static volatile int running = 1;

#define TYPE      float
#define GEMM_FUNC cblas_sgemm
/*
#define TYPE      double
#define GEMM_FUNC cblas_dgemm
*/
static void run_gemm(const TYPE *A, const TYPE *B, TYPE *C)
{
    GEMM_FUNC( (const enum CBLAS_ORDER)CblasColMajor, 
               (const enum CBLAS_TRANSPOSE)CblasNoTrans, (const enum CBLAS_TRANSPOSE)CblasNoTrans,
               NB /* A.rows */, 
               NB /* A.cols */,
               NB /* B.rows */,
               1.0, (TYPE*)A, NB /* A.data_stride */,
                    (TYPE*)B, NB /* B.data_stride */,
               1.0, (TYPE*)C, NB /* C.data_stride */);
}

static TYPE *init_matrix(void)
{
    TYPE *res;
    int i, j;
    res = (TYPE*)calloc(NB*NB, sizeof(TYPE));
    for(i = 0; i < NB; i++)
        for(j = 0; j < NB; j++)
            res[i*NB+j] = (TYPE)rand() / (TYPE)RAND_MAX;
    return res;
}

static void *thread_loop(void *_proc)
{
    TYPE *A, *B, *C;
    unsigned long int proc = (unsigned long int)_proc;
    int i;
    unsigned long long int time;
    struct timespec start, end;

    A = init_matrix();
    B = init_matrix();
    C = init_matrix();

#if defined(HAVE_SCHED_SETAFFINITY) && defined (__USE_GNU)
    {
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
        CPU_SET(proc, &cpuset);

        if( -1 == sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) ) {
            printf( "Unable to set the thread affinity (%s)\n", strerror(errno) );
        }
    }
#endif  /* HAVE_SCHED_SETAFFINITY */

    if( proc == 0 ) {
        for(i = 0; i < 100; i++) {
            /* Ensures that all threads run cblas_dgemm on another core */
            /* while warming up */
            run_gemm(A, B, C);
        }

        for(i = 0; i < 1000; i++) {
            /* Take the time */
            clock_gettime(CLOCK_REALTIME, &start);
            run_gemm(A, B, C);
            clock_gettime(CLOCK_REALTIME, &end);
            time = end.tv_nsec - start.tv_nsec + ( (end.tv_sec - start.tv_sec) * 1000000000ULL );
            printf("NB = %d TIME = %llu ns  %f GFlops\n", NB, time,
                   (2*(NB/1e3)*(NB/1e3)*(NB/1e3)) / ((double)time  / 1e9));
        }

        running = 0;
    } else {
        while( running ) {
            run_gemm(A, B, C);
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    unsigned long int i, nbcores;
    pthread_t threads[MAX_THREADS];

    if( argc != 3 ) {
        fprintf(stderr, "usage: %s <Matrix Size> <Nb Cores>\n", argv[0]);
        return 1;
    }

    NB = atoi(argv[1]);
    nbcores = atoi(argv[2]);

    if( NB <= 1 || nbcores < 1 ) {
        fprintf(stderr, "usage: %s <Matrix Size> <Nb Cores>\n", argv[0]);
        return 1;
    }

    for(i = 0; i < nbcores; i++) {
        pthread_create(&threads[i], NULL, thread_loop, (void*)(i+1));
    }
    
    thread_loop((void*)0);

    for(i = 0; i < nbcores; i++) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}
