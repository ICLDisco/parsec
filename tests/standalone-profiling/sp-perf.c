#include <stdlib.h>
#include <pthread.h>
#ifdef PARSEC_HAVE_PTHREAD_BARRIER_H
/* Mac OS X pthread.h does not provide the pthread_barrier by default */
#include <pthread-barrier.h>
#endif PARSEC_HAVE_PTHREAD_BARRIER_H
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <string.h>
#include <stdio.h>
#include "profiling.h"

typedef struct {
    pthread_t                 pthread_id;
    int                       thread_index;
    parsec_thread_profiling_t *prof;
    struct timeval            duration;
    double                    dummy;
} per_thread_info_t;

static int event_startkey, event_endkey;
static pthread_barrier_t barrier;
static uint32_t tasks_per_thread = 100;
static int profiling;

#define D 32
static void cpuburn(double *a, double *b, double *c)
{
    int i, j, k;
    for(i = 0; i < D; i++)
        for(j = 0; j < D; j++)
            for(k = 0; k < D; k++)
                c[i*D+j] += a[i*D+k] * b[k*D+j];
}

static void *run_thread(void *_arg)
{
    per_thread_info_t *ti = (per_thread_info_t*)_arg;
    uint32_t i;
    struct timeval start, end;
    double a[D*D], b[D*D], c[D*D];

    if( profiling )
        ti->prof = parsec_profiling_thread_init(4096, "Thread %d", ti->thread_index);

    for(i = 0; i < D*D; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
        c[i] = (double)rand() / RAND_MAX;
    }
    
    pthread_barrier_wait(&barrier);
    gettimeofday(&start, NULL);
    
    for(i = 0; i < tasks_per_thread; i++) {
        if(profiling)
            parsec_profiling_trace_flags(ti->prof, event_startkey, i, ti->thread_index, NULL, 0);
        cpuburn(a, b, c);
        if(profiling)
            parsec_profiling_trace_flags(ti->prof, event_endkey, i, ti->thread_index, NULL, 0);
    }

    gettimeofday(&end, NULL);
    timersub(&end, &start, &ti->duration);
    ti->dummy = c[0];
    
    return NULL;
}

int main(int argc, char *argv[])
{
    int i, opt;
    per_thread_info_t *thread_info;
    int nbthreads = 1;
    char *filename = NULL;

    while ((opt = getopt(argc, argv, "f:n:N:h?")) != -1) {
        switch (opt) {
        case 'f':
            filename = strdup(optarg);
            break;
        case 'n':
            nbthreads = atoi(optarg);
            break;
        case 'N':
            tasks_per_thread = atoi(optarg);
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s [-f filename] [-n number of threads] [-N number of tasks per thread]\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if(NULL == filename) {
        profiling = 0;
    } else {
        profiling = 1;
    }

    if( profiling ) {
        parsec_profiling_init();
        if( parsec_profiling_dbp_start(filename, "PaRSEC profiling system performance evaluation" ) == -1 )
            exit(EXIT_FAILURE);

        parsec_profiling_add_dictionary_keyword("Event", "#FF0000", 0, NULL, &event_startkey, &event_endkey);

        parsec_profiling_start();
    }

    pthread_barrier_init(&barrier, NULL, nbthreads);
    thread_info = (per_thread_info_t *)calloc(nbthreads, sizeof(per_thread_info_t));
    
    for(i = 1; i < nbthreads; i++) {
        thread_info[i].thread_index = i;
        pthread_create(&thread_info[i].pthread_id, NULL, run_thread, &thread_info[i]);
    }

    thread_info[0].thread_index = 0;
    run_thread(&thread_info[0]);

    for(i = 1; i < nbthreads; i++)
        pthread_join(thread_info[i].pthread_id, NULL);

    if( profiling ) {
        parsec_profiling_dbp_dump();
        parsec_profiling_fini();
    }

    for(i = 0; i < nbthreads; i++) {
        fprintf(stderr, "Thread %d Total Time (s): %d.%06d\n", i, (int)thread_info[i].duration.tv_sec, (int)thread_info[i].duration.tv_usec);
    }
    free(thread_info);

    exit(EXIT_SUCCESS);
}
