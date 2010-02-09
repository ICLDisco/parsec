#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include "buf2cache2.h"
#include "timer.h"

int tile_size = 120*120*8;
int NP=6;
cache_t *L2[6], *L3;

void *thread_main(void *arg);

int main(int argc, char **argv){
    int i, ids[NP];
    pthread_t tID[NP];

    L3 = cache_create(6, NULL, (int)(6*1024*1024/tile_size)-1);
    for(i=0; i<6; i++){
        L2[i] = cache_create(1, L3, (int)(512*1024/tile_size)-1);
    }

    for(i=0; i<NP; i++){
        ids[i] = i;
        pthread_create(&tID[i], NULL, thread_main, &ids[i]);
    }
    for(i=0; i<NP; i++){
        pthread_join(tID[i], NULL);
    }

    fflush(stdout);
    return 0;
}

void *thread_main(void *arg){
    int A[64][2];
    int i, mytid, indx;
    long n, N, failed =0;
    ticks_t t1, t2;
    cache_t *myL2;

    mytid = (*(int *)arg);
    n = 0;

    myL2 = L2[mytid];
    srand48(mytid*(time(NULL)%123456));

    indx = 0;

    cache_buf_referenced(myL2, A[indx+0]);
    cache_buf_referenced(myL2, A[indx+1]);
    cache_buf_referenced(myL2, A[indx+2]);
    cache_buf_referenced(myL2, A[indx+3]);
    cache_buf_referenced(myL2, A[indx+4]);

    if( cache_buf_isLocal(myL2,  A[indx+2]) == 1 )
        printf("[tid:%d] pass 1\n", mytid);
    else{
        printf("[tid:%d] fail 1\n", mytid);
        ++failed;
    }

    if( cache_buf_isLocal(myL2,  A[indx+0]) == 0 )
        printf("[tid:%d] pass 2\n", mytid);
    else{
        printf("[tid:%d] fail 2\n", mytid);
        ++failed;
    }

    if( cache_buf_distance(myL2, A[indx+0]) == 1)
        printf("[tid:%d] pass 3\n", mytid);
    else{
        printf("[tid:%d] fail 3\n", mytid);
        ++failed;
    }

    if( cache_buf_age(myL2, A[indx+0]) == -1 )
        printf("[tid:%d] pass 4\n", mytid);
    else{
        printf("[tid:%d] fail 4\n", mytid);
        ++failed;
    }

    cache_buf_referenced(myL2, A[indx+0]);

    if( cache_buf_distance(myL2, A[indx+0]) == 0 )
        printf("[tid:%d] pass 5\n", mytid);
    else{
        printf("[tid:%d] fail 5\n", mytid);
        ++failed;
    }

    if( cache_buf_age(myL2, A[indx+0]) == 0 )
        printf("[tid:%d] pass 6\n", mytid);
    else{
        printf("[tid:%d] fail 6\n", mytid);
        ++failed;
    }
    fflush(stdout);
    usleep(50000);

    if( failed == 0 )
        printf("[tid:%d] All tests passed\n", mytid);

    fflush(stdout);
    usleep(100000);

    if( mytid == 0 )
        printf("---- Running Performance Test ----\n");

    N=20000;
    t1 = getticks();
    for(i=0; i<N; ++i){
        indx = (int)(lrand48()%54);

        cache_buf_referenced(myL2, A[indx]);
        cache_buf_referenced(myL2, A[indx+1]);
        cache_buf_referenced(myL2, A[indx+2]);
        n += cache_buf_isLocal(myL2, A[indx+1]);
        n += cache_buf_isLocal(myL2, A[indx+2]);
        n += cache_buf_distance(myL2, A[indx+2]);
        cache_buf_referenced(myL2, A[indx+3]);
        cache_buf_referenced(myL2, A[indx+4]);
        n += cache_buf_isLocal(myL2, A[indx+6]);
        n += cache_buf_distance(myL2, A[indx+5]);
        n += cache_buf_age(myL2, A[indx+10]);
        n += cache_buf_age(myL2, A[indx+2]);
        n += cache_buf_age(myL2, A[indx+6]);
        n += cache_buf_distance(myL2, A[indx+2]);
        n += cache_buf_isLocal(myL2, A[indx+1]);
        n += cache_buf_isLocal(myL2, A[indx+5]);
        n += cache_buf_isLocal(myL2, A[indx+8]);
        n += cache_buf_distance(myL2, A[indx+8]);
        n += cache_buf_isLocal(myL2, A[indx]);
        n += cache_buf_distance(myL2, A[indx]);
        n += cache_buf_age(myL2, A[indx]);
        cache_buf_referenced(myL2, A[indx]);
        cache_buf_referenced(myL2, A[indx+5]);
        cache_buf_referenced(myL2, A[indx+6]);
        cache_buf_referenced(myL2, A[indx+7]);
        cache_buf_referenced(myL2, A[indx+1]);
        cache_buf_referenced(myL2, A[indx+8]);
        n += cache_buf_isLocal(myL2, A[indx+6]);
        n += cache_buf_distance(myL2, A[indx+5]);
        n += cache_buf_age(myL2, A[indx+10]);
        n += cache_buf_age(myL2, A[indx+2]);
        n += cache_buf_age(myL2, A[indx+6]);
        n += cache_buf_distance(myL2, A[indx+2]);
        n += cache_buf_isLocal(myL2, A[indx+1]);
        n += cache_buf_isLocal(myL2, A[indx+5]);
        n += cache_buf_isLocal(myL2, A[indx+8]);
        n += cache_buf_distance(myL2, A[indx+8]);
        n += cache_buf_isLocal(myL2, A[indx]);
        n += cache_buf_distance(myL2, A[indx]);
        n += cache_buf_age(myL2, A[indx]);

    }
    t2 = getticks();
    double dt = elapsed(t2,t1);
    printf("[tid:%d] time per operation = %.3lf (usec)\n",mytid, dt/(40*N));
    return NULL;
}

