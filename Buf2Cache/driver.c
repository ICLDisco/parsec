#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include "buf2cache.h"
#include "timer.h"

int checkAll(void *array_ptr, int tid);
int tile_size = 120*120*8;
int NP=16;

void *thread_main(void *arg);

int main(int argc, char **argv){
    int i, ids[NP];
    pthread_t tID[NP];

    dplasma_hwloc_init_cache(16, 1, 1, 64*1024, tile_size);
    dplasma_hwloc_init_cache(16, 2, 2, 4*1024*1024, tile_size);

    srand48(time(NULL));
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
    long n, N;
    ticks_t t1, t2;

    mytid = (*(int *)arg);
    n = 0;

    N=20000;
    t1 = getticks();
    for(i=0; i<N; ++i){
        indx = (int)(lrand48()%54);

        dplasma_hwloc_insert_buffer(A[indx], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+1], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+2], tile_size, mytid);

        n += checkAll(A[indx+1], mytid);

        dplasma_hwloc_insert_buffer(A[indx+3], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+4], tile_size, mytid);

        n += checkAll(A[indx], mytid);

        n += checkAll(A[indx+6], mytid);
        n += checkAll(A[indx+5], mytid);

        dplasma_hwloc_insert_buffer(A[indx+7], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+8], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+9], tile_size, mytid);
        dplasma_hwloc_insert_buffer(A[indx+10], tile_size, mytid);

        n += checkAll(A[indx+10], mytid);
        n += checkAll(A[indx+2], mytid);
        n += checkAll(A[indx+8], mytid);

    }
    t2 = getticks();
    double dt = elapsed(t2,t1);
    printf("[tid:%d] time per operation = %.3lf (usec)\n",mytid, dt/(16*N));
    return NULL;
}


int checkAll(void *array_ptr, int tid) {
    int isLcl, level;

    for(level=1; level<3; level++){
        isLcl = dplasma_hwloc_isLocal(array_ptr, level, tid);
        if( isLcl )
            return 1;
//            printf(" --> Array %p is in the L%d cache of PU:%d\n", array_ptr, level, myPID);
    }
    return 0;
}
