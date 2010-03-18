#include "atomic_tester.h"

uint32_t NP = 40, N = 2000;
volatile uint32_t glb_counter;
volatile uint32_t sync_counter;

int main(int argc, char **argv){
    int i, *ids;
    pthread_t tID[NP];

    glb_counter = 0;
    sync_counter = 0;
    ids = (int *)malloc( NP*sizeof(int) );

    for(i=0; i<NP; i++){
        ids[i] = i;
        pthread_create(&tID[i], NULL, thread_main, &ids[i]);
    }
    for(i=0; i<NP; i++){
        pthread_join(tID[i], NULL);
    }

    if( glb_counter != N*NP )
        printf("glb_counter = %u, expected value = %u\n",glb_counter, N*NP);
    else
        printf("Pass\n");

    return 0;
}


void *thread_main(void *arg){
    int i, tid;

    tid = *((int *)arg);

    /* This is like a barrier, only it relies on atomics */
    dplasma_atomic_inc_32b(&sync_counter);
    while(sync_counter < NP );

    if( sync_counter != NP ){
        printf("[%d] Error: sync_counter != NP (%u!=%u)\n",tid, sync_counter, NP);
        return NULL;
    }

    for(i=0; i<N; ++i)
        incr_glb_val();

    return NULL;
}


void incr_glb_val(void){
    uint32_t oldV;
    do {
        oldV = glb_counter;
    } while( !dplasma_atomic_cas_xxb( &glb_counter, oldV, oldV+1, sizeof(glb_counter) ) );
}
