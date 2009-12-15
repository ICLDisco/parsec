#include <stdio.h>
#include <sys/time.h>

#include "dplasma.h"

static double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

int main(int argc, char *argv[])
{
    int NB, NT, N;
    double time_elapsed;

    N = atoi(argv[1]);
    NB = atoi(argv[2]);

    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    time_elapsed = get_cur_time();
    load_dplasma_objects();
    time_elapsed = get_cur_time() - time_elapsed;

    printf("load_dplasma_objects(): %gs\n", time_elapsed);

    {
        expr_t* constant;
        
        constant = expr_new_int( NB );
        dplasma_assign_global_symbol( "NB", constant );
        
        constant = expr_new_int( NT );
        dplasma_assign_global_symbol( "SIZE", constant );
    }

    time_elapsed = get_cur_time();
    load_dplasma_hooks();
    time_elapsed = get_cur_time() - time_elapsed;

    printf("load_dplasma_hooks(): %gs\n", time_elapsed);

    time_elapsed = get_cur_time();
    enumerate_dplasma_tasks();
    time_elapsed = get_cur_time() - time_elapsed;
    printf("enumerate_dplasma_tasks(): %gs\n", time_elapsed);

    return 0;
}
