#include "papime.h"

static void handle_error(int e){
  char m[512];

  PAPI_perror(e, m, 512);
  fprintf(stderr,"Error '%s' Occured. Exiting\n", m);
//  exit(e);
}

static int Events[4] = {PAPI_TLB_TL, PAPI_L1_DCM, PAPI_L1_DCA, PAPI_L2_DCM};
static long long int values[4] = { 0, 0, 0, 0 };

void papime_start_thread_counters(void){
  PAPI_start_counters(Events, 1);
}

void papime_acc_thread_counters(void){
  PAPI_acc_counters(Events, values);
}

void papime_stop_thread_counters(void){
  long long int values[4];
  int ret;

  if( (ret = PAPI_stop_counters(values, 1)) != PAPI_OK )
    handle_error(ret);

  printf("Thread: %d says L1_DCA: %lld  L1_DCM: %lld  L2_DCM: %lld  TLB_TL: %lld\n",pthread_self(), values[0], values[1], values[2], values[3]);
}

void papime_start(void){
    (void)PAPI_num_counters();

    PAPI_thread_init(pthread_self);
}

void papime_stop(void){
  PAPI_shutdown();
}
