#include <sys/time.h>

typedef double ticks_t;
static inline ticks_t getticks(void){
     ticks_t ret;
     struct timeval tv;

     gettimeofday(&tv, NULL);
     ret = 1000*1000*(ticks_t)tv.tv_sec + (ticks_t)tv.tv_usec;
     return ret;
}

static inline double elapsed(ticks_t t1, ticks_t t0){ 
     return (double)t1 - (double)t0;
} 
