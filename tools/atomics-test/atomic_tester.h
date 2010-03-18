#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//#include <time.h>
#include <pthread.h>
#include "atomic.h"

void incr_glb_val(void);
void *thread_main(void *arg);
