#ifndef _PAPIME_H_
#define _PAPIME_H_
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "dague_config.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

void papime_start_thread_counters(void);
void papime_stop_thread_counters(void);
void papime_start(void);
void papime_stop(void);
#endif

