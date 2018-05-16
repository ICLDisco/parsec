#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "papi.h"

void __attribute__ ((constructor)) init(void);

extern char **environ;
static int nbcounters = 0;

void *thread_fct(void *_)
{
    int i, retval;
    char **ev, *c;
    long long *counts;
    int eventset = PAPI_NULL;
    char **counternames = NULL;

    counternames = (char**)malloc(nbcounters * sizeof(char**));
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr, "STD_QUICK_TEST:: Wrong PAPI at runtime: expected %d, got %d\n", PAPI_VER_CURRENT, retval);
        return NULL;
    }
    retval=PAPI_create_eventset(&eventset);
    if (retval!=PAPI_OK) {
        fprintf(stderr, "SDE_QUICK_TEST:: Unable to create event set\n");
        return NULL;
    }

    nbcounters = 0;
    for(ev = environ; *ev != NULL; ev++) {
        if( strncmp(*ev, "SDE_QUICK_TEST", 14) == 0 ) {
            for( c = *ev; *c != '\0' && *c != '='; c++) /* nothing */;
            if( *c == '\0' )
                continue;
            c++;
            if( *c == '\0' )
                continue;
            
            retval=PAPI_add_named_event(eventset, c);
            if (retval!=PAPI_OK) {
                fprintf(stderr, "SDE_QUICK_TEST:: unable to add event '%s' to eventset (PAPI error: '%s')\n",
                        c, PAPI_strerror(retval));
                continue;
            }
            counternames[nbcounters] = strdup(c);
            nbcounters++;
        }
    }
    if( nbcounters == 0 ) {
        fprintf(stderr, "SDE_QUICK_TEST:: no counter succesfully defined, bailing out (have you set PAPI_SDE_QUICK_TEST environment variables?)\n");
        return NULL;
    }
    
    counts = (long long*)malloc(sizeof(long long) * nbcounters);
    PAPI_reset(eventset);
    PAPI_start(eventset);
    while(1) {
        usleep(10000);
        PAPI_stop(eventset, counts);
        PAPI_reset(eventset);
        PAPI_start(eventset);
        for(i = 0; i < nbcounters; i++) {
            fprintf(stderr, "%s=%lld\t", counternames[i], counts[i]);
        }
        fprintf(stderr, "\n");
    }
}

void init(void)
{
    pthread_t my_thread;
    char **ev, *c;

    for(ev = environ; *ev != NULL; ev++) {
        if( strncmp(*ev, "SDE_QUICK_TEST", 14) == 0 ) {
            for( c = *ev; *c != '\0' && *c != '='; c++) /* nothing */;
            if( *c == '\0' )
                continue;
            c++;
            if( *c == '\0' )
                continue;
            nbcounters++;
        }
    }
    
    if( 0 == nbcounters ) {
        fprintf(stderr, "SDE_QUICK_TEST:: No environment variable starting with SDE_QUICK_TEST defined, I don't know what events to log, bailing out\n");
        return;
    }
    
    pthread_create(&my_thread, NULL, thread_fct, NULL);
    pthread_detach(my_thread);   
}
