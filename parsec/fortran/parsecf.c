/*
 * Copyright (c) 2013-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include <string.h>
#include <stdlib.h>

/**
 * @brief A function callable from Fortran allowing to initialize the PaRSEC runtime
 * 
 * @param nbcores 
 * @param context 
 * @param ierr 
 * 
 * As there is no standard way to find the arg[cv] in Fortran, we build them using
 * the PARSEC_ARGS environment variable.
 */
void parsec_init_f08(int nbcores, parsec_context_t** context, int* ierr)
{
    char *args = NULL, *token, **argv = NULL;
    int argc = 0;

    if( NULL != (args = getenv("PARSEC_ARGS"))) {
        args = token = strdup(args);
        while( NULL != (token = strpbrk(token, "=;")) ) {  /* count the number of argv */
            argc++;
        }
        argv = (char**)malloc((2+argc) * sizeof(char*));
        argc = 1;
        argv[0] = "myapp";  /* No idea how to extract the real application name from Fortran */
        token = strtok(args, "=;");
        while( NULL != token ) {
            argv[argc++] = token;
            token = strtok(NULL, " =;");
        }
        argv[argc] = NULL;
    }
    *context = parsec_init(nbcores, &argc, &argv);
    /* Cleanup the locals used to initialize parsec */
    free(argv);
    if( NULL != args )
        free(args);
    *ierr = (NULL == *context) ? 0 : -1;
}

void parsec_fini_f08(parsec_context_t** context, int* ierr)
{
    *ierr = parsec_fini(context);
}

void
parsec_taskpool_set_complete_callback_f08(parsec_taskpool_t** tp,
                                          parsec_event_cb_t complete_cb,
                                          void* cb_data, int* ierr)
{
    *ierr = parsec_taskpool_set_complete_callback(*tp, complete_cb, cb_data);
}

void
parsec_taskpool_get_complete_callback_f08(parsec_taskpool_t** tp,
                                          parsec_event_cb_t* complete_cb,
                                          void** cb_data, int* ierr)
{
    *ierr = parsec_taskpool_get_complete_callback(*tp, complete_cb, cb_data);
}

void
parsec_taskpool_set_enqueue_callback_f08(parsec_taskpool_t** tp,
                                         parsec_event_cb_t enqueue_cb,
                                         void* cb_data, int* ierr)
{
    *ierr = parsec_taskpool_set_enqueue_callback(*tp, enqueue_cb, cb_data);
}

void
parsec_taskpool_get_enqueue_callback_f08(parsec_taskpool_t** tp,
                                         parsec_event_cb_t* enqueue_cb,
                                         void** cb_data, int* ierr)
{
    *ierr = parsec_taskpool_get_enqueue_callback(*tp, enqueue_cb, cb_data);
}

void
parsec_taskpool_set_priority_f08(parsec_taskpool_t** tp, int priority, int* ierr)
{
    *ierr = parsec_taskpool_set_priority(*tp, priority);
}

