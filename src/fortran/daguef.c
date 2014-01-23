/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>
#include <string.h>
#include <stdlib.h>

void dague_init_f08(int nbcores, dague_context_t** context, int* ierr)
{
    char *args, *token, **argv = NULL;
    int argc = 0;

    if( NULL != (args = getenv("DAGUE_ARGS"))) {
        args = token = strdup(args);
        while(NULL != strsep(&args, ";=")) argc++;
        argv = (char**)malloc((2+argc) * sizeof(char*));
        free(token);
        args = strdup(getenv("DAGUE_ARGS"));
        argc = 1;
        argv[0] = "myapp";  /* No idea how to extract the real application name from Fortran */
        while( NULL != (token = strsep(&args, ";=")) ) {
            argv[argc] = token;
            argc++;
        }
        argv[argc] = NULL;
    }
    *context = dague_init(nbcores, &argc, &argv);
    *ierr = (NULL == *context) ? 0 : -1;
}

void dague_fini_f08(dague_context_t** context, int* ierr)
{
    *ierr = dague_fini(context);
}

void dague_set_complete_callback_f08(dague_handle_t** object,
                                     dague_completion_cb_t complete_cb,
                                     void* cb_data, int* ierr)
{
    *ierr = dague_set_complete_callback(*object, complete_cb, cb_data);
}

void dague_get_complete_callback_f08(dague_handle_t** object,
                                     dague_completion_cb_t* complete_cb,
                                     void** cb_data, int* ierr)
{
    *ierr = dague_get_complete_callback(*object, complete_cb, cb_data);
}

void dague_set_priority_f08(dague_handle_t** object, int priority, int* ierr)
{
    *ierr = dague_set_priority(*object, priority);
}

