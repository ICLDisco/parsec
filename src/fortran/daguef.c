/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>

void dague_init_f08(int nbcores, dague_context_t** context, int* ierr)
{
    *context = dague_init(nbcores, NULL, NULL);
    *ierr = (NULL == *context) ? 0 : -1;
}

void dague_fini_f08(dague_context_t** context, int* ierr)
{
    *ierr = dague_fini(context);
}

dague_handle_t* dague_compose_f08(dague_handle_t** start, dague_handle_t** next)
{
    dague_handle_t* object = dague_compose(*start, *next);
    return object;
}

void dague_enqueue_f08(dague_context_t** context, dague_handle_t** object, int* ierr)
{
    *ierr = dague_enqueue(*context, *object);
}

void dague_progress_f08(dague_context_t** context, int* ierr)
{
    *ierr = dague_progress(*context);
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

