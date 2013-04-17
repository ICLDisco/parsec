/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>

dague_context_t* dague_init_f08(int nbcores, int* ierr)
{
    dague_context_t* context;

    context = dague_init(nbcores, NULL, NULL);
    *ierr = (NULL == context) ? 0 : -1;
    return context;
}

void dague_fini_f08(dague_context_t** context, int* ierr)
{
    *ierr = dague_fini(context);
}

dague_object_t* dague_compose_f08(dague_object_t** start, dague_object_t** next)
{
    dague_object_t* object = dague_compose(*start, *next);
    return object;
}

void dague_enqueue_f08(dague_context_t** context, dague_object_t** object, int* ierr)
{
    *ierr = dague_enqueue(*context, *object);
}

void dague_progress_f08(dague_context_t** context, int* ierr)
{
    *ierr = dague_progress(*context);
}

void dague_set_complete_callback_f08(dague_object_t** object,
                                     dague_completion_cb_t complete_cb,
                                     void* cb_data, int* ierr)
{
    *ierr = dague_set_complete_callback(*object, complete_cb, cb_data);
}

void dague_get_complete_callback_f08(dague_object_t** object,
                                     dague_completion_cb_t* complete_cb,
                                     void** cb_data, int* ierr)
{
    *ierr = dague_get_complete_callback(*object, complete_cb, cb_data);
}

void dague_set_priority_f08(dague_object_t** object, int priority, int* ierr)
{
    *ierr = dague_set_priority(*object, priority);
}

