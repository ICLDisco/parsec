/*
 * Copyright (c) 2015-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _PARSEC_RECURSIVE_H_
#define _PARSEC_RECURSIVE_H_

#include "parsec/execution_stream.h"
#include "parsec/scheduling.h"
#include "parsec/mca/device/device.h"
#include "parsec/data_dist/matrix/matrix.h"

typedef struct parsec_recursive_callback_s parsec_recursive_callback_t;
typedef void (*parsec_recursive_callback_f)(parsec_taskpool_t*, const parsec_recursive_callback_t* );

struct parsec_recursive_callback_s {
    parsec_task_t                *task;
    parsec_recursive_callback_f   callback;
    int nbdesc;
    parsec_data_collection_t      *desc[1];
};

static inline int parsec_recursivecall_callback(parsec_taskpool_t* tp, void* cb_data)
{
    parsec_recursive_callback_t* data = (parsec_recursive_callback_t*)cb_data;
    parsec_execution_stream_t *es = parsec_my_execution_stream();
    int i, rc = 0;

    /* first trigger the internal taskpool completion callback */
    data->callback( tp, data );
    /* then complete the generator task */
    rc = __parsec_complete_execution(es, data->task);
    /* and finally release the data associated with the inner taskpool */
    for( i = 0; i < data->nbdesc; i++ ) {
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)(data->desc[i]) );
        free( data->desc[i] );
    }
    free(data);
    return rc;
}

static inline int
parsec_recursivecall( parsec_task_t                *task,
                      parsec_taskpool_t            *tp,
                      parsec_recursive_callback_f   callback,
                      int nbdesc,
                      ... )
{
    parsec_recursive_callback_t *cbdata = NULL;
    va_list ap;

    /* Set mask to be used only on CPU */
    parsec_mca_device_taskpool_restrict( tp, PARSEC_DEV_CPU );

    /* Callback */
    cbdata = (parsec_recursive_callback_t *) malloc( sizeof(parsec_recursive_callback_t) + (nbdesc-1)*sizeof(parsec_data_collection_t*));
    cbdata->task     = task;
    cbdata->callback = callback;
    cbdata->nbdesc   = nbdesc;

    /* Get descriptors */
    va_start(ap, nbdesc);
    for(int i = 0; i < nbdesc; i++ ) {
        cbdata->desc[i] = va_arg(ap, parsec_data_collection_t *);
    }
    va_end(ap);

    parsec_taskpool_set_complete_callback( tp, &parsec_recursivecall_callback,
                                           (void *)cbdata );

    parsec_context_add_taskpool( task->taskpool->context, tp);

    return -1;
}

#endif /* _PARSEC_RECURSIVE_H_ */
