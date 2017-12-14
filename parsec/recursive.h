/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _PARSEC_RECURSIVE_H_
#define _PARSEC_RECURSIVE_H_

#include "parsec/parsec_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/scheduling.h"
#include "parsec/devices/device.h"
#include "parsec/data_dist/matrix/matrix.h"

typedef struct cb_data_s {
    parsec_execution_stream_t    *es;
    parsec_task_t *context;
    void (*destruct)( parsec_taskpool_t * );
    int nbdesc;
    parsec_data_collection_t *desc[1];
} cb_data_t;

static inline int parsec_recursivecall_callback(parsec_taskpool_t* tp, void* cb_data)
{
    int i, rc = 0;
    cb_data_t* data = (cb_data_t*)cb_data;

    rc = __parsec_complete_execution(data->es, data->context);

    for(i=0; i<data->nbdesc; i++){
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)(data->desc[i]) );
        free( data->desc[i] );
    }

    data->destruct( tp );
    free(data);

    return rc;
}

static inline int
parsec_recursivecall( parsec_execution_stream_t    *es,
                      parsec_task_t *context,
                      parsec_taskpool_t            *tp,
                      void (*taskpool_destroy)(parsec_taskpool_t *),
                      int nbdesc,
                      ... )
{
    cb_data_t *cbdata = NULL;
    int i;
    va_list ap;

    /* Set mask to be used only on CPU */
    parsec_devices_taskpool_restrict( tp, PARSEC_DEV_CPU );

    /* Callback */
    cbdata = (cb_data_t *) malloc( sizeof(cb_data_t) + (nbdesc-1)*sizeof(parsec_data_collection_t*));
    cbdata->es       = es;
    cbdata->context  = context;
    cbdata->destruct = taskpool_destroy;
    cbdata->nbdesc   = nbdesc;

    /* Get descriptors */
    va_start(ap, nbdesc);
    for(i=0; i<nbdesc; i++){
        cbdata->desc[i] = va_arg(ap, parsec_data_collection_t *);
    }
    va_end(ap);

    parsec_taskpool_set_complete_callback( tp, &parsec_recursivecall_callback,
                                           (void *)cbdata );

    parsec_enqueue( es->virtual_process->parsec_context, tp);

    return -1;
}

#endif /* _PARSEC_RECURSIVE_H_ */
