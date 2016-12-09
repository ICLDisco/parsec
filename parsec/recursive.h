#ifndef _PARSEC_RECURSIVE_H_
#define _PARSEC_RECURSIVE_H_

#include "parsec/parsec_internal.h"
#include "parsec/execution_unit.h"
#include "parsec/scheduling.h"
#include "parsec/devices/device.h"
#include "data_dist/matrix/matrix.h"

typedef struct cb_data_s {
    parsec_execution_unit_t    *eu;
    parsec_execution_context_t *context;
    void (*destruct)( parsec_handle_t * );
    int nbdesc;
    parsec_ddesc_t *desc[1];
} cb_data_t;

static inline int parsec_recursivecall_callback(parsec_handle_t* parsec_handle, void* cb_data)
{
    int i, rc = 0;
    cb_data_t* data = (cb_data_t*)cb_data;

    rc = __parsec_complete_execution(data->eu, data->context);

    for(i=0; i<data->nbdesc; i++){
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)(data->desc[i]) );
        free( data->desc[i] );
    }

    data->destruct( parsec_handle );
    free(data);

    return rc;
}

static inline int parsec_recursivecall( parsec_execution_unit_t    *eu,
                                       parsec_execution_context_t *context,
                                       parsec_handle_t            *handle,
                                       void (*handle_destroy)(parsec_handle_t *),
                                       int nbdesc,
                                       ... )
{
    cb_data_t *cbdata = NULL;
    int i;
    va_list ap;

    /* Set mask to be used only on CPU */
    parsec_devices_handle_restrict( handle, PARSEC_DEV_CPU );

    /* Callback */
    cbdata = (cb_data_t *) malloc( sizeof(cb_data_t) + (nbdesc-1)*sizeof(parsec_ddesc_t*));
    cbdata->eu       = eu;
    cbdata->context  = context;
    cbdata->destruct = handle_destroy;
    cbdata->nbdesc   = nbdesc;

    /* Get descriptors */
    va_start(ap, nbdesc);
    for(i=0; i<nbdesc; i++){
        cbdata->desc[i] = va_arg(ap, parsec_ddesc_t *);
    }
    va_end(ap);

    parsec_set_complete_callback( handle, parsec_recursivecall_callback,
                                 (void *)cbdata );

    parsec_enqueue( eu->virtual_process->parsec_context,
                   handle );

    return -1;
}

#endif /* _PARSEC_RECURSIVE_H_ */
