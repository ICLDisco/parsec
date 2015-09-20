#ifndef _DAGUE_RECURSIVE_H_
#define _DAGUE_RECURSIVE_H_

#include "dague/dague_internal.h"
#include "dague/execution_unit.h"
#include "dague/scheduling.h"
#include "data_dist/matrix/matrix.h"

typedef struct cb_data_s {
    dague_execution_unit_t    *eu;
    dague_execution_context_t *context;
    void (*destruct)( dague_handle_t * );
    int nbdesc;
    dague_ddesc_t *desc[1];
} cb_data_t;

static inline int dague_recursivecall_callback(dague_handle_t* dague_handle, void* cb_data)
{
    int i, rc = 0;
    cb_data_t* data = (cb_data_t*)cb_data;

    rc = __dague_complete_execution(data->eu, data->context);

    for(i=0; i<data->nbdesc; i++){
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)(data->desc[i]) );
        free( data->desc[i] );
    }

    data->destruct( dague_handle );
    free(data);

    return rc;
}

static inline int dague_recursivecall( dague_execution_unit_t    *eu,
                                       dague_execution_context_t *context,
                                       dague_handle_t            *handle,
                                       void (*handle_destroy)(dague_handle_t *),
                                       int nbdesc,
                                       ... )
{
    cb_data_t *cbdata = NULL;
    int i;
    va_list ap;

    /* Set mask to be used only on CPU */
    handle->devices_mask = 1;

    /* Callback */
    cbdata = (cb_data_t *) malloc( sizeof(cb_data_t) + (nbdesc-1)*sizeof(dague_ddesc_t*));
    cbdata->eu       = eu;
    cbdata->context  = context;
    cbdata->destruct = handle_destroy;
    cbdata->nbdesc   = nbdesc;

    /* Get descriptors */
    va_start(ap, nbdesc);
    for(i=0; i<nbdesc; i++){
        cbdata->desc[i] = va_arg(ap, dague_ddesc_t *);
    }
    va_end(ap);

    dague_set_complete_callback( handle, dague_recursivecall_callback,
                                 (void *)cbdata );

    dague_enqueue( eu->virtual_process->dague_context,
                   handle );

    return -1;
}

#endif /* _DAGUE_RECURSIVE_H_ */
