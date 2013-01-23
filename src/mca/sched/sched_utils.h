#ifndef SCHED_UTILS_H
#define SCHED_UTILS_H

#include "dague_config.h"
#include "execution_unit.h"

static int no_scheduler_is_active( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            if( vp->execution_units[t]->scheduler_object != NULL ) {
                return 0;
            }
        }
    }

    return 1;
}

#endif /* SCHED_UTILS_H */

