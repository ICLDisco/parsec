#ifndef PINS_PAPI_UTILS_H
#define PINS_PAPI_UTILS_H

#include "dague.h"
#include "execution_unit.h"

#define WHICH_CORE_IN_SOCKET 1
/* mostly, just don't choose 0; it interferes with PaRSEC's thread handling
 * at this point, any value for WHICH should be acceptable, due to refactoring of PINS finalization code */

/* CORES_PER_SOCKET is now in CMAKE config,
 * until dague-hwloc is updated to support dynamic determination */

void pins_papi_init(dague_context_t * master_context);
void pins_papi_thread_init(dague_execution_unit_t * exec_unit);

#endif
