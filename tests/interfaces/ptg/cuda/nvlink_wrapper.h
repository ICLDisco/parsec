#ifndef _NVLINK_WRAPPER_H
#define _NVLINK_WRAPPER_H

#include "parsec.h"

parsec_taskpool_t* testing_nvlink_New( parsec_context_t *ctx, int depth, int mb );

void testing_nvlink_Destruct( parsec_taskpool_t *tp );

#endif /* _NVLINK_WRAPPER_H */
