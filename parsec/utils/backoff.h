/**
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"

#if defined(PARSEC_HAVE_RAND_R)
#include <stdlib.h>
#define PARSEC_BACKOFF_RAND_MAX RAND_MAX
#else
#define PARSEC_BACKOFF_RAND_MAX 32767U
#endif

#define TIME_STEP 5410

static inline unsigned int parsec_backoff_rand(parsec_execution_stream_t *es)
{
#if defined(PARSEC_HAVE_RAND_R)
    return (unsigned int)rand_r(&es->rand_seed);
#else
    es->rand_seed = 1103515245U * es->rand_seed + 12345U;
    return (es->rand_seed / 65536U) & PARSEC_BACKOFF_RAND_MAX;
#endif
}

static inline unsigned long parsec_exponential_backoff(parsec_execution_stream_t *es, uint64_t k)
{
    unsigned int n = (64 < k ? 64 : k);
    unsigned int r = (unsigned int)((double)n * ((double)parsec_backoff_rand(es) / (double)PARSEC_BACKOFF_RAND_MAX));

    return r * TIME_STEP;
}

#undef PARSEC_BACKOFF_RAND_MAX
