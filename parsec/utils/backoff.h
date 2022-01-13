/**
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#define TIME_STEP 5410

static inline unsigned long parsec_exponential_backoff(parsec_execution_stream_t *es, uint64_t k)
{
    unsigned int n = (64 < k ? 64 : k);
    unsigned int r = (unsigned int) ((double)n * ((double)rand_r(&es->rand_seed)/(double)RAND_MAX));

    return r * TIME_STEP;
}

