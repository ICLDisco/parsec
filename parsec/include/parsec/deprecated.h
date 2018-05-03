/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEPRECATED_H_HAS_BEEN_INCLUDED
#define PARSEC_DEPRECATED_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

BEGIN_C_DECLS

/**
 * @defgroup parsec_deprecated Deprecated Runtime System
 * @ingroup parsec_deprecated
 *   PaRSEC Core routines that have been deprecated but remain temporarily
 *   provided by the project.
 *
 *  @{
 */
static inline int parsec_enqueue(parsec_context_t* context, parsec_taskpool_t* tp)
{
    return parsec_context_add_taskpool(context, tp);
}

/**  @} */

END_C_DECLS

#endif  /* PARSEC_DEPRECATED_H_HAS_BEEN_INCLUDED */
