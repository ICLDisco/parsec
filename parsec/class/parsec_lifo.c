/*
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
/**
 * Define LIFO_STATIC_INLINE to empty in order to generate the fallback implementation for
 * the atomic operations. We need to have this version to be able to compile and link
 * with a different compiler than the one PaRSEC has been compiled with. This approach also
 * covers the case of including PaRSEC header files from C++ application.
 *
 * Note that we do not change ATOMIC_STATIC_INLINE because we do want to use the
 * fast atomics in this file, the LIFO interface itself will be produced as functions
 * instead of individual atomic operations.
 */
#define LIFO_STATIC_INLINE
#include "parsec/sys/atomic.h"
#include "parsec/class/lifo.h"

static inline void parsec_lifo_construct( parsec_lifo_t* lifo )
{
    /* Don't allow strange alignemnts */
    lifo->alignment = PARSEC_LIFO_ALIGNMENT_DEFAULT;
    lifo->lifo_head.data.item = NULL;
    lifo->lifo_head.data.guard.counter = 0;
    parsec_atomic_lock_init(&lifo->lifo_head.data.guard.lock);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_lifo_t, parsec_object_t,
                   parsec_lifo_construct, NULL);

