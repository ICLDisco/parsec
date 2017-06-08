/*
 * Copyright (c) 2017     The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
/**
 * Define ATOMIC_STATIC_INLINE to empty in order to generate the fallback implementation for
 * the atomic operations. We need to have this version to be able to compile and link
 * with a different compiler than the one PaRSEC has been compiled with. This approach also
 * covers the case of including PaRSEC header files from C++ application.
 */
#define ATOMIC_STATIC_INLINE
#include "parsec/sys/atomic.h"
