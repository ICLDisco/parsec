/*
 * Copyright (c) 2018-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_TLS_H
#define PARSEC_TLS_H

/* This file does not need BEGIN_C_DECL, as it only holds macros,
 * and should only be compiled within the PaRSEC sources */

#if !defined(BUILDING_PARSEC)
#error "This file is compiler / architecture dependent and should not be included from a source file that does not belong to PaRSEC core code"
#endif

#include "parsec/parsec_config.h"

#if defined(PARSEC_HAVE_THREAD_LOCAL)

/**
 * @ingroup parsec_internal_runtime
 * @{
 */

/**
 * Declare a TLS variable / key.
 *
 * @details
 *   To preserve compatibility with pthread specifics, we only deal with
 *   pointers. That variable is static and can be accessed only in the
 *   c file that declared it.
 *
 *  @param[INOUT] var the handle to use for the thread specific pointer
 */
#define PARSEC_TLS_DECLARE(var)              static __thread void *var

/**
 * Create the key associated with a handle.
 *
 * @details
 *  This is used for compatibility only. _Thread_local and __thread
 *  approaches do not require this call. It must be issued, when using
 *  pthread, before any get/set specific call, by a single thread.
 *
 *  @param[INOUT] key the handle to use for the thread specific pointer
 */
#define PARSEC_TLS_KEY_CREATE(key)           do {} while(0)

/**
 * Returns the current value associated with the handle, for the calling thread.
 *
 * @details
 *  Undefined behavior in pthread mode, if this is called before KEY_CREATE.
 *  The undefined behavior is non-detectable: it will appear as working well,
 *  but the results will be inconsistent.
 *
 *  @return the current value of the thread specific handle associated with
 *    the calling thread. Warning: do not use this to set the variable, as
 *    this is not portable code. Use PARSEC_TLS_SET_SPECIFIC.
 */
#define PARSEC_TLS_GET_SPECIFIC(var)         (var)

/**
 * Sets the value associated with the handle for the calling thread.
 *
 * @details
 *  Undefined behavior in pthread mode, if this is called before KEY_CREATE.
 *  The undefined behavior is non-detectable: it will appear as working well,
 *  but the result will be inconsistent.
 *
 *  @param[INOUT] var the handle of the thread-specific variable
 *  @param[IN] value  a pointer, to set the value to.
 */
#define PARSEC_TLS_SET_SPECIFIC(var, value)  (var) = (value)

#elif defined(PARSEC_HAVE_PTHREAD_GETSPECIFIC)

#include <pthread.h>

#define PARSEC_TLS_DECLARE(key)              static pthread_key_t key
#define PARSEC_TLS_KEY_CREATE(key)           pthread_key_create(&key, NULL)
#define PARSEC_TLS_GET_SPECIFIC(key)         pthread_getspecific(key)
#define PARSEC_TLS_SET_SPECIFIC(key, value)  pthread_setspecific(key, value)

#else
#error "No Thread Local Storage API defined (CMake should have failed)"
#endif

/**
 * @}
 */

#endif /* PARSEC_TLS_H */
