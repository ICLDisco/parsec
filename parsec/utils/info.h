/*
 * Copyright (c) 2022      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * Generic routines for "info" handling.  This is a sinple key-value store that
 * is independent from the main MCA param key-value store. Helpful for creating
 * info strings without affecting MCA params.
 */

#ifndef PARSEC_INFO_H
#define PARSEC_INFO_H

#include "parsec/parsec_config.h"

#ifdef PARSEC_HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef PARSEC_HAVE_STDBOOL_H
#include <stdbool.h>
#endif

BEGIN_C_DECLS


typedef char* parsec_info_key_t;
typedef char* parsec_info_value_t;


struct parsec_info_value_s {
    char val[PARSEC_INFO_VALUE_LEN];
} parsec_info_value_t;


struct parsec_info_s {
    parsec_hashtable_t super;
} parsec_info_t;

PARSEC_OBJ_CLASS(parsec_info_t);

} parsec_info_t;

#define PARSEC_INFO_VALUE_NULL NULL


/**
 * Set the key-value pair in the target info object.
 *
 * @param info Pointer to the target info object.  Must not be
 * NULL.
 * @param key String for the key, max length PARSEC_INFO_KEY_LEN.
 * @param value String for the value to attache to the key, max 
 *        length PARSEC_INFO_VALUE_LEN.
 *
 * @retval PARSEC_SUCCESS On success
 * @retval PARSEC_ERROR On failure, values lower than PARSEC_ERROR may
 * elaborate on the particular error type.
 *
 * This function associates a string value to the string key in the
 * key-value store info. The value string is copied to the internal
 * storage of the key-value store. To reiterate, there is no need to
 * keep a copy of the value.
 *
 * If the key is already set to a value, the previous value is
 * overwritten for te key. In particular, if the value is
 * PARSEC_INFO_VALUE_NULL, the previous key-value pair is removed
 * from the info.
 */
parsec_info_set(parsec_info_t *info, const char *key, const char *value);

/**
 * Get the value associated with the key in the target info object.
 *
 * @param info Pointer to the target info object.  Must not be
 * NULL.
 * @param key String for the key, max length PARSEC_INFO_KEY_LEN.
 * @param value Pointer to the string of the value to retrieve;
 *
 * @retval PARSEC_SUCCESS On success
 * @retval PARSEC_ERROR On failure, values lower than PARSEC_ERROR may
 * elaborate on the particular error type.
 *
 * This function obtains the string value associated with the string
 * key in the key-value store info. A pointer to the internal value
 * string is returned. To reiterate, the value is shared with the
 * internal storage of the info object and must not be freed.
 *
 * If the key is not set to a value, the value is set to
 * PARSEC_INFO_VALUE_NULL.
 */
parsec_info_get(parsec_info_t *info, const char *key, char **value);

/**
 * Clear all key-values from the target info object.
 *
 * @param info Pointer to the target info object. Must not be
 * NULL.
 *
 * @retval PARSEC_SUCCESS On success
 * @retcal PARSEC_ERROR On failure, values lower than PARSEC_ERROR may
 * elaborate on the particular error type.
 *
 * This function clears all previously set key-values from the target
 * key-value store object; that is, it is equivalent to calling
 * parsec_info_set with the value PARSEC_INFO_VALUE_NULL, for any key
 * previously set on the info.
 */
parsec_info_clear(parserc_info_t *info);

/**
 * Obtain the info object associated with the givent context
 *
 * @param ctx The context to query
 * @param info Pointer to the info object associated with the context
 *
 * @retval PARSEC_SUCCESS On success
 * @retval PARSEC_ERROR On failure
 *
 * The returned info object may be empty (that is, it has no values
 * associated with any keys), but is always returns a reference to a
 * valid info object.
 *
 * Key-values added, modified, or removed to/from the info object will
 * be implicitly applied to the context, that is, behavior for calls
 * using that context may be altered accordingly to the set key-values.
 *
 * The returned info is a reference to the info object internally held
 * by the context. The Info object should never be freed, and it ceases
 * to be a valid reference when the concext has been freed.
 */
parsec_context_get_info(parsec_context_t *ctx, parsec_info_t **info);

/**
 * Obtain the info object associated with the givent taskpool
 *
 * @param tp The taskpool to query
 * @param info Pointer to the info object associated with the context
 *
 * @retval PARSEC_SUCCESS On success
 * @retval PARSEC_ERROR On failure
 *
 * The returned info object may be empty (that is, it has no values
 * associated with any keys), but is always returns a reference to a
 * valid info object.
 *
 * Key-values added, modified, or removed to/from the info object will
 * be implicitly applied to the taskpool, that is, behavior for calls
 * using that taskpool may be altered accordingly to the set key-values.
 *
 * The returned info is a reference to the info object internally held
 * by the taskpool. The Info object should never be freed, and it ceases
 * to be a valid reference when the taskpool has been freed.
 */
parsec_taskpool_get_info(parsec_taskpool_t *tp, parsec_info_t **info);

END_C_DECLS

#endif /* PARSEC_INFO_H */
