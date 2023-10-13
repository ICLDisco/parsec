/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef PARSEC_FUTURE_H_HAS_BEEN_INCLUDED
#define PARSEC_FUTURE_H_HAS_BEEN_INCLUDED

/*
 * @defgroup parsec_internal_classes_future Future Structure
 * @ingroup parsec_internal_classes
 * @{
 *
 * @brief Future structures that provide async and callback mechanism
 *
 * @details A simple future structure that provides similar functionality 
 *          to C++ Promise/Future structure. Set function, blocking get
 *          function and an async callback APIs are provided. Extension 
 *          to countable future is provided as well
 *
 * @remark Base future will allow one set to ready state, and contain the 
 *         value. The countable future extension allows a way to 
 *         aggregate a set of events for a trigger event
 *
 *
 */

#include <stdarg.h> 
#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_object.h"
#include "parsec/class/list.h"

BEGIN_C_DECLS

typedef struct  parsec_base_future_t             parsec_base_future_t;
typedef struct  parsec_future_fn_t               parsec_future_fn_t;

/* Callback routines types */
typedef void  (*parsec_future_cb_fulfill)       ();
typedef void  (*parsec_future_cb_nested)        ();
typedef int   (*parsec_future_cb_match)         ();
typedef void  (*parsec_future_cb_cleanup)       ();

typedef int   (*parsec_future_is_ready_t)       (parsec_base_future_t*);
typedef void* (*parsec_future_get_or_trigger_t) ();
typedef void  (*parsec_future_set_t)            (parsec_base_future_t*, void*);
typedef void* (*parsec_future_get_t)            (parsec_base_future_t*);
typedef void  (*parsec_future_init_t)           ();

#define PARSEC_DATA_FUTURE_STATUS_INIT      ((uint8_t)0x01) /* Future has been initialized. */
#define PARSEC_DATA_FUTURE_STATUS_TRIGGERED ((uint8_t)0x02) /* Future has been triggered. */
#define PARSEC_DATA_FUTURE_STATUS_COMPLETED ((uint8_t)0x04) /* Future has been completed. Note, it is possible to
                                                             * initialized a future and completed it, without triggering it.
                                                             */

/*
 * @brief future functions structure that includes the future API functions
 */
struct parsec_future_fn_t {
    parsec_future_is_ready_t       is_ready;        /**< check whether the future is ready */
    parsec_future_set_t            set;             /**< set value on a specific future */
    parsec_future_get_or_trigger_t get_or_trigger;  /**< trigger data generation on a specific future */
    parsec_future_get_t            get;             /**< get the value from a future, blocking */
    parsec_future_init_t           future_init;     /**< initialize the future with callback, count etc */
};

/*
 * @brief Base future structure
 */
struct parsec_base_future_t {
    parsec_list_item_t       item;          /**< a base future type is list item (also a PaRSEC object) */
    parsec_future_fn_t      *future_class;  /**< struct that holds all the common function pointers */
    volatile uint8_t         status;        /**< status of the future */
    void                    *tracked_data;  /**< a pointer to the data this future is tracking */
    parsec_future_cb_fulfill cb_fulfill;    /**< callback function */
    parsec_atomic_lock_t     future_lock;   /**< lockable for multithread access */
};

/*
 * @brief Countable future structure
 */
typedef struct parsec_countable_future_t {
    parsec_base_future_t  super;
    volatile int32_t      count; /**< extension of basic future with a count before ready, manipulate atomically */
} parsec_countable_future_t;

/*
 * @brief Data future structure
 */
typedef struct parsec_datacopy_future_t {
    parsec_base_future_t              super;
    void                             *cb_fulfill_data_in;  /**< a pointer to hold the data the callback function may need */
    parsec_future_cb_match            cb_match;            /**< callback function to check if target data matches */
    void                             *cb_match_data_in;    /**< callback arguments to pass to the callback to check
                                                                if the target data of this future matches */
    parsec_future_cb_cleanup          cb_cleanup;          /**< callback function for cleanup */
    parsec_list_t                    *nested_futures;      /**< a pointer to the list of nested futures this future tracks */
    int                               nested_enable;       /**< flag to indicate whether or not this future can have nested futures */
} parsec_datacopy_future_t;

/*
 * @brief Convenience macros for the future functions
 */
#define parsec_future_is_ready(future) \
    (((parsec_base_future_t*)(future))->future_class)->is_ready(((parsec_base_future_t*)(future)))

#define parsec_future_set(future, data) \
    (((parsec_base_future_t*)(future))->future_class)->set(((parsec_base_future_t*)(future)), data )

#define parsec_future_get_or_trigger(future, ...) \
    (((parsec_base_future_t*)(future))->future_class)->get_or_trigger(((parsec_base_future_t*)(future)), __VA_ARGS__ )

#define parsec_future_get(future) \
    (((parsec_base_future_t*)(future))->future_class)->get(((parsec_base_future_t*)(future)))

#define parsec_future_init(future, ...) \
    (((parsec_base_future_t*)(future))->future_class)->future_init(((parsec_base_future_t*)(future)), __VA_ARGS__ )

/* For creating objects of class parsec_future_t */
PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_base_future_t);

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_countable_future_t);

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_datacopy_future_t);

END_C_DECLS

/*
 * @}
 */

#endif /* PARSEC_FUTURE_H_HAS_BEEN_INCLUDED */
