/*
 * Copyright (c) 2020-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_INFO_H_HAS_BEEN_INCLUDED
#define PARSEC_INFO_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/list.h"
#include "parsec/class/parsec_rwlock.h"

/**
 * @remarks
 *
 *   info.h / info.c implement optional informations managed as
 *    a key-value store, where keys are strings, and values are
 *    arbitrary pointers, for the parsec_class_t / parsec_object_t.
 *   Current implementation targets efficiency when the number of
 *    info types is small: it uses a list in the class to store the
 *    keys, an array per object to store the values, and a rw_lock
 *    to redimension dynamic structures
 */

/**
 * @brief An info identifier
 * @details Such identifier is used to lookup or set info
 *    objects in private arrays.
 */
typedef int  parsec_info_id_t;

#define PARSEC_INFO_ID_UNDEFINED  ((parsec_info_id_t)-1)

/**
 * @brief An info entry
 * @details these entries belong the a parsec_info_t structure.
 *   they give a unique parsec_info_id_t to an info name, and
 *   record the destructor of the info objects
 */
typedef struct parsec_info_entry_s parsec_info_entry_t;

/**
 * @brief An object info array: this is the structure that holds the
 *   info objects, in an array indexed by the index associated with the
 *   keys
 */
typedef struct parsec_info_object_array_s parsec_info_object_array_t;

/**
 * @brief An info collection: this is the structure that holds the
 *   keys and associates an index to them
 */
typedef struct parsec_info_s parsec_info_t;

/******************* Structures implementation ***************************/

/**
 * @brief Prototype of a destructor function
 *
 * @details
 *   A function with this prototype will be called on each pointer
 *   set on any info, if a destructor is passed to info_register.
 */
typedef void (*parsec_info_destructor_t)(void *elt, void *cb_data);

/**
 * @brief Prototype of a constructor function
 *
 * @details
 *   A function with this prototype will be called each time a
 *   parsec_info_get is called on an info object array that
 *   does not hold an info for it
 *
 *   @param[inout] cb_data: opaque pointer provided by the user
 *   @param[in]        obj: Info-level dependent pointer.
 *                          This pointer is set by the runtime system
 *                          and corresponds to the object that holds
 *                          the info object array (typically the taskpool
 *                          for taskpool-level infos, the CUDA stream for
 *                          per-stream infos, etc...)
 */
typedef void *(*parsec_info_constructor_t)(void *obj, void *cb_data);

/**
 * @brief The descriptor of a single info
 */
struct parsec_info_entry_s {
    parsec_list_item_t       list_item;  /**< Descriptors are chained */
    parsec_info_t           *info;       /**< Backpointer to the info collection */
    char                    *name;       /**< Name of the info, as provided by the user */
    parsec_info_constructor_t constructor; /**< Constructor for the objects if they are gotten for the first time */
    void                   *cons_data;   /**< Parameter of the constructor */
    parsec_info_destructor_t destructor; /**< Destructor that will be called on each info item when this entry is
                                          *   unregistered */
    void                    *des_data;   /**< Opaque pointer passed to the destructor at time of call */
    void                    *cb_data;    /**< A value shared between all info_objects belonging to this info_entry */
    parsec_info_id_t         iid;        /**< Unique identifier of the info (within that info collection) */
};

/**
 * @brief An array of info objects
 * 
 * @details This structure holds the info objects
 *   in an array indexed by the iid
 */
struct parsec_info_object_array_s {
    parsec_list_item_t     list_item;    /**< IOAs are chained together, so the info_t can iterate over them to
                                          *   call the destructor */
    parsec_atomic_rwlock_t rw_lock;      /**< R/W lock to redimension info_entries and info_objects when needed */
    int                    known_infos;  /**< Size of the arrays below */
    parsec_info_t         *infos;        /**< Back pointer to the collection of infos */
    void                 **info_objects; /**< Info objects are stored in this array indexed by
                                          *   info_entry->iid */
    void                  *cons_obj;     /**< obj pointer for the constructor */
};

PARSEC_OBJ_CLASS_DECLARATION(parsec_info_object_array_t);

/**
 * @brief a collection of infos
 */
struct parsec_info_s {
    parsec_list_t        info_list;  /**< The list of infos */
    parsec_list_t        ioa_list;   /**< the list of IOAs */
    int                  max_id;     /**< The largest id known today in the list */
};

BEGIN_C_DECLS

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_info_t);

/**
 * @brief registers a new info key in an info collection
 *
 * @details
 *   @param[INOUT]      nfo: the info collection in which to register the new info
 *   @param[IN]        name: the name of the info
 *   @param[IN]  destructor: function called on each item at time of unregistration
 *   @param[IN]    des_data: opaque pointer passed to the destructor
 *   @param[IN] constructor: function called the first time an object is gotten from
 *                           an object info array
 *   @param[IN]   cons_data: the parameter to constructor
 *   @param[IN]     cb_data: a common pointer passed to all info entries with this
 *                           info ID.
 *   @return the new info identifier, or PARSEC_INFO_ID_UNDEFINED if the info name
 *           already exists.
 */
parsec_info_id_t parsec_info_register(parsec_info_t *nfo, const char *name,
                                      parsec_info_destructor_t destructor, void *des_data,
                                      parsec_info_constructor_t constructor, void *cons_data,
                                      void *cb_data);
    
/**
 * @brief unregisters an info key using its ID.
 *
 * @details
 *   @param[IN]       nfo: info collection holding the info
 *   @param[IN]   info_id: the identifier of the info to unregister
 *   @param[OUT] pcb_data: set the cb_data passed at info_register in pcb_data, if it is not NULL
 *   @return      info_id if it is unregistered, PARSEC_INFO_ID_UNDEFINED
 *                if the info could not be found
 */
parsec_info_id_t parsec_info_unregister(parsec_info_t *nfo, parsec_info_id_t info_id,
                                        void **pcb_data);

/**
 * @brief lookup an info identifier from its name
 *
 * @details
 *   @param[IN]      nfo: the info collection holding the info
 *   @param[IN]     name: the info (unique) name
 *   @param[OUT] pcb_data: set the cb_data passed at info_register in pcb_data, if it is not NULL
 *   @return the info id, or PARSEC_INFO_ID_UNDEFINED if there is no such info registered.
 */
parsec_info_id_t parsec_info_lookup(parsec_info_t *nfo, const char *name, void **pcb_data);

/**
 * @brief initializes an empty info object array to store the objects of this info collection
 *
 * @details
 *   @param[OUT]      oa: the info object array to initialize
 *   @param[IN]      nfo: the info collection that defines its keys
 *   @param[IN] cons_obj: pointer to the object holding the object array (passed as
 *                        first argument to constructor calls)
 *
 * @remark when constructing an object array with PARSE_OBJ_CONSTRUCT or PARSEC_OBJ_NEW,
 *   the parsec_info_t cannot be associated to the object array automatically as the constructor
 *   do not take parameters. It is thus needed to initialize the object_array with the info
 *   after it is constructed.
 */
void parsec_info_object_array_init(parsec_info_object_array_t *oa, parsec_info_t *nfo,
                                   void *cons_obj);

/**
 * @brief Set an info in an array of objects
 *
 * @details
 *   @param[IN]      oa: the info object array in which to set the info
 *   @param[IN]     iid: the id of the info key to set
 *   @param[IN]    info: the value of the info
 *   @return the old value of the info, or (void*)-1 if a parameter is invalid.
 *
 * @remark obj should belong to a descendent of the class that registered the
 *    info with this iid
 */
void *parsec_info_set(parsec_info_object_array_t *oa, parsec_info_id_t iid, void *info);

/**
 * @brief Test and Set an info in an info object array
 *
 * @details
 *   @param[IN]      oa: the info object array in which to set the info
 *   @param[IN]     iid: the id of the info key to set
 *   @param[IN]    info: the new value of the info
 *   @param[IN]     old: the old value of the info
 *   @return the value of the info object after the operation completed, 
 *           or (void*)-1 if a parameter is invalid.
 *
 * @remark obj should belong to a descendent of the class that registered the
 *    info with this iid
 */
void *parsec_info_test_and_set(parsec_info_object_array_t *oa, parsec_info_id_t iid, void *info, void *old);

/**
 * @brief Get an info from an info object array
 *
 * @details
 *   @param[IN]  oa: the info object array from which to get the info
 *   @param[IN] iid: the index of the info to get
 *   @return the value of the info, or PARSEC_INFO_ID_UNDEFINED if a parameter is invalid.
 *
 * @remark obj should belong to a descendent of the class that registered the
 *    info with this iid
 */
void *parsec_info_get(parsec_info_object_array_t *oa, parsec_info_id_t info_id);


END_C_DECLS
#endif /* PARSEC_INFO_H_HAS_BEEN_INCLUDED */
