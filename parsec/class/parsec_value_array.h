/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#ifndef PARSEC_VALUE_ARRAY_H
#define PARSEC_VALUE_ARRAY_H

#include <string.h>

#include "parsec/class/parsec_object.h"
#include "parsec/constants.h"

/**
 * @defgroup parsec_internal_classes_valuearray Value Arrays
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Value Arrays for PaRSEC
 *
 *  @details Array of elements maintained by value.
 */

BEGIN_C_DECLS

/**
 * @brief Value Array Structure
 */
struct parsec_value_array_t
{
    parsec_object_t  super;            /**< A value array is a parsec object */
    unsigned char*  array_items;       /**< Items are stored here */
    size_t          array_item_sizeof; /**< Size of items */
    size_t          array_size;        /**< Occupied size of array_items */
    size_t          array_alloc_size;  /**< Allocated size of array_items */
};
/**
 * @brief Value Array Type
 */
typedef struct parsec_value_array_t parsec_value_array_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_value_array_t);

/**
 * @brief
 *  Initialize the array to hold items by value. This routine must 
 *  be called prior to using the array.
 *
 *  @param[inout]   array       The array to initialize.
 *  @param[in]      item_size   The sizeof each array element.
 *  @return          error code
 *
 * @remark Note that there is no corresponding "finalize" function -- use
 * OBJ_DESTRUCT (for stack arrays) or OBJ_RELEASE (for heap arrays) to
 * delete it.
 */
static inline int parsec_value_array_init(parsec_value_array_t *array, size_t item_sizeof)
{
    array->array_item_sizeof = item_sizeof;
    array->array_alloc_size = 1; 
    array->array_size = 0;
    array->array_items = (unsigned char*)realloc(array->array_items, item_sizeof * array->array_alloc_size);
    return (NULL != array->array_items) ? PARSEC_SUCCESS : PARSEC_ERR_OUT_OF_RESOURCE;
}


/**
 * @brief
 *  Reserve space in the array for new elements, but do not change the size.
 *
 *  @param[inout]   array   The input array.
 *  @param[in]      size    The anticipated size of the array.
 *  @return  error code.
 */
static inline int parsec_value_array_reserve(parsec_value_array_t* array, size_t size)
{
     if(size > array->array_alloc_size) {
         array->array_items = (unsigned char*)realloc(array->array_items, array->array_item_sizeof * size);
         if(NULL == array->array_items) {
             array->array_size = 0;
             array->array_alloc_size = 0;
             return PARSEC_ERR_OUT_OF_RESOURCE;
         }
         array->array_alloc_size = size;
     }
     return PARSEC_SUCCESS;
}



/**
 * @brief
 *   Retreives the number of elements in the array.
 *
 *  @param[in]   array   The input array.
 *  @return  The number of elements currently in use.
 */
static inline size_t parsec_value_array_get_size(parsec_value_array_t* array)
{
    return array->array_size;
}


/**
 * @brief
 *  Set the number of elements in the array.
 *
 * @detail
 *  Note that resizing the array to a smaller size may not change
 *  the underlying memory allocated by the array. However, setting
 *  the size larger than the current allocation will grow it. In either
 *  case, if the routine is successful, parsec_value_array_get_size() will 
 *  return the new size.
 *
 *  @param[inout]  array   The input array.
 *  @param[in]  size    The new array size.
 *
 *  @return  error code.
 */
PARSEC_DECLSPEC int parsec_value_array_set_size(parsec_value_array_t* array, size_t size);

/** 
 * @brief
 *  Macro to retrieve an item from the array by value. 
 *
 * @detail
 *  Note that this does not change the size of the array - this macro is 
 *  strictly for performance - the user assumes the responsibility of 
 *  ensuring the array index is valid (0 <= item index < array size).
 *
 *  @param  array       The input array (IN).
 *  @param  item_type   The C datatype of the array item (IN).
 *  @param  item_index  The array index (IN).
 *
 *  @returns item       The requested item.
 */
#define PARSEC_VALUE_ARRAY_GET_ITEM(array, item_type, item_index) \
    ((item_type*)((array)->array_items))[item_index]

/**
 * @brief
 *  Retrieve an item from the array by reference.
 *
 * @detail
 *  Note that if the specified item_index is larger than the current
 *  array size, the array is grown to satisfy the request.
 *
 *  @param[in]  array          The input array.
 *  @param[in]  item_index     The array index.
 *
 *  @return ptr Pointer to the requested item.
 */
static inline void* parsec_value_array_get_item(parsec_value_array_t *array, size_t item_index)
{
    if(item_index >= array->array_size && parsec_value_array_set_size(array, item_index+1) != PARSEC_SUCCESS)
        return NULL;
    return array->array_items + (item_index * array->array_item_sizeof);
}

/** 
 * @brief
 *  Macro to set an array element by value.
 *
 * @detail
 *  Note that this does not change the size of the array - this macro is 
 *  strictly for performance - the user assumes the responsibility of 
 *  ensuring the array index is valid (0 <= item index < array size).
 *
 *  @param[in]  array       The input array.
 *  @param[in]  item_type   The C datatype of the array item.
 *  @param[in]  item_index  The array index.
 *  @param[in]  item_value  The new value for the specified index.
 *
 * @remark It is safe to free the item after returning from this call;
 * it is copied into the array by value.
 */
#define PARSEC_VALUE_ARRAY_SET_ITEM(array, item_type, item_index, item_value) \
    (((item_type*)((array)->array_items))[item_index] = item_value)

/** 
 * @brief
 *  Set an array element by value.
 *
 * @detail
 *  @param[inout]   array       The input array.
 *  @param[in]      item_index  The array index.
 *  @param[in]      item_value  A pointer to the item, which is copied into 
 *                              the array.
 *  @return  error code.
 *
 * @remark It is safe to free the item after returning from this call;
 * it is copied into the array by value.
 */
static inline int parsec_value_array_set_item(parsec_value_array_t *array, size_t item_index, const void* item)
{
    int rc;
    if(item_index >= array->array_size && 
       (rc = parsec_value_array_set_size(array, item_index+1)) != PARSEC_SUCCESS)
        return rc;
    memcpy(array->array_items + (item_index * array->array_item_sizeof), item, array->array_item_sizeof);
    return PARSEC_SUCCESS;
}


/**
 * @brief
 *  Appends an item to the end of the array. 
 *
 * @detail 
 *  This will grow the array if it is not large enough to
 *  contain the item.  It is safe to free the item after returning
 *  from this call; it is copied by value into the array.  
 *
 *  @param[inout] array The input array
 *  @param[in] item A pointer to the item to append, which is copied into the array.
 *  @return  error code 
 *
 */
static inline int parsec_value_array_append_item(parsec_value_array_t *array, const void *item)
{
    return parsec_value_array_set_item(array, array->array_size, item);
}


/**
 * @brief
 *  Remove a specific item from the array. 
 *
 * @detail
 * All elements following this index are shifted down.
 *  @param[inout]   array       The input array.
 *  @param[in]      item_index  The index to remove, which must be less than
 *                              the current array size.
 *  @return  error code.
 */
static inline int parsec_value_array_remove_item(parsec_value_array_t *array, size_t item_index)
{
    if (item_index >= array->array_size) {
        return PARSEC_ERR_BAD_PARAM;
    }
    memmove(array->array_items+(array->array_item_sizeof * item_index), 
            array->array_items+(array->array_item_sizeof * (item_index+1)),
            array->array_item_sizeof * (array->array_size - item_index - 1));
    array->array_size--;
    return PARSEC_SUCCESS;
}

/**
 * @brief
 *   Get the base pointer of the underlying array.
 * 
 * @detail
 * This function is helpful when you need to iterate through an
 * entire array; simply get the base value of the array and use native
 * C to iterate through it manually.  This can have better performance
 * than looping over PARSEC_VALUE_ARRAY_GET_ITEM() and
 * PARSEC_VALUE_ARRAY_SET_ITEM() because it will [potentially] reduce the
 * number of pointer dereferences.
 * @param[in] array The input array.
 * @param[in] array_type The C datatype of the array.
 *
 * @returns ptr Pointer to the actual array.
 */
#define PARSEC_VALUE_ARRAY_GET_BASE(array, item_type) \
  ((item_type*) ((array)->array_items))

END_C_DECLS

/** @} */

#endif

