/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef PARSEC_OBJECT_H
#define PARSEC_OBJECT_H

#include "parsec/parsec_config.h"
#include <assert.h>
#include <stdlib.h>

/**
 * @defgroup parsec_internal_classes_object PaRSEC Objects
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Base Object of the entire PaRSEC hierarchy.  A simple
 *         C-language object-oriented system with single inheritance
 *         and ownership-based memory management using a
 *         retain/release model.
 *
 *  @details A class consists of a struct and singly-instantiated
 * class descriptor.  The first element of the struct must be the
 * parent class's struct.  The class descriptor must be given a
 * well-known name based upon the class struct name (if the struct is
 * sally_t, the class descriptor should be sally_t_class) and must be
 * statically initialized as discussed below.
 *
 * (a) To define a class
 *
 * In a interface (.h) file, define the class.  The first element
 * should always be the parent class, for example
 * @code
 *   struct sally_t
 *   {
 *     parent_t parent;
 *     void *first_member;
 *     ...
 *   };
 *   typedef struct sally_t sally_t;
 *
 *   PARSEC_OBJ_CLASS_DECLARATION(sally_t);
 * @endcode
 * All classes must have a parent which is also class.
 *
 * In an implementation (.c) file, instantiate a class descriptor for
 * the class like this:
 * @code
 *   PARSEC_OBJ_CLASS_INSTANCE(sally_t, parent_t, sally_construct, sally_destruct);
 * @endcode
 * This macro actually expands to
 * @code
 *   parsec_class_t sally_t_class = {
 *     "sally_t",
 *     PARSEC_OBJ_CLASS(parent_t),  // pointer to parent_t_class
 *     sally_construct,
 *     sally_destruct,
 *     0, 0, NULL, NULL,
 *     sizeof ("sally_t")
 *   };
 * @endcode
 * This variable should be declared in the interface (.h) file using
 * the PARSEC_OBJ_CLASS_DECLARATION macro as shown above.
 *
 * sally_construct, and sally_destruct are function pointers to the
 * constructor and destructor for the class and are best defined as
 * static functions in the implementation file.  NULL pointers maybe
 * supplied instead.
 *
 * Other class methods may be added to the struct.
 *
 * (b) Class instantiation: dynamic
 *
 * To create a instance of a class (an object) use PARSEC_OBJ_NEW:
 * @code
 *   sally_t *sally = PARSEC_OBJ_NEW(sally_t);
 * @endcode
 * which allocates memory of sizeof(sally_t) and runs the class's
 * constructors. An optional second argument names the allocator to
 * obtain the memory from (see (b') below); NULL selects malloc.
 *
 * Use PARSEC_OBJ_RETAIN, PARSEC_OBJ_RELEASE to do reference-count-based
 * memory management:
 * @code
 *   PARSEC_OBJ_RETAIN(sally);
 *   PARSEC_OBJ_RELEASE(sally);
 *   PARSEC_OBJ_RELEASE(sally);
 * @endcode
 * When the reference count reaches zero, the class's destructor, and
 * those of its parents, are run and the memory is freed.
 *
 * N.B. There is no explicit free/delete method for dynamic objects in
 * this model.
 *
 * (b') Class instantiation: custom allocators
 *
 * The storage of an object, and what happens to it once its reference
 * count drops to zero, is controlled by the parsec_obj_allocator_t the
 * object was created with. By default, PARSEC_OBJ_NEW uses malloc/free,
 * but an object can be obtained from any other source (e.g., a free-list,
 * see parsec_obj_freelist.h) by passing an allocator:
 * @code
 *   static void *my_alloc(parsec_obj_allocator_t *a, parsec_class_t *cls) {
 *       my_pool_t *pool = (my_pool_t*)a;  // allocator is the first member
 *       return my_pool_get(pool, cls->cls_sizeof);
 *   }
 *   static void my_release(parsec_obj_allocator_t *a, parsec_object_t *obj) {
 *       parsec_obj_destruct(obj);         // run the destructors
 *       my_pool_put((my_pool_t*)a, obj);  // and recycle the storage
 *   }
 *   my_pool->super = (parsec_obj_allocator_t){ my_alloc, my_release };
 *
 *   sally_t *sally = PARSEC_OBJ_NEW(sally_t, &my_pool->super);
 *   PARSEC_OBJ_RELEASE(sally);           // calls my_release
 * @endcode
 * The release callback has full control over the object: it is
 * expected to run the destructors (parsec_obj_destruct) and to dispose
 * of the storage. The allocator must outlive all objects created
 * from it.
 *
 * (c) Class instantiation: static
 *
 * For an object with static (or stack) allocation, it is only
 * necessary to initialize the memory, which is done using
 * PARSEC_OBJ_CONSTRUCT:
 * @code
 *   sally_t sally;
 *
 *   PARSEC_OBJ_CONSTRUCT(&sally, sally_t);
 * @endcode
 * The retain/release model is not necessary here, but before the
 * object goes out of scope, PARSEC_OBJ_DESTRUCT should be run to release
 * initialized resources:
 * @code
 *   PARSEC_OBJ_DESTRUCT(&sally);
 * @endcode
 *
 * If the storage was obtained elsewhere and should be disposed of
 * through the retain/release model, pass an allocator as third argument
 * to PARSEC_OBJ_CONSTRUCT; its release callback is invoked once the
 * reference count drops to zero (only its obj_release member is used):
 * @code
 *   PARSEC_OBJ_CONSTRUCT(obj, sally_t, &my_allocator);
 * @endcode
 * NULL selects the default, which only runs the destructors.
 * The release callback must be able to dispose of that storage: pooling
 * allocators such as free-lists and thread-mempools only accept objects
 * that were allocated from them.
 */

BEGIN_C_DECLS

#if defined(PARSEC_DEBUG_PARANOID)
/* Any kind of unique ID should do the job */
#define PARSEC_OBJ_MAGIC_ID ((0xdeafbeedULL << 32) + 0xdeafbeedULL)
#endif

/* typedefs ***********************************************************/

typedef struct parsec_object_t parsec_object_t;
typedef struct parsec_class_t parsec_class_t;
typedef void (*parsec_construct_t) (parsec_object_t *);
typedef void (*parsec_destruct_t) (parsec_object_t *);
typedef struct parsec_obj_allocator_s parsec_obj_allocator_t;
/**
 * Provide storage for an object of class cls (at least cls->cls_sizeof
 * bytes). Return NULL on failure.
 */
typedef void *(*parsec_obj_alloc_fn_t) (parsec_obj_allocator_t *allocator, parsec_class_t *cls);
/**
 * Called once the reference count of object drops to zero. Responsible
 * for running the destructors (parsec_obj_destruct) and disposing of
 * the storage.
 */
typedef void (*parsec_obj_release_fn_t) (parsec_obj_allocator_t *allocator, parsec_object_t *object);


/* types **************************************************************/

/**
 * Class descriptor.
 *
 * There should be a single instance of this descriptor for each class
 * definition.
 */
struct parsec_class_t {
    const char *cls_name;           /**< symbolic name for class */
    parsec_class_t *cls_parent;       /**< parent class descriptor */
    parsec_construct_t cls_construct; /**< class constructor */
    parsec_destruct_t cls_destruct;   /**< class destructor */
    int cls_initialized;            /**< is class initialized */
    int cls_depth;                  /**< depth of class hierarchy tree */
    parsec_construct_t *cls_construct_array;
                                    /**< array of parent class constructors */
    parsec_destruct_t *cls_destruct_array;
                                    /**< array of parent class destructors */
    size_t cls_sizeof;              /**< size of an object instance */
};

/**
 * Allocator descriptor.
 *
 * Determines where the storage of an object comes from and what
 * happens to the object once its reference count drops to zero. The
 * descriptor is typically embedded (as first member) into a larger
 * structure holding the state of the allocator (e.g., a free-list),
 * which the callbacks can then retrieve from the allocator pointer.
 */
struct parsec_obj_allocator_s {
    parsec_obj_alloc_fn_t   obj_alloc;    /**< allocate storage (may be NULL if only used for construct) */
    parsec_obj_release_fn_t obj_release;  /**< destruct and dispose of the object */
};

/**
 * Allocator used by PARSEC_OBJ_NEW: malloc the storage, destruct and free on release.
 */
PARSEC_DECLSPEC extern parsec_obj_allocator_t parsec_obj_malloc_allocator;
/**
 * Allocator used by PARSEC_OBJ_CONSTRUCT: only run the destructors on release.
 */
PARSEC_DECLSPEC extern parsec_obj_allocator_t parsec_obj_static_allocator;

/**
 * For static initializations of OBJects.
 *
 * @param BASE_CLASS   Name of the class to initialize
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_OBJ_STATIC_INIT(BASE_CLASS) { PARSEC_OBJ_MAGIC_ID, PARSEC_OBJ_CLASS(BASE_CLASS), 1, &parsec_obj_static_allocator, __FILE__, __LINE__ }
#else
#define PARSEC_OBJ_STATIC_INIT(BASE_CLASS) { PARSEC_OBJ_CLASS(BASE_CLASS), 1, &parsec_obj_static_allocator }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

/**
 * Base object.
 *
 * This is special and does not follow the pattern for other classes.
 */
struct parsec_object_t {
#if defined(PARSEC_DEBUG_PARANOID)
    /** Magic ID -- want this to be the very first item in the
        struct's memory */
    uint64_t obj_magic_id;
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
    parsec_class_t *obj_class;            /**< class descriptor */
    volatile int32_t obj_reference_count;   /**< reference count */
    parsec_obj_allocator_t *obj_allocator;  /**< allocator providing the release callback */
#if defined(PARSEC_DEBUG_PARANOID)
    const char* cls_init_file_name;        /**< In debug mode store the file where the object get constructed */
    int   cls_init_lineno;           /**< In debug mode store the line number where the object get constructed */
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
};

/* macros ************************************************************/

/**
 * Return a pointer to the class descriptor associated with a
 * class type.
 *
 * @param NAME          Name of class
 * @return              Pointer to class descriptor
 */
#define PARSEC_OBJ_CLASS(NAME)     (&(NAME ## _class))


/**
 * Static initializer for a class descriptor
 *
 * @param NAME          Name of class
 * @param PARENT        Name of parent class
 * @param CONSTRUCTOR   Pointer to constructor
 * @param DESTRUCTOR    Pointer to destructor
 *
 * Put this in NAME.c
 */
#define PARSEC_OBJ_CLASS_INSTANCE(NAME, PARENT, CONSTRUCTOR, DESTRUCTOR)       \
    parsec_class_t NAME ## _class = {                                     \
        # NAME,                                                         \
        PARSEC_OBJ_CLASS(PARENT),                                              \
        (parsec_construct_t) CONSTRUCTOR,                                 \
        (parsec_destruct_t) DESTRUCTOR,                                   \
        0, 0, NULL, NULL,                                               \
        sizeof(NAME)                                                    \
    }


/**
 * Declaration for class descriptor
 *
 * @param NAME          Name of class
 *
 * Put this in NAME.h
 */
#define PARSEC_OBJ_CLASS_DECLARATION(NAME)             \
    extern parsec_class_t NAME ## _class


/* Pick a macro by the number of arguments. PARSEC_OBJ_EXPAND forces a
 * rescan, for preprocessors that would pass __VA_ARGS__ as a single
 * argument. The trailing ~ keeps the variadic part non-empty. */
#define PARSEC_OBJ_EXPAND(x) x
#define PARSEC_OBJ_SELECT_3RD(_1, _2, NAME, ...) NAME
#define PARSEC_OBJ_SELECT_4TH(_1, _2, _3, NAME, ...) NAME

static inline parsec_object_t *parsec_obj_new(parsec_class_t * cls);
static inline parsec_object_t *parsec_obj_new_wallocator(parsec_class_t * cls,
                                                         parsec_obj_allocator_t *allocator);
#if defined(PARSEC_DEBUG_PARANOID)
static inline parsec_object_t *parsec_obj_new_debug(parsec_object_t* object, const char* file, int line)
{
    if (NULL != object) {
        object->cls_init_file_name = file;
        object->cls_init_lineno = line;
    }
    return object;
}
#define PARSEC_OBJ_NEW_RECORD(object) parsec_obj_new_debug((object), __FILE__, __LINE__)
#else
#define PARSEC_OBJ_NEW_RECORD(object) (object)
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

/**
 * Create an object: obtain storage and run the class constructor.
 *
 * PARSEC_OBJ_NEW(type) allocates the storage with malloc, and frees it
 * once the reference count drops to zero. PARSEC_OBJ_NEW(type, allocator)
 * obtains the storage from allocator->obj_alloc and calls
 * allocator->obj_release once the reference count drops to zero; a NULL
 * allocator is the same as PARSEC_OBJ_NEW(type).
 *
 * @param type          Type (class) of the object
 * @param allocator     Optional pointer to a parsec_obj_allocator_t
 * @return              Pointer to the object
 */
#define PARSEC_OBJ_NEW(...)                                             \
    PARSEC_OBJ_EXPAND(PARSEC_OBJ_SELECT_3RD(__VA_ARGS__, PARSEC_OBJ_NEW_ARGS2, \
                                            PARSEC_OBJ_NEW_ARGS1, ~)(__VA_ARGS__))
/* implementation of PARSEC_OBJ_NEW, do not use directly */
#define PARSEC_OBJ_NEW_ARGS1(type)                                      \
    ((type *)PARSEC_OBJ_NEW_RECORD(parsec_obj_new(PARSEC_OBJ_CLASS(type))))
#define PARSEC_OBJ_NEW_ARGS2(type, allocator)                           \
    ((type *)PARSEC_OBJ_NEW_RECORD(parsec_obj_new_wallocator(PARSEC_OBJ_CLASS(type), (allocator))))

/**
 * Retain an object (by incrementing its reference count)
 *
 * @param object        Pointer to the object
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_OBJ_RETAIN(object)                                              \
    do {                                                                \
        assert(NULL != ((parsec_object_t *) (object))->obj_class);        \
        assert(PARSEC_OBJ_MAGIC_ID == ((parsec_object_t *) (object))->obj_magic_id); \
        parsec_obj_update((parsec_object_t *) (object), 1);                 \
        assert(((parsec_object_t *) (object))->obj_reference_count >= 0); \
    } while (0)
#else
#define PARSEC_OBJ_RETAIN(object)  parsec_obj_update((parsec_object_t *) (object), 1);
#endif

/**
 * Helper macro for the debug mode to store the locations where the status of
 * an object change.
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( OBJECT, FILE, LINENO )    \
    do {                                                        \
        ((parsec_object_t*)(OBJECT))->cls_init_file_name = FILE;  \
        ((parsec_object_t*)(OBJECT))->cls_init_lineno = LINENO;   \
    } while(0)
#define PARSEC_OBJ_SET_MAGIC_ID( OBJECT, VALUE )                       \
    do {                                                        \
        ((parsec_object_t*)(OBJECT))->obj_magic_id = (VALUE);     \
    } while(0)
#else
#define PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( OBJECT, FILE, LINENO )
#define PARSEC_OBJ_SET_MAGIC_ID( OBJECT, VALUE )
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

/**
 * Run the destructors of an object and invalidate it. To be used from
 * the release callback of a custom allocator.
 */
PARSEC_DECLSPEC void parsec_obj_destruct(parsec_object_t * object);

/**
 * Invoke the release callback of the object's allocator.
 * Do not use directly: use PARSEC_OBJ_RELEASE() instead.
 */
static inline void parsec_obj_release_internal(parsec_object_t *object)
{
    object->obj_allocator->obj_release(object->obj_allocator, object);
}

/**
 * Release an object (by decrementing its reference count).  If the
 * reference count reaches zero, destruct (finalize) the object and
 * free its storage.
 *
 * Note: If the object is freed, then the value of the pointer is set
 * to NULL.
 *
 * @param object        Pointer to the object
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_OBJ_RELEASE(object)                                     \
    do {                                                        \
        assert(NULL != ((parsec_object_t *) (object))->obj_class);        \
        assert(PARSEC_OBJ_MAGIC_ID == ((parsec_object_t *) (object))->obj_magic_id); \
        if (0 == parsec_obj_update((parsec_object_t *) (object), -1)) {     \
            PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( object, __FILE__, __LINE__ ); \
            parsec_obj_release_internal((parsec_object_t*)object);  \
            object = NULL;                                      \
        }                                                       \
    } while (0)
#else
#define PARSEC_OBJ_RELEASE(object)                                     \
    do {                                                        \
        if (0 == parsec_obj_update((parsec_object_t *) (object), -1)) {     \
            parsec_obj_release_internal((parsec_object_t*)object); \
            object = NULL;                                      \
        }                                                       \
    } while (0)
#endif

/**
 * Construct (initialize) objects that are not dynamically allocated.
 *
 * PARSEC_OBJ_CONSTRUCT(object, type) only runs the destructors once the
 * reference count drops to zero. PARSEC_OBJ_CONSTRUCT(object, type,
 * allocator) disposes of the object through allocator->obj_release
 * instead, for storage obtained by other means; a NULL allocator is the
 * same as PARSEC_OBJ_CONSTRUCT(object, type). The allocator must be able
 * to dispose of that storage: pooling allocators (free-lists,
 * thread-mempools) only accept objects allocated from them.
 *
 * @param object        Pointer to the object
 * @param type          The object type
 * @param allocator     Optional pointer to a parsec_obj_allocator_t
 */
#define PARSEC_OBJ_CONSTRUCT(...)                                       \
    PARSEC_OBJ_EXPAND(PARSEC_OBJ_SELECT_4TH(__VA_ARGS__, PARSEC_OBJ_CONSTRUCT_ARGS3, \
                                            PARSEC_OBJ_CONSTRUCT_ARGS2, ~)(__VA_ARGS__))
/* implementation of PARSEC_OBJ_CONSTRUCT, do not use directly */
#define PARSEC_OBJ_CONSTRUCT_ARGS2(object, type)                        \
    PARSEC_OBJ_CONSTRUCT_ARGS3(object, type, NULL)
#define PARSEC_OBJ_CONSTRUCT_ARGS3(object, type, allocator)             \
do {                                                                    \
    parsec_obj_construct((parsec_object_t *) (object), PARSEC_OBJ_CLASS(type), (allocator)); \
    PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( object, __FILE__, __LINE__ );  \
} while (0)


/**
 * Destruct (finalize) an object that is not dynamically allocated.
 *
 * @param object        Pointer to the object
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_OBJ_DESTRUCT(object)                                    \
do {                                                            \
    assert(PARSEC_OBJ_MAGIC_ID == ((parsec_object_t *) (object))->obj_magic_id); \
    parsec_obj_run_destructors((parsec_object_t *) (object));     \
    PARSEC_OBJ_SET_MAGIC_ID((object), 0);                              \
    PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( object, __FILE__, __LINE__ ); \
} while (0)
#else
#define PARSEC_OBJ_DESTRUCT(object)                                    \
do {                                                            \
    parsec_obj_run_destructors((parsec_object_t *) (object));     \
    PARSEC_OBJ_REMEMBER_FILE_AND_LINENO( object, __FILE__, __LINE__ ); \
} while (0)
#endif

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_object_t);

/* declarations *******************************************************/

/**
 * Lazy initialization of class descriptor.
 *
 * Specifically cache arrays of function pointers for the constructor
 * and destructor hierarchies for this class.
 *
 * @param class    Pointer to class descriptor
 */
PARSEC_DECLSPEC void parsec_class_initialize(parsec_class_t *cls);

/**
 * Shut down the class system and release all memory
 *
 * This function should be invoked as the ABSOLUTE LAST function to
 * use the class subsystem.  It frees all associated memory with ALL
 * classes, rendering all of them inoperable.  It is here so that
 * tools like valgrind and purify don't report still-reachable memory
 * upon process termination.
 */
PARSEC_DECLSPEC void parsec_class_finalize(void);

/**
 * Run the hierarchy of class constructors for this object, in a
 * parent-first order.
 *
 * Do not use this function directly: use PARSEC_OBJ_CONSTRUCT() instead.
 *
 * WARNING: This implementation relies on a hardwired maximum depth of
 * the inheritance tree!!!
 *
 * Hardwired for fairly shallow inheritance trees
 * @param object          Pointer to the object.
 */
static inline void parsec_obj_run_constructors(parsec_object_t * object)
{
    parsec_construct_t* cls_construct;

    assert(NULL != object->obj_class);

    cls_construct = object->obj_class->cls_construct_array;
    while( NULL != *cls_construct ) {
        (*cls_construct)(object);
        cls_construct++;
    }
}


/**
 * Run the hierarchy of class destructors for this object, in a
 * parent-last order.
 *
 * Do not use this function directly: use PARSEC_OBJ_DESTRUCT() instead.
 *
 * @param object          Pointer to the object.
 */
static inline void parsec_obj_run_destructors(parsec_object_t * object)
{
    parsec_destruct_t* cls_destruct;

    assert(NULL != object->obj_class);

    cls_destruct = object->obj_class->cls_destruct_array;
    while( NULL != *cls_destruct ) {
        (*cls_destruct)(object);
        cls_destruct++;
    }
}


/**
 * Initialize the header of an object and run the class constructors.
 *
 * Do not use this function directly: use PARSEC_OBJ_NEW() or
 * PARSEC_OBJ_CONSTRUCT() instead.
 *
 * @param object        Pointer to the storage of the object (may be NULL)
 * @param cls           Pointer to the class descriptor of this object
 * @param allocator     Allocator providing the release callback (not NULL)
 * @return              Pointer to the object
 */
static inline parsec_object_t *parsec_obj_init(parsec_object_t *object,
                                               parsec_class_t *cls,
                                               parsec_obj_allocator_t *allocator)
{
    if (0 == cls->cls_initialized) {
        parsec_class_initialize(cls);
    }
    if (NULL != object) {
        PARSEC_OBJ_SET_MAGIC_ID(object, PARSEC_OBJ_MAGIC_ID);
        object->obj_class = cls;
        object->obj_reference_count = 1;
        object->obj_allocator = allocator;
        parsec_obj_run_constructors(object);
    }
    return object;
}

/**
 * Construct an object in place: NULL selects the default allocator,
 * which only runs the destructors on release.
 *
 * Do not use this function directly: use PARSEC_OBJ_CONSTRUCT() instead.
 */
static inline void parsec_obj_construct(parsec_object_t *object,
                                        parsec_class_t *cls,
                                        parsec_obj_allocator_t *allocator)
{
    parsec_obj_init(object, cls, (NULL != allocator) ? allocator : &parsec_obj_static_allocator);
}

/**
 * Create new object: dynamically allocate storage and run the class
 * constructor.
 *
 * Do not use this function directly: use PARSEC_OBJ_NEW() instead.
 *
 * @param cls           Pointer to the class descriptor of this object
 * @return              Pointer to the object
 */
static inline parsec_object_t *parsec_obj_new(parsec_class_t * cls)
{
    assert(cls->cls_sizeof >= sizeof(parsec_object_t));
    /* call malloc directly to avoid the indirect call on the common path */
    return parsec_obj_init((parsec_object_t *) malloc(cls->cls_sizeof),
                           cls, &parsec_obj_malloc_allocator);
}

/**
 * Create new object: obtain storage from allocator and run the class
 * constructor. NULL selects malloc, as parsec_obj_new.
 *
 * Do not use this function directly: use PARSEC_OBJ_NEW() instead.
 *
 * @param cls           Pointer to the class descriptor of this object
 * @param allocator     Allocator providing the storage and release callback
 * @return              Pointer to the object
 */
static inline parsec_object_t *parsec_obj_new_wallocator(parsec_class_t * cls,
                                                         parsec_obj_allocator_t *allocator)
{
    if (NULL == allocator) {
        return parsec_obj_new(cls);
    }
    assert(cls->cls_sizeof >= sizeof(parsec_object_t));
    assert(NULL != allocator->obj_alloc && NULL != allocator->obj_release);
    return parsec_obj_init((parsec_object_t *) allocator->obj_alloc(allocator, cls),
                           cls, allocator);
}

#if defined(BUILDING_PARSEC)
#include "parsec/sys/atomic.h"

/**
 * Atomically update the object's reference count by some increment.
 *
 * This function should not be used directly: it is called via the
 * macros PARSEC_OBJ_RETAIN and PARSEC_OBJ_RELEASE
 *
 * @param object        Pointer to the object
 * @param inc           Increment by which to update reference count
 * @return              New value of the reference count
 */
static inline int parsec_obj_update(parsec_object_t *object, int inc) __parsec_attribute_always_inline__;
static inline int parsec_obj_update(parsec_object_t *object, int inc)
{
    return parsec_atomic_fetch_add_int32(&(object->obj_reference_count), inc ) + inc;
}
#else
/* Read the comment in parsec_object.c regarding the use of this function */
PARSEC_DECLSPEC int parsec_obj_update_not_inline(parsec_object_t *object, int inc);
#define parsec_obj_update parsec_obj_update_not_inline
#endif  /* defined(BUILDING_PARSEC) */
END_C_DECLS

/**
 * @}
 */

#endif
