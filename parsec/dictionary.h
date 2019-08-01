/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PROFILING_DICTIONARY_H
#define PROFILING_DICTIONARY_H

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"

#include "parsec/mca/pins/pins.h"
#include "parsec/include/parsec.h"
#include "parsec/parsec_description_structures.h"
#include "parsec/class/parsec_hash_table.h"

/**
 * @defgroup parsec_dictionnary_profiling Profiling: Dictionary
 * @ingroup parsec_public_profiling
 * @{
 *
 * @brief The PaRSEC dictionary profiling system allows to expose properties
 *    at runtime through a shared memory region.
 * @details
 *    The dictionnary system explores the submitted taskpools and the PINS modules,
 *    collects properties exposed by those entities.
 *    It then compare this list with the list of requested properties submitted
 *    by the user through mca parameters:
 *    profiling_properties=dgemm_NN_summa:GEMM:flops;dgemm_NN_summa:*:flops;dgemm_NN_summa:*:*;
 *
 * @remark Note about thread safety:
 *    Some functions are thread safe, others are not.
 *    The shared memory pages are exclusive to threads.
 *
 * # Concepts
 * A task's properties are evaluated during the completion of the task. The thread
 * will update the memory region before proceeding to the scheduler.
 *
 * One shared memory region is opened per node on the node. The memory region
 * starts with 3 integers specifying the number of pages of the memory region,
 * the running state of the memory region, and the dictionnary version.
 *
 * Dictionnary version increases when the producers adds or removes properties. It
 * also increases when the consumers change their list of requested properties.
 *
 * After reading the three integers, it is advised to re-open the shared memory region
 * with the correct size. Starting after those three integers is the XML header string.
 * <?xml version="1.0"?>
 * <root>
 * <application>
 *   <prank>0</prank>
 *   <psize>1</psize>
 *   <nb_vp>1</nb_vp>
 *   <nb_eu>2</nb_eu>
 *   <pages_per_nd>0</pages_per_nd>
 *   <pages_per_vp>0</pages_per_vp>
 *   <per_eu_properties>
 *     <dgemm_NN_summa>
 *       <GEMM>
 *         <flops><t>i</t><o>0</o></flops>
 *         <kflops><t>d</t><o>4</o></kflops>
 *       </GEMM>
 *     </dgemm_NN_summa>
 *   </per_eu_properties>
 *   <pages_per_eu>1</pages_per_eu>
 * </application>
 * </root>
 * It gives information about the MPI rank running on this node, the parsec_context
 * structure, and for each entity type, the number of pages allocated.
 * This should allow anyone to navigate the memory region. Following section gives
 * a detailed structure of what's happening in each page.
 * Properties are described by a name in square brackets, then a 1 character type.
 * The type is the python format. And then, the offset rom the beginning of the page.
 *
 * #Notes
 * There are no synchronization mechanism with the outside world
 * The application will update the memory region when it pleases, and as long as short
 * element are exported, there is no chance of reading an entry in the middle of its
 * update.
 *
 * A very basic tool to open and read a shared memory region will be provided in
 * parsec/tools/aggregator_visu/reader.
 * XML header previously shown is obtained with,
 * ./reader parsec_shmem header
 *
 * The content of the shared memory region can be read with,
 * ./reader parsec_shmem data
 * output should look like:
 * Node 0/1 {
 *   VP 0 {
 *   }
 *   EU 0 {
 *     dgemm_NN_summa:GEMM:flops    986000000
 *     dgemm_NN_summa:GEMM:kflops   986000.000000
 *   }
 *   EU 1 {
 *     dgemm_NN_summa:GEMM:flops    1014000000
 *     dgemm_NN_summa:GEMM:kflops   1014000.000000
 *   }
 * }
 */

#define PROPERTY_NO_STATE       0
#define PROPERTY_REQUESTED      1
#define PROPERTY_PROVIDED       2

#define DICT_PAGE_SIZE 4096

/**
 *  Profiling shmem is a container with a list of reference to the properties requested by the user, a buffer to write into, sizes, nb of prop, offset array, etc
 *  Page is
 **/
typedef struct parsec_profiling_shmem_s parsec_profiling_shmem_t;

/**
 *  Dictionary is the hashtable container for the properties defined by the taskpools or the platform and exposed to the user for dump
 *  Each entry, a property, expose a type and a function pointer.
 **/

/* Producer side, type and typed function pointer in union */
typedef struct parsec_property_function_s parsec_property_function_t;

/* Item from the properties hashtable */
typedef struct parsec_profiling_property_s parsec_profiling_property_t;
/* Item from the namespace hashtable, contains a hastable for properties */
typedef struct parsec_profiling_task_class_s parsec_profiling_task_class_t;
/* Item from the dictionary hashtable, contains a hastable for task_classes */
typedef struct parsec_profiling_namespace_s parsec_profiling_namespace_t;
/* at the intersection between consumers and producers */
typedef struct parsec_profiling_dictionary_s parsec_profiling_dictionary_t;
/* a node in a tree */
typedef struct parsec_profiling_node_s parsec_profiling_node_t;
/* the tree to map on top of namespace, task_class, and property to determineif a property is available and/or requested */
typedef struct parsec_profiling_tree_s parsec_profiling_tree_t;
/* special function pointer to request the creation of a bucket in each node */
typedef parsec_hash_table_item_t *(*create_bucket_fn)(char *);

/**
 * @brief System support a three-level naming mechanism namespace:task_class:property
 */
#define PROF_ROOT       0
#define PROF_NAMESPACE  1
#define PROF_TASK_CLASS 2
#define PROF_PROPERTY   3

/**
 * @brief Enum specifying which entity of the context is exporting a property
 */
typedef enum {
    PROFILING_UNINITIALIZED = -1,
    PROFILING_PER_NODE      =  0,
    PROFILING_PER_VP,
    PROFILING_PER_EU,
    /* MAX_IDX must be the highest rank of this enum for obvious reasons */
    PROFILING_MAX_IDX
} parsec_profiling_index_t;

/**
 * @brief Enum for property datatypes, must match whatever is in the ptg compiler
 */
typedef enum {
    PROPERTIES_INT32 = 0,
    PROPERTIES_INT64,
    PROPERTIES_FLOAT,
    PROPERTIES_DOUBLE,
    PROPERTIES_ULONGLONG,
    PROPERTIES_QUAD,
    PROPERTIES_UNKNOWN
} parsec_profiling_datatype_t;

/**
 * @brief Structure description of the Shared Memory Region
 * @details Gives sizes for each section, and offset for quick dispacement in it
 */
struct parsec_profiling_shmem_s {
    int                                 nb_xml_pages;  /**< Number of pages for the XML header */
    int                                 nb_node_pages; /**< Number of pages for the node */
    int                                 nb_vp_pages;   /**< Number of pages per vp */
    int                                 nb_eu_pages;   /**< Number of pages per execution unit */
    int                                 nb_vp;         /**< Number of vp reporting */
    int                                 nb_eu;         /**< Number of execution unit reporting */
    int                                 first_node;    /**< Offset in pages to the node exported data */
    int                                 first_vp;      /**< Offset in pages to the first vp exported data */
    int                                 first_eu;      /**< Offset in pages to the first exec unit exported data */
    size_t                              xml_sz;        /**< size of header string */
    char                               *header;        /**< String containing XML description of exported data */
    int                                 shm_fd;        /**< File descriptor associated with shared memory region */
    char                               *shmem_name;    /**< Opened shared memory region name */
    void                               *buffer;        /**< Pointer to the shared memory region */
    int                                 nb_pages;      /**< Total number of exported pages */
};

/**
 * @brief Structure describing required information for properties
 */
struct parsec_property_function_s {
    size_t                              offset;             /**< Offset  */
    parsec_profiling_datatype_t         type;
    union {
	expr_op_int32_inline_func_t     inline_func_int32;
	expr_op_int64_inline_func_t     inline_func_int64;
	expr_op_float_inline_func_t     inline_func_float;
	expr_op_double_inline_func_t    inline_func_double;
    } func;
};

/**
 * @brief Data instant value has a meaning
 * @details Actually used to modify the initial value
 */
#define NON_CUMULATIVE    0
/**
 * @brief Data values have to be accumulated
 * @details Final value is obtained by adding initial value with evaluated value
 */
#define     CUMULATIVE    1

/**
 * @brief Bucket for a property
 * Properties can be accumulative or not.
 * Properties type tell which unit is exporting them.
 * Properties have a 2-bit state, they can be REQUESTED and/or PROVIDED.
 * Properties have a func(-tion) that is evaluated every freq completion of tasks
 */
struct parsec_profiling_property_s {
    parsec_hash_table_item_t            super;    
    parsec_profiling_index_t            type;
    int                                 accumulate;
    PINS_FLAG                           event;
    int                                 freq;
    int                                 counter;
    int                                 state;
    parsec_property_function_t          func;
};

/**
 * @brief Bucket of a task_class, contains a hashtable full of properties
 */
struct parsec_profiling_task_class_s {
    parsec_hash_table_item_t            super;
    parsec_hash_table_t                 properties;
};

/**
 * @brief Bucket of a namespace, contains a hashtable full of buckets of task_class
 */
struct parsec_profiling_namespace_s {
    parsec_hash_table_item_t            super;    
    parsec_hash_table_t                 task_classes;
};

/**
 * @brief Dictionnary is a hashtable of namespaces
 * @details Dictionnary is versioned to signal to observers that the XML
 *        description changed and has to be reloaded.
 *        running depicts if the dictionnary is ready, in initialization,
 *        or disabled.
 */
struct parsec_profiling_dictionary_s {
    parsec_hash_table_t                 properties;
    int                                 version;
    int                                 running; /* 0: turned off, 1: initial state waiting for stuff, 2: running */
    parsec_context_t                   *context; /* I should consider a hashtable to store the contexts */
    parsec_profiling_tree_t            *tree;
    parsec_profiling_shmem_t           *shmem;
};

/**
 * @brief Node of the tree
 */
struct parsec_profiling_node_s {
    parsec_profiling_node_t            *parent;
    parsec_profiling_node_t            *left;
    parsec_profiling_node_t            *right;
    parsec_profiling_node_t            *next_sibling;
    int                                 depth;
    int                                 wildcard;
    char                               *str;
    parsec_hash_table_item_t           *bucket;
    parsec_hash_table_t                *ht;
    create_bucket_fn                    new_bucket;
};

/**
 * @brief Tree structure to compute the intersection between PROVIDED and REQUESTED properties and wildcards
 */
struct parsec_profiling_tree_s {
    parsec_profiling_node_t            *root;
    int                                 depth;
    parsec_profiling_node_t           **first_nodes;
};

#define MAX_LENGTH_NAME 256

parsec_profiling_dictionary_t *parsec_profiling_dictionary;

/**
 * @brief Initialize the dictionnary by exploring the context.
 * @details Exploring the PINS modules is not yet supported.
 */
int parsec_profiling_dictionary_init(parsec_context_t *master_context,
				     int num_modules,
				     parsec_pins_module_t **modules);

/**
 * @brief Free all the data structure and close the shared memory region.
 */
int parsec_profiling_dictionary_free(void);

/**
 * @brief Explore the dictionnary to find a namespace
 */
parsec_profiling_namespace_t *find_namespace(const char *ns);

/**
 * @brief Explore the namespace to find a task_class
 */
parsec_profiling_task_class_t  *find_task_class(parsec_profiling_namespace_t *ns, const char *tc);

/**
 * @brief Explore the task_class to find a property
 */
parsec_profiling_property_t  *find_property(parsec_profiling_task_class_t* tc, const char *pr);

/**
 * @brief Extract the provided properties from the taskpool
 *
 * @details Call this ONCE per per taskpool. Will increase the version of the dictionnary
 * @return 0    if success, never fails!
 *
 * @remark not thread safe
 */
int parsec_profiling_add_taskpool_properties(parsec_taskpool_t *h);

/**
 * @brief Manually forge a property and add it to the dictionnary
 *
 * @details Modules can use this function to add their own property to the dictionnary
 * @return 0    if success, never fails!
 *
 * @remark not thread safe
 */
int parsec_profiling_register_property(parsec_property_function_t *func,
				       const char *namespace,
				       const char *task_class,
				       const char *property,
				       parsec_profiling_index_t who,
				       int accumulative);

/**
 * @brief This function evaluates the requested properties into the shared memory region
 *
 * @remark thread safety ensured by construction of the shared memory region, and functions are stateless
 */
void parsec_profiling_evaluate_property(void *item, void *cb_data);

#endif /* PROFILING_DICTIONARY_H */

/**
 * @}
 */
