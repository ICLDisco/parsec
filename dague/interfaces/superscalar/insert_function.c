#include "dague_config.h"
#include <stdarg.h>
#include "dague.h"
#include "data_distribution.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include "dplasma/lib/memory_pool.h"
#include "data.h"
#include "debug.h"
#include "scheduling.h"
#include "dague/mca/pins/pins.h"
#include "remote_dep.h"
#include "datarepo.h"
#include "dague_prof_grapher.h"
#include "dague/dague_prof_grapher.c"
#include "mempool.h"
#include "dague/devices/device.h"
#include "dague/constants.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

#define MAX_TASK_CLASS 10
#define TASK_HASH_TABLE_SIZE (100*1000)
#define TILE_HASH_TABLE_SIZE (100*10)
#define FUNCTION_HASH_TABLE_SIZE (10)

/**
 * All the static functions should be declared before being defined.
 */
static int
test_hook_of_dtd_task(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task);
static int
dtd_startup_tasks(dague_context_t * context,
                  __dague_dtd_internal_handle_t * __dague_handle,
                  dague_execution_context_t ** pready_list);
static int
dtd_is_ready(const dtd_task_t *dest, int dest_flow_index);

static void
iterate_successors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg);
static void
_internal_insert_task(dague_dtd_handle_t *__dague_handle,
                      task_func* fpointer,
                      task_param_t *param_list_head,
                      char* name, int no_of_param,
                      int size_of_value_param);
static int
release_deps_of_dtd(struct dague_execution_unit_s *,
                    dague_execution_context_t *,
                    uint32_t, dague_remote_deps_t *);

static dague_hook_return_t
complete_hook_of_dtd(struct dague_execution_unit_s *,
                     dague_execution_context_t *);


//#define GRAPH_COLOR
//#define PRINT_F_STRUCTURE

void
dague_dtd_unpack_args(dague_execution_context_t *this_task, ...)
{
    dtd_task_t *current_task = (dtd_task_t *)this_task;
    task_param_t *current_param = current_task->param_list;
    int next_arg;
    void **tmp;
    dague_data_copy_t *tmp_data;
    va_list arguments;
    va_start(arguments, this_task);
    next_arg = va_arg(arguments, int);

    while (current_param != NULL){
         tmp = va_arg(arguments, void**);
        if(UNPACK_VALUE == next_arg){
            /*tmp = current_param->pointer_to_tile; */
             memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }else if (UNPACK_DATA == next_arg){
            tmp_data = ((dtd_tile_t*)(current_param->pointer_to_tile))->data_copy;
            memcpy(tmp, &tmp_data, sizeof(dague_data_copy_t *));
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}


/* To create object of class dtd_task_t that inherits dague_execution_context_t class */
static dague_mempool_t *context_mempool;
OBJ_CLASS_INSTANCE(dtd_task_t, dague_execution_context_t,
                   NULL, NULL);

#if defined(DAGUE_PROF_TRACE)
static int *zpotrf_dtd_profiling_array;
int *task_class_count;
uint64_t function_pointer_tracker[20];
#endif


static inline char*
color_hash(char *name)
{
    int c, i, r1, r2, g1, g2, b1, b2;
    char *color=(char *)calloc(6,sizeof(char));

    r1 = 0xA3;
    r2 = 7;
    g1 = 0x2C;
    g2 = 135;
    b1 = 0x97;
    b2 = 49;

    for(i=0; i<strlen(name); i++){
        c = name[i];
        c &= 0xFF; // Make sure we don't get a Unicode or something.

        r1 ^= c;
        r2 *= c;
        r2 %= 1<<24;

        g1 ^= c;
        g2 *= c;
        g2 %= 1<<24;

        b1 ^= c;
        b2 *= c;
        b2 %= 1<<24;
    }
    r1 ^= (r2)&0xFF;
    r1 ^= (r2>>8)&0xFF;
    r1 ^= (r2>>16)&0xFF;

    g1 ^= (g2)&0xFF;
    g1 ^= (g2>>8)&0xFF;
    g1 ^= (g2>>16)&0xFF;

    b1 ^= (b2)&0xFF;
    b1 ^= (b2>>8)&0xFF;
    b1 ^= (b2>>16)&0xFF;

#if defined(DAGUE_PROF_GRAPHER_ddd)
    fprintf(grapher_file,"#%02x%02x%02x",r1,g1,b1);
#endif
    snprintf(color,6,"%02x%02x%02x",r1,g1,b1);
    return(color);
}

static inline char*
fill_color(char *name)
{
    char *str;
    str = (char *)calloc(12,sizeof(char));
    snprintf(str,12,"fill:%s",color_hash(name));
    return str;
}

#if defined(DAGUE_PROF_TRACE)
void
profiling_trace(dague_dtd_handle_t *__dague_handle,
                int task_class_counter,  task_func *fpointer,
                char* name, int flow_index)
{
    int l_counter, task_class_flag=0;

    __dague_handle->super.profiling_array = zpotrf_dtd_profiling_array;

    if(task_class_counter < *task_class_count) {
        for(l_counter=0; l_counter<*task_class_count;l_counter++) {
            if(function_pointer_tracker[l_counter] == (uint64_t)fpointer) {
                task_class_flag = 1;
            }
        }
        if(!task_class_flag) {
            function_pointer_tracker[task_class_counter] = (uint64_t)fpointer;
            dague_profiling_add_dictionary_keyword(name, fill_color(name),
                                               sizeof(dague_profile_ddesc_info_t) + flow_index * sizeof(assignment_t),
                                               dague_profile_ddesc_key_to_string,
                                               (int *) &__dague_handle->super.profiling_array[0 +
                                                                                                    2 *
                                                                                                    task_class_counter
                                                                                                    /* start key */
                                                                                                    ],
                                               (int *) &__dague_handle->super.profiling_array[1 +
                                                                                                    2 *
                                                                                                    task_class_counter
                                                                                                    /*  end key */
                                                                                                    ]);


            task_class_counter++;
        }
    }
}
#endif /* defined(DAGUE_PROF_TRACE) */

/* Hash table creation function */
void *
generic_create_hash_table(int size_of_table, int size_of_each_bucket)
{
    void *new_table;
    new_table = malloc(sizeof(hash_table));
    ((hash_table *)new_table)->buckets = calloc(size_of_table, size_of_each_bucket);
    ((hash_table *)new_table)->size = size_of_table;
    return new_table;
}

/* Hashing function */
uint32_t
hash_key (uintptr_t key, int size)
{
    uint32_t hash_val = key % size;
    return hash_val;
}

/* To find function structure(task class type) */
dague_function_t *
find_function(hash_table *hash_table,
              task_func *key, int h_size)
{
    uint32_t hash_val = hash_key((uintptr_t)key, h_size);
    bucket_element_f_t *current;

    current = hash_table->buckets[hash_val];

    if(current != NULL) {    /* Finding the elememnt, the pointer to the tile in the bucket of Hash table is returned if found, else NULL is returned */
        while(current != NULL) {
            if(current->key == key) {
                break;
            }
            current = current->next;
        }
        if(NULL != current) {
            return current->dtd_function;
        }else {
            return NULL;
        }
    }else {
        return NULL;
    }
}

void
function_insert_h_t(hash_table *hash_table,
                    task_func *key, dague_function_t *dtd_function,
                    int h_size)
{
    uint32_t hash_val = hash_key((uintptr_t)key, h_size);
    bucket_element_f_t *new_list, *current_table_element;

    /** Assigning values to new element **/
    new_list =(bucket_element_f_t *) malloc(sizeof(bucket_element_f_t));
    new_list->next = NULL;
    new_list->key = key;
    new_list->dtd_function = dtd_function;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        while(current_table_element->next != NULL) {    /* Finding the last element of the list */
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/** Tile Hash Function **/
dtd_tile_t *
find_tile(hash_table *hash_table,
          uint32_t key, int h_size,
          dague_ddesc_t* belongs_to)
{
    uint32_t hash_val = hash_key(key, h_size);
    bucket_element_tile_t *current;

    current = hash_table->buckets[hash_val];

    if(current != NULL) {    /* Finding the elememnt, the pointer to the tile in the bucket of Hash table is returned if found, else NULL is returned */
        while(current != NULL) {
            if(current->key == key && current->belongs_to == belongs_to) {
                break;
            }
            current = current->next;
        }
        if(NULL != current) {
            return (dtd_tile_t *)current->tile;
        }else {
            return NULL;
        }
    }else {
        return NULL;
    }
}

/* Hash Table insert function */
void
tile_insert_h_t(hash_table *hash_table,
                uint32_t key, dtd_tile_t *tile,
                int h_size, dague_ddesc_t* belongs_to)
{
    uint32_t hash_val = hash_key(key, h_size);
    bucket_element_tile_t *new_list, *current_table_element;

    /** Assigning values to new element **/
    new_list =(bucket_element_tile_t *) malloc(sizeof(bucket_element_tile_t));
    new_list->next = NULL;
    new_list->key = key;
    new_list->tile = tile;
    new_list->belongs_to = belongs_to;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        while(current_table_element->next != NULL) {    /* Finding the last element of the list */
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to manage tiles once insert_task() is called.
 * This function checks if the tile structure(dtd_tile_t) is created for the data already or not
*/
dtd_tile_t*
tile_manage(dague_dtd_handle_t *dague_dtd_handle,
            dague_ddesc_t *ddesc, int i, int j)
{
    dtd_tile_t *tmp = find_tile(dague_dtd_handle->tile_h_table,
                                ddesc->data_key(ddesc, i, j),
                                dague_dtd_handle->tile_hash_table_size,
                                ddesc);

    if( NULL == tmp) {
        dtd_tile_t *temp_tile = (dtd_tile_t*) malloc(sizeof(dtd_tile_t));
        temp_tile->key = ddesc->data_key(ddesc, i, j);
        temp_tile->rank = ddesc->rank_of_key(ddesc, temp_tile->key);
        temp_tile->vp_id = ddesc->vpid_of_key(ddesc, temp_tile->key);
        temp_tile->data = ddesc->data_of_key(ddesc, temp_tile->key);
        temp_tile->data_copy = temp_tile->data->device_copies[0];
        temp_tile->ddesc = ddesc;
        temp_tile->last_user.flow_index = 0;
        temp_tile->last_user.op_type = 0;
        temp_tile->last_user.task = NULL;
        tile_insert_h_t(dague_dtd_handle->tile_h_table,
                        temp_tile->key,
                        temp_tile,
                        dague_dtd_handle->tile_hash_table_size,
                        ddesc);
        return temp_tile;
    }else {
        return tmp;
    }
}

/** task_Hashtable **/
bucket_element_task_t*
find_task(hash_table* hash_table,
          int key, int task_h_size)
{
    uint32_t hash_val = hash_key(key, task_h_size);
    bucket_element_task_t *current;

    current = hash_table->buckets[hash_val];

    if(current != NULL) {    /* Finding the elememnt, the pointer to the list in the Hash table is returned if found, else NILL is returned */
        while(current!=NULL) {
            if(current->key == key) {
                break;
            }
            current = current->next;
        }
        return current;
    }else {
        return NULL;
    }
}

void
task_insert_h_t(hash_table* hash_table, int key,
                dtd_task_t *task, int task_h_size)
{
    int hash_val = hash_key(key, task_h_size);
    bucket_element_task_t *new_list, *current_table_element;

    /** Assigning values to new element **/
    new_list =(bucket_element_task_t *) malloc(sizeof(bucket_element_task_t));
    new_list->next = NULL;
    new_list->key = key;
    new_list->task = task;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        while( current_table_element->next != NULL) {    // Finding the last element of the list
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/** New Set Descendant list **/
void
set_descendant(dtd_task_t *parent_task, uint8_t parent_flow_index,
               dtd_task_t *desc_task, uint8_t desc_flow_index,
               uint8_t parent_op_type, uint8_t desc_op_type)
{
    parent_task->desc[parent_flow_index].op_type_parent = parent_op_type;
    parent_task->desc[parent_flow_index].op_type = desc_op_type;
    parent_task->desc[parent_flow_index].flow_index = desc_flow_index;
    parent_task->desc[parent_flow_index].task = desc_task;
}

/* hook of task() */
static int
test_hook_of_dtd_task(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;

    dtd_task_t * current_task = (dtd_task_t*)this_task;
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[0], this_task);
    current_task->fpointer(this_task);

    return 0;
}

/* chores and dague_function_t structure intitalization */
static const __dague_chore_t dtd_chore[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = test_hook_of_dtd_task },
    {.type =  DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = NULL},             /* End marker */
};

/* for GRAPHER purpose */
static symbol_t symb_dtd_taskid = {
    .name = "task_id",
    .context_index = 0,
    .min = NULL,
    .max = NULL,
    .cst_inc = 1,
    .expr_inc = NULL,
    .flags = 0x0
};

static inline uint64_t DTD_identity_hash(const __dague_dtd_internal_handle_t * __dague_handle,
                                                         const assignment_t * assignments)
{
    return (uint64_t)assignments[0].value;
}

/* dtd_startup_tasks() */
static int
dtd_startup_tasks(dague_context_t * context,
                  __dague_dtd_internal_handle_t * __dague_handle,
                  dague_execution_context_t ** pready_list)
{
    dague_dtd_handle_t* dague_dtd_handle = (dague_dtd_handle_t*)__dague_handle;
    int vpid = 0;

    /* from here dtd_task specific loop starts*/
    dague_list_item_t *tmp_task = (dague_list_item_t*) dague_dtd_handle->ready_task;
    dague_list_item_t *ring;

    while(NULL != tmp_task) {
        ring = dague_list_item_ring_chop (tmp_task);
        DAGUE_LIST_ITEM_SINGLETON(tmp_task);

        if (NULL != pready_list[vpid]) {
            dague_list_item_ring_merge((dague_list_item_t *)tmp_task,
                           (dague_list_item_t *) (pready_list[vpid]));
        }

        pready_list[vpid] = (dague_execution_context_t*)tmp_task;
        vpid = (vpid+1)%context->nb_vp; /* spread the tasks across all the VPs */
        tmp_task = ring;
    }

    return 0;
}

/* Destruct function */
static void
dtd_destructor(__dague_dtd_internal_handle_t * handle)
{
    uint32_t i;

    free(handle->super.super.functions_array);
    handle->super.super.functions_array = NULL;
    handle->super.super.nb_functions = 0;

    for (i = 0; i < (uint32_t) handle->super.arenas_size; i++) {
            if (handle->super.arenas[i] != NULL) {
                free(handle->super.arenas[i]);
                handle->super.arenas[i] = NULL;
        }
    }

    free(handle->super.arenas);
    handle->super.arenas = NULL;
    handle->super.arenas_size = 0;

    /* Destroy the data repositories for this object */
    data_repo_destroy_nothreadsafe(handle->dtd_data_repository);
    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_destruct_dependencies(handle->super.super.dependencies_array[i]);
        handle->super.super.dependencies_array[i] = NULL;
    }

    free(handle->super.super.dependencies_array);
    handle->super.super.dependencies_array = NULL;

    /* Unregister the handle from the devices */
    for (i = 0; i < dague_nb_devices; i++) {
        if (!(handle->super.super.devices_mask & (1 << i)))
            continue;
        handle->super.super.devices_mask ^= (1 << i);
        dague_device_t *device = dague_devices_get(i);
        if ((NULL == device) || (NULL == device->device_handle_unregister))
            continue;
        if (DAGUE_SUCCESS != device->device_handle_unregister(device, &handle->super.super))
            continue;
    }

    /* dtd handle specific */
    dague_mempool_destruct(context_mempool);
    free(handle->super.task_h_table);
    /* end */

    dague_handle_unregister(&handle->super.super);
    free(handle);
}


/* startup_hook() */
void
dtd_startup(dague_context_t * context,
            dague_handle_t * dague_handle,
            dague_execution_context_t ** pready_list)
{
    uint32_t supported_dev = 0;
    __dague_dtd_internal_handle_t *__dague_handle = (__dague_dtd_internal_handle_t *) dague_handle;
    dague_handle->context = context;

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    __dague_handle->super.super.context = context;
    (void) pins_handle_init(&__dague_handle->super.super);
#endif /* defined(PINS_ENABLE) */

    uint32_t wanted_devices = dague_handle->devices_mask;
    dague_handle->devices_mask = 0;

    for (uint32_t _i = 0; _i < dague_nb_devices; _i++) {
        if (!(wanted_devices & (1 << _i)))
            continue;
        dague_device_t *device = dague_devices_get(_i);
        dague_ddesc_t *dague_ddesc;

        if (NULL == device)
            continue;
        if (NULL != device->device_handle_register)
            if (DAGUE_SUCCESS != device->device_handle_register(device, (dague_handle_t *) dague_handle))
            continue;

        supported_dev |= (1 << device->type);
        dague_handle->devices_mask |= (1 << _i);
    }

    dtd_startup_tasks(context, (__dague_dtd_internal_handle_t *) dague_handle, pready_list);
}

/* dague_dtd_new() */
dague_dtd_handle_t *
dague_dtd_new(int task_class_counter,
              int arena_count, int *info)
{
    int i;
    int tile_hash_table_size = TILE_HASH_TABLE_SIZE;    /* Size of hash table */
    int task_hash_table_size = TASK_HASH_TABLE_SIZE;    /* Size of task hash table */
    int function_hash_table_size = FUNCTION_HASH_TABLE_SIZE;    /* Size of function hash table */

    __dague_dtd_internal_handle_t *__dague_handle = (__dague_dtd_internal_handle_t *) calloc (1, sizeof(__dague_dtd_internal_handle_t) );
    __dague_handle->super.tile_hash_table_size = tile_hash_table_size;
    __dague_handle->super.task_hash_table_size = task_hash_table_size;
    __dague_handle->super.function_hash_table_size = function_hash_table_size;
    __dague_handle->super.ready_task = NULL;
    __dague_handle->super.total_task_class = task_class_counter;

    __dague_handle->super.task_h_table = (hash_table *)(generic_create_hash_table(__dague_handle->super.task_hash_table_size,
                                                                                  sizeof(bucket_element_task_t*)));
    __dague_handle->super.tile_h_table = (hash_table *)(generic_create_hash_table(__dague_handle->super.tile_hash_table_size,
                                                                                  sizeof(bucket_element_tile_t*)));
    __dague_handle->super.function_h_table = (hash_table *)(generic_create_hash_table(__dague_handle->super.function_hash_table_size,
                                                                                      sizeof(bucket_element_f_t *)));
    __dague_handle->super.super.devices_mask = DAGUE_DEVICES_ALL;
    __dague_handle->super.INFO = info; /* zpotrf specific; should be removed */

    __dague_handle->super.super.nb_functions = DAGUE_dtd_NB_FUNCTIONS;
    __dague_handle->super.super.functions_array = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));
    for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++){
        __dague_handle->super.super.functions_array[i] = calloc(1, sizeof(dague_function_t));
    }
        //memcpy((dague_function_t *) __dague_handle->super.super.functions_array[0], &dtd_function, sizeof(dague_function_t));
    __dague_handle->super.super.dependencies_array = (dague_dependencies_t **) calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    __dague_handle->super.arenas_size = arena_count;
    __dague_handle->super.arenas = (dague_arena_t **) malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t *));
    for (i = 0; i < __dague_handle->super.arenas_size; i++) {
        __dague_handle->super.arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

    __dague_handle->dtd_data_repository= data_repo_create_nothreadsafe(DTD_TASK_COUNT, MAX_DEP_OUT_COUNT);

#if defined(DAGUE_PROF_TRACE)
    task_class_count = (int *) malloc(sizeof(int));
    *task_class_count = task_class_counter;
    zpotrf_dtd_profiling_array =  calloc(2*task_class_counter, sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */

    __dague_handle->super.super.startup_hook = dtd_startup;
    __dague_handle->super.super.destructor = (dague_destruct_fn_t) dtd_destructor;

    /* initializing mempool_context here */
    context_mempool = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dtd_task_t fake_task;
    dague_mempool_construct( context_mempool,
                             OBJ_CLASS(dtd_task_t), sizeof(dtd_task_t),
                             ((char*)&fake_task.super.mempool_owner) - ((char*)&fake_task),
                             1/* no. of threads*/ );


    (void) dague_handle_reserve_id((dague_handle_t *) __dague_handle);
    return (dague_dtd_handle_t*) __dague_handle;
}

/* DTD version of is_completed() */
static int
dtd_is_ready(const dtd_task_t *dest,
             int dest_flow_index)
{
    dague_dtd_handle_t* dague_dtd_handle = (dague_dtd_handle_t*) dest->super.dague_handle;
    dtd_task_t *dest_task = (dtd_task_t*)dest;

    if ( dest_task->total_flow == dague_atomic_inc_32b(&(dest_task->flow_count))) {
        return 1;
    }
    return 0;
}

/* DTD ontask() */
dague_ontask_iterate_t
dtd_release_dep_fct( dague_execution_unit_t *eu,
                     const dague_execution_context_t* new_context,
                     const dague_execution_context_t * old_context,
                     const dep_t * deps,
                     dague_dep_data_description_t * data,
                     int src_rank, int dst_rank, int dst_vpid,
                     void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    dague_dtd_handle_t* dague_dtd_handle = (dague_dtd_handle_t*) old_context->dague_handle;
    int task_id = new_context->locals[0].value, is_ready;
    char *parent, *dest;
    dtd_task_t *current_task = (dtd_task_t*) new_context;
    dtd_task_t *parent_task = (dtd_task_t*)old_context;

#if defined(DAGUE_PROF_GRAPHER)
    if(NULL!=grapher_file) {
        int i=0;

        parent = parent_task->name;
        for(i=0;i<strlen(parent);i++){
            fprintf(grapher_file,"%c",parent[i]);
        }
        fprintf(grapher_file,"_%d -> ", parent_task->task_id);
        dest = current_task->name;
        for(i=0;i<strlen(dest);i++){
            fprintf(grapher_file,"%c",dest[i]);
        }
        fprintf(grapher_file,"_%d ",task_id);
        fprintf(grapher_file,"[label=tile_%d]\n",deps->dep_index);
        for(i=0;i<strlen(parent);i++){
            fprintf(grapher_file,"%c",parent[i]);
        }
        fprintf(grapher_file,"_%d ", parent_task->task_id);
        fprintf(grapher_file,"[shape=\"polygon\",style=\"filled\",color=\"");
        color_hash(parent);
        fprintf(grapher_file,"\"]\n");
        for(i=0;i<strlen(dest);i++){
            fprintf(grapher_file,"%c",dest[i]);
        }
        fprintf(grapher_file,"_%d ", current_task->task_id);
        fprintf(grapher_file,"[shape=\"polygon\",style=\"filled\",color=\"");
        color_hash(dest);
        fprintf(grapher_file,"\"]\n");
        fflush(grapher_file);
    }
#endif

    is_ready = dtd_is_ready(current_task, deps->flow->flow_index);
    if(is_ready){
        arg->ready_lists[dst_vpid] = (dague_execution_context_t*)
        dague_list_item_ring_push_sorted( (dague_list_item_t*)arg->ready_lists[dst_vpid],
                                           &current_task->super.list_item,
                                           dague_execution_context_priority_comparator );

   }
  return DAGUE_ITERATE_CONTINUE;
}

/* iterate_successors_function */
static void
iterate_successors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg)
{
    dague_dtd_handle_t *dague_dtd_handle = (dague_dtd_handle_t*) this_task->dague_handle;
    dague_dep_data_description_t data;
    dague_arena_t *arena = NULL;
    dtd_task_t *current_task = (dtd_task_t*) this_task;
    dtd_task_t *current_desc_task, *tmp_task;
    dague_data_t *tile_data; // affinity
    dague_ddesc_t * ddesc;
    dague_data_key_t key;
    dep_t* deps;

    uint32_t rank_src=0, rank_dst=0;
    int __nb_elt = -1, vpid_dst = 0, i;
    uint8_t tmp_flow_index, last_iterate_flag=0;

    tmp_task = current_task;
    for(i=0; i<current_task->total_flow; i++) {
        if( (NULL == current_task->desc[i].task ) || (INPUT == current_task->desc[i].op_type_parent) ) {
            continue;
        }

        deps = (dep_t*) malloc (sizeof(dep_t));
        deps->dep_index =  i; //src_flow_index
        tmp_flow_index = i;

        current_desc_task = current_task->desc[i].task;
        if ( current_task->desc[i].op_type == INOUT || current_task->desc[i].op_type == OUTPUT) {
            last_iterate_flag = 1;
        }
        while(NULL != current_desc_task) {
            dague_flow_t* dst_flow = (dague_flow_t*) malloc(sizeof(dague_flow_t));
            dst_flow->flow_index = tmp_task->desc[tmp_flow_index].flow_index;
            tmp_flow_index = dst_flow->flow_index;

            rank_dst = 0;
            deps->flow =  dst_flow;

            ontask(eu, (dague_execution_context_t*)current_desc_task, (dague_execution_context_t*)current_task, deps, &data, rank_src, rank_dst, vpid_dst, ontask_arg);
            if(1 == last_iterate_flag ) {
                break;
            }
            tmp_task = current_desc_task;
            if ( current_desc_task->desc[dst_flow->flow_index].op_type == INOUT ||  current_desc_task->desc[dst_flow->flow_index].op_type == OUTPUT) {
                   last_iterate_flag = 1;
            }
            current_desc_task = tmp_task->desc[dst_flow->flow_index].task;
        }
    }
}

static void
iterate_predecessors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg)
{
}

/* release_deps() */
static int
release_deps_of_dtd(dague_execution_unit_t* eu,
                    dague_execution_context_t* this_task,
                    uint32_t action_mask,
                    dague_remote_deps_t* deps)
{
    dague_release_dep_fct_arg_t arg;
    int __vp_id;

    arg.action_mask = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
    #ifdef HAVE_MPI
    arg.remote_deps = NULL;
    #endif /* HAVE_MPI */
    arg.ready_lists = (NULL != eu) ? alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp) : NULL;

    if (NULL != eu)
        for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL);

    iterate_successors_of_dtd_task(eu, (dague_execution_context_t*)this_task, action_mask, dtd_release_dep_fct, &arg);

    struct dague_vp_s **vps = eu->virtual_process->dague_context->virtual_processes;
    for (__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
        if (NULL == arg.ready_lists[__vp_id]) {
            continue;
        }
        if (__vp_id == eu->virtual_process->vp_id) {
            __dague_schedule(eu, arg.ready_lists[__vp_id]);
        }else {
            __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
        }
        arg.ready_lists[__vp_id] = NULL;
    }

    return 0;
}

/* complete_hook() */
static int
complete_hook_of_dtd(dague_execution_unit_t* context,
                     dague_execution_context_t* this_task)
{
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[1],
                          this_task);
    release_deps_of_dtd(context, (dague_execution_context_t*)this_task, 0xFFFF, NULL);
    return 0;
}

/* prepare_input function */
int
data_lookup_of_dtd_task(dague_execution_unit_t * context,
                        dague_execution_context_t * this_task)
{
    return DAGUE_HOOK_RETURN_DONE;
}

/* Inert Task with one allocation of memory */
void
insert_task_generic_fptr(dague_dtd_handle_t * __dague_handle, 
                      task_func *kernel_pointer, char *name, ...)
{
    va_list args, args_copy;
    int next_arg, counter = 0, tile_op_type, tile_type_index, total_value_size=0;
    int offset = 1;
    long unsigned int total_size;
    void *tmp;
    void *value_block, *current_val, *start_of_allocation;
    task_param_t *param_head = NULL, *current_param = NULL, *tmp_param;

    va_start(args, name);
    va_copy(args_copy, args);    
   
    /* checking with template */ 
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               kernel_pointer,
                                               __dague_handle->function_hash_table_size); /* Hash table lookup to check if the function structure exists or not */

    if(NULL == function){
        next_arg = va_arg(args, int);
        while(next_arg != 0){
            tmp = va_arg(args, void*);
            tile_op_type = va_arg(args, int);
             
            if(tile_op_type == INPUT || tile_op_type == OUTPUT || tile_op_type == INOUT) {
                tile_type_index = va_arg(args, int);
            }else{
                total_value_size += next_arg;
            }
            counter ++;
            next_arg = va_arg(args, int);
        }

    }else {
       counter = function->nb_parameters; 
       total_value_size = function->nb_locals; 
    }

    va_end(args);

    total_size = counter*sizeof(task_param_t) + total_value_size;
    start_of_allocation = (void *) malloc(total_size);
    param_head = (task_param_t *) start_of_allocation;
    value_block = ((char *)start_of_allocation) + counter*sizeof(task_param_t);
    
    next_arg = va_arg(args_copy, int);
    current_param = param_head;        
    current_val = value_block;

    while(next_arg != 0){
        tmp = va_arg(args_copy, void*);
        tile_op_type = va_arg(args_copy, int);
        current_param->tile_type_index = DEFAULT;

        if(tile_op_type == INPUT || tile_op_type == OUTPUT || tile_op_type == INOUT) {
            tile_type_index = va_arg(args_copy, int);
            current_param->tile_type_index = tile_type_index;
            current_param->pointer_to_tile = tmp;
        } else {
            memcpy(current_val, tmp, next_arg);
            current_param->pointer_to_tile = current_val;
            current_val = ((char *)current_val) + next_arg;      
        }

        current_param->operation_type = tile_op_type;
        
        tmp_param = current_param;
        current_param = current_param + offset;
        tmp_param->next = current_param;

        next_arg = va_arg(args_copy, int);

    }
    tmp_param->next = NULL;
    va_end(args_copy);

    _internal_insert_task(__dague_handle, kernel_pointer, param_head, 
                          name, counter, total_value_size);

}

/** PaRSEC INSERT Task Function **/
void
insert_task_generic_fptr_old(dague_dtd_handle_t *__dague_handle,
                         task_func* kernel_pointer, char* name, ...)
{
    va_list arguments;
    int next_arg, tile_type_index, tile_op_type, ii=0;
    void *tmp, *tmp_value;
    task_func *fpointer;
    task_param_t *info, *param_list_head = NULL, *current_param = NULL ;

    va_start(arguments, name);
    next_arg = va_arg(arguments, int);
    while(next_arg != 0) {
        info = (task_param_t *) calloc(1, sizeof(task_param_t));
        info->next = NULL;
        info->tile_type_index = 0;

        tmp = va_arg(arguments, void*);
        tile_op_type = va_arg(arguments, int);

        if(tile_op_type == INPUT || tile_op_type == OUTPUT || tile_op_type == INOUT) {
            tile_type_index = va_arg(arguments, int);
            info->tile_type_index = tile_type_index;
            info->pointer_to_tile = tmp;
        } else {
            tmp_value = (void *) malloc (next_arg);
            memcpy( tmp_value, tmp, next_arg );
            info->pointer_to_tile = tmp_value;
        }

        info->operation_type = tile_op_type;

        if( NULL == param_list_head) {
            param_list_head = info;
        } else {
            assert(NULL != current_param);
            current_param->next = info;
        }

        current_param = info;

        next_arg = va_arg(arguments, int);
    }
    va_end(arguments);

    fpointer = kernel_pointer;
    /* _internal_insert_task(__dague_handle, fpointer, param_list_head, name); */
}

/* Dependencies setting function for Function structures */
void
set_dependencies_for_function(dague_handle_t* dague_handle,
                              dague_function_t *parent_function,
                              dague_function_t *desc_function,
                              uint8_t parent_flow_index,
                              uint8_t desc_flow_index,
                              int tile_type_index)
{
    dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
    uint8_t i, dep_exists = 0, j;

    if (NULL == parent_function) {   /* Data is not coming from any other task */
        if(desc_function->in[desc_flow_index]){
            dague_flow_t *tmp_d_flow = (dague_flow_t *)desc_function->in[desc_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL != tmp_d_flow->dep_in[i]) {
                    if (tmp_d_flow->dep_in[i]->function_id == 100 ) {
                        dep_exists = 1;
                        break;
                    }
                }
            }
        }
        if (!dep_exists) {
#if defined (PRINT_F_STRUCTURE)
            printf("LOCAL -> %s\n", desc_function->name);
#endif
            desc_dep->cond = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index = desc_flow_index;
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */
            desc_dep->belongs_to = desc_function->in[desc_flow_index];

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(desc_function->in[desc_flow_index]);
                    (*desc_in)->dep_in[i] = (dep_t *)desc_dep; /* Setting dep in the next available dep_in array index */
                    break;
                }
            }
        }
        return;
    } else {
        dep_t *parent_dep = (dep_t *) malloc(sizeof(dep_t));
        dague_flow_t *tmp_flow = (dague_flow_t *) parent_function->out[parent_flow_index]; /* just to make code look better */
        uint8_t function_id;

        if (NULL == tmp_flow) {
            dague_flow_t *tmp_p_flow = NULL;
            tmp_flow =(dague_flow_t *) parent_function->in[parent_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL != tmp_flow->dep_in[i]) {
                    if(tmp_flow->dep_in[i]->dep_index == parent_flow_index && tmp_flow->dep_in[i]->dep_datatype_index == tile_type_index) {
                        if(tmp_flow->dep_in[i]->function_id == 100) {
                            set_dependencies_for_function(dague_handle, NULL, desc_function, 0, desc_flow_index, tile_type_index);
                            return;
                        }
                        tmp_p_flow = (dague_flow_t *)tmp_flow->dep_in[i]->flow;
                        parent_function =(dague_function_t *) dague_handle->functions_array[tmp_flow->dep_in[i]->function_id];
                        for(j=0; j<MAX_DEP_OUT_COUNT; j++) {
                            if(NULL != tmp_p_flow->dep_out[j]) {
                                if((dague_flow_t *)tmp_p_flow->dep_out[j]->flow == tmp_flow) {
                                    /* printf("found the actual flow between function %s -> %s\n", parent_function->name, desc_function->name); */
                                    parent_flow_index = tmp_p_flow->dep_out[j]->dep_index;
                                    set_dependencies_for_function(dague_handle, parent_function, desc_function, parent_flow_index, desc_flow_index, tile_type_index);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            dep_exists = 1;
        }

        for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
            if (NULL != tmp_flow->dep_out[i]) {
                if (tmp_flow->dep_out[i]->function_id == desc_function->function_id &&
                    tmp_flow->dep_out[i]->flow == desc_function->in[desc_flow_index] &&
                    tmp_flow->dep_out[i]->dep_datatype_index == tile_type_index) {
                    /* printf("flow_found between functions: %s and %s\n",parent_function->name, desc_function->name); */
                    dep_exists = 1;
                    break;
                }
            }
        }

        if(!dep_exists){
#if defined (PRINT_F_STRUCTURE)
            printf("%s -> %s\n", parent_function->name, desc_function->name);
#endif
            /* setting out-dependency for parent */
            parent_dep->cond = NULL;
            parent_dep->ctl_gather_nb = NULL;
            parent_dep->function_id = desc_function->function_id;
            parent_dep->flow = desc_function->in[desc_flow_index];
            parent_dep->dep_index = parent_flow_index;
            parent_dep->dep_datatype_index = tile_type_index;
            parent_dep->belongs_to = parent_function->out[parent_flow_index];

            for(i=0; i<MAX_DEP_OUT_COUNT; i++) {
                if(NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* to bypass constness in function structure */
                    dague_flow_t **parent_out = (dague_flow_t **)&(parent_function->out[parent_flow_index]);
                    (*parent_out)->dep_out[i] = (dep_t *)parent_dep;
                    break;
                }
            }

            /* setting in-dependency for descendant */
            desc_dep->cond = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id = parent_function->function_id;
            desc_dep->flow = parent_function->out[parent_flow_index];
            desc_dep->dep_index = desc_flow_index;
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->belongs_to = desc_function->in[desc_flow_index];

            for(i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in funciton strucutre */
                    dague_flow_t **desc_in = (dague_flow_t **)&(desc_function->in[desc_flow_index]);
                    (*desc_in)->dep_in[i] = (dep_t *)desc_dep;
                    break;
                }
            }
        }

    }
    return;
}

/* Function structure declaration and initializing */
dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, task_func* fpointer, char* name)
{
    static int handle_id = 0;
    static uint8_t function_counter = 0;

    if(__dague_handle->super.handle_id != handle_id){
        handle_id = __dague_handle->super.handle_id;
        function_counter = 0;
    }

    dague_function_t *function = (dague_function_t *) calloc(1, sizeof(dague_function_t));

    /*
       To bypass const in function structure.
       Getting address of the const members in local mutable pointers.
    */
    char **name_not_const = (char **)&(function->name);
    symbol_t **params = (symbol_t **) &function->params;
    symbol_t **locals = (symbol_t **) &function->locals;
    expr_t **priority = (expr_t **)&function->priority;
    __dague_chore_t **incarnations = (__dague_chore_t **)&(function->incarnations);

    *name_not_const = name;
    function->function_id = function_counter;
    function->nb_flows = 0;
    function->nb_parameters = 0; /* using now to store info about how many parameters this take */
    function->nb_locals = 0; /* using to store total size of value parameters */
    params[0] = &symb_dtd_taskid;
    locals[0] = &symb_dtd_taskid;
    function->data_affinity = NULL;
    function->initial_data = NULL;
    *priority = NULL;
    function->flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK;
    function->dependencies_goal = 0;
    function->key = (dague_functionkey_fn_t *)DTD_identity_hash;
    function->fini = NULL;
    *incarnations = (__dague_chore_t *)dtd_chore;
    function->iterate_successors = iterate_successors_of_dtd_task;
    function->iterate_predecessors = iterate_predecessors_of_dtd_task;
    function->release_deps = release_deps_of_dtd;
    function->prepare_input = data_lookup_of_dtd_task;
    function->prepare_output = NULL;
    function->complete_execution = complete_hook_of_dtd;

    function_insert_h_t(__dague_handle->function_h_table, fpointer, (dague_function_t *)function, __dague_handle->function_hash_table_size); /* Inserting Fucntion structure in the hash table to keep track for each class of task */
    __dague_handle->super.functions_array[function_counter] = (dague_function_t *)function;
    function_counter++;
    return function;
}

/* Setting flows in function (for each task class) */
void
set_flow_in_function(dague_dtd_handle_t *__dague_handle,
                     dtd_task_t *temp_task, int tile_op_type,
                     int flow_index, int tile_type_index)
{
    dague_flow_t* flow = (dague_flow_t *) calloc(1, sizeof(dague_flow_t));
    flow->name = "Random";
    flow->sym_type = 0;
    flow->flow_index = flow_index;
    flow->flow_datatype_mask = 1<<tile_type_index;
    if (tile_op_type == INPUT) {
        flow->flow_flags = FLOW_ACCESS_READ;
    } else if (tile_op_type == OUTPUT) {
        flow->flow_flags = FLOW_ACCESS_WRITE;
    } else if (tile_op_type == INOUT) {
        flow->flow_flags = FLOW_ACCESS_RW;
    }
    if (tile_op_type == INPUT || tile_op_type == INOUT) {
        dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
        *in = flow;
    }
    if (tile_op_type == OUTPUT || tile_op_type == INOUT) {
        dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
        *out = flow;
    }
}

/*
 * Internal INSERT Task Function.
 * Each time the user calls it a task is created with the respective parameters the user has passed.
 * For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
 * structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are tracked from this function.
*/
static void
_internal_insert_task(dague_dtd_handle_t *__dague_handle,
                      task_func* fpointer,
                      task_param_t *param_list_head,
                      char* name, int no_of_param,
                      int size_of_value_param)
{
    static int handle_id = 0;
    static uint32_t task_id = 0, _internal_task_counter=0;
    static int task_class_counter = 0;
    static uint8_t flow_set_flag[MAX_TASK_CLASS];
    int next_arg, tile_type_index,  tile_op_type, i, flow_index = 0;

    if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        task_class_counter = 0;
        for (int i=0; i<MAX_TASK_CLASS; i++)
            flow_set_flag[i] = 0;
    }

    task_param_t *current_param = param_list_head; /* Parameters passed by user when called Insert_task() */
    dtd_tile_t *tile;
    dtd_task_t *current_task = NULL, *temp_task, *task_to_be_in_hasht = NULL;

    temp_task = (dtd_task_t *) dague_thread_mempool_allocate(context_mempool->thread_mempools); /* Creating Task object */
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->flow_count = 0;
    for(int i=0;i<MAX_DESC;i++){
        temp_task->desc[i].op_type_parent = 0;
        temp_task->desc[i].op_type        = 0;
        temp_task->desc[i].flow_index     = 0;
        temp_task->desc[i].task           = NULL;
    }
    for(int i=0;i<MAX_PARAM_COUNT;i++){
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in = NULL;
        temp_task->super.data[i].data_out = NULL;
    }

    /* Creating master function structures */
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size); /* Hash table lookup to check if the function structure exists or not */
    if( NULL == function ) {
        function = create_function(__dague_handle, fpointer, name);
        function->nb_parameters = no_of_param;
        function->nb_locals = size_of_value_param;
        track_function_created_or_not = 1;
    }

    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];

    while( current_param != NULL) {
        tile = (dtd_tile_t*)current_param->pointer_to_tile;
        tile_op_type = current_param->operation_type;
        tile_type_index = current_param->tile_type_index;

        if(tile != NULL){ /* For test purpose */
            if(tile_op_type == INPUT || tile_op_type == OUTPUT || tile_op_type == INOUT) {
                if(0 == flow_set_flag[temp_task->belongs_to_function]){
                    /*setting flow in function structure */
                    set_flow_in_function(__dague_handle, temp_task, tile_op_type, flow_index, tile_type_index);
                }

                if (NULL != tile->last_user.task) {
                    if (tile->last_user.task == temp_task){
                        temp_task->flow_count++;
                    }
                    set_descendant(tile->last_user.task, tile->last_user.flow_index,
                                   temp_task, flow_index, tile->last_user.op_type,
                                   tile_op_type);
                    /*set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)tile->last_user.task->super.function,
                                                  (dague_function_t *)temp_task->super.function,
                                                  tile->last_user.flow_index, flow_index, tile_type_index);
                    */
                } else {
                    temp_task->flow_count++;
                    /*set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                                  (dague_function_t *)temp_task->super.function,
                                                  0, flow_index, tile_type_index);
                    */
                }

                tile->last_user.flow_index = flow_index;
                tile->last_user.op_type = tile_op_type;
                tile->last_user.task = temp_task;
                flow_index++;
            }
        }
        current_param = current_param->next;
    }
    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->super.locals[0].value = task_id;
    temp_task->fpointer = fpointer;
    temp_task->param_list = param_list_head;
    temp_task->task_id = task_id;
    temp_task->total_flow = flow_index;
    temp_task->ready_mask = 0;
    temp_task->name = name;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;


    /* Building list of initial ready task */
    if(temp_task->total_flow == temp_task->flow_count) {
        DAGUE_LIST_ITEM_SINGLETON(temp_task);
        if(NULL != __dague_handle->ready_task) {
            dague_list_item_ring_push((dague_list_item_t*)__dague_handle->ready_task,
                                      (dague_list_item_t*)temp_task);
        }
        __dague_handle->ready_task = temp_task;
    }

#if defined(DAGUE_PROF_TRACE)
    profiling_trace(__dague_handle, task_class_counter, fpointer, name, flow_index);
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    __dague_handle->super.nb_local_tasks = _internal_task_counter;
}
