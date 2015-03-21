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

//#define PRINT_F_STRUCTURE
//#define SPIT_TRAVERSAL_INFO
#define DAG_BUILD_2
#define OVERLAP_STRATEGY_1
//#define OVERLAP_STRATEGY_2

#if defined (OVERLAP_STRATEGY_1) || defined (OVERLAP_STRATEGY_2)
    #include "dague/interfaces/superscalar/overlap_strategies.c"
#else
    #include "dague/interfaces/superscalar/insert_function_internal.h"
#endif


 
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
static int
release_deps_of_dtd(struct dague_execution_unit_s *,
                    dague_execution_context_t *,
                    uint32_t, dague_remote_deps_t *);

static dague_hook_return_t
complete_hook_of_dtd(struct dague_execution_unit_s *,
                     dague_execution_context_t *);

void 
increment_task_counter(dague_dtd_handle_t *handle)
{
    dague_atomic_add_32b((int *) &handle->super.nb_local_tasks, -1);
}

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
        }else if (UNPACK_SCRATCH == next_arg){
             memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}


/* To create object of class dtd_task_t that inherits dague_execution_context_t class */
OBJ_CLASS_INSTANCE(dtd_task_t, dague_execution_context_t,
                   NULL, NULL);


static inline char*
color_hash(char *name)
{
    int c, i, r1, r2, g1, g2, b1, b2;
    char *color=(char *)calloc(7,sizeof(char));

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

    snprintf(color,7,"%02X%02X%02X",r1,g1,b1);
    return(color);
}

#if defined(DAGUE_PROF_GRAPHER)
static inline void
print_color_graph(char* name)
{
    char *color = color_hash(name);
    fprintf(grapher_file,"#%s",color);
    free(color); 
}
char *
print_color_graph_str(char* name)
{
    char *color = color_hash(name);
    return color;
}
#endif

#if defined(DAGUE_PROF_TRACE)
static inline char*
fill_color(char *name)
{
    char *str, *color;
    str = (char *)calloc(12,sizeof(char));
    color = color_hash(name);    
    snprintf(str,12,"fill:%s",color);
    free(color);
    return str;
}

void
profiling_trace(dague_dtd_handle_t *__dague_handle,
                          dague_function_t *function, char* name,
                          int flow_count)
{
    char *str = fill_color(name);
    dague_profiling_add_dictionary_keyword(name, str,
                                           sizeof(dague_profile_ddesc_info_t) + flow_count * sizeof(assignment_t),
                                           dague_profile_ddesc_key_to_string,
                                           (int *) &__dague_handle->super.profiling_array[0 +
                                                                                    2 *
                                                                                    function->function_id
                                                                                    /* start key */
                                                                                    ],
                                           (int *) &__dague_handle->super.profiling_array[1 +
                                                                                    2 *
                                                                                    function->function_id
                                                                                    /*  end key */
                                                                                    ]);

    free(str);

}
#endif /* defined(DAGUE_PROF_TRACE) */

/* Hash table creation function */
void *
generic_create_hash_table(int size_of_table, int size_of_each_bucket)
{
    void *new_table;
    new_table                           = malloc(sizeof(hash_table));
    ((hash_table *)new_table)->buckets  = calloc(size_of_table, size_of_each_bucket);
    ((hash_table *)new_table)->size     = size_of_table;
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
    new_list                = (bucket_element_f_t *) malloc(sizeof(bucket_element_f_t));
    new_list->next          = NULL;
    new_list->key           = key;
    new_list->dtd_function  = dtd_function;

    current_table_element   = hash_table->buckets[hash_val];

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
    bucket_element_tile_t *current;

    uint32_t hash_val = hash_key(key, h_size);
    current           = hash_table->buckets[hash_val];

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
    new_list             = (bucket_element_tile_t *) malloc(sizeof(bucket_element_tile_t));
    new_list->next       = NULL;
    new_list->key        = key;
    new_list->tile       = tile;
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
tile_manage_for_testing(dague_dtd_handle_t *dague_dtd_handle,
            dague_ddesc_t *ddesc, dague_data_key_t key)
{
    dtd_tile_t *tmp = find_tile(dague_dtd_handle->tile_h_table,
                                key,
                                dague_dtd_handle->tile_hash_table_size,
                                ddesc);

    if( NULL == tmp) {
        dtd_tile_t *temp_tile           = (dtd_tile_t*) malloc(sizeof(dtd_tile_t));
        temp_tile->key                  = key;
        temp_tile->rank                 = 0; //ddesc->rank_of_key(ddesc, temp_tile->key);
        temp_tile->vp_id                = 0; //ddesc->vpid_of_key(ddesc, temp_tile->key);
        temp_tile->data                 = (dague_data_t*)ddesc; //ddesc->data_of_key(ddesc, temp_tile->key);
        temp_tile->data_copy            = temp_tile->data->device_copies[0];
        temp_tile->ddesc                = NULL;
        temp_tile->last_user.flow_index = 0;
        temp_tile->last_user.op_type    = 0;
        temp_tile->last_user.task       = NULL;
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
        dtd_tile_t *temp_tile           = (dtd_tile_t*) malloc(sizeof(dtd_tile_t));
        temp_tile->key                  = ddesc->data_key(ddesc, i, j);
        temp_tile->rank                 = ddesc->rank_of_key(ddesc, temp_tile->key);
        temp_tile->vp_id                = ddesc->vpid_of_key(ddesc, temp_tile->key);
        temp_tile->data                 = ddesc->data_of_key(ddesc, temp_tile->key);
        temp_tile->data_copy            = temp_tile->data->device_copies[0];
        temp_tile->ddesc                = ddesc;
        temp_tile->last_user.flow_index = 0;
        temp_tile->last_user.op_type    = 0;
        temp_tile->last_user.task       = NULL;
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
    new_list       = (bucket_element_task_t *) malloc(sizeof(bucket_element_task_t));
    new_list->next = NULL;
    new_list->key  = key;
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
               int parent_op_type, int desc_op_type)
{
    parent_task->desc[parent_flow_index].op_type_parent = parent_op_type;
    parent_task->desc[parent_flow_index].op_type        = desc_op_type;
    parent_task->desc[parent_flow_index].flow_index     = desc_flow_index;
    parent_task->desc[parent_flow_index].task           = desc_task;
}

/* hook of task() */
static int
test_hook_of_dtd_task(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;

    dtd_task_t * current_task                = (dtd_task_t*)this_task;
    
    int rc = 0;
    
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id],
                          this_task);

    /* Check to see which interface */
    if(current_task->orig_task != NULL){
        rc = current_task->fpointer(context, current_task->orig_task);
        if(rc == DAGUE_HOOK_RETURN_DONE) {
           // __dague_complete_execution(context, current_task->orig_task);
            dague_atomic_add_32b(&(((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_scheduled),1);
        }
        if(((dague_dtd_handle_t*)current_task->super.dague_handle)->total_tasks_to_be_exec == ((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_scheduled){
            //__dague_complete_execution(context, current_task->orig_task);
            //dague_atomic_dec_32b( &(context->virtual_process->dague_context->active_objects) );
            dague_handle_update_nbtask(current_task->orig_task->dague_handle, -1); 
            //dague_atomic_dec_32b( &(context->virtual_process->dague_context->active_objects) );
        }
    } else {
        current_task->fpointer(context, this_task);
    }

    return DAGUE_HOOK_RETURN_DONE;
}

/* chores and dague_function_t structure intitalization */
static const __dague_chore_t dtd_chore[] = {
    {.type      = DAGUE_DEV_CPU,
     .evaluate  = NULL,
     .hook      = test_hook_of_dtd_task },
    {.type      =  DAGUE_DEV_NONE,
     .evaluate  = NULL,
     .hook      = NULL},             /* End marker */
};

/* for GRAPHER purpose */
static symbol_t symb_dtd_taskid = {
    .name           = "task_id",
    .context_index  = 0,
    .min            = NULL,
    .max            = NULL,
    .cst_inc        = 1,
    .expr_inc       = NULL,
    .flags          = 0x0
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
        vpid              = (vpid+1)%context->nb_vp; /* spread the tasks across all the VPs */
        tmp_task          = ring;
    }
    (dague_list_item_t*) dague_dtd_handle->ready_task = NULL; /* can not be any contention */

    return 0;
}

/* Destruct function */
 void
dtd_destructor(__dague_dtd_internal_handle_t * handle)
{
    uint32_t i, j;

#if defined(DAGUE_PROF_TRACE)
        free((void *)handle->super.super.profiling_array);
#endif /* defined(DAGUE_PROF_TRACE) */

    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_function_t *func = (dague_function_t *) handle->super.super.functions_array[i];
        if (func != NULL){
            free(func);
        }
    }
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
    handle->super.arenas      = NULL;
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
    /*dague_mempool_destruct(context_mempool);
    context_mempool = NULL;*/
    for (i=0;i<TASK_HASH_TABLE_SIZE;i++){
        free((bucket_element_task_t *)handle->super.task_h_table->buckets[i]);
    }
    for (i=0;i<TILE_HASH_TABLE_SIZE;i++){
        bucket_element_tile_t *bucket = handle->super.tile_h_table->buckets[i];
        if( bucket != NULL)
            free((dtd_tile_t *)bucket->tile);
        free((bucket_element_tile_t *)handle->super.tile_h_table->buckets[i]);
    }
    for (i=0;i<FUNCTION_HASH_TABLE_SIZE;i++){
        free((bucket_element_f_t *)handle->super.function_h_table->buckets[i]);
    }
    free(handle->super.task_h_table->buckets);
    free(handle->super.tile_h_table->buckets);
    free(handle->super.function_h_table->buckets);
    free(handle->super.task_h_table);
    free(handle->super.tile_h_table);
    free(handle->super.function_h_table);
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
    /*dague_handle->context = context;*/ /* doing it in dts_new */

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    __dague_handle->super.super.context = context;
    //(void) pins_handle_init(&__dague_handle->super.super);
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

/* dague_dtd_new()  
 * For correct profiling the task_class_counter should be correct   
*/
dague_dtd_handle_t *
dague_dtd_new(dague_context_t* context, 
              int task_class_counter,
              int arena_count, int *info)
{
    int i;
    int tile_hash_table_size = TILE_HASH_TABLE_SIZE;    /* Size of hash table */
    int task_hash_table_size = TASK_HASH_TABLE_SIZE;    /* Size of task hash table */
    int function_hash_table_size = FUNCTION_HASH_TABLE_SIZE;    /* Size of function hash table */


    __dague_dtd_internal_handle_t *__dague_handle   = (__dague_dtd_internal_handle_t *) calloc (1, sizeof(__dague_dtd_internal_handle_t) );
    __dague_handle->super.tile_hash_table_size      = tile_hash_table_size;
    __dague_handle->super.task_hash_table_size      = task_hash_table_size;
    __dague_handle->super.function_hash_table_size  = function_hash_table_size;
    __dague_handle->super.ready_task                = NULL;
    __dague_handle->super.total_task_class          = task_class_counter;
    __dague_handle->super.super.context             = context;

    /* Adding new tile table to track control flows */
    //__dague_handle->super.tile_ctl_table          = (hash_table *)(generic_create_hash_table(__dague_handle->super.tile_hash_table_size,
      //                                                                            sizeof(bucket_element_tile_t*)));

    __dague_handle->super.task_h_table              = (hash_table *)(generic_create_hash_table(__dague_handle->super.task_hash_table_size,
                                                                                  sizeof(bucket_element_task_t*)));
    __dague_handle->super.tile_h_table              = (hash_table *)(generic_create_hash_table(__dague_handle->super.tile_hash_table_size,
                                                                                  sizeof(bucket_element_tile_t*)));
    __dague_handle->super.function_h_table          = (hash_table *)(generic_create_hash_table(__dague_handle->super.function_hash_table_size,
                                                                                  sizeof(bucket_element_f_t *)));
    __dague_handle->super.super.devices_mask        = DAGUE_DEVICES_ALL;
    __dague_handle->super.INFO                      = info; /* zpotrf specific; should be removed */

    __dague_handle->super.super.nb_functions        = DAGUE_dtd_NB_FUNCTIONS;
    __dague_handle->super.super.functions_array     = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));
    for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++){
        __dague_handle->super.super.functions_array[i] = NULL;
    }
        //memcpy((dague_function_t *) __dague_handle->super.super.functions_array[0], &dtd_function, sizeof(dague_function_t));
    __dague_handle->super.super.dependencies_array  = (dague_dependencies_t **) calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    __dague_handle->super.arenas_size               = arena_count/arena_count;
    __dague_handle->super.arenas = (dague_arena_t **) malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t *));
    for (i = 0; i < __dague_handle->super.arenas_size; i++) {
        __dague_handle->super.arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

    __dague_handle->dtd_data_repository             = data_repo_create_nothreadsafe(DTD_TASK_COUNT, MAX_DEP_OUT_COUNT);

#if defined(DAGUE_PROF_TRACE)
    __dague_handle->super.super.profiling_array     = calloc (2 * task_class_counter, sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */

    __dague_handle->super.tasks_created         = 0; /* For the bounded window, starting with 0 tasks */ 
    __dague_handle->super.tasks_scheduled       = 0; /* For the bounded window, starting with 0 tasks */ 
    __dague_handle->super.super.nb_local_tasks  = 1; /* For the bounded window, starting with +1 task */ 
    __dague_handle->super.super.startup_hook    = dtd_startup;
    __dague_handle->super.super.destructor      = (dague_destruct_fn_t) dtd_destructor;


    /* for testing interface*/
    __dague_handle->super.total_tasks_to_be_exec = arena_count;

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

    if ( dest_task->flow_count == dague_atomic_inc_32b(&(dest_task->flow_satisfied))) {
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
    int task_id = new_context->locals[0].value, is_ready = 0;
    char *parent, *dest;
    dtd_task_t *current_task = (dtd_task_t*) new_context;
    dtd_task_t *parent_task  = (dtd_task_t*)old_context;

    is_ready = dtd_is_ready(current_task, deps->flow->flow_index);

#if defined(DAGUE_PROF_GRAPHER)
    if(!parent_task->first_and_input[deps->dep_index]){ /* Check to not print stuff redundantly */

        dague_flow_t * origin_flow = (dague_flow_t*) calloc(1, sizeof(dague_flow_t));
        dague_flow_t * dest_flow = (dague_flow_t*) calloc(1, sizeof(dague_flow_t));
        
        char aa ='A';   
 
        origin_flow->name = &aa;
        dest_flow->name = &aa;
        dest_flow->flow_flags = FLOW_ACCESS_RW;

        dague_prof_grapher_dep(old_context, new_context, is_ready, origin_flow, dest_flow);
    
        free(origin_flow);
        free(dest_flow);
    }
#endif

    if(is_ready){
        #if defined (SPIT_TRAVERSAL_INFO) 
            printf("------\ntask Ready: %s \t %d\nTotal flow: %d  flow_count: %d\n-----\n", current_task->super.function->name, current_task->task_id, current_task->total_flow, current_task->flow_count);
        #endif
        arg->ready_lists[dst_vpid] = (dague_execution_context_t*)
        dague_list_item_ring_push_sorted( (dague_list_item_t*)arg->ready_lists[dst_vpid],
                                           &current_task->super.list_item,
                                           dague_execution_context_priority_comparator );

        return DAGUE_ITERATE_CONTINUE; /* Returns the status of the task being activated, whether 
                                          it is ready or not */
    } else {
        return DAGUE_ITERATE_STOP;
    }
    
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
    dague_arena_t *arena     = NULL;
    dtd_task_t *current_task = (dtd_task_t*) this_task;
    dtd_task_t *current_desc_task, *tmp_task;
    dague_data_t *tile_data; // affinity
    dague_ddesc_t * ddesc;
    dague_data_key_t key;
    dep_t* deps;

    uint32_t rank_src=0, rank_dst=0;
    int __nb_elt = -1, vpid_dst = 0, i;
    uint8_t tmp_flow_index, last_iterate_flag=0;

#if defined (OVERLAP_STRATEGY_1)
    ordering_correctly_1(eu, this_task, action_mask, ontask, ontask_arg);
#else
    deps = (dep_t*) malloc (sizeof(dep_t));
    dague_flow_t* dst_flow = (dague_flow_t*) malloc(sizeof(dague_flow_t));

    for(i=0; i<current_task->total_flow; i++) {
        tmp_task = current_task;
        last_iterate_flag = 0;
        /** 
          * Not iterating if there's no descendant for this flow or
          * if the operation type of current task is READ ONLY
         */
        if( (NULL == current_task->desc[i].task ) || INPUT == (current_task->desc[i].op_type_parent & GET_OP_TYPE) ) {
            /* Checking if this is the first task and
             * the operation is input or not.
             * We can not bypass this task even if the operation type is
             * INPUT only
             */
            if (current_task->first_and_input[i]){
                continue;
            }
        }

        deps->dep_index = i; /* src_flow_index */
        tmp_flow_index  = i;

        current_desc_task = current_task->desc[i].task;

        if ( (current_task->desc[i].op_type & GET_OP_TYPE) == INOUT || (current_task->desc[i].op_type & GET_OP_TYPE) == OUTPUT) {
            last_iterate_flag = 1;
        }
        while(NULL != current_desc_task) {
    #if defined(SPIT_TRAVERSAL_INFO)
            printf("Current successor: %s \t %d\nTotal flow: %d  flow_satisfied: %d\n", current_desc_task->super.function->name, current_desc_task->task_id, current_desc_task->flow_count, current_desc_task->flow_satisfied);
    #endif

            dst_flow->flow_index = tmp_task->desc[tmp_flow_index].flow_index;
            tmp_flow_index = dst_flow->flow_index;

            rank_dst   = 0;
            deps->flow = dst_flow;

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
    free(dst_flow);
    free(deps);
#endif
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

    arg.action_mask  = action_mask;
    arg.output_usage = 0;
    arg.output_entry = NULL;
    #ifdef HAVE_MPI
    arg.remote_deps  = NULL;
    #endif /* HAVE_MPI */
    arg.ready_lists  = (NULL != eu) ? alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp) : NULL;

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
    dtd_task_t *task = (dtd_task_t*) this_task; 
#if defined(SPIT_TRAVERSAL_INFO)
    static int counter= 0;
    dague_atomic_add_32b(&counter,1);
    //printf("------------------------------------------------\n"); 
    printf("------------------------------------------------\nexecution done of task: %s \t %d\ntask done %d \n", this_task->function->name, task->task_id, counter);
    //printf("task done %d \n", counter);
#endif

/* Experimenting with ptg prof grapher */
#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id, 
                            task->task_id);
#endif /* defined(DAGUE_PROF_GRAPHER) */
    
    DAGUE_TASK_PROF_TRACE(context->eu_profile,
                          this_task->dague_handle->profiling_array[2 * this_task->function->function_id + 1],
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

/* Dependencies setting function for Function structures */
void
set_dependencies_for_function(dague_handle_t* dague_handle,
                              dague_function_t *parent_function,
                              dague_function_t *desc_function,
                              uint8_t parent_flow_index,
                              uint8_t desc_flow_index,
                              int tile_type_index)
{
    uint8_t i, dep_exists = 0, j;

    if (NULL == desc_function) {   /* Data is not going to any other task */
        if(parent_function->out[parent_flow_index]){
            dague_flow_t *tmp_d_flow = (dague_flow_t *)parent_function->out[parent_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL != tmp_d_flow->dep_out[i]) {
                    if (tmp_d_flow->dep_out[i]->function_id == 100 ) {
                        dep_exists = 1;
                        break;
                    }
                }
            }
        }
        if (!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
#if defined (PRINT_F_STRUCTURE)
            printf("%s -> LOCAL\n", parent_function->name);
#endif
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = parent_flow_index;
            desc_dep->belongs_to    = parent_function->out[parent_flow_index];
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(parent_function->out[parent_flow_index]);
                    (*desc_in)->dep_out[i] = (dep_t *)desc_dep; /* Setting dep in the next available dep_in array index */
                    break;
                }
            }
        }
        return;
    }

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
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
#if defined (PRINT_F_STRUCTURE)
            printf("LOCAL -> %s\n", desc_function->name);
#endif
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(desc_function->in[desc_flow_index]);
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep; /* Setting dep in the next available dep_in array index */
                    break;
                }
            }
        }
        return;
    } else {
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
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            dep_t *parent_dep = (dep_t *) malloc(sizeof(dep_t));
#if defined (PRINT_F_STRUCTURE)
            printf("%s -> %s\n", parent_function->name, desc_function->name);
#endif
            /* setting out-dependency for parent */
            parent_dep->cond            = NULL;
            parent_dep->ctl_gather_nb   = NULL;
            parent_dep->function_id     = desc_function->function_id;
            parent_dep->flow            = desc_function->in[desc_flow_index];
            parent_dep->dep_index       = parent_flow_index;
            parent_dep->belongs_to      = parent_function->out[parent_flow_index];
            parent_dep->dep_datatype_index = tile_type_index;

            for(i=0; i<MAX_DEP_OUT_COUNT; i++) {
                if(NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* to bypass constness in function structure */
                    dague_flow_t **parent_out = (dague_flow_t **)&(parent_function->out[parent_flow_index]);
                    (*parent_out)->dep_out[i] = (dep_t *)parent_dep;
                    break;
                }
            }

            /* setting in-dependency for descendant */
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = parent_function->function_id;
            desc_dep->flow          = parent_function->out[parent_flow_index];
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->dep_datatype_index = tile_type_index;

            for(i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in funciton strucutre */
                    dague_flow_t **desc_in = (dague_flow_t **)&(desc_function->in[desc_flow_index]);
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep;
                    break;
                }
            }
        }

    }
    return;
}

/* Function structure declaration and initializing 
   Also creates the mempool_context for each task class     
*/
dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, task_func* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param)
{
    static int handle_id = 0;
    static uint8_t function_counter = 0;

    if(__dague_handle->super.handle_id != handle_id){
        handle_id = __dague_handle->super.handle_id;
        function_counter = 0;
    }

    dague_dtd_function_t *dtd_function = (dague_dtd_function_t *) calloc(1, sizeof(dague_dtd_function_t));
    dague_function_t *function = (dague_function_t *) dtd_function;

    dtd_function->count_of_params = count_of_params;
    dtd_function->size_of_param = size_of_param;

    /* Allocating mempool according to the size and param count */
    dtd_function->context_mempool = (dague_mempool_t*) malloc (sizeof(dague_mempool_t));
    dtd_task_t fake_task;
    int total_size = sizeof(dtd_task_t) + count_of_params * sizeof(task_param_t) + size_of_param;
    dague_mempool_construct( dtd_function->context_mempool,
                             OBJ_CLASS(dtd_task_t), total_size,
                             ((char*)&fake_task.super.mempool_owner) - ((char*)&fake_task),
                             1/* no. of threads*/ );


    /*
       To bypass const in function structure.
       Getting address of the const members in local mutable pointers.
    */
    char **name_not_const = (char **)&(function->name);
    symbol_t **params     = (symbol_t **) &function->params;
    symbol_t **locals     = (symbol_t **) &function->locals;
    expr_t **priority     = (expr_t **)&function->priority;
    __dague_chore_t **incarnations = (__dague_chore_t **)&(function->incarnations);

    *name_not_const                 = name;
    function->function_id           = function_counter;
    function->nb_flows              = 0;
    function->nb_parameters         = 1; /* set to one so that prof_grpaher prints the task id properly */ 
    function->nb_locals             = 0; 
    params[0]                       = &symb_dtd_taskid;
    locals[0]                       = &symb_dtd_taskid;
    function->data_affinity         = NULL;
    function->initial_data          = NULL;
    *priority                       = NULL;
    function->flags                 = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK;
    function->dependencies_goal     = 0;
    function->key                   = (dague_functionkey_fn_t *)DTD_identity_hash;
    function->fini                  = NULL;
    *incarnations                   = (__dague_chore_t *)dtd_chore;
    function->iterate_successors    = iterate_successors_of_dtd_task;
    function->iterate_predecessors  = iterate_predecessors_of_dtd_task;
    function->release_deps          = release_deps_of_dtd;
    function->prepare_input         = data_lookup_of_dtd_task;
    function->prepare_output        = NULL;
    function->complete_execution    = complete_hook_of_dtd;

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
    dague_flow_t* flow  = (dague_flow_t *) calloc(1, sizeof(dague_flow_t));
    flow->name          = "Random";
    flow->sym_type      = 0;
    flow->flow_index    = flow_index;
    flow->flow_datatype_mask = 1<<tile_type_index;

    if ((tile_op_type & GET_OP_TYPE) == INPUT) {
        flow->flow_flags = FLOW_ACCESS_READ;
    } else if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
        flow->flow_flags = FLOW_ACCESS_WRITE;
    } else if ((tile_op_type & GET_OP_TYPE) == INOUT) {
        flow->flow_flags = FLOW_ACCESS_RW;
    }
    if ((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
        dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
        *in = flow;
    }
    if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE || (tile_op_type & GET_OP_TYPE) == INOUT) {
        dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
        *out = flow;
    }
}

/*
* INSERT Task Function.
* Each time the user calls it a task is created with the respective parameters the user has passed.
* For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
* structures are created for each task class).
* The flow of data from each task to others and all other dependencies are tracked from this function.
*/
void
insert_task_generic_fptr(dague_dtd_handle_t *__dague_handle,
                         task_func* fpointer,
                         char* name, ...)
{
    va_list args, args_for_size;
    static int handle_id = 0;
    static uint32_t task_id = 0, _internal_task_counter=0;
    static int task_class_counter = 0;
    static uint8_t flow_set_flag[MAX_TASK_CLASS];
    int next_arg, tile_type_index, i, flow_index = 0;
    int tile_op_type;
    int track_function_created_or_not = 0;
    task_param_t *head_of_param_list, *current_param, *tmp_param;
    void *tmp, *value_block, *current_val; 

    if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        task_class_counter = 0;
        for (int i=0; i<MAX_TASK_CLASS; i++)
            flow_set_flag[i] = 0;
    }


    va_start(args, name);

    /* Creating master function structures */
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size); /* Hash table lookup to check if the function structure exists or not */
    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        int count_of_params = 0;
        long unsigned int size_of_param = 0;
        va_copy(args_for_size, args);
        next_arg = va_arg(args_for_size, int);
        while(next_arg != 0){
            count_of_params ++;
            tmp = va_arg(args_for_size, void *);
            tile_op_type = va_arg(args_for_size, int);

            if((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) {
                size_of_param += next_arg;    
            }
            next_arg = va_arg(args_for_size, int);
        } 

        va_end(args_for_size);
#if defined (PRINT_F_STRUCTURE)
        printf("Function Created for task Class: %s\n Has %d parameters\n Total Size: %lu\n", name, count_of_params, size_of_param);
#endif
        function = create_function(__dague_handle, fpointer, name, count_of_params, size_of_param);
        track_function_created_or_not = 1;
    }

    dague_mempool_t * context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dtd_tile_t *tile;
    dtd_task_t *current_task = NULL, *temp_task, *task_to_be_in_hasht = NULL;

    temp_task = (dtd_task_t *) dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools); /* Creating Task object */
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->flow_satisfied = 0;
    temp_task->orig_task = NULL;
    for(int i=0;i<MAX_DESC;i++){
        temp_task->desc[i].op_type_parent = -1;
        temp_task->desc[i].op_type        = -1;
        temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;

        temp_task->first_and_input[i]     = 0;
        temp_task->dont_skip_releasing_data[i] = 0;
    }
    for(int i=0;i<MAX_PARAM_COUNT;i++){
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in   = NULL;
        temp_task->super.data[i].data_out  = NULL;
    }

    

    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];

    head_of_param_list = (task_param_t *) (((char *)temp_task) + sizeof(dtd_task_t)); /* Getting the pointer allocated from mempool */
    current_param = head_of_param_list;  
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(task_param_t);  
    current_val = value_block;


    next_arg = va_arg(args, int);

    while(next_arg != 0){
        tmp = va_arg(args, void *);
        tile = (dtd_tile_t *) tmp;
        tile_op_type = va_arg(args, int);
        current_param->tile_type_index = DEFAULT;

        if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
            tile_type_index = tile_op_type & GET_REGION_INFO;
            current_param->tile_type_index = tile_type_index;
            current_param->pointer_to_tile = tmp;                

            if(tile !=NULL) {
                if(0 == flow_set_flag[temp_task->belongs_to_function]){
                    /*setting flow in function structure */
                    set_flow_in_function(__dague_handle, temp_task, tile_op_type, flow_index, tile_type_index);
                }

                if(NULL != tile->last_user.task) {
                    if (tile->last_user.task == temp_task){
                        temp_task->flow_satisfied++;
                    } //else { /* Setting descendant only if the tasks are diferrent from each other */
                        set_descendant(tile->last_user.task, tile->last_user.flow_index,
                                   temp_task, flow_index, tile->last_user.op_type,
                                   tile_op_type);
                    //}
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)temp_task->super.function, NULL,
                                                  flow_index, 0, tile_type_index);
                    } else {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)tile->last_user.task->super.function,
                                                      (dague_function_t *)temp_task->super.function,
                                                      tile->last_user.flow_index, flow_index, tile_type_index);
                    }
                } else {
                    if(INPUT == (tile_op_type & GET_OP_TYPE) || ATOMIC_WRITE == (tile_op_type & GET_OP_TYPE)){
                        temp_task->first_and_input[flow_index] = 1; /* Saving the Flow for which a Task is the first one to use the data and the operation is INPUT */
                    }
                    temp_task->flow_satisfied++;
                    if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT)
                        set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                                  (dague_function_t *)temp_task->super.function,
                                                  0, flow_index, tile_type_index);
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE)
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)temp_task->super.function, NULL,
                                                  flow_index, 0, tile_type_index);
                    
                }

                tile->last_user.flow_index       = flow_index;
                tile->last_user.op_type          = tile_op_type;
                tile->last_user.task             = temp_task;
                temp_task->desc[flow_index].tile = tile; /* Saving tile pointer foreach flow in a task*/
                temp_task->desc[flow_index].op_type_parent = tile_op_type;            

                flow_index++;
            }
        } else if ((tile_op_type & GET_OP_TYPE) == SCRATCH){
            if(NULL == tmp){
                current_param->pointer_to_tile = current_val;
                current_val = ((char*)current_val) + next_arg;
                //current_param->pointer_to_tile = malloc(next_arg);        
            }else {
                current_param->pointer_to_tile = tmp;        
            }
        } else {
            memcpy(current_val, tmp, next_arg);
            current_param->pointer_to_tile = current_val;
            current_val = ((char*)current_val) + next_arg;
        }
        current_param->operation_type = tile_op_type;
        
        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;
        
        next_arg = va_arg(args, int);
    }

    tmp_param->next = NULL;
    va_end(args);


    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->super.locals[0].value = task_id;
    temp_task->fpointer = fpointer;
    temp_task->param_list = head_of_param_list;
    temp_task->task_id = task_id;
    temp_task->total_flow = flow_index;
    temp_task->flow_count = flow_index;
    temp_task->ready_mask = 0;
    temp_task->name = name;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    if(!__dague_handle->super.context->active_objects) {
        //printf("Context not attached\n");
        task_id++;
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0], (dague_execution_context_t *)temp_task);  /* executing the tasks as soon as we find it if no engine is attached */        
        return;
    }
 
    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied) {
        DAGUE_LIST_ITEM_SINGLETON(temp_task);
        if(NULL != __dague_handle->ready_task) {
            dague_list_item_ring_push((dague_list_item_t*)__dague_handle->ready_task,
                                      (dague_list_item_t*)temp_task);
        }
        __dague_handle->ready_task = temp_task;
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not){
        profiling_trace(__dague_handle, function, name, flow_index); 
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    //printf("internal task counter: %s \t %d\n", temp_task->super.function->name, task_id);
    /* __dague_handle->super.nb_local_tasks = _internal_task_counter; 
    __dague_handle->super.nb_local_tasks++; */
    /* Atomically increasing the nb_local_tasks_counter */
    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_local_tasks),1);
    __dague_handle->tasks_created = _internal_task_counter;

    int task_window_size = 1;
    static int first_time = 1;

    if((__dague_handle->tasks_created % task_window_size) == 0 ){
        dague_execution_context_t **startup_list;
        startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*));

        int vpid = 0;
        /* from here dtd_task specific loop starts*/
        dague_list_item_t *tmp_task = (dague_list_item_t*) __dague_handle->ready_task;
        dague_list_item_t *ring;

        while(NULL != tmp_task) {
            ring = dague_list_item_ring_chop (tmp_task);
            DAGUE_LIST_ITEM_SINGLETON(tmp_task);

            if (NULL != startup_list[vpid]) {
                dague_list_item_ring_merge((dague_list_item_t *)tmp_task,
                               (dague_list_item_t *) (startup_list[vpid]));
            }

            startup_list[vpid] = (dague_execution_context_t*)tmp_task;
            vpid = (vpid+1)%__dague_handle->super.context->nb_vp; /* spread the tasks across all the VPs */
            tmp_task = ring;
        }
        (dague_list_item_t*) __dague_handle->ready_task = NULL; /* can not be any contention */

        int p;
        for(p = 0; p < vpmap_get_nb_vp(); p++) {
            if( NULL != startup_list[p] ) {
                dague_list_t temp;

                OBJ_CONSTRUCT( &temp, dague_list_t );
                /* Order the tasks by priority */
                dague_list_chain_sorted(&temp, (dague_list_item_t*)startup_list[p],
                                        dague_execution_context_priority_comparator);
                startup_list[p] = (dague_execution_context_t*)dague_list_nolock_unchain(&temp);
                OBJ_DESTRUCT(&temp);
                /* We should add these tasks on the system queue when there is one */
                __dague_schedule( __dague_handle->super.context->virtual_processes[p]->execution_units[0], startup_list[p] );
            }
        }
        free(startup_list);
        first_time = 0;
    }
}

/*
* INSERT Task Function.
* Each time the user calls it a task is created with the respective parameters the user has passed.
* For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
* structures are created for each task class).
* The flow of data from each task to others and all other dependencies are tracked from this function.
*/
void
insert_task_generic_fptr_for_testing(dague_dtd_handle_t *__dague_handle,
                         task_func* fpointer, dague_execution_context_t *orig_task,
                         char* name, task_param_t *head_paramm)
{
    task_param_t *current_paramm;
    static int handle_id = 0;
    static uint32_t task_id = 0, _internal_task_counter=0;
    static int task_class_counter = 0;
    static uint8_t flow_set_flag[MAX_TASK_CLASS];
    int next_arg, tile_type_index, i, flow_index = 0;
    int tile_op_type;
    int track_function_created_or_not = 0;
    task_param_t *head_of_param_list, *current_param, *tmp_param;
    void *tmp, *value_block, *current_val; 

    

    if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        task_class_counter = 0;
        for (int i=0; i<MAX_TASK_CLASS; i++)
            flow_set_flag[i] = 0;
    }

    /* Creating master function structures */
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size); /* Hash table lookup to check if the function structure exists or not */
    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        int count_of_params = 0;
        long unsigned int size_of_param = 0;

        current_paramm = head_paramm;
    
        while(current_paramm != NULL){
            count_of_params++;
            current_paramm = current_paramm->next;
        }
        
#if defined (PRINT_F_STRUCTURE)
        printf("Function Created for task Class: %s\n Has %d parameters\n Total Size: %lu\n", name, count_of_params, size_of_param);
#endif
        function = create_function(__dague_handle, fpointer, name, count_of_params, size_of_param);
        track_function_created_or_not = 1;
    }

    dague_mempool_t * context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dtd_tile_t *tile;
    dtd_task_t *current_task = NULL, *temp_task, *task_to_be_in_hasht = NULL;

    temp_task = (dtd_task_t *) dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools); /* Creating Task object */
    DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->flow_satisfied = 0;
    for(int i=0;i<MAX_DESC;i++){
        temp_task->desc[i].op_type_parent = -1;
        temp_task->desc[i].op_type        = -1;
        temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;

        temp_task->first_and_input[i]     = 0;
        temp_task->dont_skip_releasing_data[i] = 0;
    }
    for(int i=0;i<MAX_PARAM_COUNT;i++){
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in   = NULL;
        temp_task->super.data[i].data_out  = NULL;
    }

    
    dague_execution_context_t *orig_task_copy =(dague_execution_context_t *) malloc(sizeof(dague_execution_context_t));
    memcpy(orig_task_copy, orig_task, sizeof(dague_execution_context_t));
    temp_task->orig_task = orig_task_copy;

    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];

    head_of_param_list = (task_param_t *) (((char *)temp_task) + sizeof(dtd_task_t)); /* Getting the pointer allocated from mempool */
    current_param = head_of_param_list;  
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(task_param_t);  
    current_val = value_block;

    current_paramm = head_paramm;

    while(current_paramm != NULL){
        tmp = current_paramm->pointer_to_tile;
        tile = (dtd_tile_t *) tmp;
        tile_op_type = current_paramm->operation_type;
        current_param->tile_type_index = DEFAULT;

        if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
            tile_type_index = tile_op_type & GET_REGION_INFO;
            current_param->tile_type_index = tile_type_index;
            current_param->pointer_to_tile = tmp;                

            if(tile !=NULL) {
                if(0 == flow_set_flag[temp_task->belongs_to_function]){
                    /*setting flow in function structure */
                    set_flow_in_function(__dague_handle, temp_task, tile_op_type, flow_index, tile_type_index);
                }

                if(NULL != tile->last_user.task) {
                    if (tile->last_user.task == temp_task){
                        temp_task->flow_satisfied++;
                    } //else { /* Setting descendant only if the tasks are diferrent from each other */
                        set_descendant(tile->last_user.task, tile->last_user.flow_index,
                                   temp_task, flow_index, tile->last_user.op_type,
                                   tile_op_type);
                    //}
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)temp_task->super.function, NULL,
                                                  flow_index, 0, tile_type_index);
                    } else {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)tile->last_user.task->super.function,
                                                      (dague_function_t *)temp_task->super.function,
                                                      tile->last_user.flow_index, flow_index, tile_type_index);
                    }
                } else {
                    if(INPUT == (tile_op_type & GET_OP_TYPE)){
                        temp_task->first_and_input[flow_index] = 1; /* Saving the Flow for which a Task is the first one to use the data and the operation is INPUT */
                    }
                    temp_task->flow_satisfied++;
                    if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT)
                        set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                                  (dague_function_t *)temp_task->super.function,
                                                  0, flow_index, tile_type_index);
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE)
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)temp_task->super.function, NULL,
                                                  flow_index, 0, tile_type_index);
                    
                }

                tile->last_user.flow_index       = flow_index;
                tile->last_user.op_type          = tile_op_type;
                tile->last_user.task             = temp_task;
                temp_task->desc[flow_index].tile = tile; /* Saving tile pointer foreach flow in a task*/
                temp_task->desc[flow_index].op_type_parent = tile_op_type;            
    
                flow_index++;
            }
        } else if ((tile_op_type & GET_OP_TYPE) == SCRATCH){
            if(NULL == tmp){
                current_param->pointer_to_tile = current_val;
                current_val = ((char*)current_val) + next_arg;
                //current_param->pointer_to_tile = malloc(next_arg);        
            }else {
                current_param->pointer_to_tile = tmp;        
            }
        } else {
            memcpy(current_val, tmp, next_arg);
            current_param->pointer_to_tile = current_val;
            current_val = ((char*)current_val) + next_arg;
        }
        current_param->operation_type = tile_op_type;
        
        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;
    
        current_paramm = current_paramm->next;    
    }

    //tmp_param->next = NULL;


    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->super.locals[0].value = task_id;
    temp_task->fpointer = fpointer;
    temp_task->param_list = head_of_param_list;
    temp_task->task_id = task_id;
    temp_task->total_flow = flow_index;
    temp_task->flow_count = flow_index;
    temp_task->ready_mask = 0;
    temp_task->name = name;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    if(!__dague_handle->super.context->active_objects) {
        //printf("Context not attached\n");
        task_id++;
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0], (dague_execution_context_t *)temp_task);  /* executing the tasks as soon as we find it if no engine is attached */        
        return;
    }
 
    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied) {
        DAGUE_LIST_ITEM_SINGLETON(temp_task);
        if(NULL != __dague_handle->ready_task) {
            dague_list_item_ring_push((dague_list_item_t*)__dague_handle->ready_task,
                                      (dague_list_item_t*)temp_task);
        }
        __dague_handle->ready_task = temp_task;
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not){
        profiling_trace(__dague_handle, function, name, flow_index); 
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    //printf("internal task counter: %s \t %d\n", temp_task->super.function->name, task_id);
    /* __dague_handle->super.nb_local_tasks = _internal_task_counter; 
    __dague_handle->super.nb_local_tasks++; */
    /* Atomically increasing the nb_local_tasks_counter */
    dague_atomic_add_32b((int *) &(__dague_handle->super.nb_local_tasks),1);
    __dague_handle->tasks_created = _internal_task_counter;

    int task_window_size = 1;
    static int first_time = 1;

    if((__dague_handle->tasks_created % task_window_size) == 0 ){
        dague_execution_context_t **startup_list;
        startup_list = (dague_execution_context_t**)calloc( vpmap_get_nb_vp(), sizeof(dague_execution_context_t*));

        int vpid = 0;
        /* from here dtd_task specific loop starts*/
        dague_list_item_t *tmp_task = (dague_list_item_t*) __dague_handle->ready_task;
        dague_list_item_t *ring;

        while(NULL != tmp_task) {
            ring = dague_list_item_ring_chop (tmp_task);
            DAGUE_LIST_ITEM_SINGLETON(tmp_task);

            if (NULL != startup_list[vpid]) {
                //dague_list_item_ring_merge((dague_list_item_t *)tmp_task,
                  //             (dague_list_item_t *) (startup_list[vpid]));
                dague_list_item_ring_merge((dague_list_item_t *) (startup_list[vpid]), 
                                           (dague_list_item_t *)tmp_task);
            } else{
                startup_list[vpid] = (dague_execution_context_t*)tmp_task;
            }
            vpid = (vpid+1)%__dague_handle->super.context->nb_vp; /* spread the tasks across all the VPs */
            tmp_task = ring;
        }
        (dague_list_item_t*) __dague_handle->ready_task = NULL; /* can not be any contention */

        int p;
        for(p = 0; p < vpmap_get_nb_vp(); p++) {
            if( NULL != startup_list[p] ) {
                dague_list_t temp;

                OBJ_CONSTRUCT( &temp, dague_list_t );
                /* Order the tasks by priority */
                dague_list_chain_sorted(&temp, (dague_list_item_t*)startup_list[p],
                                        dague_execution_context_priority_comparator);
                startup_list[p] = (dague_execution_context_t*)dague_list_nolock_unchain(&temp);
                OBJ_DESTRUCT(&temp);
                /* We should add these tasks on the system queue when there is one */
                __dague_schedule( __dague_handle->super.context->virtual_processes[p]->execution_units[0], startup_list[p] );
            }
        }
        free(startup_list);
        first_time = 0;
    }
}

static dague_ontask_iterate_t copy_content(dague_execution_unit_t *eu, const dague_execution_context_t *newcontext,
              const dague_execution_context_t *oldcontext, const dep_t *dep, dague_dep_data_description_t *data,
              int src_rank, int dst_rank, int dst_vpid, void *param) {

    uint8_t *val = (uint8_t *) &(oldcontext->unused);
    *val += 1;    

    printf("pred called\n");

    memcpy(param, newcontext, sizeof(dague_execution_context_t));
    return DAGUE_ITERATE_STOP;
}

static int
fake_hook_for_testing(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    //OBJ_RETAIN((dague_object_t *)this_task);
    dague_dtd_handle_t *dtd_handle = __dtd_handle;
    int total_flows = 0;  
    const char *name = this_task->function->name; 
    task_param_t *head_param = NULL, *current_param, *tmp_param;
    dague_ddesc_t *ddesc;
    dague_data_key_t key;

    //TODO: Build list of parameters and call insert task

    for (int i=0; this_task->function->in[i] != NULL ; i++){
        tmp_param = (task_param_t *) malloc(sizeof(task_param_t));

        int tmp_op_type = this_task->function->in[i]->flow_flags;    
        int op_type;
        int mask, pred_found = 0;

        if ((tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_RW) {
              op_type = INOUT | REGION_FULL;  
        } else if((tmp_op_type & FLOW_ACCESS_READ) == FLOW_ACCESS_READ){
              op_type = INPUT | REGION_FULL;  
        } else if((tmp_op_type & FLOW_ACCESS_WRITE) == FLOW_ACCESS_WRITE) {
              op_type = OUTPUT | REGION_FULL;  
        } else if((tmp_op_type & FLOW_ACCESS_NONE) == FLOW_ACCESS_NONE) {

            /*this_task->unused = 0; 
            dague_execution_context_t *T1 = malloc (sizeof(dague_execution_context_t)); 
            mask = 1 << i;
            this_task->function->iterate_predecessors(context, this_task,  mask, copy_content, (void*)T1);
            if (this_task->unused != 0) {
                pred_found = 1;
            }   

            if (pred_found) {
                printf("Has pred\n");
            } else {
                printf("first _task\n");

            } */         
    
            continue;
        }else {
            continue;
        } 

        //ddesc = this_task->data[i].data_in->original->ddesc;
        //if (!pred_found) {
            ddesc = (dague_ddesc_t *)this_task->data[i].data_in->original;
            key = this_task->data[i].data_in->original->key;
            dtd_tile_t *tile = tile_manage_for_testing(dtd_handle, ddesc, key);    
        //}    

        tmp_param->pointer_to_tile = (void *)tile;
        tmp_param->operation_type = op_type;
        tmp_param->tile_type_index = REGION_FULL; 
        tmp_param->next = NULL;

        if(head_param == NULL){
            head_param = tmp_param;
        } else {
            current_param->next = tmp_param;
        }
        current_param = tmp_param;
    }

    /* testing Insert Task */
    
    insert_task_generic_fptr_for_testing(dtd_handle, __dtd_handle->actual_hook[this_task->function->function_id].hook,
                         this_task,
                         (char *)name, head_param);

    //__dtd_handle->actual_hook[this_task->function->function_id].hook(context, this_task);
    
    //if(dtd_handle->total_tasks_to_be_exec-1 == dtd_handle->tasks_scheduled){
      //  return DAGUE_HOOK_RETURN_NEXT;
    //}else {
        return DAGUE_HOOK_RETURN_DONE;
    //}
}

void 
copy_chores(dague_handle_t *handle, dague_dtd_handle_t *dtd_handle)
{
    int total_functions = handle->nb_functions;
    int i;
    for (i=0; i<total_functions; i++){
        dtd_handle->actual_hook[i].hook = handle->functions_array[i]->incarnations->hook;
        //actual_hook[i].hook = handle->functions_array[i]->incarnations->hook;
        dague_hook_t **hook_not_const = (dague_hook_t **)&(handle->functions_array[i]->incarnations->hook);
        *hook_not_const = &fake_hook_for_testing;
    } 


}


