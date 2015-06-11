#include <stdarg.h>
#include "dague_config.h"
#include "dague.h"
#include "dague/data_distribution.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include "dplasma/lib/memory_pool.h"
#include "dague/data.h"
#include "dague/data_internal.h"
#include "dague/debug.h"
#include "dague/scheduling.h"
#include "dague/mca/pins/pins.h"
#include "dague/remote_dep.h"
#include "dague/datarepo.h"
#include "dague/dague_prof_grapher.h"
#include "dague/mempool.h"
#include "dague/devices/device.h"
#include "dague/constants.h"
#include "dague/vpmap.h"
#include "dague/utils/mca_param.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

static int task_hash_table_size = (10+1);
static int tile_hash_table_size = (100*10+1);

/* To create object of class dtd_task_t that inherits dague_execution_context_t
 * class 
 */
OBJ_CLASS_INSTANCE(dtd_task_t, dague_execution_context_t,
                   NULL, NULL);

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

/* This function infact decrements the dague handles task counter by executing
 * one fake task after all the real tasks has been inserted in the dague context.
 * Arguments:   - handle (dague_dtd_handle_t *)
 * Returns:     - void
*/
void 
increment_task_counter(dague_dtd_handle_t *__dague_handle)
{

    /* Scheduling all the remaining tasks */
    schedule_tasks (__dague_handle);

    /* decrementing the extra task we initialized the handle with */
    __dague_complete_task( &(__dague_handle->super), __dague_handle->super.context); 
}

/* Unpacks all the arguments of a task, the variables(in which the actual values
 * will be copied) are passed from the body of this task and the parameters of each
 * task is copied back on the passed variables * Arguments:   - this_task
 * (dague_execution_context_t *).
                - variadic arguments (the number of arguments depends on the
                  arguments supplied while inserting this task) 
 * Returns:     - void
*/
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

    while (current_param != NULL) {
         tmp = va_arg(arguments, void**);
        if(UNPACK_VALUE == next_arg) {
             memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }else if (UNPACK_DATA == next_arg) {
            tmp_data = ((dtd_tile_t*)(current_param->pointer_to_tile))->data_copy;
            memcpy(tmp, &tmp_data, sizeof(dague_data_copy_t *));
        }else if (UNPACK_SCRATCH == next_arg) {
             memcpy(tmp, &(current_param->pointer_to_tile), sizeof(uintptr_t));
        }
        next_arg = va_arg(arguments, int);
        current_param = current_param->next;
    }
    va_end(arguments);
}

/* For generating color code required for profiling and dot generation 
 * Keeping this function as GET_UNIQUE_RGB_COLOR (PaRSECs default)
 * has worse API and according to me this is better[period]
 */
static inline char*
color_hash(char *name)
{
    int c, r1, r2, g1, g2, b1, b2;
    uint32_t i;
    char *color=(char *)calloc(7,sizeof(char));

    r1 = 0xA3;
    r2 = 7;
    g1 = 0x2C;
    g2 = 135;
    b1 = 0x97;
    b2 = 49;

    for(i=0; i<strlen(name); i++) {
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
    /*fprintf(grapher_file,"#%s",color); */
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

/* Generic function to produce hash from any key
 * Arguments:   - the kay to be hashed (uintptr_t)
                - the size of the hash table (int)
 * Returns:     - the hash value (uint32_t)
*/
uint32_t
hash_key (uintptr_t key, int size)
{
    uint32_t hash_val = key % size;
    return hash_val;
}

/* Function to search for a specific master_structure (named as Function in
 * PaRSEC) from the hash table that stores that structures 
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to search the hash table with (task_func *)
                - size of the hash table (int)
 * Returns:     - the function structure (dague_function_t *) if found / Null if not
 */
dague_function_t *
find_function(hash_table *hash_table,
              task_func *key, int h_size)
{
    uint32_t hash_val = hash_table->hash((uintptr_t)key, h_size);
    bucket_element_f_t *current;

    current = hash_table->buckets[hash_val];

    /* Finding the elememnt, the pointer to the tile in the bucket of Hash table
     * is returned if found, else NULL is returned 
     */ 
    if(current != NULL) {
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

/* Function to insert master structure in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (task_func *)
                - the function structure to be stored (dague_function_t *)
                - the size of the hash table (int)
 * Returns:     - void
 */
void
function_insert_h_t(hash_table *hash_table,
                    task_func *key, dague_function_t *dtd_function,
                    int h_size)
{
    uint32_t hash_val = hash_table->hash((uintptr_t)key, h_size);
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
        /* Finding the last element of the list */
        while(current_table_element->next != NULL) {    
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to search for a specific Tile(used in insert_task interface as
 * representative of a data) from the hash table that stores the tiles
 * Arguments:
                - hash table that stores the Tiles (hash_table *)
                - key to search the hash table with (uint32_t)
                - size of the hash table (int)
                - the data descriptor, in case of tile we use both the
                  descriptor this tile belongs to and the key to uniquely
                  identify a tile (dague_ddesc_t *) 
 * Returns:     - the tile (dtd_tile_t *) if found / Null if not
 */
dtd_tile_t *
find_tile(hash_table *hash_table,
          uint32_t key, int h_size,
          dague_ddesc_t* belongs_to)
{
    bucket_element_tile_t *current;

    uint32_t hash_val = hash_table->hash(key, h_size);
    current           = hash_table->buckets[hash_val];

    /* Finding the elememnt, the pointer to the tile in the bucket of Hash table
     * is returned if found, else NULL is returned 
     */ 
    if(current != NULL) {    
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

/* Function to insert tiles in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (uint32_t)
                - the tile to be stored (dtd_tile_t *)
                - the size of the hash table (int)
                - data descriptor used along key to uniqely identify a tile (dague_ddesc_t *)
 * Returns:     - void
 */
void
tile_insert_h_t(hash_table *hash_table,
                uint32_t key, dtd_tile_t *tile,
                int h_size, dague_ddesc_t* belongs_to)
{
    uint32_t hash_val = hash_table->hash(key, h_size);
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
        /* Finding the last element of the list */
        while(current_table_element->next != NULL) {    
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to search for a specific Task in the hash table that stores the tasks 
 * Arguments:   - hash table that stores the Tiles (hash_table *)
                - key to search the hash table with (uint32_t)
                - size of the hash table (int)
 * Returns:     - the task (dtd_task_t *) if found / Null if not
 */
dtd_task_t*
find_task(hash_table* hash_table,
          int32_t key, int task_h_size)
{
    uint32_t hash_val = hash_table->hash(key, task_h_size);
    bucket_element_task_t *current;

    current = hash_table->buckets[hash_val];
        
    /* Finding the elememnt, the pointer to the list in the Hash table is returned
     * if found, else NILL is returned 
     */
    if(current != NULL) {    
        while(current!=NULL) {
            if(current->key == key) {
                break;
            }
            current = current->next;
        }
        return current->task;
    }else {
        return NULL;
    }
}

/* Function to insert tasks in the hash table
 * Arguments:   - hash table that stores the function structures (hash_table *)
                - key to store it in the hash table (uint32_t)
                - the task to be stored (dtd_task_t *)
                - the size of the hash table (int)
 * Returns:     - void
 */
void
task_insert_h_t(hash_table* hash_table, uint32_t key,
                dtd_task_t *task, int task_h_size)
{
    uint32_t hash_val = hash_table->hash(key, task_h_size);
    bucket_element_task_t *new_list, *current_table_element;

    /* Assigning values to new element */
    new_list       = (bucket_element_task_t *) malloc(sizeof(bucket_element_task_t));
    new_list->next = NULL;
    new_list->key  = key;
    new_list->task = task;

    current_table_element = hash_table->buckets[hash_val];

    if(current_table_element == NULL) {
        hash_table->buckets[hash_val]= new_list;
    }else {
        /* Finding the last element of the list */
        while( current_table_element->next != NULL) {    
            current_table_element = current_table_element->next;
        }
        current_table_element->next = new_list;
    }
}

/* Function to manage tiles once insert_task() is called, this functions is to
 * generate tasks from PTG and insert it using insert task interface.
 * This function checks if the tile structure(dtd_tile_t) is created for the data
 * already or not.
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - data descriptor (dague_ddesc_t *)
                - key of this data (dague_data_key_t)
 * Returns:     - tile, creates one if not already created, and returns that
                  tile, (dtd_tile_t *)
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
        temp_tile->rank                 = 0; 
        temp_tile->vp_id                = 0; 
        temp_tile->data                 = (dague_data_t*)ddesc; 
        temp_tile->data_copy            = temp_tile->data->device_copies[0];
        temp_tile->ddesc                = NULL;
        temp_tile->last_user.flow_index = -1;
        temp_tile->last_user.op_type    = -1;
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

/* Function to manage tiles once insert_task() is called 
 * This function checks if the tile structure(dtd_tile_t) is created for the
 * data already or not 
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - data descriptor (dague_ddesc_t *)
                - key of this data (dague_data_key_t)
 * Returns:     - tile, creates one if not already created, and returns that
                  tile, (dtd_tile_t *)
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
        temp_tile->last_user.flow_index = -1;
        temp_tile->last_user.op_type    = -1;
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

/* This function sets the descendant of a task, the descendant checks the "last
 * user" of a tile to see if there's a last user.  
 * If there is, the descsendant calls this function and sets itself as the
 * descendant of the parent task * Arguments:   
                - parent task (dtd_task_t *)
                - flow index of parent for which we are setting the descendant (uint8_t)
                - the descendant task (dtd_task_t *)
                - flow index of descendant task (uint8_t)
                - operation type of parent on the data (int)
                - operation type of descendant on the data (int)
 * Returns:     - void 
 */
void
set_descendant(dtd_task_t *parent_task, uint8_t parent_flow_index,
               dtd_task_t *desc_task, uint8_t desc_flow_index,
               int parent_op_type, int desc_op_type)
{
    parent_task->desc[parent_flow_index].flow_index     = desc_flow_index;
    parent_task->desc[parent_flow_index].op_type        = desc_op_type;
    parent_task->desc[parent_flow_index].task           = desc_task;
}

/* This function acts as the hook to connect the PaRSEC task with the actual task. 
 * The function users passed while inserting task in PaRSEC is called in this procedure.
 * Called internally by the scheduler
 * Arguments:   - the execution unit (dague_execution_unit_t *)
                - the PaRSEC task (dague_execution_context_t *)
 */
static int
test_hook_of_dtd_task(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    dtd_task_t * current_task                = (dtd_task_t*)this_task;

    DAGUE_TASK_PROF_TRACE(context->eu_profile,
              this_task->dague_handle->profiling_array[2 * this_task->function->function_id],
              this_task);

    int rc = 0;
    /* Check to see which interface, if it is the PTG inserting task in DTD then
     * this condition will be true 
     */

    if(current_task->orig_task != NULL) {
        rc = current_task->fpointer(context, current_task->orig_task);
        if(rc == DAGUE_HOOK_RETURN_DONE) {
            dague_atomic_add_32b(&(((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_scheduled),1);
        }
        if(((dague_dtd_handle_t*)current_task->super.dague_handle)->total_tasks_to_be_exec
            == ((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_scheduled) {

            dague_handle_update_nbtask(current_task->orig_task->dague_handle, -1); 

        } else if (current_task->orig_task->dague_handle->nb_local_tasks == 1 &&
                ((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_created ==
                ((dague_dtd_handle_t*)current_task->super.dague_handle)->tasks_scheduled) {

            dague_handle_update_nbtask(current_task->orig_task->dague_handle, -1); 

        }
    } else { /* this is the default behavior */
        current_task->fpointer(context, this_task);
    }

    return DAGUE_HOOK_RETURN_DONE;
}

/* chores and dague_function_t structure intitalization */
static const __dague_chore_t dtd_chore[] = {
    {.type      = DAGUE_DEV_CPU,
     .evaluate  = NULL,
     .hook      = test_hook_of_dtd_task },
    {.type      = DAGUE_DEV_NONE,
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

/* To amke it consistent with PaRSEC we need to intialize and have function we do not use at this point */
static inline uint64_t DTD_identity_hash(const __dague_dtd_internal_handle_t * __dague_handle,
                                                         const assignment_t * assignments)
{
    return (uint64_t)assignments[0].value;
}

/* This function is called when the handle is enqueued in the context.
 * This function checks if there is any initial ready task to be scheduled and schedules if any
 * Arguments:   - the dague context (dague_context_t *)
                - dague handle (dague_internal_handle_t *)
                - list of PaRSEC tasks (dague_execution_context_t **)
 * Returns:     - 0 (int)
 */
static int
dtd_startup_tasks(dague_context_t * context,
                  __dague_dtd_internal_handle_t * __dague_handle,
                  dague_execution_context_t ** pready_list)
{
    dague_dtd_handle_t* dague_dtd_handle = (dague_dtd_handle_t*)__dague_handle;
    int vpid = 0;

    /* from here dtd_task specific loop starts */
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
        /* spread the tasks across all the VPs */
        vpid              = (vpid+1)%context->nb_vp; 
        tmp_task          = ring;
    }
    dague_dtd_handle->ready_task = NULL;

    return 0;
}

/* Clean up function to clean memory allocated dynamically for the run
 * Arguments:   - the dague handle (dague_dtd_internal_handle_t *)
 * Returns:     - void
 */
void
dtd_destructor(__dague_dtd_internal_handle_t * handle)
{
    int i, j, k;
#if defined(DAGUE_PROF_TRACE)
        free((void *)handle->super.super.profiling_array);
#endif /* defined(DAGUE_PROF_TRACE) */

    for (i = 0; i <DAGUE_dtd_NB_FUNCTIONS; i++) {
        dague_function_t *func = (dague_function_t *) handle->super.super.functions_array[i];
        
        dague_dtd_function_t *func_parent = (dague_dtd_function_t *)func;
        if (func != NULL) {
            for (j=0; j< func->nb_flows; j++) {
                if(func->in[j] != NULL && func->in[j]->flow_flags == FLOW_ACCESS_READ) {
                    for(k=0; k<MAX_DEP_IN_COUNT; k++) {
                        if (func->in[j]->dep_in[k] != NULL) {  
                            free((void*)func->in[j]->dep_in[k]);
                        }
                    }
                    for(k=0; k<MAX_DEP_OUT_COUNT; k++) {
                        if (func->in[j]->dep_out[k] != NULL) {
                            free((void*)func->in[j]->dep_out[k]);
                        }
                    }
                    free((void*)func->in[j]);
                }
                if(func->out[j] != NULL) {
                    for(k=0; k<MAX_DEP_IN_COUNT; k++) {
                        if (func->out[j]->dep_in[k] != NULL) {
                            free((void*)func->out[j]->dep_in[k]);
                        }
                    }
                    for(k=0; k<MAX_DEP_OUT_COUNT; k++) { 
                        if (func->out[j]->dep_out[k] != NULL) {
                            free((void*)func->out[j]->dep_out[k]);
                        }
                    }
                    free((void*)func->out[j]);
                }
            }
            dague_mempool_destruct(func_parent->context_mempool);
            free (func_parent->context_mempool);
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
    for (i=0;i<task_hash_table_size;i++) {
        free((bucket_element_task_t *)handle->super.task_h_table->buckets[i]);
    }
    for (i=0;i<tile_hash_table_size;i++) {
        bucket_element_tile_t *bucket = handle->super.tile_h_table->buckets[i];
        bucket_element_tile_t *tmp_bucket;
        if( bucket != NULL) {
            /* cleaning chains */
            while (bucket != NULL) {
                tmp_bucket = bucket;
                free((dtd_tile_t *)bucket->tile);
                bucket = bucket->next;
                free(tmp_bucket);
            }
        }
    }
    for (i=0;i<DAGUE_dtd_NB_FUNCTIONS;i++){
        free((bucket_element_f_t *)handle->super.function_h_table->buckets[i]);
    }
    hash_table_fini(handle->super.task_h_table);
    hash_table_fini(handle->super.tile_h_table);
    hash_table_fini(handle->super.function_h_table);

    dague_handle_unregister(&handle->super.super);
    free(handle);
}

/* This is the hook that connects the function to start initial ready tasks with the context.
 * Called internally by PaRSEC
 * Arguments:   - dague context (dague_context_t *)
                - dague handle (dague_handle_t *)
                - list of ready tasks (dague_execution_context_t **)
 * Returns:     - void
 */
void
dtd_startup(dague_context_t * context,
            dague_handle_t * dague_handle,
            dague_execution_context_t ** pready_list)
{
    uint32_t supported_dev = 0;
    __dague_dtd_internal_handle_t *__dague_handle = (__dague_dtd_internal_handle_t *) dague_handle;

    /* Create the PINS DATA pointers if PINS is enabled */
#if defined(PINS_ENABLE)
    __dague_handle->super.super.context = context;
#endif /* defined(PINS_ENABLE) */

    uint32_t wanted_devices = dague_handle->devices_mask;
    dague_handle->devices_mask = 0;

    for (uint32_t _i = 0; _i < dague_nb_devices; _i++) {
        if (!(wanted_devices & (1 << _i)))
            continue;
        dague_device_t *device = dague_devices_get(_i);

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
 * Intializes all the needed members and returns the DAGUE handle  
 * Arguments:   - Dague context
                - Total number of task CLASSES
                - Total number of arenas
                - An integer for checking some of the Kernels (TODO: remove it)  
 * Returns:     - DAGUE handle
 * For correct profiling the task_class_counter should be correct    
 */
dague_dtd_handle_t *
dague_dtd_new(dague_context_t* context, 
              int task_class_counter,
              int arena_count, int *info)
{

    if (testing_ptg_to_dtd == 99) {
        testing_ptg_to_dtd = 0;
    } else {
        testing_ptg_to_dtd = 1;
    }

    /* Registering mca param for printing out traversal info */
    (void)dague_mca_param_reg_int_name("dtd", "traversal_info",
                                       "Show graph traversal info",
                                           false, false, 0, &dump_traversal_info);

    /* Registering mca param for printing out function_structure info */
    (void)dague_mca_param_reg_int_name("dtd", "function_info",
                                       "Show master structure info",
                                       false, false, 0, &dump_function_info);

    /* Registering mca param for tile hash table size */
    (void)dague_mca_param_reg_int_name("dtd", "tile_hash_size",
                                       "Registers the supplied size overriding the default size of tile hash table",
                                       false, false, tile_hash_table_size, &tile_hash_table_size);

    /* Registering mca param for task hash table size */
    (void)dague_mca_param_reg_int_name("dtd", "task_hash_size",
                                       "Registers the supplied size overriding the default size of task hash table",
                                       false, false, task_hash_table_size, &task_hash_table_size);

    int i;

    __dague_dtd_internal_handle_t *__dague_handle   = (__dague_dtd_internal_handle_t *) calloc (1, sizeof(__dague_dtd_internal_handle_t) );

    __dague_handle->super.tile_hash_table_size      = tile_hash_table_size;
    __dague_handle->super.task_hash_table_size      = task_hash_table_size;
    __dague_handle->super.function_hash_table_size  = DAGUE_dtd_NB_FUNCTIONS;
    __dague_handle->super.ready_task                = NULL;
    __dague_handle->super.total_task_class          = task_class_counter;
    __dague_handle->super.task_h_table              = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.task_h_table, 
                    __dague_handle->super.task_hash_table_size, 
                    sizeof(bucket_element_task_t*), &hash_key);
    __dague_handle->super.tile_h_table              = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.tile_h_table, 
                    __dague_handle->super.tile_hash_table_size,
                    sizeof(bucket_element_tile_t*), &hash_key);
    __dague_handle->super.function_h_table          = OBJ_NEW(hash_table);
    hash_table_init(__dague_handle->super.function_h_table, 
                    __dague_handle->super.function_hash_table_size,
                    sizeof(bucket_element_f_t *), &hash_key);
    __dague_handle->super.INFO                      = info; /* zpotrf specific; should be removed */
    __dague_handle->super.super.context             = context;
    __dague_handle->super.super.devices_mask        = DAGUE_DEVICES_ALL;
    __dague_handle->super.super.nb_functions        = DAGUE_dtd_NB_FUNCTIONS;
    __dague_handle->super.super.functions_array     = (const dague_function_t **) malloc( DAGUE_dtd_NB_FUNCTIONS * sizeof(dague_function_t *));

    for(i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
        __dague_handle->super.super.functions_array[i] = NULL;
    }

    __dague_handle->super.super.dependencies_array  = (dague_dependencies_t **) calloc(DAGUE_dtd_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
    __dague_handle->super.arenas_size               = arena_count/arena_count;
    __dague_handle->super.arenas = (dague_arena_t **) malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t *));

    for (i = 0; i < __dague_handle->super.arenas_size; i++) {
        __dague_handle->super.arenas[i] = (dague_arena_t *) calloc(1, sizeof(dague_arena_t));
    }

    __dague_handle->dtd_data_repository             = data_repo_create_nothreadsafe(DTD_TASK_COUNT, MAX_DEP_OUT_COUNT);
#if defined(DAGUE_PROF_TRACE)
    __dague_handle->super.super.profiling_array     = calloc (2 * DAGUE_dtd_NB_FUNCTIONS , sizeof(int));
#endif /* defined(DAGUE_PROF_TRACE) */

    __dague_handle->super.tasks_created         = 0; /* For the testing of PTG inserting in DTD */ 
    __dague_handle->super.tasks_scheduled       = 0; /* For the testing of PTG inserting in DTD */ 
    __dague_handle->super.super.nb_local_tasks  = 1; /* For the bounded window, starting with +1 task */ 
    __dague_handle->super.super.startup_hook    = dtd_startup;
    __dague_handle->super.super.destructor      = (dague_destruct_fn_t) dtd_destructor;

    /* for testing interface*/
    __dague_handle->super.total_tasks_to_be_exec = arena_count;

    (void) dague_handle_reserve_id((dague_handle_t *) __dague_handle);
    return (dague_dtd_handle_t*) __dague_handle;
}

/* DTD version of is_completed() 
 * Input:   - dtd task (dtd_task_t *)
            - flow index (int)
 * Return:  - 1 - indicating task in ready / 0 - indicating task is not ready
 */
static int
dtd_is_ready(const dtd_task_t *dest,
             int dest_flow_index)
{
    dtd_task_t *dest_task = (dtd_task_t*)dest;
    if ( dest_task->flow_count == dague_atomic_inc_32b(&(dest_task->flow_satisfied))) {
        return 1;
    }
    return 0;
}

/* Checks whether the task is ready or not and packs the ready tasks in a list 
 *
 */
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
    int is_ready = 0;
    dtd_task_t *current_task = (dtd_task_t*) new_context;
    dtd_task_t *parent_task  = (dtd_task_t*)old_context;

    is_ready = dtd_is_ready(current_task, deps->flow->flow_index);

#if defined(DAGUE_PROF_GRAPHER)
    /* Check to not print stuff redundantly */
    if(!parent_task->dont_skip_releasing_data[deps->dep_index]) { 
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

    if(is_ready) {
        if(dump_traversal_info) {  
            printf("------\ntask Ready: %s \t %d\nTotal flow: %d  flow_count:"
                   "%d\n-----\n", current_task->super.function->name, current_task->task_id,
                   current_task->total_flow, current_task->flow_count); 
        }

        int ii = dague_atomic_cas(&(current_task->ready_mask), 0, 1);
        if(ii) { 
            arg->ready_lists[dst_vpid] = (dague_execution_context_t*)
            dague_list_item_ring_push_sorted( (dague_list_item_t*)arg->ready_lists[dst_vpid],
                                               &current_task->super.list_item,
                                               dague_execution_context_priority_comparator );
            return DAGUE_ITERATE_CONTINUE; /* Returns the status of the task being activated */
         }else {
             return DAGUE_ITERATE_STOP;
         }
    } else {
        return DAGUE_ITERATE_STOP;
    }
    
}

/* This function iterates over all the successors of a task and activates them
 * and builds a list of the ones that got ready by this activation 
 *
 */
static void
iterate_successors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg)
{
    ordering_correctly_1(eu, this_task, action_mask, ontask, ontask_arg);
}

/* To be consistent with PaRSECs structures, is not used or implemented */
static void
iterate_predecessors_of_dtd_task(dague_execution_unit_t * eu,
                               const dague_execution_context_t * this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t * ontask,
                               void *ontask_arg)
{
}

/* Release dependencies after a task is done.
 * Calls iterate successors function that returns a list of tasks that are ready to go.
 * Those ready tasks are scheduled in here
 */
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

/* This function is called internally by PaRSEC once a task is done 
 *
 */
static int
complete_hook_of_dtd(dague_execution_unit_t* context,
                     dague_execution_context_t* this_task)
{
    dtd_task_t *task = (dtd_task_t*) this_task; 
    if (dump_traversal_info) {
        static int counter= 0;
        dague_atomic_add_32b(&counter,1);
        printf("------------------------------------------------\nexecution done"
               "of task: %s \t %d\ntask done %d \n", this_task->function->name, task->task_id,
                counter); 
    }

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

/* prepare_input function, to be consistent with PaRSEC */
int
data_lookup_of_dtd_task(dague_execution_unit_t * context,
                        dague_execution_context_t * this_task)
{
    return DAGUE_HOOK_RETURN_DONE;
}

/* This function creates relationship between different types of task classes.
 * Arguments:   - dague handle (dague_handle_t *)
                - parent master structure (dague_function_t *) 
                - child master structure (dague_function_t *)
                - flow index of task that belongs to the class of "parent master structure" (int)
                - flow index of task that belongs to the class of "child master structure" (int)
                - the type of data (the structure of the data like square,
                  triangular and etc) this dependency is about (int)
 * Returns:     - void
 */
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
        if(parent_function->out[parent_flow_index]) {
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
            if (dump_function_info) {
                printf("%s -> LOCAL\n", parent_function->name);
            }

            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = parent_flow_index;
            desc_dep->belongs_to    = parent_function->out[parent_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            /* specific for cholesky, will need to change */
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = NULL;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == parent_function->out[parent_flow_index]->dep_out[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(parent_function->out[parent_flow_index]);
                    /* Setting dep in the next available dep_in array index */ 
                    (*desc_in)->dep_out[i] = (dep_t *)desc_dep; 
                    break;
                }
            }
        }
        return;
    }

    if (NULL == parent_function) {   /* Data is not coming from any other task */
        if(desc_function->in[desc_flow_index]) {
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
            if(dump_function_info) {
                printf("LOCAL -> %s\n", desc_function->name);
            }
            desc_dep->cond          = NULL;
            desc_dep->ctl_gather_nb = NULL;
            desc_dep->function_id   = 100; /* 100 is used to indicate data is coming from memory */
            desc_dep->dep_index     = desc_flow_index;
            desc_dep->belongs_to    = desc_function->in[desc_flow_index];
            desc_dep->flow          = NULL;
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = tile_type_index; /* specific for cholesky, will need to change */
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = NULL;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if (NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function structure */
                    dague_flow_t **desc_in = (dague_flow_t**)&(desc_function->in[desc_flow_index]);
                    /* Setting dep in the next available dep_in array index */
                    (*desc_in)->dep_in[i]  = (dep_t *)desc_dep; 
                    break;
                }
            }
        }
        return;
    } else {
        dague_flow_t *tmp_flow = (dague_flow_t *) parent_function->out[parent_flow_index];

        if (NULL == tmp_flow) {
            dague_flow_t *tmp_p_flow = NULL;
            tmp_flow =(dague_flow_t *) parent_function->in[parent_flow_index];
            for (i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL != tmp_flow->dep_in[i]) {
                    if(tmp_flow->dep_in[i]->dep_index == parent_flow_index && 
                       tmp_flow->dep_in[i]->dep_datatype_index == tile_type_index) {
                        if(tmp_flow->dep_in[i]->function_id == 100) {
                            set_dependencies_for_function(dague_handle, 
                                                          NULL, desc_function, 0, 
                                                          desc_flow_index, tile_type_index);
                            return;
                        }
                        tmp_p_flow = (dague_flow_t *)tmp_flow->dep_in[i]->flow;
                        parent_function =(dague_function_t *) dague_handle->functions_array[tmp_flow->dep_in[i]->function_id];
                        for(j=0; j<MAX_DEP_OUT_COUNT; j++) {
                            if(NULL != tmp_p_flow->dep_out[j]) {
                                if((dague_flow_t *)tmp_p_flow->dep_out[j]->flow == tmp_flow) {
                                    parent_flow_index = tmp_p_flow->dep_out[j]->dep_index;
                                    set_dependencies_for_function(dague_handle,
                                                                  parent_function, 
                                                                  desc_function, 
                                                                  parent_flow_index, 
                                                                  desc_flow_index,
                                                                  tile_type_index); 
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
                    dep_exists = 1;
                    break;
                }
            }
        }

        if(!dep_exists) {
            dep_t *desc_dep = (dep_t *) malloc(sizeof(dep_t));
            dep_t *parent_dep = (dep_t *) malloc(sizeof(dep_t));

            if (dump_function_info) {
                printf("%s -> %s\n", parent_function->name, desc_function->name);
            }

            /* setting out-dependency for parent */
            parent_dep->cond            = NULL;
            parent_dep->ctl_gather_nb   = NULL;
            parent_dep->function_id     = desc_function->function_id;
            parent_dep->flow            = desc_function->in[desc_flow_index];
            parent_dep->dep_index       = parent_flow_index;
            parent_dep->belongs_to      = parent_function->out[parent_flow_index];
            parent_dep->direct_data     = NULL;
            parent_dep->dep_datatype_index = tile_type_index;
            parent_dep->datatype.type.cst     = 0;
            parent_dep->datatype.layout.cst   = NULL;
            parent_dep->datatype.count.cst    = 0;
            parent_dep->datatype.displ.cst    = 0;

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
            desc_dep->direct_data   = NULL;
            desc_dep->dep_datatype_index = tile_type_index;
            desc_dep->datatype.type.cst     = 0;
            desc_dep->datatype.layout.cst   = NULL;
            desc_dep->datatype.count.cst    = 0;
            desc_dep->datatype.displ.cst    = 0;

            for(i=0; i<MAX_DEP_IN_COUNT; i++) {
                if(NULL == desc_function->in[desc_flow_index]->dep_in[i]) {
                    /* Bypassing constness in function strucutre */
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
 * Also creates the mempool_context for each task class     
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - function pointer to the actual task (task_func *)
                - name of the task class (char *)
                - count of parameter each task of this class has, to estimate the memory we need 
                  to allocate for the mempool (int)
                - total size of memory required in bytes to hold the values of those paramters (int)
                - flow count of the tasks belonging to a particular class (int)
 * Returns:     - the master structure (dague_function_t *)
 */
dague_function_t*
create_function(dague_dtd_handle_t *__dague_handle, task_func* fpointer, char* name,
                int count_of_params, long unsigned int size_of_param, int flow_count)
{
    static int handle_id = 0;
    static uint8_t function_counter = 0;

    /* TODO: Instead of resetting counter we need to keep track of which handles
             we have already encountered already */
     if(__dague_handle->super.handle_id != handle_id) { 
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

    /*int total_size = sizeof(dtd_task_t) + count_of_params * sizeof(task_param_t) 
     + size_of_param + 2; */ /* this is for memory alignment */

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
    function->nb_flows              = flow_count;
    /* set to one so that prof_grpaher prints the task id properly */
    function->nb_parameters         = 1;  
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

    /* Inserting Fucntion structure in the hash table to keep track for each class of task */
    function_insert_h_t(__dague_handle->function_h_table, fpointer,
                       (dague_function_t *)function, __dague_handle->function_hash_table_size);
    __dague_handle->super.functions_array[function_counter] = (dague_function_t *) function;
    function_counter++;
    return function;
}

/* For each flow in the task class we call this function to set up a flow 
 * Arguments:   - dague handle (dague_dtd_handle_t *)
                - the task to extract the class this task belongs to (dtd_task_t *)
                - the operation this flow does on the data (int)
                - the index of this flow for this task class (int)
                - the data type(triangular, square and etc) this flow works on (int)
 * Returns:     - void
 */
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

    int i;
    for (i=0; i<MAX_DEP_IN_COUNT; i++) {
        flow->dep_in[i] = NULL;
    }
    for (i=0; i<MAX_DEP_OUT_COUNT; i++) {
        flow->dep_out[i] = NULL;
    }

    if ((tile_op_type & GET_OP_TYPE) == INPUT) {
        flow->flow_flags = FLOW_ACCESS_READ;
    } else if ((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
        flow->flow_flags = FLOW_ACCESS_WRITE;
    } else if ((tile_op_type & GET_OP_TYPE) == INOUT) {
        flow->flow_flags = FLOW_ACCESS_RW;
    }
        
    /* 
        cannot pack the flows like PTG as it creates 
        a lot more complicated dependency building 
        between master structures.
    */
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
 * Each time the user calls it a task is created with the respective parameters
   the user has passed.
 * For each task class a structure known as "function" is created as well. 
   (e.g. for Cholesky 4 function structures are created for each task class).
 * The flow of data from each task to others and all other dependencies are
   tracked from this function.  
 */
void
insert_task_generic_fptr(dague_dtd_handle_t *__dague_handle,
                         task_func* fpointer,
                         char* name, ...)
{
    va_list args, args_for_size;
    static int handle_id=0;
    static uint32_t task_id=0, _internal_task_counter=0;
    static uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS];
    int next_arg, i, flow_index=0;
    int tile_op_type;
    int track_function_created_or_not=0;
    task_param_t *head_of_param_list, *current_param, *tmp_param;
    void *tmp, *value_block, *current_val; 

    /* resetting static variables for each handle */
    if(__dague_handle->super.handle_id != handle_id) { 
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
            flow_set_flag[i] = 0;
        }
    }

    va_start(args, name);

    /* Creating master function structures */
    /* Hash table lookup to check if the function structure exists or not */
    dague_function_t *function = find_function(__dague_handle->function_h_table,
                                               fpointer,
                                               __dague_handle->function_hash_table_size); 

    if( NULL == function ) {
        /* calculating the size of parameters for each task class*/
        int flow_count_master=0;
        int count_of_params = 0;
        long unsigned int size_of_param = 0;
        va_copy(args_for_size, args);
        next_arg = va_arg(args_for_size, int);
        while(next_arg != 0) {
            count_of_params ++;
            tmp = va_arg(args_for_size, void *);
            tile_op_type = va_arg(args_for_size, int);

            if((tile_op_type & GET_OP_TYPE) == VALUE || (tile_op_type & GET_OP_TYPE) == SCRATCH) {
                size_of_param += next_arg;    
            } else {
                flow_count_master++;
            }
            next_arg = va_arg(args_for_size, int);
        } 

        va_end(args_for_size);

        if (dump_function_info) {
            printf("Function Created for task Class: %s\n Has %d parameters\n"
                   "Total Size: %lu\n", name, count_of_params, size_of_param); 
        }

        function = create_function(__dague_handle, fpointer, name, count_of_params, 
                                   size_of_param, flow_count_master);
        track_function_created_or_not = 1;
    }

    dague_mempool_t *context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dtd_tile_t *tile;
    dtd_task_t *temp_task;

    /* Creating Task object */
    temp_task = (dtd_task_t *)dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools);
  
    /*printf("Orignal Address : %p\t", temp_task);
    int n = ((uintptr_t)temp_task) & 0xF; 
    printf("n is : %lx\n", n);

    uintptr_t ptrr =  ((((uintptr_t)temp_task)+16)/16)*16;
    printf("New Address :%lx \t", ptrr);
    n = ((uintptr_t)ptrr) & 0xF; 
    printf("n is : %lx\n", n);*/
     
    for(i=0;i<MAX_DESC;i++) {
        temp_task->desc[i].op_type_parent = -1;
        temp_task->desc[i].op_type        = -1;
        temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;
        temp_task->dont_skip_releasing_data[i] = 0;
    }
    for(i=0;i<MAX_PARAM_COUNT;i++) {
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in   = NULL;
        temp_task->super.data[i].data_out  = NULL;
    }

    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];
    temp_task->flow_satisfied = 0;
    temp_task->orig_task = NULL;
    temp_task->ready_mask = 0;
    temp_task->task_id = task_id;
    temp_task->total_flow = temp_task->super.function->nb_flows;
    /* +1 to make sure the task is completely ready before it gets executed */
    temp_task->flow_count = temp_task->super.function->nb_flows+1;
    temp_task->fpointer = fpointer;
    temp_task->super.locals[0].value = task_id;
    temp_task->name = name;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    /* Getting the pointer to allocated memory by mempool */
    head_of_param_list = (task_param_t *) (((char *)temp_task) + sizeof(dtd_task_t)); 
    current_param = head_of_param_list;  
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(task_param_t);  
    current_val = value_block;

    next_arg = va_arg(args, int);

    struct user *last_user = (struct user *) malloc(sizeof(struct user));
    while(next_arg != 0) {
        tmp = va_arg(args, void *);
        tile = (dtd_tile_t *) tmp;
        tile_op_type = va_arg(args, int);
        current_param->tile_type_index = REGION_FULL;

        set_task(temp_task, tmp, tile,
                 tile_op_type, current_param,
                 last_user, flow_set_flag, &current_val, 
                 __dague_handle, &flow_index, &next_arg);

        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;
        
        next_arg = va_arg(args, int);
    }
    free(last_user);

    tmp_param->next = NULL;
    va_end(args);


    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->param_list = head_of_param_list;
    

    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_local_tasks),1);
    /* in attempt to make the task not ready till the whole body is constructed */
    dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1); 

    if(!__dague_handle->super.context->active_objects) {
        task_id++;
         /* executing the tasks as soon as we find it, if no engine is attached */
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0], 
                        (dague_execution_context_t *)temp_task);         
        return;
    }

    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied) {
        int ii = dague_atomic_cas(&(temp_task->ready_mask), 0, 1);
        if(ii) {
            DAGUE_LIST_ITEM_SINGLETON(temp_task);
            if(NULL != __dague_handle->ready_task) {
                dague_list_item_ring_push((dague_list_item_t*)__dague_handle->ready_task,
                                          (dague_list_item_t*)temp_task);
            }
            __dague_handle->ready_task = temp_task;
        }
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not) {
        profiling_trace(__dague_handle, function, name, flow_index); 
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    /* Atomically increasing the nb_local_tasks_counter */
    __dague_handle->tasks_created = _internal_task_counter;

    static int task_window_size = 30;

    if((__dague_handle->tasks_created % task_window_size) == 0 ) {
        schedule_tasks (__dague_handle);
    }
}

/* Function that sets all dependencies between tasks according to the operation type of that task 
 * on the data and also created relationship between master structures.
 * Arguments:   - the current task (dtd_task_t *)
                - pointer to the tile/data (void *)
                - pointer to the tile/data (tile *)
                - operation type on the data (int)
                - task parameters (task_param_t *)
                - structure to hold information about the last user of the data, 
                  if any (struct user)
                - array of int to indicate whether we have set a flow for this 
                  task class in the master structure or not (uint8_t [])
                - pointer to the memory allocated by mempool for holding the 
                  parameter of this task (void **)
                - dague handle (dague_dtd_handle_t)
                - current flow index (int *)
                - next argument sent to insert task (int *)
 * Returns:     - void
 */
void
set_task(dtd_task_t *temp_task, void *tmp, dtd_tile_t *tile,
         int tile_op_type, task_param_t *current_param,
         struct user *last_user, uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS], void **current_val, 
         dague_dtd_handle_t *__dague_handle, int *flow_index, int *next_arg)
{
    int tile_type_index;
    if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == INOUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
        tile_type_index = tile_op_type & GET_REGION_INFO;
        current_param->tile_type_index = tile_type_index;
        current_param->pointer_to_tile = tmp;                

        if(tile !=NULL) {
            if(0 == flow_set_flag[temp_task->belongs_to_function]) {
                /*setting flow in function structure */
                set_flow_in_function(__dague_handle, temp_task, tile_op_type, *flow_index, tile_type_index);
            }

            last_user->flow_index   = tile->last_user.flow_index;
            last_user->op_type      = tile->last_user.op_type;
            last_user->task         = tile->last_user.task;
                
            tile->last_user.flow_index       = *flow_index;
            tile->last_user.op_type          = tile_op_type;
            temp_task->desc[*flow_index].op_type_parent = tile_op_type;
            /* Saving tile pointer foreach flow in a task*/
            temp_task->desc[*flow_index].tile = tile; 

            dtd_task_t *parent;
            int no_parent = 1;
            if(NULL != (parent = last_user->task)) {
                int ii = dague_atomic_cas(&(tile->last_user.task), parent, temp_task);
                if(ii) {
                    no_parent = 0;
                    set_descendant(parent, last_user->flow_index,
                               temp_task, *flow_index, last_user->op_type,
                               tile_op_type);
                    if (parent == temp_task) {
                        dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1);
                    }
                    if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                        if (testing_ptg_to_dtd) {
                            set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                  (dague_function_t *)temp_task->super.function, NULL,
                                                  *flow_index, 0, tile_type_index);
                        }
                       
                    } else {
                        if (testing_ptg_to_dtd) {
                            set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                                      (dague_function_t *)parent->super.function,
                                                      (dague_function_t *)temp_task->super.function,
                                                      last_user->flow_index, *flow_index, tile_type_index);
                        }
                    }
                } 
            } 
            if (no_parent) {
                dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1);

                if(INPUT == (tile_op_type & GET_OP_TYPE) || ATOMIC_WRITE == (tile_op_type & GET_OP_TYPE)) {
                    /* Saving the Flow for which a Task is the first one to
                       use the data and the operation is INPUT or ATOMIC_WRITE
                    */
                    temp_task->dont_skip_releasing_data[*flow_index] = 1;
                }

                if((tile_op_type & GET_OP_TYPE) == INPUT || (tile_op_type & GET_OP_TYPE) == INOUT) {
                    if (testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle, NULL,
                                              (dague_function_t *)temp_task->super.function,
                                              0, *flow_index, tile_type_index);
                    }
                }
                if((tile_op_type & GET_OP_TYPE) == OUTPUT || (tile_op_type & GET_OP_TYPE) == ATOMIC_WRITE) {
                    if (testing_ptg_to_dtd) {
                        set_dependencies_for_function((dague_handle_t *)__dague_handle,
                                              (dague_function_t *)temp_task->super.function, NULL,
                                              *flow_index, 0, tile_type_index);
                    }
                }
                
            }

            tile->last_user.task = temp_task; /* at the end to maintain order */
            *flow_index += 1;
        }
    } else if ((tile_op_type & GET_OP_TYPE) == SCRATCH){
        if(NULL == tmp) {
            current_param->pointer_to_tile = *current_val;
            *current_val = ((char*)*current_val) + *next_arg;
        }else {
            current_param->pointer_to_tile = tmp;        
        }
    } else {
        memcpy(*current_val, tmp, *next_arg);
        current_param->pointer_to_tile = *current_val;
        *current_val = ((char*)*current_val) + *next_arg;
    }
    current_param->operation_type = tile_op_type;
}

/* Funciton to schedule tasks in PaRSEC's scheduler
 * Arguments:   - Dague handle that has the list of ready tasks (dague_dtd_handle_t *)
 * Returns:     - void 
 */
void 
schedule_tasks (dague_dtd_handle_t *__dague_handle)
{
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
        /* spread the tasks across all the VPs */
        vpid = (vpid+1)%__dague_handle->super.context->nb_vp; 
        tmp_task = ring;
    }
    __dague_handle->ready_task = NULL; 

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
            __dague_schedule( __dague_handle->super.context->virtual_processes[p]->execution_units[0], 
                            startup_list[p] );
        }
    }
    free(startup_list);
}

/* ------------ */

/*  Everything under this is for testing the insert task interface by using the
    existing PTG tests so not to be counted as a part of insert task interface
 */
/* ------------ */
/*
* INSERT Task Function.
* Each time the user calls it a task is created with the respective parameters the user has passed.
* For each task class a structure known as "function" is created as well. (e.g. for Cholesky 4 function
  structures are created for each task class).
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
    static uint8_t flow_set_flag[DAGUE_dtd_NB_FUNCTIONS];
    int next_arg=-1, i, flow_index = 0;
    int tile_op_type;
    int track_function_created_or_not = 0;
    task_param_t *head_of_param_list, *current_param, *tmp_param;
    void *tmp, *value_block, *current_val; 

    if(__dague_handle->super.handle_id != handle_id) {
        handle_id = __dague_handle->super.handle_id;
        task_id = 0;
        _internal_task_counter = 0;
        for (i=0; i<DAGUE_dtd_NB_FUNCTIONS; i++) {
            flow_set_flag[i] = 0;
        }
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
    
        while(current_paramm != NULL) {
            count_of_params++;
            current_paramm = current_paramm->next;
        }
        
        if (dump_function_info) {
            printf("Function Created for task Class: %s\n Has %d parameters\n Total Size: %lu\n", name, count_of_params, size_of_param);
        }

        function = create_function(__dague_handle, fpointer, name, count_of_params, size_of_param, count_of_params);
        track_function_created_or_not = 1;
    } else { /* Because I am tracking things that does not make a lot of sense hence */
        int count_of_params = 0;

        current_paramm = head_paramm;
    
        while(current_paramm != NULL) { 
            count_of_params++;
            current_paramm = current_paramm->next;
        }
        
        if(function->nb_flows < count_of_params) {
            function->nb_flows = count_of_params;
            flow_set_flag[function->function_id] = 0;
        }
        else if(function->nb_flows > count_of_params) {
            function->nb_flows = count_of_params;
        }
    }

    dague_mempool_t * context_mempool_in_function = ((dague_dtd_function_t*) function)->context_mempool;

    dtd_tile_t *tile;
    dtd_task_t *temp_task;

    temp_task = (dtd_task_t *) dague_thread_mempool_allocate(context_mempool_in_function->thread_mempools); /* Creating Task object */
    for(i=0;i<MAX_DESC;i++) {
        temp_task->desc[i].op_type_parent = -1;
        temp_task->desc[i].op_type        = -1;
        temp_task->desc[i].flow_index     = -1;
        temp_task->desc[i].task           = NULL;

        temp_task->dont_skip_releasing_data[i] = 0;
    }
    for(i=0;i<MAX_PARAM_COUNT;i++) {
        temp_task->super.data[i].data_repo = NULL;
        temp_task->super.data[i].data_in   = NULL;
        temp_task->super.data[i].data_out  = NULL;
    }

    
    dague_execution_context_t *orig_task_copy =(dague_execution_context_t *) malloc(sizeof(dague_execution_context_t));
    memcpy(orig_task_copy, orig_task, sizeof(dague_execution_context_t));
    temp_task->orig_task = orig_task_copy;

    temp_task->super.dague_handle = (dague_handle_t*)__dague_handle;
    temp_task->belongs_to_function = function->function_id;
    temp_task->super.function = __dague_handle->super.functions_array[(temp_task->belongs_to_function)];
    temp_task->flow_satisfied = 0;
    temp_task->ready_mask = 0;
    temp_task->task_id = task_id;
    temp_task->total_flow = temp_task->super.function->nb_flows;
    temp_task->flow_count = temp_task->super.function->nb_flows+1; /* +1 to make sure the task is completely ready before it gets executed */
    temp_task->fpointer = fpointer;
    temp_task->super.locals[0].value = task_id;
    temp_task->name = name;
    temp_task->super.priority = 0;
    temp_task->super.hook_id = 0;
    temp_task->super.chore_id = 0;
    temp_task->super.unused = 0;

    head_of_param_list = (task_param_t *) (((char *)temp_task) + sizeof(dtd_task_t)); /* Getting the pointer allocated from mempool */
    current_param = head_of_param_list;  
    value_block = ((char *)head_of_param_list) + ((dague_dtd_function_t*)function)->count_of_params * sizeof(task_param_t);  
    current_val = value_block;

    current_paramm = head_paramm;

    struct user *last_user = (struct user *) malloc(sizeof(struct user));
    while(current_paramm != NULL) {
        tmp = current_paramm->pointer_to_tile;
        tile = (dtd_tile_t *) tmp;
        tile_op_type = current_paramm->operation_type;
        current_param->tile_type_index = REGION_FULL;
    
        set_task(temp_task, tmp, tile,
                 tile_op_type, current_param,
                 last_user, flow_set_flag, &current_val, 
                 __dague_handle, &flow_index, &next_arg);
 
        tmp_param = current_param;
        current_param = current_param + 1;
        tmp_param->next = current_param;
    
        current_paramm = current_paramm->next;    
    }

    /* Bypassing constness in function structure */
    dague_flow_t **in = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->in[flow_index]);
    *in = NULL;
    dague_flow_t **out = (dague_flow_t **)&(__dague_handle->super.functions_array[temp_task->belongs_to_function]->out[flow_index]);
    *out = NULL;
    flow_set_flag[temp_task->belongs_to_function] = 1;

    /* Assigning values to task objects  */
    temp_task->param_list = head_of_param_list;

    dague_atomic_add_32b((int *)&(__dague_handle->super.nb_local_tasks),1);
    dague_atomic_add_32b((int *)&(temp_task->flow_satisfied),1); /* in attempt to make the task not ready till the whole body is constructed */

    if(!__dague_handle->super.context->active_objects) {
        task_id++;
        __dague_execute(__dague_handle->super.context->virtual_processes[0]->execution_units[0], (dague_execution_context_t *)temp_task);  /* executing the tasks as soon as we find it if no engine is attached */        
        return;
    }
 
    /* Building list of initial ready task */
    if(temp_task->flow_count == temp_task->flow_satisfied && !temp_task->ready_mask) {
        DAGUE_LIST_ITEM_SINGLETON(temp_task);
        if(NULL != __dague_handle->ready_task) {
            dague_list_item_ring_push((dague_list_item_t*)__dague_handle->ready_task,
                                      (dague_list_item_t*)temp_task);
        }
        __dague_handle->ready_task = temp_task;
    }

#if defined(DAGUE_PROF_TRACE)
    if(track_function_created_or_not) {
        profiling_trace(__dague_handle, function, name, flow_index); 
        track_function_created_or_not = 0;
    }
#endif /* defined(DAGUE_PROF_TRACE) */

    /* task_insert_h_t(__dague_handle->task_h_table, task_id, temp_task, __dague_handle->task_h_size); */
    task_id++;
    _internal_task_counter++;
    /* Atomically increasing the nb_local_tasks_counter */
    __dague_handle->tasks_created = _internal_task_counter;

    static int task_window_size = 1;

    if((__dague_handle->tasks_created % task_window_size) == 0 ) {
        schedule_tasks (__dague_handle);
    }
}

/* To copy the dague_context_t of the predecessor needed for tracking control flow
 * 
 */
static dague_ontask_iterate_t copy_content(dague_execution_unit_t *eu, 
                const dague_execution_context_t *newcontext,
                const dague_execution_context_t *oldcontext, 
                const dep_t *dep, dague_dep_data_description_t *data,
                int src_rank, int dst_rank, int dst_vpid, void *param) 
{
    /* assinging 1 to "unused" field in dague_context_t of the successor to indicate we found a predecesor */
    uint8_t *val = (uint8_t *) &(oldcontext->unused);     
    *val += 1;    

    /* Saving the flow index of the parent in the "unused" field of the predecessor */
    uint8_t *val1 = (uint8_t *) &(newcontext->unused);
    dague_flow_t* parent_outflow = (dague_flow_t*)(dep->flow);
    *val1 = parent_outflow->flow_index;

    memcpy(param, newcontext, sizeof(dague_execution_context_t));
    return DAGUE_ITERATE_STOP;
}

static int
fake_hook_for_testing(dague_execution_unit_t * context,
                      dague_execution_context_t * this_task)
{
    int count = 0;
    dague_dtd_handle_t *dtd_handle = __dtd_handle;
    const char *name = this_task->function->name; 
    task_param_t *head_param = NULL, *current_param = NULL, *tmp_param = NULL;
    dague_ddesc_t *ddesc;
    dague_data_key_t key;
    int tmp_op_type;

    int i;

    data_repo_entry_t *entry;
    dague_execution_context_t *T1;

    for (i=0; this_task->function->in[i] != NULL ; i++) {
        tmp_param = (task_param_t *) malloc(sizeof(task_param_t));

        dague_data_copy_t* copy;
        tmp_op_type = this_task->function->in[i]->flow_flags;    
        int op_type;
        int mask, pred_found = 0;

        if ((tmp_op_type & FLOW_ACCESS_RW) == FLOW_ACCESS_RW) {
              op_type = INOUT | REGION_FULL;  
        } else if((tmp_op_type & FLOW_ACCESS_READ) == FLOW_ACCESS_READ) {
              op_type = INPUT | REGION_FULL;  
        } else if((tmp_op_type & FLOW_ACCESS_WRITE) == FLOW_ACCESS_WRITE) {
              op_type = OUTPUT | REGION_FULL;  
        } else if((tmp_op_type) == FLOW_ACCESS_NONE || tmp_op_type == FLOW_HAS_IN_DEPS) {
            op_type = INOUT | REGION_FULL;  

            this_task->unused = 0; 
            T1 = malloc (sizeof(dague_execution_context_t)); 
            mask = 1 << i;
            this_task->function->iterate_predecessors(context, this_task,  mask, copy_content, (void*)T1);
            if (this_task->unused != 0) {
                pred_found = 1;
            } else {
                pred_found = 2;
                continue;
            }  

            if (pred_found == 1) {
                uint64_t id = T1->function->key(T1->dague_handle, T1->locals);
                entry = data_repo_lookup_entry(T1->dague_handle->repo_array[T1->function->function_id], id); 
                copy = entry->data[T1->unused];
            } else {
            }          
        }else {
            continue;
        } 

        if (pred_found == 0) {
            ddesc = (dague_ddesc_t *)this_task->data[i].data_in->original;
            key = this_task->data[i].data_in->original->key;
            OBJ_RETAIN(this_task->data[i].data_in); 
        } else if (pred_found == 1) {
            ddesc = (dague_ddesc_t *)copy->original;
            key   =  copy->original->key; 
        }
        dtd_tile_t *tile = tile_manage_for_testing(dtd_handle, ddesc, key);    

        tmp_param->pointer_to_tile = (void *)tile;
        tmp_param->operation_type = op_type;
        tmp_param->tile_type_index = REGION_FULL; 
        tmp_param->next = NULL;

        if(head_param == NULL) {
            head_param = tmp_param;
        } else {
            current_param->next = tmp_param;
        }
        count ++;
        current_param = tmp_param;
    }

    for (i=0; this_task->function->out[i] != NULL; i++) {
        int op_type;
        tmp_op_type = this_task->function->out[i]->flow_flags;    
        dague_data_copy_t* copy;
        tmp_param = (task_param_t *) malloc(sizeof(task_param_t));
        int pred_found = 0;
        if((tmp_op_type & FLOW_ACCESS_WRITE) == FLOW_ACCESS_WRITE) {
              op_type = OUTPUT | REGION_FULL;  
        } else if((tmp_op_type) == FLOW_ACCESS_NONE || tmp_op_type == FLOW_HAS_IN_DEPS) {
            pred_found = 1;
            op_type = INOUT | REGION_FULL;  

            dague_data_t *fake_data = OBJ_NEW(dague_data_t);
            fake_data->key = rand(); 
            dague_data_copy_t *fake_data_copy = OBJ_NEW(dague_data_copy_t);
            copy = fake_data_copy;
            fake_data_copy->original = fake_data; 
            this_task->data[this_task->function->out[i]->flow_index].data_out = fake_data_copy;

            ddesc = (dague_ddesc_t *)fake_data;
            key   =  fake_data->key;
        } else {
            continue;
        }

        if (pred_found == 0) {
            ddesc = (dague_ddesc_t *)this_task->data[i].data_out->original;
            key = this_task->data[i].data_out->original->key;
            OBJ_RETAIN(this_task->data[i].data_out);
        } else if (pred_found == 1) {
            ddesc = (dague_ddesc_t *)copy->original;
            key   = copy->original->key; 
        }
        dtd_tile_t *tile = tile_manage_for_testing(dtd_handle, ddesc, key);    

            tmp_param->pointer_to_tile = (void *)tile;
            tmp_param->operation_type = op_type;
            tmp_param->tile_type_index = REGION_FULL; 
            tmp_param->next = NULL;

            if(head_param == NULL) {
                head_param = tmp_param;
            } else {
                current_param->next = tmp_param;
            }
            count ++;
            current_param = tmp_param;

    } 

    /* testing Insert Task */
    insert_task_generic_fptr_for_testing(dtd_handle, __dtd_handle->actual_hook[this_task->function->function_id].hook,
                                         this_task, (char *)name, head_param);

    return DAGUE_HOOK_RETURN_DONE;
}

void 
copy_chores(dague_handle_t *handle, dague_dtd_handle_t *dtd_handle)
{
    int total_functions = handle->nb_functions;
    int i, j;
    for (i=0; i<total_functions; i++) {
        for (j =0; handle->functions_array[i]->incarnations[j].hook != NULL; j++) {
            /* saving the CPU hook only */
            if (handle->functions_array[i]->incarnations[j].type == DAGUE_DEV_CPU) {
                dtd_handle->actual_hook[i].hook = handle->functions_array[i]->incarnations[j].hook;
            }
        }
        for (j =0; handle->functions_array[i]->incarnations[j].hook != NULL; j++) { 
            /* copying ther dake hook in all the hooks (CPU, GPU etc) */
            dague_hook_t **hook_not_const = (dague_hook_t **)&(handle->functions_array[i]->incarnations[j].hook);
            *hook_not_const = &fake_hook_for_testing;
        }
    } 
}
