#include <pthread.h>
#include "buf2cache.h"

////////////////////////////////////////////////////////////////////////////////
// Types

// TODO: Pad the following structs appropriately 
typedef struct _arrayInfo_t arrayInfo_t;
struct _arrayInfo_t{
    void *array;
    size_t size;
    arrayInfo_t *next;
    arrayInfo_t *prev;
};

typedef struct{
    uint32_t available;
    pthread_mutex_t mutex;
    arrayInfo_t *arrays_start;
    arrayInfo_t *arrays_head;
    arrayInfo_t *arrays_tail;
} cacheInfo_t;

////////////////////////////////////////////////////////////////////////////////
// Global variables

// This is the actual structure with the info about the tiles that are cached
static cacheInfo_t *cache_info[3] = {NULL,NULL,NULL};
// Number of caches of each level per node 
static int cache_count[3] = {0,0,0};
// Number of processing units per node
static int pu_count = 0;

static int cache_tile_capacity[3];

////////////////////////////////////////////////////////////////////////////////
// Function code

void dplasma_hwloc_init_cache(int npu, int level, int npu_per_cache, int cache_size, int tile_size){
    int count = npu/npu_per_cache, tile_count, i;

    pu_count = npu;

    tile_count = cache_size/tile_size;

    if( count > 0 && cache_size > tile_size ){
        cache_info[level-1] = (cacheInfo_t *)calloc(count, sizeof(cacheInfo_t));
        cache_count[level-1] = count;

        cache_tile_capacity[level-1] = tile_count;
        for(i=0; i<count; ++i){
            cache_info[level-1][i].available = cache_size;
            pthread_mutex_init( &(cache_info[level-1][i].mutex), NULL );
            cache_info[level-1][i].arrays_start = calloc( tile_count, sizeof(arrayInfo_t) );
            cache_info[level-1][i].arrays_head = cache_info[level-1][i].arrays_start;
            // The following shouldn't be necessary since we calloc()ed, but let's be paranoid.
            cache_info[level-1][i].arrays_tail = NULL;
        }
    }

    return;
}



void dplasma_hwloc_insert_buffer(void *array_ptr, int bufSize, int myPUID){
    cacheInfo_t *curr;
    arrayInfo_t *ptr, *new_array;

    int level;

    for(level=0; level<3; ++level){
        // If this cache can't fit nuten, skip it.
        if( cache_count[level] == 0 )
            continue;

        int indx = (myPUID*cache_count[level])/pu_count;
        curr = &cache_info[level][indx];
        pthread_mutex_lock( &(curr->mutex) );

        // If we haven't inserted any element already
        if( curr->arrays_head->array == NULL ){
            curr->arrays_head->array = array_ptr;
            curr->arrays_head->size = bufSize;
            curr->arrays_head->next = NULL;
            curr->arrays_head->prev = NULL;
            curr->arrays_tail = curr->arrays_head;
            curr->available -= bufSize;
            // jump to the next iteration
            pthread_mutex_unlock( &(curr->mutex) );
            continue;
        }

        // if the array is already in the cache (and it's not already the head), 
        // just move the pointers around.
//        for(ptr = curr->arrays_head; ptr != curr->arrays_tail; ptr = ptr->next){
        int i=0;
        for(ptr = curr->arrays_start; i<cache_tile_capacity[level] ; ++i, ++ptr){
            if( ptr->array == array_ptr )
                break;
        }
        if( (ptr->array == array_ptr) && (ptr != curr->arrays_head) ){
            if( ptr->next != NULL ){
                ptr->next->prev = ptr->prev;
            }else{
                if( ptr->prev != NULL )
                    curr->arrays_tail = ptr->prev;
            }

            if( ptr->prev != NULL ){
                ptr->prev->next = ptr->next;
            }
            ptr->prev = NULL;
            ptr->next = curr->arrays_head;
            curr->arrays_head->prev = ptr;
            curr->arrays_head = ptr;
            // jump to the next iteration
            pthread_mutex_unlock( &(curr->mutex) );
            continue;
        }

        if( curr->available > bufSize ){
            // If the cache is not full, just add the new array at the first empty spot
            // (let the compiler do the pointer arithmetic).
            for(ptr = curr->arrays_head; ptr->array != NULL; ++ptr);

            new_array = ptr;
            curr->available -= bufSize;
        }else{
            // If the cache is full, we will evict the tail by putting the new element there
            new_array = curr->arrays_tail;
            // The element before the tail (if there is such an element) is pointing to the tail
            // as its "next" element.  Now that we are evicting the tail element, we should clean
            // that "next" pointer.
            curr->arrays_tail->prev->next = NULL;
            // Make the element we just cleaned the new tail (since it was the "prev" of the
            // old tail). Don't worry about losing the pointer, we have it in "new_array".
            curr->arrays_tail = curr->arrays_tail->prev;
        }

        new_array->array = array_ptr;
        new_array->size = bufSize;
        // Since the new element is the new head, what used to be the head
        // becomes the new head's next element.
        new_array->next = curr->arrays_head;
        // The new head becomes the old head's previous element.
        curr->arrays_head->prev = new_array;;
        // No element is before the head, so no "prev" pointer.
        new_array->prev = NULL;
        // Update the head pointer (we updated the tail already).
        curr->arrays_head = new_array;

        pthread_mutex_unlock( &(curr->mutex) );
    }
}

int dplasma_hwloc_isLocal(void *array_ptr, int cacheLevel, int myPUID){
    int i;
    cacheInfo_t *curr;
    arrayInfo_t *tmp;// *end;

    if( cache_count[cacheLevel-1] == 0 )
        return 0;

    int indx = (myPUID*cache_count[cacheLevel-1])/pu_count;
    curr = &cache_info[cacheLevel-1][indx];

//    pthread_mutex_lock( &(curr->mutex) );
//    for(tmp = curr->arrays_head; tmp != curr->arrays_tail; tmp = tmp->next){
    for(i=0, tmp = curr->arrays_start; i<cache_tile_capacity[cacheLevel-1] ; ++i, ++tmp){
        if( tmp->array == array_ptr ){
//            pthread_mutex_unlock( &(curr->mutex) );
            return 1;
        }
    }
    // Compare against the tail as well
//    if( end != NULL ){
//        if( tmp->array == array_ptr ){
//            pthread_mutex_unlock( &(curr->mutex) );
//            return 1;
//        }
//    }
//
//    pthread_mutex_unlock( &(curr->mutex) );
    return 0;
}

