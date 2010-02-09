#include "buf2cache2.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// function forward declarations

static void *do_cache_buf_referenced(cache_t *cache, void *tile_ptr, int *succeeded);


///////////////////////////////////////////////////////////////////////////////////////////////
// function code

//--------------------------------------------------------------------------------
cache_t *cache_create(int core_count, cache_t *parent, int tile_capacity){

    cache_t *cache = (cache_t *)calloc( 1, sizeof(cache_t) );

    cache->tile_capacity = tile_capacity;
    cache->parent = parent;
    cache->entries = (cache_entry_t *)calloc( tile_capacity, sizeof(cache_entry_t) );

    return cache;
}


//--------------------------------------------------------------------------------
// This function returns the element we evicted (the tile pointer) or NULL if nothing was evicted.
void *cache_buf_referenced(cache_t *cache, void *tile_ptr){
    int succeeded = 0;
    void *old_tile_ptr;

    if( cache == NULL )
        return NULL;

    do{
        old_tile_ptr = do_cache_buf_referenced(cache, tile_ptr, &succeeded);
    }while(!succeeded);

    // Update all the parent caches as well
    cache_buf_referenced(cache->parent, tile_ptr);

    return old_tile_ptr;
}


//--------------------------------------------------------------------------------
// This function does the actual job. It sets its last argument, "succeeded", to "0" if
// it can get the lock (or if the queue changes in the middle of a search) so it can be
// called again by the caller "cache_buf_referenced()".
//
// The ordering of the elements is defined by their age. The smaller the age value, the
// younger the element.  Therefore when we add a new (or reference again an existing)
// element, we set its age value to the minimum age value minus one. Ages are always
// negative numbers.
//
static void *do_cache_buf_referenced(cache_t *cache, void *tile_ptr, int *succeeded){
    cache_entry_t *cur, *oldest_elem = NULL, *found_elem = NULL;
    int i, N;
    int64_t oldest_age, youngest_age;

    N = cache->tile_capacity;
    oldest_age = 1;
    youngest_age = 0;

    // First find the youngest and oldest elements, and maybe the element we are looking for
    for( i=0, cur = cache->entries; i<N; ++cur, ++i){
        // If the element is locked, wait until the change has finished.
        while( cur->lock == 1 );

        // if we hit an empty element, stop looking
        if( cur->tile_ptr == NULL ){
            break;
        }

        // keep track of the oldest element, we might need to evict it.
        if( (oldest_age == 1) || (cur->age > oldest_age) ){
            oldest_age = cur->age;
            oldest_elem = cur;
        }

        // remember the age of the youngest element.
        if( cur->age < youngest_age ){
            youngest_age = cur->age;
        }

        // We do not break when we find the element because we are also looking for the overall
        // youngest and oldest elements.
        if( cur->tile_ptr == tile_ptr ){
            found_elem = cur;
        }
    }

    // if the element already exists and nobody changed it, just make it the youngest.
    if( found_elem && (found_elem->tile_ptr == tile_ptr) ){
        if( dplasma_atomic_cas_xxb(&(found_elem->lock), 0, 1, sizeof(found_elem->lock)) ){
            // Ok we got the lock, now we have to make sure that no other thread changed
            // the pointer between if(found_elem->tile_ptr == tile_ptr) and the atomic
            if( found_elem->tile_ptr == tile_ptr ){
                // Here I can safely update the age, because every other thread has lost the race.
                // If another thread updated some other element to youngest_age-1 between
                // setting the variable "youngest" and here, more than one elements will have
                // the same age.  That makes the replacement policy an imperfect LRU, but reduces
                // waiting on locks, which should speed up performance.
                found_elem->age = youngest_age-1;
                // release the lock
                found_elem->lock = 0;
                // notify success to the caller
                *succeeded = 1;
                return NULL;
            }else{
                // If the pointer was changed, release the lock and go search the queue again
                found_elem->lock = 0;
                // notify failure to the caller
                *succeeded = 0;
                return NULL;
            }
        }else{
            // I couldn't get a lock, so somebody is changing the queue, so I must
            // traverse it again.
            // notify failure to the caller
            *succeeded = 0;
            return NULL;
        }
    }

    // If we didn't find it, but there is still room in the queue, append the info at the end
    if( i<N ){
        if( dplasma_atomic_cas_xxb(&(cur->lock), 0, 1, sizeof(cur->lock)) ){
            // If this element is still empty
            if( cur->tile_ptr == NULL ){
                cur->age = youngest_age-1;
                cur->tile_ptr = tile_ptr;
                // release the lock
                cur->lock = 0;
                // notify success to the caller
                *succeeded = 1;
                return NULL;
            }else{
                // release the lock
                cur->lock = 0;
                // notify failure to the caller
                *succeeded = 0;
                return NULL;
            }
        }else{ // otherwise the queue has changed, so we need to search all over again.
            // notify failure to the caller
            *succeeded = 0;
            return NULL;
        }
    }

    // If the queue is full, replace (evict) the oldest element, unless some other thread
    // has changed it already.
    if( dplasma_atomic_cas_xxb(&(oldest_elem->lock), 0, 1, sizeof(oldest_elem->lock)) ){
        // If the element hasn't changed, evict it.
        if( oldest_elem->age == oldest_age ){
            void *old_ptr = oldest_elem->tile_ptr;
            oldest_elem->tile_ptr = tile_ptr;
            oldest_elem->age = youngest_age-1;
            // release the lock
            oldest_elem->lock = 0;
            // notify success to the caller
            *succeeded = 1;
            return old_ptr;
        }else{
            // release the lock
            oldest_elem->lock = 0;
            // notify failure to the caller
            *succeeded = 0;
            return NULL;
        }
    }

    // notify failure to the caller
    *succeeded = 0;
    return NULL;
}


//--------------------------------------------------------------------------------
int cache_buf_isLocal(cache_t *cache, void *tile_ptr){
    int i, N;
    cache_entry_t *cur;
    N = cache->tile_capacity;

    for( i=0, cur = cache->entries; i<N; ++cur, ++i){
        // If the element is locked, wait until the change has finished.
        while( cur->lock == 1 );

        if( cur->tile_ptr == tile_ptr )
            return 1;

        // if we hit an empty element, stop looking
        if( cur->tile_ptr == NULL )
            break;
    }
    return 0;
}


//--------------------------------------------------------------------------------
int cache_buf_distance(cache_t *cache, void *tile_ptr){
    int i=0;
    cache_t *cur_cache;
    for(cur_cache=cache; cur_cache != NULL; cur_cache=cur_cache->parent ){ 
        if( cache_buf_isLocal(cur_cache, tile_ptr) )
            break;
        ++i;
    }
    return i;
}


//--------------------------------------------------------------------------------
// Return the age of a tile.  "Age" in this context means a relative ordering
// of the element in comparison to other elements.  That is, the number of
// elements that are younger than this element.
// If the element is not found, the function returns "-1"
int cache_buf_age(cache_t *cache, void *tile_ptr){
    int64_t abs_age=1;
    int i, N, ret_val=0;
    cache_entry_t *cur;
    N = cache->tile_capacity;

    for( i=0, cur = cache->entries; i<N; ++cur, ++i){
        // If the element is locked, wait until the change has finished.
        while( cur->lock == 1 );

        if( cur->tile_ptr == tile_ptr )
            abs_age = cur->age;

        // if we hit an empty element, stop looking
        if( cur->tile_ptr == NULL )
            break;
    }

    // If we didn't find the element, return an error value
    if( abs_age == 1 )
        return -1;

    ret_val = 0;
    for( i=0, cur = cache->entries; i<N; ++cur, ++i){
        // If the element is locked, wait until the change has finished.
        while( cur->lock == 1 );

        if( cur->age < abs_age )
            ret_val++;

        // if we hit an empty element, stop looking
        if( cur->tile_ptr == NULL )
            break;
    }

    return ret_val;
}
