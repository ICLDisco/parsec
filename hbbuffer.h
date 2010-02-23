#ifndef HBBUFFER_H_HAS_BEEN_INCLUDED
#define HBBUFFER_H_HAS_BEEN_INCLUDED

#include "dplasma.h"
#include "atomic.h"

/**
 * Hierarchical Bounded Buffers:
 *
 *   bounded buffers with a parent storage, to store elements
 *   that will be ejected from the current buffer at push time.
 */

/**
 * ranking function: takes an element that was stored in the buffer,
 * and serves as input to the pop_best function.
 * pop_best will pop the first element it finds in the bounded buffer
 * that has the highest score with this ranking function
 */
typedef unsigned int (*dplasma_hbbuffer_ranking_fct_t)(void *elt, void *param);

/** 
 * parent push function: takes a pointer to the parent store object, and
 * a pointer to the element that is ejected out of this bounded buffer because
 * of a push. elt must be stored in the parent store (linked list, hbbuffer, or
 * dequeue, etc...) before the function returns
 */
typedef void (*dplasma_hbbuffer_parent_push_fct_t)(void *store, void *elt);

typedef struct dplasma_hbbuffer_t {
    size_t size;       /**< the size of the buffer, in number of void* */
    uint32_t w_pos;    /**< the position to push the next element in the buffer */
    uint32_t lock;     /**< lock on the buffer */
    void    *parent_store; /**< pointer to this buffer parent store */
    /** function to push element to the parent store */
    dplasma_hbbuffer_parent_push_fct_t parent_push_fct;
    void    *items[1]; /**< array of elements */
} dplasma_hbbuffer_t;

static inline dplasma_hbbuffer_t *dplasma_hbbuffer_new(size_t size, 
                                                       dplasma_hbbuffer_parent_push_fct_t parent_push_fct,
                                                       void *parent_store)
{
    /** Must use calloc to ensure that all ites are set to NULL */
    dplasma_hbbuffer_t *n = (dplasma_hbbuffer_t*)calloc(1, sizeof(dplasma_hbbuffer_t) + (size-1)*sizeof(void*));
    n->size = size;
    /** Not needed since using calloc 
     *  n->w_pos = 0;
     *  n->version = 0;
     */
    n->parent_push_fct = parent_push_fct;
    n->parent_store = parent_store;
    return n;
}

static inline void dplasma_hbbuffer_destroy(dplasma_hbbuffer_t *b)
{
    free(b);
}

static inline void dplasma_hbbuffer_push(dplasma_hbbuffer_t *b, volatile dplasma_list_item_t *elt)
{
    void *victim;
    volatile dplasma_list_item_t *n;
    int i = 0;

    while( elt ) {
        n = elt->list_next;
        if( n == elt ) {
            n = NULL;
        }

        elt->list_next->list_prev = elt->list_prev;
        elt->list_prev->list_next = elt->list_next;

        elt->list_prev = elt;
        elt->list_next = elt;

        DEBUG(("trying to push %p in %p\n", elt, b));

        dplasma_atomic_lock(&b->lock);
        victim = b->items[b->w_pos];
        b->items[b->w_pos] = (dplasma_list_item_t*)elt;
        b->w_pos = (b->w_pos + 1) % b->size;
        dplasma_atomic_unlock(&b->lock);
        if( NULL != victim ) {
            DEBUG(("%p is full -> moving %p to %p\n", b, victim, b->parent_store));
            b->parent_push_fct(b->parent_store, victim);
        }

        elt = n;
        i++;
    }
    DEBUG(("pushed %d elements\n", i));
}

static inline int dplasma_hbbuffer_is_empty(dplasma_hbbuffer_t *b)
{
    int ret = 1, i;
    dplasma_atomic_lock(&b->lock);
    for(i = 0; i < b->size; i++) {
        if( NULL != b->items[i] ) {
            ret = 0;
            break;
        }
    }
    dplasma_atomic_unlock(&b->lock);
    return ret;
}

static inline void *dplasma_hbbuffer_pop_best(dplasma_hbbuffer_t *b, 
                                              dplasma_hbbuffer_ranking_fct_t rank_function, 
                                              void *rank_function_param)
{
    int idx;
    void *best_elt = NULL;
    int best_idx = 0;   
    unsigned int best_rank = 0, rank;

    dplasma_atomic_lock(&b->lock);
    for(idx = 0; idx != b->size; idx++) {
        if( NULL == b->items[idx] )
            continue;

        rank = rank_function(b->items[idx], rank_function_param);
        if( NULL == best_elt ||
            rank > best_rank ) {
            best_rank = rank;
            best_elt  =  b->items[idx];
            best_idx  = idx;
        }
    }
    /** Removes the element from the buffer.
     *  If no element is found, b->items[0] == NULL already */
    b->items[best_idx] = NULL;
    dplasma_atomic_unlock(&b->lock);

    DEBUG(("pop best %p from %p\n", best_elt, b));

    return best_elt;
}

#endif /* HBBUFFER_H_HAS_BEEN_INCLUDED */
