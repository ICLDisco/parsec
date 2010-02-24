#ifndef HBBUFFER_H_HAS_BEEN_INCLUDED
#define HBBUFFER_H_HAS_BEEN_INCLUDED

#include "dplasma.h"
#include "atomic.h"
#include "lifo.h"

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
typedef unsigned int (*dplasma_hbbuffer_ranking_fct_t)(dplasma_list_item_t *elt, void *param);

/** 
 * parent push function: takes a pointer to the parent store object, and
 * a pointer to the element that is ejected out of this bounded buffer because
 * of a push. elt must be stored in the parent store (linked list, hbbuffer, or
 * dequeue, etc...) before the function returns
 */
typedef void (*dplasma_hbbuffer_parent_push_fct_t)(void *store, dplasma_list_item_t *elt);

typedef struct dplasma_hbbuffer_t {
    size_t size;       /**< the size of the buffer, in number of void* */
	size_t nbelt;      /**< Number of elemnts in the buffer currently */
    volatile uint32_t lock;     /**< lock on the buffer */
    void    *parent_store; /**< pointer to this buffer parent store */
    /** function to push element to the parent store */
    dplasma_hbbuffer_parent_push_fct_t parent_push_fct;
    dplasma_list_item_t *items[1]; /**< array of elements */
} dplasma_hbbuffer_t;

static inline dplasma_hbbuffer_t *dplasma_hbbuffer_new(size_t size, 
                                                       dplasma_hbbuffer_parent_push_fct_t parent_push_fct,
                                                       void *parent_store)
{
    /** Must use calloc to ensure that all ites are set to NULL */
    dplasma_hbbuffer_t *n = (dplasma_hbbuffer_t*)calloc(1, sizeof(dplasma_hbbuffer_t) + (size-1)*sizeof(dplasma_list_item_t*));
    n->size = size;
	/** n->nbelt = 0; <not needed because callc */
    n->parent_push_fct = parent_push_fct;
    n->parent_store = parent_store;
    DEBUG(("Created a new hierarchical buffer of %d elements\n", size));
    return n;
}

static inline void dplasma_hbbuffer_destroy(dplasma_hbbuffer_t *b)
{
    free(b);
}

static inline void dplasma_hbbuffer_push(dplasma_hbbuffer_t *b, dplasma_list_item_t *elt)
{
    dplasma_list_item_t *next;
    int nbelt, i;

    nbelt = 0;
    next = elt;
    dplasma_atomic_lock(&b->lock);
    for(i = 0; (i < b->size) && (NULL != elt); i++) {
        if( NULL != b->items[i] )
            continue;

        next = (dplasma_list_item_t *)elt->list_next;
        if(next == elt) {
            next = NULL;
        }
        elt->list_next->list_prev = elt->list_prev;
        elt->list_prev->list_next = elt->list_next;
        elt->list_prev = elt;
        elt->list_next = elt;
        DEBUG(("Pushing %p in %p\n", elt, b));
        b->items[i] = elt;
        nbelt++;

        elt = next;
    }
	b->nbelt += nbelt;
    dplasma_atomic_unlock(&b->lock);

    DEBUG(("pushed %d elements. %s\n", nbelt, NULL != elt ? "More to push, go to father" : "Everything pushed - done"));

    if( NULL != next ) {
        b->parent_push_fct(b->parent_store, next);
    }
}

static inline int dplasma_hbbuffer_is_empty(dplasma_hbbuffer_t *b)
{
    int ret = 1, i;
    dplasma_atomic_lock(&b->lock);
	ret = (b->nbelt == 0);
    dplasma_atomic_unlock(&b->lock);
    return ret;
}

static inline dplasma_list_item_t *dplasma_hbbuffer_pop_best(dplasma_hbbuffer_t *b, 
                                                             dplasma_hbbuffer_ranking_fct_t rank_function, 
                                                             void *rank_function_param)
{
    int idx;
    dplasma_list_item_t *best_elt = NULL;
    int best_idx = 0;   
    unsigned int best_rank = 0, rank;

    dplasma_atomic_lock(&b->lock);
    for(idx = 0; idx < b->size; idx++) {
        if( NULL == b->items[idx] )
            continue;

        DEBUG(("Found non NULL element in %p at position %d/%d\n", b, idx, (int)b->size));

        rank = rank_function(b->items[idx], rank_function_param);
        if( (NULL == best_elt) || (rank > best_rank) ) {
            best_rank = rank;
            best_elt  =  b->items[idx];
            best_idx  = idx;
        }
    }
    /** Removes the element from the buffer. */
	if( best_elt != NULL ) {
      b->items[best_idx] = NULL;
	  b->nbelt--;
	}
    dplasma_atomic_unlock(&b->lock);

    DEBUG(("pop best %p from %p\n", best_elt, b));

    return best_elt;
}

#endif /* HBBUFFER_H_HAS_BEEN_INCLUDED */
