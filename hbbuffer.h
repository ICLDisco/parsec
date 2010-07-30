#ifndef HBBUFFER_H_HAS_BEEN_INCLUDED
#define HBBUFFER_H_HAS_BEEN_INCLUDED

#include "debug.h"
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
 * 
 * @return an integer value for the element. Bigger is better.
 *         DPLASMA_RANKING_FUNCTION_BEST means that no other element 
 *         can be better than this element.
 */
#define DPLASMA_RANKING_FUNCTION_BEST 0xffffff
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
    size_t ideal_fill; /**< hint on the number of elements that should be there to increase parallelism */
    void    *parent_store; /**< pointer to this buffer parent store */
    /** function to push element to the parent store */
    dplasma_hbbuffer_parent_push_fct_t parent_push_fct;
    volatile dplasma_list_item_t *items[1]; /**< array of elements */
} dplasma_hbbuffer_t;

static inline dplasma_hbbuffer_t *dplasma_hbbuffer_new(size_t size,  size_t ideal_fill,
                                                   dplasma_hbbuffer_parent_push_fct_t parent_push_fct,
                                                   void *parent_store)
{
    /** Must use calloc to ensure that all ites are set to NULL */
    dplasma_hbbuffer_t *n = (dplasma_hbbuffer_t*)calloc(1, sizeof(dplasma_hbbuffer_t) + (size-1)*sizeof(dplasma_list_item_t*));
    n->size = size;
    n->ideal_fill = ideal_fill;
	/** n->nbelt = 0; <not needed because callc */
    n->parent_push_fct = parent_push_fct;
    n->parent_store = parent_store;
    DEBUG(("Created a new hierarchical buffer of %d elements\n", (int)size));
    return n;
}

static inline void dplasma_hbbuffer_destroy(dplasma_hbbuffer_t *b)
{
    free(b);
}

static inline void dplasma_hbbuffer_push_all(dplasma_hbbuffer_t *b, dplasma_list_item_t *elt)
{
    dplasma_list_item_t *next;
    int i, nbelt;


    nbelt = 0;
    next = elt;
    while( NULL != elt ) {
        /* Assume that we're going to push elt.
         * Remove the first element from the list, keeping the rest of the list in next
         */
        next = (dplasma_list_item_t *)elt->list_next;
        if(next == elt) {
            next = NULL;
        }
        elt->list_next->list_prev = elt->list_prev;
        elt->list_prev->list_next = elt->list_next;

        elt->list_prev = elt;
        elt->list_next = elt;

        /* Try to find a room for elt */
        for(i = 0; i < b->size; i++) {
            if( dplasma_atomic_cas(&b->items[i], NULL, elt) == 0 )
                continue;

            /* Found an empty space to push the first element. */
            nbelt++;
            break;
        }

        if( i == b->size ) {
            /* It was impossible to push elt */
            break;
        }

        elt = next;
    }

    DEBUG(("pushed %d elements. %s\n", nbelt, NULL != elt ? "More to push, go to father" : "Everything pushed - done"));

    if( elt != NULL ) {

        if( NULL != next ) {
            /* Rechain elt to next */
            elt->list_next = next;
            elt->list_prev = next->list_prev;
            next->list_prev->list_next = elt;
            next->list_prev = elt;
        }

        b->parent_push_fct(b->parent_store, elt);
    }
}

/* This code is unsafe, since another thread may be inserting new elements.
 * Use is_empty in safe-checking only 
 */
static inline int dplasma_hbbuffer_is_empty(dplasma_hbbuffer_t *b)
{
    int i;
    for(i = 0; i < b->size; i++)
        if( NULL != b->items[i] )
            return 0;
    return 1;
}

static inline dplasma_list_item_t *dplasma_hbbuffer_pop_best(dplasma_hbbuffer_t *b, 
                                                         dplasma_hbbuffer_ranking_fct_t rank_function, 
                                                         void *rank_function_param)
{
    int idx;
    dplasma_list_item_t *best_elt = NULL;
    int best_idx = -1;
    unsigned int best_rank = 0, rank;
    dplasma_list_item_t *candidate;
    
    do {
        best_elt = NULL;
        best_idx = -1;
        best_rank = 0;

        for(idx = 0; idx < b->size; idx++) {
            if( NULL == (candidate = (dplasma_list_item_t *)b->items[idx]) )
                continue;

            rank = rank_function(candidate, rank_function_param);
            if( (NULL == best_elt) || (rank == DPLASMA_RANKING_FUNCTION_BEST) || (rank > best_rank) ) {
                best_rank = rank;
                best_elt  = candidate;
                best_idx  = idx;

                if( DPLASMA_RANKING_FUNCTION_BEST == rank )
                    break;
            }
        }
        
        if( NULL == best_elt)
            break;

    } while( dplasma_atomic_cas( &b->items[best_idx], best_elt, NULL ) == 0 );

    /** Removes the element from the buffer. */
	if( best_elt != NULL ) {
        DEBUG(("Found best element at position %d\n", best_idx));
	}

    return best_elt;
}

#endif /* HBBUFFER_H_HAS_BEEN_INCLUDED */
