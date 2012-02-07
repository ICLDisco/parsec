#ifndef MAXHEAP_H_HAS_BEEN_INCLUDED
#define MAXHEAP_H_HAS_BEEN_INCLUDED

#include "dague.h"
#include "dague_config.h"

#include "debug.h"
#include "atomic.h"
#include "lifo.h"
#include <stdlib.h>

#define H_LEFT 0
#define H_RIGHT 1

typedef struct heap_node_t {
	struct heap_node_t* leaf[2];
	dague_execution_context_t* elem;
} heap_node_t;

// main struct holding size info and ID
typedef struct dague_heap {
	dague_list_item_t list_item; // to be compatible with the lists
	int size; // used only during building
	int id;   // used only during building
	heap_node_t * top;
	dague_execution_context_t * to_use;
} dague_heap_t;

void heap_destroy(dague_heap_t* heap);
dague_heap_t* heap_create(int id);
dague_heap_t* heap_insert(dague_heap_t * heap, dague_execution_context_t * elem);
dague_heap_t* heap_split_and_steal(dague_heap_t * heap);
static void emptyHeap(heap_node_t* node);

void heap_destroy(dague_heap_t* heap) {
	if (heap != NULL) {
		if (heap->top != NULL) {
			emptyHeap(heap->top);
			free(heap->top);
		}
		if (heap->to_use != NULL) {
			printf("oh, that's not a good thing at all.... to_use of %x isn't NULL!\n", heap);
		}
		free(heap);
		heap = NULL;
	}
}

// private version
static void emptyHeap(heap_node_t* node) {
	if (node != NULL) {
		emptyHeap(node->leaf[H_RIGHT]);
		node->leaf[H_RIGHT] = NULL;
		emptyHeap(node->leaf[H_LEFT]);
		node->leaf[H_LEFT] = NULL;
		if (node->elem != NULL)
			printf("ERROR: element of heap node is NOT NULL during heap destruction!\n");
		free(node);
	}
}

dague_heap_t* heap_create(int id) {
	dague_heap_t* heap = calloc(sizeof(dague_heap_t), 1); // makes sure everything is zeroed
	heap->id = id;
	heap->list_item.list_next = heap;
	heap->list_item.list_prev = heap;
	return heap;
}

// not remotely thread-safe
/*
 * Insertion is O(lg n), as we know exactly how to get to the next insertion point,
 * and the tree is manually balanced.
 * Overall build is O(n lg n)
 */
dague_heap_t * heap_insert(dague_heap_t * heap, dague_execution_context_t * elem) {
	heap_node_t * node = malloc(sizeof(heap_node_t));
	node->leaf[H_LEFT] = NULL;
	node->leaf[H_RIGHT] = NULL;
	node->elem = elem;
	// now find the place to put it
	heap->size++;
	if (heap == NULL)
		printf("heap is NULL!\n");
	else if (heap->size == 1) 
		heap->top = node;
	else {
		heap_node_t * parent = heap->top;
		unsigned int bitmask = 1;
		int size = heap->size;
		// prime the bitmask
		int level_counter = 0;
		int parents_size = 0;
		while (bitmask <= size) {
			bitmask = bitmask << 1;
			level_counter++;
		}
		parents_size = level_counter;

		heap_node_t ** parents = calloc(sizeof(heap_node_t *), level_counter);
		// now the bitmask is two places farther than we want it, so back down
		bitmask = bitmask >> 2;
		
		parents[--level_counter] = heap->top;;
		// now move through tree
		while (bitmask > 1) {
			parent = parent->leaf[(bitmask & size) ? H_RIGHT : H_LEFT];
			parents[--level_counter] = parent; // save parent
			bitmask = bitmask >> 1;
		}
		parent->leaf[bitmask & size] = node;
		// now bubble up to preserve max heap org.
		if (parents[level_counter]->elem  == NULL)
			printf("elem is NULL!\n");
		while (level_counter < parents_size && 
				 parents[level_counter] != NULL && 
				 node->elem->priority > parents[level_counter]->elem->priority) {
			// make swap
			dague_execution_context_t * temp = node->elem;
			node->elem = parents[level_counter]->elem;
			parents[level_counter]->elem = temp;
			level_counter++;
		}
	}

	return heap;
}

// not remotely thread-safe
// but tolerant of NULLs
/*
 * split-and-steal (remove) is O(1), although the preceding
 * list search is probably O(n), technically, since eventually we
 * end up with a list of n/2 trees with single nodes
 */
dague_heap_t* heap_split_and_steal(dague_heap_t * heap) {
	// if tree is empty, return NULL
	// if tree has only one node (top), return new heap with single node
	//    moved into to_use slot
	// if tree has left child but not right child, put left child in new tree
	//    
	if (heap == NULL || heap->top == NULL) {
		if (heap != NULL)
			printf("this heap (%x) should have been deleted...\n", heap);
		return NULL; // this heap should be deleted
	}
	if (heap->top->leaf[H_LEFT] == NULL) {
		heap->to_use = heap->top->elem;
		heap->top->elem = NULL;
		free(heap->top);
		heap->top = NULL;
		heap->size = 0;

		return heap;
	}
	else {
		if (heap->top->leaf[H_RIGHT] == NULL) {
			heap_node_t* temp = heap->top->leaf[H_LEFT];
			heap->to_use = heap->top->elem;
			heap->top->elem = NULL;
			free(heap->top);
			heap->top = temp;
			heap->size--;

			return heap;
		}
		else { // heap has at least 3 nodes
			dague_heap_t* new_heap = heap_create(0); // ID is unimportant
			heap_node_t* temp = heap->top->leaf[H_RIGHT];
			new_heap->top = heap->top->leaf[H_LEFT];
			new_heap->to_use = heap->top->elem;
			heap->top->elem = NULL;
			free(heap->top);
			heap->top = temp;

			return new_heap;
		}
	}
}

#endif
