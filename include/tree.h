#ifndef TREE_H_HAS_BEEN_INCLUDED
#define TREE_H_HAS_BEEN_INCLUDED

#include "dague.h"
#include "dague_config.h"

#include "debug.h"
#include "atomic.h"
#include "lifo.h"
#include <stdlib.h>

static int max(int lhs, int rhs);
static int height(dague_avltree tree);
static int elem_gt(dague_exection_context_t l_ec, dague_exection_context_t r_ec);
static dague_avltree node_cmp(dague_avltree l_tree, dague_avltree r_tree);
dague_avltree emptyTree(dague_avltree tree);
dague_avltree insert(dague_execution_context_t * ec, dague_avltree tree);

// NOTE: may want to save space by having a tree head node that contains the
// list_item thing.
// but for now i'm ignoring this
typedef struct avl_head {
	dague_list_item_t list_item;
	avl_node_t *      head;
} avl_head_t;


typedef struct avl_node {
	dague_list_item_t          list_item; // so that the tree can be a member of an hbbuffer
	avl_node_t                 left;
	avl_node_t                 right;
	int                        height; // keep this up-to-date during insertion
	dague_execution_context_t *elem;
} avl_node_t;

typedef struct dague_avltree *avl_node_t;

// with preference to right hand side if equal
static int max(int lhs, int rhs) {
	return lhs > rhs ? lhs : rhs;
}

static int height(dague_avltree tree) {
	if (tree != NULL)
		return tree->height;
	else
		return 0;
}

static int elem_gt(dague_exection_context_t l_ec, dague_exection_context_t r_ec) {
	return l_ec->dague_object > r_ec->dague_object;
}

static dague_avltree node_cmp(dague_avltree l_tree, dague_avltree r_tree) {
	return elem_gt(l_tree->elem, r_tree->elem) ? l_tree : r_tree;
}

// returns a NULL pointer, so that its return value can be used in assignment
dague_avltree emptyTree(dague_avltree tree) {
	if (tree != NULL) {
		emptyTree(tree->left);
		emptyTree(tree->right);
		free(ec); // should this happen or not?
		free(tree);
	}
	return NULL;
}


// notes:
/*
  Tree will need to act as a standard AVL tree on insertion,
  but not rebalance during removal.
  when removing one at a time, should we remove
  by row (i.e. looking for highest-valued leaf perhaps?),
  or by value (looking for highest-valued node, period)?
 */

dague_avltree insert(dague_execution_context_t * ec, dague_avltree tree) {
	// make a new node
	if (tree == NULL) {
		tree = malloc(sizeof(avl_node_t));
		tree->left = NULL;
		tree->right = NULL;
		tree->height = 1;
		tree->elem = ec;
	}
	else {
		dague_avltree newNode = NULL;
		if (elem_gt(ec, tree->elem)) {
			newNode = insert(ec, tree->right);
			if (tree->right == NULL)
				tree->right = newNode;
		}
		else {
			newNode = insert(ec, tree->left);
			if (tree->left == NULL)
				tree->left = newNode;
		}
		// no balancing so far
		return newNode;
	}
}

dague_avltree steal_half(dague_avltree tree) {
	dague_avltree save;
	if (tree->left != NULL) {
		save = tree->left;
		tree->left = NULL;
	}
	else
		save = steal_half(tree->right);

	return save;
}

dague_execution_context_t remove_single_elem(dague_avltree tree) {

}

#endif
