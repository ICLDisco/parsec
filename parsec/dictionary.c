/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"

#include "parsec/mca/mca.h"
#include "parsec/utils/mca_param.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/execution_stream.h"

#include "parsec/dictionary.h"

#define dict parsec_profiling_dictionary
#define PROFILING_HASH_SIZE 4096

static size_t     sizeof_for_type(parsec_profiling_datatype_t dt);
static char       code_for_type(  parsec_profiling_datatype_t dt);

static parsec_hash_table_item_t *create_pr_bucket(char *pr);
static parsec_hash_table_item_t *create_tc_bucket(char *tc);
static parsec_hash_table_item_t *create_ns_bucket(char *ns);
static parsec_profiling_namespace_t *find_or_insert_ns(parsec_hash_table_t *ht, char *ns);
static parsec_profiling_task_class_t  *find_or_insert_tc(parsec_hash_table_t *ht, char *tc);
static parsec_profiling_property_t  *find_or_insert_pr(parsec_hash_table_t *ht, char *pr);

static void       dump_property(   void *item, void *cb_data);
static void       dump_task_class( void *item, void *cb_data);
static void       dump_namespace(  void *item, void *cb_data);
static void       namespace_print( void *item, void *cb_data);
static void       task_class_print(void *item, void *cb_data);
static void       property_print(  void *item, void *cb_data);

static void       parsec_profiling_tree_reload_buckets(parsec_profiling_tree_t *tree, int pos, const char *new);
static int        parsec_profiling_dictionary_update(void);
static parsec_hash_table_item_t *parsec_profiling_tree_add_missing_buckets(parsec_profiling_tree_t *tree);
static int        parsec_profiling_tree_look_for_path(parsec_profiling_tree_t *tree);
static void       parsec_profiling_tree_recursive_delete(parsec_profiling_node_t *node);
static void       parsec_profiling_tree_delete(parsec_profiling_tree_t *tree);

static int        parsec_profiling_update_xml_header(void);
static int        parsec_profiling_change_shmem(void);
static int        parsec_profiling_dump_header_shmem(void);


static int dict_key_compare(parsec_key_t keyA, parsec_key_t keyB, void *data)
{
    (void)data;
    return strcmp((char*)keyA, (char*)keyB);
}

static char * dict_key_print(char *buffer, size_t buffer_size,
			     parsec_key_t k, void *data) {
  (void)data;
  char *key = (char*)k;
  snprintf(buffer, buffer_size, "<%s>", key);
  return buffer;
}

static uint64_t dict_key_hash(parsec_key_t key, void *data)
{
    (void)data;
    char *str = (char*)key;
    uint64_t h = 1125899906842597ULL; // Large prime
    int len = strlen(str);

    for (int i = 0; i < len; i++) {
        h = 31*h + str[i];
    }
    return h;
}

static parsec_key_fn_t dict_key_fns = {
    .key_compare = dict_key_compare,
    .key_print = dict_key_print,
    .key_hash = dict_key_hash
};

static void parsec_profiling_chain_siblings(parsec_profiling_tree_t *tree)
{
    parsec_profiling_node_t *head = tree->root;
    parsec_profiling_node_t *tail = tree->root;
    while (head) {
	if (head->left) {
	    tail->next_sibling = head->left;
	    tail = head->left;
	}
	if (head->right) {
	    tail->next_sibling = head->right;
	    tail = head->right;
	}
	head = head->next_sibling;
    }

    parsec_profiling_node_t *node = tree->root;
    while (node) { /* the next_sibling chain ends up with a NULL on each row */
	node->next_sibling = NULL;
	node = node->right;
    }
    int i = 0;
    node = tree->root;
    while (node) { /* we store the first node of the next_sibling chain for each row */
	tree->first_nodes[i] = node;
	i++;
	node = node->left;
    }
}

static void recursive_init_tree(parsec_profiling_node_t *node, int depth, int max_depth)
{
    node->depth = depth;

    if (max_depth <= depth) { /* called on a leaf */
        node->left  = NULL;
        node->right = NULL;
        return;
    }

    switch(depth) {
    case PROF_ROOT: /* root dictionary, inserts namespaces */
        node->new_bucket = create_ns_bucket;
        break;
    case PROF_NAMESPACE:
        node->new_bucket = create_tc_bucket;
        break;
    case PROF_TASK_CLASS:
        node->new_bucket = create_pr_bucket;
        break;
    default:
        break;
    }

    node->left  = (parsec_profiling_node_t*)calloc(1, sizeof(parsec_profiling_node_t));
    node->left->wildcard = 0;
    node->left->parent = node;
    node->left->depth = node->depth+1;
    node->left->next_sibling = NULL;
    recursive_init_tree(node->left, depth+1, max_depth);

    node->right = (parsec_profiling_node_t*)calloc(1, sizeof(parsec_profiling_node_t));
    node->right->wildcard = 1;
    node->right->parent = node;
    node->right->depth = node->depth+1;
    node->right->next_sibling = NULL;
    node->right->str = (char*)calloc(2, sizeof(char));
    sprintf(node->right->str, "%s", "*");
    recursive_init_tree(node->right, depth+1, max_depth);
}

static parsec_profiling_tree_t *parsec_profiling_init_tree(int depth)
{
    parsec_profiling_tree_t *tree = (parsec_profiling_tree_t*)calloc(1, sizeof(parsec_profiling_tree_t));
    parsec_profiling_node_t *root = (parsec_profiling_node_t*)calloc(1, sizeof(parsec_profiling_node_t));
    tree->root = root;
    tree->depth = depth;
    tree->root->str = "PaRSEC";
    tree->first_nodes = (parsec_profiling_node_t**)calloc(tree->depth+1, sizeof(parsec_profiling_node_t*));

    recursive_init_tree(root, 0, depth);
    return tree;
}

static parsec_hash_table_item_t *create_pr_bucket(char *pr)
{
    char *str = (char*)calloc(strlen(pr)+1, sizeof(char));
    sprintf(str, "%s", pr);
    parsec_profiling_property_t *bucket = calloc(1, sizeof(parsec_profiling_property_t));
    bucket->super.key = (parsec_key_t)str;
    bucket->state = PROPERTY_NO_STATE;
    bucket->accumulate = 0;
    return (parsec_hash_table_item_t*)bucket;
}

static parsec_hash_table_item_t *create_tc_bucket(char *tc)
{
    char *str = (char*)calloc(strlen(tc)+1, sizeof(char));
    sprintf(str, "%s", tc);
    parsec_profiling_task_class_t *bucket = calloc(1, sizeof(parsec_profiling_task_class_t));
    bucket->super.key = (parsec_key_t)str;
    parsec_hash_table_init(&bucket->properties, 0, 8, dict_key_fns, NULL);
    return (parsec_hash_table_item_t*)bucket;
}

static parsec_hash_table_item_t *create_ns_bucket(char *ns)
{
    char *str = (char*)calloc(strlen(ns)+1, sizeof(char));
    sprintf(str, "%s", ns);
    parsec_profiling_namespace_t *bucket = calloc(1, sizeof(parsec_profiling_namespace_t));
    bucket->super.key = (parsec_key_t)str;
    parsec_hash_table_init(&bucket->task_classes, 0, 8, dict_key_fns, NULL);
    return (parsec_hash_table_item_t*)bucket;
}

static parsec_profiling_namespace_t *find_or_insert_ns(parsec_hash_table_t *ht, char *ns)
{
    char *str = NULL;
    parsec_profiling_namespace_t *ns_bucket = NULL;
    ns_bucket = parsec_hash_table_nolock_find(ht, (parsec_key_t)ns);
    if (!ns_bucket) {
	/* Namespace doesn't exist, therefore insert it */
	ns_bucket = calloc(1, sizeof(parsec_profiling_namespace_t));
	str = (char*)calloc(strlen(ns)+1, sizeof(char));
	sprintf(str, "%s", ns);
	ns_bucket->super.key = (uint64_t)str;
	parsec_hash_table_init(&ns_bucket->task_classes, 0, 8, dict_key_fns, NULL);
	parsec_hash_table_nolock_insert(ht, &ns_bucket->super);
    }
    return ns_bucket;
}

static parsec_profiling_task_class_t *find_or_insert_tc(parsec_hash_table_t *ht, char *tc)
{
    char *str = NULL;
    parsec_profiling_task_class_t *tc_bucket = NULL;
    if (NULL == (tc_bucket = parsec_hash_table_nolock_find(ht, (parsec_key_t)tc))) {
	/* Function doesn't exist, therefore insert it */
	tc_bucket = calloc(1, sizeof(parsec_profiling_task_class_t));
	str = (char*)calloc(strlen(tc)+1, sizeof(char));
	sprintf(str, "%s", tc);
	tc_bucket->super.key = (uint64_t)str;
	parsec_hash_table_init(&tc_bucket->properties, 0, 8, dict_key_fns, NULL);

	parsec_hash_table_nolock_insert(ht, &tc_bucket->super);
    }
    return tc_bucket;
}

static parsec_profiling_property_t *find_or_insert_pr(parsec_hash_table_t *ht, char *pr)
{
    char *str = NULL;
    parsec_profiling_property_t *pr_bucket = NULL;
    if (NULL == (pr_bucket = parsec_hash_table_nolock_find(ht, (parsec_key_t)pr))) {
	/* Namespace doesn't exist, therfore create everything */
	pr_bucket = calloc(1, sizeof(parsec_profiling_property_t));
	str = (char*)calloc(strlen(pr)+1, sizeof(char));
	sprintf(str, "%s", pr);
	pr_bucket->super.key = (uint64_t)str;
	pr_bucket->state = PROPERTY_NO_STATE;
	pr_bucket->accumulate = 0;
	parsec_hash_table_nolock_insert(ht, &pr_bucket->super);
    }
    return pr_bucket;
}

static void property_print(void *item, void *cb_data)
{
    (void)cb_data;
    parsec_profiling_property_t *bucket = (parsec_profiling_property_t*)item;
    fprintf(stdout, "        - %s\n", (char*)(bucket->super.key));
}

static void task_class_print(void *item, void *cb_data)
{
    (void)cb_data;
    parsec_profiling_task_class_t *bucket = (parsec_profiling_task_class_t*)item;
    fprintf(stdout, "    - %s {\n", (char*)(bucket->super.key));
    parsec_hash_table_for_all(&bucket->properties, property_print, cb_data);
    fprintf(stdout, "      }\n");
}

static void namespace_print(void *item, void *cb_data)
{
    (void)cb_data;
    parsec_profiling_namespace_t *bucket = (parsec_profiling_namespace_t*)item;
    fprintf(stdout, "%s {\n", (char*)(bucket->super.key));
    parsec_hash_table_for_all(&bucket->task_classes, task_class_print, cb_data);
    fprintf(stdout, "}\n");
}

/* place string buffer for every other node, not a wildcard node */
static void parsec_profiling_tree_setstr(parsec_profiling_tree_t *tree, char *str, int pos)
{
    parsec_profiling_node_t *node = tree->first_nodes[pos];
    while (node) { /* will overshoot to next line */
	if (!node->wildcard)
	    node->str = str;
	node = node->next_sibling;
    }
    return;
}

/* call after pushing a different string in a buffer at position pos to reload the buckets */
static void parsec_profiling_tree_reload_buckets(parsec_profiling_tree_t *tree, int pos, const char *name)
{
    if (!strcmp(name, dict->tree->first_nodes[pos]->str)) return;

    sprintf(dict->tree->first_nodes[pos]->str, "%s", name);

    parsec_profiling_node_t *node = tree->first_nodes[pos];
    while (node) {
	node->bucket = NULL;
	if (node->parent->ht) {
  	    node->bucket = parsec_hash_table_nolock_find(node->parent->ht, (parsec_key_t)node->str);
	}
	if (node->bucket) {
	    switch(pos) {
	    case PROF_NAMESPACE:
		node->ht = &((parsec_profiling_namespace_t*)node->bucket)->task_classes;
		break;
	    case PROF_TASK_CLASS:
		node->ht = &((parsec_profiling_task_class_t*)node->bucket)->properties;
		break;
	    case PROF_PROPERTY: /* No hashtable to catch */
		break;
	    default: /* how did you end up here? */
		break;
	    }
	}
	node = node->next_sibling;
    }
    return;
}

/* create the missing subtree on the left path */
static parsec_hash_table_item_t *parsec_profiling_tree_add_missing_buckets(parsec_profiling_tree_t *tree)
{
    parsec_hash_table_item_t *bucket = NULL;
    parsec_profiling_node_t *node = tree->root->left;
    while(node) {
	bucket = NULL;
	if (node->parent->ht)
	  bucket = (parsec_hash_table_item_t*)parsec_hash_table_nolock_find(node->parent->ht, (parsec_key_t)node->str);
	if (node->parent->ht && !bucket && node->parent->new_bucket) {
	  bucket = node->parent->new_bucket(node->str);
	  parsec_hash_table_nolock_insert(node->parent->ht, bucket);
	  if (node->depth == 1)
	    node->ht = &((parsec_profiling_namespace_t*)bucket)->task_classes;
	  else if (node->depth == 2)
	    node->ht = &((parsec_profiling_task_class_t*)bucket)->properties;
  	}

	node = node->left;
    }
    return bucket; /* return the leaf, if everything goes well */
}

static int parsec_profiling_tree_look_for_path(parsec_profiling_tree_t *tree)
{
    parsec_profiling_node_t *node = tree->first_nodes[tree->depth];
    while(node) {
	parsec_profiling_property_t *pr = (parsec_profiling_property_t*)node->bucket;
	if (pr && pr->state == PROPERTY_REQUESTED)
	    return 1;
	node = node->next_sibling;
    }
    return 0;
}

static void parsec_profiling_tree_recursive_delete(parsec_profiling_node_t *node)
{
    if (node->left)
	parsec_profiling_tree_recursive_delete(node->left);
    if (node->right)
	parsec_profiling_tree_recursive_delete(node->right);
    free(node);
}

static void parsec_profiling_tree_delete(parsec_profiling_tree_t *tree)
{
    parsec_profiling_tree_recursive_delete(tree->root);
    free(tree->first_nodes);
    free(tree);
}

/**
 * @brief High level method to update a dictionnary
 * @details Will start by recomputing the XML header and its size.
 *          Will reshape the shared memory region if necessary.
 *          Will export the new metadata in the shared memory region.
 *          Will implicitely see its version incremented by 1
 */
static int parsec_profiling_dictionary_update()
{
  if (dict->shmem) {
    /* Update header xml string */
    parsec_profiling_update_xml_header();
    /* If the version number of the dictionary changed, you should reload the whole shmem, just in case */
    /* Reopen a new shmem if the number of pages changes */
    parsec_profiling_change_shmem();
    /* Push the new header */
    parsec_profiling_dump_header_shmem();
    /* Close the old shmem, now that threads are working with the new one */
    parsec_debug_verbose(11, parsec_debug_output, "Header dumped in shared memory:\n%s", (char*)(dict->shmem->buffer)+3*sizeof(int));
  }
    return PARSEC_SUCCESS;
}

/**
 * @brief Converts the datatype into its size in bytes.
 */
static size_t sizeof_for_type(parsec_profiling_datatype_t dt)
{
    /* Python pack format */
    /* c    char                  string lgth    1 */
    /* b    signed char           integer        1 */
    /* B    unsigned char         integer        1 */
    /* ?    _Bool                 bool           1 */
    /* h    short                 integer        2 */
    /* H    unsigned short        integer        2 */
    if (dt == PROPERTIES_INT32)     return sizeof(int32_t);  /* i    int                   integer        4 */
    /* I    unsigned int          integer        4 */
    /* l    long                  integer        4 */
    /* L    unsigned long         integer        4 */
    if (dt == PROPERTIES_INT64)     return sizeof(int64_t);  /* q    long long             integer        8 */
    if (dt == PROPERTIES_ULONGLONG) return sizeof(unsigned long long); /* Q    unsigned long long    integer        8 */
    if (dt == PROPERTIES_FLOAT)     return sizeof(float);    /* f    float                 float          4 */
    if (dt == PROPERTIES_DOUBLE)    return sizeof(double);  /* d    double                float          8 */
    /* s    char[]                string */
    /* p    char[]                string */
    /* P    void *                integer */
    return sizeof(int64_t); /* python code for padding */
}

/**
 * @brief Converts the datatype into the python datatype character
 */
static char code_for_type(parsec_profiling_datatype_t dt)
{
    /* Python pack format */
    /* c    char                  string lgth    1 */
    /* b    signed char           integer        1 */
    /* B    unsigned char         integer        1 */
    /* ?    _Bool                 bool           1 */
    /* h    short                 integer        2 */
    /* H    unsigned short        integer        2 */
    if (dt == PROPERTIES_INT32)     return 'i';  /* i    int                   integer        4 */
    /* I    unsigned int          integer        4 */
    /* l    long                  integer        4 */
    /* L    unsigned long         integer        4 */
    if (dt == PROPERTIES_INT64)     return 'q';  /* q    long long             integer        8 */
    if (dt == PROPERTIES_ULONGLONG) return 'Q'; /* Q    unsigned long long    integer        8 */
    if (dt == PROPERTIES_FLOAT)     return 'f';  /* f    float                 float          4 */
    if (dt == PROPERTIES_DOUBLE)    return 'd'; /* d    double                float          8 */
    /* s    char[]                string */
    /* p    char[]                string */
    /* P    void *                integer */
    return 'x'; /* python code for padding */
}

/**
 * @brief Temporary structure used for building the XML header
 */
struct param_s {
    size_t                        *offset;
    parsec_profiling_index_t       type;
    parsec_profiling_namespace_t  *ns;
    parsec_profiling_task_class_t   *tc;
    char                          *ns_buff;
    char                          *tc_buff;
    int                            tc_count;
    char                          *pr_buff; /* tmp buf for string storage before export */
    int                            pr_count;
};

/**
 * @brief XML header string is written into the shared memory region.
 * @details String is offset by three integers. Those 3 integers are
 *          the size of the region in pages, the running state of the
 *          dictionnary, and the version of the dictionnary
 */
static int parsec_profiling_dump_header_shmem(void)
{
    if (dict->shmem && dict->shmem->buffer) {
	/* Shmem object is activated using mca param, and shmem area is opened */
	sprintf(dict->shmem->buffer+3*sizeof(int), "%s", dict->shmem->header);
    }
    return PARSEC_SUCCESS;
}

/**
 * @brief Check if property has to be exported and export its metadata
 * @details Pushes a metadata string for the property into the temporary
 *          buffer used by the calling function to decide if there is
 *          anything to propagate to higher levels.
 */
static void dump_property(void *item, void *cb_data)
{
    parsec_profiling_property_t *pr = (parsec_profiling_property_t*)item;
    struct param_s *params = (struct param_s*)cb_data;
    if (pr->state == (PROPERTY_REQUESTED|PROPERTY_PROVIDED) && pr->type == params->type) {
	char *name = (char*)pr->super.key;
	char *buff = params->pr_buff+strlen(params->pr_buff);
	sprintf(buff, "        <%s><t>%c</t><o>%zu</o></%s>\n", name, code_for_type(pr->func.type), *(params->offset), name);
	params->pr_count++;
	pr->func.offset = *(params->offset);
	*(params->offset) += sizeof_for_type(pr->func.type);
    }
}

/**
 * @brief Explore a task_class, looking for properties to export
 */
static void dump_task_class(void *item, void *cb_data)
{
    parsec_profiling_task_class_t *tc = (parsec_profiling_task_class_t*)item;
    struct param_s *params = (struct param_s*)cb_data;

    params->tc = tc;
    params->pr_count = 0;
    params->pr_buff[0] = '\0'; /* Resets the tmp buff as an empty string */
    /* Gather everything */
    parsec_hash_table_for_all(&tc->properties, dump_property, cb_data);
    /* Dump if necessary */
    if (0 < strlen(params->pr_buff)) {
	char *name = (char*)tc->super.key;
	char *buff = params->tc_buff+strlen(params->tc_buff);
	sprintf(buff, "      <%s>\n%s      </%s>\n", name, params->pr_buff, name);
	params->tc_count++;
    }
}

/**
 * @brief Explore a given namespace, looking for task_classes to export
 */
static void dump_namespace(void *item, void *cb_data)
{
    parsec_profiling_namespace_t *ns = (parsec_profiling_namespace_t*)item;
    struct param_s *params = (struct param_s*)cb_data;

    params->ns = ns;
    params->tc_count = 0;
    params->tc_buff[0] = '\0'; /* Resets the tmp buff as an empty string */
    /* Gather everything */
    parsec_hash_table_for_all(&ns->task_classes, dump_task_class, cb_data);
    /* Dump if necessary */
    if (0 < strlen(params->tc_buff)) {
	char *name = (char*)ns->super.key;
	char *buff = params->ns_buff+strlen(params->ns_buff);
	sprintf(buff, "    <%s>\n%s    </%s>\n", name, params->tc_buff, name);
    }
}

/**
 * @brief rebuild the header XML based on the properties requested and provided, and the context running
 */
static int parsec_profiling_update_xml_header(void)
{
    char buff[DICT_PAGE_SIZE];
    char pr_buff[DICT_PAGE_SIZE];
    char tc_buff[DICT_PAGE_SIZE];
    char ns_buff[DICT_PAGE_SIZE];

    struct param_s tmp;
    tmp.ns_buff = ns_buff;
    tmp.tc_buff = tc_buff;
    tmp.pr_buff = pr_buff;

    char *desc = dict->shmem->header;
    sprintf(desc, "<?xml version=\"1.0\"?>\n<root>\n");
    desc += 29;
    sprintf(desc, "<application>\n");
    desc += 14;

    sprintf(buff, "  <prank>%d</prank>\n", dict->context->my_rank);
    sprintf(desc, "%s", buff);
    desc += strlen(buff);

    sprintf(buff, "  <psize>%d</psize>\n", dict->context->nb_nodes);
    sprintf(desc, "%s", buff);
    desc += strlen(buff);

    sprintf(buff, "  <nb_vp>%d</nb_vp>\n", dict->shmem->nb_vp);
    sprintf(desc, "%s", buff);
    desc += strlen(buff);

    sprintf(buff, "  <nb_eu>%d</nb_eu>\n", dict->shmem->nb_eu);
    sprintf(desc, "%s", buff);
    desc += strlen(buff);

    size_t per_nd_offset = 0;
    tmp.offset = &per_nd_offset;
    tmp.type = PROFILING_PER_NODE;
    tmp.ns_buff[0] = '\0';
    parsec_hash_table_for_all(&dict->properties, dump_namespace, &tmp);

    if (0 < strlen(tmp.ns_buff)) {
	sprintf(desc, "  <per_nd_properties>\n");
	desc += 22;
	sprintf(desc, "%s", tmp.ns_buff);
	desc += strlen(tmp.ns_buff);
	sprintf(desc, "  </per_nd_properties>\n");
	desc += 23;
    }

    if (dict->shmem) {
	dict->shmem->nb_node_pages = 1+(per_nd_offset-1)/DICT_PAGE_SIZE;
	sprintf(buff, "  <pages_per_nd>%d</pages_per_nd>\n", dict->shmem->nb_node_pages);
	sprintf(desc, "%s", buff);
	desc += strlen(buff);
    }

    size_t per_vp_offset = 0;
    tmp.offset = &per_vp_offset;
    tmp.type = PROFILING_PER_VP;
    tmp.ns_buff[0] = '\0';
    parsec_hash_table_for_all(&dict->properties, dump_namespace, &tmp);

    if (0 < strlen(tmp.ns_buff)) {
	sprintf(desc, "  <per_vp_properties>\n");
	desc += 22;
	sprintf(desc, "%s", tmp.ns_buff);
	desc += strlen(tmp.ns_buff);
	sprintf(desc, "  </per_vp_properties>\n");
	desc += 23;
    }

    if (dict->shmem) {
	dict->shmem->nb_vp_pages = 1+(per_vp_offset-1)/DICT_PAGE_SIZE;
	sprintf(buff, "  <pages_per_vp>%d</pages_per_vp>\n", dict->shmem->nb_vp_pages);
	sprintf(desc, "%s", buff);
	desc += strlen(buff);
    }

    size_t per_th_offset = 0;
    tmp.offset = &per_th_offset;
    tmp.type = PROFILING_PER_EU;
    tmp.ns_buff[0] = '\0';

    parsec_hash_table_for_all(&dict->properties, dump_namespace, &tmp);

    if (0 < strlen(tmp.ns_buff)) {
	sprintf(desc, "  <per_eu_properties>\n");
	desc += 22;
	sprintf(desc, "%s", tmp.ns_buff);
	desc += strlen(tmp.ns_buff);
	sprintf(desc, "  </per_eu_properties>\n");
	desc += 23;
    }

    if (dict->shmem) {
	dict->shmem->nb_eu_pages = 1+(per_th_offset-1)/DICT_PAGE_SIZE;
	sprintf(buff, "  <pages_per_eu>%d</pages_per_eu>\n", dict->shmem->nb_eu_pages);
	sprintf(desc, "%s", buff);
	desc += strlen(buff);
    }

    sprintf(buff, "%s", "</application>\n</root>\n");
    sprintf(desc, "%s", buff);
    desc += strlen(buff);

    dict->shmem->nb_xml_pages = 1+(strlen(dict->shmem->header)-1+3*sizeof(int))/DICT_PAGE_SIZE;

    return PARSEC_SUCCESS;
}

static int parsec_profiling_change_shmem(void)
{
    /* I don't want to keep a number of readers on a shmem area */
    /* How do I switch from the old one to the new one? */

    int nb_pages = dict->shmem->nb_xml_pages + dict->shmem->nb_node_pages
	+ dict->shmem->nb_vp * dict->shmem->nb_vp_pages + dict->shmem->nb_eu * dict->shmem->nb_eu_pages;
    if (dict->shmem->nb_pages != nb_pages) {
	parsec_mca_param_reg_string_name("pins", "shmem_name",
					 "Name of the Shared memory area to save App level perf counters.\n",
					 false, false,
					 "parsec_shmem", &dict->shmem->shmem_name);

	dict->shmem->shm_fd = shm_open(dict->shmem->shmem_name, O_RDWR | O_CREAT, 0666);
	int ret = ftruncate(dict->shmem->shm_fd, nb_pages*DICT_PAGE_SIZE);
	if (0 > ret) {
	    perror("parsec_profiling_change_shmem, error:");
	    return PARSEC_ERROR;
	}

	if (MAP_FAILED == (dict->shmem->buffer = mmap(0, nb_pages*DICT_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dict->shmem->shm_fd, 0))) {
	    fprintf(stderr, "Map failed to area %s\n", dict->shmem->shmem_name);
	    /* Turn off feature */

	    return PARSEC_ERROR;
	}
	memset(dict->shmem->buffer, '\0', nb_pages*DICT_PAGE_SIZE);
	((int*)dict->shmem->buffer)[0] = nb_pages;
	((int*)dict->shmem->buffer)[1] = dict->running;
	((int*)dict->shmem->buffer)[2] = dict->version;
	parsec_debug_verbose(10, parsec_debug_output, "Opened a shared memory area named %s.", dict->shmem->shmem_name);
	dict->shmem->first_vp = dict->shmem->nb_xml_pages + dict->shmem->nb_node_pages;
	dict->shmem->first_eu = dict->shmem->first_vp + dict->shmem->nb_vp * dict->shmem->nb_vp_pages;
    }

    return PARSEC_SUCCESS;
}

void parsec_profiling_evaluate_property(void *item, void* cb_data)
{
    parsec_profiling_property_t *pr = (parsec_profiling_property_t *)item;
    if (!(pr->state & PROPERTY_REQUESTED)) return;
    if (!(pr->state & PROPERTY_PROVIDED)) return;

    void **tmp = (void**)cb_data;
    struct parsec_execution_stream_s *exec_unit = (struct parsec_execution_stream_s*)tmp[0];
    struct parsec_task_s *task = (struct parsec_task_s*)tmp[1];

    void *buf = dict->shmem->buffer + (dict->shmem->first_eu + exec_unit->th_id)*DICT_PAGE_SIZE + pr->func.offset;

    switch (pr->func.type) {
    case PROPERTIES_INT32:
	*(int32_t*)buf = pr->accumulate*(*(int32_t*)buf) + pr->func.func.inline_func_int32(task->taskpool, task->locals);
	break;
    case PROPERTIES_INT64:
	*(int64_t*)buf = pr->accumulate*(*(int64_t*)buf) + pr->func.func.inline_func_int64(task->taskpool, task->locals);
	break;
    case PROPERTIES_FLOAT:
	*(float*)buf   = pr->accumulate*(*(float*)buf)   + pr->func.func.inline_func_float(task->taskpool, task->locals);
	break;
    case PROPERTIES_DOUBLE:
	*(double*)buf  = pr->accumulate*(*(double*)buf)  + pr->func.func.inline_func_double(task->taskpool, task->locals);
	break;
    case PROPERTIES_ULONGLONG:
    default:
	/* Unknown type */
	break;
    }
}

int parsec_profiling_dictionary_init(parsec_context_t *master_context,
				     int num_modules,
				     parsec_pins_module_t **modules)
{
    (void)num_modules;
    (void)modules;

    dict = (parsec_profiling_dictionary_t*)calloc(1, sizeof(parsec_profiling_dictionary_t));

    parsec_hash_table_init(&dict->properties, 0, 8, dict_key_fns, NULL);

    dict->version = 0;
    dict->running = 1;
    dict->context = master_context;

    dict->tree = parsec_profiling_init_tree(3);
    parsec_profiling_chain_siblings(dict->tree);

    dict->tree->root->ht = &dict->properties;

    char *a = (char*)calloc(MAX_LENGTH_NAME, sizeof(char));
    char *b = (char*)calloc(MAX_LENGTH_NAME, sizeof(char));
    char *d = (char*)calloc(MAX_LENGTH_NAME, sizeof(char));

    parsec_profiling_tree_setstr(dict->tree, a, PROF_NAMESPACE);
    parsec_profiling_tree_setstr(dict->tree, b, PROF_TASK_CLASS);
    parsec_profiling_tree_setstr(dict->tree, d, PROF_PROPERTY);

    /* This section registers the properties requested by the user */
    char *user_props, *c, *s, *ns, *tc, *pr;
    parsec_mca_param_reg_string_name("profiling", "properties",
				    "Application Level Performance events to be saved.\n",
				    false, false,
				    "", &user_props);

    parsec_profiling_namespace_t *ns_bucket = NULL;
    parsec_profiling_task_class_t  *tc_bucket = NULL;
    parsec_profiling_property_t  *pr_bucket = NULL;
    s = user_props;
    while( ( c = strtok_r(s, ";", &s) ) != NULL ) {
	ns = strtok_r(c, ":", &c);
	tc = strtok_r(c, ":", &c);
	pr = strtok_r(c, ":", &c);
	ns_bucket = find_or_insert_ns(&dict->properties, ns);
	tc_bucket = find_or_insert_tc(&ns_bucket->task_classes, tc);
	pr_bucket = find_or_insert_pr(&tc_bucket->properties, pr);

	pr_bucket->event      = EXEC_END;
	pr_bucket->freq       = 1;
	pr_bucket->counter    = 0;
	pr_bucket->state     |= PROPERTY_REQUESTED;
    }

    dict->shmem = NULL;
    int mca_shmem_activate;
    parsec_mca_param_reg_int_name("pins", "shmem_activate",
				  "Application level shared memory will be used to export profiling counters.\n",
				  false, false, 0, &mca_shmem_activate);

    /* This section prepares the shared memory module */
    if (mca_shmem_activate) {
	dict->shmem           = (parsec_profiling_shmem_t*)calloc(1, sizeof(parsec_profiling_shmem_t));
	dict->shmem->nb_pages = 0;
	dict->shmem->header   = (char*)calloc(10*DICT_PAGE_SIZE, sizeof(char));
	dict->shmem->nb_vp    = master_context->nb_vp;
	dict->shmem->nb_eu    = 0;
	dict->shmem->first_vp = -1;
	dict->shmem->first_eu = -1;

	int i;
	for (i = 0; i < dict->shmem->nb_vp; ++i)
	    dict->shmem->nb_eu += master_context->virtual_processes[i]->nb_cores;

	/* Activate the shmem PINS module */
    }

    /* Dictionary and shmem are intialized, let's see if modules have something to say */
    /* int m; */
    /* for (m = 0; m < num_modules; ++m) */
    /*     if (modules[m]->init_profiling.register_properties) */
    /*         modules[m]->init_profiling.register_properties(); */

    return PARSEC_SUCCESS;
}

int parsec_profiling_dictionary_free()
{
    int i;
    for (i = 0; i < dict->tree->depth; ++i)
	if (dict->tree->first_nodes[i]->str)
	    free(dict->tree->first_nodes[i]->str);
    parsec_profiling_tree_delete(dict->tree);

    return PARSEC_SUCCESS;
}

void print_dict_content(parsec_hash_table_t *ht)
{
    parsec_hash_table_for_all(ht, namespace_print, NULL);
}

parsec_profiling_namespace_t *find_namespace(const char *ns)
{
    return parsec_hash_table_nolock_find(&dict->properties, (parsec_key_t)ns);
}

parsec_profiling_task_class_t *find_task_class(parsec_profiling_namespace_t *ns, const char *tc)
{
    return parsec_hash_table_nolock_find(&ns->task_classes, (parsec_key_t)tc);
}

parsec_profiling_property_t *find_property(parsec_profiling_task_class_t* tc, const char *pr)
{
    return parsec_hash_table_nolock_find(&tc->properties, (parsec_key_t)pr);
}

int parsec_profiling_add_taskpool_properties(parsec_taskpool_t *h)
{
    uint16_t i;
    /* PROF_NAMESPACE: ns, * */
    /* PROF_TASK_CLASS:  ns:tc, ns:*, *:tc, *:* */
    /* PROF_PROPERTY:  ns:tc:pr, ns:tc:*, ns:*:pr, ns:*:*, *:tc:pr, *:tc:*, *:*:pr, *:*:* */

    parsec_task_class_t *f;
    parsec_property_t *p;

    parsec_profiling_tree_reload_buckets(dict->tree, PROF_NAMESPACE, h->taskpool_name);
    for (i = 0; i < h->nb_task_classes; i++) {
	f = (parsec_task_class_t*)(h->task_classes_array[i]);
	parsec_profiling_tree_reload_buckets(dict->tree, PROF_TASK_CLASS, f->name);

	p = (parsec_property_t*)(f->properties);
	while (p && NULL != p->expr) {
	    parsec_profiling_tree_reload_buckets(dict->tree, PROF_PROPERTY, p->name);

	    /* check if anyone requested ns:tc:pr or the wilcard combinations */
	    int exists = parsec_profiling_tree_look_for_path(dict->tree);

	    parsec_profiling_property_t* pr_bucket = (parsec_profiling_property_t*) parsec_profiling_tree_add_missing_buckets(dict->tree);
	    pr_bucket->func.type = p->expr->u_expr.v_func.type;
	    pr_bucket->state |= PROPERTY_PROVIDED;
	    pr_bucket->type = PROFILING_PER_EU;
	    pr_bucket->accumulate = CUMULATIVE;

	    if (exists) pr_bucket->state |= PROPERTY_REQUESTED;

	    switch (pr_bucket->func.type) {
	    case PROPERTIES_INT32:
		pr_bucket->func.func.inline_func_int32  = p->expr->u_expr.v_func.func.inline_func_int32;
		break;
	    case PROPERTIES_INT64:
		pr_bucket->func.func.inline_func_int64  = p->expr->u_expr.v_func.func.inline_func_int64;
		break;
	    case PROPERTIES_FLOAT:
		pr_bucket->func.func.inline_func_float  = p->expr->u_expr.v_func.func.inline_func_float;
		break;
	    case PROPERTIES_DOUBLE:
		pr_bucket->func.func.inline_func_double = p->expr->u_expr.v_func.func.inline_func_double;
		break;
	    case PROPERTIES_UNKNOWN:
	    default:
		break;
	    }
	    p++;
	}
    }

    dict->version++;
    parsec_profiling_dictionary_update();

    return PARSEC_SUCCESS;
}

int parsec_profiling_register_property(parsec_property_function_t *func,
				       const char *namespace,
				       const char *task_class,
				       const char *property,
				       parsec_profiling_index_t who,
				       int cumulative)
{
    parsec_profiling_tree_reload_buckets(dict->tree, PROF_NAMESPACE, namespace);
    parsec_profiling_tree_reload_buckets(dict->tree, PROF_TASK_CLASS, task_class);
    parsec_profiling_tree_reload_buckets(dict->tree, PROF_PROPERTY, property);

    int exists = parsec_profiling_tree_look_for_path(dict->tree);

    parsec_profiling_property_t* pr_bucket = (parsec_profiling_property_t*) parsec_profiling_tree_add_missing_buckets(dict->tree);
    pr_bucket->state |= PROPERTY_PROVIDED;
    if (exists) pr_bucket->state |= PROPERTY_REQUESTED;
    pr_bucket->type = who;
    pr_bucket->accumulate = cumulative;

    /* Lazy copy of the task_class pointer */
    memcpy(&pr_bucket->func, func, sizeof(parsec_property_function_t));

    dict->version++;
    parsec_profiling_dictionary_update();

    return PARSEC_SUCCESS;
}
