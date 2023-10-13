/*
 * Copyright (c) 2021-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <string.h>
#include <stdio.h>
#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_hash_table.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/mca_repository.h"

static parsec_hash_table_t parsec_termdet_opened_modules;

typedef struct {
    parsec_hash_table_item_t ht_item;
    char                    *name;
    mca_base_component_t    *component;
    mca_base_module_t       *module;
} parsec_termdet_opened_module_t;

static int string_key_equal(parsec_key_t a, parsec_key_t b, void *user_data)
{
    char *stra = (char*)a;
    char *strb = (char*)b;
    (void)user_data;
    return !strcmp(stra, strb);
}

static char *string_key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data)
{
    (void)user_data;
    snprintf(buffer, buffer_size, "%s", (char*)k);
    return buffer;
}

static uint64_t string_key_hash(parsec_key_t k, void *user_data)
{
    int i;
    char *strk = (char*)k;
    uint64_t h = 0;
    (void)user_data;
    for(i = 0; strk[i] != 0; i++) {
        h += strk[i] << (i % 64) ;
    }
    return h;
}

parsec_key_fn_t parsec_termdet_opened_module_key_fn = {
        string_key_equal,
        string_key_print,
        string_key_hash
};

int parsec_termdet_init(void)
{
    parsec_hash_table_init(&parsec_termdet_opened_modules, offsetof(parsec_termdet_opened_module_t, ht_item), 4,
                           parsec_termdet_opened_module_key_fn, NULL);
    return PARSEC_SUCCESS;
}

int parsec_termdet_open_module(parsec_taskpool_t *tp, char *name)
{
    parsec_termdet_opened_module_t *omod;

    assert(NULL == tp->tdm.module);

    parsec_hash_table_lock_bucket(&parsec_termdet_opened_modules, (parsec_key_t)name);
    omod = parsec_hash_table_nolock_find(&parsec_termdet_opened_modules, (parsec_key_t)name);
    if(NULL == omod) {
        omod = malloc(sizeof(parsec_termdet_opened_module_t));
        omod->name = strdup(name);
        omod->ht_item.key = (parsec_key_t)omod->name;
        omod->component = mca_component_open_byname("termdet", name);
        if(NULL == omod->component) {
            free(omod->name);
            free(omod);
            parsec_fatal("Could not find a MCA module named '%s' of type termination detection (termdet)",
                         name);
            parsec_hash_table_unlock_bucket(&parsec_termdet_opened_modules, (parsec_key_t)name);
            return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
        }
        omod->module = mca_component_query(omod->component);
        if(NULL == omod->module) {
            free(omod->name);
            mca_component_close(omod->component);
            free(omod);
            parsec_fatal("Component of name '%s' of type termdet exists, but could not load (query failed)",
                         name);
            parsec_hash_table_unlock_bucket(&parsec_termdet_opened_modules, (parsec_key_t)name);
            return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
        }
        parsec_hash_table_nolock_insert(&parsec_termdet_opened_modules, &omod->ht_item);
    }
    parsec_hash_table_unlock_bucket(&parsec_termdet_opened_modules, (parsec_key_t)name);

    tp->tdm.module = &((parsec_termdet_module_t*)omod->module)->module;

    return PARSEC_SUCCESS;
}

int parsec_termdet_open_dyn_module(parsec_taskpool_t *tp)
{
    //TODO Once there are multiple choices for a dynamic module, this function should evolve
    return parsec_termdet_open_module(tp, "fourcounter");
}

static void parsec_termdet_close_module(void *item, void *data)
{
    parsec_hash_table_t *ht = (parsec_hash_table_t*)data;
    parsec_termdet_opened_module_t *omod = (parsec_termdet_opened_module_t *)item;
    if(NULL != omod->component->mca_close_component )
        omod->component->mca_close_component();
    parsec_hash_table_nolock_remove(ht, omod->ht_item.key);
    free(omod->name);
    free(omod);
}

int parsec_termdet_fini(void)
{
    parsec_hash_table_for_all(&parsec_termdet_opened_modules, parsec_termdet_close_module,
                              &parsec_termdet_opened_modules);
    parsec_hash_table_fini(&parsec_termdet_opened_modules);
    return PARSEC_SUCCESS;
}
