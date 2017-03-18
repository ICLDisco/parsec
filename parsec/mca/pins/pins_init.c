/*
 * Copyright (c) 2012-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/mca/mca_repository.h"
#include "parsec/execution_unit.h"
#include "parsec/profiling.h"
#include "parsec/utils/mca_param.h"

#define MAX_NAME_SIZE 100 /* arbitrary module name limit for 'safety' */

static int num_modules_activated = 0;
static parsec_pins_module_t **modules_activated = NULL;

static mca_base_component_t **pins_components = NULL;

/**
 * pins_init() should be called once and only once per runtime of a PaRSEC execution.
 * It should be called near the beginning of execution, preferably when most
 * other components have been initialized, so as to allow the interfacing of
 * PINS measurements with working PaRSEC subsystems.
 */
void pins_init(parsec_context_t* master_context)
{
    int i = 0, err, priority = -1;
    parsec_pins_module_t *module = NULL;
    char **user_list;

#if defined(PARSEC_PROF_TRACE)
    char * modules_activated_str = NULL;
#endif /* PARSEC_PROF_TRACE */

    parsec_debug_verbose(10, parsec_debug_output, "Initialized PaRSEC PINS callbacks to pins_empty_callback()");
    user_list = mca_components_get_user_selection("pins");
    if( NULL == user_list ) {
        /* No PINS component requested by user */
        return;
    }
    pins_components = mca_components_open_bytype("pins");
    for(i = 0; pins_components[i] != NULL; i++) /* nothing just counting */;
    modules_activated = (parsec_pins_module_t**)malloc(sizeof(parsec_pins_module_t*) * i);
#if defined(PARSEC_PROF_TRACE)
    modules_activated_str = (char*)malloc( (MAX_NAME_SIZE+1) * i);
    modules_activated_str[0] = '\0';
#endif /* PARSEC_PROF_TRACE */
    num_modules_activated = 0;

    for(i = 0; pins_components[i] != NULL; i++) {
        if( mca_components_belongs_to_user_list(user_list, pins_components[i]->mca_component_name) ) {
            if (pins_components[i]->mca_query_component != NULL) {
                err = pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
                if( err != MCA_SUCCESS ) {
                    parsec_debug_verbose(4, parsec_debug_output, "query function for component %s return no module", pins_components[i]->mca_component_name);
                    continue;
                }
                parsec_debug_verbose(10, parsec_debug_output, "query function for component %s[%d] returns priority %d",
                       pins_components[i]->mca_component_name, i, priority);
                if (NULL != module->module.init) {
                    module->module.init(master_context);
                }
                parsec_debug_verbose(4, parsec_debug_output, "Activated PINS module %s.",
                       module->component->base_version.mca_component_name);
                modules_activated[num_modules_activated++] = module;
#if defined(PARSEC_PROF_TRACE)
                strncat(modules_activated_str, pins_components[i]->mca_component_name, MAX_NAME_SIZE);
                strncat(modules_activated_str, ",", 1);
#endif
            }
        }
    }
    mca_components_free_user_list(user_list);
    parsec_debug_verbose(20, parsec_debug_output, "Found %d components, activated %d", i, num_modules_activated);
#if defined(PARSEC_PROF_TRACE)
    /* replace trailing comma with \0 */
    if ( strlen(modules_activated_str) > 1) {
        if( modules_activated_str[ strlen(modules_activated_str) - 1 ] == ',' ) {
            modules_activated_str[strlen(modules_activated_str) - 1] = '\0';
        }
    }
    parsec_profiling_add_information("PINS_MODULES", modules_activated_str);
    free(modules_activated_str);
#endif
}

/**
 * pins_fini must call fini methods of all modules
 */
void pins_fini(parsec_context_t* master_context)
{
    int i = 0;

    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if( NULL != modules_activated[i]->module.fini ) {
                modules_activated[i]->module.fini(master_context);
                parsec_debug_verbose(20, parsec_debug_output, "Finalized PINS module %s.",
                       modules_activated[i]->component->base_version.mca_component_name);
            }
        }
        free(modules_activated);
        modules_activated = NULL;
        num_modules_activated = -1;
    }

    if( NULL != pins_components ) {
        mca_components_close(pins_components);
        pins_components = NULL;
    }
}


/**
 * pins_thread_init() should be called once per thread runtime of a PaRSEC execution.
 * It should be called near the beginning of the thread's lifetime, preferably
 * once most other thread components have been initialized, so as to allow the
 * interfacing of PINS measurements with working PaRSEC subsystems.
 * It MUST NOT be called BEFORE pins_init().
 */
void pins_thread_init(parsec_execution_unit_t* exec_unit)
{
    int i;

    for( i = 0; i < PINS_FLAG_COUNT; i++ ) {
        exec_unit->pins_events_cb[i].cb_func = NULL;
        exec_unit->pins_events_cb[i].cb_data = NULL;
    }
    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if ( NULL != modules_activated[i]->module.thread_init)
                modules_activated[i]->module.thread_init(exec_unit);
        }
    }

    PINS(exec_unit, THREAD_INIT, NULL);
}

/**
 * called in scheduling.c, which is not ideal
 */
void pins_thread_fini(parsec_execution_unit_t* exec_unit)
{
    int i = 0;

    PINS(exec_unit, THREAD_FINI, NULL);

    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if ( NULL != modules_activated[i]->module.thread_fini)
                modules_activated[i]->module.thread_fini(exec_unit);
        }
    }

    for( i = 0; i < PINS_FLAG_COUNT; i++ ) {
        assert(NULL == exec_unit->pins_events_cb[i].cb_func);
        assert(NULL == exec_unit->pins_events_cb[i].cb_data);
    }
}

/**
 * pins_handle_init() should be called once per PaRSEC handle instantiation.
 * It should be called near the beginning of the handle's lifetime, preferably
 * once most other handle components have been initialized, so as to allow the
 * interfacing of PINS measurements with working PaRSEC subsystems.
 *
 * It MUST NOT be called BEFORE pins_init().
 */
void pins_handle_init(parsec_handle_t* handle)
{
    int i = 0;

    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if ( NULL != modules_activated[i]->module.handle_init)
                modules_activated[i]->module.handle_init(handle);
        }
    }
}

/**
 * Currently uncalled in the PaRSEC DPLAMSA testing executables
 */
void pins_handle_fini(parsec_handle_t * handle)
{
    int i = 0;

    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if ( NULL != modules_activated[i]->module.handle_fini)
                modules_activated[i]->module.handle_fini(handle);
        }
    }
}

/**
 * Convenient functions for application that want to overwrite the MCA
 *  default behavior.
 */
void pins_enable_modules (const char * const modules[])
{
    int i, l;
    char *str;
    int idx;

    l = 0;
    for(i = 0; NULL != modules[i]; i++) {
        l += strlen(modules[i])+1;
    }
    str = (char *)malloc(l);
    str[0] = '\0';
    for(i = 0; NULL != modules[i]; i++) {
        strcat(str, modules[i]);
        strcat(str, ",");
    }
    if(l > 0)
        str[l-1] = '\0';

    idx = parsec_mca_param_find("mca", NULL, "pins");
    parsec_mca_param_set_string(idx, str);

    free(str);
}

/*
  This function is not currently useful if the module limiting functionality
  is not in use, because this function checks based on that array.
  A future version could assemble a list of modules enabled based on which
  modules are "init"ed, and check against that array.
 */
int parsec_pins_is_module_enabled(char * name)
{
    int i = 0;

    if (NULL != modules_activated) {
        for(i = 0; i < num_modules_activated; i++) {
            if ( strcmp(name, modules_activated[i]->component->base_version.mca_component_name) == 0)
                return 1;
        }
    }
    return 0; /* no, this module is not enabled */
}
