/*
 * Copyright (c) 2012-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins.h"
#include "dague/mca/mca_repository.h"
#include "execution_unit.h"
#include "profiling.h"

static int allowable_modules_in_use;  /* allows all modules if not set */
static int allowable_modules_defined; /* flag to prevent multiple definitions */
static const char * const default_modules_array[] = {NULL};
char ** allowable_modules;
#define MAX_NAME_SIZE 100 /* arbitrary module name limit for 'safety' */
#define MAX_ENABLED_MODULES 30 /* arbitrary limit for safety */

extern parsec_pins_callback * pins_array[];

static mca_base_component_t **pins_components = NULL;

/**
 * pins_init() should be called once and only once per runtime of a PaRSEC execution.
 * It should be called near the beginning of execution, preferably when most
 * other components have been initialized, so as to allow the interfacing of
 * PINS measurements with working PaRSEC subsystems.
 */
void pins_init(dague_context_t * master_context) {
    int i = 0;
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int enabled_module_count = 0;

#if defined(DAGUE_PROF_TRACE)
    char modules_enabled[(MAX_NAME_SIZE + 1) * MAX_ENABLED_MODULES + 1];
    modules_enabled[0] = '\0';
#endif /* DAGUE_PROF_TRACE */

    for (; i < PINS_FLAG_COUNT; i++) {
        if (pins_array[i] == NULL)
            pins_array[i] = &pins_empty_callback;
    }
    DEBUG(("Initialized PaRSEC PINS callbacks to pins_empty_callback()"));

    /* We no longer default to the use of any particular module. */
    /* set_allowable_pins_modules(default_modules_array); */
    allowable_modules_defined = 1; /* block post-init definitions */
    i = 0;

    pins_components = mca_components_open_bytype("pins");
    while (pins_components[i] != NULL && enabled_module_count <= MAX_ENABLED_MODULES) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * init_this_module = NULL;
            if (!allowable_modules_in_use)
                init_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE)) {
                        init_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != init_this_module) {
                enabled_module_count++;
                if (NULL != init_this_module->module.init) {
                    init_this_module->module.init(master_context);
                    DEBUG(("Activated PINS module %s.\n",
                           init_this_module->component->base_version.mca_component_name));
                }
#if defined(DAGUE_PROF_TRACE)
                /* accumulate the names of modules used */
                strncat(modules_enabled, 
                        init_this_module->component->base_version.mca_component_name, 
                        MAX_NAME_SIZE);
                strncat(modules_enabled, ",", 1);
#endif
            }
        }
        i++;
    }
#if defined(DAGUE_PROF_TRACE)
    modules_enabled[strlen(modules_enabled) - 1] = '\0';
    dague_profiling_add_information("PINS_MODULES", modules_enabled);
#endif
}

/**
 * pins_fini must call fini methods of all modules
 */
void pins_fini(dague_context_t * master_context) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    /*
     * Call all fini methods in reverse order in order to preserve
     * cleanup semantics.
     */
    while (pins_components[i] != NULL)
        i++; /* count */
    i--; /* back down to before NULL element */

    while (i >= 0) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * fini_this_module = NULL;
            if (!allowable_modules_in_use)
                fini_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE))  {
                        fini_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != fini_this_module && NULL != fini_this_module->module.fini) {
                fini_this_module->module.fini(master_context);
                DEBUG(("Finalized PINS module %s.\n",
                       fini_this_module->component->base_version.mca_component_name));
            }
        }
        i--;
    }
    free(allowable_modules);
    allowable_modules = NULL;
    allowable_modules_in_use = 0;

    mca_components_close(pins_components);
    pins_components = NULL;
}


/**
 * pins_thread_init() should be called once per thread runtime of a PaRSEC execution.
 * It should be called near the beginning of the thread's lifetime, preferably
 * once most other thread components have been initialized, so as to allow the
 * interfacing of PINS measurements with working PaRSEC subsystems.
 * It MUST NOT be called BEFORE pins_init().
 */
void pins_thread_init(dague_execution_unit_t * exec_unit) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    while (pins_components[i] != NULL) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * t_init_this_module = NULL;
            if (!allowable_modules_in_use)
                t_init_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE)) {
                        t_init_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != t_init_this_module && NULL != t_init_this_module->module.thread_init)
                t_init_this_module->module.thread_init(exec_unit);
        }
        i++;
    }

    parsec_instrument(THREAD_INIT, exec_unit, NULL, NULL);
}

/**
 * called in scheduling.c, which is not ideal
 */
void pins_thread_fini(dague_execution_unit_t * exec_unit) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    /*
     * Call all fini methods in reverse order in order to preserve
     * cleanup semantics.
     */
    while (pins_components[i] != NULL)
        i++; /* count modules */
    i--; /* back down by one to skip NULL array element */

    while (i >= 0) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * t_fini_this_module = NULL;
            if (!allowable_modules_in_use)
                t_fini_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE)) {
                        t_fini_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != t_fini_this_module && NULL != t_fini_this_module->module.thread_fini) 
                t_fini_this_module->module.thread_fini(exec_unit);
        }
        i--;
    }

    parsec_instrument(THREAD_FINI, exec_unit, NULL, NULL);
}


/**
 * pins_handle_init() should be called once per PaRSEC handle instantiation.
 * It should be called near the beginning of the handle's lifetime, preferably
 * once most other handle components have been initialized, so as to allow the
 * interfacing of PINS measurements with working PaRSEC subsystems.
 *
 * It MUST NOT be called BEFORE pins_init().
 */
void pins_handle_init(dague_handle_t * handle) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    while (pins_components[i] != NULL) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * h_init_this_module = NULL;
            if (!allowable_modules_in_use)
                h_init_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE)) {
                        h_init_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != h_init_this_module && NULL != h_init_this_module->module.handle_init) 
                h_init_this_module->module.handle_init(handle);
        }
        i++;
    }

    parsec_instrument(HANDLE_INIT, NULL, NULL, (void *)handle);
}

/**
 * Currently uncalled in the PaRSEC DPLAMSA testing executables
 */
void pins_handle_fini(dague_handle_t * handle) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    /*
     * Call all fini methods in reverse order in order to preserve
     * cleanup semantics.
     */
    while (pins_components[i] != NULL)
        i++; 
    i--;

    while (i >= 0) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            dague_pins_module_t * h_fini_this_module = NULL;
            if (!allowable_modules_in_use)
                h_fini_this_module = module;
            else {
                while (allowable_modules[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     allowable_modules[j], MAX_NAME_SIZE)) {
                        h_fini_this_module = module;
                        break;
                    }
                    j++;
                }
            }
            if (NULL != h_fini_this_module && NULL != h_fini_this_module->module.handle_fini) 
                h_fini_this_module->module.handle_fini(handle);
        }
        i--;
    }

    parsec_instrument(HANDLE_FINI, NULL, NULL, (void *)handle);
}

/**
 * Addon method to allow for limiting the 'allowable modules.'
 * It is only possible to call this method only before pins_init, so as not to introduce
 * new modules without proper initialization, so as not to overwrite
 * currently-enabled modules, and so as not to cause threading complications.
 * (this method is certainly NOT THREAD-SAFE).
 *
 * The method will only allow itself to be called a single time.
 *
 * The array of modules should be terminated by a NULL pointer.
 */
void set_allowable_pins_modules (const char * const modules[]) {
    if (dague_atomic_cas(&allowable_modules_defined, 0, 1)) {
        int count = 0;
        allowable_modules_in_use = 1; /* activates the module limiting functionality */
        while (modules[count] != NULL)
            count++;
        allowable_modules = calloc(sizeof(char *), count + 1);
        if (allowable_modules != NULL) {
            allowable_modules[count] = NULL;
            for (count--; count >= 0; count--) {
                allowable_modules[count] = calloc(sizeof(char), MAX_NAME_SIZE + 1);
                if (NULL != allowable_modules[count]) {
                    strncpy(allowable_modules[count], modules[count], MAX_NAME_SIZE);
                    DEBUG(("Allowing PINS module %s\n", allowable_modules[count]));
                }
                else {
                    DEBUG(("Memory allocation failed in "
                           "'set_allowable_pins_modules.' "
                           "Module %s not enabled\n", modules[count]));
                }
            }
        }
        else {
            DEBUG(("Memory allocation failed in 'set_allowable_pins_modules.'"
                   " All modules disabled\n"));
        }
    }
    else {
        DEBUG3(("PINS modules have already been set and cannot be set again.\n"));
    }
}

/*
  This function is not currently useful if the module limiting functionality
  is not in use (i.e. if allowable_modules_in_use is not set), because
  this function checks based on that array. 
  A future version could assemble a list of modules enabled based on which 
  modules are "init"ed, and check against that array.
 */
int is_pins_module_enabled(char * name) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    while (pins_components[i] != NULL) {
        if (pins_components[i]->mca_query_component != NULL) {
            pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
            int j = 0;
            while (allowable_modules[j] != NULL) {
                if ((!allowable_modules_in_use ||
                     0 == strncmp(module->component->base_version.mca_component_name,
                                  allowable_modules[j], MAX_NAME_SIZE)) &&
                    0 == strncmp(module->component->base_version.mca_component_name,
                                 name, MAX_NAME_SIZE)) {
                    return 1; /* yes, this module is enabled */
                }
                j++;
            }
        }
        i++;
    }
    return 0; /* no, this module is not enabled */
}
