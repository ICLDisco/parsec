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

#define MAX_NAME_SIZE 100 /* arbitrary module name limit for 'safety' */

static int num_modules_enabled = -1;
static char ** modules_enabled;

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

    if (num_modules_enabled < 0)
        num_modules_enabled = 0; /* disable future activations */
#if defined(DAGUE_PROF_TRACE)
    int size = (num_modules_enabled * (MAX_NAME_SIZE + 1)) + 1;
    char * modules_enabled_str = calloc(size, sizeof(char));
    modules_enabled_str[0] = '\0';
#endif /* DAGUE_PROF_TRACE */

    for (; i < PINS_FLAG_COUNT; i++) {
        if (pins_array[i] == NULL)
            pins_array[i] = &pins_empty_callback;
    }
    DEBUG(("Initialized PaRSEC PINS callbacks to pins_empty_callback()"));

    if (NULL != modules_enabled) {
        i = 0;
        pins_components = mca_components_open_bytype("pins");
        while (pins_components[i] != NULL) {
            if (pins_components[i]->mca_query_component != NULL) {
                pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
                int j = 0;
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE)) {
                        if (NULL != module->module.init) {
                            module->module.init(master_context);
                            DEBUG(("Activated PINS module %s.\n",
                                   module->component->base_version.mca_component_name));
                        }
#if defined(DAGUE_PROF_TRACE)
                        /* accumulate the names of modules used */
                        strncat(modules_enabled_str,
                                module->component->base_version.mca_component_name,
                                MAX_NAME_SIZE);
                        strncat(modules_enabled_str, ",", 1);
#endif /* DAGUE_PROF_TRACE */
                        break;
                    }
                    j++;
                }
            }
            i++;
        }
    }
#if defined(DAGUE_PROF_TRACE)
    /* replace trailing comma with \0 */
    if (strlen(modules_enabled_str) > 1)
        modules_enabled_str[strlen(modules_enabled_str) - 2] = '\0';
    dague_profiling_add_information("PINS_MODULES", modules_enabled_str);
    free(modules_enabled_str);
    modules_enabled_str = NULL;
#endif
}

/**
 * pins_fini must call fini methods of all modules
 */
void pins_fini(dague_context_t * master_context) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    if (NULL != modules_enabled) {
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
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE))  {
                        if (NULL != module->module.fini) {
                            module->module.fini(master_context);
                            DEBUG(("Finalized PINS module %s.\n",
                                   module->component->base_version.mca_component_name));
                        }
                        break;
                    }
                    j++;
                }
            }
            i--;
        }
        /* cleanup memory */
        for (i=0; i < num_modules_enabled; i++) {
            if (modules_enabled[i] != NULL) {
                free(modules_enabled[i]);
                modules_enabled[i] = NULL;
            }
        }
        free(modules_enabled);
        modules_enabled = NULL;
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
void pins_thread_init(dague_execution_unit_t * exec_unit) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    if (NULL != modules_enabled) {
        while (pins_components[i] != NULL) {
            if (pins_components[i]->mca_query_component != NULL) {
                pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
                int j = 0;
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE)) {
                        if (NULL != module->module.thread_init)
                            module->module.thread_init(exec_unit);
                        break;
                    }
                    j++;
                }
            }
            i++;
        }
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

    if (modules_enabled != NULL) {
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
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE)) {
                        if (NULL != module->module.thread_fini)
                            module->module.thread_fini(exec_unit);
                        break;
                    }
                    j++;
                }
            }
            i--;
        }
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

    if (NULL != modules_enabled) {
        while (pins_components[i] != NULL) {
            if (pins_components[i]->mca_query_component != NULL) {
                pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
                int j = 0;
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE)) {
                        if (NULL != module->module.handle_init)
                            module->module.handle_init(handle);
                        break;
                    }
                    j++;
                }
            }
            i++;
        }
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

    if (NULL != modules_enabled) {
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
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE)) {
                        if (NULL != module->module.handle_fini)
                            module->module.handle_fini(handle);
                        break;
                    }
                    j++;
                }
            }
            i--;
        }
    }

    parsec_instrument(HANDLE_FINI, NULL, NULL, (void *)handle);
}

/**
 * Addon method to allow for limiting the 'enabled modules.'
 * It is only possible to call this method only before pins_init, so as not to introduce
 * new modules without proper initialization, so as not to overwrite
 * currently-enabled modules, and so as not to cause threading complications.
 * (this method is certainly NOT THREAD-SAFE).
 *
 * The method will only allow itself to be called a single time.
 *
 * The array of modules should be terminated by a NULL pointer.
 */
void pins_enable_modules (const char * const modules[]) {
    int counter = 0;
    if (dague_atomic_cas(&num_modules_enabled, -1, 0)) {
        while (modules[num_modules_enabled] != NULL)
            num_modules_enabled++;
        counter = num_modules_enabled;
        modules_enabled = calloc(num_modules_enabled + 1, sizeof(char));
        if (modules_enabled != NULL) {
            modules_enabled[counter] = NULL;
            for (counter--; counter >= 0; counter--) {
                modules_enabled[counter] = calloc(sizeof(char), MAX_NAME_SIZE + 1);
                if (NULL != modules_enabled[counter]) {
                    strncpy(modules_enabled[counter], modules[counter], MAX_NAME_SIZE);
                    DEBUG(("Allowing PINS module %s to be activated.\n", modules_enabled[counter]));
                }
                else {
                    DEBUG(("Memory allocation failed in "
                           "'pins_set_modules_enabled.' "
                           "Module %s not enabled\n", modules[counter]));
                }
            }
        }
    }
    else
        DEBUG3(("PINS modules have already been set and cannot be set again.\n"));
}

/*
  This function is not currently useful if the module limiting functionality
  is not in use, because this function checks based on that array.
  A future version could assemble a list of modules enabled based on which
  modules are "init"ed, and check against that array.
 */
int pins_is_module_enabled(char * name) {
    dague_pins_module_t * module = NULL;
    int priority = -1;
    int i = 0;

    if (modules_enabled != NULL) {
        while (pins_components[i] != NULL) {
            if (pins_components[i]->mca_query_component != NULL) {
                pins_components[i]->mca_query_component((mca_base_module_t**)&module, &priority);
                int j = 0;
                while (modules_enabled[j] != NULL) {
                    if (0 == strncmp(module->component->base_version.mca_component_name,
                                     modules_enabled[j], MAX_NAME_SIZE) &&
                        0 == strncmp(module->component->base_version.mca_component_name,
                                     name, MAX_NAME_SIZE)) {
                        return 1; /* yes, this module is enabled */
                    }
                    j++;
                }
            }
            i++;
        }
    }
    return 0; /* no, this module is not enabled */
}
