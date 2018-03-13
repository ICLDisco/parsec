#include <stdio.h>
#include <stddef.h>
#include "papi_sde_interface.h"

#pragma weak papi_sde_init
#pragma weak papi_sde_register_counter
#pragma weak papi_sde_describe_counter

papi_handle_t 
__attribute__((weak)) 
papi_sde_init(const char *name_of_library, int event_count)
{
    (void) name_of_library;
    (void) event_count;

    return NULL;
}

void 
__attribute__((weak)) 
papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter)
{
    (void) handle;
    (void) event_name;
    (void) cntr_mode;
    (void) cntr_type;
    (void) counter;

    /* do nothing */
}

void 
__attribute__((weak)) 
papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t func_ptr, void *param )
{
    (void) handle;
    (void) event_name;
    (void) cntr_mode;
    (void) cntr_type;
    (void) func_ptr;
    (void) param;

    /* do nothing */
}

void 
__attribute__((weak)) 
papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description)
{
    (void) handle;
    (void) event_name;
    (void) event_description;

    /* do nothing */
}
