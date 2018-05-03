#include <stdio.h>
#include <stddef.h>
#include "papi_sde_interface.h"

#pragma weak papi_sde_init
#pragma weak papi_sde_register_counter
#pragma weak papi_sde_register_fp_counter
#pragma weak papi_sde_unregister_counter
#pragma weak papi_sde_describe_counter

papi_handle_t 
__attribute__((weak)) 
papi_sde_init(const char *name_of_library)
{
    (void) name_of_library;

    return NULL;
}

int 
__attribute__((weak)) 
papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter)
{
    (void) handle;
    (void) event_name;
    (void) cntr_mode;
    (void) cntr_type;
    (void) counter;

    /* do nothing */

    return 0;
}

int 
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

    return 0;
}

int 
__attribute__((weak))
papi_sde_unregister_counter( void *handle, const char *event_name)
{
    (void) handle;
    (void) event_name;

    /* do nothing */

    return 0;
}

int 
__attribute__((weak)) 
papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description)
{
    (void) handle;
    (void) event_name;
    (void) event_description;

    /* do nothing */

    return 0;
}

int
__attribute__((weak)) 
papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags)
{
    (void) handle;
    (void) event_name;
    (void) group_name;
    (void) group_flags;

    /* do nothing */

    return 0;
}
