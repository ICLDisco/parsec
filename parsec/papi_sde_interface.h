#ifndef PAPI_SDE_INTERFACE_H
#define PAPI_SDE_INTERFACE_H

#define SDE_RO       0x00
#define SDE_RW       0x01
#define SDE_DELTA    0x00
#define SDE_INSTANT  0x10

#define PAPI_SDE_long_long 0x0
#define PAPI_SDE_int       0x1
#define PAPI_SDE_double    0x2
#define PAPI_SDE_float     0x3

typedef long long int (*papi_sde_fptr_t)( void * );
typedef void* papi_handle_t;

papi_handle_t papi_sde_init(const char *name_of_library, int event_count);
void papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter);
void papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t func_ptr, void *param);
void papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description );

#endif
