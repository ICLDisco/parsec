/*
 * Copyright (c) 2021      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_LEVEL_ZERO_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_LEVEL_ZERO_INTERNAL_H_HAS_BEEN_INCLUDED

#include "parsec.h"
#include "parsec/mca/device/level_zero/device_level_zero.h"

#if defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT)

BEGIN_C_DECLS

/* From MCA parameters */
extern int parsec_device_level_zero_enabled;
extern int parsec_level_zero_max_streams;
extern int parsec_level_zero_memory_block_size, parsec_level_zero_memory_percentage, parsec_level_zero_memory_number_of_blocks;
extern char* parsec_level_zero_lib_path;

PARSEC_DECLSPEC extern const parsec_device_module_t parsec_device_level_zero_module;

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

int parsec_level_zero_module_init( int device_id, parsec_device_level_zero_driver_t *driver, ze_device_handle_t ze_device,
                                   ze_device_properties_t *device_properties, parsec_device_module_t** module );
int parsec_level_zero_module_fini(parsec_device_module_t* device);

END_C_DECLS

#if defined(PARSEC_PROF_TRACE)
typedef struct {
    uint64_t size;
    uint64_t data_key;
    uint64_t dc_id;
} parsec_device_level_zero_memory_prof_info_t;
#define PARSEC_DEVICE_LEVEL_ZERO_MEMORY_PROF_INFO_CONVERTER "size{int64_t};data_key{uint64_t};dc_id{uint64_t}"
int parsec_device_level_zero_one_profiling_stream_per_gpu_stream;
#endif /* PARSEC_PROF_TRACE */

#endif /* defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT) */

#endif  /* PARSEC_DEVICE_LEVEL_ZERO_INTERNAL_H_HAS_BEEN_INCLUDED */
