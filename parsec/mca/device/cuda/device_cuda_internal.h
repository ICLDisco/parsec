/*
 * Copyright (c) 2010-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED

#include "parsec/mca/device/cuda/device_cuda.h"

#if defined(PARSEC_HAVE_CUDA)

BEGIN_C_DECLS

/* From MCA parameters */
extern int use_cuda_index, use_cuda;
extern int cuda_mask, cuda_verbosity;
extern int cuda_memory_block_size, cuda_memory_percentage, cuda_memory_number_of_blocks;
extern char* cuda_lib_path;
extern int32_t parsec_CUDA_d2h_max_flows;
extern int32_t parsec_CUDA_sort_pending_list;

PARSEC_DECLSPEC extern const parsec_device_module_t parsec_device_cuda_module;

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

int parsec_cuda_module_init( int device, parsec_device_module_t** module );
int parsec_cuda_module_fini(parsec_device_module_t* device);

END_C_DECLS

#endif /* defined(PARSEC_HAVE_CUDA) */

#endif  /* PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED */
