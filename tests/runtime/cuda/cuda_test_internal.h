/*
 * Copyright (c) 2019-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef CUDA_TEST_GET_BEST_DEVICE_H
#define CUDA_TEST_GET_BEST_DEVICE_H

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "tests/tests_timing.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include "parsec/mca/device/cuda/device_cuda.h"
#endif

int parsec_get_best_device_check(parsec_context_t *parsec,
                parsec_tiled_matrix_t *A);

#endif /* CUDA_TEST_GET_BEST_DEVICE_H */
