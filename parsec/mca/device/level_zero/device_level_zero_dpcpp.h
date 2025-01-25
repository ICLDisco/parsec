#ifndef PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H
#define PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H
/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

typedef struct parsec_sycl_wrapper_platform_s parsec_sycl_wrapper_platform_t;
typedef struct parsec_sycl_wrapper_device_s parsec_sycl_wrapper_device_t;
typedef struct parsec_sycl_wrapper_queue_s  parsec_sycl_wrapper_queue_t;

#include <level_zero/ze_api.h>

#if defined(c_plusplus) || defined(__cplusplus)
#include <sycl/ext/oneapi/backend/level_zero.hpp>

struct parsec_sycl_wrapper_platform_s {
    sycl::platform platform;
    sycl::context  context;
};

struct parsec_sycl_wrapper_device_s {
    sycl::device device;
};

struct parsec_sycl_wrapper_queue_s {
    sycl::queue queue;
};

extern "C" {
#endif

parsec_sycl_wrapper_platform_t *parsec_sycl_wrapper_platform_create(ze_driver_handle_t ze_driver);
void parsec_sycl_wrapper_platform_add_context(parsec_sycl_wrapper_platform_t *swp, ze_context_handle_t ze_context, parsec_sycl_wrapper_device_t **swd, uint32_t num_device);
parsec_sycl_wrapper_device_t *parsec_sycl_wrapper_device_create(ze_device_handle_t ze_device);
parsec_sycl_wrapper_queue_t  *parsec_sycl_wrapper_queue_create(parsec_sycl_wrapper_platform_t *swp, parsec_sycl_wrapper_device_t *swd, ze_command_queue_handle_t ze_queue);

int parsec_sycl_wrapper_platform_destroy(parsec_sycl_wrapper_platform_t *swp);
int parsec_sycl_wrapper_device_destroy(parsec_sycl_wrapper_device_t *swd);
int parsec_sycl_wrapper_queue_destroy(parsec_sycl_wrapper_queue_t *swq);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif //PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H
