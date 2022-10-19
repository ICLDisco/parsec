#ifndef INTERFACE_DPCPP_H
#define INTERFACE_DPCPP_H

typedef struct sycl_wrapper_driver_s sycl_wrapper_driver_t;
typedef struct sycl_wrapper_device_s sycl_wrapper_device_t;
typedef struct sycl_wrapper_queue_s  sycl_wrapper_queue_t;

#if defined(c_plusplus) || defined(__cplusplus)
#include "sycl/ext/oneapi/backend/level_zero.hpp"

struct sycl_wrapper_driver_s {
    sycl::platform platform;
    sycl::context  context;
};

struct sycl_wrapper_device_s {
    sycl::device device;
};

struct sycl_wrapper_queue_s {
    sycl::queue queue;
};

extern "C" {
#endif

sycl_wrapper_driver_t *sycl_wrapper_platform_create(ze_driver_handle_t ze_driver);
void sycl_wrapper_platform_add_context(sycl_wrapper_driver_t *swp, ze_context_handle_t ze_context, sycl_wrapper_device_t **swd, uint32_t num_device);
sycl_wrapper_device_t *sycl_wrapper_device_create(ze_device_handle_t ze_device);
sycl_wrapper_queue_t  *sycl_wrapper_queue_create(sycl_wrapper_driver_t *swp, sycl_wrapper_device_t *swd, ze_command_queue_handle_t ze_queue);

sycl_wrapper_t *sycl_queue_create(ze_driver_handle_t ze_driver,
                         ze_device_handle_t ze_device,
                         ze_context_handle_t ze_context,
                         ze_command_queue_handle_t ze_queue);
int sycl_queue_destroy(sycl_wrapper_t *_queue);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif //INTERFACE_DPCPP_H
