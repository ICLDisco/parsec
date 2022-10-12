#ifndef PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H
#define PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H

#if defined(c_plusplus) || defined(__cplusplus)
#include "sycl/ext/oneapi/backend/level_zero.hpp"

typedef struct {
    sycl::platform platform;
    sycl::device   device;
    sycl::context  context;
    sycl::queue    queue;
} parsec_dpcpp_object_t;

extern "C" {
#endif

void * parsec_dpcpp_queue_create(ze_driver_handle_t ze_driver,
                                 ze_device_handle_t ze_device,
                                 ze_context_handle_t ze_context,
                                 ze_command_queue_handle_t ze_queue);
int parsec_dpcpp_queue_destroy(void *_dpcpp_obj);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif //PARSEC_DEVICE_LEVEL_ZERO_DPCPP_H
