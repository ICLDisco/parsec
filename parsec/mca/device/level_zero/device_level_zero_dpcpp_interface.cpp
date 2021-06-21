#include "level_zero/ze_api.h"
#include "device_level_zero_dpcpp.h"

void * parsec_dpcpp_queue_create(ze_driver_handle_t ze_driver,
                                 ze_device_handle_t ze_device,
                                 ze_context_handle_t ze_context,
                                 ze_command_queue_handle_t ze_queue)
{
    parsec_dpcpp_object_t *dpcpp_obj;
    std::vector<sycl::device>devices;

    dpcpp_obj = new parsec_dpcpp_object_t;
    dpcpp_obj->platform = sycl::level_zero::make<sycl::platform>(ze_driver);
    dpcpp_obj->device = sycl::level_zero::make<sycl::device>(dpcpp_obj->platform, ze_device);
    devices.push_back(dpcpp_obj->device);
    dpcpp_obj->context = sycl::level_zero::make<sycl::context>(devices, ze_context);
    dpcpp_obj->queue = sycl::level_zero::make<sycl::queue>(dpcpp_obj->context, ze_queue);

    return static_cast<void*>(dpcpp_obj);
}

int parsec_dpcpp_queue_destroy(void *_dpcpp_obj)
{
    parsec_dpcpp_object_t *dpcpp_obj = reinterpret_cast<parsec_dpcpp_object_t*>(_dpcpp_obj);
    delete dpcpp_obj;

    return 0;
}
