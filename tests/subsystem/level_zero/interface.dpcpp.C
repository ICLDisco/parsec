#include "level_zero/ze_api.h"
#include "interface.dpcpp.h"

sycl_wrapper_t *sycl_queue_create(ze_driver_handle_t ze_driver,
                         ze_device_handle_t ze_device,
                         ze_context_handle_t ze_context,
                         ze_command_queue_handle_t ze_queue)
{
    std::vector<sycl::device>devices;

    for(uint32_t i = 0; i < num_device; i++) {
        devices.push_back(swd[i]->device);
    }
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> hContextInteropInput = {ze_context, devices};
    res->context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(hContextInteropInput);
    res->queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(ze_queue, res->context);

    return res;
}

int sycl_queue_destroy(sycl_wrapper_t *sycl_obj)
{
    delete sycl_obj;
    return 0;
}

