#include "level_zero/ze_api.h"
#include "interface.dpcpp.h"

sycl_wrapper_driver_t *sycl_wrapper_platform_create(ze_driver_handle_t ze_driver)
{
    sycl_wrapper_driver_t *res = new sycl_wrapper_driver_t;
    res->platform = sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(ze_driver);
    return res;
}

void sycl_wrapper_platform_add_context(sycl_wrapper_driver_t *swp, ze_context_handle_t ze_context, sycl_wrapper_device_t **swd, uint32_t num_device)
{
    std::vector<sycl::device>devices;

    for(uint32_t i = 0; i < num_device; i++) {
        devices.push_back(swd[i]->device);
    }
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> hContextInteropInput = {ze_context, devices};
    swp->context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(hContextInteropInput);
}

sycl_wrapper_device_t *sycl_wrapper_device_create(ze_device_handle_t ze_device)
{
    sycl_wrapper_device_t *res = new sycl_wrapper_device_t;

    res->device = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(ze_device);

    return res;
}

sycl_wrapper_queue_t *sycl_wrapper_queue_create(sycl_wrapper_driver_t *swp, sycl_wrapper_device_t *swd, ze_command_queue_handle_t ze_queue)
{
    sycl_wrapper_queue_t *swq = new sycl_wrapper_queue_t;
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> hQueueInteropInput = { ze_queue, swd->device };
    swq->queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(hQueueInteropInput, swp->context);
    return swq;
}

int sycl_wrapper_driver_destroy(sycl_wrapper_driver_t *swp)
{
    delete swp;
    return 0;
}

int sycl_wrapper_device_destroy(sycl_wrapper_device_t *swd)
{
    delete swd;
    return 0;
}

int sycl_wrapper_queue_destroy(sycl_wrapper_queue_t *swq)
{
    delete swq;
    return 0;
}

