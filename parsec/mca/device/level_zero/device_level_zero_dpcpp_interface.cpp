#include <level_zero/ze_api.h>
#include "parsec/mca/device/level_zero/device_level_zero_dpcpp.h"
#include "parsec/mca/device/level_zero/device_level_zero_internal.h"

parsec_sycl_wrapper_platform_t *parsec_sycl_wrapper_platform_create(ze_driver_handle_t ze_driver)
{
    parsec_sycl_wrapper_platform_t *res = new parsec_sycl_wrapper_platform_t;
    res->platform = sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(ze_driver);
    return res;
}

void parsec_sycl_wrapper_platform_add_context(parsec_sycl_wrapper_platform_t *swp, ze_context_handle_t ze_context, parsec_sycl_wrapper_device_t **swd, uint32_t num_device)
{
    std::vector<sycl::device>devices;

    for(uint32_t i = 0; i < num_device; i++) {
        devices.push_back(swd[i]->device);
    }
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> hContextInteropInput = {ze_context, devices};
    swp->context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(hContextInteropInput);
}

parsec_sycl_wrapper_device_t *parsec_sycl_wrapper_device_create(ze_device_handle_t ze_device)
{
    parsec_sycl_wrapper_device_t *res = new parsec_sycl_wrapper_device_t;

    res->device = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(ze_device);

    return res;
}

parsec_sycl_wrapper_queue_t *parsec_sycl_wrapper_queue_create(parsec_sycl_wrapper_platform_t *swp, parsec_sycl_wrapper_device_t *swd, ze_command_queue_handle_t ze_queue)
{
    parsec_sycl_wrapper_queue_t *swq = new parsec_sycl_wrapper_queue_t;
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> hQueueInteropInput = { ze_queue, swd->device };
    swq->queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(hQueueInteropInput, swp->context);
    return swq;
}

int parsec_sycl_wrapper_platform_destroy(parsec_sycl_wrapper_platform_t *swp)
{
    if(nullptr != swp)
        delete swp;
    return 0;
}

int parsec_sycl_wrapper_device_destroy(parsec_sycl_wrapper_device_t *swd)
{
    if(nullptr != swd)
        delete swd;
    return 0;
}

int parsec_sycl_wrapper_queue_destroy(parsec_sycl_wrapper_queue_t *swq)
{
    if(nullptr != swq)
        delete swq;
    return 0;
}
