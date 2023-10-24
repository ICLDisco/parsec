set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

if(PARSEC_HAVE_CUDA OR PARSEC_HAVE_HIP)
  list(APPEND MCA_${COMPONENT}_SOURCES mca/device/device_gpu.c mca/device/transfer_gpu.c)
endif()

set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/device/device.h
                                    mca/device/device_gpu.h)

set(PARSEC_HAVE_DEV_CPU_SUPPORT 1 CACHE BOOL "PaRSEC has support for CPU kernels")
set(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT 0 CACHE BOOL  "PaRSEC has support for CPU kernels")
if(PARSEC_HAVE_CUDA)
  set(PARSEC_HAVE_DEV_CUDA_SUPPORT 1 CACHE BOOL "PaRSEC support for CUDA")
endif(PARSEC_HAVE_CUDA)
if(PARSEC_HAVE_HIP)
  set(PARSEC_HAVE_DEV_HIP_SUPPORT 1 CACHE BOOL "PaRSEC support for HIP")
endif(PARSEC_HAVE_HIP)
