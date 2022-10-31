set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

if(PARSEC_HAVE_CUDA OR PARSEC_HAVE_HIP)
  list(APPEND MCA_${COMPONENT}_SOURCES mca/device/device_gpu.c mca/device/transfer_gpu.c)
endif()

set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/device/device.h
                                    mca/device/device_gpu.h)
