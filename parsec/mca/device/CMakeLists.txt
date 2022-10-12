set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

list(APPEND MCA_${COMPONENT}_SOURCES mca/device/device_gpu.c mca/device/transfer_gpu.c)

set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/device/device.h
                                    mca/device/device_gpu.h)
