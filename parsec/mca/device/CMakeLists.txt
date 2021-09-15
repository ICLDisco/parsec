set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

if( PARSEC_HAVE_CUDA )
  LIST(APPEND MCA_${COMPONENT}_SOURCES mca/device/device_gpu.c mca/device/transfer_gpu.c)
endif( PARSEC_HAVE_CUDA )

set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/device/device.h
                                    mca/device/device_gpu.h)
