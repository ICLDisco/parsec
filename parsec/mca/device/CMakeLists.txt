set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

if( PARSEC_HAVE_CUDA )
  LIST(APPEND MCA_${COMPONENT}_SOURCES mca/device/device_gpu.c mca/device/transfer_gpu.c)
endif( PARSEC_HAVE_CUDA )

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/device.h ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/device_gpu.h
        DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/mca/device )
