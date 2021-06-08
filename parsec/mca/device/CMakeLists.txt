set(MCA_${COMPONENT}_SOURCES mca/device/device.c mca/device/device_gpu.c mca/device/transfer_gpu.c)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mca/device/device.h
        DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/mca/device )
