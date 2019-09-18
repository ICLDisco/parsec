set(MCA_${COMPONENT}_SOURCES mca/device/device.c)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/device/device.h
        DESTINATION include/parsec/mca/device )
