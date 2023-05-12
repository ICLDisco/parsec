# Always look for PAPI to avoid CMake complaints about manually-specified
# variables not used.
find_package(PAPI QUIET)

# Save it in the cache we will need it to generate the PaRSECConfig.cmake file
set(PARSEC_HAVE_PAPI ${PAPI_FOUND} CACHE INTERNAL "Support for PAPI has been found")

if (PARSEC_PROF_PINS)
  if (NOT PARSEC_PROF_TRACE)
    message(STATUS "Module ${MODULE} not selectable: PARSEC_PROF_TRACE disabled.")
  else (NOT PARSEC_PROF_TRACE)
    if (PARSEC_HAVE_PAPI)
      set(MCA_${COMPONENT}_${MODULE} ON)
      file(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES mca/pins/pins_papi_utils.c ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
      set(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")

      list(APPEND EXTRA_LIBS ${PAPI_LIBRARIES})
      include_directories( ${PAPI_INCLUDE_DIRS} )
    else (PARSEC_HAVE_PAPI)
      message(STATUS "Module ${MODULE} not selectable: does not have PAPI")
      set(MCA_${COMPONENT}_${MODULE} OFF)
    endif( PARSEC_HAVE_PAPI )
  endif (NOT PARSEC_PROF_TRACE)
else (PARSEC_PROF_PINS)
  message(STATUS "Module ${MODULE} not selectable: PINS disabled.")
  set(MCA_${COMPONENT}_${MODULE} OFF)
endif (PARSEC_PROF_PINS)
