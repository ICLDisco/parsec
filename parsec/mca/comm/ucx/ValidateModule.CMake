# The UCX backend is optional and is bootstrapped through PMIx.  The first
# implementation is intentionally restricted to builds where PaRSEC owns the
# datatype representation, because MPI-enabled builds still expose MPI_Datatype
# as parsec_datatype_t.
if(PARSEC_HAVE_UCX AND PARSEC_HAVE_PMIX AND NOT PARSEC_HAVE_MPI)
  set(MCA_${COMPONENT}_${MODULE} ON)
  file(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  set(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
else()
  if(PARSEC_DIST_WITH_UCX AND PARSEC_HAVE_MPI)
    message(STATUS "Module ${MODULE} not selectable: UCX currently requires PARSEC_DIST_WITH_MPI=OFF")
  elseif(PARSEC_DIST_WITH_UCX)
    message(STATUS "Module ${MODULE} not selectable: UCX and PMIx were not both found")
  endif()
  set(MCA_${COMPONENT}_${MODULE} OFF)
endif()
