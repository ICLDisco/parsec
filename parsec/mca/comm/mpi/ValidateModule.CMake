# The MPI backend is the current implementation of the distributed
# communication engine.  Build it only when PaRSEC itself was configured with
# MPI support; otherwise the runtime keeps using the existing local-only path.
if(PARSEC_HAVE_MPI)
  set(MCA_${COMPONENT}_${MODULE} ON)
  file(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  set(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
else()
  set(MCA_${COMPONENT}_${MODULE} OFF)
endif()
