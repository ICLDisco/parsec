# This file provides a module with the opportunity to check the availability
# of prerequisite libraries and decide if the module can or cannot be build.
# It also allow the component to add the source files necessary to build
# the component, and decide how the component will be built (shared or
# static).
# - if shared then the component should create it's own targets and install
#   them in the PaRSEC shared object directory (lib/parsec)
# - if static the component can add sources either via a CMAKE variable
#   ($MCA_${COMPONENT}_${MODULE}_SOURCES) or add them directly to the
#   parsec target.

# if the component will be built or not
SET(MCA_${COMPONENT}_${MODULE} ON)

# create a cmake variable with all the sources to be included. This variable will
# be automatically included in the parsec target sources
FILE(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)

# define the name of the static compoenent that behave as a constructor for the object
# when linked statically. It points to a visible base_compomponent_t exposing the
# open, close, and query functions.
SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")

# add the list of files to be installed. This step can be conditional, and must
# take in account the WITH_DEVEL_HEADER request.

# install(FILES
#   ${CMAKE_CURRENT_SOURCE_DIR}/${COMPONENT}/${MODULE}/file.h
#   DESTINATION include/parsec/mca/${COMPONENT}/${MODULE}/ )
