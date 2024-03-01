find_program(OTF2_CONFIG NAMES otf2-config
             PATHS "${OTF2_DIR}/bin"
                   "${OTF2_CONFIG_PATH}"
                   "/opt/scorep/bin")

if(NOT OTF2_CONFIG)
    message(STATUS "no otf2-config found")
    set(OTF2_FOUND false)
else(NOT OTF2_CONFIG)
    message(STATUS "OTF2 library found. (using ${OTF2_CONFIG})")
    #
    # Get the OTF2 version. The output of otf2-config depends on the version itself...
    #   Newish versions follows the template "otf2-config: version x.y.z"
    #   Oldish versions directly return the version number.
    execute_process(COMMAND ${OTF2_CONFIG} "--version" OUTPUT_VARIABLE OTF2_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(OTF2_OUTPUT MATCHES ".*version.*")
      string(REPLACE " " ";" OTF2_LIST ${OTF2_OUTPUT})
      list(GET OTF2_LIST 2 OTF2_VERSION)
    else(OTF2_OUTPUT MATCHES ".*version.*")
      set(OTF2_VERSION "${OTF2_OUTPUT}")
    endif(OTF2_OUTPUT MATCHES ".*version.*")
    if(OTF2_VERSION VERSION_LESS ${OTF2_FIND_VERSION})
      message(STATUS "OTF2 library is version ${OTF2_VERSION}; Version 2.1.1 or later is needed.")
      set(OTF2_FOUND false)
    else(OTF2_VERSION VERSION_LESS ${OTF2_FIND_VERSION})
      execute_process(COMMAND ${OTF2_CONFIG} "--cflags" OUTPUT_VARIABLE OTF2_CONFIG_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

      string(REGEX MATCHALL "-I[^ ]*" OTF2_CONFIG_INCLUDES "${OTF2_CONFIG_FLAGS}")
      foreach(inc ${OTF2_CONFIG_INCLUDES})
        string(SUBSTRING ${inc} 2 -1 inc2)
        list(APPEND OTF2_INCLUDE_DIRS ${inc2})
      endforeach()

      string(REGEX MATCHALL "(^| +)-[^I][^ ]*" OTF2_CONFIG_CXXFLAGS "${OTF2_CONFIG_FLAGS}")
      foreach(flag ${OTF2_CONFIG_CXXFLAGS})
        string(STRIP ${flag} flag)
        list(APPEND OTF2_CXX_FLAGS ${flag})
      endforeach()

      unset(OTF2_CONFIG_FLAGS)
      unset(OTF2_CONFIG_INCLUDES)
      unset(OTF2_CONFIG_CXXFLAGS)

      execute_process(COMMAND ${OTF2_CONFIG} "--ldflags" OUTPUT_VARIABLE _LINK_LD_ARGS OUTPUT_STRIP_TRAILING_WHITESPACE)
      string( REPLACE " " ";" _LINK_LD_ARGS ${_LINK_LD_ARGS} )
      foreach( _ARG ${_LINK_LD_ARGS} )
        if(${_ARG} MATCHES "^-L")
          string(REGEX REPLACE "^-L" "" _ARG ${_ARG})
          set(OTF2_LINK_DIRS ${OTF2_LINK_DIRS} ${_ARG})
        else(${_ARG} MATCHES "^-L")
          set(OTF2_LINK_OPTIONS ${OTF2_LINK_OPTIONS} ${_ARG})
        endif(${_ARG} MATCHES "^-L")
      endforeach(_ARG)

      execute_process(COMMAND ${OTF2_CONFIG} "--libs" OUTPUT_VARIABLE _LINK_LD_ARGS OUTPUT_STRIP_TRAILING_WHITESPACE)
      string( REPLACE " " ";" _LINK_LD_ARGS ${_LINK_LD_ARGS} )
      foreach( _ARG ${_LINK_LD_ARGS} )
        if(${_ARG} MATCHES "^-l")
          string(REGEX MATCH "otf2" _OTF2_LIBRARY_TEST ${_ARG})
          string(REGEX REPLACE "^-l" "" _ARG ${_ARG})
          find_library(_OTF2_LIB_FROM_ARG NAMES ${_ARG}
            PATHS
            ${OTF2_LINK_DIRS}
            )
          if(_OTF2_LIB_FROM_ARG)
            if(_OTF2_LIBRARY_TEST)
              set(OTF2_LIBRARY ${_OTF2_LIB_FROM_ARG})
            endif(_OTF2_LIBRARY_TEST)
            set(OTF2_LIBRARIES ${OTF2_LIBRARIES} ${_OTF2_LIB_FROM_ARG})
          endif(_OTF2_LIB_FROM_ARG)
          unset(_OTF2_LIB_FROM_ARG CACHE)
        endif(${_ARG} MATCHES "^-l")
      endforeach(_ARG)

      set(OTF2_FOUND true)
    endif(OTF2_VERSION VERSION_LESS ${OTF2_FIND_VERSION})
endif(NOT OTF2_CONFIG)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OTF2 DEFAULT_MSG
    OTF2_CONFIG
    OTF2_LIBRARIES
    OTF2_INCLUDE_DIRS
)

if(NOT TARGET OTF2::OTF2)
  add_library(OTF2::OTF2 UNKNOWN IMPORTED GLOBAL)
endif()

set_property(TARGET OTF2::OTF2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${OTF2_INCLUDE_DIRS}")
set_property(TARGET OTF2::OTF2 PROPERTY INTERFACE_LINK_LIBRARIES "${OTF2_LIBRARIES}")
set_property(TARGET OTF2::OTF2 PROPERTY INTERFACE_LINK_DIRECTORIES "${OTF2_LINK_DIRS}")
set_property(TARGET OTF2::OTF2 PROPERTY INTERFACE_LINK_OPTIONS "${OTF2_LINK_OPTIONS}")
set_property(TARGET OTF2::OTF2 PROPERTY IMPORTED_LOCATION "${OTF2_LIBRARY}")

mark_as_advanced(OTF2_CONFIG)
