# Find the Cython compiler.
#
# This code sets the following variables:
#
#  CYTHON_EXECUTABLE
#
# See also UseCython.cmake

#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Use the Cython executable that lives next to the Python executable
# if it is a local installation.
find_package( PythonInterp )
if( PYTHONINTERP_FOUND )
  get_filename_component( _python_path ${PYTHON_EXECUTABLE} PATH )
  find_program( CYTHON_EXECUTABLE
    NAMES cython cython.bat
    HINTS ${_python_path}
    )
else()
  find_program( CYTHON_EXECUTABLE
    NAMES cython cython.bat
    )
endif()

#This is hot fix. A better approach should check
# for the version only if necessary.
# target version should not be hard coded.
if( CYTHON_EXECUTABLE )
    execute_process(COMMAND ${CYTHON_EXECUTABLE} --version
                    RESULT_VARIABLE CYTHON_RESULT
                    ERROR_VARIABLE CYTHON_OUTPUT
                    OUTPUT_QUIET
                    ERROR_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "Cython version " "" CYTHON_VERSION "${CYTHON_OUTPUT}")
    if( "${CYTHON_VERSION}" VERSION_LESS "0.19.1" )
        MESSAGE(STATUS "Cython version ${CYTHON_VERSION} found -- too old for current code")
        unset(CYTHON_EXECUTABLE CACHE)
    else()
        MESSAGE(STATUS "Cython version ${CYTHON_VERSION} found")
    endif()
endif()

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Cython REQUIRED_VARS CYTHON_EXECUTABLE )

mark_as_advanced( CYTHON_EXECUTABLE )

