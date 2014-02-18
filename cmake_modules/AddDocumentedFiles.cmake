#
# See http://stackoverflow.com/questions/13103018/how-to-prepend-cmake-current-source-dir-to-all-files-in-variable
#
function( add_documented_files _var _path )
    unset( _tmp )
    foreach( _src ${ARGN} )
        list(APPEND _tmp ${_path}${_src} )
    endforeach()
    set( ${_var} ${${_var}} ${_tmp} CACHE INTERNAL "List of sources with internal documentation" )
endfunction()
