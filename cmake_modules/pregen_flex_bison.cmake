# incorrect arguments passed
if(archive AND missing)
  message(FATAL_ERROR "Wrong argument -Darchive=xyz and -Dmissing=xyz passed to script simultaneously")
endif()

# Missing tool mode: FLEX not found or BISON not found, just check that sources have not been
# tampered with, if possible (i.e., this is a git worktree)
if(missing)
  if(NOT source OR NOT srcdir)
    # Script Argument errors
    message(FATAL_ERROR "Script called incorrectly, you need to set -Dsource=xyz.l and -Dsrcdir=\${PROJECT_SOURCE_DIR}")
  endif()

  execute_process(
    COMMAND git status --porcelain -- ${source}
    WORKING_DIRECTORY ${srcdir}
    RESULT_VARIABLE ret
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(ret)
    # git not found, not a worktree, etc. Assume we are not in a dev environment and just bail-out
    #message(STATUS "git status could not verify if ${source} has been modified: ${stderr}")
    return()
  endif()

  if(NOT stdout)
    # git found, version controlled file, no changes
    #message(STATUS "${source} has not been modified")
    return()
  endif()

  # git found changes, yet we do not have flex/bison, abort
  message(FATAL_ERROR "${source} has been modified: you need ${missing} to regenerate dependent files!")
endif(missing)

# Archive mode: FLEX_FOUND and BISON_FOUND, we can regenerate the bundle
if(archive)
  if(NOT srcdir OR NOT builddir OR NOT archive)
    message(FATAL_ERROR "Missing argument -Dsrcdir=xyz -Dbuilddir=xyz -Darchive=xyz.tar")
  endif()

  file(GLOB_RECURSE generated_files RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "*.l.c" "*.y.c" "*.y.h")
  foreach(file IN LISTS generated_files)
    # strip out the '.[ch]' from '.[ly].[ch]'
    string(REPLACE ".l.c" ".l" source ${file})
    string(REPLACE ".y.c" ".y" source ${source})
    string(REPLACE ".y.h" ".y" source ${source})

    # Check for problems: stray files in build directory
    execute_process(
      COMMAND git ls-files --error-unmatch ${srcdir}/${source}
      WORKING_DIRECTORY ${srcdir}
      RESULT_VARIABLE ret
      OUTPUT_VARIABLE stdout
      ERROR_VARIABLE stderr
      ECHO_ERROR_VARIABLE)
    if(ret)
      # git not found, not a worktree, file not under version control, etc.
      message(FATAL_ERROR "git could not find ${source} under version control; Is this a clean git worktree and build directory (notably stray files in ${CMAKE_CURRENT_BINARY_DIR}?")
    endif()

    # Check for modifications
    execute_process(
      COMMAND git status --porcelain -- ${srcdir}/${source}
      WORKING_DIRECTORY ${srcdir}
      RESULT_VARIABLE ret
      OUTPUT_VARIABLE stdout
      ERROR_VARIABLE stderr
      ECHO_ERROR_VARIABLE)
    if(ret)
      message(FATAL_ERROR "git status could not verify if ${source} has been modified")
    endif()

    if(NOT stdout)
      # we already checked the file exists, so it means no changes
      message(STATUS "${source}: not modified")
      continue()
    endif()

    string(STRIP "${stdout}" stdout)
    if(stdout MATCHES "^R?M ")
      message(STATUS "${stdout}: modified")
      list(APPEND modified ${file})
    else()
      message(FATAL_ERROR "${stdout}: the worktree is not clean (merging/rebase in progress?; removed files in git still in build?); ABORTING; run again when merge is complete and both worktree and build dir are clean")
    endif()
  endforeach()

  if(NOT modified)
    file(RELATIVE_PATH archive_rp ${srcdir} ${archive})

    message(WARNING "No modified .y/.l files. The archive has been created (for when a dev forgot to update the archive in a prior commit). Should you -really- commit the resulting archive? You can revert changes to the worktree with `git checkout -- ${archive_rp}`")
  endif()
  file(ARCHIVE_CREATE
    OUTPUT ${archive}
    PATHS ${generated_files})
endif(archive)

