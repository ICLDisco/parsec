#include_directories(BEFORE ../ )

add_executable(lhr lhr.c cholesky.c)
set_target_properties(lhr PROPERTIES LINKER_LANGUAGE C)
set_target_properties(lhr PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(lhr ${EXTRA_LIBS} dague -lm)

add_custom_command (
  OUTPUT cholesky.c cholesky.h
  COMMAND ../dc -i cholesky_ideal.jdf -o cholesky
  MAIN_DEPENDENCY cholesky_ideal.jdf
  DEPENDS dc
  )
 
