BISON_TARGET(dague_yacc dague.y ${CMAKE_CURRENT_BINARY_DIR}/dague.y.c)
FLEX_TARGET(dague_flex dague.l  ${CMAKE_CURRENT_BINARY_DIR}/dague.l.c)
ADD_FLEX_BISON_DEPENDENCY(dague_flex dague_yacc)

# Bison and Flex are supposed to generate good code.
# But they don't.
# This approach is damageable, because we can't catch C errors in our .l or .y code
# But if we don't do that, we'll keep having reports of compilation warnings forever.
SET_SOURCE_FILES_PROPERTIES(${BISON_dague_yacc_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")
SET_SOURCE_FILES_PROPERTIES(${FLEX_dague_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(daguepp main.c jdf.c jdf2c.c ${BISON_dague_yacc_OUTPUTS} ${FLEX_dague_flex_OUTPUTS})
set_target_properties(daguepp PROPERTIES LINKER_LANGUAGE C)
set_target_properties(daguepp PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(daguepp -lm)

install(TARGETS daguepp RUNTIME DESTINATION bin)
