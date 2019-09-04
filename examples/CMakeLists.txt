function(parsec_addexample lang target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang})
  target_link_libraries(${target}
    PRIVATE
      parsec
      $<$<STREQUAL:Fortran,${lang}>:parsec_fortran>)
endfunction(parsec_addexample)

parsec_addexample(C ex00 Ex00_StartStop.c)
parsec_addexample(C ex01)
target_ptg_sources(ex01 PRIVATE "Ex01_HelloWorld.jdf")
parsec_addexample(C ex02)
target_ptg_sources(ex02 PRIVATE "Ex02_Chain.jdf")
parsec_addexample(C ex03)
target_ptg_sources(ex03 PRIVATE "Ex03_ChainMPI.jdf")
parsec_addexample(C ex04)
target_ptg_sources(ex04 PRIVATE "Ex04_ChainData.jdf")
parsec_addexample(C ex05)
target_ptg_sources(ex05 PRIVATE "Ex05_Broadcast.jdf")
parsec_addexample(C ex06)
target_ptg_sources(ex06 PRIVATE "Ex06_RAW.jdf")
parsec_addexample(C ex07)
target_ptg_sources(ex07 PRIVATE "Ex07_RAW_CTL.jdf")

add_Subdirectory(interfaces/superscalar)
