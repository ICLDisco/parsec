
add_executable(grapher grapher.c)
ADD_DEPENDENCIES(grapher grapher_create_objects.c)
target_link_libraries(grapher dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

Add_Subdirectory(dague-compiler)

add_custom_target(
      grapher_create_objects.c
      COMMAND ./grapher_create_objects.sh > grapher_create_objects.c
      MAIN_DEPENDENCY ./grapher_create_objects.sh
      DEPENDS ../dplasma/lib/LU.h
      DEPENDS ../dplasma/lib/LU_sd.h
      DEPENDS ../dplasma/lib/QR.h
      DEPENDS ../dplasma/lib/TSQR.h
      DEPENDS ../dplasma/lib/dpotrf.h
      DEPENDS ../dplasma/lib/sgeqrt.h
      DEPENDS ../dplasma/lib/spotrf_ll.h
      DEPENDS ../dplasma/lib/spotrf_rl.h
)
