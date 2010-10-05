
add_executable(grapher grapher.c)
target_link_libraries(grapher dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

Add_Subdirectory(dague-compiler)

add_custom_command(
      OUTPUT grapher_create_objects.c
      COMMAND ./grapher_create_objects.sh > grapher_create_objects.c
      MAIN_DEPENDENCY ./grapher_create_objects.sh ../dplasma/lib/LU.h ../dplasma/lib/LU_sd.h ../dplasma/lib/QR.h ../dplasma/lib/TSQR.h ../dplasma/lib/dpotrf.h ../dplasma/lib/sgeqrt.h ../dplasma/lib/spotrf_ll.h ../dplasma/lib/spotrf_rl.h
) 
