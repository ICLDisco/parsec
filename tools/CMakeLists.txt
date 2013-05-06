Add_Subdirectory(dague-compiler)
Add_Subdirectory(profiling)
#
# If building cross-compile there is no reason
# to build these.
#
IF( NOT CMAKE_CROSSCOMPILING )
    IF (CMAKE_CXX_COMPILER_WORKS)
        Add_Subdirectory(q2j)
    ENDIF (CMAKE_CXX_COMPILER_WORKS)

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
ENDIF( NOT CMAKE_CROSSCOMPILING )

