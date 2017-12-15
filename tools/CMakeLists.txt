Add_Subdirectory(profiling)
#
# If building cross-compile there is no reason
# to build these.
#
IF( NOT CMAKE_CROSSCOMPILING )
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

#
# If building the argobots steering server
#
if (LIBARGOBOTS_FOUND AND BUILD_ARGO_DEMO2)
  set (SERVER_SRCS steering_server/demo_server.c)
  add_executable (demo_server ${SERVER_SRCS})
endif (LIBARGOBOTS_FOUND AND BUILD_ARGO_DEMO2)
