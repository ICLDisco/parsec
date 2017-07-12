# Prevent cmake from complaining about variable substitution
IF (CMAKE_VERSION VERSION_GREATER "3.1")
  cmake_policy(SET CMP0054 NEW)
ENDIF (CMAKE_VERSION VERSION_GREATER "3.1")

#
# Shared Memory Testings
#

set(PTG2DTD "ptg2dtd")
set(PTD2DTD_OPTIONS "--;--mca;mca_pins;ptg_to_dtd")
set(OPTIONS "-x;-v=5")
#set(OPTIONS "")

macro(dplasma_add_test m_nameradix m_dependsradix m_types)
  foreach (m_type "${m_types}")
    if (m_type MATCHES "gpu")
      set(m_gpus "${CTEST_GPU_LAUNCHER_OPTIONS}")
    else()
      unset(m_gpus)
    endif()
    if (m_type MATCHES "shm")
      string(REPLACE "${PTG2DTD}_" "" m_suffix ${m_type})
      add_test(dplasma_${prec}${m_nameradix}_${m_suffix} ${SHM_TEST_CMD_LIST} ${m_gpus} ./testing_${prec}${m_nameradix} ${ARGN})
      set_tests_properties(dplasma_${prec}${m_nameradix}_${m_suffix} PROPERTIES LABELS "dplasma;shm")
      if (NOT "" STREQUAL "${m_dependsradix}")
        set_tests_properties(dplasma_${prec}${m_nameradix}_${m_suffix} PROPERTIES DEPENDS "launcher_shm;dplasma_${prec}${m_dependsradix}_shm")
      endif()
      if (m_type MATCHES ${PTG2DTD})
        add_test(dplasma_${prec}${m_nameradix}_${PTG2DTD}_${m_suffix} ${SHM_TEST_CMD_LIST} ${m_gpus} ./testing_${prec}${m_nameradix} ${ARGN} ${PTG2DTD_OPTIONS})
        set_tests_properties(dplasma_${prec}${m_nameradix}_${PTG2DTD}_${m_suffix} PROPERTIES DEPENDS dplasma_${prec}${m_nameradix}_${m_suffix} LABELS "dplasma;shm;${PTG2DTD}")
      endif()
    endif()

    if (m_type MATCHES "mpi")
      string(REGEX REPLACE ".*mpi:" "" m_procs ${m_type})
      string(REGEX REPLACE ":.*" "" m_suffix ${m_type})
     add_test(dplasma_${prec}${m_nameradix}_${m_suffix} ${MPI_TEST_CMD_LIST} ${m_procs} ${m_gpus} ./testing_${prec}${m_nameradix} ${ARGN})
      set_tests_properties(dplasma_${prec}${m_nameradix}_${m_suffix} PROPERTIES LABELS "dplasma;mpi")
      if (NOT "" STREQUAL "${m_dependsradix}")
        set_tests_properties(dplasma_${prec}${m_nameradix}_${m_suffix} PROPERTIES DEPENDS "launcher_mpi;dplasma_${prec}${m_dependsradix}_mpi")
      else()
        set_tests_properties(dplasma_${prec}${m_nameradix}_mpi PROPERTIES DEPENDS launcher_mpi)
      endif()
    endif()
  endforeach()
endmacro()

#
# Check BLAS/Lapack subroutines in shared memory
#
foreach(prec ${DPLASMA_PRECISIONS} )

  # check the control and test matrices generation (zplrnt, zplghe, zplgsy, zpltmg) in shared memory
  dplasma_add_test(print "" ${PTG2DTD}_shm -N 64 -t 7 ${OPTIONS})

  # check the norms that are used in all other testings
  dplasma_add_test(lange print ${PTG2DTD}_shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})

  dplasma_add_test(lanm2 print shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})

  # Need to add testings on zlacpy, zlaset, zgeadd, zlascal, zger, (zlaswp?)

  # BLAS Shared memory
  dplasma_add_test(trmm lange ${PTG2DTD}_shm -M 106 -N 150 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(trsm trmm ${PTG2DTD}_shm -M 106 -N 150 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(gemm lange ${PTG2DTD}_shm -M 106 -N 150 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(symm lange ${PTG2DTD}_shm -M 106 -N 150 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(syrk "" ${PTG2DTD}_shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(syr2k "" ${PTG2DTD}_shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})

  if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
    dplasma_add_test(hemm "" ${PTG2DTD}_shm -M 106 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(herk "" ${PTG2DTD}_shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(her2k "" ${PTG2DTD}_shm -M 287 -N 283 -K 97 -t 56 ${OPTIONS})
  endif()

  # Cholesky
  dplasma_add_test(potrf trsm ${PTG2DTD}_shm -N 378 -t 93        ${OPTIONS})
  dplasma_add_test(posv potrf ${PTG2DTD}_shm -N 457 -t 93 -K 367 ${OPTIONS})

  # QR / LQ
  dplasma_add_test(geqrf "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(gelqf "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
    dplasma_add_test(unmqr "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(unmlq "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  else()
    dplasma_add_test(ormqr "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(ormlq "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  endif()

  # QR / LQ: HQR
  dplasma_add_test(geqrf_hqr "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(gelqf_hqr "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
    dplasma_add_test(unmqr_hqr "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(unmlq_hqr "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  else()
    dplasma_add_test(ormqr_hqr "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(ormlq_hqr "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  endif()

  # QR / LQ: systolic
  dplasma_add_test(geqrf_systolic "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
  dplasma_add_test(gelqf_systolic "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
    dplasma_add_test(unmqr_systolic "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(unmlq_systolic "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  else()
    dplasma_add_test(ormqr_systolic "" ${PTG2DTD}_shm -M 487 -N 283 -K 97 -t 56 ${OPTIONS})
    dplasma_add_test(ormlq_systolic "" ${PTG2DTD}_shm -M 287 -N 383 -K 97 -t 56 ${OPTIONS})
  endif()

  # LU
  dplasma_add_test(getrf "" shm -N 378 -t 93       ${OPTIONS})
  dplasma_add_test(getrf_incpiv "" ${PTG2DTD}_shm -N 378 -t 93 -i 17 ${OPTIONS})
  dplasma_add_test(getrf_nopiv "" ${PTG2DTD}_shm -N 378 -t 93       ${OPTIONS})
  dplasma_add_test(getrf_qrf "" shm -N 378 -t 93 -i 17 ${OPTIONS})

  #dplasma_add_test(gesv "" ${PTG2DTD}_shm -N 874 -K 367 -t 76       ${OPTIONS})
  dplasma_add_test(gesv_incpiv "" ${PTG2DTD}_shm -N 874 -K 367 -t 76 -i 23 ${OPTIONS})

  # The headnode lack GPUs so we need MPI in order to get the test to run on
  # one of the nodes.
  if (CUDA_FOUND AND MPI_C_FOUND)
    dplasma_add_test(potrf potrf ${PTG2DTD}_1gpu_shm -N 8000 -t 320 ${OPTIONS} -g 1)
    set_tests_properties(dplasma_${prec}potrf_1gpu_shm PROPERTIES LABEL "dplasma;shm;gpu")
    dplasma_add_test(potrf potrf ${PTG2DTD}_2gpu_shm -N 8000 -t 320 ${OPTIONS} -g 2)
    set_tests_properties(dplasma_${prec}potrf_2gpu_shm PROPERTIES LABEL "dplasma;shm;gpu")
  endif (CUDA_FOUND AND MPI_C_FOUND)

  #   if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
  #     dplasma_add_test(heev "" ${PTG2DTD}_shm -N 4000 ${OPTIONS})
  #   else()
  #     dplasma_add_test(syev "" ${PTG2DTD}_shm -N 4000 ${OPTIONS})
  #   endif()
  #   dplasma_add_test(geqrf_p0 "" ${PTG2DTD}_shm -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
  #   dplasma_add_test(geqrf_p1 "" ${PTG2DTD}_shm -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
  #   dplasma_add_test(geqrf_p2 "" ${PTG2DTD}_shm -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
  #   dplasma_add_test(geqrf_p3 "" ${PTG2DTD}_shm -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)
  #
  # The insert_task interface
  dplasma_add_test(potrf_dtd "" shm -N 874 -K 367 -t 76 -i 23 ${OPTIONS})
  dplasma_add_test(geqrf_dtd "" shm -N 874 -K 367 -t 76 -i 23 ${OPTIONS})
  dplasma_add_test(getrf_incpiv_dtd "" shm -N 874 -K 367 -t 76 -i 23 ${OPTIONS})
endforeach()

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  set(PROCS 4)
  set(CORES "")
  #set(CORES "-c;1")

  foreach(prec ${DPLASMA_PRECISIONS})

    # check the control and test matrices generation (zplrnt, zplghe, zplgsy, zpltmg) in distributed memory
    dplasma_add_test(print "" mpi:${PROCS} -N 64 -t 7 ${OPTIONS})

    # check the norms that are used in all other testings
    dplasma_add_test(lange print mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(lanm2 print mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})

    # Need to add testings on zlacpy, zlaset, zgeadd, zlascal, zger, (zlaswp?)

    # BLAS Shared memory
    dplasma_add_test(trmm lange mpi:${PROCS} -M 106 -N 150 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(trsm trmm mpi:${PROCS} -M 106 -N 150 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(gemm lange mpi:${PROCS} -M 106 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(gemm_dtd lange mpi:${PROCS} -M 106 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(symm lange mpi:${PROCS} -M 106 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(syrk "" mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(syr2k "" mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})
    if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
      dplasma_add_test(hemm "" mpi:${PROCS} -M 106 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(herk "" mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(her2k "" mpi:${PROCS} -M 287 -N 283 -K 97 -t 19 ${OPTIONS})
    endif()

    # Cholesky
    dplasma_add_test(potrf               "" mpi:${PROCS} -N 378 -t 19        ${OPTIONS})
    dplasma_add_test(potrf_dtd           "" mpi:${PROCS} -N 378 -t 19        ${OPTIONS})
    dplasma_add_test(potrf_dtd_untied    "" mpi:${PROCS} -N 378 -t 19        ${OPTIONS})
    dplasma_add_test(posv                "" mpi:${PROCS} -N 457 -t 19 -K 367 ${OPTIONS})

    # QR / LQ
    dplasma_add_test(geqrf               "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(geqrf_dtd           "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(gelqf               "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
      dplasma_add_test(unmqr             "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(unmlq             "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    else()
      dplasma_add_test(ormqr             "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(ormlq             "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    endif()

    # QR / LQ: HQR
    dplasma_add_test(geqrf_hqr           "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(gelqf_hqr           "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
      dplasma_add_test(unmqr_hqr         "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(unmlq_hqr         "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    else()
      dplasma_add_test(ormqr_hqr         "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(ormlq_hqr         "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    endif()

    # QR / LQ: systolic
    dplasma_add_test(geqrf_systolic      "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
    dplasma_add_test(gelqf_systolic      "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
      dplasma_add_test(unmqr_systolic    "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(unmlq_systolic    "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    else()
      dplasma_add_test(ormqr_systolic    "" mpi:${PROCS} -M 487 -N 283 -K 97 -t 19 ${OPTIONS})
      dplasma_add_test(ormlq_systolic    "" mpi:${PROCS} -M 287 -N 383 -K 97 -t 19 ${OPTIONS})
    endif()

    # LU
    dplasma_add_test(getrf               "" mpi:${PROCS} -N 378 -t 19 -P 1 ${OPTIONS})
    dplasma_add_test(getrf_incpiv        "" mpi:${PROCS} -N 378 -t 19 -i 7 ${OPTIONS})
    dplasma_add_test(getrf_incpiv_dtd    "" mpi:${PROCS} -N 378 -t 19 -i 7 ${OPTIONS})
    dplasma_add_test(getrf_nopiv         "" mpi:${PROCS} -N 378 -t 19      ${OPTIONS})
    dplasma_add_test(getrf_qrf           "" mpi:${PROCS} -N 378 -t 19 -i 7 ${OPTIONS})

    #dplasma_add_test(gesv "" mpi:${PROCS} -N 874 -K 367 -t 76       ${OPTIONS})
    dplasma_add_test(gesv_incpiv         "" mpi:${PROCS} -N 874 -K 367 -t 17 -i 7 ${OPTIONS})

    # GPU Cholesky tests
    if (CUDA_FOUND)
        dplasma_add_test(potrf potrf      1gpu_mpi:${PROCS} -N 8000 -t 320 ${OPTIONS} -g 1 -P 2)
        dplasma_add_test(potrf potrf_1gpu 2gpu_mpi:${PROCS} -N 8000 -t 320 ${OPTIONS} -g 2 -P 2)
    endif (CUDA_FOUND)

    # dplasma_add_test(potrf_pbq "" mpi:${PROCS} -N 4000 ${OPTIONS} -o PBQ)
    # dplasma_add_test(geqrf_pbq "" mpi:${PROCS} -N 4000 ${OPTIONS} -o PBQ)
    # dplasma_add_test(geqrf_p0 "" mpi:${PROCS} -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 0 --tsrr=0 -v=5)
    # dplasma_add_test(geqrf_p1 "" mpi:${PROCS} -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 1 --tsrr=0 -v=5)
    # dplasma_add_test(geqrf_p2 "" mpi:${PROCS} -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 2 --tsrr=0 -v=5)
    # dplasma_add_test(geqrf_p3 "" mpi:${PROCS} -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 3 --tsrr=0 -v=5)

    # if ( ${prec} STREQUAL "c" OR ${prec} STREQUAL "z" )
    #   dplasma_add_test(heev "" mpi:${PROCS} -N 2000 ${OPTIONS})
    # else()
    #   dplasma_add_test(syev "" mpi:${PROCS} -N 2000 ${OPTIONS})
    # endif()
  endforeach()

endif( MPI_C_FOUND )

