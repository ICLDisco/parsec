#
# Shared Memory Testings
#

foreach(prec ${DPLASMA_PRECISIONS})
  # check the control in shared memory
  add_test(${prec}print ${SHM_TEST_CMD_LIST} ./testing_${prec}print -N 40 -t 7 -x -v=5)
endforeach()

#
# Check BLAS/Lapack subroutines in shared memory
#
foreach(prec ${DPLASMA_PRECISIONS})

  # check the norms that are used in all other testings
  add_test(${prec}lange ${SHM_TEST_CMD_LIST} ./testing_${prec}lange -N 1500 -t 233 -x -v=5)

  # Need to add here check on lacpy (Tile => Lapack) and geadd

  # BLAS Shared memory
  # add_test(${prec}trmm  ${SHM_TEST_CMD_LIST} ./testing_${prec}trmm         -N 150 -K 97 -t 56 -x -v=5)
  # add_test(${prec}trsm  ${SHM_TEST_CMD_LIST} ./testing_${prec}trsm         -N 150 -K 97 -t 56 -x -v=5)
  # add_test(${prec}gemm  ${SHM_TEST_CMD_LIST} ./testing_${prec}gemm  -M 106 -N 283 -K 97 -t 56 -x -v=5)
  # add_test(${prec}symm  ${SHM_TEST_CMD_LIST} ./testing_${prec}symm  -M 106 -N 283 -K 97 -t 56 -x -v=5)
  # add_test(${prec}syrk  ${SHM_TEST_CMD_LIST} ./testing_${prec}syrk  -M 287 -N 283 -K 97 -t 56 -x -v=5)
  # add_test(${prec}syr2k ${SHM_TEST_CMD_LIST} ./testing_${prec}syr2k -M 287 -N 283 -K 97 -t 56 -x -v=5)

  # if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
  #   add_test(${prec}hemm  ${SHM_TEST_CMD_LIST} ./testing_${prec}hemm  -M 106 -N 283 -K 97 -t 56 -x -v=5)
  #   add_test(${prec}herk  ${SHM_TEST_CMD_LIST} ./testing_${prec}herk  -M 287 -N 283 -K 97 -t 56 -x -v=5)
  #   add_test(${prec}her2k ${SHM_TEST_CMD_LIST} ./testing_${prec}her2k -M 287 -N 283 -K 97 -t 56 -x -v=5)
  # endif()

#   # LAPACK shared memory
#   add_test(${prec}potrf     ${SHM_TEST_CMD_LIST} ./testing_${prec}potrf -N 4000 -x -v=5)
#   add_test(${prec}posv      ${SHM_TEST_CMD_LIST} ./testing_${prec}posv  -N 4000 -K 367 -x -v=5)
# 
#   add_test(${prec}getrf           ${SHM_TEST_CMD_LIST} ./testing_${prec}getrf        -N 4000 -x -v=5)
#   add_test(${prec}getrf_incpiv    ${SHM_TEST_CMD_LIST} ./testing_${prec}getrf_incpiv -N 4000 -x -v=5)
#   add_test(${prec}getrf_incpiv_sd ${SHM_TEST_CMD_LIST} ./testing_${prec}getrf_incpiv -N 4000 -x -v=5)
#   add_test(${prec}gesv_incpiv     ${SHM_TEST_CMD_LIST} ./testing_${prec}gesv_incpiv  -N 4000 -K 367 -x -v=5)
#   add_test(${prec}geqrf           ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf -N 4000 -x -v=5)
#   add_test(${prec}geqrf_systolic  ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf -N 4000 -x -v=5)
#   if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
#     add_test(${prec}unmqr        ${SHM_TEST_CMD_LIST} ./testing_${prec}unmqr -M 2873 -N 1067 -K 987 -x -v=5)
#     add_test(${prec}heev         ${SHM_TEST_CMD_LIST} ./testing_${prec}heev  -N 4000 -x -v=5)
#   else()
#     add_test(${prec}ormqr        ${SHM_TEST_CMD_LIST} ./testing_${prec}ormqr -M 2873 -N 1067 -K 987 -x -v=5)
#     add_test(${prec}syev         ${SHM_TEST_CMD_LIST} ./testing_${prec}syev  -N 4000 -x -v=5)
#   endif()
#   add_test(${prec}gelqf        ${SHM_TEST_CMD_LIST} ./testing_${prec}gelqf -N 4000 -x -v=5)
#   add_test(${prec}geqrf_p0     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
#   add_test(${prec}geqrf_p1     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
#   add_test(${prec}geqrf_p2     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
#   add_test(${prec}geqrf_p3     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)
# 
endforeach()

#  # Specific cases
#  # Do we want to test them in all precisions ?
#  add_test(dpotrf_pbq ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 4000 -x -v=5 -o PBQ)
#  add_test(dgeqrf_pbq ${SHM_TEST_CMD_LIST} ./testing_dgeqrf -N 4000 -x -v=5 -o PBQ)
#  
#  # The headnode lack GPUs so we need MPI in order to get the test to run on
#  # one of the nodes.
#  if (CUDA_FOUND AND MPI_C_FOUND)
#    add_test(dpotrf_g1  ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 8000 -x -v=5 -g 1)
#    add_test(dpotrf_g2  ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 8000 -x -v=5 -g 2)
#  endif (CUDA_FOUND AND MPI_C_FOUND)
#  
#  #
#  # Distributed Memory Testings
#  #
#  if( MPI_C_FOUND )
#    # Check MPI
#    add_test(mpi_test   ${MPI_TEST_CMD_LIST} -np 8 /bin/true)
#  
#    foreach(prec ${DPLASMA_PRECISIONS})
#      # check the control in shared memory
#      add_test(mpi_${prec}print         ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}print        -p 2 -N 40 -t 7 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}print" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}lange        ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}lange        -p 4 -N 1500 -t 233 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}lange" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}trmm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}trmm         -p 2 -N 1500 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}trmm" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}trsm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}trsm         -p 4 -N 1500 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}trsm" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}gemm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}gemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}gemm" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}potrf        ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}potrf        -p 2 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}potrf" PROPERTIES DEPENDS "mpi_test")
#  
#      if (CUDA_FOUND)
#        add_test(mpi_${prec}potrf_g1     ${MPI_TEST_CMD_LIST} -np 8 -mca btl_openib_flags 1 ./testing_${prec}potrf        -p 2 -N 8000 -x -v=5 -g 1)
#        SET_TESTS_PROPERTIES("mpi_${prec}potrf_g1" PROPERTIES DEPENDS "mpi_test")
#      endif (CUDA_FOUND)
#      add_test(mpi_${prec}posv         ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}posv         -p 4 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}posv" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}potrf_pbq    ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}potrf        -p 2 -N 4000 -x -v=5 -o PBQ)
#      SET_TESTS_PROPERTIES("mpi_${prec}potrf_pbq" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}getrf        ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}getrf        -p 1 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}getrf" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}getrf_incpiv ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}getrf_incpiv -p 4 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}getrf_incpiv" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}gesv_incpiv  ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}gesv_incpiv  -p 4 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}gesv_incpiv" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf        ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf        -p 4 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}gelqf        ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}gelqf        -p 4 -N 4000 -x -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}gelqf" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf_pbq    ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf        -p 4 -N 4000 -x -v=5 -o PBQ)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf_pbq" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf_p0     ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 0 --tsrr=0 -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf_p0" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf_p1     ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 1 --tsrr=0 -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf_p1" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf_p2     ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 2 --tsrr=0 -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf_p2" PROPERTIES DEPENDS "mpi_test")
#  
#      add_test(mpi_${prec}geqrf_p3     ${MPI_TEST_CMD_LIST} -np 8 ./testing_${prec}geqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 3 --tsrr=0 -v=5)
#      SET_TESTS_PROPERTIES("mpi_${prec}geqrf_p3" PROPERTIES DEPENDS "mpi_test")
#      if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
#        add_test(mpi_${prec}heev         ${MPI_TEST_CMD_LIST} -np 4 ./testing_${prec}heev -p 2 -N 2000 -x -v=5)
#      else()
#        add_test(mpi_${prec}syev         ${MPI_TEST_CMD_LIST} -np 4 ./testing_${prec}syev -p 2 -N 2000 -x -v=5)
#      endif()
#    endforeach()
#  
#    add_test(mpi_dsymm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_dsymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
#    add_test(mpi_csymm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_csymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
#    add_test(mpi_chemm         ${MPI_TEST_CMD_LIST} -np 8 ./testing_chemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
#    add_test(mpi_dsyrk         ${MPI_TEST_CMD_LIST} -np 8 ./testing_dsyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
#    add_test(mpi_csyrk         ${MPI_TEST_CMD_LIST} -np 8 ./testing_csyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
#    add_test(mpi_cherk         ${MPI_TEST_CMD_LIST} -np 8 ./testing_cherk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
#  
#  endif( MPI_C_FOUND )
