include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

parsec_addtest(C testing_redistribute "testing_redistribute.c;common.c")
target_link_libraries(testing_redistribute PRIVATE m)
target_ptg_sources(testing_redistribute PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check2.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_bound.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_no_optimization.jdf")

parsec_addtest(C testing_redistribute_random "testing_redistribute_random.c;common.c")
target_link_libraries(testing_redistribute_random PRIVATE m)
target_ptg_sources(testing_redistribute_random PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_check2.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_bound.jdf;${CMAKE_CURRENT_SOURCE_DIR}/redistribute_no_optimization.jdf")

if( MPI_C_FOUND )
   add_test(redistribute_mpi ${MPI_TEST_CMD_LIST} 8 ./testing_redistribute -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)
    add_test(redistribute_random_mpi ${MPI_TEST_CMD_LIST} 8 ./testing_redistribute_random -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)
else( MPI_C_FOUND )
    add_test(redistribute_shm ${MPI_TEST_CMD_LIST} ./testing_redistribute -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x)
    add_test(redistribute_random_shm ${MPI_TEST_CMD_LIST} ./testing_redistribute_random -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x)
endif( MPI_C_FOUND )
