
parsec_addtest_executable(C write_check SOURCES vector.c)
target_ptg_sources(write_check PRIVATE "write_check.jdf")

parsec_addtest_executable(C jdf_forward_RW_NULL)
target_ptg_sources(jdf_forward_RW_NULL PRIVATE "forward_RW_NULL.jdf")
parsec_addtest_executable(C jdf_forward_READ_NULL)
target_ptg_sources(jdf_forward_READ_NULL PRIVATE "forward_READ_NULL.jdf")

parsec_addtest_executable(C must_fail_too_many_in_deps NODEFAULTBUILD)
target_ptg_sources(must_fail_too_many_in_deps PRIVATE "too_many_in_deps.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_in_deps PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
parsec_addtest_executable(C must_fail_too_many_out_deps NODEFAULTBUILD)
target_ptg_sources(must_fail_too_many_out_deps PRIVATE "too_many_out_deps.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_out_deps PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
parsec_addtest_executable(C must_fail_too_many_read_flows NODEFAULTBUILD)
target_ptg_sources(must_fail_too_many_read_flows PRIVATE "too_many_read_flows.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_read_flows PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
parsec_addtest_executable(C must_fail_too_many_write_flows NODEFAULTBUILD)
target_ptg_sources(must_fail_too_many_write_flows PRIVATE "too_many_write_flows.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_write_flows PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
parsec_addtest_executable(C must_fail_too_many_local_vars NODEFAULTBUILD)
target_ptg_sources(must_fail_too_many_local_vars PRIVATE "too_many_local_vars.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_local_vars PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
