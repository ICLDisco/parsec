include(ParsecCompilePTG)

parsec_addtest_executable(C async)
target_ptg_sources(async PRIVATE "async.jdf")

# Define Testings
include(Testings.cmake)
