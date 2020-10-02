include(ParsecCompilePTG)

parsec_addtest(C async)
target_ptg_sources(async PRIVATE "async.jdf")

# Define Testings
include(Testings.cmake)
