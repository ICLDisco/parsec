# CTest delayed initialization is broken, so we put the
# CTestConfig.cmake info here.
set(CTEST_PROJECT_NAME "PaRSEC")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "icl.cs.utk.edu")
set(CTEST_DROP_LOCATION "/cdash/submit.php?project=PaRSEC")
set(CTEST_DROP_SITE_CDASH TRUE)
