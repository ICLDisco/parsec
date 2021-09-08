include(ParsecCompilePTG)

parsec_addtest_cmd(dsl/ptg/user-defined-functions/udf ${SHM_TEST_CMD_LIST} dsl/ptg/user-defined-functions/udf -N 100 -n 10)
