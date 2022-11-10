include(ParsecCompilePTG)

parsec_addtest_cmd(parsec/dsl/ptg/branching/hashtable ${SHM_TEST_CMD_LIST} dsl/ptg/branching/branching_ht)
parsec_addtest_cmd(parsec/dsl/ptg/branching/idxarray ${SHM_TEST_CMD_LIST} dsl/ptg/branching/branching_idxarr)
