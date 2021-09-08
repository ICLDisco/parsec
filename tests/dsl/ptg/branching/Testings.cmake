include(ParsecCompilePTG)

parsec_addtest_cmd(dsl/ptg/branching/hashtable ${SHM_TEST_CMD_LIST} dsl/ptg/branching/branching_ht)
parsec_addtest_cmd(dsl/ptg/branching/idxarray ${SHM_TEST_CMD_LIST} dsl/ptg/branching/branching_idxarr)
