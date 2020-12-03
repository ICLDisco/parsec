include(ParsecCompilePTG)

set(JDF_SOURCES "local_no_reshape.jdf;local_read_reshape.jdf;local_output_reshape.jdf;local_input_reshape.jdf;remote_read_reshape.jdf;remote_no_re_reshape.jdf;local_input_LU_LL.jdf;")
parsec_addtest(C reshape "testing_reshape.c;common.c")
target_ptg_sources(reshape PRIVATE ${JDF_SOURCES})

parsec_addtest(C input_dep_reshape_single_copy "testing_input_dep_reshape_single_copy.c;common.c")
target_ptg_sources(input_dep_reshape_single_copy PRIVATE "input_dep_single_copy_reshape.jdf;")

parsec_addtest(C remote_multiple_outs_same_pred_flow "testing_remote_multiple_outs_same_pred_flow.c;common.c")
target_ptg_sources(remote_multiple_outs_same_pred_flow PRIVATE "remote_multiple_outs_same_pred_flow.jdf;remote_multiple_outs_same_pred_flow_multiple_deps.jdf;")

set(JDF_SOURCES "avoidable_reshape.jdf;")
parsec_addtest(C avoidable_reshape "testing_avoidable_reshape.c;common.c")
target_ptg_sources(avoidable_reshape PRIVATE ${JDF_SOURCES})

include(Testings.cmake)


