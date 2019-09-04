include(ParsecCompilePTG)

parsec_addtest(C merge_sort "main.c;merge_sort_wrapper.c;sort_data.c")
target_ptg_sources(merge_sort PRIVATE "merge_sort.jdf")

include(Testings.cmake)
