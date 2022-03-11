if(TARGET parsec-ptgpp)
  add_subdirectory(pingpong)
  add_subdirectory(all2all)
  add_subdirectory(generalized_reduction)
  add_subdirectory(stencil)
  add_subdirectory(merge_sort)
  add_subdirectory(haar_tree)
endif(TARGET parsec-ptgpp)

