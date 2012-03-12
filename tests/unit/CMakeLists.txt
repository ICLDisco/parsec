set(DAGUE_DIR ../../)
include(FindDAGUE)

dague_addexec(C lifo lifo.c)
dague_addexec(C list list.c)
