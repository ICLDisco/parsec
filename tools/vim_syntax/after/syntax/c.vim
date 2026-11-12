" Vim syntax file (loaded after the standard C syntax)
" Language:	    C with the PaRSEC Dynamic Task Discovery (DTD) interface
" File Types:	.c .h
"
" DTD (parsec/interfaces/dtd/insert_function.h) is a C API, not a language of
" its own: tasks are inserted at run time with parsec_dtd_insert_task() and
" friends, and the way every argument is accessed is described by a flag.
" This file only adds the DTD names on top of the C highlighting.  All the
" names carry the parsec_ / PARSEC_ prefix, so they are harmless in any other
" C file.
"
" Installation: same as syntax/parsec.vim (the file must end up in an
" 'after/syntax' directory of the runtimepath).

" Types: parsec_dtd_taskpool_t, parsec_dtd_task_t, parsec_dtd_tile_t, ...
syn match   cType       "\<parsec_dtd_\w*_t\>"
syn keyword cType       parsec_arena_datatype_t

" API: parsec_dtd_taskpool_new, parsec_dtd_insert_task, parsec_dtd_data_flush,
"      parsec_dtd_unpack_args, parsec_dtd_tile_of, parsec_insert_dtd_task, ...
syn match   cFunction   "\<parsec_dtd_\h\w*\>\%(_t\)\@<!"
syn keyword cFunction   parsec_insert_dtd_task

" Access mode of an argument: operation type, then modifier flags that can be
" or-ed with it (PARSEC_INOUT | PARSEC_AFFINITY | PARSEC_PUSHOUT).
syn keyword cConstant   PARSEC_INPUT PARSEC_OUTPUT PARSEC_INOUT PARSEC_ATOMIC_WRITE
syn keyword cConstant   PARSEC_SCRATCH PARSEC_VALUE PARSEC_REF
syn keyword cConstant   PARSEC_AFFINITY PARSEC_DONT_TRACK PARSEC_PUSHOUT PARSEC_PULLIN
syn keyword cConstant   PARSEC_PROFILE_INFO
syn keyword cConstant   PARSEC_DTD_ARG_END PARSEC_DTD_EMPTY_FLAG PASSED_BY_REF
syn keyword cConstant   PARSEC_DTD_MAX_PARAMS

" Tile helpers are macros
syn keyword cConstant   PARSEC_DTD_TILE_OF PARSEC_DTD_TILE_OF_KEY
