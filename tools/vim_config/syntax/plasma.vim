" Vim syntax file
" Language:	    PLASMA conf
" Maintainer:	
" Last change:	
" File Types:	.jdf

" For version 5.x: Clear all syntax items
" For version 6.x: Quit when a syntax file was already loaded
"if version < 600
  "syntax clear
"elseif exists("b:current_syntax")
  "finish
"endif

" Read the C syntax to start with
if version < 600
  so <sfile>:p:h/c.vim
else
  runtime! syntax/c.vim
  unlet b:current_syntax
endif

syn keyword plasmaCInclusion    %{ %}
syn keyword plasmaDirection     INOUT IN OUT
syn keyword plasmaOperator      -> <- min max
syn keyword plasmaRange         ..
syn match plasmaStructuring     "\c\<body\>"
syn match plasmaStructuring     "\c\<end\>"
syn match plasmaPartition       /^:/

" Default highlighting
if version >= 508 || !exists("did_cg_syntax_inits")
  if version < 508
    let did_cg_syntax_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink plasmaCInclusion       Tag
  HiLink plasmaStructuring      Keyword
  HiLink plasmaDirection        Keyword

  HiLink plasmaPartition        Delimiter
  HiLink pasmaOperator          Operator

  delcommand HiLink
endif
let b:current_syntax = "plasma"

