
" Vim syntax file
" Language:	    PaRSEC
" Maintainer:	
" Last change:	
" File Types:	.jdf

" For version 5.x: Clear all syntax items
" For version 6.x: Quit when a syntax file was already loaded
if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

" avoid displaying annoying errors du to twisted
" embedded C syntax
let c_no_curly_error = 1
let c_no_bracket_error = 1

" Read the C syntax to start with
if version < 600
  so <sfile>:p:h/c.vim
else
  runtime! syntax/c.vim
  unlet b:current_syntax
endif

syn keyword parsecCInclusion    %{ %}
syn keyword parsecOldDirection  INOUT IN OUT
syn keyword parsecDirection     READ RW WRITE
syn keyword parsecOperator      -> <- min max
syn keyword parsecRange         ..
syn match parsecDataType        "\[\w\+\]"
syn match parsecTopLevel        "^\w\+"
syn match parsecStructuring     "\c\<body\>"
syn match parsecStructuring     "\c\<end\>"
syn match parsecPartition       /^:/

" Default highlighting
if version >= 508 || !exists("did_cg_syntax_inits")
  if version < 508
    let did_cg_syntax_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink parsecCInclusion       Tag
  HiLink parsecTopLevel         Identifier
  HiLink parsecStructuring      Keyword
  HiLink parsecDirection        Keyword
  HiLink parsecOldDirection     Error
  HiLink parsecDataType         Type
  HiLink parsecPartition        Delimiter

  delcommand HiLink
endif
let b:current_syntax = "parsec"

