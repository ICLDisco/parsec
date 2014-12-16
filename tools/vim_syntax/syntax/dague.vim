
" Vim syntax file
" Language:	    Dague
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

syn keyword dagueCInclusion    %{ %}
syn keyword dagueOldDirection  INOUT IN OUT
syn keyword dagueDirection     READ RW WRITE
syn keyword dagueOperator      -> <- min max
syn keyword dagueRange         ..
syn match dagueDataType        "\[\w\+\]"
syn match dagueTopLevel        "^\w\+"
syn match dagueStructuring     "\c\<body\>"
syn match dagueStructuring     "\c\<end\>"
syn match daguePartition       /^:/

" Default highlighting
if version >= 508 || !exists("did_cg_syntax_inits")
  if version < 508
    let did_cg_syntax_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink dagueCInclusion       Tag
  HiLink dagueTopLevel         Identifier
  HiLink dagueStructuring      Keyword
  HiLink dagueDirection        Keyword
  HiLink dagueOldDirection     Error
  HiLink dagueDataType         Type
  HiLink daguePartition        Delimiter

  delcommand HiLink
endif
let b:current_syntax = "dague"

