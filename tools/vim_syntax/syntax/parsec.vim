" Vim syntax file
" Language:	    PaRSEC JDF (Job Data Flow, the PTG input language)
" Maintainer:	
" Last change:	
" File Types:	.jdf
"
" The grammar followed here is parsec/interfaces/ptg/ptg-compiler/parsec.l
" and parsec.y.  A JDF file is a mix of two languages: the JDF language
" itself (globals, task classes, execution space, partitioning, flows and
" dependencies) and C, which only appears in a few delimited places:
"
"     extern "C" %{ ... %}      verbatim prologue / epilogue
"     %{ ... %}                 inline C expression (also: inline_c %{ ... %})
"     BODY [ properties ] ... END
"
" C highlighting is therefore confined to those regions instead of being
" applied to the whole file.
"
" Installation: copy or link the content of this directory into
" ~/.vim (or ~/.config/nvim), or add it to 'runtimepath'.

" Quit when a syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

" avoid displaying annoying errors due to twisted
" embedded C syntax
let c_no_curly_error = 1
let c_no_bracket_error = 1

" Make the C syntax available as a cluster, usable inside the C regions only.
syn include @parsecC syntax/c.vim
unlet! b:current_syntax

" The lexer is case sensitive, except for BODY, END and %option.
syn case match

" --- Operators -------------------------------------------------------------
" Defined first: where two items start at the same position, the one defined
" last wins, so the more specific items below take precedence.
syn match parsecOperator    "[-+*/%^<>!?]"
syn match parsecOperator    "==\|!=\|<=\|>=\|<<\|>>\|&&\=\|||\="
syn match parsecOperator    ":"
syn match parsecRange       "\.\."
syn match parsecArrow       "->\|<-"
syn match parsecNumber      "\<\d\+\>"

" --- Names -----------------------------------------------------------------
" Anything followed by an opening parenthesis is either a task class or a data
" collection: Name(k, n), -> A TaskRecv(k, 0 .. NB), : descA(m, n)
syn match parsecCall        "\<\h\w*\ze\s*("

" Dataflow declaration: [READ|RW|...] FlowName <- ... / -> ...
syn keyword parsecFlowType  READ RW WRITE RO WO CTL nextgroup=parsecFlowName skipwhite
syn match   parsecFlowName  "\h\w*" contained
syn match   parsecFlowName  "^\s*\zs\h\w*\ze\s*\%(<-\|->\)"

" Flow referenced on the other side of a dependency: -> A TaskRecv(k)
" (the name that precedes the task class).
" (a look-behind is needed, the operators are consumed by their own items).
syn match parsecFlowRef     "\%(\%(<-\|->\|?\|:\)\s*\)\@<=\h\w*\ze\s\+\h\w*\s*("

" Special dependency endpoints
syn keyword parsecSpecialData NEW NULL

" Local variables of the execution space:  k = 0 .. NB .. 2
syn match parsecLocal       "^\s*\zs\h\w*\ze\s*=\%(=\)\@!"

" Partitioning (task affinity)  ': descA(m, n)'  and priority  '; expr'
syn match parsecPartition   "^\s*:"
syn match parsecPriority    "^\s*;"
syn keyword parsecStructuring SIMCOST

" --- Strings and comments --------------------------------------------------
syn region parsecString     start=/"/ skip=/\\\\\|\\"/ end=/"/ oneline contains=parsecEscape
syn match  parsecEscape     "\\\%([0-7]\{1,3}\|.\)" contained

syn keyword parsecTodo      TODO FIXME XXX NOTE contained
syn match   parsecComment   "//.*$" contains=parsecTodo
syn region  parsecComment   start="/\*" end="\*/" contains=parsecTodo

" --- Properties: [ key = value  key = value ... ] ----------------------------
" Used on globals, task classes, dependencies and BODY.  The same keys
" are commonly written as type, type_remote, type_data, layout, count, displ,
" hidden, default, profile, high_priority, ...
syn region parsecProperties matchgroup=parsecPropDelim start="\[" end="\]"
      \ contains=parsecPropKey,parsecPropBool,parsecString,parsecNumber,parsecComment,parsecInlineC
syn match  parsecPropKey    "\<\h\w*\%(\.\h\w*\)*\ze\s*=" contained
" the lexer accepts these in any case
syn case ignore
syn keyword parsecPropBool  contained on off true false yes no
syn case match

" --- %option name = value ---------------------------------------------------
syn case ignore
syn match parsecOption      "^%option\>" nextgroup=parsecOptionName skipwhite
syn case match
syn match parsecOptionName  "\h\w*" contained

" --- Embedded C ------------------------------------------------------------
" extern "C" %{ ... %}
syn region parsecExternC matchgroup=parsecCInclusion
      \ start=/\<extern\s\+"C"\s*%{/ end=/%}/
      \ contains=@parsecC keepend

" %{ ... %}, %c{ ... %} and inline_c %{ ... %}
syn region parsecInlineC matchgroup=parsecCInclusion
      \ start=/\%(\<inline_c\>\s*\)\=%c\={/ end=/%}/
      \ contains=@parsecC keepend

" BODY [ properties ] ... END  (both keywords are case insensitive and the
" lexer only recognizes them at the beginning of a line).
syn case ignore
syn region parsecBody matchgroup=parsecStructuring
      \ start=/^\s*body\>/ end=/^\s*end\>/
      \ contains=parsecBodyProperties,@parsecC keepend
syn region parsecBodyProperties matchgroup=parsecPropDelim contained
      \ start=/\%(^\s*body\s*\)\@<=\[/ end=/\]/
      \ contains=parsecPropKey,parsecPropBool,parsecString,parsecNumber,parsecComment,parsecInlineC
syn case match

" The embedded C regions can be long (prologues, main()): never guess.
syn sync fromstart

" --- Highlighting ----------------------------------------------------------
hi def link parsecCInclusion    Tag
hi def link parsecStructuring   Keyword
hi def link parsecOption        PreProc
hi def link parsecOptionName    Identifier
hi def link parsecFlowType      Keyword
hi def link parsecFlowName      Identifier
hi def link parsecFlowRef       Identifier
hi def link parsecCall          Function
hi def link parsecLocal         Identifier
hi def link parsecSpecialData   Constant
hi def link parsecNumber        Number
hi def link parsecString        String
hi def link parsecEscape        SpecialChar
hi def link parsecOperator      Operator
hi def link parsecArrow         Statement
hi def link parsecRange         Operator
hi def link parsecPartition     Delimiter
hi def link parsecPriority      Delimiter
hi def link parsecPropDelim     Delimiter
hi def link parsecPropKey       Type
hi def link parsecPropBool      Boolean
hi def link parsecComment       Comment
hi def link parsecTodo          Todo

let b:current_syntax = "parsec"
