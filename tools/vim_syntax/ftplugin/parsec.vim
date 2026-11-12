" Vim filetype plugin
" Language:	    PaRSEC JDF
" File Types:	.jdf

if exists("b:did_ftplugin")
  finish
endif
let b:did_ftplugin = 1

" JDF comments are C-style: // to end of line, and /* ... */
setlocal commentstring=//\ %s
setlocal comments=s1:/*,mb:*,ex:*/,://
setlocal formatoptions-=t formatoptions+=croql

let b:undo_ftplugin = "setlocal commentstring< comments< formatoptions<"
