# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# petitlex.py
#
# A lexer for Petit output.
# ----------------------------------------------------------------------

import ply.lex as lex

tokens = (
  "ID", "QUOTE",
  "MINUS", "LT", "LE",
  "LAND", "LOR", "ARROW",
  "LBRACKET", "RBRACKET", "LBRACE", "RBRACE", "COMMA", "COLON",
  "ICONST"
)

# Completely ignored characters
t_ignore           = ' \t\x0c'

# Newlines
def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

t_QUOTE            = r"'"

t_MINUS            = r"-"
t_LT               = r"<"
t_LE               = r"<="

t_LAND             = r"&&"
t_LOR              = r"\|\|"

t_ARROW            = r"->"

t_LBRACKET         = r"\["
t_RBRACKET         = r"\]"
t_LBRACE           = r"\{"
t_RBRACE           = r"\}"
t_COMMA            = r","
t_COLON            = r":"

def t_ID(t):
    r"[A-Za-z_][\w_]*"
    t.type ="ID"
    return t

# Integer literal
t_ICONST = r'\d+([uU]|[lL]|[uU][lL]|[lL][uU])?'

t_ignore = " \t\x0c"

def t_error(t):
    print "Illegal character %s" % repr(t.value[0])
    t.lexer.skip(1)

lexer = lex.lex(optimize=1)

if __name__ == "__main__":
    lex.runmain(lexer)
