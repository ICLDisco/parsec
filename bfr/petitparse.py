# -*- coding: utf-8 -*-
# --------------------------------
# petitparse.py
#
# Simple parser for Petit output.
# --------------------------------

import sys
import petitlex
import ply.yacc as yacc

# Get the token map
tokens = petitlex.tokens

def p_translation_unit_1(p):
  "translation_unit : constrain_definition"

def p_translation_unit_2(p):
  "translation_unit : translation_unit constrain_definition"

def p_constrain_definition(p):
  "constrain_definition : LBRACE variable_mapping COLON logical_or_expression RBRACE"

def p_variable_mapping(p):
  "variable_mapping : bracket_expression ARROW bracket_expression"

def p_bracket_expression(p):
  "bracket_expression : LBRACKET variable_list RBRACKET"

def p_variable_list_1(p):
  "variable_list : variable_expression"

def p_variable_list_2(p):
  "variable_list : variable_list COMMA variable_expression"

def p_variable_expression_1(p):
  "variable_expression : ID"

def p_variable_expression_2(p):
  "variable_expression : ID QUOTE"

def p_logical_or_expression_1(p):
  "logical_or_expression : logical_and_expression"

def p_logical_or_expression_2(p):
  "logical_or_expression : logical_or_expression LOR logical_and_expression"

def p_logical_and_expression_1(p):
  "logical_and_expression : relational_expression"

def p_logical_and_expression_2(p):
  "logical_and_expression : logical_and_expression LAND relational_expression"

def p_relational_expression_1(p):
  "relational_expression : variable_expression"

def p_relational_expression_2(p):
  "relational_expression : ICONST"

def p_relational_expression_3(p):
  "relational_expression : relational_expression LT variable_expression"

def p_relational_expression_4(p):
  "relational_expression : relational_expression LE variable_expression"

def p_error(p):
  global G_text

  print "Whoa. We're hosed", p
  print "Type     :", p.type
  print "Value    :", repr(p.value)
  print "Line No. :", p.lineno
  print "Offending:", G_text.split("\n")[p.lineno-1]


yacc.yacc(method='LALR')

def parseFile(fname):
  global G_text
  f = open(fname)
  G_text = f.read()
  print yacc.parse(G_text, tracking=True)
  #ast = bfrast.File(None, yacc.parse(G_text, tracking=True))
  #return ast

def main(argv):
  global G_text

  # Build the parser

  if len(argv) > 1:
    fname = argv[1]
  else:
    #f = sys.stdin
    return 0

  ast = parseFile(fname)

  print
  print ast

  return 0

if "__main__" == __name__:
    sys.exit(main(sys.argv))
