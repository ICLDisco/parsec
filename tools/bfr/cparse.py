# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# cparse.py
#
# Simple parser for ANSI C.  Based on the grammar in K&R, 2nd Ed.
# -----------------------------------------------------------------------------

import sys
import clex
import ply.yacc as yacc

import bfrast

# Get the token map
tokens = clex.tokens

# translation-unit:

def p_translation_unit_1(p):
    'translation_unit : external_declaration'
    p[0] = p[1]

def p_translation_unit_2(p):
    'translation_unit : translation_unit external_declaration'
    pass

# external-declaration:

def p_external_declaration_1(p):
    "external_declaration : function_definition"
    p[0] = p[1]

def p_external_declaration_2(p):
    'external_declaration : declaration'
    pass

# function-definition:

def p_function_definition_1(p):
    "function_definition : declaration_specifiers declarator declaration_list compound_statement"
    print "p_function_definition_1", p[4]

def p_function_definition_2(p):
    "function_definition : declarator declaration_list compound_statement"
    print "p_function_definition_2", p[3]

def p_function_definition_3(p):
    'function_definition : declarator compound_statement'
    print "p_function_definition_3", p[2]

def p_function_definition_4(p):
    "function_definition : declaration_specifiers declarator compound_statement"
    p[0] = bfrast.Function(p[2].expr.name, p[2].params, p[1], p[3])
    print "p_function_definition_4", p[1], p[2], p[3]

# declaration:

def p_declaration_1(p):
    "declaration : declaration_specifiers init_declarator_list SEMI"
    print "p_declaration_1", p[1], p[2]
    p[0] = bfrast.Declare(p[1], p[2])

def p_declaration_2(p):
    "declaration : declaration_specifiers SEMI"
    p[0] = p[1]

# declaration-list:

def p_declaration_list_1(p):
    "declaration_list : declaration"
    p[0] = [p[1]]
    print "p_declaration_list_1", p[1]

def p_declaration_list_2(p):
    "declaration_list : declaration_list declaration"
    p[0] = p[1] + [p[2]]
    print "p_declaration_list_2", p[1], p[2]

# declaration-specifiers
def p_declaration_specifiers_1(p):
    "declaration_specifiers : storage_class_specifier declaration_specifiers"
    p[0] = bfrast.DeclSpec(p[1], p[2])
    print "p_declaration_specifiers_1", p[1], p[2]

def p_declaration_specifiers_2(p):
    "declaration_specifiers : type_specifier declaration_specifiers"
    p[0] = p[1]
    print "p_declaration_specifiers_2", p[1], p[2]

def p_declaration_specifiers_3(t):
    "declaration_specifiers : type_qualifier declaration_specifiers"
    print "p_declaration_specifiers_3", p[1], p[2]

def p_declaration_specifiers_4(t):
    "declaration_specifiers : storage_class_specifier"
    print "p_declaration_specifiers_4", p[1]

def p_declaration_specifiers_5(p):
    "declaration_specifiers : type_specifier"
    p[0] = p[1]
    print "p_declaration_specifiers_5", p[1]

def p_declaration_specifiers_6(t):
    "declaration_specifiers : type_qualifier"
    pass

# storage-class-specifier
def p_storage_class_specifier(p):
    """storage_class_specifier : AUTO
                               | REGISTER
                               | STATIC
                               | EXTERN
                               | TYPEDEF
                               """
    pass

# type-specifier:
def p_type_specifier(p):
    """type_specifier : VOID
                      | CHAR
                      | SHORT
                      | INT
                      | LONG
                      | FLOAT
                      | DOUBLE
                      | SIGNED
                      | UNSIGNED
                      | struct_or_union_specifier
                      | enum_specifier
                      | TYPEID
                      """
    p[0] = bfrast.TypeID(p[1])

# type-qualifier:
def p_type_qualifier(t):
    '''type_qualifier : CONST
                      | VOLATILE'''
    pass

# struct-or-union-specifier

def p_struct_or_union_specifier_1(t):
    'struct_or_union_specifier : struct_or_union ID LBRACE struct_declaration_list RBRACE'
    pass

def p_struct_or_union_specifier_2(t):
    'struct_or_union_specifier : struct_or_union LBRACE struct_declaration_list RBRACE'
    pass

def p_struct_or_union_specifier_3(t):
    'struct_or_union_specifier : struct_or_union ID'
    pass

# struct-or-union:
def p_struct_or_union(t):
    '''struct_or_union : STRUCT
                       | UNION
                       '''
    pass

# struct-declaration-list:

def p_struct_declaration_list_1(t):
    'struct_declaration_list : struct_declaration'
    pass

def p_struct_declaration_list_2(t):
    'struct_declaration_list : struct_declaration_list struct_declaration'
    pass

# init-declarator-list:

def p_init_declarator_list_1(p):
    "init_declarator_list : init_declarator"
    p[0] = [p[1]]
    print "p_init_declarator_list_1", p[1]

def p_init_declarator_list_2(p):
    'init_declarator_list : init_declarator_list COMMA init_declarator'
    p[0] = p[1] + [p[3]]

# init-declarator

def p_init_declarator_1(p):
    "init_declarator : declarator"
    p[0] = p[1]

def p_init_declarator_2(p):
    "init_declarator : declarator EQUALS initializer"
    p[0] = bfrast.InitDeclarator(p[1], p[3])
    print "p_init_declarator_2", p[1], p[3]

# struct-declaration:

def p_struct_declaration(t):
    "struct_declaration : specifier_qualifier_list struct_declarator_list SEMI"
    pass

# specifier-qualifier-list:

def p_specifier_qualifier_list_1(t):
    "specifier_qualifier_list : type_specifier specifier_qualifier_list"
    pass

def p_specifier_qualifier_list_2(t):
    "specifier_qualifier_list : type_specifier"
    pass

def p_specifier_qualifier_list_3(t):
    "specifier_qualifier_list : type_qualifier specifier_qualifier_list"
    pass

def p_specifier_qualifier_list_4(t):
    "specifier_qualifier_list : type_qualifier"
    pass

# struct-declarator-list:

def p_struct_declarator_list_1(t):
    "struct_declarator_list : struct_declarator"
    pass

def p_struct_declarator_list_2(t):
    'struct_declarator_list : struct_declarator_list COMMA struct_declarator'
    pass

# struct-declarator:

def p_struct_declarator_1(t):
    'struct_declarator : declarator'
    pass

def p_struct_declarator_2(t):
    'struct_declarator : declarator COLON constant_expression'
    pass

def p_struct_declarator_3(t):
    'struct_declarator : COLON constant_expression'
    pass

# enum-specifier:

def p_enum_specifier_1(t):
    'enum_specifier : ENUM ID LBRACE enumerator_list RBRACE'
    pass

def p_enum_specifier_2(t):
    'enum_specifier : ENUM LBRACE enumerator_list RBRACE'
    pass

def p_enum_specifier_3(t):
    'enum_specifier : ENUM ID'
    pass

# enumerator_list:
def p_enumerator_list_1(t):
    'enumerator_list : enumerator'
    pass

def p_enumerator_list_2(t):
    'enumerator_list : enumerator_list COMMA enumerator'
    pass

# enumerator:
def p_enumerator_1(t):
    "enumerator : ID"
    pass

def p_enumerator_2(t):
    "enumerator : ID EQUALS constant_expression"
    pass

# declarator:

def p_declarator_1(p):
    "declarator : pointer direct_declarator"
    p[0] = bfrast.PtrDecl(p[2])

def p_declarator_2(p):
    "declarator : direct_declarator"
    p[0] = p[1]
    print "p_declarator_2", p[1]

# direct-declarator:

def p_direct_declarator_1(p):
    "direct_declarator : ID"
    p[0] = bfrast.SimpleDeclarator(bfrast.Name(p[1]))
    print "p_direct_declarator_1", p[1]

def p_direct_declarator_2(p):
    "direct_declarator : LPAREN declarator RPAREN"
    print "p_direct_declarator_2", p[1]

def p_direct_declarator_3(p):
    "direct_declarator : direct_declarator LBRACKET constant_expression_opt RBRACKET"
    p[0] = bfrast.ArrayDeclarator(p[1], p[3])
    print "p_direct_declarator_3", p[1], p[3]

def p_direct_declarator_4(p):
    "direct_declarator : direct_declarator LPAREN parameter_type_list RPAREN"
    p[0] = bfrast.FunctionDeclarator(p[1], p[3])
    print "p_direct_declarator_4", p[1], p[3]

def p_direct_declarator_5(p):
    "direct_declarator : direct_declarator LPAREN identifier_list RPAREN"
    p[0] = bfrast.FunctionDeclarator(p[1], p[3])
    print "p_direct_declarator_5", p[1], p[3]

def p_direct_declarator_6(p):
    "direct_declarator : direct_declarator LPAREN RPAREN"
    p[0] = bfrast.FunctionDeclarator(p[1])
    print "p_direct_declarator_6", p[1]

# pointer:
def p_pointer_1(t):
    "pointer : TIMES type_qualifier_list"
    pass

def p_pointer_2(t):
    "pointer : TIMES"
    pass

def p_pointer_3(t):
    'pointer : TIMES type_qualifier_list pointer'
    pass

def p_pointer_4(t):
    'pointer : TIMES pointer'
    pass

# type-qualifier-list:

def p_type_qualifier_list_1(t):
    'type_qualifier_list : type_qualifier'
    pass

def p_type_qualifier_list_2(t):
    'type_qualifier_list : type_qualifier_list type_qualifier'
    pass

# parameter-type-list:

def p_parameter_type_list_1(p):
    "parameter_type_list : parameter_list"
    p[0] = p[1]
    print "p_parameter_type_list_1", p[1]

def p_parameter_type_list_2(p):
    "parameter_type_list : parameter_list COMMA ELLIPSIS"
    pass

# parameter-list:

def p_parameter_list_1(p):
    "parameter_list : parameter_declaration"
    p[0] = [p[1]]

def p_parameter_list_2(p):
    "parameter_list : parameter_list COMMA parameter_declaration"
    p[0] = p[1] + [p[3]]

# parameter-declaration:
def p_parameter_declaration_1(p):
    "parameter_declaration : declaration_specifiers declarator"
    p[0] = bfrast.Declare(p[1], [p[2]])
    print "p_parameter_declaration_1", p[1], p[2]

def p_parameter_declaration_2(t):
    'parameter_declaration : declaration_specifiers abstract_declarator_opt'
    pass

# identifier-list:
def p_identifier_list_1(p):
    "identifier_list : ID"
    p[0] = [bfrast.Name(p[1])]

def p_identifier_list_2(p):
    "identifier_list : identifier_list COMMA ID"
    p[0] = p[1] + p[3]

# initializer:

def p_initializer_1(p):
    "initializer : assignment_expression"
    p[0] = p[1]

def p_initializer_2(t):
    """initializer : LBRACE initializer_list RBRACE
                   | LBRACE initializer_list COMMA RBRACE"""
    pass

# initializer-list:

def p_initializer_list_1(t):
    'initializer_list : initializer'
    pass

def p_initializer_list_2(t):
    'initializer_list : initializer_list COMMA initializer'
    pass

# type-name:

def p_type_name(t):
    'type_name : specifier_qualifier_list abstract_declarator_opt'
    pass

def p_abstract_declarator_opt_1(t):
    'abstract_declarator_opt : empty'
    pass

def p_abstract_declarator_opt_2(t):
    'abstract_declarator_opt : abstract_declarator'
    pass

# abstract-declarator:

def p_abstract_declarator_1(t):
    "abstract_declarator : pointer"
    pass

def p_abstract_declarator_2(t):
    'abstract_declarator : pointer direct_abstract_declarator'
    pass

def p_abstract_declarator_3(t):
    'abstract_declarator : direct_abstract_declarator'
    pass

# direct-abstract-declarator:

def p_direct_abstract_declarator_1(t):
    'direct_abstract_declarator : LPAREN abstract_declarator RPAREN'
    pass

def p_direct_abstract_declarator_2(t):
    'direct_abstract_declarator : direct_abstract_declarator LBRACKET constant_expression_opt RBRACKET'
    pass

def p_direct_abstract_declarator_3(t):
    'direct_abstract_declarator : LBRACKET constant_expression_opt RBRACKET'
    pass

def p_direct_abstract_declarator_4(t):
    'direct_abstract_declarator : direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN'
    pass

def p_direct_abstract_declarator_5(t):
    'direct_abstract_declarator : LPAREN parameter_type_list_opt RPAREN'
    pass

# Optional fields in abstract declarators

def p_constant_expression_opt_1(t):
    'constant_expression_opt : empty'
    pass

def p_constant_expression_opt_2(t):
    'constant_expression_opt : constant_expression'
    pass

def p_parameter_type_list_opt_1(t):
    'parameter_type_list_opt : empty'
    pass

def p_parameter_type_list_opt_2(t):
    'parameter_type_list_opt : parameter_type_list'
    pass

# statement:

def p_statement(p):
    """
    statement : labeled_statement
              | expression_statement
              | compound_statement
              | selection_statement
              | iteration_statement
              | jump_statement
              """
    p[0] = p[1]
    print "p_statement", p[1]

# labeled-statement:

def p_labeled_statement_1(t):
    'labeled_statement : ID COLON statement'
    pass

def p_labeled_statement_2(t):
    'labeled_statement : CASE constant_expression COLON statement'
    pass

def p_labeled_statement_3(t):
    "labeled_statement : DEFAULT COLON statement"
    pass

# expression-statement:
def p_expression_statement(p):
    "expression_statement : expression_opt SEMI"
    p[0] = p[1]

# compound-statement:

def p_compound_statement_1(p):
    "compound_statement : LBRACE declaration_list statement_list RBRACE"
    p[0] = bfrast.CompoundStatement(p[2] + p[3])
    print "p_compound_statement_1", p[2], p[3]

def p_compound_statement_2(p):
    "compound_statement : LBRACE statement_list RBRACE"
    p[0] = bfrast.CompoundStatement(p[2])

def p_compound_statement_3(p):
    "compound_statement : LBRACE declaration_list RBRACE"
    p[0] = bfrast.CompoundStatement(p[2])

def p_compound_statement_4(p):
    'compound_statement : LBRACE RBRACE'
    p[0] = bfrast.CompoundStatement()

# statement-list:

def p_statement_list_1(p):
    "statement_list : statement"
    p[0] = [p[1]]

def p_statement_list_2(p):
    "statement_list : statement_list statement"
    p[0] = p[1] + [p[2]]

# selection-statement

def p_selection_statement_1(t):
    'selection_statement : IF LPAREN expression RPAREN statement'
    pass

def p_selection_statement_2(t):
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement '
    pass

def p_selection_statement_3(t):
    'selection_statement : SWITCH LPAREN expression RPAREN statement '
    pass

# iteration_statement:

def p_iteration_statement_1(t):
    'iteration_statement : WHILE LPAREN expression RPAREN statement'
    pass

def p_iteration_statement_2(p):
    "iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement"
    p[0] = bfrast.For(p[3], p[5], p[7], p[9])

def p_iteration_statement_3(t):
    'iteration_statement : DO statement WHILE LPAREN expression RPAREN SEMI'
    pass

# jump_statement:

def p_jump_statement_1(t):
    'jump_statement : GOTO ID SEMI'
    pass

def p_jump_statement_2(t):
    'jump_statement : CONTINUE SEMI'
    pass

def p_jump_statement_3(t):
    'jump_statement : BREAK SEMI'
    pass

def p_jump_statement_4(p):
    'jump_statement : RETURN expression_opt SEMI'
    p[0] = bfrast.Return(p[1])

def p_expression_opt_1(p):
    'expression_opt : empty'
    p[0] = bfrast.Empty()

def p_expression_opt_2(p):
    'expression_opt : expression'
    p[0] = p[1]
    print "p_expression_opt_2", p[1]

# expression:
def p_expression_1(p):
    'expression : assignment_expression'
    p[0] = p[1]

def p_expression_2(t):
    'expression : expression COMMA assignment_expression'
    pass

# assigment_expression:
def p_assignment_expression_1(p):
    "assignment_expression : conditional_expression"
    p[0] = p[1]
    print "p_assignment_expression_1", p[1]

def p_assignment_expression_2(p):
    "assignment_expression : unary_expression assignment_operator assignment_expression"
    if p[2] == "=":
      p[0] = bfrast.Assign(p[1], p[3])
    else:
      p[0] = bfrast.AugAssign(p[1], p[2], p[3])

# assignment_operator:
def p_assignment_operator(p):
    """
    assignment_operator : EQUALS
                        | TIMESEQUAL
                        | DIVEQUAL
                        | MODEQUAL
                        | PLUSEQUAL
                        | MINUSEQUAL
                        | LSHIFTEQUAL
                        | RSHIFTEQUAL
                        | ANDEQUAL
                        | OREQUAL
                        | XOREQUAL
                        """
    p[0] = p[1]

# conditional-expression
def p_conditional_expression_1(p):
    'conditional_expression : logical_or_expression'
    p[0] = p[1]
    print "p_conditional_expression_1", p[1]

def p_conditional_expression_2(t):
    'conditional_expression : logical_or_expression CONDOP expression COLON conditional_expression '
    pass

# constant-expression

def p_constant_expression(p):
    "constant_expression : conditional_expression"
    p[0] = p[1]
    print "p_constant_expression", p[1]

# logical-or-expression

def p_logical_or_expression_1(p):
    'logical_or_expression : logical_and_expression'
    p[0] = p[1]
    print "p_logical_or_expression_1", p[1]

def p_logical_or_expression_2(t):
    'logical_or_expression : logical_or_expression LOR logical_and_expression'
    pass

# logical-and-expression

def p_logical_and_expression_1(p):
    'logical_and_expression : inclusive_or_expression'
    p[0] = p[1]
    print "p_logical_and_expression_1", p[1]

def p_logical_and_expression_2(t):
    'logical_and_expression : logical_and_expression LAND inclusive_or_expression'
    pass

# inclusive-or-expression:

def p_inclusive_or_expression_1(p):
    'inclusive_or_expression : exclusive_or_expression'
    p[0] = p[1]
    print "p_inclusive_or_expression_1", p[1]

def p_inclusive_or_expression_2(t):
    'inclusive_or_expression : inclusive_or_expression OR exclusive_or_expression'
    pass

# exclusive-or-expression:

def p_exclusive_or_expression_1(p):
    'exclusive_or_expression :  and_expression'
    p[0] = p[1]
    print "p_exclusive_or_expression_1", p[1]

def p_exclusive_or_expression_2(t):
    'exclusive_or_expression :  exclusive_or_expression XOR and_expression'
    pass

# AND-expression

def p_and_expression_1(p):
    "and_expression : equality_expression"
    p[0] = p[1]
    print "p_and_expression_1", p[1]

def p_and_expression_2(t):
    'and_expression : and_expression AND equality_expression'
    pass


# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    "equality_expression : equality_expression EQ relational_expression"

def p_equality_expression_3(t):
    'equality_expression : equality_expression NE relational_expression'
    pass


# relational-expression:
def p_relational_expression_1(p):
    "relational_expression : shift_expression"
    p[0] = p[1]
    print "p_relational_expression_1", p[1]

def p_relational_expression_2(p):
    "relational_expression : relational_expression LT shift_expression"
    p[0] = bfrast.Compare(p[1], p[2], p[3])

def p_relational_expression_3(p):
    "relational_expression : relational_expression GT shift_expression"
    p[0] = bfrast.Compare(p[1], p[2], p[3])

def p_relational_expression_4(p):
    "relational_expression : relational_expression LE shift_expression"
    p[0] = bfrast.Compare(p[1], p[2], p[3])

def p_relational_expression_5(p):
    "relational_expression : relational_expression GE shift_expression"
    p[0] = bfrast.Compare(p[1], p[2], p[3])

# shift-expression

def p_shift_expression_1(p):
    'shift_expression : additive_expression'
    p[0] = p[1]

def p_shift_expression_2(t):
    'shift_expression : shift_expression LSHIFT additive_expression'
    pass

def p_shift_expression_3(t):
    'shift_expression : shift_expression RSHIFT additive_expression'
    pass

# additive-expression

def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]
    print "p_additive_expression_1", p[1]

def p_additive_expression_2(p):
    'additive_expression : additive_expression PLUS multiplicative_expression'
    p[0] = bfrast.BinAdd(p[1], p[3])

def p_additive_expression_3(p):
    'additive_expression : additive_expression MINUS multiplicative_expression'
    p[0] = bfrast.BinSub(p[1], p[3])

# multiplicative-expression

def p_multiplicative_expression_1(p):
    'multiplicative_expression : cast_expression'
    p[0] = p[1]

def p_multiplicative_expression_2(p):
    "multiplicative_expression : multiplicative_expression TIMES cast_expression"
    p[0] = bfrast.BinMul(p[1], p[3])
    print "p_multiplicative_expression_2", p[0], p[1], p[3]

def p_multiplicative_expression_3(t):
    "multiplicative_expression : multiplicative_expression DIVIDE cast_expression"
    pass

def p_multiplicative_expression_4(t):
    "multiplicative_expression : multiplicative_expression MOD cast_expression"
    pass

# cast-expression:

def p_cast_expression_1(p):
    'cast_expression : unary_expression'
    p[0] = p[1]

def p_cast_expression_2(t):
    "cast_expression : LPAREN type_name RPAREN cast_expression"
    pass

# unary-expression:
def p_unary_expression_1(p):
    "unary_expression : postfix_expression"
    p[0] = p[1]

def p_unary_expression_2(p):
    "unary_expression : PLUSPLUS unary_expression"
    p[0] = bfrast.PreInc(p[1])

def p_unary_expression_3(p):
    "unary_expression : MINUSMINUS unary_expression"
    p[0] = bfrast.PreDec(p[1])

def p_unary_expression_4(p):
    "unary_expression : unary_operator cast_expression"
    if "&" == p[1]:
      p[0] = bfrast.TakeAddr(p[2])
    else:
      raise ValueError, repr(p[1])

def p_unary_expression_5(p):
    "unary_expression : SIZEOF unary_expression"
    pass

def p_unary_expression_6(p):
    "unary_expression : SIZEOF LPAREN type_name RPAREN"
    pass

#unary-operator
def p_unary_operator(p):
    """unary_operator : AND
                    | TIMES
                    | PLUS
                    | MINUS
                    | NOT
                    | LNOT"""
    p[0] = p[1]

# postfix-expression:
def p_postfix_expression_1(p):
    'postfix_expression : primary_expression'
    p[0] = p[1]
    print "p_postfix_expression_1", p[1]

def p_postfix_expression_2(p):
    'postfix_expression : postfix_expression LBRACKET expression RBRACKET'
    pass

def p_postfix_expression_3(p):
    "postfix_expression : postfix_expression LPAREN argument_expression_list RPAREN"
    p[0] = bfrast.CallFunc(p[1], p[3])

def p_postfix_expression_4(p):
    'postfix_expression : postfix_expression LPAREN RPAREN'
    p[0] = bfrast.CallFunc(p[1])

def p_postfix_expression_5(p):
    'postfix_expression : postfix_expression PERIOD ID'
    pass

def p_postfix_expression_6(p):
    'postfix_expression : postfix_expression ARROW ID'
    pass

def p_postfix_expression_7(p):
    "postfix_expression : postfix_expression PLUSPLUS"
    p[0] = bfrast.PostInc(p[1])

def p_postfix_expression_8(p):
    'postfix_expression : postfix_expression MINUSMINUS'
    p[0] = bfrast.PostDec(p[1])

# primary-expression:
def p_primary_expression(p):
    """primary_expression :  ID
                        |  constant
                        |  SCONST
                        |  LPAREN expression RPAREN"""
    if len(p) > 2:
        p[0] = p[2]
    else:
        if isinstance(p[1], bfrast.Const):
          p[0] = p[1]
        elif p[1].isalpha():
          p[0] = bfrast.Name(p[1])
        else:
          p[0] = bfrast.Const(p[1])
    print "p_primary_expression", p[1]

# argument-expression-list:
def p_argument_expression_list(p):
    """argument_expression_list :  assignment_expression
                              |  argument_expression_list COMMA assignment_expression"""
    if len(p) > 2:
      p[0] = p[1] + [p[3]]
    else:
      p[0] = [p[1]]

# constant:
def p_constant(p):
   """constant : ICONST
              | FCONST
              | CCONST"""
   p[0] = bfrast.Const(p[1])


def p_empty(t):
   "empty : "
   t[0] = bfrast.Empty()

def p_error(t):
    print("Whoa. We're hosed")

# Build the grammar

yacc.yacc(method='LALR')

#import profile
#profile.run("yacc.yacc(method='LALR')")

def parseFile(fname):
  global G_text
  f = open(fname)
  G_text = f.read()
  ast = bfrast.File(fname, yacc.parse(G_text, tracking=True))
  return ast

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
