#! /usr/bin/env python
# -*- coding: utf-8 -*-

def xpf(f):
  def anonymous(p, func=f):
    print func.__name__, p
    return func(p)
  return anonymous

def _lstr(l, surround_with_brackets=0):
  sep = ""
  s = ""

  if surround_with_brackets:
    s += "["

  for i in l:
    s += sep
    if isinstance(i, list):
      s += _lstr(i, 1)
    elif isinstance(i, str):
      s += '"' + i + '"'
    else:
      s += str(i)
    sep = ","

  if surround_with_brackets:
    s += "]"

  return s

def _smerge(name, *args):
  return name + "(" + _lstr(args) + ")"

#####
# Internal classes for storing intermediate results during parsing.
#####
class i_pDecl:
  def __init__(self, decl, lst=[]):
    self.decl = decl
    self.lst = lst
#####

class Assign:
  def __init__(self, target, expr):
    self.target = target
    self.expr = expr
  def __str__(self):
    return _smerge("Assign", self.target, self.expr)

class UnaryOp:
  def __init__(self, expr):
    self.expr = expr
  def __str__(self):
    return _smerge(self.name, self.expr)

class TakeAddr(UnaryOp):
  name = "TakeAddr"

class BinOp:
  def __init__(self, expr_left, expr_right):
    self.expr_left = expr_left
    self.expr_right = expr_right
  def __str__(self):
    return _smerge(self.name, self.expr_left, self.expr_right)

class BinAdd(BinOp):
  name = "BinAdd"
class BinMul(BinOp):
  name = "BinMul"
class BinSub(BinOp):
  name = "BinSub"

class CallFunc:
  def __init__(self, expr, args=[]):
    self.expr = expr
    self.args = args
  def __str__(self):
    return _smerge("CallFunc", self.expr, self.args)

class Compare:
  def __init__(self, left, op, right):
    self.left = left
    self.op = op
    self.right = right
  def __str__(self):
    return _smerge("Compare" , self.left, self.op, self.right)

class CompoundStatement:
  def __init__(self, slist=[]):
    if slist is None: raise ValueError
    self.slist = slist
  def __str__(self):
    return _smerge("CompoundStatement", self.slist)

class Const:
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return _smerge("Const", self.value)

class Declare:
  def __init__(self, typeid, lst):
    self.typeid = typeid
    self.lst = lst
  def __str__(self):
    return _smerge("Declare", self.typeid, self.lst)

class InitDeclarator:
  def __init__(self, dtor, value):
    self.dtor = dtor
    self.value = value
  def __str__(self):
    return _smerge("InitDeclarator", self.dtor, self.value)

class FunctionDeclarator:
  def __init__(self, expr, params):
    self.expr = expr
    self.params = params
  def __str__(self):
    return _smerge("FunctionDeclarator", self.expr, self.params)

class SimpleDeclarator:
  def __init__(self, name):
    self.name = name
  def __str__(self):
    return _smerge("SimpleDeclarator", self.name)

class Empty:
  pass

class File:
  def __init__(self, fname, ast):
    self.fname = fname
    self.ast = ast

  def __str__(self):
    return _smerge("File", self.fname, self.ast)

class For:
  def __init__(self, expr1, expr2, expr3, code):
    self.expr1 = expr1
    self.expr2 = expr2
    self.expr3 = expr3
    self.code = code
  def __str__(self):
    return _smerge("For", self.expr1, self.expr2, self.expr3, self.code)

class Function:
  def __init__(self, name, args, retarg, code):
    self.name = name
    self.args = args
    self.retarg = retarg
    self.code = code
  def __str__(self):
    return _smerge("Function", self.name, self.args, self.retarg, self.code)

class Name:
  def __init__(self, name):
    self.name = name
  def __str__(self):
    return _smerge("Name", self.name)

class Return:
  def __init__(self, expr):
    self.expr = expr
  def __str__(self):
   return "Return(" + str(self.expr) + ")"

class AnyInc: # any increment
  def __init__(self, expr):
    self.expr = expr
  def __str__(self):
    return self.name + "(" + str(self.expr) + ")"

class PostInc(AnyInc): # post increment
  name = "PostInc"

class PtrDecl:
  def __init__(self, decl):
    self.decl = decl
  def __str__(self):
    return _smerge("PtrDecl", self.decl)

class TypeID:
  def __init__(self, name):
    self.name = name
  def __str__(self):
    return _smerge("TypeID", self.name)
