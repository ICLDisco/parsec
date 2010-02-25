import sys

import ast

def isi(obj, class_): return isinstance(obj, class_)

def walk_ast0(ast_):
  for node in ast.walk(ast):
    if isi(node, ast.FunctionDef):
      print node

def walk_ast(module):
  for node in module.body:
    if isi(node, ast.FunctionDef) and node.name.startswith("p_"):
      e = node.body[0]
      if isi(e, ast.Expr) and (e.value, ast.Str):
        print e.value.s

def main(argv):
  walk_ast(compile(open(argv[1]).read(), argv[1], "exec", ast.PyCF_ONLY_AST))
  return 0

if __name__ == "__main__":
  sys.exit(main(sys.argv))
