from lexer import run_lexer
from parser import run_parser
import pprint
pp = pprint.PrettyPrinter(indent=4)

tokens = []
run_lexer(lambda x: tokens.append(x))
ast = run_parser(tokens)

print("AST:", end=' ')
pp.pprint(ast)
