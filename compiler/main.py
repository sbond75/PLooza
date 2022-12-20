from lexer import run_lexer
from parser import run_parser

tokens = []
run_lexer(lambda x: tokens.append(x))
ast = run_parser(tokens)
