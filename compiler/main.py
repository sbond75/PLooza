from lexer import run_lexer
from parser import run_parser
from semantics import run_semantic_analyzer
import pprint
pp = pprint.PrettyPrinter(indent=4)

tokens = []
run_lexer(lambda x: tokens.append(x))
ast = run_parser(tokens)

#print("AST:", end=' ')
#pp.pprint(ast)
#exit(0)

s, state = run_semantic_analyzer(ast)
print("--AAST:",s)
print("--Type constraints (for each key and value, the key and value are equal):")
pp.pprint(state.typeConstraints)
