from sys import argv, exit
if len(argv) > 2 and argv[2] == '1': # debug mode -- will drop into a pdb prompt if an exception occurs
    import debug
from plexception import PLException
from lexer import run_lexer
from parser import run_parser
from semantics import run_semantic_analyzer, State
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Returns False if errors occurred.
def run(f, state):
    try:
        tokens = []
        run_lexer(lambda x: tokens.append(x), f)
        ast = run_parser(tokens)

        #print("AST:", end=' ')
        #pp.pprint(ast)
        #exit(0)

        s, state = run_semantic_analyzer(ast, state)
        print("--AAST:",s)
        print("--Type constraints (for each key and value, the key and value are equal):")
        pp.pprint(state.typeConstraints)
    except PLException:
        return False
    return True

if __name__ == '__main__':
    if len(argv) > 1:
        with open(argv[1], 'r') as f:
            hadNoErrors = run(f, State())
            exit(0 if hadNoErrors else 1)
    else:
        # Open REPL
        import repl
        state = State()
        repl.run(lambda f: run(f, state))
