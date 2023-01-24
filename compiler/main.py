from sys import argv, exit
from plexception import PLException
from lexer import run_lexer
from parser import run_parser
from semantics import run_semantic_analyzer, State
import argparse
import builtins
from debugOutput import print, input, pp
import debugOutput

# Returns False if errors occurred.
def run(f, state, rethrow=False, skipSecondPass=False):
    try:
        tokens = []
        run_lexer(lambda x: tokens.append(x), f)
        ast = run_parser(tokens)

        #print("AST:", end=' ')
        #pp.pprint(ast)
        #exit(0)

        aast, state = run_semantic_analyzer(ast, state, skipSecondPass)
        print("--AAST:",aast)
        print("--Type constraints (for each key and value, the key and value are equal):")
        pp.pprint(state.typeConstraints)
    except PLException as e:
        if rethrow:
            raise
        else:
            builtins.print(e)

            if debugOutput.debugOutput:
                # For debugging
                import traceback
                traceback.print_exc()

            return False, None, state
    return True, aast, state

def main():
    # https://docs.python.org/3/library/argparse.html
    argparser = argparse.ArgumentParser(description='Run the PLooza compiler.')
    argparser.add_argument('src', metavar='source file', type=str, nargs='?',
                           help='the source file to compile, or none to enter the REPL')
    argparser.add_argument('--debug', dest='debugMode', action='store_true',
                           help='whether to enable debug mode -- will drop into a pdb prompt if an unhandled exception within the compiler occurs')
    argparser.add_argument('--debug-output', dest='debugOutput', action='store_true',
                           help='whether to enable printing of debug information')

    args = argparser.parse_args()
    
    if args.debugMode:
        import debug # Installs handlers for pdb
    debugOutput.debugOutput = args.debugOutput

    if args.src is not None:
        with open(args.src, 'r') as f:
            hadNoErrors, aast, state = run(f, State())
            exit(0 if hadNoErrors else 1)
    else:
        # Open REPL
        import repl
        state = State()
        def replIter(f):
            nonlocal state
            hadNoErrors, aast, state = run(f, state)
            
            # Auto-print expressions
            if aast is None: return
            from tree_walk_interpreter import unwrapAll
            from semantics import Type
            for x in aast:
                x = unwrapAll(x)
                if x != Type.Void and not isinstance(x, list): # TODO: slight hack, may be wrong
                    builtins.print(x)
        repl.run(replIter)

if __name__ == '__main__':
    main()
