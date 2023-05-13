from sys import argv, exit
from plexception import PLException
from lexer import run_lexer
try:
    from parser import run_parser
except ImportError:
    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    import os
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("parser", os.path.join(os.path.dirname(os.path.abspath(__file__)), "parser.py"))
    foo = importlib.util.module_from_spec(spec)
    sys.modules["parser"] = foo
    spec.loader.exec_module(foo)
    from parser import run_parser
from semantics import run_semantic_analyzer, State
import argparse
import builtins
from debugOutput import print, input, pp
import debugOutput

def handleException(e, state):
    builtins.print(e)

    if debugOutput.debugErr:
        # For debugging
        import traceback
        traceback.print_exc()

        import pdb
        pdb.set_trace()

    return False, None, state

# Returns False if errors occurred.
def run(f, state, rethrow=False, skipInterpreter=False):
    try:
        tokens = []
        run_lexer(lambda x: tokens.append(x), f)
        ast = run_parser(tokens)

        print("--AST:", end=' ')
        pp.pprint(ast)
        #exit(0)

        aast, state = run_semantic_analyzer(ast, state, skipInterpreter)
        print("--AAST:",aast)
        print("--Type constraints (for each key and value, the key and value are equal):")
        pp.pprint(state.typeConstraints)
    except PLException as e:
        if rethrow:
            raise
        else:
            return handleException(e, state)

    if debugOutput.debugOutput:
        # For debugging: inspecting the `state` after execution
        import pdb
        pdb.set_trace()
        
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
    argparser.add_argument('--debug-err', dest='debugErr', action='store_true',
                           help='whether to enable pdb prompt when there is a syntax, semantic, etc. error in the compilation')

    args = argparser.parse_args()
    
    if args.debugMode:
        import debug # Installs handlers for pdb
    debugOutput.debugOutput = args.debugOutput
    debugOutput.debugErr = args.debugErr

    if args.src is not None:
        with open(args.src, 'r') as f:
            hadNoErrors, aast, state = run(f, State())
            exit(0 if hadNoErrors else 1)
    else:
        # Open REPL
        import repl
        state = State()
        def replIter(line):
            nonlocal state
            import io
            try:
                f = io.StringIO(line)
                hadNoErrors, aast, state = run(f, state, rethrow=True)
            except PLException as eOrig:
                if not line.rstrip().endswith(';') and not len(line.strip()) == 0:
                    # Try again with auto-added semicolon
                    try:
                        f = io.StringIO(line + ";")
                        hadNoErrors, aast, state = run(f, state, rethrow=True)
                    except PLException as e2:
                        # Show new exception
                        hadNoErrors, aast, state = handleException(e2, state)
                else:
                    hadNoErrors, aast, state = handleException(eOrig, state)
            
            # Auto-print expressions
            if aast is None: return
            from tree_walk_interpreter import unwrapAll
            from semantics import Type
            for x in aast:
                x = unwrapAll(x, present=True)
                if x != Type.Void and not isinstance(x, list): # TODO: slight hack, may be wrong
                    builtins.print(x)
        repl.run(replIter)

if __name__ == '__main__':
    main()
