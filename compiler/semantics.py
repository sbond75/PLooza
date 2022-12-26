# Semantic analyzer; happens after parsing. Includes type-checking.

from collections import namedtuple
import pprint
pp = pprint.PrettyPrinter(indent=4)

AST = namedtuple("AST", ["lineno", "type", "args"])

def proc(state, ast):
    ast = AST(ast[0], ast[1], ast[2:])
    procMap[ast.type](state, ast)

def stmtBlock(state, ast, i=0):
    stmts = ast.args[0]
    whereclause = ast.args[1]
    ret = []
    procRestIndex = state.newProcRestIndex()
    for x in stmts[i:]:
        print('stmt in block:', x)
        state.setProcRest(lambda: stmtBlock(state, ast, i+1), procRestIndex)
        ret.append(proc(state, x))
        if state.processedRest():
            ret += state.rest
            break
        i += 1
    print(ret)
    return ret

def stmtInit(state, ast):
    pass

def stmtDecl(state, ast):
    type = AST(*ast.args[0])
    ident = AST(*ast.args[1])
    name = ident.args
    typename = type.args
    with state.newBindings([name], [Identifier(name, typename)]):
        return state.procRest([]) # Continuation-passing style of some sort?

def identifier(state, ast):
    pass

def mapAccess(state, ast):
    pass

def functionCall(state, ast):
    pass

def assign(state, ast):
    pass

def rangeExclusive(state, ast):
    pass

def rangeInclusive(state, ast):
    pass

def escaped(state, ast):
    pass

def old(state, ast):
    pass

def rangeGT(state, ast):
    pass

def rangeLE(state, ast):
    pass

def rangeLT(state, ast):
    pass

def rangeGE(state, ast):
    pass

def listExpr(state, ast):
    pass

def lambda_(state, ast):
    pass

def braceExpr(state, ast):
    pass

def new(state, ast):
    pass

def plus(state, ast):
    pass

def times(state, ast):
    pass

def minus(state, ast):
    pass

def divide(state, ast):
    pass

def negate(state, ast):
    pass

def lt(state, ast):
    pass

def le(state, ast):
    pass

def eq(state, ast):
    pass

def not_(state, ast):
    pass

def exprIdentifier(state, ast):
    pass

def integer(state, ast):
    pass

def float(state, ast):
    pass

def string(state, ast):
    pass

def true(state, ast):
    pass

def false(state, ast):
    pass

procMap = {
    'stmt_block': stmtBlock,
    'stmt_init': stmtInit,
    'stmt_decl': stmtDecl,
    'identifier': identifier,
    # Exps:
    'mapAccess': mapAccess,
    'functionCall': functionCall,
    'assign': assign,
    'range_exclusive': rangeExclusive,
    'range_inclusive': rangeInclusive,
    'escaped': escaped,
    'old': old,
    'range_gt': rangeGT,
    'range_le': rangeLE,
    'range_lt': rangeLT,
    'range_ge': rangeGE,
    'list_expr': listExpr,
    'lambda': lambda_,
    'brace_expr': braceExpr,
    'new': new,
    'plus': plus,
    'times': times,
    'minus': minus,
    'divide': divide,
    'negate': negate,
    'lt': lt,
    'le': le,
    'eq': eq,
    'not': not_,
    'expr_identifier': exprIdentifier,
    'integer': integer,
    'float': float,
    'string': string,
    'true': true,
    'false': false,
}

# O: map from identifier to Identifier
# maps: map from identifier to Map
class State:
    def __init__(self):
        self.O = dict()
        self.maps = dict()

        # Continuation-passing style stuff
        self.currentProcRest = [] # Stack of lambdas
        self.rest = None

    def processedRest(self):
        return self.rest is not None

    def addProcRest(self, procRest):
        self.rest = None
        self.currentProcRest.append(procRest)

    def newProcRestIndex(self):
        return (len(self.currentProcRest) - 1) if len(self.currentProcRest) > 0 else 0
        
    def setProcRest(self, procRest, index):
        self.rest = None
        if len(self.currentProcRest) == 0:
            assert index == 0
            self.currentProcRest.append(procRest)
        else:
            self.currentProcRest[index] = procRest
    
    def procRest(self, ret):
        if len(self.currentProcRest) > 0:
            ret_ = self.currentProcRest.pop()()
            self.rest = ret + ret_
            return self.rest
        return []

    # Creates bindings from `idents` (a list of variable names), to `bindings` (a list of `Identifier`s) for blocks' local bindings.
    # Usage: `with state.newBindings(...):`
    def newBindings(self, idents, bindings):
            return self.ContextManager(self, idents, bindings)

    # Based on https://docs.python.org/3/library/contextlib.html#contextlib.closing and https://www.geeksforgeeks.org/context-manager-in-python/
    class ContextManager():
            def __init__(self, s, idents, bindings):
                    self.s = s
                    self.idents = idents
                    self.bindings = bindings
                    self.prevValues = []

            def __enter__(self):
                    # Back up the current bindings before overwriting with the new one:
                    for ident, binding in zip(self.idents, self.bindings):
                            self.prevValues.append(self.s.O.get(ident))
                            self.s.O[ident] = binding
                    return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                    # Restore backups or remove newly added bindings that have no backups
                    for ident, prevValue in zip(self.idents, self.prevValues):
                            if prevValue is not None:
                                    # Restore backup
                                    self.s.O[ident] = prevValue
                            else:
                                    # Remove our newly added binding
                                    del self.s.O[ident]
                    self.prevValues.clear()


# annotated AST
class AAST:
    def __init__(self, lineNumber, resolvedType, astType, values):
        self.lineNumber = lineNumber
        self.type = resolvedType
        self.astType = astType
        self.values = values

    def __repr__(self):
        return "AAST:\n  \tline " + str(self.lineNumber) + "\n  \ttype " + str(self.type) +  "\n  \tAST type: " + str(self.astType) + "\n  \tvalues: " + str(self.values)

class Identifier:
    def __init__(self, name, type):
        self.name = name
        self.type = type

# PLooza map
class Map:
    def __init__(self):
        pass

def run_semantic_analyzer(ast):
    state = State()
    def run(x):
        proc(state, x)
    for x in ast:
        #pp.pprint(x)
        if isinstance(x, list):
            for y in x:
                run(y)
        else:
            run(x)
