# Semantic analyzer; happens after parsing. Includes type-checking.

import sys
from collections import namedtuple
from enum import Enum
import pprint
pp = pprint.PrettyPrinter(indent=4)

AST = namedtuple("AST", ["lineno", "type", "args"])

class Type(Enum):
    Func = 1
    Map = 2
    Int = 3
    Float = 4
    String = 5
    Atom = 6
    Bool = 7

def proc(state, ast, type=None):
    ret = []
    if type is None:
        ast = AST(ast[0], ast[1], ast[2:])
        ret = procMap[ast.type](state, ast)
    elif type == "args":
        for x in ast:
            x = AST(x[0], x[1], x[2:])
            ret.append(procMap[x.type](state, x))
    
    pp.pprint(("proc:", ast, "->", ret))
    return ret

def ensure(bool, msg, lineno):
    if not bool:
        print("ERROR: " + str(lineno) + ": Type-Check: " + msg)
        1 / 0
        sys.exit(1)

def stmtBlock(state, ast, i=0):
    stmts = ast.args[0]
    whereclause = ast.args[1]
    ret = []
    procRestIndex = state.newProcRestIndex()
    iStart=i
    for x in stmts[i:]:
        print(f'[iStart={iStart},procRestIndex={procRestIndex}] stmt with index', i, 'in block:', x)
        state.setProcRest(lambda: stmtBlock(state, ast, i+1), procRestIndex)
        ret.append(proc(state, x))
        if state.processedRest():
            ret += state.rest
            break
        i += 1
    print(ret)
    return ret

def stmtInit(state, ast):
    type = AST(*ast.args[0])
    ident = AST(*ast.args[1])
    name = ident.args
    typename = type.args

    rhs = proc(state, ast.args[2])
    
    def p():
        identO = state.O[name]
        ensure(rhs.type == identO.type, "Right-hand side of initializer must have the same type as the declaration type (" + str(typename) + ")", rhs.lineno)
        if identO.type == Type.Func:
            fnargs = rhs.args[2]
            identO.value = (fnargs,)
            print(identO.value)
            input()

    state.setProcRest(p, state.newProcRestIndex())
    return stmtDecl(state, ast)

def typenameToType(typename, lineno):
    if typename == "l":
        return Type.Func
    if typename == "i":
        return Type.Int
    if typename == "f":
        return Type.Float
    if typename == "b":
        return Type.Bool
    if typename == "m":
        return Type.Map
    if typename == "s":
        return Type.String
    ensure(False, "Unknown type " + str(typename), lineno)

def stmtDecl(state, ast):
    type = AST(*ast.args[0])
    ident = AST(*ast.args[1])
    name = ident.args
    typename = type.args
    with state.newBindings([name], [Identifier(name, typenameToType(typename, ast.lineno), None)]):
        return state.procRest([]) # Continuation-passing style of some sort?

def identifier(state, ast):
    name = ast.args[0]
    identO = state.O.get(name)
    return AAST(lineNumber=ast.lineno, resolvedType=identO.type if identO is not None else None, astType=ast.type, values=ast.args)

def mapAccess(state, ast):
    return functionCall(state, ast)

def functionCall(state, ast):
    print("functionCall:", ast)
    fnname = proc(state, ast.args[0])
    fnargs = proc(state, ast.args[1], type="args" if isinstance(ast, list) else None)
    ret = []

    # Lookup the function in the environment to get its prototype
    fnident = state.O.get(fnname.values[0])
    ensure(fnident is not None, "Undeclared function or map: " + str(fnname), ast.lineno)
    ensure(fnident.type == Type.Func or fnident.type == Type.Map, "Expected type function or map", ast.lineno)
    
    if fnident.type == Type.Func:
        # Check length of args
        ensure(len(fnident.value.args) == len(fnargs), "Calling function " + str(fnname) + " with wrong number of arguments (" + str(len(fnargs)) + "). Expected " + str(len(fnident.value.args)) + (" arguments" if len(fnident.value.args) != 1 else " argument"), ast.lineno)
        # Check type of arguments
        ts = []
        for arg,protoArg in zip(fnargs,fnident.value.args):
            t = proc(state, arg)
            ensure(t.type == protoArg.type)
            ts.append(t)
    elif fnident.type == Type.Map:
        # Look up the identifier (rhs of dot) in the parent identifier (lhs of dot)
        fnidentReal = state.maps[fnname.values[0]].get(fnargs.name)
        ensure(fnidentReal is not None, "Map has no such key: " + str(fnargs.name), ast.lineno)
        keyType = fnidentReal.type[0]
        valueType = fnidentReal.type[1]
        ensure(keyType == fnargs.type, "Key type is not what the map expects", ast.lineno)
        return AAST(lineNumber=ast.lineno, resolvedType=valueType, astType=ast.type, values=ast.args)
    else:
        assert False # Should never be reached

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
    name = proc(state, ast.args[0])
    ensure(name.type is not None, "Unknown identifier " + str(name.values[0]), ast.lineno)
    return name

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
        return len(self.currentProcRest)
        
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
                            print("New binding:", ident, "->", binding)
                    return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                    # Restore backups or remove newly added bindings that have no backups
                    for ident, prevValue in zip(self.idents, self.prevValues):
                            print("Remove binding:", ident, "->", self.s.O[ident])
                            1/0
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
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    def __repr__(self):
        return "Identifier:\n  \tname " + str(self.name) + "\n  \ttype " + str(self.type) + "\n  \tvalue: " + str(self.value)

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
