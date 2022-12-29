# Semantic analyzer; happens after parsing. Includes type-checking.

import sys
from collections import namedtuple
from enum import Enum
from intervaltree import Interval, IntervalTree
from functools import reduce
import builtins
from autorepr import AutoRepr
import pprint
pp = pprint.PrettyPrinter(indent=4)

AST = namedtuple("AST", ["lineno", "type", "args"])

# Holds a shared object
class Box(AutoRepr):
    def __init__(self, item):
        self.item = item
    
    def toString(self):
        return "Box: item " + str(self.item)

class Type(Enum):
    Func = 1
    Map = 2
    Int = 3
    Float = 4
    String = 5
    Atom = 6
    Bool = 7
    Template = 8 # Type variable (can be anything basically). Ununified type (waiting to be resolved).
    Void = 9 # No return type etc. (for statements, side effects)
    #Array = 10 #  This is also a map type. It is a compile-time map that is an array which means the keys are from 0 to n-1 where n is the size of the array.
def typeToString(type):
    return {  Type.Func: "function"
            , Type.Map: "map"
            , Type.Int: "integer"
            , Type.Float: "float"
            , Type.String: "string"
            , Type.Atom: "atom"
            , Type.Bool: "boolean"
            , Type.Template: "any"
            , Type.Void: "void" }[type]

def toASTObj(ast):
    return AST(ast[0], ast[1], ast[2:])

def proc(state, ast, type=None):
    pp.pprint(("proc: about to proc:", ast))
    
    ret = []
    if type is None:
        ast = toASTObj(ast)
        ret = procMap[ast.type](state, ast)
    elif type == "args":
        for x in ast:
            x = AST(x[0], x[1], x[2:])
            ret.append(procMap[x.type](state, x))
    
    pp.pprint(("proc:", ast, "->", ret))
    return ret

def ensure(bool, msg, lineno):
    if not bool:
        msg = "ERROR: " + str(lineno) + ": Type-Check: " + msg()
        print(msg)
        raise Exception(msg)
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
        #print('x:',x);input()
        ret.append(proc(state, x))
        #1/0
        if state.processedRest():
            ret = state.rest.pop()
            break
        i += 1
    #print(ret)
    return AAST(lineNumber=ast.lineno, resolvedType=ret[-1].type, astType=ret[-1].astType, values=ret) # Type of the block becomes the type of the last statement in the block (return value)

def stmtInit(state, ast):
    type = AST(*ast.args[0])
    ident = AST(*ast.args[1])
    name = ident.args
    typename = type.args

    rhs = proc(state, ast.args[2])
    
    def p():
        identO = state.O[name]
        ensure(rhs.type == identO.type, lambda: "Right-hand side of initializer must have the same type as the declaration type (" + str(typename) + ")", rhs.lineno)
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
    if typename == "t":
        return Type.Template
    # if typename == "v":
    #     return Type.Void
    ensure(False, lambda: "Unknown type " + str(typename), lineno)

def stmtDecl(state, ast):
    type = AST(*ast.args[0])
    ident = AST(*ast.args[1])
    name = ident.args
    typename = type.args
    t = typenameToType(typename, ast.lineno)
    with state.newBindings([name], [Identifier(name, t,
                                               None if t != Type.Map else PLMap(state.O["$map"] # Maps start out with default methods for maps
                                                                                , dict() # map contents
                                                                                , Type.Template, Type.Template     # key and value types are ununified so far
                                                                           )
                                               )]):
        retval= state.procRest([]) # Continuation-passing style of some sort?
    #1/0
    return retval

class PLMap(AutoRepr):
    def __init__(self, prototype, contents, keyType, valueType):
        self.prototype = prototype
        self.contents = contents
        self.contents_intervalTree = IntervalTree()
        self.keyType = keyType
        self.valueType = valueType

    def printIntervalTree(self):
        return str(self.contents_intervalTree)

    # Adds all of the `other` PLMap's contents to `self`.
    def update(self, other):
        assert self.prototype == other.prototype
        assert self.keyType == other.keyType
        assert self.valueType == other.valueType
        self.contents.update(other.contents)
        self.contents_intervalTree.update(other.contents_intervalTree)

    def get(self, key, onNotFoundError):
        # Search prototype first
        temp_ = self.prototype.value.get(key)
        if temp_ is not None:
            return temp_
        
        temp = self.contents.get(key)
        if temp is None:
            if isinstance(key, DelayedMapInsert):
                # TODO: implement..
                onNotFoundError()
                return None
            if isinstance(key, (builtins.int, builtins.float)): # https://stackoverflow.com/questions/33311258/python-check-if-variable-isinstance-of-any-type-in-list
                temp2 = self.contents_intervalTree.overlap(key-1, key+1) # Endpoints are excluded in IntervalTree so we adjust for that here by "-1" and "+1"
                # temp2 is a set. So we return only the first element
                if len(temp2) != 0:
                    onNotFoundError()
                    return None
                assert len(temp2) == 1
                return next(iter(temp2)).data # Get first item in the set, then get its value (`.data`).
        return temp
        
    def toString(self):
        return "PLMap:\n  \tprototype " + str(self.prototype) + "\n  \tcontents " + str(self.contents) + f", {self.printIntervalTree()}\n  \tkeyType " + str(self.keyType) + "\n  \tvalueType " + str(self.valueType)

def identifier(state, ast):
    name = ast.args[0]
    identO = state.O.get(name)
    return AAST(lineNumber=ast.lineno, resolvedType=identO.type if identO is not None else None, astType=ast.type, values=ast.args)

def mapAccess(state, ast):
    return functionCall(state, ast, mapAccess=True)

def functionCall(state, ast, mapAccess=False):
    print("functionCall:", ast, f'mapAccess={mapAccess}')
    fnname = proc(state, ast.args[0])
    fnargs = proc(state, ast.args[1], type="args" if isinstance(ast.args[1], list) else None)
    ret = []

    if isinstance(fnname.values[0], AAST):
        assert not mapAccess
        # Resolved already; lookup the function in the map to get its prototype
        temp = fnname.values[0].values[0]
        temp = temp if not isinstance(temp, Identifier) else temp.name
        fncallee = state.O.get(temp if not isinstance(temp, Identifier) else temp.name)
        #print('\n\n',fncallee); input(); print('\n\n',fnname.values[1].values[0]); input()
        #print("fncallee:",fncallee,'\n\n'); print("fnname:",fnname); print("fnname.values[0].values[0]:",fnname.values[0].values[0]); print("fnargs:",fnargs); input()
        def onNotFoundError():
            assert False
        fnname_ = fncallee.value.get(fnname.values[1].values[0], onNotFoundError=onNotFoundError)
        fnident = Identifier(temp + "." + fnname.values[1].values[0], Type.Func, fnname_)
        #print(fnident);input()
    elif isinstance(fnname.values, DelayedMapInsert):
        # Evaluate lambda
        shift = fnname.values.fn(0)
        fnident = fnname.values.mapIdent
    else:
        # May be resolved already
        if isinstance(fnname.values[0], Identifier) and isinstance(fnname.values[0].value, PLMap) and fnname.values[0].value.prototype is not None:
            fnident = fnname.values[0]
        else:
            # Lookup the function in the environment to get its prototype
            #print('aaaaa',fnname); print('\n\n', fnname.values[0]);input()
            fnident = state.O.get(fnname.values[0])
        ensure(fnident is not None, lambda: "Undeclared function or map: " + str(fnname.values[0]), ast.lineno)
        ensure(fnident.type == Type.Func or fnident.type == Type.Map, lambda: "Expected type function or map", ast.lineno)
    # else:
    #     assert False, f"Unknown object type given: {fnname}"

    if fnident.type == Type.Func:
        # Check length of args
        ensure(len(fnident.value.paramTypes) == len(fnargs), lambda: "Calling function " + str(fnname) + " with wrong number of arguments (" + str(len(fnargs)) + "). Expected " + str(len(fnident.value.args)) + (" arguments" if len(fnident.value.args) != 1 else " argument"), ast.lineno)
        # Check types of arguments
        ts = []
        for arg,protoArgT,i in zip(fnargs,fnident.value.paramTypes,range(len(fnargs))):
            #print(arg,protoArgT);input()
            if protoArgT == Type.Template:
                # Unify (now we require arg.type as input to the function)
                protoArgT = arg.type
                fnident.value.paramTypes[i] = protoArgT # Save it to the prototype
            ensure(arg.type == protoArgT, lambda: f"Expected type {typeToString(protoArgT)} but got type {typeToString(arg.type)} for argument {i+1}", ast.lineno)
            ts.append(arg.type)
        return AAST(lineNumber=ast.lineno, resolvedType=fnident.value.returnType, astType=ast.type, values=(fnname,fnargs))
    elif fnident.type == Type.Map:
        # Look up the identifier (rhs of dot) in the parent identifier (lhs of dot)
        theMap = fnident.value
        ensure(fnident.type == Type.Map, lambda: "Name " + fnident.name + " refers to type " + typeToString(fnident.type) + ", not map, but it is being used as a map", ast.lineno)
        #print(fnargs.values[0], theMap);input()
        #print(fnargs);input()
        k = fnargs.values[0] if not isinstance(fnargs, list) else fnargs[0].values
        if isinstance(fnargs, list):
            assert isinstance(fnargs[0], AAST)
        fnidentReal = theMap.get(k, onNotFoundError=lambda: ensure(False, lambda: f"Map {fnident.name} doesn't contain key: {k}",
                                                                   fnargs.values[0].lineNumber if not isinstance(fnargs, list)
                                                                   else fnargs[0].lineNumber))
        ensure(fnidentReal is not None, lambda: "Map has no such key: " + str(k), ast.lineno)
        #print(fnidentReal);input()
        #print(fnargs);input()
        #print(fnident.type);input()
        #keyType = fnidentReal.type[0]
        #valueType = fnidentReal.type[1]
        #ensure(keyType == fnargs.type, lambda: "Key type is not what the map expects", ast.lineno)
        values = proc(state, ast.args, type="args")
        #print("values:",values);input()
        return AAST(lineNumber=ast.lineno, resolvedType=fnidentReal, astType=ast.type, values=values)
    else:
        assert False # Should never be reached

def assign(state, ast):
    pass

# Helper function
def rangeProc(state, ast):
    # This is a map type. It is a compile-time map that is an array which means the keys are from 0 to n-1 where n is the size of the array.
    m = PLMap(state.O["$map"], dict(), Type.Int, Type.Int)
    if ast.type == 'range_inclusive' or ast.type == 'range_exclusive':
        print(ast.args)
        start = proc(state, ast.args[0])
        end = proc(state, ast.args[1])
        ensure(start.type == Type.Int, f"Range start must be an integer, not {typeToString(start.type)}", start.lineNumber)
        ensure(end.type == Type.Int, f"Range end must be an integer, not {typeToString(start.type)}", start.lineNumber)
        # TODO: handle non-compile-time start and end, then it becomes non-compile-time map.

        # TODO: ensure integers below..
        startInt = start.values
        endInt = end.values
        size = endInt - startInt
        ensure(startInt <= endInt, f'Range starts after its end', ast.lineno)
    else:
        assert False,'not yet impl'
    def insert(shift):
        m.contents_intervalTree.addi(shift, shift + size, (startInt, endInt if ast.type == 'range_inclusive' else endInt - 1))
        return size
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Map,
                astType=ast.type, values=DelayedMapInsert(
                    Identifier(f'$tempMap_{state.newID()}', Type.Map, m),
                    insert
                ))

class DelayedMapInsert:
    def __init__(self, mapIdent, fn):
        self.mapIdent = mapIdent
        self.fn = fn

    def __getitem__(self, index):
        return self

def rangeExclusive(state, ast):
    return rangeProc(state, ast)

def rangeInclusive(state, ast):
    return rangeProc(state, ast)

def escaped(state, ast):
    pass

def old(state, ast):
    pass

def rangeGT(state, ast):
    return rangeProc(state, ast)

def rangeLE(state, ast):
    return rangeProc(state, ast)

def rangeLT(state, ast):
    return rangeProc(state, ast)

def rangeGE(state, ast):
    return rangeProc(state, ast)

def listExpr(state, ast):
    # Get contents of list
    # print(ast.args[0])
    # input()
    values = proc(state, ast.args[0], type='args')
    # Evaluate maps with their shifts
    # print(values)
    # input()
    shifts = list(reduce(lambda acc,x: acc + [x.values.fn(acc[len(acc)-1])], values, [0])) # (Mutates stuff in `values` by inserting to the maps)
    # Combine the maps into one "list_expr"
    m = PLMap(state.O["$map"], dict(), Type.Int, Type.Int)
    acc = Identifier(f'$tempMap_{state.newID()}', Type.Map, m)
    for x in values:
        acc.value.update(x.values.mapIdent.value)
    state.addTempIdentifier(acc)
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Map, astType=ast.type, values=[acc])

def lambda_(state, ast):
    args = list(map(lambda x: toASTObj(x), ast.args[0]))
    # Bind parameters
    t = [Box(Type.Template) for x in args]
    ti = iter(t)
    with state.newBindings(
            map(lambda x: toASTObj(x.args[0]).args[0], args),
            map(lambda x: Identifier(toASTObj(x.args[0]).args[0], next(ti) # starts out as template, until first and second pass of unification
                                     , None # no value yet
                                     ), args)):
        # Evaluate lambda body
        lambdaBody = proc(state, ast.args[1])
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Func, astType=ast.type, values=FunctionPrototype(t, Type.Template, lambdaBody))

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
    ensure(name.type is not None, lambda: "Unknown identifier " + str(name.values[0]), ast.lineno)
    return name

def integer(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Int, astType=ast.type, values=int(ast.args[0]))

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

class FunctionPrototype(AutoRepr):
    def __init__(self, paramTypes, returnType, body=None):
        self.paramTypes = paramTypes
        self.returnType = returnType
        self.body = body

    def toString(self):
        return "FunctionPrototype:\n  \tparamTypes " + str(self.paramTypes) + "\n  \treturnType " + str(self.returnType) +  "\n  \tbody: " + str(self.body)

# O: map from identifier to Identifier
class State:
    def __init__(self):
        self.O = dict()

        # Add stdlib #
        # Map prototype
        self.O.update({"$map" : Identifier("$map", Type.Map, {
              'add': FunctionPrototype([Type.Template, Type.Template], Type.Void) # format: ([param types], return type)
            , 'map': FunctionPrototype([Type.Template], Type.Template)
        })
                       })
        # #

        # Continuation-passing style stuff
        self.currentProcRest = [] # Stack of lambdas
        self.rest = []

        # Counter for temp identifier names
        self.lastID = 0

    # Adds an identifier forever, usually use this for temporary objects with unique names that start with dollar signs
    def addTempIdentifier(self, ident):
        assert ident.name.startswith("$")
        self.O[ident.name] = ident
        
    def newID(self):
        retval = self.lastID
        self.lastID += 1
        return retval

    def processedRest(self):
        return len(self.rest) >= len(self.currentProcRest)

    def addProcRest(self, procRest):
        self.currentProcRest.append(procRest)

    def newProcRestIndex(self):
        return len(self.currentProcRest)
        
    def setProcRest(self, procRest, index):
        if len(self.currentProcRest) == 0:
            assert index == 0
            self.currentProcRest.append(procRest)
        else:
            self.currentProcRest[index-1] = procRest
    
    def procRest(self, ret):
        if len(self.currentProcRest) > 0:
            ret_ = self.currentProcRest.pop()()
            retval = ret + [ret_]
            self.rest.append(retval)
            return retval
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
                            #1/0
                    return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                    # Restore backups or remove newly added bindings that have no backups
                    for ident, prevValue in zip(self.idents, self.prevValues):
                            print("Remove binding:", ident, "->", self.s.O[ident])
                            #1/0
                            if prevValue is not None:
                                    # Restore backup
                                    self.s.O[ident] = prevValue
                            else:
                                    # Remove our newly added binding
                                    del self.s.O[ident]
                    self.prevValues.clear()


# annotated AST
class AAST(AutoRepr):
    def __init__(self, lineNumber, resolvedType, astType, values):
        self.lineNumber = lineNumber
        self.type = resolvedType
        self.astType = astType
        self.values = values

    def toString(self):
        return "AAST:\n  \tline " + str(self.lineNumber) + "\n  \ttype " + str(self.type) +  "\n  \tAST type: " + str(self.astType) + "\n  \tvalues: " + str(self.values)

class Identifier(AutoRepr):
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    def toString(self):
        return "Identifier:\n  \tname " + str(self.name) + "\n  \ttype " + str(self.type) + "\n  \tvalue: " + str(self.value)

# PLooza map
class Map:
    def __init__(self):
        pass

def run_semantic_analyzer(ast, state = None):
    if state is None:
        state = State()
    i = 0
    ret = []
    didProcessRest = False
    def run(x):
        nonlocal ret
        nonlocal didProcessRest
        index = state.newProcRestIndex()
        state.setProcRest(lambda: run_semantic_analyzer(ast[i+1:], state), index)
        ret.append(proc(state, x))
        if state.processedRest():
            ret = state.rest.pop()
            didProcessRest = True
    for x in ast:
        #pp.pprint(x)
        if isinstance(x, list):
            for y in x:
                run(y)
        else:
            run(x)
        if didProcessRest:
            return ret
        i += 1
    return ret
