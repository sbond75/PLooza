# Semantic analyzer; happens after parsing. Includes type-checking.

import sys
from collections import namedtuple
from enum import Enum
from intervaltree import Interval, IntervalTree
from functools import reduce
import builtins
from autorepr import AutoRepr, indent, indentInc
from bidict import bidict
from plexception import PLException
from debugOutput import print, input, pp # Replace default `print` and `input` to use them for debugging purposes
import debugOutput

AST = namedtuple("AST", ["lineno", "type", "args"])
def astSemanticDescription(ast):
    if isinstance(ast, AST):
        t = ast.type
    elif isinstance(ast, AAST):
        t = ast.astType
    else:
        assert False
    retval = {'plus': 'addition'
              , 'minus': 'subtraction'
              , 'times': 'multiplication'
              , 'divide': 'division'}.get(t)
    if retval is None:
        return ast.type
    return retval

def strWithDepth(x, depth):
    if depth > 20: # depth limit
        return f"<depth limit reached on {type(x)}>"
    if isinstance(x, AutoRepr):
        return indent(x.toString(depth + 1), indentInc)
    else:
        return str(x)

# Holds a shared object
class Box(AutoRepr):
    def __init__(self, item):
        self.item = item

    # Forwarded methods for Identifier #
    # `@property` lets you do self.name instead of self.name() for function calls, like a computed property in Swift ( https://www.pythonmorsels.com/making-read-only-attribute/ )
    # @property
    # def name(self):
    #     return self.item.name
    # @property
    # def type(self):
    #     return self.item.type
    # @property
    # def value(self):
    #     return self.item.value
    # #
    
    def toString(self, depth):
        return "Box: item " + strWithDepth(self.item, depth)
    
    def __eq__(self, obj):
        return isinstance(obj, Box) and obj.item == self.item

class Type(Enum):
    Func = 1
    Map = 2
    Int = 3
    Float = 4
    String = 5
    Atom = 6
    Bool = 7
    #Template = 8 # Type variable (can be anything basically). Ununified type (waiting to be resolved).
    Void = 9 # No return type etc. (for statements, side effects)
    #Array = 10 #  This is also a map type. It is a compile-time map that is an array which means the keys are from 0 to n-1 where n is the size of the array.

    def clone(self, state, lineno):
        return self

    def __eq__(self, other):
        if isinstance(other, Box): # Then unbox `other`
            return self is other.item or self.value == other.item
        # if isinstance(other, TypeVar):
        #     return True
        if isinstance(other, State.TypeEither):
            return other.t1 == self or other.t2 == self
        return self is other or self.value == other

    # https://stackoverflow.com/questions/72664763/python-enum-with-eq-method-no-longer-hashable
    def __hash__(self):
        return hash(self.value)

class TypeResolution:
    pass
    
# Type variable, aka template type from C++
class TypeVar(TypeResolution, AutoRepr):
    def __init__(self, name):
        # if name == "T_5_14":
        #     import pdb; pdb.set_trace()
        self.name = name

    def toString(self, depth):
        return strWithDepth(self.name, depth)

    def clone(self, state, lineno):
        return TypeVar(self.name + f"_{state.newID()}")

    # def __eq__(self, other):
    #     # if isinstance(other, Type.Template):
    #     #     return True
    #     return self is other
    
    def __eq__(self, other):
        if isinstance(other, TypeVar):
            return self.name == other.name
        if isinstance(other, State.TypeEither):
            return self == other.t1 or self == other.t2
        if isinstance(other, Type):
            return False # need state.resolveType to get an answer of different semantics than this
        return self is other
    
    # def __hash__(self):
    #     return hash(self.name)

    
def typeToString(type):
    retval = {  Type.Func: "function"
              , Type.Map: "map"
              , Type.Int: "integer"
              , Type.Float: "float"
              , Type.String: "string"
              , Type.Atom: "atom"
              , Type.Bool: "boolean"
              #, Type.Template: "any"
              , Type.Void: "void" }.get(type)
    if retval is None:
        assert isinstance(retval, TypeVar)
        return "any"
    return retval

def toASTObj(ast):
    return AST(ast[0], ast[1], ast[2:])

def proc(state, ast, type=None):
    pp.pprint(("proc: about to proc:", ast))
    
    ret = []
    used = []
    if type is None:
        ast = toASTObj(ast)
        processor = procMap[ast.type]
        print(f"proc: using: {processor}")
        ret = processor(state, ast)
        used.append(processor)
    elif type == "args":
        i=0
        for x in ast:
            x = AST(x[0], x[1], x[2:])
            processor = procMap[x.type]
            print(f"proc: arg {i}: using: {processor}")
            ret.append(processor(state, x))
            used.append(processor)
            i+=1
    
    pp.pprint(("proc:", ast, f"--[{used}]->", ret))
    return ret

def ensure(bool, msg, lineno, tryIfFailed=None, exceptionType=None):
    if tryIfFailed is None and debugOutput.debugErr:
        tryIfFailed = debugOutput.handleErr
    def error():
        nonlocal msg
        msg = "ERROR: " + str(lineno if not callable(lineno) else lineno()) + ": Type-Check: " + msg()
        raise PLException(msg) if exceptionType is None else exceptionType(msg)
    if not bool:
        if tryIfFailed is not None:
            if not tryIfFailed(error):
                error()
        else:
            error()

def stmtBlock(state, ast):
    stmts = ast.args[0]
    whereclause = ast.args[1]
    ret = []
    state.pushBlock() # make a new scope for the declarations in this block
    for x in stmts:
        #print('x:',x);input()
        ret.append(proc(state, x))

    state.popBlock()
    #print(ret)
    return AAST(lineNumber=ast.lineno, resolvedType=ret[-1].type, astType=ast.type, values=ret) # Type of the block becomes the type of the last statement in the block (return value)

# A base type is Type.Int, Type.Bool, etc.
def isBaseType(t):
    return not isinstance(t, TypeVar) and not isinstance(t, FunctionPrototype) and isinstance(t, Type)

def stmtInit(state, ast):
    type = AST(*ast.args[0])
    assert len(ast.args[1]) == 1, "Expected one identifier for name of variable being initialized in stmtInit"
    ident = AST(*ast.args[1][0])
    name = ident.args
    typename = type.args

    rhs = proc(state, ast.args[2])
    
    def p():
        identO = state.O[name]
        # import builtins
        # builtins.print("RHS")
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        def makeRhsType_(rhs):
            if isinstance(rhs.type, TypeVar):
                return rhs.type
            rhsType_ = rhs.type if isinstance(rhs.type, FunctionPrototype) else rhs.values
            rhsType_ = rhsType_.type if isinstance(rhsType_, AAST) else rhsType_
            if isinstance(rhs.type, Type) and not isinstance(rhsType_, FunctionPrototype):
                return rhs.type
            return rhsType_
        # if identO.name == 'kestrelIdX':
        #     import pdb
        #     pdb.set_trace()
        print("identO.type:",identO.type,"rhs.type:",rhs.type,"rhs.values:",rhs.values)
        rhsType_ = makeRhsType_(rhs)
        print("rhsType_ final:", rhsType_)

        # if rhsType_ == Type.Func:
        #     assert rhs.astType == 'functionCall'
        #     # Currying must be happening. We have to make a new wrapping FunctionPrototype for this function call, whose `.body` is `rhs`.
        #     fp = rhs.values[0].values[0]
        #     if isinstance(fp, Identifier):
        #         fp = fp.value
        #     assert isinstance(fp, FunctionPrototype)

        #     # Remove args we applied in currying
            
            
        #     rhs = AAST(rhs.lineno, rhs.type, 'lambda', fp)
        #     rhsType_ = makeRhsType_(rhs)
        
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        
        rhsType = state.resolveType(rhsType_)
        print(f'{state.resolveType(rhsType.returnType) if isinstance(rhsType, FunctionPrototype) else None} -++++++++++++++++')# {pp.pformat(state.typeConstraints)}')
        ensure(rhsType == identO.type or (isinstance(rhsType, FunctionPrototype) and identO.type == Type.Func), lambda: f"Right-hand side of initializer (of type {rhsType}) must have the same type as the declaration type (" + str(typename) + ")", type.lineno #rhs.lineNumber
               )
        # DONETODO: uncomment the above 2 lines
        # if isBaseType(rhsType):
            # import code
            # code.InteractiveConsole(locals=locals()).interact()
        identO.value = rhs
        # else:
        #     identO.value = rhsType if isinstance(rhsType_, TypeVar) else rhs.values
        # if identO.type == Type.Func: # TODO: fix below
        #     fnargs = rhs.args[2]
        #     identO.value = (fnargs,)
        #     print(identO.value)
        #     input()
        # builtins.print("RHS END")
        # import code
        # code.InteractiveConsole(locals=locals()).interact()

        if isinstance(rhsType, FunctionPrototype):
            # Add clarified name
            rhsType.presentableNames.append(identO.name)
        
        return AAST(lineNumber=ast.lineno, resolvedType=None # just a statement with side effects only
                    , astType=ast.type, values=[identO,rhs.values])

    retval = stmtDecl(state, ast)
    return p()

def typenameToType(state, typename, lineno, onNotFound=None):
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
        return state.newTypeVar()
    # if typename == "v":
    #     return Type.Void
    if onNotFound:
        return onNotFound()
    else:
        ensure(False, lambda: "Unknown type " + str(typename), lineno)

def stmtDecl(state, ast):
    def tryTreatingThisAsAFunctionCall():
        # Wrap the single "argument" in an expr_identifier AST node, and put it in a list. Then wrap the "function name" into an expr_identifier.
        arguments = ast.args[1:]
        functionName = ast.args[0]
        arguments = arguments[0]
        functionName = AST(lineno=toASTObj(functionName).lineno, type='expr_identifier', args=functionName)
        astNew = AST(ast.lineno, 'functionCall', args=(functionName,arguments))
        try:
            return functionCall(state, astNew)
        except PLUnknownIdentifierException:
            # Try treating it as an import expression.
            if len(arguments) > 1:
                raise # Not an import since there's more args
            possiblyImportCall = functionName.args[2]
            if possiblyImportCall == "import":
                astNew = AST(ast.lineno, 'import', args=arguments[0])
                return import_(state, astNew)
            raise # https://nedbatchelder.com/blog/200711/rethrowing_exceptions_in_python.html : "Here the raise statement means, “throw the exception last caught”."
    
    type = AST(*ast.args[0])
    if len(ast.args[1]) > 1:
        return tryTreatingThisAsAFunctionCall()
    assert len(ast.args[1]) == 1
    ident = AST(*ast.args[1][0])
    name = ident.args
    if isinstance(name, tuple): # unwrap the expr_identifier
        assert ident.type == 'expr_identifier'
        innerAST = AST(*name)
        assert innerAST.type == 'identifier'
        name = innerAST.args
    typename = type.args
            
    if state.O.get(type.args) is not None: # We may be doing a function call, and `type` is actually the function identifier
        return tryTreatingThisAsAFunctionCall();
    t = typenameToType(state, typename, ast.lineno, onNotFound=tryTreatingThisAsAFunctionCall)
    if isinstance(t, AAST): # We treated it as a function call
        return t
    state.addBindingsToCurrentBlock([name], [Identifier(name, t,
                                                        None if t != Type.Map else PLMap(state.O["$map"] # Maps start out with default methods for maps
                                                                                         , dict() # map contents
                                                                                         , state.newTypeVar(), state.newTypeVar()     # key and value types are ununified so far
                                                                                         )
                                                        )])
    return AAST(lineNumber=ast.lineno, resolvedType=None # just a statement with side effects only
                    , astType=ast.type, values=[])

class Presentable:
    def presentableString(self):
        raise Exception("Subclasses should override presentableString()")

class PLMap(AutoRepr, Presentable):
    def __init__(self, prototype, contents, keyType, valueType):
        self.prototype = prototype
        self.contents = contents
        self.contents_intervalTree = IntervalTree()
        self.contentsOrder = [] # List of intervals (tuples) or keys (any type) in the order they were inserted at compile-time
        self.requestedKeys = dict() # When you try to get a key from a map but the key isn't found, we store it in here since this is a key that is requested, and the key could even be a function parameter which, since it is a function parameter, hasn't been bound yet. This dict `requestedKeys` is from key to an instance of the Identifier class. The Identifier instance acts as a placeholder.
        self.keyType = keyType
        self.valueType = valueType

    def addKeyValuePair(self, key, value):
        self.contentsOrder.append(key)
        self.contents[key] = value

    def addInterval(self, begin, end, data):
        self.contentsOrder.append((begin, end, data))
        self.contents_intervalTree.addi(begin, end, data)

    def printIntervalTree(self, depth):
        return strWithDepth(self.contents_intervalTree, depth)

    # Adds all of the `other` PLMap's contents to `self`.
    def update(self, other):
        assert self.prototype == other.prototype
        assert self.keyType == other.keyType
        assert self.valueType == other.valueType
        # self.contents.update(other.contents)
        # self.contents_intervalTree.update(other.contents_intervalTree)

        for intervalOrKey in other.contentsOrder:
            self.contentsOrder.append(intervalOrKey)
            if isinstance(intervalOrKey, tuple):
                begin, end, data = intervalOrKey
                self.contents_intervalTree.addi(begin, end, data)
            else:
                # Some key
                self.contents[intervalOrKey] = other.contents[intervalOrKey]

    def get(self, key, onNotFoundError):
        # Search prototype first
        temp_ = self.prototype.value.get(key)
        if temp_ is not None:
            return temp_

        # Check for intervals
        temp = None
        if isinstance(key, int):
            # Find the value by using the key as the index into the intervaltree
            counter = 0
            for intervalOrKey in self.contentsOrder:
                if isinstance(intervalOrKey, tuple):
                    intervalOrKey = intervalOrKey[2] # grab "data" of the interval
                    counter += intervalOrKey[1] - intervalOrKey[0] + 1 # `+ 1` to include both endpoints
                    if counter >= key:
                        indexOfTemp = intervalOrKey[0] - counter
                        # assert indexOfTemp == key
                        # temp = intervalOrKey[0] + (intervalOrKey[1] - key)
                        temp = indexOfTemp
                else:
                    # Some key
                    counter += 1
                    if counter >= key:
                        temp = self.contents.get(key)
                        break
        else:
            temp = self.contents.get(key)
        
        #temp = self.contents.get(key)
        if temp is None:
            if isinstance(key, DelayedMapInsert):
                assert False, 'TODO: implement..'
                res = onNotFoundError()
                return res
            # if isinstance(key, (builtins.int, builtins.float)): # https://stackoverflow.com/questions/33311258/python-check-if-variable-isinstance-of-any-type-in-list
            #     temp2 = self.contents_intervalTree.overlap(key-1, key+1) # Endpoints are excluded in IntervalTree so we adjust for that here by "-1" and "+1"
            #     # temp2 is a set. So we return only the first element
            #     if len(temp2) == 0:
            #         res = onNotFoundError()
            #         return res
            #     assert len(temp2) == 1, f"Expected {temp2} to have length 1"
            #     return next(iter(temp2)).data # Get first item in the set, then get its value (`.data`).
            else:
                res = onNotFoundError()
                return res
        return temp

    # Gets from the map, but returns a placeholder type if not found
    def lazyGet(self, key, state):
        res = self.get(key, onNotFoundError=lambda: self.getOrAddRequestedKey(key, state))
        return res

    def getOrAddRequestedKey(self, key, state):
        if key in self.requestedKeys:
            return self.requestedKeys[key]
        retval = Identifier(key, state.newTypeVar(), None)
        self.requestedKeys[key] = retval
        return retval

    def items(self):
        return (list(self.contents.items())
                #+ sorted(self.contents_intervalTree.items()) # `sorted` seems to be required to get the iteration order to be correct; otherwise, it starts with indices later in the "array" (map)
                + self.contents_intervalTree.items()
                )
        
    def toString(self, depth):
        return "PLMap:\n  \tprototype " + strWithDepth(self.prototype, depth) + "\n  \tcontents " + strWithDepth(self.contents, depth) + f", {self.printIntervalTree(depth)}\n  \tkeyType " + str(self.keyType) + "\n  \tvalueType " + strWithDepth(self.valueType, depth)

    def presentableString(self):
        import io
        acc = io.StringIO()
        counter = 0
        for intervalOrKey in self.contentsOrder:
            if isinstance(intervalOrKey, tuple):
                intervalOrKey = intervalOrKey[2] # grab "data" of the interval
                next = counter + intervalOrKey[1] - intervalOrKey[0] + 1 # `+ 1` to include both endpoints
                acc.write(f"\t{counter}..<={next - 1}    {intervalOrKey[0]}..<={intervalOrKey[1]}\n")
            else:
                # Some key
                next = counter + 1
                acc.write(f"\t{intervalOrKey}    {self.contents[intervalOrKey]}\n")
            counter = next
        return acc.getvalue()

def identifier(state, ast):
    name = ast.args[0]
    identO = state.O.get(name)
    return AAST(lineNumber=ast.lineno, resolvedType=identO.type if identO is not None else None, astType=ast.type, values=ast.args[0])

def mapAccess(state, ast):
    return functionCall(state, ast, mapAccess=True)

def functionCall(state, ast, mapAccess=False, tryIfFailed=None):
    print("functionCall:", ast, f'mapAccess={mapAccess}')
    def unwrapType(x):
        return x.type if x.type is not Type.Func else (State.unwrap(x.values, noFurtherThan=[AAST])[0].type if isinstance(State.unwrap(x.values, noFurtherThan=[AAST])[0], AAST) else State.unwrap(x.values, noFurtherThan=[AAST])[0])
    try:
        fnname = proc(state, ast.args[0])
    except PLUnknownIdentifierException:
        # print(ast.args[1])
        # print(ast.args[0][2][2])
        # exit(0)
        if (len(ast.args[1]) == 1
            and ast.args[0][2][2] == 'import' # Reach into `(1, 'expr_identifier', (1, 'identifier', 'import'))` to get 'import'
            ):
            # Try treating it as an import expression
            return import_(state, AST(lineno=ast.args[0][0], type='import', args=ast.args[1][0]))
        raise
    fnargs = proc(state, ast.args[1], type="args" if isinstance(ast.args[1], list) else None)
    def clone(val):
        orig = val
        val = val.values
        print('[(((((((((((((',val)

        if not hasattr(val, 'clone'):
            return orig
        val = val.clone() if not isinstance(val, FunctionPrototype) else val
        def getIt1():
            return val.value.values
        def getIt2():
            return val.value
        def changeIt1(new):
            val.value.values = new
        def changeIt2(new):
            val.value = new
        # Provide functions to both get (`getIt`) and set (`changeIt`) the function prototype contained somewhere within `val` depending on the structure/type of `val`:
        if isinstance(val, FunctionPrototype):
            orig.values = val.clone(state, ast.lineno, cloneConstraints=True)
            return orig
        if isinstance(val.value, AAST):
            changeIt = changeIt1
            getIt = getIt1
        else:
            changeIt = changeIt2
            getIt = getIt2

        print('--------n---------------',val)
        if isinstance(getIt(), FunctionPrototype):
            # Clone types
            changeIt(getIt().clone(state, ast.lineno, cloneConstraints=True))
        else:
            return orig

        print('[(2(((((((((((((',val)
        orig.values = val
        return orig
    fnargs = [clone(x) for x in fnargs] if isinstance(fnargs, list) else fnargs
    print("fnname:", fnname, "fnargs:", fnargs); input()
    ret = []

    if isinstance(fnname.values, (list,tuple)) and isinstance(fnname.values[0], AAST):
        assert not mapAccess
        # Resolved already; lookup the function in the map to get its prototype
        temp = fnname
        #temp = fnname.values[0].values[0]
        #temp = temp if not isinstance(temp, Identifier) else temp.name
        # if isinstance(temp, AAST):
        #     if len(temp.values) == 1:
        #         lookup = temp.values[0]
        #     else:
        
        fn = state.resolveType(temp.type)
        skipLookup = False
        if isinstance(fn, TypeVar):
            # A function that isn't resolvable yet and could be a parameter to an enclosing lambda somewhere above this functionCall in the AST tree, so we process it specially:
            fnident = Identifier(f"$tempFnFromTypeVar_${state.newID()}", fn, fnname)
            skipLookup=True
        elif not isinstance(temp.type, FunctionPrototype):
            if not isinstance(temp.type, TypeVar):
                assert temp.type != Type.Func, f"Type isn't resolved properly: {temp.type} for {temp}"

                # If we get this far, this must be calling something as a function that isn't meant to be called (i.e. fnname has type Type.Void or Type.Int).
                ensure(False, lambda: f"Type {typeToString(temp.type)} isn't callable", temp.lineNumber)
            # We have a function to handle or some type thing
            assert isinstance(fn, FunctionPrototype)
            lookup = fn
        else:
            lookup = temp.type
        # else:
        #     lookup = temp
        if not skipLookup:
            lookup = lookup if not isinstance(lookup, Identifier) else lookup.name

            if isinstance(lookup, str):
                fncallee = state.O.get(lookup)
            else:
                fncallee = Identifier(f"$tempFn_${state.newID()}", Type.Func, lookup)

            #print('\n\n',fncallee); input(); print('\n\n',fnname.values[1].values[0]); input()
            #print("fncallee:",fncallee,'\n\n'); print("fnname:",fnname); print("fnname.values[0].values[0]:",fnname.values[0].values[0]); print("fnargs:",fnargs); input()
            def onNotFoundError():
                assert False
            print('>>>>>>>>>',fncallee)
            if isinstance(fncallee.value, FunctionPrototype):
                fnident = fncallee
            else:
                fnname_ = fncallee.value.get(fnname.values[1].values[0], onNotFoundError=onNotFoundError)
                fnident = Identifier(temp + "." + fnname.values[1].values[0], Type.Func, fnname_)
            #print(fnident);input()
    elif isinstance(fnname.values, DelayedMapInsert):
        # Evaluate lambda
        shift = fnname.values.fn(0)
        fnident = fnname.values.mapIdent
    else:
        # print('ppppppppppp',mapAccess,fnname.values[0]); input()
        # May be resolved already
        if isinstance(fnname.values, Identifier) and ((isinstance(fnname.values.value, PLMap) and fnname.values.value.prototype is not None) or isinstance(fnname.values.value, FunctionPrototype)):
            fnident = fnname.values
        elif isinstance(fnname.values, FunctionPrototype):
            aNewType = state.newTypeVar()
            state.unify(aNewType, fnname.values, fnname.lineNumber)
            fnident = Identifier(f'$tempLambda_{state.newID()}', aNewType, fnname.values)
        elif (isinstance(fnname.values.value, PLMap) and fnname.values.value.prototype is not None):
            assert False, f"Map with no prototype: {fnname.values}"
        else:
            # Lookup the function in the environment to get its prototype
            #print('aaaaa',fnname); print('\n\n', fnname.values);input()
            # print(fnname,'=============================================================1')
            fnident = state.O.get(fnname.values) if not isinstance(fnname.values, Identifier) else fnname.values
        ensure(fnident is not None, lambda: "Undeclared function or map: " + str(fnname.values), fnname.lineNumber)
        #ensure(fnident.type == Type.Func or fnident.type == Type.Map, lambda: "Expected type function or map", fnname.lineNumber)
    # else:
    #     assert False, f"Unknown object type given: {fnname}"

    def procFnCall(fnname, fnidentResolved, shouldCloneParamBindings):
        # Check types of arguments
        # Based on the body of the function `type_ptr ast_app::typecheck(type_mgr& mgr, const type_env& env) const` from https://danilafe.com/blog/03_compiler_typechecking/
        returnType = state.newTypeVar()
        # import code
        # code.InteractiveConsole(locals=locals()).interact()

        arrow = FunctionPrototype(list(map(lambda x: unwrapType(x), fnargs)), returnType, receiver=fnname).clone(state, fnname.lineNumber)
        #arrow = FunctionPrototype(list(map(lambda x: x.type if x.type is not Type.Func else (State.unwrap(x.values, noFurtherThan=[AAST])[0].type if isinstance(State.unwrap(x.values, noFurtherThan=[AAST])[0], AAST) else State.unwrap(x.values, noFurtherThan=[AAST])[0]), fnargs)), returnType, receiver=fnname)
        #arrow = FunctionPrototype(list(map(lambda x: x.type if x.type is not Type.Func else (State.unwrap(x.values, noFurtherThan=[AAST])[0].type if isinstance(State.unwrap(x.values, noFurtherThan=[AAST])[0], AAST) else State.unwrap(x.values, noFurtherThan=[AAST])[0]), fnargs)), returnType, receiver=fnname, paramBindings=([f'$arg_{i}' for i in range(len(fnargs))],None))
        
        # Allow for parametric polymorphism (template functions from C++ basically -- i.e. if we have a function `id` defined to be `x in x`, i.e. the identity function which returns its input, then `id` can be invoked with any type as a parameter.) #
        # print(fnname, fnident.value)
        # exit()

        valueNew = fnidentResolved.clone(state, fnname.lineNumber, cloneConstraints=True)
        # #

        print(valueNew,f'++++++++++++++++')# {pp.pformat(state.typeConstraints)}')

        
        # def getLessArgs():
        #     nonlocal aast
        #     # It is possible that the function is being applied with *more* arguments than it appears to take, due to currying. So, we split up the AST accordingly into two function calls and try again, thereby allowing currying.
        #     if len(fnargs) > 0:
        #         # Try with less args (one less arg at a time)
        #         restOfFnArgs = fnargs[-1:]
        #         reduced = fnargs[:-1]
        #         print("arg count reduced for type var function:\n", fnargs, "->\n", reduced, "with rest of args", restOfFnArgs)
        #         if len(reduced) > 0:
        #             arrow = FunctionPrototype(list(map(lambda x: x.type if x.type is not Type.Func else x.values, reduced)), returnType, receiver=fnname)
        #             return arrow
            
        #     # Return None since we couldn't resolve the error.
        #     return None
        # while arrow is not None:
        #     try:
        #         state.unify(arrow, valueNew, fnname.lineNumber)
        #         break
        #     except PLDiffNumArgsException as e:
        #         # import pdb
        #         # pdb.set_trace()
                
        #         arrow = getLessArgs() # (On the next loop iteration we will try again)

        # import pdb; pdb.set_trace()
        
        # try:
        state.unify(arrow, valueNew, fnname.lineNumber)
        # except PLDiffNumArgsException as e:
        #     aast = None
        #     tryLessArgs()
        #     if aast is not None:
        #         return aast
        #     else:
        #         raise

        
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        
        # ts = []
        # for arg,protoArgT,i in zip(fnargs,fnident.value.paramTypes,range(len(fnargs))):
        #     #print(arg,protoArgT);input()
        #     if protoArgT == Type.Template:
        #         # Unify (now we require arg.type as input to the function)
        #         protoArgT = arg.type #protoArgT = propagateTypes(protoArgT, arg)
        #         fnident.value.paramTypes[i] = protoArgT # Save it to the prototype
        #     ensure(arg.type == protoArgT, lambda: f"Expected type {typeToString(protoArgT)} but got type {typeToString(arg.type)} for argument {i+1}", ast.lineno)
        #     ts.append(arg.type)
        # return AAST(lineNumber=ast.lineno, resolvedType=fnident.value.returnType, astType=ast.type, values=(fnname,fnargs))

        # Clone param bindings so we can modify them without permanently binding arguments to the original function's prototype.
        #fn = fnident.value.cloneParamBindings(state)

        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        
        fnnameOld = fnname
        #fnname, fn = cloneParamBindings(fnname, state) if shouldCloneParamBindings else (fnname, fnname.type)
        fn = valueNew.cloneParamBindings(state)

        # # Bind the args in the prototype since this is a function call
        # for x,y in zip(fn.paramBindings[1], fnargs):
        #     x.value = y

        if shouldCloneParamBindings:
            # Bind the args in the prototype since this is a function call
            for x,y in zip(fn.paramBindings[1], fnargs):
                x.value = y
        
        #newFnname = AAST(lineNumber=fnname.lineNumber, resolvedType=fnident.type, astType='lambda', values=fn)

        # for i in range(len(fnargs)):
        #     if isinstance(fnargs[i].values, FunctionPrototype):
        #         fnargs[i].values = fnargs[i].values.cloneParamBindings()

        # import code
        # code.InteractiveConsole(locals=locals()).interact()

        # TODO: why do we need `valueNew` *and* `arrow`? (fnident.value is just the prototype for the constraints of `valueNew` (a clone to get its own separate constraints) so I get that one.)
        return AAST(lineNumber=fnname.lineNumber, resolvedType=valueNew.returnType, astType=ast.type, values=(fnname,fnargs))
    
    if fnident.type == Type.Func:
        print("fnident:",fnident)
        # Check length of args
        import tree_walk_interpreter
        fnidentResolved=fnident.value
        if isinstance(fnidentResolved, AAST):
            fnidentResolved = fnidentResolved.type if isinstance(fnidentResolved.type, TypeVar) else fnidentResolved.values
        fnidentResolved = tree_walk_interpreter.unwrapAll(fnidentResolved, preferFullType=True)
        if isinstance(fnidentResolved, TypeVar):
            fnidentResolved = state.resolveType(fnidentResolved)
        aast = None
        def tryLessArgs(e):
            nonlocal aast
            # It is possible that the function is being applied with *more* arguments than it appears to take, due to currying. So, we split up the AST accordingly into two function calls and try again, thereby allowing currying.
            if len(fnargs) > len(fnidentResolved.paramTypes):
                # Try with less args (one less arg at a time)
                restOfFnArgs = ast.args[1][-1:]
                args = ast.args[0:1] + (ast.args[1][:-1],) + ast.args[2:]
                res = (fnname.lineNumber, ast.type, (fnname.lineNumber, ast.type, *args), restOfFnArgs)
                print("arg count reduced:\n", ast.args, "->\n", args, "with rest of args", restOfFnArgs, "; proc call", pp.pformat(res))
                if len(args) > 0:
                    aast = proc(state, res)
                    return True
            
            # Return False since we couldn't resolve the error.
            return False
        ensure(len(fnidentResolved.paramTypes) == len(fnargs), lambda: "Calling function " + str(fnname) + " with wrong number of arguments (" + str(len(fnargs)) + "). Expected " + str(len(fnidentResolved.paramTypes)) + (" arguments" if len(fnidentResolved.paramTypes) != 1 else " argument") + f" but got {len(fnargs)}", fnname.lineNumber, tryIfFailed=tryLessArgs)
        if aast is not None:
            return aast
        
        return procFnCall(fnname, fnidentResolved, True)
    elif fnident.type == Type.Map:
        # Look up the identifier (rhs of dot) in the parent identifier (lhs of dot)
        theMap_ = fnident.value
        import tree_walk_interpreter
        theMap = tree_walk_interpreter.unwrapAll(theMap_, preferFullType=True) # Resolve it
        ensure(fnident.type == Type.Map, lambda: "Name " + fnident.name + " refers to type " + typeToString(fnident.type) + ", not map, but it is being used as a map", ast.lineno)
        #print(fnargs.values, theMap);input()
        #print(fnargs);input()
        k = fnargs.values if not isinstance(fnargs, list) else fnargs[0].values
        if isinstance(fnargs, list):
            ensure(len(fnargs) == 1, lambda: "Map lookup requires one argument as the key", ast.lineno)
        fnidentRealOrFn = theMap.lazyGet(k, state)
        # fnidentReal = theMap.get(k, onNotFoundError=lambda: ensure(False, lambda: f"Map {fnident.name} doesn't contain key: {k}",
        #                                                            fnargs.values.lineNumber if not isinstance(fnargs, list)
        #                                                            else fnargs[0].lineNumber))
        # ensure(fnidentReal is not None, lambda: "Map has no such key: " + str(k), ast.lineno)
        
        #print(fnidentReal);input()
        #print(fnargs);input()
        #print(fnident.type);input()
        #keyType = fnidentReal.type[0]
        #valueType = fnidentReal.type[1]
        #ensure(keyType == fnargs.type, lambda: "Key type is not what the map expects", ast.lineno)
        #values = proc(state, ast.args, type="args")
        # print("values:",values);input()
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        return AAST(lineNumber=fnname.lineNumber, resolvedType=(fnidentRealOrFn.type if not isinstance(fnidentRealOrFn, int) else Type.Int) if not isinstance(fnidentRealOrFn, FunctionPrototype) else fnidentRealOrFn, astType=ast.type, values=(fnname,fnargs))
    else:
        assert isinstance(fnident.type, TypeVar)

        # ensure(False, lambda: f"Function {fnname.values.name} has infinite type", fnname.lineNumber)

        # Enforce that this is a function call in the type constaints.
        returnType = state.newTypeVar()
        # builtins.print(fnargs)
        # builtins.print("------")
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        arrow = FunctionPrototype(list(map(lambda x: unwrapType(x), fnargs)), returnType, receiver=fnname, paramBindings=([f'$arg_{i}' for i in range(len(fnargs))],None)) # `noFurtherThan=[AAST]` since we don't want to unwrap the AAST for function args -- so we unwrap no further than the AAST
        #assert len(arrow.paramBindings[1]) == 1, f'size not 1 is not yet implemented ; {arrow} ; {fnargs}'
        #assert fnident.value is not None, fnident
        #arrow.body = fnident.value.body
        valueNew = fnident.type

        # # def getLessArgs():
        # #     nonlocal aast
        # #     # It is possible that the function is being applied with *more* arguments than it appears to take, due to currying. So, we split up the AST accordingly into two function calls and try again, thereby allowing currying.
        # #     if len(fnargs) > 0:
        # #         # Try with less args (one less arg at a time)
        # #         restOfFnArgs = fnargs[-1:]
        # #         reduced = fnargs[:-1]
        # #         print("arg count reduced for type var function:\n", fnargs, "->\n", reduced, "with rest of args", restOfFnArgs)
        # #         if len(reduced) > 0:
        # #             arrow = FunctionPrototype(list(map(lambda x: x.type if x.type is not Type.Func else x.values, reduced)), returnType, receiver=fnname)
        # #             return arrow
            
        # #     # Return None since we couldn't resolve the error.
        # #     return None
        # # while arrow is not None:
        # #     try:
        # #         state.unify(arrow, valueNew, fnname.lineNumber)
        # #         break
        # #     except PLDiffNumArgsException as e:
        # #         # import pdb
        # #         # pdb.set_trace()
                
        # #         arrow = getLessArgs() # (On the next loop iteration we will try again)

        state.unify(arrow, valueNew, fnname.lineNumber)
        
        # # # It is a variable function being applied to something, so just wrap it up an an AAST since this call can't be type-checked yet due to parametric polymorphism (the functionCall AST node could have any type signature (it is a TypeVar as asserted above)).
        # # print(fnident,'=============================2')
        # # print(fnname,'=============================3')
        # # print(fnargs,'=============================4')
        #return AAST(lineNumber=ast.lineno, resolvedType=returnType, astType=ast.type, values=(fnname,fnargs))

        # shouldCloneParamBindings = True

        # fnnameOld = fnname
        # fnname, fn = cloneParamBindings(fnname, state) if shouldCloneParamBindings else (fnname, fnname.type)

        # # # Bind the args in the prototype since this is a function call
        # # for x,y in zip(fn.paramBindings[1], fnargs):
        # #     x.value = y

        # if shouldCloneParamBindings:
        #     # Bind the args in the prototype since this is a function call
        #     for x,y in zip(fn.paramBindings[1], fnargs):
        #         x.value = y

        return AAST(lineNumber=ast.lineno, resolvedType=returnType, astType=ast.type, values=(fnname,fnargs))
                
        #return procFnCall(fnname, arrow, False)

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
        size = endInt - startInt - (1 if ast.type == 'range_exclusive' else 0)
        ensure(size > 0, f'Range starts at or after its end', ast.lineno)
    else:
        assert False,'not yet impl'
    def insert(shift):
        m.addInterval(shift, shift + size, (startInt, endInt if ast.type == 'range_inclusive' else endInt - 1))
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

def import_(state, ast):
    # It is an error to import non-"constexpr" things -- so the expression must not perform IO. Then it can be used.
    # TODO: (check for IO as mentioned above)

    toImport = proc(state, ast.args)
    
    # Now load in the code from `toImport`:
    import tree_walk_interpreter, main
    aast, state = tree_walk_interpreter.second_pass([toImport], state)
    # `aast` is now resolved as an Executed type.
    # print(aast)
    # exit(0)
    assert len(aast) == 1
    toImportRes = tree_walk_interpreter.unwrapAll(aast[0])
    try:
        with open(toImportRes, 'r') as f:
            hadNoErrors, aast, state = main.run(f, state, rethrow=True, skipSecondPass=True)
    except FileNotFoundError:
        ensure(False, lambda: f"File not found: {toImportRes}", ast.lineno)
    return aast

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
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Map, astType=ast.type, values=acc)

def lambda_(state, ast):
    args = list(map(lambda x: toASTObj(x), ast.args[0]))
    # Check for special parameter `_` which indicates no parameters (syntax: `_ in 123`)
    if toASTObj(args[0]).args[0][0][2] == '_':
        ensure(len(args) == 1, lambda: f"No parameter specifier was provided (`_`), but there should only be one parameter, `_`, not {len(args)} parameters", toASTObj(args[0]).lineno)
        
        # No parameters exist for this function
        t = []
        bindings = ([], [])
    else:
        # Bind parameters
        t = [state.newTypeVar() for x in args]
        ti = iter(t)
        bindings = (list(map(lambda x: toASTObj(x.args[0]).args[0], args)),
                    list(map(lambda x: Identifier(toASTObj(x.args[0]).args[0], next(ti) # starts out as template, until first and second pass of unification
                                                  , None # no value yet
                                                  ), args)))
        # Ensure no duplicates
        a = bindings[0]
        seen = set()
        dupes = [x for x in a if x in seen or seen.add(x)] # https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
        dupe = dupes[0] if len(dupes) > 0 else None
        # print(toASTObj(args[0]).args[0][0][2])
        ensure(len(dupes) == 0, lambda: f"Duplicate parameter name: {dupe}", lambda: next(x for x in args if toASTObj(x).args[0][0][2] == dupe).lineno)
            
    with state.newBindings(*bindings):
        # Evaluate lambda body
        lambdaBody = proc(state, ast.args[1])
    # Add type constraint for return value
    v = state.newTypeVar()
    print(lambdaBody);input('asd')
    state.unify(v, lambdaBody.type if lambdaBody.type != Type.Func else lambdaBody.values, ast.lineno)
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Func, astType=ast.type, values=FunctionPrototype(t, v, str(ast.args[0]), lambdaBody, paramBindings=bindings, functionID=state.newID()))

def braceExpr(state, ast):
    #exprs = list(map(lambda x: toASTObj(x), ast.args[2]))
    exprs = ast.args[0]
    retval = []
    for x in exprs:
        retval.append(proc(state, x))
    return AAST(lineNumber=ast.lineno, resolvedType=retval[-1].type, astType=ast.type, values=retval)

def new(state, ast):
    pass

# def isFunction(state, aast, possibleRetTypes):
#     def processFn(fn):
#         # Will fully evaluate later -- add a type constraint for now
#         # print(fn, fn.returnType, possibleRetTypes); input();input();input()
#         state.constrainTypeVariableToBeOneOfTypes(fn.returnType, possibleRetTypes)
#         return True
#     isFn = aast.type == Type.Func
#     if isFn:
#         # print(aast,'888888888')
#         fn = aast.values[0].value
#         if fn.returnType in possibleRetTypes:
#             return True
#         elif isinstance(fn.returnType, TypeVar):
#             return processFn(fn)
#     elif isinstance(aast.type, TypeVar):
#         fn = aast.values[0].values[0].value
#         return processFn(fn)
#     return False

def arith(state, ast):
    e1 = proc(state, ast.args[0])
    e2 = proc(state, ast.args[1])
    # ensure(e1.type == Type.Int or e1.type == Type.Float or isFunction(state, e1, possibleRetTypes={Type.Int, Type.Float}), lambda: f"First operand of {astSemanticDescription(ast)} must be an integer, float, or function returning an integer or float", ast.lineno)
    # ensure(e2.type == Type.Int or e2.type == Type.Float or isFunction(state, e2, possibleRetTypes={Type.Int, Type.Float}), lambda: f"Second operand of {astSemanticDescription(ast)} must be an integer, float, or function returning an integer or float", ast.lineno)
    t3 = state.newTypeVar()

    # Make t3 be the lub of e1 and e2
    #builtins.print(e1,e2)
    state.constrainTypeLeastUpperBound(t3, e1.type, e2.type, zeroType=Type.Int, zeroRes=Type.Int, oneType=Type.Float, oneRes=Type.Float, lineno=ast.lineno) # Says that e1 may be float or int, e2 may be float or int, but the result will be float
    # Logic table: notice this is "or" where "float" is 1 and "int" is 0:
    # e1      e2    |  res
    # -----------------------
    # int     int   |  int
    # float   int   |  float
    # int     float |  float
    # float   float |  float
    
    return AAST(lineNumber=ast.lineno, resolvedType=t3, astType=ast.type, values=(e1, e2))

def plus(state, ast):
    return arith(state, ast)

def times(state, ast):
    return arith(state, ast)

def minus(state, ast):
    return arith(state, ast)

def divide(state, ast):
    return arith(state, ast)

def negate(state, ast):
    e = proc(state, ast.args[0])
    ensure(e.type == Type.Int or e.type == Type.Float, lambda: "You can only negate integers or floats", ast.lineno)
    return AAST(lineNumber=ast.lineno, resolvedType=e.type, astType=ast.type, values=e)

def lt(state, ast):
    pass

def le(state, ast):
    pass

def eq(state, ast):
    e1 = proc(state, ast.args[0])
    e2 = proc(state, ast.args[1])
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Bool, astType=ast.type, values=(e1, e2))

def not_(state, ast):
    e = proc(state, ast.args[0])
    ensure(e.type == Type.Bool, lambda: "Only bools can be not'ed", ast.lineno)
    return AAST(lineNumber=ast.lineno, resolvedType=e.type, astType=ast.type, values=ast.args[0])

class PLUnknownIdentifierException(PLException):
    pass

def exprIdentifier(state, ast):
    name = proc(state, ast.args[0])
    ensure(name.type is not None, lambda: "Unknown identifier " + str(name.values), ast.lineno, exceptionType=PLUnknownIdentifierException)
    val = state.O.get(name.values)
    print(name,'(((((((((((((',val)

    def getIt():
        return State.unwrap(val)[0]

    # if val.name == 'true_':
    #     import pdb; pdb.set_trace()
    
    # print('--------n---------------',val)
    # if isinstance(getIt(), FunctionPrototype):
    #     # Clone types
    #     changeIt(getIt().clone(state, ast.lineno, cloneConstraints=True))
        
    # print(name,'(2(((((((((((((',val)

    retunwrapped = AAST(lineNumber=name.lineNumber, resolvedType=name.type, astType=name.astType, values=val)
        
    if isinstance(getIt(), FunctionPrototype) and len(getIt().paramTypes) == 0:
        # Special case of function call with no arguments. Make `val` into a function call.
        return AAST(lineNumber=name.lineNumber, resolvedType=getIt().returnType, astType='functionCall', values=(retunwrapped,
                                                                                                                   [] # No args
                                                                                                                   ))
    return retunwrapped

def integer(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Int, astType=ast.type, values=int(ast.args[0]))

def float(state, ast):
    pass

def string(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=Type.String, astType=ast.type, values=str(ast.args[0]))

def true(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Bool, astType=ast.type, values=True)

def false(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=Type.Bool, astType=ast.type, values=False)

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
    'import': import_,
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

class FunctionPrototype(AutoRepr, Presentable):
    def __init__(self, paramTypes, returnType, bodyAST=None, body=None, receiver=None, paramBindings=None, presentableNames=None, functionID=None):
        self.paramTypes = paramTypes
        self.returnType = returnType
        assert Type.Func not in self.paramTypes and self.returnType is not Type.Func, f"{self.paramTypes} -> {self.returnType}"
        self.bodyAST = bodyAST if bodyAST is not None else (body if isinstance(body, str) else None)
        self.body = body
        self.receiver = receiver
        if paramBindings is not None and paramBindings[1] is None:
            # Populate it automatically
            paramBindings = (paramBindings[0], [Identifier(x, y, None) for x,y in zip(paramBindings[0], self.paramTypes)])
        self.paramBindings = paramBindings
        self.presentableNames = [] if presentableNames is None else presentableNames
        self.functionID = functionID

    # When copying the functionprototype (which is just the arrow type), this function resolves and
    # then clones its type variables but for any nested functionprototypes within the
    # functionprototype's parameters or return types, it constrains the cloned original name of it
    # (before resolving) to that nested functionprototype. This is because the variables could
    # themselves resolve to functionprototypes. It puts all those resolved and cloned types into a
    # list, which only contains type variables, and then finds duplicates in the list. For all the
    # duplicates, it unifies those types with each other, which allows preservation of only the
    # necessary constraints from the original functionprototype.
    def clone(self, state, lineno, cloneConstraints=False):
        # Fixup the function prototype we are about to make to abstract embedded function prototypes into type vars
        otherParamTypes = []
        def proc(x):
            # if isinstance(x, FunctionPrototype):
            #     import pdb; pdb.set_trace()
            # # assert not isinstance(x, FunctionPrototype) # Else, need to constrain a new type variable to it first. Potential impl is below, just uncomment it:
            # if not isinstance(x, TypeVar):
            #     x2 = state.newTypeVar()
            #     state.unify(x2, x, lineno)
            #     x = x2
            # assert isinstance(x, TypeVar)
            
            y = state.resolveType(x)
            # x = x if hasattr(x, 'clone') else State.unwrap(x)[0]
            if isinstance(y, FunctionPrototype):
                other = x.clone(state, lineno)
                state.unify(other, y.clone(state, lineno), lineno)
            else:
                other = y.clone(state, lineno)
            return other
        for x in self.paramTypes:
            other = proc(x)
            otherParamTypes.append(other)
        otherReturnType = proc(self.returnType)
        
        retval = FunctionPrototype(otherParamTypes,
                                   otherReturnType,
                                   self.bodyAST,
                                   self.body,
                                   self.receiver,
                                   self.paramBindings,
                                   self.presentableNames,
                                   self.functionID)# .cloneParamBindings(state) # clone param bindings too so we can update its types below:
        # for x,y in zip(retval.paramBindings[1], otherParamTypes):
        #     x.type = y
        # # builtins.print(retval)
        
        # retval = FunctionPrototype(list(map(lambda x: x.clone(state, lineno), self.paramTypes)),
        #                            self.returnType.clone(state, lineno),
        #                            self.body,
        #                            self.receiver,
        #                            self.paramBindings)

        if True: #if cloneConstraints:
            # Make new constraints that point to the original function
            # Demo:
            # Given constraints:
            # T_1 : T_0
            # T_2 : T_1
            # T_3 : T_2
            # And we want to clone this function:
            # T_1 -> T_3
            # Then we resolve this function to:
            # T_0 -> T_0
            # And the clone becomes:
            # clone: T_0' -> T_0'
            
            (myTypes, myTypesDupes), (otherTypes, otherTypesDupes) = (self.allTypes(state), retval.allTypes(state))
            if len(otherTypesDupes) != len(myTypesDupes): # If this if statement is not true, then sometimes they are already "constrained" since they resolve to the same thing already
                for i,j in myTypesDupes:
                    state.unify(otherTypes[i], otherTypes[j], lineno)

            # import code
            # code.InteractiveConsole(locals=locals()).interact()
            
            # myTypes, myResolvedTypes, otherTypes = (self.paramTypes + [self.returnType]
            #                                         , list(map(lambda x: state.resolveType(x), self.paramTypes)) + [state.resolveType(self.returnType)]
            #                                         , retval.paramTypes + [retval.returnType])
            # for x,c,other in zip(myTypes,myResolvedTypes,otherTypes):
            #     # Only constrain to types that resolve to types contained within the function
            #     if c in myTypes:
            #         state.unify(x, c, lineno)
            #         # import code
            #         # code.InteractiveConsole(locals=locals()).interact()

            pass

        assert not any((x == y if isinstance(x, TypeVar) or isinstance(y, TypeVar)
                        else False # Consider Type.Void "!=" another Type.Void
                        for x,y in zip(self.allTypes(state)[0], retval.allTypes(state)[0]))) # All must be not equal ("The any() function returns True if any [at least one] element of an iterable is True. If not, it returns False." ( https://www.programiz.com/python-programming/methods/built-in/any ))
        return retval

    def allTypes(self, state, topLevel=True):
        retval = []
        for x in (self.paramTypes + [self.returnType]):
            x = state.resolveType(x)
            if isinstance(x, FunctionPrototype):
                retval.extend(x.allTypes(state, topLevel=False))
            else:
                retval.append(x)
        if topLevel:
            seen = set()
            dupes = [i for x,i in zip(retval,range(len(retval))) if (x.name if isinstance(x, TypeVar) # Sometimes we get function prototypes for `x` since it gets resolved to that
                                                                     else id(x)) in seen or seen.add((x.name if isinstance(x, TypeVar) else id(x)))]
            dupes2 = []
            for i in dupes:
                # Add first occurrence too
                j=i
                findIt = retval[j]
                dupes2.append((
                    next(i for i,v in enumerate(retval) if retval[i].name == findIt.name) # https://stackoverflow.com/questions/1701211/python-return-the-index-of-the-first-element-of-a-list-which-makes-a-passed-fun
                    , i))
            return retval, dupes2
        else:
            return retval

    def cloneParamBindings(self, state):
        import copy
        memoDict = dict() # deepcopy preserves objects that point to the same memory location (have the same id()) if you use the same memoDict for calls to it, and we use this since identifiers share objects between self.body and self.paramBindings[1] (i.e., Identifier objects in one should have the same memory address as those in the other)
        retval = FunctionPrototype(self.paramTypes,
                                   self.returnType,
                                   self.bodyAST,
                                   copy.deepcopy(self.body, memoDict),
                                   self.receiver,
                                   (self.paramBindings[0], copy.deepcopy(self.paramBindings[1], memoDict)),
                                   self.presentableNames,
                                   self.functionID)

        # # Update the type of param bindings using self.paramTypes since we have a copy of the paramBindings now. Technically could have been done in clone(self) but we do it here since it makes sense to update types here since we're cloning the body here but not in clone(self):
        # for x,y in zip(retval.paramBindings[1], self.paramTypes):
        #     x.type = y
        # builtins.print(retval.paramBindings[1])
        # import pdb
        # pdb.set_trace()
        
        # import builtins
        # builtins.print("CPB")
        # import code
        # code.InteractiveConsole(locals=locals()).interact()
        return retval

    # def equalsResolvingTypes(self, other, state):
    #     return self.body == other.body and self.allTypes(state) == other.allTypes(state)

    # def equalsName(self, other):
    #     return self.bodyAST == other.bodyAST and self.presentableNames == other.presentableNames

    def equalsID(self, other):
        assert self.functionID is not None and other.functionID is not None
        return self.functionID == other.functionID
    
    def toString(self, depth, state=None):
        return "FunctionPrototype" + ("[resolved]" if state is not None else "") + ":\n  \tparamTypes " + (strWithDepth(self.paramTypes, depth) if state is None else strWithDepth(list(map(lambda x: (x, state.resolveType(x)), self.paramTypes)), depth)) + "\n  \treturnType " + (strWithDepth(self.returnType, depth) if state is None else strWithDepth((self.returnType, state.resolveType(self.returnType)), depth)) + (("\n  \tpresentableNames " + strWithDepth(self.presentableNames, depth)) if self.presentableNames is not None else '') + (("\n  \tbodyAST " + strWithDepth(self.bodyAST, depth)) if self.bodyAST is not None else '') + "\n  \tbody " + strWithDepth(self.body, depth) + (("\n  \treceiver " + strWithDepth(self.receiver, depth)) if self.receiver is not None else '') + (("\n  \tparamBindings " + (strWithDepth(self.paramBindings, depth) if state is None else strWithDepth((self.paramBindings[0], list(map(lambda x: Identifier(x.name, (x.type, state.resolveType(x.type)), x.value), self.paramBindings[1]))), depth))) if self.paramBindings is not None else '')

    def presentableString(self):
        def prettyPrint(ast): return str(ast) # TODO: implement
        return f"<lambda {self.presentableNames if len(self.presentableNames) > 0 else prettyPrint(self.bodyAST)}>"

# `aast` must contain a FunctionPrototype. The topmost one will be used.
def cloneParamBindings(aast, state):
    import copy
    aast = copy.deepcopy(aast) # Prevent modifying the identifiers for functions
    if isinstance(aast.type, TypeVar):
        fn = state.resolveType(aast.type)
        ensure(isinstance(fn, FunctionPrototype), lambda: "Identifier {aast} must be a function", aast.lineNumber)
        fn = fn.cloneParamBindings(state)
        aast.type = fn
        return aast, aast.type
    elif isinstance(aast.values, tuple):
        assert False # Trying to call this on a functionCall type, for example, isn't good
        # if isinstance(aast.values[0], FunctionPrototype):
        #     aast.values = (aast.values[0].cloneParamBindings(state), aast.values[1])
        #     return aast, aast.values[0]
        # elif isinstance(aast.values[0], Identifier):
        #     aast.values[0].value = aast.values[0].value.cloneParamBindings(state)
        #     return aast, aast.values[0].value
        # elif isinstance(aast.values[0], AAST):
        #     temp, retval = cloneParamBindings(aast.values[0], state)
        #     aast.values = (temp, aast.values[1])
        #     return aast, retval
    elif isinstance(aast.values, FunctionPrototype):
        aast.values = aast.values.cloneParamBindings(state)
        return aast, aast.values
    elif isinstance(aast.values, Identifier):
        temp, retval = cloneParamBindings(aast.values.value, state)
        aast.values.value = temp
        return aast, retval
    elif isinstance(aast.values, list):
        assert aast.astType == 'mapAccess' # map access
        aast.type = aast.type.cloneParamBindings(state)
        return aast, aast.type
    assert False

class PLDiffNumArgsException(PLException):
    pass

class PLTypesDontMatchException(PLException):
    pass

# O: map from identifier to Identifier
class State:
    def __init__(self):
        # # Continuation-passing style stuff
        # self.currentProcRest = [] # Stack of lambdas
        # self.rest = []

        # Stack of ContextManagers for variable bindings
        self.bindingBlocks = []
        self.pushBlock() # Global scope block

        # Counter for temp identifier names
        self.lastID = 0

        # Hindley-Milner type-checking stuff
        self.typeConstraints = dict() # Map from type variable's name to "resolved type"

        # Variable name to Identifier map
        self.O = dict()

        # Add stdlib #
        # Map prototype
        self.O.update({"$map" : Identifier("$map", Type.Map, { # "member variables" present within $map, including methods (FunctionPrototype), etc.:
              'add': FunctionPrototype([self.newTypeVar(), self.newTypeVar()], Type.Void, body='$map.add', paramBindings=(['key', 'value'], None), receiver='$self', functionID=self.newID()) # format: ([param types], return type)
            , 'map': FunctionPrototype([ # 1-arg version of .map
                FunctionPrototype([self.newTypeVar()], self.newTypeVar(), functionID=self.newID()) # (This is PLMap.valueType -> Type.Template to be specific, for when only one type is used in the values)
                                        ], Type.Map, body='$map.map', receiver='$self', paramBindings=(['fn'], None), functionID=self.newID())
            # , 'map$2': FunctionPrototype([ # 2-arg version of .map
            #     ...
            #     , Type.Template # This is PLMap.valueType to be specific
            #                             ], Type.Template)
            
            # `.get` is an alias for "calling" the PLMap as if it were a function (with one argument: the key)
            , 'get': FunctionPrototype([self.newTypeVar()], self.newTypeVar(), body='$map.get', paramBindings=(['key'], None), receiver='$self', functionID=self.newID())
        })
                       })
        # IO library map
        self.O["io"] = Identifier("io", Type.Map, PLMap(Identifier("$emptyMap", Type.Map, {}), {
            'print': FunctionPrototype([self.newTypeVar() # any type
                                        ], Type.Void, body='$io.print', receiver='$self', paramBindings=(['x']
                                                                                                         ,None # Will auto-populate this one
                                                                                                         ), functionID=self.newID())
            # "Read integer" function (like Lua's readint):
            , 'readi': FunctionPrototype([], Type.Int, body='$io.readi', receiver='$self', functionID=self.newID())
        }, Type.String, Type.Func))
        # #

    # Adds an identifier forever, usually use this for temporary objects with unique names that start with dollar signs
    def addTempIdentifier(self, ident):
        assert ident.name.startswith("$")
        self.O[ident.name] = ident
        
    def newID(self):
        retval = self.lastID
        self.lastID += 1
        return retval

    # Begin types #
    
    def newTypeVar(self):
        i = self.newID()
        return TypeVar(f"T_{i}")

    # Returns a TypeVar pointing to the type resolved from the current set of type constraints, or if there is no resolution, returns a re-boxed version of the given TypeVar that hasn't been resolved (unified) yet.
    def resolveType(self, t):
        if isinstance(t, str): # Look up identifiers
            t = self.O[t].type
        # if not (isinstance(t, TypeVar) or isinstance(t, FunctionPrototype) or isinstance(t, Type)):
        #     t = State.unwrap(t)[0]
        if isinstance(t, TypeResolution) and not isinstance(t, TypeVar): # i.e., if it's a TypeEither, it can be resolved now
            return t
        assert isinstance(t, TypeVar) or isinstance(t, FunctionPrototype) or isinstance(t, Type)
        while isinstance(t, TypeVar):
            it = self.typeConstraints.get(t.name)
            if isinstance(it, TypeVar):
                t = it
                continue
            elif isinstance(it, State.TypeLeastUpperBound):
                res = it.evaluate(self)
                if res is None: # Can't evaluate yet, so just return the typevar
                    return t
            elif it is not None:
                return it
            return TypeVar(t.name)
        return t
        
    # Constraints TypeVar `l` to equal `r`.
    def constrainTypeVariable(self, l, r, lineno):
        assert r != Type.Func, "Need more info than just Type.Func -- i.e., need its actual FunctionPrototype"
        assert isinstance(l, TypeVar), f"{l}, {r}, {lineno}, {pp.pformat(self.typeConstraints)}"

        #assert l != r, "a = a is not a very useful equation to have." # https://danilafe.com/blog/03_compiler_typechecking/
        if l == r: return

        # if l.name == "T_9":
        #     import pdb; pdb.set_trace()
        existing = self.typeConstraints.get(l.name)
        assert existing is None, f"{l} {r} {existing}"
        if isinstance(r, tuple):
            1/0
            for rr in r:
                self.constrainTypeVariable(l, rr, lineno)
            return
        # existingR = self.typeConstraints.get(r.name)
        # assert existingR is None, f"{l}, {r}, {existing}, {existingR}, {pp.pformat(self.typeConstraints)}"
        
        #assert existing is None, f"{l} {r} {existing}" # Otherwise, we might need to make a type error or maybe support multiple constraints?
        #self.typeConstraints[l.name] = r

        print("88888888888888888888888888888888888888888")
        #pp.pprint(self.typeConstraints)
        print(l, r)
        #if existing is None:
        self.typeConstraints[l.name] = r
        # else:
        #     self.typeConstraints[l.name] = self.unify(existing, r, lineno)
    # def constrainTypeVariableToBeOneOfTypes(self, l, rOneOf):
    #     assert isinstance(l, TypeVar)
    #     assert isinstance(rOneOf, set)
    #     #assert self.typeConstraints.get(l.name) is None # Otherwise, we might need to make a type error or maybe support multiple constraints?
    #     c = self.typeConstraints.get(l.name)
    #     if c is not None:
    #         # We have to unify with existing constraints
    #         self.typeConstraints[l.name] = rOneOf.union(c if isinstance(c,set) else set([c]))
    #     else:    
    #         self.typeConstraints[l.name] = rOneOf

    # Unwraps `item` to get its type.
    def unwrap(item, assumeFunctionCall=False, noFurtherThan=[]):
        itemType = item
        itemOrig = item
        if isinstance(item, FunctionPrototype):
            if assumeFunctionCall:
                itemType = item.returnType
                # import code
                # code.InteractiveConsole(locals=locals()).interact()
            else:
                itemType = Type.Func
        elif isinstance(item, Type):
            return item, item
        elif not isinstance(item, TypeVar):
            if isinstance(item, str):
                return item, None
            # assert isinstance(item,AAST), f"Expected this to be an AAST: {item}"
            if isinstance(item,AAST) and AAST not in noFurtherThan:

                if item.type is not None:
                    print(item,item.type,'[[[[[[[[[[[[[[');input('123')
                    if item.type != Type.Func: # Func's can have more info so we continue past this if statement if it is a Func.
                        return item, item.type

                # print(item);input()
                item = item.values
                if isinstance(item, (tuple,list)):
                    assert len(item)==1
                    item = item[0]
                # print(item);input()
            if isinstance(item, Identifier) and Identifier not in noFurtherThan:
                while isinstance(item, Identifier) and Identifier not in noFurtherThan:
                    if item.type == Type.Func: # More info to process
                        item, itemType = State.unwrap(item.value, assumeFunctionCall)
                    else:
                        itemType = item.type
                        item = item.value
            else:
                if id(item) == id(itemOrig):
                    # Couldn't unwrap
                    # import builtins
                    # builtins.print("t:", item, itemType, itemOrig)
                    return item, None
            
                #assert isinstance(item, str), f"Expected this to be a string: {item}"
                #itemType = None

                # import code
                # code.InteractiveConsole(locals=locals()).interact()
                item, itemType = State.unwrap(item, assumeFunctionCall)
            #print(item,'dddddddddd',isinstance(item,AAST))
        #assert isinstance(item, TypeVar), f"Invalid unwrapping of: {item} of type {type(item)}"
        return item, itemType

    class TypeLeastUpperBound(AutoRepr):
        def __init__(self, t1, t2, zeroType, zeroRes, oneType, oneRes):
            self.t1 = t1
            self.t2 = t2
            self.zeroType = zeroType
            self.zeroRes = zeroRes
            self.oneType = oneType
            self.oneRes = oneRes

        def evaluate(self, state):
            t1 = state.resolveType(self.t1)
            t2 = state.resolveType(self.t2)
            zeroType = state.resolveType(self.zeroType)
            zeroRes = state.resolveType(self.zeroRes)
            oneType = state.resolveType(self.oneType)
            oneRes = state.resolveType(self.oneRes)
            if (t1 == oneType or t2 == oneType) and (t1 == zeroType or t2 == zeroType):
                return oneRes # Coerce any remaining zeroTypes into oneTypes
            elif t1 == zeroType and t2 == zeroType:
                return zeroRes
            else:
                return None # Not yet resolved

        def toString(self, depth):
            return strWithDepth(f"LUB({self.t1}, {self.t2}) ~> {self.oneRes}", depth) # (weird random custom notation)
    class TypeEither(TypeResolution, AutoRepr):
        def __init__(self, t1, t2):
            self.t1 = t1
            self.t2 = t2

        def clone(self, state, lineno):
            return State.TypeEither(self.t1.clone(state, lineno), self.t2.clone(state, lineno))
        
        def __eq__(self, other):
            if isinstance(other, TypeVar):
                return self.t1 == other or self.t2 == other
            if isinstance(other, State.TypeEither):
                return self.t1 == other.t1 and self.t2 == other.t2
            if isinstance(other, Type):
                return self.t1 == other or self.t2 == other
            return self is other
        
        def __hash__(self):
            return hash((self.t1, self.t2)) # https://stackoverflow.com/questions/2909106/whats-a-correct-and-good-way-to-implement-hash

        def toString(self, depth):
            return strWithDepth(f"Either({self.t1}, {self.t2})", depth)
    def constrainTypeLeastUpperBound(self, dest, t1, t2, zeroType, zeroRes, oneType, oneRes, lineno):
        assert isinstance(dest, TypeVar)
        # if not isinstance(t1, TypeVar):
        #     t1_ = self.newTypeVar()
        #     self.constrainTypeVariable(t1_, t1, lineno)
        #     t1 = t1_
        # if not isinstance(t2, TypeVar):
        #     t1_ = self.newTypeVar()
        #     self.constrainTypeVariable(t1_, t1, lineno)
        #     t1 = t1_
        
        test = self.typeConstraints.get(dest.name)
        assert test is None, f"Type {dest} is already constrained to {test}"
        self.typeConstraints[dest.name] = State.TypeLeastUpperBound(t1, t2, zeroType, zeroRes, oneType, oneRes)

        if isinstance(t1, TypeVar):
            test = self.typeConstraints.get(t1.name)
            assert test is None, f"Type {t1} is already constrained to {test}"
            self.typeConstraints[t1.name] = State.TypeEither(zeroType, oneType)

        if isinstance(t2, TypeVar):        
            test = self.typeConstraints.get(t2.name)
            assert test is None, f"Type {t2} is already constrained to {test}"
            self.typeConstraints[t2.name] = State.TypeEither(zeroType, oneType)
    
    # Calling a function `dest` ("left") using `src` ("right") as arguments for example
    def unify(self, dest, src, lineno, _check=None):
        if _check=='paramTypes':
            # Compare param types
            if len(dest) != len(src):
                print(1)
            #     import pdb
            #     pdb.set_trace()
            # retval = None
            # def tryLessArgs(e):
            #     import pdb
            #     pdb.set_trace()
                
            #     nonlocal retval
            #     if len(src) > len(dest):
            #         # Try with less args (one less arg at a time)
            #         restOfArgs = src[:-1]
            #         if len(restOfArgs) > 0:
            #             # Remove old constraint
                        
            #             self.unify(dest, restOfArgs, lineno, _check)
            #             retval = True
            #             return True
            #     return False
            ensure(len(dest) == len(src), lambda: f"Functions have different numbers of arguments: {dest} vs {src}", lineno, exceptionType=PLDiffNumArgsException)#, tryIfFailed=tryLessArgs)
            # if retval is not None:
            #     return
            for p1,p2 in zip(dest,src):
                print("p1:",p1)
                print("p2:",p2)
                # try:
                self.unify(p1, p2, lineno)
                # except AssertionError:
                #     1/0
                #     pass
            return

        print(dest,'bbbbbbbbb')
        print(src, f'ccccccccc {lineno}')
        dest, destType = State.unwrap(dest)
        src, srcType = State.unwrap(src)

        print(dest,'aaaaaaaaaaa',src,'aaa-',destType,srcType)
        l = self.resolveType(dest if not isinstance(destType, TypeVar) else destType)
        r = self.resolveType(src if not isinstance(srcType, TypeVar) else srcType)
        print(l,'aaa-2',r)
        #assert (not isinstance(l, FunctionPrototype) or not isinstance(r, FunctionPrototype)) or id(l) != id(r) # Otherwise an infinite loop seems to happen
        if isinstance(l, TypeVar): # Type variable
            self.constrainTypeVariable(l, r, lineno) # Set l to r with existing type variable l
        elif isinstance(r, TypeVar): # Type variable
            self.constrainTypeVariable(r, l, lineno) # Set r to l with existing type variable r
        elif (destType == Type.Func and srcType == Type.Func) or (isinstance(l, FunctionPrototype) and isinstance(r, FunctionPrototype)): # Unifying two arrows to be the same type (two functions to be of the same function type)
            # Handle the FunctionPrototype itself
            # Unify the argument we gave and the function prototype
            left,right = l, r

            # Check if the actual function we're applying has less args than the one declared within this function appears to have, and if so, change the number of params for the one that appears. For example, "the C combinator" -- `l cardinal = f in (a in (b in f b a));` -- has a function `f` that can appear to take 2 arguments in its usage, but when `cardinal` is invoked, `f` need not take 2 arguments but could instead just take one and return a function (i.e., currying).
            # if len(src) > len(dest):
            #     # Try with less args (one less arg at a time)
            #     restOfArgs = src[:-1]
            #     if len(restOfArgs) > 0:
            #         # Remove old constraint

            #         self.unify(dest, restOfArgs, lineno, _check)
            #         retval = True
            #         return True
            
            #print('left:',left,'right:',right);input('ppppppp')
            while True:
                try:
                    self.unify(left.paramTypes, right.paramTypes, lineno, _check='paramTypes') # Corresponds to `unify(larr->left, rarr->left);` on https://danilafe.com/blog/03_compiler_typechecking/
                    break
                except PLDiffNumArgsException as e:
                    print('-------1',left,right)
                    # import pdb
                    # pdb.set_trace()
                    

                    prevRP = right.paramTypes # i.e., [    T_7_16,     T_6_17]
                    prevRT = right.returnType # i.e., T_8_18
                    right.paramTypes = right.paramTypes[:-1] # i.e., [    T_7_16]
                    
                    newT = self.newTypeVar()
                    self.unify(newT, FunctionPrototype(prevRP[-1:], prevRT), lineno)
                    right.returnType = newT # i.e., instead of returning T_8_18, `right` will return a function from T_6_17 to T_8_18 since it's "waiting for another parameter."        # Almost like bubble pushing from digital logic, but for some sort of modded lambda calculus..
                    
                    print('-------2',left,right)
                    # self.unify(left.paramTypes, right.paramTypes, lineno, _check='paramTypes')
                    # print('-------3',left,right)
                    #break
            self.unify(left.returnType, right.returnType, lineno) # Corresponds to `unify(larr->right, rarr->right);` on the above website
        else:
            # Just check type equality
            ensure(l == r, lambda: f"Types don't match: {dest} ({l})\n    \tand {src} ({r})", lineno, exceptionType=PLTypesDontMatchException) # TODO: better msg etc.

    # End types #

    # def processedRest(self, index):
    #     return len(self.rest) == index + 1

    # def addProcRest(self, procRest):
    #     self.currentProcRest.append(procRest)

    # def newProcRestIndex(self):
    #     return len(self.currentProcRest)
        
    # def setProcRest(self, procRest, index):
    #     # Grow the list to fit
    #     origSize = len(self.currentProcRest)
    #     numIters = 0
    #     for i in range(origSize, index+1):
    #         self.currentProcRest.append(None)
    #         numIters += 1
    #     assert numIters <= 1, f"Adding procRest {procRest} at index {index} that leaves {numIters-1} empty spot(s) in the array of size {origSize}"

    #     assert self.currentProcRest[index] is None, "No overwriting supported for now (just comment out this line to support it)"
    #     self.currentProcRest[index] = procRest
    
    # def procRest(self, ret):
    #     retval = ret
    #     while len(self.currentProcRest) > 0:
    #         ret_ = self.currentProcRest.pop()()
    #         retval.append(ret_)
    #         self.rest.append(retval)
    #     return retval

    # Creates bindings from `idents` (a list of variable names), to `bindings` (a list of `Identifier`s) for blocks' local bindings.
    # Usage: `with state.newBindings(...):`
    def newBindings(self, idents, bindings):
            return self.ContextManager(self, idents, bindings)

    def pushBlock(self):
        self.bindingBlocks.append(self.ContextManager(self, [], []))

    def popBlock(self):
        self.bindingBlocks.pop().__exit__(None, None, None)

    def addBindingsToCurrentBlock(self, idents, bindings):
        cm = self.bindingBlocks[-1]
        i = len(cm.idents)
        cm.idents += idents
        cm.bindings += bindings
        cm.__enter__(i=i)

    # Based on https://docs.python.org/3/library/contextlib.html#contextlib.closing and https://www.geeksforgeeks.org/context-manager-in-python/
    class ContextManager():
            def __init__(self, s, idents, bindings):
                    self.s = s
                    self.idents = idents
                    self.bindings = bindings
                    self.prevValues = []

            def __enter__(self, i=0):
                    assert len(self.idents) == len(self.bindings)
                
                    # Back up the current bindings before overwriting with the new one:
                    for ident, binding in zip(self.idents[i:], self.bindings[i:]):
                            self.prevValues.append(self.s.O.get(ident))
                            self.s.O[ident] = binding
                            print("New binding:", ident, "->", binding)
                            #1/0
                    return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                    assert len(self.idents) == len(self.bindings)
                    
                    # Restore backups or remove newly added bindings that have no backups, in reverse in case of duplicate bindings -- we want the earliest one to be saved to self.s.O
                    for ident, prevValue in reversed(list(zip(self.idents, self.prevValues))):
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
        # if astType == 'functionCall':
        #     import pdb; pdb.set_trace()
        self.lineNumber = lineNumber
        self.type = resolvedType
        self.astType = astType
        self.values = values

    def toString(self, depth):
        return "AAST:\n  \tline " + strWithDepth(self.lineNumber, depth) + "\n  \ttype " + strWithDepth(self.type, depth) +  "\n  \tAST type: " + strWithDepth(self.astType, depth) + "\n  \tvalues: " + strWithDepth(self.values, depth)

    def clone(self):
        return AAST(self.lineNumber, self.type, self.astType, self.values)

class Identifier(AutoRepr):
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    def toString(self, depth):
        return "Identifier:\n  \tname " + strWithDepth(self.name, depth) + "\n  \ttype " + strWithDepth(self.type, depth) + "\n  \tvalue: " + strWithDepth(self.value, depth)

    def clone(self):
        return Identifier(self.name, self.type, self.value.clone() if isinstance(self.value, AAST) else self.value)

# PLooza map
class Map:
    def __init__(self):
        pass

def run_semantic_analyzer(ast, state=None, skipInterpreter=False):
    if state is None:
        state = State()
    
    def first_pass(state):
        i = 0
        j = 0
        ret = []
        def run(x):
            ret.append(proc(state, x))
        for x in ast:
            #pp.pprint(x)
            if isinstance(x, list):
                j = 0
                for y in x:
                    run(y)
                    j += 1
            else:
                run(x)
            i += 1
        return ret, state

    # Perform first pass
    aast, state = first_pass(state)
    print('--AAST after first pass:',aast)
    #import code; code.InteractiveConsole(locals=locals()).interact()
    # Perform dataflow analysis (second pass)
    import dataflow_analysis
    dataflow_analysis.run_dataflow_analysis(aast, state)
    # Perform third pass: tree-walk interpreter to populate maps' contents, while unifying the map keyType and valueType with the type least-upper-bound of the .add x y calls (doing: keyType lub x, valueType lub y)
    if not skipInterpreter:
        import tree_walk_interpreter
        aast, state = tree_walk_interpreter.run_interpreter(aast, state)
    
    return aast, state
