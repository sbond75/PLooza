import semantics
from semantics import pp, Type, FunctionPrototype, Identifier, State, TypeVar, ensure, astSemanticDescription
from intervaltree import Interval, IntervalTree
from autorepr import AutoRepr

def passthru(state, ast):
    return ast

def unwrap(state, ast):
    return ast.values

# Represents the result of executing something (interpreting it as a program all the way through)
class Executed(AutoRepr):
    def __init__(self, type, value=None):
        self.type = type
        self.value = value
    
    def unwrapAll(self):
        val = self.value
        while isinstance(val, Executed):
            val = val.value
        return val
        
    def toString(self):
        return "Executed:  \ttype " + str(self.type) + (("  \tvalue " + str(self.value)) if self.value is not None else '') + '\n'

def proc(state, aast):
    pp.pprint(("proc 2nd pass: about to proc:", aast))

    if isinstance(aast, list):
        ret = []
        for x in aast:
            ret.append(procMap[x.astType](state, x))
    else:
        ret = procMap[aast.astType](state, aast)
    
    pp.pprint(("proc 2nd pass:", aast, "->", ret))
    return ret


def stmtBlock(state, ast):
    return passthru(state, ast)

def stmtInit(state, ast):
    return passthru(state, ast)

def stmtDecl(state, ast):
    return passthru(state, ast)

def identifier(state, ast):
    assert len(ast.values) == 1
    temp = ast.values[0]
    if isinstance(temp, Identifier):
        temp2 = temp.value
        return temp2
    return temp

def mapAccess(state, ast):
    return functionCall(state, ast, mapAccess=True)

def functionCall(state, ast, mapAccess=False):
    print('pppppppp',ast)
    fnname_ = ast.values[0]
    fnname = proc(state, fnname_)
    if isinstance(fnname, (list,tuple)):
        assert len(fnname)==1
        fnname = fnname[0]
    #print(ast.values[1]); input()
    fnargs = proc(state, ast.values[1])
    if mapAccess:
        assert mapAccess == (fnname_.type == Type.Map)
        # Grab the value from the map
        print('oooooooooo',fnname_)
        plmap = fnname_.value if isinstance(fnname_, Identifier) else fnname_.values[0].value
        print('oo11111111111111',plmap)
        #assert hasattr(fnargs, 'values') and len(fnargs.values) == 1, f"{fnargs}"
        #key = fnargs.values[0]
        key = fnargs
        def onNotFoundError():
            assert False, f"Key not found but should be there: {key}"
        value = plmap.get(key, onNotFoundError=onNotFoundError)
        return value
    else:
        assert isinstance(fnname, FunctionPrototype), f"{fnname} ; {ast.values[0]}"
        print(ast,'0000000000000000')
        receiver = ast.values[0]
        receiverIsPLMap = isinstance(receiver.type, FunctionPrototype) and receiver.type.receiver == '$self'
        if receiverIsPLMap:
            receiverPLMap = receiver.values[0].values[0].value
        else:
            # It must be a regular function
            receiverPLMap = None
        
        # Evaluate function body with the args put in
        def evalMapMap(): # $map.map: Must take in a lambda from T1 to T2, returns a Map<KeyT, T2>.
            assert len(fnargs) == 1
            theLambda = fnargs[0]
            fnProto = theLambda.values
            # Put the args in
            with state.newBindings(*fnProto.paramBindings):
                def evalBody(arg):
                    for name,ident in zip(*fnProto.paramBindings):
                        assert ident.value is None

                        # Give it a value of the argument we put in
                        ident.value = arg
                        
                    # Eval body
                    retval = proc(state, fnProto.body)
                    
                    for name,ident in zip(*fnProto.paramBindings):
                        assert ident.value is not None

                        # Reset ident.value
                        ident.value = None
                    return retval
                
                # Call the lambda for each item in the map and accumulate their return values
                retvals = []
                for arg_ in receiverPLMap.items():
                    if isinstance(arg_, Interval):
                        for arg in range(arg_.data[0], arg_.data[1]+1):
                            retvals.append(evalBody(arg)) # Calls the lambda
                    else: # dictionary item
                        key,value = arg_
                        retvals.append(evalBody(value)) # Calls the lambda
                # Unify retvals' types, and check if it is only one type or something
                # print(retvals)
                # input('a')
                return Executed(Type.Map, semantics.PLMap(receiverPLMap.prototype
                                                          , {k: v for k, v in enumerate(retvals)} # list to dict with indices of elements in the list as the keys      ( https://stackoverflow.com/questions/36459969/how-to-convert-a-list-to-a-dictionary-with-indexes-as-values )
                                                          , receiverPLMap.keyType, receiverPLMap.valueType
                                                ))
        def evalMapAdd(): # Take in a keyType and valueType, and put it in the map. We take the lub of keyType and valueType each time we do this $map.add method call.
            assert len(fnargs) == 2
            #key = proc(state, fnargs[0])
            #value = proc(state, fnargs[1])
            key = fnargs[0]
            value = fnargs[1]
            
            # import code
            # code.InteractiveConsole(locals=locals()).interact()

            # Set it in the map
            #receiverPLMap.contents[key.values[0].value] = value.values[0].value
            receiverPLMap.contents[key] = value

            # Return void
            return Executed(Type.Void)
        def evalIOPrint(): # Takes any type, returns void
            assert len(fnargs) == 1
            #value = proc(state, fnargs[0])
            value = fnargs[0]

            # # We don't evaluate it since it is IO.
            # return ast

            print('evalIOPrint:', value)
            return Type.Void
        evalMap = {'$map.map': evalMapMap
                   , '$map.add': evalMapAdd
                   , '$io.print': evalIOPrint}
        if isinstance(fnname.body, str): # Built-in method/function
            fn = evalMap.get(fnname.body)
            if fn is None:
                #print(fnname.body,'kkkkkkk')
                1/0
                return
            # print(fn(),'123123', fnname.body)
            return fn()
        else:
            #fnname.receiver...
            # print(fnname,'-----------',ast.lineNumber)

            fnProto = fnname
            
            # Evaluate function body
            # Put the args in
            with state.newBindings(*fnProto.paramBindings):
                def evalBody(args):
                    for name,ident,arg in zip(*fnProto.paramBindings,args):
                        #assert ident.value is not None, f"{ident}"

                        # Give it a value of the argument we put in
                        ident.value = arg
                        
                    # Eval body
                    retval = proc(state, fnProto.body)
                    
                    for name,ident,arg in zip(*fnProto.paramBindings,args):
                        #assert ident.value is None, f"{ident}"

                        # Reset ident.value
                        ident.value = arg
                    return retval

            # Call the lambda
            print(fnargs,'===============')
            retval = evalBody(fnargs) # Calls the lambda
            
            # Unify retvals' types, and check if it is only one type or something
            # print(retvals)
            # input('a')
            return Executed(fnProto.returnType, retval)
        

def assign(state, ast):
    pass

# Helper function
def rangeProc(state, ast):
    assert False, 'Should never be encountered in the second pass'

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
    return unwrap(state, ast)

def lambda_(state, ast):
    # Abstraction from lambda calculus interpreter, shouldn't be too bad
    return passthru(state, ast)

def braceExpr(state, ast):
    pass

def new(state, ast):
    pass

def arith(state, ast):
    print(ast,'999999999999999')
    # pp.pprint((state.typeConstraints,'9999999999999992222222222'))
    # pp.pprint((state.resolveType(state.typeConstraints[ast.type.name]),'9999999999999992222222222333333'))
    t1 = State.unwrap(ast.values[0], assumeFunctionCall=True)[1]
    t2 = State.unwrap(ast.values[1], assumeFunctionCall=True)[1]
    print(t1)
    print(t2)
    def input(x): print(x)
    input('ooooooooooooo')
    e1 = t1
    e2 = t2
    # e1 = state.typeConstraints.get(t1.name) if isinstance(t1, TypeVar) else t1
    # e2 = state.typeConstraints.get(t2.name) if isinstance(t2, TypeVar) else t2
    # if e1 is None:
    #     e1 = state.newTypeVar()
    # if e2 is None:
    #     e2 = state.newTypeVar()
    if not isinstance(e1, set):
        e1 = set([e1])
    if not isinstance(e2, set):
        e2 = set([e2])
    print(e1)
    print(e2)
    input('ppppppppppppppppp333333333333333333')
    import code
    code.InteractiveConsole(locals=locals()).interact()
    
    # https://towardsdatascience.com/python-tricks-flattening-lists-75aeb1102337
    def unwrap(l):
        from collections import Iterable
        
        flat_list = []
        for item in l:
            if isinstance(item, Iterable):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list
    
    e1 = set(unwrap(map(lambda x: state.resolveType(x), e1)))
    e2 = set(unwrap(map(lambda x: state.resolveType(x), e2)))
    print(e1)
    print(e2)
    input('ppppppppppppppppp2222222222222222')
    # for x,y in zip(e1, e2):
    #     state.unify(x, y, ast.lineNumber)
    print(e1)
    print(e2)
    input('ppppppppppppppppp')
    pp.pprint(state.typeConstraints)
    ensure(e1.issubset({Type.Int, Type.Float}), lambda: f"First operand of {astSemanticDescription(ast)} must be an integer, float, or function returning an integer or float", ast.lineNumber)
    ensure(e2.issubset({Type.Int, Type.Float}), lambda: f"Second operand of {astSemanticDescription(ast)} must be an integer, float, or function returning an integer or float", ast.lineNumber)

           
    # if e1.type == Type.Float or e2.type == Type.Float:
    #     t3 = Type.Float # Coerce any remaining ints into floats
    # else:
    #     t3 = Type.Int
    
    # pp.pprint(State.unwrap(ast)
    #return passthru(state, ast)

    # e1 and e2 are assumed to be function calls as well if they are function identifiers.
    a1 = ast.values[0]
    a2 = ast.values[1]
    e1 = proc(state, a1)
    e2 = proc(state, a2)
    print("a1:", a1, "a2:", a2, "e1:", e1, "e2:", e2)
    # # def makeFnCall(e, a):
    # #     return proc(state, semantics.AAST(lineNumber=a.lineNumber, resolvedType=e.returnType, astType='functionCall', values=(e,[])))
    # # eNew = makeFnCall(e1, a1), makeFnCall(e2, a2)
    # print("eNew:", eNew)
    # return eNew[0] + eNew[1]
    return e1.unwrapAll() + e2.unwrapAll()

def plus(state, ast):
    return arith(state, ast)

def times(state, ast):
    return arith(state, ast)

def minus(state, ast):
    return arith(state, ast)

def divide(state, ast):
    return arith(state, ast)

def negate(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=e.type, astType=ast.type, values=ast.args[0])

def lt(state, ast):
    pass

def le(state, ast):
    pass

def eq(state, ast):
    pass

def not_(state, ast):
    return AAST(lineNumber=ast.lineno, resolvedType=e.type, astType=ast.type, values=ast.args[0])

def exprIdentifier(state, ast):
    return name

def integer(state, ast):
    return Executed(ast.type, ast.values)

def float(state, ast):
    return Executed(ast.type, ast.values)

def string(state, ast):
    return Executed(ast.type, ast.values)

def true(state, ast):
    return Executed(ast.type, ast.values)

def false(state, ast):
    return Executed(ast.type, ast.values)


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

# At this point, all identifiers are resolved, so we don't need to track those. We need to fully evaluate some things if possible, like map inserts, and to determine which maps are compile-time or run-time. We also need to propagate more type info.
def second_pass(aast, state):
    ret = []
    for x in aast:
        print("PROC")
        input()

        ret.append(proc(state, x))
    return ret, state
