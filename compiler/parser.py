import sys
import ply.lex as lex
import ply.yacc as yacc
from lexer import tokens
from plexception import PLException
from debugOutput import print, input # Replace default `print` and `input` to use them for debugging purposes
import debugOutput

# Precedence order (lowest to highest; higher precedence ones happen before lower ones)
precedence = (
    ('right', 'LARROW'),
    ('left', 'NOT'),
    ('nonassoc', 'LE', 'LT', 'EQUALS', 'GT'), # "It is also possible to specify non-associativity in the precedence table. This would be used when you don't want operations to chain together. For example, suppose you wanted to support comparison operators like < and > but you didn't want to allow combinations like a < b < c." ( https://www.dabeaz.com/ply/ply.html#ply_nn27 )
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'UMINUS'), # Unary minus operator ( https://www.dabeaz.com/ply/ply.html -- at "One problem with the precedence specifier technique is that it is sometimes necessary to change the precedence of an operator in certain contexts. For example, consider a unary-minus operator in "3 + 4 * -5". Mathematically, the unary minus is normally given a very high precedence--being evaluated before the multiply. However, in our precedence specifier, MINUS has a lower precedence than TIMES. To deal with this, precedence rules can be given for so-called "fictitious tokens" like this")
    ('left', 'DOT'),
    ('right', 'FUNCTION_CALL'),
)
 
# program = [class]
def p_program(p):
    'program : stmtlist'
    p[0] = p[1]

def p_stmtlist_none(p):
    'stmtlist : '
    p[0] = []
    
def p_stmtlist_some(p):
    'stmtlist : stmt SEMI stmtlist'
    p[0] = [p[1]] + p[3]

# stmt can be a block, variable initializer, variable declaration, or expression statement.

def p_stmt_block(p):
    'stmt : stmtblock'
    p[0] = p[1]

# block
def p_stmtblock(p):
    'stmtblock : LBRACE stmtlist RBRACE whereclause' # Note: no `SEMI` is needed after the `whereclause` here since if the block is part of a stmtlist that covers semicolons already.
    p[0] = (p[2][0], 'stmt_block', p[2], p[4])

def p_whereclause_none(p):
    'whereclause : '
    p[0] = []

def p_whereclause_some(p):
    'whereclause : WHERE wherelist'
    p[0] = p[1]

def p_wherelist_empty(p):
    'wherelist : '
    # Parsing for anticipated errors ("error productions" (in the grammar)) is the technical term. It is an error to have an empty wherelist:
    errorMsg(p, 'Expected bindings after "where"')

def p_wherelist_1(p):
    'wherelist : wherelist1'
    p[0] = p[1]

def p_wherelist1_only(p):
    'wherelist1 : wherebinding'
    p[0] = [p[1]]

def p_wherelist1_some(p):
    'wherelist1 : wherebinding COMMA wherelist1'
    p[0] = [p[1]] + p[3]

def p_wherebinding(p):
    'wherebinding : identifier IS expr'
    p[0] = (p[1][0], p[1], 'wherebinding', p[3])

# variable initializer
def p_stmt_init(p):
    'stmt : identifier identifier LET_EQUALS expr'
    p[0] = (p[1][0], "stmt_init", p[1], p[2], p[4])

# variable declaration
def p_stmt_decl(p):
    'stmt : identifier identifier' # exprlist can be an identifier, in which case this is a variable declaration; or, it can be exprs.
    #print(list(p))
    p[0] = (p[1][0], "stmt_decl", p[1], p[2])

# expression statement
def p_stmt_expr(p):
    'stmt : expr'
    p[0] = p[1]

# identifier along with line number
def p_identifier(p):
    'identifier : IDENTIFIER'
    p[0] = (p.lineno(1), "identifier", p[1])

def p_exprlist_none(p):
    'exprlist : '
    p[0] = []

def p_exprlist_some(p):
    'exprlist : expr exprlist'
    p[0] = [p[1]] + p[2]

# exprlist1: 1 or more exprs

def p_exprlist1_only(p):
    'exprlist1 : expr'
    p[0] = [p[1]]

def p_exprlist1_some(p):
    'exprlist1 : expr exprlist1'
    p[0] = [p[1]] + p[2]

# # exprlistcomma: 0 or more exprs separated by commas

# def p_exprlistcomma_empty(p):
#     'exprlistcomma : '
#     p[0] = []

# def p_exprlistcomma_1(p):
#     'exprlistcomma : exprlistcomma1'
#     p[0] = p[1]

# # exprlistcomma1: 1 or more exprs separated by commas

# def p_exprlistcomma1_only(p):
#     'exprlistcomma1 : expr'
#     p[0] = [p[1]]

# def p_exprlistcomma1_some(p):
#     'exprlistcomma1 : expr COMMA exprlistcomma1'
#     p[0] = [p[1]] + p[3]
    
# formallist: 0 or more formals

def p_formallist_empty(p):
    'formallist : '
    p[0] = []

def p_formallist_1(p):
    'formallist : formallist1'
    p[0] = p[1]

# formallist1: 1 or more formals

def p_formallist1_only(p):
    'formallist1 : formal'
    p[0] = [p[1]]

def p_formallist1_some(p):
    'formallist1 : formal COMMA formallist1'
    p[0] = [p[1]] + p[3]

# formal is a variable with a type, or without one:

# def p_formal_withType(p):
#     'formal : identifier identifier'
#     p[0] = (p[1], p[2])

def p_formal_noType(p):
    'formal : identifier'
    p[0] = (p[1][0], 'formal_identifier', p[1])

# now we have a whole bunch of possible expressions.....

# variable declaration or function call (which one it is will be checked later by the type-checker)
def p_expr_mapAccess_ident(p):
    'expr : expr DOT identifier' # exprlist: optional args
    p[0] = (p.lineno(2), "mapAccess", p[1], p[3])
def p_expr_mapAccess_escapedident(p):
    'expr : expr DOT ESCAPE identifier' # exprlist: optional args
    p[0] = (p.lineno(2), "mapAccess", p[1], p[4])
def p_expr_mapAccess_num(p):
    'expr : expr DOT INTEGER' # exprlist: optional args
    p[0] = (p.lineno(2), "mapAccess", p[1], p[3])

def p_expr_exprFnCall(p):
    'expr : exprFnCall'
    p[0] = p[1]
def p_expr_functionCall(p):
    'exprFnCall : expr exprlist1 whereclause %prec FUNCTION_CALL'
    p[0] = (p[1][0], "functionCall", p[1], p[2], p[3])
    
def p_expr_assign(p):
    'expr : identifier LARROW expr'
    p[0] = (p.lineno(2), 'assign', p[1], p[3])

def p_expr_range_exclusive(p):
    'expr : expr ELLIPSIS LT expr' # use parens to make function calls since we use expr instead of expr here to support spaces between list elements.
    p[0] = (p.lineno(2), 'range_exclusive', p[1], p[4])
    
def p_expr_range_inclusive(p):
    'expr : expr ELLIPSIS LE expr'
    p[0] = (p.lineno(2), 'range_inclusive', p[1], p[4])

def p_expr_escaped(p):
    'expr : ESCAPE expr'
    p[0] = (p.lineno(2), 'escaped', p[2])
def p_expr_old(p):
    'expr : OLD expr'
    p[0] = (p.lineno(2), 'old', p[2])

def p_expr_import(p):
    'expr : IMPORT expr'
    p[0] = (p.lineno(2), 'import', p[2])

def p_expr_range_gt(p):
    'expr : GT expr'
    p[0] = (p.lineno(2), 'range_gt', p[2])
def p_expr_range_le(p):
    'expr : LE expr'
    p[0] = (p.lineno(2), 'range_le', p[2])
def p_expr_range_lt(p):
    'expr : LT expr'
    p[0] = (p.lineno(2), 'range_lt', p[2])
def p_expr_range_ge(p):
    'expr : GE expr'
    p[0] = (p.lineno(2), 'range_ge', p[2])

def p_expr_list(p):
    'expr : LBRACKET exprlist RBRACKET'
    p[0] = (p.lineno(1), 'list_expr', p[2])

def p_expr_lambda_stmt(p):
    'expr : formallist IN stmtblock'
    p[0] = (p[1][0], 'lambda', p[1], p[3])

def p_expr_lambda_expr(p):
    'expr : formallist IN expr'
    p[0] = (p[1][0], 'lambda', p[1], p[3])

def p_expr_brace(p): # .add{this stuff here}
    'expr : LBRACE exprlist RBRACE'
    p[0] = (p.lineno(1), 'brace_expr', p[2])

def p_expr_new(p):
    'expr : NEW identifier'
    p[0] = (p.lineno(1), 'new', p[2])

def p_expr_plus(p):
    'expr : expr PLUS expr'
    p[0] = (p.lineno(2), 'plus', p[1], p[3])

def p_expr_times(p):
    'expr : expr TIMES expr'
    p[0] = (p.lineno(2), 'times', p[1], p[3])

def p_expr_minus(p):
    'expr : expr MINUS expr'
    p[0] = (p.lineno(2), 'minus', p[1], p[3])

def p_expr_divide(p):
    'expr : expr DIVIDE expr'
    p[0] = (p.lineno(2), 'divide', p[1], p[3])

# Number negation
def p_expr_negate(p):
    'expr : MINUS expr %prec UMINUS' # `%prec UMINUS`: https://www.dabeaz.com/ply/ply.html
    p[0] = (p.lineno(1), 'negate', p[2])

def p_expr_lt(p):
    'expr : expr LT expr'
    p[0] = (p.lineno(2), 'lt', p[1], p[3])

def p_expr_le(p):
    'expr : expr LE expr'
    p[0] = (p.lineno(2), 'le', p[1], p[3])

def p_expr_eq(p):
    'expr : expr EQUALS expr'
    p[0] = (p.lineno(2), 'eq', p[1], p[3])

def p_expr_not(p):
    'expr : NOT expr'
    p[0] = (p.lineno(1), 'not', p[2])

def p_expr_paren(p):
    'expr : LPAREN expr RPAREN'
    p[0] = p[2]

def p_expr_identifier(p):
    'expr : identifier'
    lineno = p[1][0] # weird stuff... p.lineno(1) returns 0 here instead so we use this..
    p[0] = (lineno, 'expr_identifier', p[1])

def p_expr_integer(p):
    'expr : INTEGER'
    p[0] = (p.lineno(1), 'integer', p[1])

def p_expr_float(p):
    'expr : FLOAT'
    p[0] = (p.lineno(1), 'float', p[1])

def p_expr_string(p):
    'expr : STRING'
    p[0] = (p.lineno(1), 'string', p[1])

def p_expr_true(p):
    'expr : TRUE'
    p[0] = (p.lineno(1), 'true')

def p_expr_false(p):
    'expr : FALSE'
    p[0] = (p.lineno(1), 'false')

# rule for syntax errors
def p_error(p):
    if p is None:
        # https://stackoverflow.com/questions/8220648/eof-error-in-parser-yacc
        raise PLException("ERROR: EOF: Parser: %s" % "Parse error in input. EOF")
    raise PLException("ERROR: %d: Parser: unexpected token %s" % (p.lineno, p.value))

def errorMsg(p, msg):
    raise PLException("ERROR: %d: Parser: %s" % (p.lineno, msg))

# read input lines as a list
# with open(sys.argv[1], 'r') as f:
#     global input_lines
#     input_lines = f.read().splitlines()

# class to generate a stream of tokens
class PA2Lexer():
  def __init__(self, input_tokens):
      self.input_tokens = input_tokens
      self.i = 0
      super().__init__()
  def token(self):
    if self.i >= len(self.input_tokens):
        return None # Finished parsing the entire file
    return_token = self.input_tokens[self.i]
    print("Processing:",return_token)
    self.i += 1
    return return_token

parser = yacc.yacc()
def run_parser(tokens):
    if debugOutput.debugOutput:
        import logging
        log = logging.getLogger()
        parsed = parser.parse(lexer = PA2Lexer(tokens),debug=log)
    else:
        parsed = parser.parse(lexer = PA2Lexer(tokens))
    #printAst(sys.stdout, parsed)
    return parsed
    

# traverse the abstract syntax tree, printing to an output file f along the way
def printAst(f,ast):
    print(ast)
    if type(ast) == type(""): # string
        f.write(ast + " ")
    elif type(ast) == type(0): # int
        f.write(str(ast) + " ")
    else: # tuple or list
        if type(ast) == type([]): # lists also print out their length
            f.write(str(len(ast)) + " ")
        for x in ast:
            printAst(f,x)
    f.write("\n")


# with open(sys.argv[1][:-4] + "-ast", 'w') as f: # replace -lex with -ast and printAst to the file
#     printAst(f, parsed)
