import sys
import ply.lex as lex
import ply.yacc as yacc
from lexer import tokens

# Precedence order
precedence = (
    ('right', 'LARROW'),
    ('left', 'NOT'),
    ('nonassoc', 'LE', 'LT', 'EQUALS', 'GT'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('left', 'NOT'),
    ('left', 'DOT'),
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

# block
def p_stmt_block(p):
    'stmt : LBRACE stmtlist RBRACE SEMI'
    p[0] = p[2]

# variable initializer
def p_stmt_init(p):
    'stmt : identifier identifier LET_EQUALS expr'
    p[0] = (p.lineno(1), "stmt_init", p[1], p[2], p[4])

# variable declaration
def p_stmt_decl(p):
    'stmt : identifier identifier' # exprlist can be an identifier, in which case this is a variable declaration; or, it can be exprs.
    print(list(p))
    p[0] = (p.lineno(1), "stmt_decl", p[1], p[2])

# expression statement
def p_stmt_expr(p):
    'stmt : expr'
    p[0] = p[1]

# identifier along with line number
def p_identifier(p):
    'identifier : IDENTIFIER'
    p[0] = (p.lineno(1), "identifier", p[1])

# 0 or more exprs
def p_exprlist_none(p):
    'exprlist : '
    p[0] = []

def p_exprlist_some(p):
    'exprlist : expr exprlist'
    p[0] = [p[1]] + p[2]

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

# formal is a variable with a type

def p_formal(p):
    'formal : identifier'
    p[0] = (p[1], p[3])

# now we have a whole bunch of possible expressions.....

# variable declaration or function call (which one it is will be checked later by the type-checker)
def p_expr_mapAccess(p):
    'expr : expr DOT identifier exprlist' # exprlist: optional args
    p[0] = (p.lineno(2), "mapAccess", p[1], p[2], p[3])

def p_expr_assign(p):
    'expr : identifier LARROW expr'
    p[0] = (p.lineno(2), 'assign', p[1], p[3])

def p_expr_range_exclusive(p):
    'expr : expr ELLIPSIS LT expr'
    p[0] = (p.lineno(2), 'range_exclusive', p[1], p[4])
    
def p_expr_range_inclusive(p):
    'expr : expr ELLIPSIS LE expr'
    p[0] = (p.lineno(2), 'range_inclusive', p[1], p[3])

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
    
def p_expr_lambda(p):
    'expr : identifier IN expr'
    p[0] = (p.lineno(1), 'list_expr', p[2])

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

def p_expr_negate(p):
    'expr : NEGATE expr'
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
    p[0] = (p[1][0], 'expr_identifier', p[1])

def p_expr_integer(p):
    'expr : INTEGER'
    p[0] = (p.lineno(1), 'integer', p[1])

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
    print("ERROR: %d: Parser: unexpected token %s" % (p.lineno, p.value))
    sys.exit(1)

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
    return_token = self.input_tokens[self.i]
    print("Processing:",return_token)
    self.i += 1
    return return_token

parser = yacc.yacc()
def run_parser(tokens):
    parsed = parser.parse(lexer = PA2Lexer(tokens))
    printAst(sys.stdout, parsed)
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
