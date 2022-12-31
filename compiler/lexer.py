# Based on https://www.dabeaz.com/ply/ply.html

import sys
import ply.lex as lex
import ply.yacc as yacc

states = (
  ('comment', 'exclusive'), # state for multiline comment
)

# List of token names.
tokens = (
    'COMMA',
    'DIVIDE',
    'DOT',
    'LET_EQUALS',
    'EQUALS',
    'FALSE',
    'IDENTIFIER',
    'IN',
    'INTEGER',
    'FLOAT',
    'LARROW',
    'LBRACE',
    'LBRACKET',
    'LE',
    'LPAREN',
    'LT',
    'MINUS',
    'NEW',
    'NOT',
    'PLUS',
    'RBRACE',
    'RPAREN',
    'RBRACKET',
    'SEMI',
    'STRING',
    'TIMES',
    'TRUE',
    'COMMENT',
    'END_OF_LINE_COMMENT',
    'comment_START',
    'comment_END',
    'comment_INSIDE',

    'ELLIPSIS',
    'GT', # Greater than
    'GE', # Greater than or equal to
    'ESCAPE',
    'WHERE',
    'IS',
    'OLD',
)

# Regular expression rules for simple tokens
t_COMMA = r','
t_DIVIDE = r'/'
t_DOT = r'\.'
t_LET_EQUALS = r'='
t_EQUALS = r'=='
t_FALSE = r'false'
t_IN = r'in'
t_LARROW = r'<-'
t_LBRACE = r'{'
t_LBRACKET = r'\['
t_LE = r'<='
t_LPAREN = r'\('
t_LT = r'<'
t_MINUS = r'-'
t_NEW = r'new'
t_NOT = r'!'
t_PLUS = r'\+'
t_RBRACE = r'}'
t_RPAREN = r'\)'
t_RBRACKET = r'\]'
t_SEMI = r';'
t_TIMES = r'\*'
t_TRUE = r'true'
t_ELLIPSIS = r'\.\.'
t_GT = r'>'
t_GE = r'>='
t_ESCAPE = r'\\'
t_WHERE = r'where'
t_IS = r'is'
t_OLD = r'old'

# keywords that are completely case insensitive
caseInsensitiveKeywords = []

# keywords that must begin with lowercase but are otherwise completely case insensitive
caseSemiSensitiveKeywords = []

keywords = ['false', 'true', 'in', 'new', 'where', 'is', 'old'] + caseInsensitiveKeywords + caseSemiSensitiveKeywords

def t_IDENTIFIER(t):
    r'[_A-Za-z]([0-9]|[_A-Za-z])*'
    # might want to override the identifier as a keyword
    if t.value.lower() in (caseInsensitiveKeywords + caseSemiSensitiveKeywords):
        t.type = t.value.upper()
    if t.value in keywords:
        t.type = t.value.upper()
    return t

# Note: order matters; t_FLOAT must come before t_INTEGER or else floats are lexed as these tokens: `INTEGER DOT INTEGER identifier` (`identifier` because of the `f` at the end) which is incorrect.
def t_FLOAT(t): # nonnegative
    r'[0-9]+\.[0-9]*f'
    return t

def t_INTEGER(t): # nonnegative
    r'[0-9]+'
    t.value = int(t.value)
    if t.value > 2147483647:
        reportError("integer literal too large (> 2147483647) '%d'" % t.value, t.lexer.lineno)
    t.value = str(t.value) # the int -> str -> int conversion is to remove trailing 0s and to make the if statement above easier
    return t

def t_STRING(t):
    r'"([^"\n\\\x00]|\\[^\n\x00])*"' # quote followed by repeatedly: either a single "normal" character, or a backslash followed by an escaped character
    t.value = t.value[1:-1] # chop off quotes
    if len(t.value) > 1024:
        reportError("string literal is too long (%d > 1024)" % len(t.value), t.lexer.lineno)
    return t

def t_COMMENT(t): # begin multiline comment
    r'\/\*'
    t.lexer.push_state('comment')
    
def t_comment_START(t): # go another layer deeper in multiline comment
    r'\/\*'
    t.lexer.push_state('comment')

def t_comment_END(t): # end a multiline comment
    r'\*\/'
    t.lexer.pop_state()

def t_comment_INSIDE(t): # characters to ignore while in multiline comment, incl. any (s or *s that the above rules didn't catch
    r'[^\r\n]'

# Define a rule so we can track line numbers
# applies to both normal and comment states
def t_ANY_newline(t):
    r'((\r\n?)|\n)'
    t.lexer.lineno += 1
 
# A string containing ignored characters (spaces, tabs, etc)
t_ignore  = ' \f\r\t\v'
# in comments we "look at" every character
t_comment_ignore = ''

# End of line comment
def t_END_OF_LINE_COMMENT(t):
    r'//[^\r\n]*'

def reportError(msg, line):
    print("ERROR: %d: Lexer: %s" %(line, msg))
    sys.exit(1)

# Error handling rules
def t_error(t):
    reportError("invalid character '%s'" % t.value[0], t.lexer.lineno)

def t_comment_error(t):
    reportError("invalid character in comment '%s'" % t.value[0], t.lexer.lineno) # TODO

def t_comment_eof(t):
    reportError("unexpected eof during multiline comment", t.lexer.lineno)

# Tokenize and output to file
def makeSerializedTokens():
    with open(sys.argv[1] + '-lex', 'w') as f:
        while True:
            tok = lexer.token()

            if not tok:
                break      # No more input
            f.write(str(tok.lineno) + '\n')
            f.write(tok.type.lower() + '\n')
            if tok.type.lower() in ['identifier', 'type', 'string', 'integer']: # these are the only types we want to give extra info about
                f.write(tok.value + '\n')

# Tokenize and process
def run_lexer(proc):
    # Build the lexer
    lexer = lex.lex()

    # Give the lexer some input
    with open(sys.argv[1], 'r') as f:
        data = f.read()
        lexer.input(data)

    f = sys.stdout
    while True:
        tok = lexer.token()

        if not tok:
            break      # No more input
        f.write(str(tok.lineno) + ' ')
        f.write(tok.type.lower() + ' ')
        if tok.type.lower() in ['identifier', 'type', 'string', 'integer']: # these are the only types we want to give extra info about
            f.write(tok.value + '\n')
        else:
            f.write('\n')
        proc(tok)

