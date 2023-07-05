# From https://poe.com/

import ply.lex as lex
import ply.yacc as yacc

# Define the token names
tokens = ['ID', 'NUMBER']

# Define the regular expressions for the tokens
t_ID = r'[a-zA-Z_][a-zA-Z_0-9]*'
t_NUMBER = r'\d+'

# Define the ignored characters
t_ignore = ' \t\n'

# Define a function to handle errors
def p_error(p):
    print("Syntax error in input!")

# Define the grammar rules
def p_expression_call(p):
    'expression : expression expression %prec juxtaposition'
    p[0] = p[1] + [p[2]]

def p_expression_id(p):
    'expression : ID'
    p[0] = [p[1]]

def p_expression_number(p):
    'expression : NUMBER'
    p[0] = [p[1]]

# Define the precedence and associativity of the juxtaposition operator
precedence = (
    ('left', 'juxtaposition'),
)

# Build the lexer
lexer = lex.lex()

# Build the parser
parser = yacc.yacc()

# Test the parser
result = parser.parse('f 1 2 3')
print(result)
