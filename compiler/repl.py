# Based on example on https://docs.python.org/3/library/cmd.html and http://pymotw.com/3/cmd/

import cmd
import io

class REPL(cmd.Cmd):
    intro = 'Welcome to the PLooza REPL. Type help or ? to list commands.\n'
    prompt = '> '

    def __init__(self, onLineEntered):
        self.onLineEntered = onLineEntered
        super().__init__()

    def parseline(self, line):
        # print('parseline({!r}) =>'.format(line), end='')
        # ret = cmd.Cmd.parseline(self, line)
        # print(ret)
        # return ret

        if line == 'EOF': # Ends program if Ctrl-D is pressed. But if you type "EOF" only in the REPL it will do this too which is weird.
            print()
            return 'EOF', '\n', 'EOF'

        # Auto-add semicolon if needed
        # if not line.rstrip().endswith(';') and not len(line.strip()) == 0:
        #     line = line + ';'
            
        self.onLineEntered(line)

        return None,None,None
    
    def do_EOF(self, line):
        print("Exit")
        return True
    
    # def precmd(self, line):
    #     print('precmd({})'.format(line))
    #     return cmd.Cmd.precmd(self, line)

    # def postcmd(self, stop, line):
    #     print('postcmd({}, {})'.format(stop, line))
    #     return cmd.Cmd.postcmd(self, stop, line)

def run(onLineEntered):
    REPL(onLineEntered).cmdloop()
