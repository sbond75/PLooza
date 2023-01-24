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

        self.onLineEntered(io.StringIO(line))

        return None,None,None
        
    # def precmd(self, line):
    #     print('precmd({})'.format(line))
    #     return cmd.Cmd.precmd(self, line)

    # def postcmd(self, stop, line):
    #     print('postcmd({}, {})'.format(stop, line))
    #     return cmd.Cmd.postcmd(self, stop, line)

def run(onLineEntered):
    REPL(onLineEntered).cmdloop()
