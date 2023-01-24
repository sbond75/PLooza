import builtins

import pprint
class PPWrapper(pprint.PrettyPrinter):
    def pprint(self, object):
        if debugOutput:
            return super().pprint(object)
pp = PPWrapper(indent=4)

global debugOutput
debugOutput = False

def print(*args, **kwargs):
    if debugOutput:
        return builtins.print(*args, **kwargs)

def input(*args, **kwargs):
    if debugOutput:
        return builtins.input(*args, **kwargs)

