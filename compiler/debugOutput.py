import builtins

import pprint
class PPWrapper(pprint.PrettyPrinter):
    def pprint(self, object):
        if debugOutput:
            return super().pprint(object)
pp = PPWrapper(indent=4)

global debugOutput
debugOutput = False
global debugErr
debugErr = False
def handleErr(e):
    assert debugErr
    try:
        e()
        assert False # Not expected to run
    except:
        # This is expected to run, since `e` throws the error message exception:
        import traceback
        traceback.print_exc()
        import pdb
        pdb.set_trace()
    return False

def print(*args, **kwargs):
    if debugOutput:
        return builtins.print(*args, **kwargs)

def input(*args, **kwargs):
    if debugOutput:
        return builtins.input(*args, **kwargs)

