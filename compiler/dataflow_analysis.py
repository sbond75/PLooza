# This file contains functionality to determine something ("dataflow facts") about every possible path of execution in the PLooza program.
# Notably, we want to determine the type of a PLMap at every point, to see if it is possible to insert an IO t, where t is some type, into the map, and where IO is an IO effect (i.e., this type was returned from readi() which reads an integer from stdin (therefore it does IO). So if there is a possible path of execution which does IO, then the map's type at that point is a runtime map instead of a compile-time map, and all other uses of the map will still be compile-time and will contain its contents excluding those in the runtime part. (Compile-time addition of things is already handled in semantics.py by adding things to the map as the source code is read, but only for the `map.add` calls that are not embedded in a function.)

# Constructs a control-flow graph (CFG) out of the AST given.
# This splits the code up into basic blocks based on "branch targets" -- in PLooza these would be map lookups and function calls only.
def buildCFG(aast, state):
    pass

def run_dataflow_analysis(aast, state):
    cfg = buildCFG(aast, state)
