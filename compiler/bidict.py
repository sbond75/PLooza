# https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table :
# {"
# Here is a class for a bidirectional dict, inspired by Finding key from value in Python dictionary [ https://stackoverflow.com/questions/7657457/finding-key-from-value-in-python-dictionar ] and modified to allow the following 2) and 3).
# 
# Note that :
# 
# 1. The inverse directory bd.inverse auto-updates itself when the standard dict bd is modified.
# 2. The inverse directory bd.inverse[value] is always a list of key such that bd[key] == value.
# 3. Unlike the bidict module from https://pypi.python.org/pypi/bidict, here we can have 2 keys having same value, this is very important.
# "}
class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
