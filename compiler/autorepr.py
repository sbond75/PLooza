from pprint import pformat

# https://stackoverflow.com/questions/8234274/how-to-indent-the-contents-of-a-multi-line-string
try:
    import textwrap
    textwrap.indent
except AttributeError:  # undefined function (wasn't added until Python 3.3)
    def indent(text, amount, ch=' '):
        padding = amount * ch
        return ''.join(padding+line for line in text.splitlines(True))
else:
    def indent(text, amount, ch=' '):
        return textwrap.indent(text, amount * ch)


# https://stackoverflow.com/questions/750908/auto-repr-method
indentInc = 4
class AutoRepr(object):
	def __repr__(self, indent_=indentInc, depth=0):
		#items = (' '*indent + "%s = %s\n" % (k, (v.__repr__(indent + indentInc) if isinstance(v, AutoRepr) else v.__repr__())) for k, v in self.__dict__.items())
		#return ' '*indent + "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items)) + "\n"
		return indent(self.toString(depth), indent_)

	def toString(self, depth):
		raise Exception("Subclasses should override toString()")
        
	def linenumber(self):
		raise Exception("Subclasses should override linenumber()")
