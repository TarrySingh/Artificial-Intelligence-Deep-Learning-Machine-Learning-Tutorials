import unittest
import textwrap
import antlr3
import testbase

class T(testbase.ANTLRTest):

    def testbug(self):
        self.compileGrammar()
        
        cStream = antlr3.StringStream("public fooze")
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)

        parser.r()


if __name__ == '__main__':
    unittest.main()

