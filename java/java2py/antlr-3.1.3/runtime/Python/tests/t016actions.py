import antlr3
import testbase
import unittest

class t016actions(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        
        
    def testValid(self):
        cStream = antlr3.StringStream("int foo;")
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        name = parser.declaration()
        assert name == 'foo', name
            

if __name__ == '__main__':
    unittest.main()
