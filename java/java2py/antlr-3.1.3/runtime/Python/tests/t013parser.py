import antlr3
import testbase
import unittest

class t013parser(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        
        
    def testValid(self):
        cStream = antlr3.StringStream('foobar')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.document()

        assert len(parser.reportedErrors) == 0, parser.reportedErrors
        assert parser.identifiers == ['foobar']


    def testMalformedInput1(self):
        cStream = antlr3.StringStream('')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)

        parser.document()

        # FIXME: currently strings with formatted errors are collected
        # can't check error locations yet
        assert len(parser.reportedErrors) == 1, parser.reportedErrors
            

if __name__ == '__main__':
    unittest.main()
