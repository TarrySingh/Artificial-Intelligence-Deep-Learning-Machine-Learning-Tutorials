import antlr3
import testbase
import unittest

class t017parser(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        
    def parserClass(self, base):
        class TestParser(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self.reportedErrors = []
        

            def emitErrorMessage(self, msg):
                self.reportedErrors.append(msg)
                
        return TestParser


    def testValid(self):
        cStream = antlr3.StringStream("int foo;")
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.program()

        assert len(parser.reportedErrors) == 0, parser.reportedErrors


    def testMalformedInput1(self):
        cStream = antlr3.StringStream('int foo() { 1+2 }')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.program()

        # FIXME: currently strings with formatted errors are collected
        # can't check error locations yet
        assert len(parser.reportedErrors) == 1, parser.reportedErrors


    def testMalformedInput2(self):
        cStream = antlr3.StringStream('int foo() { 1+; 1+2 }')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.program()

        # FIXME: currently strings with formatted errors are collected
        # can't check error locations yet
        assert len(parser.reportedErrors) == 2, parser.reportedErrors


if __name__ == '__main__':
    unittest.main()
