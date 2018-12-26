import antlr3
import testbase
import unittest


class t024finally(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid1(self):
        cStream = antlr3.StringStream('foobar')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.prog()

        assert events == ['catch', 'finally'], events


if __name__ == '__main__':
    unittest.main()

