import antlr3
import testbase
import unittest


class t029synpredgate(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid1(self):
        stream = antlr3.StringStream('ac')
        lexer = self.getLexer(stream)
        token = lexer.nextToken()


if __name__ == '__main__':
    unittest.main()



