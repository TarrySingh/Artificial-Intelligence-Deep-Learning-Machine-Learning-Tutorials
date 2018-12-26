import os
import antlr3
import testbase
import unittest

class t019lexer(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid(self):
        inputPath = os.path.splitext(__file__)[0] + '.input'
        stream = antlr3.StringStream(open(inputPath).read())
        lexer = self.getLexer(stream)

        while True:
            token = lexer.nextToken()
            if token.type == antlr3.EOF:
                break

if __name__ == '__main__':
    unittest.main()
