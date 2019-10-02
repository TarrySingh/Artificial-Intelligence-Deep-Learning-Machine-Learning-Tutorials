import antlr3
import testbase
import unittest


class t027eof(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    @testbase.broken("That's not how EOF is supposed to be used", Exception)
    def testValid1(self):
        cStream = antlr3.StringStream(' ')
        lexer = self.getLexer(cStream)
        
        tok = lexer.nextToken()
        assert tok.type == self.lexerModule.SPACE, tok
        
        tok = lexer.nextToken()
        assert tok.type == self.lexerModule.END, tok


if __name__ == '__main__':
    unittest.main()

