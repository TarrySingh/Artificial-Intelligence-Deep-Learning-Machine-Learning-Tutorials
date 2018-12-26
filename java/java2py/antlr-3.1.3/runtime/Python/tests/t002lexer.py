import antlr3
import testbase
import unittest

class t002lexer(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        
        
    def lexerClass(self, base):
        class TLexer(base):
            def emitErrorMessage(self, msg):
                # report errors to /dev/null
                pass

            def reportError(self, re):
                # no error recovery yet, just crash!
                raise re

        return TLexer
    
        
    def testValid(self):
        stream = antlr3.StringStream('01')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        self.failUnlessEqual(token.type, self.lexerModule.ZERO)

        token = lexer.nextToken()
        self.failUnlessEqual(token.type, self.lexerModule.ONE)

        token = lexer.nextToken()
        self.failUnlessEqual(token.type, self.lexerModule.EOF)
        

    def testMalformedInput(self):
        stream = antlr3.StringStream('2')
        lexer = self.getLexer(stream)

        try:
            token = lexer.nextToken()
            self.fail()

        except antlr3.NoViableAltException, exc:
            self.failUnlessEqual(exc.unexpectedType, '2')
            

if __name__ == '__main__':
    unittest.main()
