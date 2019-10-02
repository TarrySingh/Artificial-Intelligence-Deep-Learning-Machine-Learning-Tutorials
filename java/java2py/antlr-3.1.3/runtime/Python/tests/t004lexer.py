import antlr3
import testbase
import unittest

class t004lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('ffofoofooo')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 0, token.start
        assert token.stop == 0, token.stop
        assert token.text == 'f', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 1, token.start
        assert token.stop == 2, token.stop
        assert token.text == 'fo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 3, token.start
        assert token.stop == 5, token.stop
        assert token.text == 'foo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 6, token.start
        assert token.stop == 9, token.stop
        assert token.text == 'fooo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF
        

    def testMalformedInput(self):
        stream = antlr3.StringStream('2')
        lexer = self.getLexer(stream)

        try:
            token = lexer.nextToken()
            self.fail()

        except antlr3.MismatchedTokenException, exc:
            self.failUnlessEqual(exc.expecting, 'f')
            self.failUnlessEqual(exc.unexpectedType, '2')
            

if __name__ == '__main__':
    unittest.main()

