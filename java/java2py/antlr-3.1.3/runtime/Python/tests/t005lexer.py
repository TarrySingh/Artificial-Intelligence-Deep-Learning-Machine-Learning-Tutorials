import antlr3
import testbase
import unittest

class t005lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('fofoofooo')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 0, token.start
        assert token.stop == 1, token.stop
        assert token.text == 'fo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 2, token.start
        assert token.stop == 4, token.stop
        assert token.text == 'foo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 5, token.start
        assert token.stop == 8, token.stop
        assert token.text == 'fooo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF
        

    def testMalformedInput1(self):
        stream = antlr3.StringStream('2')
        lexer = self.getLexer(stream)

        try:
            token = lexer.nextToken()
            raise AssertionError

        except antlr3.MismatchedTokenException, exc:
            assert exc.expecting == 'f', repr(exc.expecting)
            assert exc.unexpectedType == '2', repr(exc.unexpectedType)


    def testMalformedInput2(self):
        stream = antlr3.StringStream('f')
        lexer = self.getLexer(stream)

        try:
            token = lexer.nextToken()
            raise AssertionError

        except antlr3.EarlyExitException, exc:
            assert exc.unexpectedType == antlr3.EOF, repr(exc.unexpectedType)
            

if __name__ == '__main__':
    unittest.main()
