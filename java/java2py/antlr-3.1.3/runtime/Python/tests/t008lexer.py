import antlr3
import testbase
import unittest

class t008lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('ffaf')
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
        assert token.text == 'fa', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 3, token.start
        assert token.stop == 3, token.stop
        assert token.text == 'f', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF


    def testMalformedInput(self):
        stream = antlr3.StringStream('fafb')
        lexer = self.getLexer(stream)

        lexer.nextToken()
        lexer.nextToken()
        try:
            token = lexer.nextToken()
            raise AssertionError, token

        except antlr3.MismatchedTokenException, exc:
            assert exc.unexpectedType == 'b', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 3, repr(exc.charPositionInLine)
            assert exc.line == 1, repr(exc.line)
            

if __name__ == '__main__':
    unittest.main()
