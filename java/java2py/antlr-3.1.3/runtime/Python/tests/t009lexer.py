import antlr3
import testbase
import unittest

class t009lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('085')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        assert token.type == self.lexerModule.DIGIT
        assert token.start == 0, token.start
        assert token.stop == 0, token.stop
        assert token.text == '0', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.DIGIT
        assert token.start == 1, token.start
        assert token.stop == 1, token.stop
        assert token.text == '8', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.DIGIT
        assert token.start == 2, token.start
        assert token.stop == 2, token.stop
        assert token.text == '5', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF


    def testMalformedInput(self):
        stream = antlr3.StringStream('2a')
        lexer = self.getLexer(stream)

        lexer.nextToken()
        try:
            token = lexer.nextToken()
            raise AssertionError, token

        except antlr3.MismatchedRangeException, exc:
            assert exc.a == '0', repr(exc.a)
            assert exc.b == '9', repr(exc.b)
            assert exc.unexpectedType == 'a', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 1, repr(exc.charPositionInLine)
            assert exc.line == 1, repr(exc.line)
            

if __name__ == '__main__':
    unittest.main()


