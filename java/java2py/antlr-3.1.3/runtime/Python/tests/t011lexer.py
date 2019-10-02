import antlr3
import testbase
import unittest

class t011lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('foobar _Ab98 \n A12sdf')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        assert token.type == self.lexerModule.IDENTIFIER
        assert token.start == 0, token.start
        assert token.stop == 5, token.stop
        assert token.text == 'foobar', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.WS
        assert token.start == 6, token.start
        assert token.stop == 6, token.stop
        assert token.text == ' ', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.IDENTIFIER
        assert token.start == 7, token.start
        assert token.stop == 11, token.stop
        assert token.text == '_Ab98', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.WS
        assert token.start == 12, token.start
        assert token.stop == 14, token.stop
        assert token.text == ' \n ', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.IDENTIFIER
        assert token.start == 15, token.start
        assert token.stop == 20, token.stop
        assert token.text == 'A12sdf', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF


    def testMalformedInput(self):
        stream = antlr3.StringStream('a-b')
        lexer = self.getLexer(stream)

        lexer.nextToken()
        try:
            token = lexer.nextToken()
            raise AssertionError, token

        except antlr3.NoViableAltException, exc:
            assert exc.unexpectedType == '-', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 1, repr(exc.charPositionInLine)
            assert exc.line == 1, repr(exc.line)

            

if __name__ == '__main__':
    unittest.main()
