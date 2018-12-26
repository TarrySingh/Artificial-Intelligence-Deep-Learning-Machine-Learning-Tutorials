import antlr3
import testbase
import unittest

class t007lexer(testbase.ANTLRTest):
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
        stream = antlr3.StringStream('fofababbooabb')
        lexer = self.getLexer(stream)

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 0, token.start
        assert token.stop == 1, token.stop
        assert token.text == 'fo', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.FOO
        assert token.start == 2, token.start
        assert token.stop == 12, token.stop
        assert token.text == 'fababbooabb', token.text

        token = lexer.nextToken()
        assert token.type == self.lexerModule.EOF


    def testMalformedInput(self):
        stream = antlr3.StringStream('foaboao')
        lexer = self.getLexer(stream)

        try:
            token = lexer.nextToken()
            raise AssertionError, token

        except antlr3.EarlyExitException, exc:
            assert exc.unexpectedType == 'o', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 6, repr(exc.charPositionInLine)
            assert exc.line == 1, repr(exc.line)
            

if __name__ == '__main__':
    unittest.main()

