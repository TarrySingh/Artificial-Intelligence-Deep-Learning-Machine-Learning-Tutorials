import antlr3
import testbase
import unittest


class t025lexerRulePropertyRef(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid1(self):
        stream = antlr3.StringStream('foobar _Ab98 \n A12sdf')
        lexer = self.getLexer(stream)

        while True:
            token = lexer.nextToken()
            if token.type == antlr3.EOF:
                break

        assert len(lexer.properties) == 3, lexer.properties

        text, type, line, pos, index, channel, start, stop = lexer.properties[0]
        assert text == 'foobar', lexer.properties[0]
        assert type == self.lexerModule.IDENTIFIER, lexer.properties[0]
        assert line == 1, lexer.properties[0]
        assert pos == 0, lexer.properties[0]
        assert index == -1, lexer.properties[0]
        assert channel == antlr3.DEFAULT_CHANNEL, lexer.properties[0]
        assert start == 0, lexer.properties[0]
        assert stop == 5, lexer.properties[0]

        text, type, line, pos, index, channel, start, stop = lexer.properties[1]
        assert text == '_Ab98', lexer.properties[1]
        assert type == self.lexerModule.IDENTIFIER, lexer.properties[1]
        assert line == 1, lexer.properties[1]
        assert pos == 7, lexer.properties[1]
        assert index == -1, lexer.properties[1]
        assert channel == antlr3.DEFAULT_CHANNEL, lexer.properties[1]
        assert start == 7, lexer.properties[1]
        assert stop == 11, lexer.properties[1]

        text, type, line, pos, index, channel, start, stop = lexer.properties[2]
        assert text == 'A12sdf', lexer.properties[2]
        assert type == self.lexerModule.IDENTIFIER, lexer.properties[2]
        assert line == 2, lexer.properties[2]
        assert pos == 1, lexer.properties[2]
        assert index == -1, lexer.properties[2]
        assert channel == antlr3.DEFAULT_CHANNEL, lexer.properties[2]
        assert start == 15, lexer.properties[2]
        assert stop == 20, lexer.properties[2]


if __name__ == '__main__':
    unittest.main()
