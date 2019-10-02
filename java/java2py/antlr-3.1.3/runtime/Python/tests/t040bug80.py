import antlr3
import testbase
import unittest


class t040bug80(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def lexerClass(self, base):
        class TLexer(base):
            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TLexer
    
        
    def testValid1(self):
        cStream = antlr3.StringStream('defined')
        lexer = self.getLexer(cStream)
        while True:
            t = lexer.nextToken()
            if t.type == antlr3.EOF:
                break
            print t


if __name__ == '__main__':
    unittest.main()


