import antlr3
import testbase
import unittest


class t034tokenLabelPropertyRef(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def lexerClass(self, base):
        class TLexer(base):
            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TLexer
    
        
    def parserClass(self, base):
        class TParser(base):
            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TParser
    
        
    def testValid1(self):
        cStream = antlr3.StringStream('   a')

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.a()


if __name__ == '__main__':
    unittest.main()

