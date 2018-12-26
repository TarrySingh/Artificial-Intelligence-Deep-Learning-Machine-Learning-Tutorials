import antlr3
import testbase
import unittest


class t035ruleLabelPropertyRef(testbase.ANTLRTest):
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
        cStream = antlr3.StringStream('   a a a a  ')

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        start, stop, text = parser.a()

        # first token of rule b is the 2nd token (counting hidden tokens)
        assert start.index == 1, start

        # first token of rule b is the 7th token (counting hidden tokens)
        assert stop.index == 7, stop

        assert text == "a a a a", text


if __name__ == '__main__':
    unittest.main()
