import antlr3
import testbase
import unittest


class t032subrulePredict(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def parserClass(self, base):
        class TParser(base):
            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TParser
    
        
    def testValid1(self):
        cStream = antlr3.StringStream(
            'BEGIN A END'
            )

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.a()


    @testbase.broken("DFA tries to look beyond end of rule b", Exception)
    def testValid2(self):
        cStream = antlr3.StringStream(
            ' A'
            )

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.b()


if __name__ == '__main__':
    unittest.main()
