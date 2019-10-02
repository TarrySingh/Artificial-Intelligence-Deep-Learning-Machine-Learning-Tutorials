import antlr3
import testbase
import unittest


class t039labels(testbase.ANTLRTest):
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
        cStream = antlr3.StringStream(
            'a, b, c, 1, 2 A FOOBAR GNU1 A BLARZ'
            )

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        ids, w = parser.a()

        assert len(ids) == 6, ids
        assert ids[0].text == 'a', ids[0]
        assert ids[1].text == 'b', ids[1]
        assert ids[2].text == 'c', ids[2]
        assert ids[3].text == '1', ids[3]
        assert ids[4].text == '2', ids[4]
        assert ids[5].text == 'A', ids[5]

        assert w.text == 'GNU1', w


if __name__ == '__main__':
    unittest.main()


