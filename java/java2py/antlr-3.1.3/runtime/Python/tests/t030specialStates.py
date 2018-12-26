import antlr3
import testbase
import unittest


class t030specialStates(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid1(self):
        cStream = antlr3.StringStream('foo')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.r()


    def testValid2(self):
        cStream = antlr3.StringStream('foo name1')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        events = parser.r()


    def testValid3(self):
        cStream = antlr3.StringStream('bar name1')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.cond = False
        events = parser.r()


    def testValid4(self):
        cStream = antlr3.StringStream('bar name1 name2')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.cond = False
        events = parser.r()


if __name__ == '__main__':
    unittest.main()

