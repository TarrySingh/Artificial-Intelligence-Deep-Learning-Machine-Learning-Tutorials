import antlr3
import testbase
import unittest
import textwrap


class t022scopes(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def parserClass(self, base):
        class TParser(base):
            def emitErrorMessage(self, msg):
                # report errors to /dev/null
                pass

            def reportError(self, re):
                # no error recovery yet, just crash!
                raise re

        return TParser

        
    def testa1(self):
        cStream = antlr3.StringStream('foobar')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.a()
        

    def testb1(self):
        cStream = antlr3.StringStream('foobar')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)

        try:
            parser.b(False)
            self.fail()
        except antlr3.RecognitionException:
            pass
        

    def testb2(self):
        cStream = antlr3.StringStream('foobar')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.b(True)
        

    def testc1(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            {
                int i;
                int j;
                i = 0;
            }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        symbols = parser.c()

        self.failUnlessEqual(
            symbols,
            set(['i', 'j'])
            )
        

    def testc2(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            {
                int i;
                int j;
                i = 0;
                x = 4;
            }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)

        try:
            parser.c()
            self.fail()
        except RuntimeError, exc:
            self.failUnlessEqual(exc.args[0], 'x')


    def testd1(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            {
                int i;
                int j;
                i = 0;
                {
                    int i;
                    int x;
                    x = 5;
                }
            }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        symbols = parser.d()

        self.failUnlessEqual(
            symbols,
            set(['i', 'j'])
            )


    def teste1(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            { { { { 12 } } } }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        res = parser.e()

        self.failUnlessEqual(res, 12)


    def testf1(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            { { { { 12 } } } }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        res = parser.f()

        self.failUnlessEqual(res, None)


    def testf2(self):
        cStream = antlr3.StringStream(
            textwrap.dedent('''\
            { { 12 } }
            '''))

        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        res = parser.f()

        self.failUnlessEqual(res, None)



if __name__ == '__main__':
    unittest.main()
