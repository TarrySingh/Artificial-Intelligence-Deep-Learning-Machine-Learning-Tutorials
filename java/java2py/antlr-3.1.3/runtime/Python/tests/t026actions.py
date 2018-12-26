import antlr3
import testbase
import unittest


class t026actions(testbase.ANTLRTest):
    def parserClass(self, base):
        class TParser(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._errors = []
                self._output = ""


            def capture(self, t):
                self._output += t


            def emitErrorMessage(self, msg):
                self._errors.append(msg)

            
        return TParser


    def lexerClass(self, base):
        class TLexer(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._errors = []
                self._output = ""


            def capture(self, t):
                self._output += t


            def emitErrorMessage(self, msg):
                self._errors.append(msg)

            
        return TLexer


    def setUp(self):
        self.compileGrammar()
        

    def testValid1(self):
        cStream = antlr3.StringStream('foobar _Ab98 \n A12sdf')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.prog()

        self.assertEqual(
            parser._output,
            'init;after;finally;')
        self.assertEqual(
            lexer._output,
            'action;u\'foobar\' 4 1 0 -1 0 0 5;attribute;action;u\'_Ab98\' 4 1 7 -1 0 7 11;attribute;action;u\'A12sdf\' 4 2 1 -1 0 15 20;attribute;')

if __name__ == '__main__':
    unittest.main()
