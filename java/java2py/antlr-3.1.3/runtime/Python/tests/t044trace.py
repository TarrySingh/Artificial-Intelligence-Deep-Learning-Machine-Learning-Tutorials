import antlr3
import testbase
import unittest


class T(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar(options='-trace')
        

    def lexerClass(self, base):
        class TLexer(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self.traces = []


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TLexer
    
        
    def parserClass(self, base):
        class TParser(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self.traces = []


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

            def getRuleInvocationStack(self):
                return self._getRuleInvocationStack(base.__module__)
            
        return TParser
    
        
    def testTrace(self):
        cStream = antlr3.StringStream('< 1 + 2 + 3 >')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.a()

        self.failUnlessEqual(
            lexer.traces,
            [ '>T__6', '<T__6', '>WS', '<WS', '>INT', '<INT', '>WS', '<WS',
              '>T__8', '<T__8', '>WS', '<WS', '>INT', '<INT', '>WS', '<WS',
              '>T__8', '<T__8', '>WS', '<WS', '>INT', '<INT', '>WS', '<WS',
              '>T__7', '<T__7']
            )

        self.failUnlessEqual(
            parser.traces,
            [ '>a', '>synpred1_t044trace_fragment', '<synpred1_t044trace_fragment', '>b', '>c',
              '<c', '>c', '<c', '>c', '<c', '<b', '<a' ]
            )
    
        
    def testInvokationStack(self):
        cStream = antlr3.StringStream('< 1 + 2 + 3 >')
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.a()

        self.failUnlessEqual(
            parser._stack,
            ['a', 'b', 'c']
            )
        
if __name__ == '__main__':
    unittest.main()
    
