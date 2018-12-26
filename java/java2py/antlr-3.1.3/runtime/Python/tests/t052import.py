import unittest
import textwrap
import antlr3
import antlr3.tree
import testbase
import sys

class T(testbase.ANTLRTest):
    def setUp(self):
        self.oldPath = sys.path[:]
        sys.path.insert(0, self.baseDir)


    def tearDown(self):
        sys.path = self.oldPath


    def parserClass(self, base):
        class TParser(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._output = ""


            def capture(self, t):
                self._output += t


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise
            
        return TParser
    

    def lexerClass(self, base):
        class TLexer(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._output = ""


            def capture(self, t):
                self._output += t


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input):
                # no error recovery yet, just crash!
                raise
            
        return TLexer
    

    def execParser(self, grammar, grammarEntry, slaves, input):
        for slave in slaves:
            parserName = self.writeInlineGrammar(slave)[0]
            # slave parsers are imported as normal python modules
            # to force reloading current version, purge module from sys.modules
            try:
                del sys.modules[parserName+'Parser']
            except KeyError:
                pass
                
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        getattr(parser, grammarEntry)()

        return parser._output
    

    def execLexer(self, grammar, slaves, input):
        for slave in slaves:
            parserName = self.writeInlineGrammar(slave)[0]
            # slave parsers are imported as normal python modules
            # to force reloading current version, purge module from sys.modules
            try:
                del sys.modules[parserName+'Parser']
            except KeyError:
                pass
                
        lexerCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)

        while True:
            token = lexer.nextToken()
            if token is None or token.type == antlr3.EOF:
                break

            lexer._output += token.text
            
        return lexer._output
    

    def testDelegatorInvokesDelegateRule(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S1;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM1.capture(t)

        }
        
        a : B { self.capture("S.a") } ;
        ''')

        master = textwrap.dedent(
        r'''
        grammar M1;
        options {
            language=Python;
        }
        import S1;
        s : a ;
        B : 'b' ; // defines B from inherited token space
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 's',
            slaves=[slave],
            input="b"
            )

        self.failUnlessEqual("S.a", found)


    def testDelegatorInvokesDelegateRuleWithArgs(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S2;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM2.capture(t)
        }
        a[x] returns [y] : B {self.capture("S.a"); $y="1000";} ;
        ''')

        master = textwrap.dedent(
        r'''
        grammar M2;
        options {
            language=Python;
        }
        import S2;
        s : label=a[3] {self.capture($label.y);} ;
        B : 'b' ; // defines B from inherited token space
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 's',
            slaves=[slave],
            input="b"
            )

        self.failUnlessEqual("S.a1000", found)


    def testDelegatorAccessesDelegateMembers(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S3;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM3.capture(t)

            def foo(self):
                self.capture("foo")
        }
        a : B ;
        ''')

        master = textwrap.dedent(
        r'''
        grammar M3;        // uses no rules from the import
        options {
            language=Python;
        }
        import S3;
        s : 'b' {self.gS3.foo();} ; // gS is import pointer
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 's',
            slaves=[slave],
            input="b"
            )

        self.failUnlessEqual("foo", found)


    def testDelegatorInvokesFirstVersionOfDelegateRule(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S4;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM4.capture(t)
        }
        a : b {self.capture("S.a");} ;
        b : B ;
        ''')

        slave2 = textwrap.dedent(
        r'''
        parser grammar T4;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM4.capture(t)
        }
        a : B {self.capture("T.a");} ; // hidden by S.a
        ''')

        master = textwrap.dedent(
        r'''
        grammar M4;
        options {
            language=Python;
        }
        import S4,T4;
        s : a ;
        B : 'b' ;
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 's',
            slaves=[slave, slave2],
            input="b"
            )

        self.failUnlessEqual("S.a", found)


    def testDelegatesSeeSameTokenType(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S5; // A, B, C token type order
        options {
            language=Python;
        }
        tokens { A; B; C; }
        @members {
            def capture(self, t):
                self.gM5.capture(t)
        }
        x : A {self.capture("S.x ");} ;
        ''')

        slave2 = textwrap.dedent(
        r'''
        parser grammar T5;
        options {
            language=Python;
        }
        tokens { C; B; A; } /// reverse order
        @members {
            def capture(self, t):
                self.gM5.capture(t)
        }
        y : A {self.capture("T.y");} ;
        ''')

        master = textwrap.dedent(
        r'''
        grammar M5;
        options {
            language=Python;
        }
        import S5,T5;
        s : x y ; // matches AA, which should be "aa"
        B : 'b' ; // another order: B, A, C
        A : 'a' ;
        C : 'c' ;
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 's',
            slaves=[slave, slave2],
            input="aa"
            )

        self.failUnlessEqual("S.x T.y", found)


    def testDelegatorRuleOverridesDelegate(self):
        slave = textwrap.dedent(
        r'''
        parser grammar S6;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM6.capture(t)
        }
        a : b {self.capture("S.a");} ;
        b : B ;
        ''')

        master = textwrap.dedent(
        r'''
        grammar M6;
        options {
            language=Python;
        }
        import S6;
        b : 'b'|'c' ;
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execParser(
            master, 'a',
            slaves=[slave],
            input="c"
            )

        self.failUnlessEqual("S.a", found)


    # LEXER INHERITANCE

    def testLexerDelegatorInvokesDelegateRule(self):
        slave = textwrap.dedent(
        r'''
        lexer grammar S7;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM7.capture(t)
        }
        A : 'a' {self.capture("S.A ");} ;
        C : 'c' ;
        ''')

        master = textwrap.dedent(
        r'''
        lexer grammar M7;
        options {
            language=Python;
        }
        import S7;
        B : 'b' ;
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execLexer(
            master,
            slaves=[slave],
            input="abc"
            )

        self.failUnlessEqual("S.A abc", found)


    def testLexerDelegatorRuleOverridesDelegate(self):
        slave = textwrap.dedent(
        r'''
        lexer grammar S8;
        options {
            language=Python;
        }
        @members {
            def capture(self, t):
                self.gM8.capture(t)
        }
        A : 'a' {self.capture("S.A")} ;
        ''')

        master = textwrap.dedent(
        r'''
        lexer grammar M8;
        options {
            language=Python;
        }
        import S8;
        A : 'a' {self.capture("M.A ");} ;
        WS : (' '|'\n') {self.skip()} ;
        ''')

        found = self.execLexer(
            master,
            slaves=[slave],
            input="a"
            )

        self.failUnlessEqual("M.A a", found)

        
if __name__ == '__main__':
    unittest.main()
