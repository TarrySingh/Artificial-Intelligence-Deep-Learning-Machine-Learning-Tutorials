import unittest
import textwrap
import antlr3
import antlr3.tree
import testbase

class T(testbase.ANTLRTest):
    def walkerClass(self, base):
        class TWalker(base):
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
            
        return TWalker
    

    def execTreeParser(self, grammar, grammarEntry, treeGrammar, treeEntry, input):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)
        walkerCls = self.compileInlineGrammar(treeGrammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        r = getattr(parser, grammarEntry)()
        nodes = antlr3.tree.CommonTreeNodeStream(r.tree)
        nodes.setTokenStream(tStream)
        walker = walkerCls(nodes)
        getattr(walker, treeEntry)()

        return walker._output
    

    def testFlatList(self):
        grammar = textwrap.dedent(
        r'''grammar T;
        options {
            language=Python;
            output=AST;
        }
        a : ID INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''tree grammar TP;
        options {
            language=Python;
            ASTLabelType=CommonTree;
        }
        a : ID INT
            {self.capture("\%s, \%s" \% ($ID, $INT))}
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("abc, 34", found)
        


    def testSimpleTree(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT -> ^(ID INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : ^(ID INT)
                {self.capture(str($ID)+", "+str($INT))}
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )
            
        self.failUnlessEqual("abc, 34", found)


    def testFlatVsTreeDecision(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : b c ;
            b : ID INT -> ^(ID INT);
            c : ID INT;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\\n') {$channel=HIDDEN;} ;
            ''')
        
        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : b b ;
            b : ID INT    {self.capture(str($ID)+" "+str($INT)+'\n')}
              | ^(ID INT) {self.capture("^("+str($ID)+" "+str($INT)+')');}
              ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 1 b 2"
            )
        self.failUnlessEqual("^(a 1)b 2\n", found)


    def testFlatVsTreeDecision2(self):
        grammar = textwrap.dedent(
            r"""grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : b c ;
            b : ID INT+ -> ^(ID INT+);
            c : ID INT+;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : b b ;
            b : ID INT+    {self.capture(str($ID)+" "+str($INT)+"\n")}
              | ^(x=ID (y=INT)+) {self.capture("^("+str($x)+' '+str($y)+')')}
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 1 2 3 b 4 5"
            )
        self.failUnlessEqual("^(a 3)b 5\n", found)


    def testCyclicDFALookahead(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT+ PERIOD;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            PERIOD : '.' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : ID INT+ PERIOD {self.capture("alt 1")}
              | ID INT+ SEMI   {self.capture("alt 2")}
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 1 2 3."
            )
        self.failUnlessEqual("alt 1", found)


##     def testTemplateOutput(self):
## 		String grammar =
## 			"grammar T;\n" +
## 			"options {output=AST;}\n" +
## 			"a : ID INT;\n" +
## 			"ID : 'a'..'z'+ ;\n" +
## 			"INT : '0'..'9'+;\n" +
## 			"WS : (' '|'\\n') {$channel=HIDDEN;} ;\n";

## 		String treeGrammar =
## 			"tree grammar TP;\n" +
## 			"options {output=template; ASTLabelType=CommonTree;}\n" +
## 			"s : a {System.out.println($a.st);};\n" +
## 			"a : ID INT -> {new StringTemplate($INT.text)}\n" +
## 			"  ;\n";

## 		String found = execTreeParser("T.g", grammar, "TParser", "TP.g",
## 				    treeGrammar, "TP", "TLexer", "a", "s", "abc 34");
## 		assertEquals("34\n", found);
## 	}


    def testNullableChildList(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT? -> ^(ID INT?);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\\n') {$channel=HIDDEN;} ;
            ''')
        
        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : ^(ID INT?)
                {self.capture(str($ID))}
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc"
            )
        self.failUnlessEqual("abc", found)


    def testNullableChildList2(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT? SEMI -> ^(ID INT?) SEMI ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : ^(ID INT?) SEMI
                {self.capture(str($ID))}
              ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc;"
            )
        self.failUnlessEqual("abc", found)


    def testNullableChildList3(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : x=ID INT? (y=ID)? SEMI -> ^($x INT? $y?) SEMI ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            WS : (' '|'\\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a : ^(ID INT? b) SEMI
                {self.capture(str($ID)+", "+str($b.text))}
              ;
            b : ID? ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc def;"
            )
        self.failUnlessEqual("abc, def", found)


    def testActionsAfterRoot(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : x=ID INT? SEMI -> ^($x INT?) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''tree grammar TP;
            options {
                language=Python;
                ASTLabelType=CommonTree;
            }
            a @init {x=0} : ^(ID {x=1} {x=2} INT?)
                {self.capture(str($ID)+", "+str(x))}
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc;"
            )
        self.failUnless("abc, 2\n", found)


    def testWildcardLookahead(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID '+'^ INT;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            PERIOD : '.' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP; 
            options {language=Python; tokenVocab=T; ASTLabelType=CommonTree;}
            a : ^('+' . INT) { self.capture("alt 1") }
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a + 2")
        self.assertEquals("alt 1", found)


    def testWildcardLookahead2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID '+'^ INT;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            PERIOD : '.' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; tokenVocab=T; ASTLabelType=CommonTree;}
            a : ^('+' . INT) { self.capture("alt 1") }
              | ^('+' . .)   { self.capture("alt 2") }
              ;
            ''')

        # AMBIG upon '+' DOWN INT UP etc.. but so what.

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a + 2")
        self.assertEquals("alt 1", found)


    def testWildcardLookahead3(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID '+'^ INT;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            PERIOD : '.' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; tokenVocab=T; ASTLabelType=CommonTree;}
            a : ^('+' ID INT) { self.capture("alt 1") }
              | ^('+' . .)   { self.capture("alt 2") }
              ;
            ''')

        # AMBIG upon '+' DOWN INT UP etc.. but so what.

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a + 2")
        self.assertEquals("alt 1", found)


    def testWildcardPlusLookahead(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID '+'^ INT;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            SEMI : ';' ;
            PERIOD : '.' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; tokenVocab=T; ASTLabelType=CommonTree;}
            a : ^('+' INT INT ) { self.capture("alt 1") }
              | ^('+' .+)   { self.capture("alt 2") }
              ;
            ''')

        # AMBIG upon '+' DOWN INT UP etc.. but so what.

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a + 2")
        self.assertEquals("alt 2", found)


if __name__ == '__main__':
    unittest.main()
