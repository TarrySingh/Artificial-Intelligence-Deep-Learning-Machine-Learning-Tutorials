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
                self.buf = ""

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
        r = getattr(walker, treeEntry)()

        if r.tree is not None:
            return r.tree.toStringTree()

        return ""
    

    def testFlatList(self):
        grammar = textwrap.dedent(
        r'''
        grammar T1;
        options {
            language=Python;
            output=AST;
        }
        a : ID INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP1;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T1;
        }
        
        a : ID INT -> INT ID;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("34 abc", found)


    def testSimpleTree(self):
        grammar = textwrap.dedent(
        r'''
        grammar T2;
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
        r'''
        tree grammar TP2;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T2;
        }
        a : ^(ID INT) -> ^(INT ID);
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("(34 abc)", found)


    def testCombinedRewriteAndAuto(self):
        grammar = textwrap.dedent(
        r'''
        grammar T3;
        options {
            language=Python;
            output=AST;
        }
        a : ID INT -> ^(ID INT) | INT ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP3;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T3;
        }
        a : ^(ID INT) -> ^(INT ID) | INT;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("(34 abc)", found)


        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "34"
            )

        self.failUnlessEqual("34", found)


    def testAvoidDup(self):
        grammar = textwrap.dedent(
        r'''
        grammar T4;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP4;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T4;
        }
        a : ID -> ^(ID ID);
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc"
            )

        self.failUnlessEqual("(abc abc)", found)


    def testLoop(self):
        grammar = textwrap.dedent(
        r'''
        grammar T5;
        options {
            language=Python;
            output=AST;
        }
        a : ID+ INT+ -> (^(ID INT))+ ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP5;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T5;
        }
        a : (^(ID INT))+ -> INT+ ID+;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a b c 3 4 5"
            )

        self.failUnlessEqual("3 4 5 a b c", found)


    def testAutoDup(self):
        grammar = textwrap.dedent(
        r'''
        grammar T6;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP6;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T6;
        }
        a : ID;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc"
            )

        self.failUnlessEqual("abc", found)


    def testAutoDupRule(self):
        grammar = textwrap.dedent(
        r'''
        grammar T7;
        options {
            language=Python;
            output=AST;
        }
        a : ID INT ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP7;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T7;
        }
        a : b c ;
        b : ID ;
        c : INT ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 1"
            )

        self.failUnlessEqual("a 1", found)


    def testAutoWildcard(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python;output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            a : ID . 
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34")
        self.assertEquals("abc 34", found)


#     def testNoWildcardAsRootError(self):
#         ErrorQueue equeue = new ErrorQueue();
#         ErrorManager.setErrorListener(equeue);
# > 
#         String treeGrammar =
#             "tree grammar TP;\n"+
#             "options {language=Python;output=AST;}
#             "a : ^(. INT) 
#             "  ;\n";
# > 
#         Grammar g = new Grammar(treeGrammar);
#         Tool antlr = newTool();
#         antlr.setOutputDirectory(null); // write to /dev/null
#         CodeGenerator generator = new CodeGenerator(antlr, g, "Java");
#         g.setCodeGenerator(generator);
#         generator.genRecognizer();
# > 
#         assertEquals("unexpected errors: "+equeue, 1, equeue.errors.size());
# > 
#         int expectedMsgID = ErrorManager.MSG_WILDCARD_AS_ROOT;
#         Object expectedArg = null;
#         antlr.RecognitionException expectedExc = null;
#         GrammarSyntaxMessage expectedMessage =
#             new GrammarSyntaxMessage(expectedMsgID, g, null, expectedArg, expectedExc);
# > 
#         checkError(equeue, expectedMessage);        
#     }

    def testAutoWildcard2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT -> ^(ID INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python;output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            a : ^(ID .) 
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34")
        self.assertEquals("(abc 34)", found)


    def testAutoWildcardWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
 
        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python;output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            a : ID c=. 
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34")
        self.assertEquals("abc 34", found)


    def testAutoWildcardWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python;output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            a : ID c+=. 
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34")
        self.assertEquals("abc 34", found)


    def testAutoDupMultiple(self):
        grammar = textwrap.dedent(
        r'''
        grammar T8;
        options {
            language=Python;
            output=AST;
        }
        a : ID ID INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP8;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T8;
        }
        a : ID ID INT
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a b 3"
            )

        self.failUnlessEqual("a b 3", found)


    def testAutoDupTree(self):
        grammar = textwrap.dedent(
        r'''
        grammar T9;
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
        r'''
        tree grammar TP9;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T9;
        }
        a : ^(ID INT)
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupTreeWithLabels(self):
        grammar = textwrap.dedent(
        r'''
        grammar T10;
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
        r'''
        tree grammar TP10;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T10;
        }
        a : ^(x=ID y=INT)
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupTreeWithListLabels(self):
        grammar = textwrap.dedent(
        r'''
        grammar T11;
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
        r'''
        tree grammar TP11;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T11;
        }
        a : ^(x+=ID y+=INT)
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupTreeWithRuleRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T12;
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
        r'''
        tree grammar TP12;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T12;
        }
        a : ^(b INT) ;
        b : ID ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupTreeWithRuleRootAndLabels(self):
        grammar = textwrap.dedent(
        r'''
        grammar T13;
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
        r'''
        tree grammar TP13;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T13;
        }
        a : ^(x=b INT) ;
        b : ID ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupTreeWithRuleRootAndListLabels(self):
        grammar = textwrap.dedent(
        r'''
        grammar T14;
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
        r'''
        tree grammar TP14;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T14;
        }
        a : ^(x+=b y+=c) ;
        b : ID ;
        c : INT ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a 3"
            )

        self.failUnlessEqual("(a 3)", found)


    def testAutoDupNestedTree(self):
        grammar = textwrap.dedent(
        r'''
        grammar T15;
        options {
            language=Python;
            output=AST;
        }
        a : x=ID y=ID INT -> ^($x ^($y INT));
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP15;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T15;
        }
        a : ^(ID ^(ID INT))
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "a b 3"
            )

        self.failUnlessEqual("(a (b 3))", found)


    def testDelete(self):
        grammar = textwrap.dedent(
        r'''
        grammar T16;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP16;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T16;
        }
        a : ID -> 
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc"
            )

        self.failUnlessEqual("", found)

    def testSetMatchNoRewrite(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {
                language=Python;
                output=AST;
                ASTLabelType=CommonTree;
                tokenVocab=T;
            }
            a : b INT;
            b : ID | INT;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("abc 34", found)

      
    def testSetOptionalMatchNoRewrite(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {
                language=Python;
                output=AST;
                ASTLabelType=CommonTree;
                tokenVocab=T;
            }
            a : (ID|INT)? INT ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34")
        
        self.failUnlessEqual("abc 34", found)


    def testSetMatchNoRewriteLevel2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : x=ID INT -> ^($x INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {
                language=Python;
                output=AST;
                ASTLabelType=CommonTree;
                tokenVocab=T;
            }
            a : ^(ID (ID | INT) ) ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("(abc 34)", found)


    def testSetMatchNoRewriteLevel2Root(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : x=ID INT -> ^($x INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {
                language=Python;
                output=AST;
                ASTLabelType=CommonTree;
                tokenVocab=T;
            }
            a : ^((ID | INT) INT) ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("(abc 34)", found)


    ## REWRITE MODE

    def testRewriteModeCombinedRewriteAndAuto(self):
        grammar = textwrap.dedent(
        r'''
        grammar T17;
        options {
            language=Python;
            output=AST;
        }
        a : ID INT -> ^(ID INT) | INT ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\\n') {$channel=HIDDEN;} ;
        ''')
        
        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP17;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T17;
            rewrite=true;
        }
        a : ^(ID INT) -> ^(ID["ick"] INT)
          | INT // leaves it alone, returning $a.start
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("(ick 34)", found)


        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            "34"
            )

        self.failUnlessEqual("34", found)


    def testRewriteModeFlatTree(self):
        grammar = textwrap.dedent(
            r'''
            grammar T18;
            options {
              language=Python;
              output=AST;
            }
            a : ID INT -> ID INT | INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP18;
            options {
              language=Python;
              output=AST;
              ASTLabelType=CommonTree;
              tokenVocab=T18;
              rewrite=true;
            }
            s : ID a ;
            a : INT -> INT["1"]
              ;
            ''')
        
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34"
            )
        self.assertEquals("abc 1", found)


    def testRewriteModeChainRuleFlatTree(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ID INT | INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : a ;
            a : b ;
            b : ID INT -> INT ID
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.assertEquals("34 abc", found)


    def testRewriteModeChainRuleTree(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID INT) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : a ;
            a : b ; // a.tree must become b.tree
            b : ^(ID INT) -> INT
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.assertEquals("34", found)


    def testRewriteModeChainRuleTree2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID INT) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            tokens { X; }
            s : a* b ; // only b contributes to tree, but it's after a*; s.tree = b.tree
            a : X ;
            b : ^(ID INT) -> INT
              ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.assertEquals("34", found)


    def testRewriteModeChainRuleTree3(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : 'boo' ID INT -> 'boo' ^(ID INT) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            tokens { X; }
            s : 'boo' a* b ; // don't reset s.tree to b.tree due to 'boo'
            a : X ;
            b : ^(ID INT) -> INT
              ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "boo abc 34")
        self.assertEquals("boo 34", found)


    def testRewriteModeChainRuleTree4(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : 'boo' ID INT -> ^('boo' ^(ID INT)) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            tokens { X; }
            s : ^('boo' a* b) ; // don't reset s.tree to b.tree due to 'boo'
            a : X ;
            b : ^(ID INT) -> INT
              ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "boo abc 34")
        self.assertEquals("(boo 34)", found)


    def testRewriteModeChainRuleTree5(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : 'boo' ID INT -> ^('boo' ^(ID INT)) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            tokens { X; }
            s : ^(a b) ; // s.tree is a.tree
            a : 'boo' ;
            b : ^(ID INT) -> INT
              ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "boo abc 34")
        self.assertEquals("(boo 34)", found)


    def testRewriteOfRuleRef(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ID INT | INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)
         
        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : a -> a ;
            a : ID INT -> ID INT ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.failUnlessEqual("abc 34", found)


    def testRewriteOfRuleRefRoot(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT INT -> ^(INT ^(ID INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(a ^(ID INT)) -> a ;
            a : INT ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 12 34")
        # emits whole tree when you ref the root since I can't know whether
        # you want the children or not.  You might be returning a whole new
        # tree.  Hmm...still seems weird.  oh well.
        self.failUnlessEqual("(12 (abc 34))", found)


    def testRewriteOfRuleRefRootLabeled(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT INT -> ^(INT ^(ID INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(label=a ^(ID INT)) -> a ;
            a : INT ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 12 34")
        # emits whole tree when you ref the root since I can't know whether
        # you want the children or not.  You might be returning a whole new
        # tree.  Hmm...still seems weird.  oh well.
        self.failUnlessEqual("(12 (abc 34))", found)


    def testRewriteOfRuleRefRootListLabeled(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT INT -> ^(INT ^(ID INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(label+=a ^(ID INT)) -> a ;
            a : INT ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 12 34")
        # emits whole tree when you ref the root since I can't know whether
        # you want the children or not.  You might be returning a whole new
        # tree.  Hmm...still seems weird.  oh well.
        self.failUnlessEqual("(12 (abc 34))", found)


    def testRewriteOfRuleRefChild(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID ^(INT INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(ID a) -> a ;
            a : ^(INT INT) ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.failUnlessEqual("(34 34)", found)


    def testRewriteOfRuleRefLabel(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID ^(INT INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(ID label=a) -> a ;
            a : ^(INT INT) ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.failUnlessEqual("(34 34)", found)


    def testRewriteOfRuleRefListLabel(self):
        grammar = textwrap.dedent(
            r"""
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID ^(INT INT));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            """)

        treeGrammar = textwrap.dedent(
            r"""
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(ID label+=a) -> a ;
            a : ^(INT INT) ;
            """)

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.failUnlessEqual("(34 34)", found)



    def testRewriteModeWithPredicatedRewrites(self):
        grammar = textwrap.dedent(
            r'''
            grammar T19;
            options {
              language=Python;
              output=AST;
            }
            a : ID INT -> ^(ID["root"] ^(ID INT)) | INT -> ^(ID["root"] INT) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP19;
            options {
              language=Python;
              output=AST;
              ASTLabelType=CommonTree;
              tokenVocab=T19;
              rewrite=true;
            }
            s : ^(ID a) { self.buf += $s.start.toStringTree() };
            a : ^(ID INT) -> {True}? ^(ID["ick"] INT)
                          -> INT
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34"
            )

        self.failUnlessEqual("(root (ick 34))", found)


    def testWildcardSingleNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID INT -> ^(ID["root"] INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {
                language=Python;
                output=AST;
                ASTLabelType=CommonTree;
                tokenVocab=T;
            }
            s : ^(ID c=.) -> $c
            ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34"
            )

        self.failUnlessEqual("34", found)

    def testWildcardUnlabeledSingleNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID INT -> ^(ID INT);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            s : ^(ID .) -> ID
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 34")
        self.assertEquals("abc", found)


    def testWildcardGrabsSubtree(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID x=INT y=INT z=INT -> ^(ID[\"root\"] ^($x $y $z));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            s : ^(ID c=.) -> $c
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 1 2 3")
        self.assertEquals("(1 2 3)", found)


    def testWildcardGrabsSubtree2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : ID x=INT y=INT z=INT -> ID ^($x $y $z);
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            s : ID c=. -> $c
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "abc 1 2 3")
        self.assertEquals("(1 2 3)", found)


    def testWildcardListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST;}
            a : INT INT INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T;}
            s : (c+=.)+ -> $c+
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "1 2 3")
        self.assertEquals("1 2 3", found)


    def testWildcardListLabel2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python; output=AST; ASTLabelType=CommonTree;}
            a  : x=INT y=INT z=INT -> ^($x ^($y $z) ^($y $z));
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        treeGrammar = textwrap.dedent(
            r'''
            tree grammar TP;
            options {language=Python; output=AST; ASTLabelType=CommonTree; tokenVocab=T; rewrite=true;}
            s : ^(INT (c+=.)+) -> $c+
              ;
            ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 's',
            "1 2 3")
        self.assertEquals("(2 3) (2 3)", found)


if __name__ == '__main__':
    unittest.main()
