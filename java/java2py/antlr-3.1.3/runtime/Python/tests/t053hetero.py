import unittest
import textwrap
import antlr3
import antlr3.tree
import testbase
import sys

class T(testbase.ANTLRTest):
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


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise
            
        return TLexer
    

    def execParser(self, grammar, grammarEntry, input):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        r = getattr(parser, grammarEntry)()

        if r is not None:
            return r.tree.toStringTree()

        return ""
    

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

        if r is not None:
            return r.tree.toStringTree()

        return ""


    # PARSERS -- AUTO AST

    def testToken(self):
        grammar = textwrap.dedent(
        r'''
        grammar T1;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : ID<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testTokenWithQualifiedType(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            @members {
            class V(CommonTree):
                def toString(self):
                    return self.token.text + "<V>"
                __str__ = toString
            }
            a : ID<TParser.V> ; // TParser.V is qualified name
            ID : 'a'..'z'+ ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testTokenWithLabel(self):
        grammar = textwrap.dedent(
        r'''
        grammar T2;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : x=ID<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testTokenWithListLabel(self):
        grammar = textwrap.dedent(
        r'''
        grammar T3;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : x+=ID<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testTokenRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T4;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : ID<V>^ ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testTokenRootWithListLabel(self):
        grammar = textwrap.dedent(
        r'''
        grammar T5;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : x+=ID<V>^ ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testString(self):
        grammar = textwrap.dedent(
        r'''
        grammar T6;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : 'begin'<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="begin"
            )

        self.failUnlessEqual("begin<V>", found)


    def testStringRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T7;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : 'begin'<V>^ ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="begin"
            )

        self.failUnlessEqual("begin<V>", found)


    # PARSERS -- REWRITE AST

    def testRewriteToken(self):
        grammar = textwrap.dedent(
        r'''
        grammar T8;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : ID -> ID<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("a<V>", found)


    def testRewriteTokenWithArgs(self):
        grammar = textwrap.dedent(
        r'''
        grammar T9;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def __init__(self, *args):
                if len(args) == 4:
                    ttype = args[0]
                    x = args[1]
                    y = args[2]
                    z = args[3]
                    token = CommonToken(type=ttype, text="")

                elif len(args) == 3:
                    ttype = args[0]
                    token = args[1]
                    x = args[2]
                    y, z = 0, 0

                else:
                    raise TypeError("Invalid args \%r" \% (args,))

                CommonTree.__init__(self, token)
                self.x = x
                self.y = y
                self.z = z
                
            def toString(self):
                txt = ""
                if self.token is not None:
                    txt += self.token.text
                txt +="<V>;\%d\%d\%d" \% (self.x, self.y, self.z)
                return txt
            __str__ = toString

        }
        a : ID -> ID<V>[42,19,30] ID<V>[$ID,99];
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a"
            )

        self.failUnlessEqual("<V>;421930 a<V>;9900", found)


    def testRewriteTokenRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T10;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : ID INT -> ^(ID<V> INT) ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="a 2"
            )

        self.failUnlessEqual("(a<V> 2)", found)


    def testRewriteString(self):
        grammar = textwrap.dedent(
        r'''
        grammar T11;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : 'begin' -> 'begin'<V> ;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="begin"
            )

        self.failUnlessEqual("begin<V>", found)


    def testRewriteStringRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T12;
        options {
            language=Python;
            output=AST;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        }
        a : 'begin' INT -> ^('begin'<V> INT) ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        found = self.execParser(
            grammar, 'a',
            input="begin 2"
            )

        self.failUnlessEqual("(begin<V> 2)", found)

    def testRewriteRuleResults(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            tokens {LIST;}
            @header {
            class V(CommonTree):
                def toString(self):
                    return self.token.text + "<V>"
                __str__ = toString

            class W(CommonTree):
                def __init__(self, tokenType, txt):
                    super(W, self).__init__(
                        CommonToken(type=tokenType, text=txt))

                def toString(self):
                    return self.token.text + "<W>"
                __str__ = toString

            }
            a : id (',' id)* -> ^(LIST<W>["LIST"] id+);
            id : ID -> ID<V>;
            ID : 'a'..'z'+ ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(
            grammar, 'a',
            input="a,b,c")

        self.failUnlessEqual("(LIST<W> a<V> b<V> c<V>)", found)

    def testCopySemanticsWithHetero(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            @header {
            class V(CommonTree):
                def dupNode(self):
                    return V(self)

                def toString(self):
                    return self.token.text + "<V>"
                __str__ = toString

            }
            a : type ID (',' ID)* ';' -> ^(type ID)+;
            type : 'int'<V> ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(
            grammar, 'a',
            input="int a, b, c;")
        self.failUnlessEqual("(int<V> a) (int<V> b) (int<V> c)", found)

    # TREE PARSERS -- REWRITE AST
        
    def testTreeParserRewriteFlatList(self):
        grammar = textwrap.dedent(
        r'''
        grammar T13;
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
        r'''
        tree grammar TP13;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T13;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        class W(CommonTree):
            def toString(self):
                return self.token.text + "<W>"
            __str__ = toString

        }
        a : ID INT -> INT<V> ID<W>
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc 34"
            )

        self.failUnlessEqual("34<V> abc<W>", found)


    def testTreeParserRewriteTree(self):
        grammar = textwrap.dedent(
        r'''
        grammar T14;
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
        r'''
        tree grammar TP14;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T14;
        }
        @header {
        class V(CommonTree):
            def toString(self):
                return self.token.text + "<V>"
            __str__ = toString

        class W(CommonTree):
            def toString(self):
                return self.token.text + "<W>"
            __str__ = toString

        }
        a : ID INT -> ^(INT<V> ID<W>)
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc 34"
            )

        self.failUnlessEqual("(34<V> abc<W>)", found)


    def testTreeParserRewriteImaginary(self):
        grammar = textwrap.dedent(
        r'''
        grammar T15;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
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
        tokens { ROOT; }
        @header {
        class V(CommonTree):
            def __init__(self, tokenType):
                CommonTree.__init__(self, CommonToken(tokenType))
                
            def toString(self):
                return tokenNames[self.token.type] + "<V>"
            __str__ = toString


        }
        a : ID -> ROOT<V> ID
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc"
            )

        self.failUnlessEqual("ROOT<V> abc", found)


    def testTreeParserRewriteImaginaryWithArgs(self):
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
        WS : (' '|'\n') {$channel=HIDDEN;} ;
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
        tokens { ROOT; }
        @header {
        class V(CommonTree):
            def __init__(self, tokenType, x):
                CommonTree.__init__(self, CommonToken(tokenType))
                self.x = x
                
            def toString(self):
                return tokenNames[self.token.type] + "<V>;" + str(self.x)
            __str__ = toString

        }
        a : ID -> ROOT<V>[42] ID
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc"
            )

        self.failUnlessEqual("ROOT<V>;42 abc", found)


    def testTreeParserRewriteImaginaryRoot(self):
        grammar = textwrap.dedent(
        r'''
        grammar T17;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        treeGrammar = textwrap.dedent(
        r'''
        tree grammar TP17;
        options {
            language=Python;
            output=AST;
            ASTLabelType=CommonTree;
            tokenVocab=T17;
        }
        tokens { ROOT; }
        @header {
        class V(CommonTree):
            def __init__(self, tokenType):
                CommonTree.__init__(self, CommonToken(tokenType))
                
            def toString(self):
                return tokenNames[self.token.type] + "<V>"
            __str__ = toString

        }
        a : ID -> ^(ROOT<V> ID)
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc"
            )

        self.failUnlessEqual("(ROOT<V> abc)", found)


    def testTreeParserRewriteImaginaryFromReal(self):
        grammar = textwrap.dedent(
        r'''
        grammar T18;
        options {
            language=Python;
            output=AST;
        }
        a : ID ;
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
        }
        tokens { ROOT; }
        @header {
        class V(CommonTree):
            def __init__(self, tokenType, tree=None):
                if tree is None:
                    CommonTree.__init__(self, CommonToken(tokenType))
                else:
                    CommonTree.__init__(self, tree)
                    self.token.type = tokenType
                
            def toString(self):
                return tokenNames[self.token.type]+"<V>@"+str(self.token.line)
            __str__ = toString

        }
        a : ID -> ROOT<V>[$ID]
          ;
        ''')

        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc"
            )

        self.failUnlessEqual("ROOT<V>@1", found)


    def testTreeParserAutoHeteroAST(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {
                language=Python;
                output=AST;
            }
            a : ID ';' ;
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
            tokens { ROOT; }
            @header {
            class V(CommonTree):
                def toString(self):
                    return CommonTree.toString(self) + "<V>"
                __str__ = toString

            }
            
            a : ID<V> ';'<V>;
            ''')
            
        found = self.execTreeParser(
            grammar, 'a',
            treeGrammar, 'a',
            input="abc;"
            )

        self.failUnlessEqual("abc<V> ;<V>", found)


if __name__ == '__main__':
    unittest.main()
