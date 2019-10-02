# $ANTLR 3.1.3 Mar 18, 2009 10:09:25 Java.g 2012-01-29 13:54:04

import sys
from antlr3 import *
from antlr3.compat import set, frozenset

from antlr3.tree import *



# for convenience in actions
HIDDEN = BaseRecognizer.HIDDEN

# token types
PACKAGE=84
EXPONENT=173
STAR=49
WHILE=103
MOD=32
MOD_ASSIGN=33
CASE=58
CHAR=60
NEW=82
DO=64
GENERIC_TYPE_PARAM_LIST=138
CLASS_INSTANCE_INITIALIZER=121
ARRAY_ELEMENT_ACCESS=115
FOR_CONDITION=129
NOT=34
VAR_DECLARATION=160
ANNOTATION_METHOD_DECL=109
EOF=-1
DIV_ASSIGN=14
LOGICAL_AND=26
BREAK=56
UNARY_PLUS=159
BIT_SHIFT_RIGHT_ASSIGN=9
TYPE=157
RPAREN=43
INC=21
FINAL=70
IMPORT=78
STRING_LITERAL=170
FOR_UPDATE=132
FLOATING_POINT_LITERAL=168
CAST_EXPR=118
NOT_EQUAL=35
VOID_METHOD_DECL=163
THIS=95
RETURN=88
DOUBLE=65
ENUM_TOP_LEVEL_SCOPE=125
VOID=101
SUPER=92
COMMENT=181
ANNOTATION_INIT_KEY_LIST=107
JAVA_ID_START=178
FLOAT_TYPE_SUFFIX=174
PRE_DEC=149
RBRACK=41
IMPLEMENTS_CLAUSE=140
SWITCH_BLOCK_LABEL_LIST=154
LINE_COMMENT=182
PRIVATE=85
STATIC=90
BLOCK_SCOPE=117
ANNOTATION_INIT_DEFAULT_KEY=106
SWITCH=93
NULL=83
VAR_DECLARATOR=161
MINUS_ASSIGN=31
ELSE=66
STRICTFP=91
CHARACTER_LITERAL=169
PRE_INC=150
ANNOTATION_LIST=108
ELLIPSIS=17
NATIVE=81
OCTAL_ESCAPE=177
UNARY_MINUS=158
THROWS=97
LCURLY=23
INT=79
FORMAL_PARAM_VARARG_DECL=135
METHOD_CALL=144
ASSERT=54
TRY=100
INTERFACE_TOP_LEVEL_SCOPE=139
SHIFT_LEFT=45
WS=180
SHIFT_RIGHT=47
FORMAL_PARAM_STD_DECL=134
LOCAL_MODIFIER_LIST=142
OR=36
LESS_THAN=25
SHIFT_RIGHT_ASSIGN=48
EXTENDS_BOUND_LIST=127
JAVA_SOURCE=143
CATCH=59
FALSE=69
INTEGER_TYPE_SUFFIX=172
DECIMAL_LITERAL=167
THROW=96
FOR_INIT=131
DEC=12
PROTECTED=86
CLASS=61
LBRACK=22
BIT_SHIFT_RIGHT=8
THROWS_CLAUSE=156
GREATER_OR_EQUAL=19
FOR=73
THIS_CONSTRUCTOR_CALL=155
LOGICAL_NOT=27
JAVADOC_COMMENT=183
FLOAT=72
ABSTRACT=53
AND=4
POST_DEC=147
AND_ASSIGN=5
STATIC_ARRAY_CREATOR=152
MODIFIER_LIST=145
ANNOTATION_SCOPE=110
LPAREN=29
IF=74
AT=7
ESCAPE_SEQUENCE=175
CONSTRUCTOR_DECL=124
LABELED_STATEMENT=141
UNICODE_ESCAPE=176
EXPR=126
SYNCHRONIZED=94
BOOLEAN=55
CLASS_TOP_LEVEL_SCOPE=123
IMPLEMENTS=75
CONTINUE=62
COMMA=11
TRANSIENT=98
EQUAL=18
XOR_ASSIGN=52
LOGICAL_OR=28
IDENT=164
QUALIFIED_TYPE_IDENT=151
ARGUMENT_LIST=112
PLUS=38
HEX_LITERAL=165
ANNOTATION_INIT_BLOCK=105
DOT=15
SHIFT_LEFT_ASSIGN=46
FORMAL_PARAM_LIST=133
GENERIC_TYPE_ARG_LIST=137
ANNOTATION_TOP_LEVEL_SCOPE=111
DOTSTAR=16
BYTE=57
XOR=51
JAVA_ID_PART=179
GREATER_THAN=20
VOLATILE=102
PARENTESIZED_EXPR=146
CLASS_STATIC_INITIALIZER=122
ARRAY_DECLARATOR_LIST=114
LESS_OR_EQUAL=24
DEFAULT=63
OCTAL_LITERAL=166
HEX_DIGIT=171
SHORT=89
INSTANCEOF=76
MINUS=30
EXTENDS_CLAUSE=128
TRUE=99
SEMI=44
STAR_ASSIGN=50
VAR_DECLARATOR_LIST=162
ARRAY_DECLARATOR=113
COLON=10
OR_ASSIGN=37
ENUM=67
QUESTION=40
FINALLY=71
RCURLY=42
PLUS_ASSIGN=39
ASSIGN=6
ANNOTATION_INIT_ARRAY_ELEMENT=104
FUNCTION_METHOD_DECL=136
INTERFACE=77
POST_INC=148
DIV=13
CLASS_CONSTRUCTOR_CALL=120
LONG=80
FOR_EACH=130
EXTENDS=68
PUBLIC=87
ARRAY_INITIALIZER=116
CATCH_CLAUSE_LIST=119
SUPER_CONSTRUCTOR_CALL=153

# token names
tokenNames = [
    "<invalid>", "<EOR>", "<DOWN>", "<UP>", 
    "AND", "AND_ASSIGN", "ASSIGN", "AT", "BIT_SHIFT_RIGHT", "BIT_SHIFT_RIGHT_ASSIGN", 
    "COLON", "COMMA", "DEC", "DIV", "DIV_ASSIGN", "DOT", "DOTSTAR", "ELLIPSIS", 
    "EQUAL", "GREATER_OR_EQUAL", "GREATER_THAN", "INC", "LBRACK", "LCURLY", 
    "LESS_OR_EQUAL", "LESS_THAN", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR", 
    "LPAREN", "MINUS", "MINUS_ASSIGN", "MOD", "MOD_ASSIGN", "NOT", "NOT_EQUAL", 
    "OR", "OR_ASSIGN", "PLUS", "PLUS_ASSIGN", "QUESTION", "RBRACK", "RCURLY", 
    "RPAREN", "SEMI", "SHIFT_LEFT", "SHIFT_LEFT_ASSIGN", "SHIFT_RIGHT", 
    "SHIFT_RIGHT_ASSIGN", "STAR", "STAR_ASSIGN", "XOR", "XOR_ASSIGN", "ABSTRACT", 
    "ASSERT", "BOOLEAN", "BREAK", "BYTE", "CASE", "CATCH", "CHAR", "CLASS", 
    "CONTINUE", "DEFAULT", "DO", "DOUBLE", "ELSE", "ENUM", "EXTENDS", "FALSE", 
    "FINAL", "FINALLY", "FLOAT", "FOR", "IF", "IMPLEMENTS", "INSTANCEOF", 
    "INTERFACE", "IMPORT", "INT", "LONG", "NATIVE", "NEW", "NULL", "PACKAGE", 
    "PRIVATE", "PROTECTED", "PUBLIC", "RETURN", "SHORT", "STATIC", "STRICTFP", 
    "SUPER", "SWITCH", "SYNCHRONIZED", "THIS", "THROW", "THROWS", "TRANSIENT", 
    "TRUE", "TRY", "VOID", "VOLATILE", "WHILE", "ANNOTATION_INIT_ARRAY_ELEMENT", 
    "ANNOTATION_INIT_BLOCK", "ANNOTATION_INIT_DEFAULT_KEY", "ANNOTATION_INIT_KEY_LIST", 
    "ANNOTATION_LIST", "ANNOTATION_METHOD_DECL", "ANNOTATION_SCOPE", "ANNOTATION_TOP_LEVEL_SCOPE", 
    "ARGUMENT_LIST", "ARRAY_DECLARATOR", "ARRAY_DECLARATOR_LIST", "ARRAY_ELEMENT_ACCESS", 
    "ARRAY_INITIALIZER", "BLOCK_SCOPE", "CAST_EXPR", "CATCH_CLAUSE_LIST", 
    "CLASS_CONSTRUCTOR_CALL", "CLASS_INSTANCE_INITIALIZER", "CLASS_STATIC_INITIALIZER", 
    "CLASS_TOP_LEVEL_SCOPE", "CONSTRUCTOR_DECL", "ENUM_TOP_LEVEL_SCOPE", 
    "EXPR", "EXTENDS_BOUND_LIST", "EXTENDS_CLAUSE", "FOR_CONDITION", "FOR_EACH", 
    "FOR_INIT", "FOR_UPDATE", "FORMAL_PARAM_LIST", "FORMAL_PARAM_STD_DECL", 
    "FORMAL_PARAM_VARARG_DECL", "FUNCTION_METHOD_DECL", "GENERIC_TYPE_ARG_LIST", 
    "GENERIC_TYPE_PARAM_LIST", "INTERFACE_TOP_LEVEL_SCOPE", "IMPLEMENTS_CLAUSE", 
    "LABELED_STATEMENT", "LOCAL_MODIFIER_LIST", "JAVA_SOURCE", "METHOD_CALL", 
    "MODIFIER_LIST", "PARENTESIZED_EXPR", "POST_DEC", "POST_INC", "PRE_DEC", 
    "PRE_INC", "QUALIFIED_TYPE_IDENT", "STATIC_ARRAY_CREATOR", "SUPER_CONSTRUCTOR_CALL", 
    "SWITCH_BLOCK_LABEL_LIST", "THIS_CONSTRUCTOR_CALL", "THROWS_CLAUSE", 
    "TYPE", "UNARY_MINUS", "UNARY_PLUS", "VAR_DECLARATION", "VAR_DECLARATOR", 
    "VAR_DECLARATOR_LIST", "VOID_METHOD_DECL", "IDENT", "HEX_LITERAL", "OCTAL_LITERAL", 
    "DECIMAL_LITERAL", "FLOATING_POINT_LITERAL", "CHARACTER_LITERAL", "STRING_LITERAL", 
    "HEX_DIGIT", "INTEGER_TYPE_SUFFIX", "EXPONENT", "FLOAT_TYPE_SUFFIX", 
    "ESCAPE_SEQUENCE", "UNICODE_ESCAPE", "OCTAL_ESCAPE", "JAVA_ID_START", 
    "JAVA_ID_PART", "WS", "COMMENT", "LINE_COMMENT", "JAVADOC_COMMENT"
]




class JavaParser(Parser):
    grammarFileName = "Java.g"
    antlr_version = version_str_to_tuple("3.1.3 Mar 18, 2009 10:09:25")
    antlr_version_str = "3.1.3 Mar 18, 2009 10:09:25"
    tokenNames = tokenNames

    def __init__(self, input, state=None, *args, **kwargs):
        if state is None:
            state = RecognizerSharedState()

        super(JavaParser, self).__init__(input, state, *args, **kwargs)

        self._state.ruleMemo = {}
        self.dfa36 = self.DFA36(
            self, 36,
            eot = self.DFA36_eot,
            eof = self.DFA36_eof,
            min = self.DFA36_min,
            max = self.DFA36_max,
            accept = self.DFA36_accept,
            special = self.DFA36_special,
            transition = self.DFA36_transition
            )

        self.dfa43 = self.DFA43(
            self, 43,
            eot = self.DFA43_eot,
            eof = self.DFA43_eof,
            min = self.DFA43_min,
            max = self.DFA43_max,
            accept = self.DFA43_accept,
            special = self.DFA43_special,
            transition = self.DFA43_transition
            )

        self.dfa86 = self.DFA86(
            self, 86,
            eot = self.DFA86_eot,
            eof = self.DFA86_eof,
            min = self.DFA86_min,
            max = self.DFA86_max,
            accept = self.DFA86_accept,
            special = self.DFA86_special,
            transition = self.DFA86_transition
            )

        self.dfa88 = self.DFA88(
            self, 88,
            eot = self.DFA88_eot,
            eof = self.DFA88_eof,
            min = self.DFA88_min,
            max = self.DFA88_max,
            accept = self.DFA88_accept,
            special = self.DFA88_special,
            transition = self.DFA88_transition
            )

        self.dfa98 = self.DFA98(
            self, 98,
            eot = self.DFA98_eot,
            eof = self.DFA98_eof,
            min = self.DFA98_min,
            max = self.DFA98_max,
            accept = self.DFA98_accept,
            special = self.DFA98_special,
            transition = self.DFA98_transition
            )

        self.dfa91 = self.DFA91(
            self, 91,
            eot = self.DFA91_eot,
            eof = self.DFA91_eof,
            min = self.DFA91_min,
            max = self.DFA91_max,
            accept = self.DFA91_accept,
            special = self.DFA91_special,
            transition = self.DFA91_transition
            )

        self.dfa106 = self.DFA106(
            self, 106,
            eot = self.DFA106_eot,
            eof = self.DFA106_eof,
            min = self.DFA106_min,
            max = self.DFA106_max,
            accept = self.DFA106_accept,
            special = self.DFA106_special,
            transition = self.DFA106_transition
            )

        self.dfa130 = self.DFA130(
            self, 130,
            eot = self.DFA130_eot,
            eof = self.DFA130_eof,
            min = self.DFA130_min,
            max = self.DFA130_max,
            accept = self.DFA130_accept,
            special = self.DFA130_special,
            transition = self.DFA130_transition
            )

        self.dfa142 = self.DFA142(
            self, 142,
            eot = self.DFA142_eot,
            eof = self.DFA142_eof,
            min = self.DFA142_min,
            max = self.DFA142_max,
            accept = self.DFA142_accept,
            special = self.DFA142_special,
            transition = self.DFA142_transition
            )

        self.dfa146 = self.DFA146(
            self, 146,
            eot = self.DFA146_eot,
            eof = self.DFA146_eof,
            min = self.DFA146_min,
            max = self.DFA146_max,
            accept = self.DFA146_accept,
            special = self.DFA146_special,
            transition = self.DFA146_transition
            )

        self.dfa153 = self.DFA153(
            self, 153,
            eot = self.DFA153_eot,
            eof = self.DFA153_eof,
            min = self.DFA153_min,
            max = self.DFA153_max,
            accept = self.DFA153_accept,
            special = self.DFA153_special,
            transition = self.DFA153_transition
            )






        self._adaptor = None
        self.adaptor = CommonTreeAdaptor()
                


        
    def getTreeAdaptor(self):
        return self._adaptor

    def setTreeAdaptor(self, adaptor):
        self._adaptor = adaptor

    adaptor = property(getTreeAdaptor, setTreeAdaptor)


    class javaSource_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.javaSource_return, self).__init__()

            self.tree = None




    # $ANTLR start "javaSource"
    # Java.g:222:1: javaSource : compilationUnit -> ^( JAVA_SOURCE compilationUnit ) ;
    def javaSource(self, ):

        retval = self.javaSource_return()
        retval.start = self.input.LT(1)
        javaSource_StartIndex = self.input.index()
        root_0 = None

        compilationUnit1 = None


        stream_compilationUnit = RewriteRuleSubtreeStream(self._adaptor, "rule compilationUnit")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 1):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:223:5: ( compilationUnit -> ^( JAVA_SOURCE compilationUnit ) )
                # Java.g:223:9: compilationUnit
                pass 
                self._state.following.append(self.FOLLOW_compilationUnit_in_javaSource4461)
                compilationUnit1 = self.compilationUnit()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_compilationUnit.add(compilationUnit1.tree)

                # AST Rewrite
                # elements: compilationUnit
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 224:9: -> ^( JAVA_SOURCE compilationUnit )
                    # Java.g:224:13: ^( JAVA_SOURCE compilationUnit )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(JAVA_SOURCE, "JAVA_SOURCE"), root_1)

                    self._adaptor.addChild(root_1, stream_compilationUnit.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 1, javaSource_StartIndex, success)

            pass
        return retval

    # $ANTLR end "javaSource"

    class compilationUnit_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.compilationUnit_return, self).__init__()

            self.tree = None




    # $ANTLR start "compilationUnit"
    # Java.g:227:1: compilationUnit : annotationList ( packageDeclaration )? ( importDeclaration )* ( typeDecls )* ;
    def compilationUnit(self, ):

        retval = self.compilationUnit_return()
        retval.start = self.input.LT(1)
        compilationUnit_StartIndex = self.input.index()
        root_0 = None

        annotationList2 = None

        packageDeclaration3 = None

        importDeclaration4 = None

        typeDecls5 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 2):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:228:5: ( annotationList ( packageDeclaration )? ( importDeclaration )* ( typeDecls )* )
                # Java.g:228:9: annotationList ( packageDeclaration )? ( importDeclaration )* ( typeDecls )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_annotationList_in_compilationUnit4497)
                annotationList2 = self.annotationList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, annotationList2.tree)
                # Java.g:229:9: ( packageDeclaration )?
                alt1 = 2
                LA1_0 = self.input.LA(1)

                if (LA1_0 == PACKAGE) :
                    alt1 = 1
                if alt1 == 1:
                    # Java.g:0:0: packageDeclaration
                    pass 
                    self._state.following.append(self.FOLLOW_packageDeclaration_in_compilationUnit4507)
                    packageDeclaration3 = self.packageDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, packageDeclaration3.tree)



                # Java.g:230:9: ( importDeclaration )*
                while True: #loop2
                    alt2 = 2
                    LA2_0 = self.input.LA(1)

                    if (LA2_0 == IMPORT) :
                        alt2 = 1


                    if alt2 == 1:
                        # Java.g:0:0: importDeclaration
                        pass 
                        self._state.following.append(self.FOLLOW_importDeclaration_in_compilationUnit4518)
                        importDeclaration4 = self.importDeclaration()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, importDeclaration4.tree)


                    else:
                        break #loop2
                # Java.g:231:9: ( typeDecls )*
                while True: #loop3
                    alt3 = 2
                    LA3_0 = self.input.LA(1)

                    if (LA3_0 == AT or LA3_0 == LESS_THAN or LA3_0 == SEMI or LA3_0 == ABSTRACT or LA3_0 == BOOLEAN or LA3_0 == BYTE or (CHAR <= LA3_0 <= CLASS) or LA3_0 == DOUBLE or LA3_0 == ENUM or LA3_0 == FINAL or LA3_0 == FLOAT or LA3_0 == INTERFACE or (INT <= LA3_0 <= NATIVE) or (PRIVATE <= LA3_0 <= PUBLIC) or (SHORT <= LA3_0 <= STRICTFP) or LA3_0 == SYNCHRONIZED or LA3_0 == TRANSIENT or (VOID <= LA3_0 <= VOLATILE) or LA3_0 == IDENT) :
                        alt3 = 1


                    if alt3 == 1:
                        # Java.g:0:0: typeDecls
                        pass 
                        self._state.following.append(self.FOLLOW_typeDecls_in_compilationUnit4529)
                        typeDecls5 = self.typeDecls()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, typeDecls5.tree)


                    else:
                        break #loop3



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 2, compilationUnit_StartIndex, success)

            pass
        return retval

    # $ANTLR end "compilationUnit"

    class typeDecls_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.typeDecls_return, self).__init__()

            self.tree = None




    # $ANTLR start "typeDecls"
    # Java.g:234:1: typeDecls : ( typeDeclaration | SEMI );
    def typeDecls(self, ):

        retval = self.typeDecls_return()
        retval.start = self.input.LT(1)
        typeDecls_StartIndex = self.input.index()
        root_0 = None

        SEMI7 = None
        typeDeclaration6 = None


        SEMI7_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 3):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:235:5: ( typeDeclaration | SEMI )
                alt4 = 2
                LA4_0 = self.input.LA(1)

                if (LA4_0 == AT or LA4_0 == LESS_THAN or LA4_0 == ABSTRACT or LA4_0 == BOOLEAN or LA4_0 == BYTE or (CHAR <= LA4_0 <= CLASS) or LA4_0 == DOUBLE or LA4_0 == ENUM or LA4_0 == FINAL or LA4_0 == FLOAT or LA4_0 == INTERFACE or (INT <= LA4_0 <= NATIVE) or (PRIVATE <= LA4_0 <= PUBLIC) or (SHORT <= LA4_0 <= STRICTFP) or LA4_0 == SYNCHRONIZED or LA4_0 == TRANSIENT or (VOID <= LA4_0 <= VOLATILE) or LA4_0 == IDENT) :
                    alt4 = 1
                elif (LA4_0 == SEMI) :
                    alt4 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 4, 0, self.input)

                    raise nvae

                if alt4 == 1:
                    # Java.g:235:9: typeDeclaration
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_typeDeclaration_in_typeDecls4549)
                    typeDeclaration6 = self.typeDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, typeDeclaration6.tree)


                elif alt4 == 2:
                    # Java.g:236:9: SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    SEMI7=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_typeDecls4559)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 3, typeDecls_StartIndex, success)

            pass
        return retval

    # $ANTLR end "typeDecls"

    class packageDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.packageDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "packageDeclaration"
    # Java.g:239:1: packageDeclaration : PACKAGE qualifiedIdentifier SEMI ;
    def packageDeclaration(self, ):

        retval = self.packageDeclaration_return()
        retval.start = self.input.LT(1)
        packageDeclaration_StartIndex = self.input.index()
        root_0 = None

        PACKAGE8 = None
        SEMI10 = None
        qualifiedIdentifier9 = None


        PACKAGE8_tree = None
        SEMI10_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 4):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:240:5: ( PACKAGE qualifiedIdentifier SEMI )
                # Java.g:240:9: PACKAGE qualifiedIdentifier SEMI
                pass 
                root_0 = self._adaptor.nil()

                PACKAGE8=self.match(self.input, PACKAGE, self.FOLLOW_PACKAGE_in_packageDeclaration4579)
                if self._state.backtracking == 0:

                    PACKAGE8_tree = self._adaptor.createWithPayload(PACKAGE8)
                    root_0 = self._adaptor.becomeRoot(PACKAGE8_tree, root_0)

                self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_packageDeclaration4582)
                qualifiedIdentifier9 = self.qualifiedIdentifier()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, qualifiedIdentifier9.tree)
                SEMI10=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_packageDeclaration4584)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 4, packageDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "packageDeclaration"

    class importDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.importDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "importDeclaration"
    # Java.g:243:1: importDeclaration : IMPORT ( STATIC )? qualifiedIdentifier ( DOTSTAR )? SEMI ;
    def importDeclaration(self, ):

        retval = self.importDeclaration_return()
        retval.start = self.input.LT(1)
        importDeclaration_StartIndex = self.input.index()
        root_0 = None

        IMPORT11 = None
        STATIC12 = None
        DOTSTAR14 = None
        SEMI15 = None
        qualifiedIdentifier13 = None


        IMPORT11_tree = None
        STATIC12_tree = None
        DOTSTAR14_tree = None
        SEMI15_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 5):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:244:5: ( IMPORT ( STATIC )? qualifiedIdentifier ( DOTSTAR )? SEMI )
                # Java.g:244:9: IMPORT ( STATIC )? qualifiedIdentifier ( DOTSTAR )? SEMI
                pass 
                root_0 = self._adaptor.nil()

                IMPORT11=self.match(self.input, IMPORT, self.FOLLOW_IMPORT_in_importDeclaration4604)
                if self._state.backtracking == 0:

                    IMPORT11_tree = self._adaptor.createWithPayload(IMPORT11)
                    root_0 = self._adaptor.becomeRoot(IMPORT11_tree, root_0)

                # Java.g:244:17: ( STATIC )?
                alt5 = 2
                LA5_0 = self.input.LA(1)

                if (LA5_0 == STATIC) :
                    alt5 = 1
                if alt5 == 1:
                    # Java.g:0:0: STATIC
                    pass 
                    STATIC12=self.match(self.input, STATIC, self.FOLLOW_STATIC_in_importDeclaration4607)
                    if self._state.backtracking == 0:

                        STATIC12_tree = self._adaptor.createWithPayload(STATIC12)
                        self._adaptor.addChild(root_0, STATIC12_tree)




                self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_importDeclaration4610)
                qualifiedIdentifier13 = self.qualifiedIdentifier()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, qualifiedIdentifier13.tree)
                # Java.g:244:45: ( DOTSTAR )?
                alt6 = 2
                LA6_0 = self.input.LA(1)

                if (LA6_0 == DOTSTAR) :
                    alt6 = 1
                if alt6 == 1:
                    # Java.g:0:0: DOTSTAR
                    pass 
                    DOTSTAR14=self.match(self.input, DOTSTAR, self.FOLLOW_DOTSTAR_in_importDeclaration4612)
                    if self._state.backtracking == 0:

                        DOTSTAR14_tree = self._adaptor.createWithPayload(DOTSTAR14)
                        self._adaptor.addChild(root_0, DOTSTAR14_tree)




                SEMI15=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_importDeclaration4615)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 5, importDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "importDeclaration"

    class typeDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.typeDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "typeDeclaration"
    # Java.g:247:1: typeDeclaration : modifierList ( classTypeDeclaration[$modifierList.tree] | interfaceTypeDeclaration[$modifierList.tree] | enumTypeDeclaration[$modifierList.tree] | annotationTypeDeclaration[$modifierList.tree] ) ;
    def typeDeclaration(self, ):

        retval = self.typeDeclaration_return()
        retval.start = self.input.LT(1)
        typeDeclaration_StartIndex = self.input.index()
        root_0 = None

        modifierList16 = None

        classTypeDeclaration17 = None

        interfaceTypeDeclaration18 = None

        enumTypeDeclaration19 = None

        annotationTypeDeclaration20 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 6):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:248:5: ( modifierList ( classTypeDeclaration[$modifierList.tree] | interfaceTypeDeclaration[$modifierList.tree] | enumTypeDeclaration[$modifierList.tree] | annotationTypeDeclaration[$modifierList.tree] ) )
                # Java.g:248:9: modifierList ( classTypeDeclaration[$modifierList.tree] | interfaceTypeDeclaration[$modifierList.tree] | enumTypeDeclaration[$modifierList.tree] | annotationTypeDeclaration[$modifierList.tree] )
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_modifierList_in_typeDeclaration4635)
                modifierList16 = self.modifierList()

                self._state.following.pop()
                # Java.g:249:9: ( classTypeDeclaration[$modifierList.tree] | interfaceTypeDeclaration[$modifierList.tree] | enumTypeDeclaration[$modifierList.tree] | annotationTypeDeclaration[$modifierList.tree] )
                alt7 = 4
                LA7 = self.input.LA(1)
                if LA7 == CLASS:
                    alt7 = 1
                elif LA7 == INTERFACE:
                    alt7 = 2
                elif LA7 == ENUM:
                    alt7 = 3
                elif LA7 == AT:
                    alt7 = 4
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 7, 0, self.input)

                    raise nvae

                if alt7 == 1:
                    # Java.g:249:13: classTypeDeclaration[$modifierList.tree]
                    pass 
                    self._state.following.append(self.FOLLOW_classTypeDeclaration_in_typeDeclaration4650)
                    classTypeDeclaration17 = self.classTypeDeclaration(modifierList16.tree)

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, classTypeDeclaration17.tree)


                elif alt7 == 2:
                    # Java.g:250:13: interfaceTypeDeclaration[$modifierList.tree]
                    pass 
                    self._state.following.append(self.FOLLOW_interfaceTypeDeclaration_in_typeDeclaration4665)
                    interfaceTypeDeclaration18 = self.interfaceTypeDeclaration(modifierList16.tree)

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, interfaceTypeDeclaration18.tree)


                elif alt7 == 3:
                    # Java.g:251:13: enumTypeDeclaration[$modifierList.tree]
                    pass 
                    self._state.following.append(self.FOLLOW_enumTypeDeclaration_in_typeDeclaration4680)
                    enumTypeDeclaration19 = self.enumTypeDeclaration(modifierList16.tree)

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, enumTypeDeclaration19.tree)


                elif alt7 == 4:
                    # Java.g:252:13: annotationTypeDeclaration[$modifierList.tree]
                    pass 
                    self._state.following.append(self.FOLLOW_annotationTypeDeclaration_in_typeDeclaration4695)
                    annotationTypeDeclaration20 = self.annotationTypeDeclaration(modifierList16.tree)

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotationTypeDeclaration20.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 6, typeDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "typeDeclaration"

    class classTypeDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classTypeDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "classTypeDeclaration"
    # Java.g:256:1: classTypeDeclaration[modifiers] : CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody -> ^( CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody ) ;
    def classTypeDeclaration(self, modifiers):

        retval = self.classTypeDeclaration_return()
        retval.start = self.input.LT(1)
        classTypeDeclaration_StartIndex = self.input.index()
        root_0 = None

        CLASS21 = None
        IDENT22 = None
        genericTypeParameterList23 = None

        classExtendsClause24 = None

        implementsClause25 = None

        classBody26 = None


        CLASS21_tree = None
        IDENT22_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_CLASS = RewriteRuleTokenStream(self._adaptor, "token CLASS")
        stream_genericTypeParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeParameterList")
        stream_classExtendsClause = RewriteRuleSubtreeStream(self._adaptor, "rule classExtendsClause")
        stream_implementsClause = RewriteRuleSubtreeStream(self._adaptor, "rule implementsClause")
        stream_classBody = RewriteRuleSubtreeStream(self._adaptor, "rule classBody")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 7):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:257:5: ( CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody -> ^( CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody ) )
                # Java.g:257:9: CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody
                pass 
                CLASS21=self.match(self.input, CLASS, self.FOLLOW_CLASS_in_classTypeDeclaration4726) 
                if self._state.backtracking == 0:
                    stream_CLASS.add(CLASS21)
                IDENT22=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_classTypeDeclaration4728) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT22)
                # Java.g:257:21: ( genericTypeParameterList )?
                alt8 = 2
                LA8_0 = self.input.LA(1)

                if (LA8_0 == LESS_THAN) :
                    alt8 = 1
                if alt8 == 1:
                    # Java.g:0:0: genericTypeParameterList
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeParameterList_in_classTypeDeclaration4730)
                    genericTypeParameterList23 = self.genericTypeParameterList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_genericTypeParameterList.add(genericTypeParameterList23.tree)



                # Java.g:257:47: ( classExtendsClause )?
                alt9 = 2
                LA9_0 = self.input.LA(1)

                if (LA9_0 == EXTENDS) :
                    alt9 = 1
                if alt9 == 1:
                    # Java.g:0:0: classExtendsClause
                    pass 
                    self._state.following.append(self.FOLLOW_classExtendsClause_in_classTypeDeclaration4733)
                    classExtendsClause24 = self.classExtendsClause()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_classExtendsClause.add(classExtendsClause24.tree)



                # Java.g:257:67: ( implementsClause )?
                alt10 = 2
                LA10_0 = self.input.LA(1)

                if (LA10_0 == IMPLEMENTS) :
                    alt10 = 1
                if alt10 == 1:
                    # Java.g:0:0: implementsClause
                    pass 
                    self._state.following.append(self.FOLLOW_implementsClause_in_classTypeDeclaration4736)
                    implementsClause25 = self.implementsClause()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_implementsClause.add(implementsClause25.tree)



                self._state.following.append(self.FOLLOW_classBody_in_classTypeDeclaration4739)
                classBody26 = self.classBody()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_classBody.add(classBody26.tree)

                # AST Rewrite
                # elements: classExtendsClause, IDENT, genericTypeParameterList, implementsClause, classBody, CLASS
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 258:9: -> ^( CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody )
                    # Java.g:258:13: ^( CLASS IDENT ( genericTypeParameterList )? ( classExtendsClause )? ( implementsClause )? classBody )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_CLASS.nextNode(), root_1)

                    self._adaptor.addChild(root_1, modifiers)
                    self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                    # Java.g:258:40: ( genericTypeParameterList )?
                    if stream_genericTypeParameterList.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                    stream_genericTypeParameterList.reset();
                    # Java.g:258:66: ( classExtendsClause )?
                    if stream_classExtendsClause.hasNext():
                        self._adaptor.addChild(root_1, stream_classExtendsClause.nextTree())


                    stream_classExtendsClause.reset();
                    # Java.g:259:15: ( implementsClause )?
                    if stream_implementsClause.hasNext():
                        self._adaptor.addChild(root_1, stream_implementsClause.nextTree())


                    stream_implementsClause.reset();
                    self._adaptor.addChild(root_1, stream_classBody.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 7, classTypeDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classTypeDeclaration"

    class classExtendsClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classExtendsClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "classExtendsClause"
    # Java.g:262:1: classExtendsClause : EXTENDS type -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] type ) ;
    def classExtendsClause(self, ):

        retval = self.classExtendsClause_return()
        retval.start = self.input.LT(1)
        classExtendsClause_StartIndex = self.input.index()
        root_0 = None

        EXTENDS27 = None
        type28 = None


        EXTENDS27_tree = None
        stream_EXTENDS = RewriteRuleTokenStream(self._adaptor, "token EXTENDS")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 8):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:263:5: ( EXTENDS type -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] type ) )
                # Java.g:263:9: EXTENDS type
                pass 
                EXTENDS27=self.match(self.input, EXTENDS, self.FOLLOW_EXTENDS_in_classExtendsClause4802) 
                if self._state.backtracking == 0:
                    stream_EXTENDS.add(EXTENDS27)
                self._state.following.append(self.FOLLOW_type_in_classExtendsClause4804)
                type28 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_type.add(type28.tree)

                # AST Rewrite
                # elements: type
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 264:9: -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] type )
                    # Java.g:264:13: ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] type )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(EXTENDS_CLAUSE, EXTENDS27, "EXTENDS_CLAUSE"), root_1)

                    self._adaptor.addChild(root_1, stream_type.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 8, classExtendsClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classExtendsClause"

    class interfaceExtendsClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceExtendsClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceExtendsClause"
    # Java.g:267:1: interfaceExtendsClause : EXTENDS typeList -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] typeList ) ;
    def interfaceExtendsClause(self, ):

        retval = self.interfaceExtendsClause_return()
        retval.start = self.input.LT(1)
        interfaceExtendsClause_StartIndex = self.input.index()
        root_0 = None

        EXTENDS29 = None
        typeList30 = None


        EXTENDS29_tree = None
        stream_EXTENDS = RewriteRuleTokenStream(self._adaptor, "token EXTENDS")
        stream_typeList = RewriteRuleSubtreeStream(self._adaptor, "rule typeList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 9):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:268:5: ( EXTENDS typeList -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] typeList ) )
                # Java.g:268:9: EXTENDS typeList
                pass 
                EXTENDS29=self.match(self.input, EXTENDS, self.FOLLOW_EXTENDS_in_interfaceExtendsClause4841) 
                if self._state.backtracking == 0:
                    stream_EXTENDS.add(EXTENDS29)
                self._state.following.append(self.FOLLOW_typeList_in_interfaceExtendsClause4843)
                typeList30 = self.typeList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_typeList.add(typeList30.tree)

                # AST Rewrite
                # elements: typeList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 269:9: -> ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] typeList )
                    # Java.g:269:13: ^( EXTENDS_CLAUSE[$EXTENDS, \"EXTENDS_CLAUSE\"] typeList )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(EXTENDS_CLAUSE, EXTENDS29, "EXTENDS_CLAUSE"), root_1)

                    self._adaptor.addChild(root_1, stream_typeList.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 9, interfaceExtendsClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceExtendsClause"

    class implementsClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.implementsClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "implementsClause"
    # Java.g:272:1: implementsClause : IMPLEMENTS typeList -> ^( IMPLEMENTS_CLAUSE[$IMPLEMENTS, \"IMPLEMENTS_CLAUSE\"] typeList ) ;
    def implementsClause(self, ):

        retval = self.implementsClause_return()
        retval.start = self.input.LT(1)
        implementsClause_StartIndex = self.input.index()
        root_0 = None

        IMPLEMENTS31 = None
        typeList32 = None


        IMPLEMENTS31_tree = None
        stream_IMPLEMENTS = RewriteRuleTokenStream(self._adaptor, "token IMPLEMENTS")
        stream_typeList = RewriteRuleSubtreeStream(self._adaptor, "rule typeList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 10):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:273:5: ( IMPLEMENTS typeList -> ^( IMPLEMENTS_CLAUSE[$IMPLEMENTS, \"IMPLEMENTS_CLAUSE\"] typeList ) )
                # Java.g:273:9: IMPLEMENTS typeList
                pass 
                IMPLEMENTS31=self.match(self.input, IMPLEMENTS, self.FOLLOW_IMPLEMENTS_in_implementsClause4880) 
                if self._state.backtracking == 0:
                    stream_IMPLEMENTS.add(IMPLEMENTS31)
                self._state.following.append(self.FOLLOW_typeList_in_implementsClause4882)
                typeList32 = self.typeList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_typeList.add(typeList32.tree)

                # AST Rewrite
                # elements: typeList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 274:9: -> ^( IMPLEMENTS_CLAUSE[$IMPLEMENTS, \"IMPLEMENTS_CLAUSE\"] typeList )
                    # Java.g:274:13: ^( IMPLEMENTS_CLAUSE[$IMPLEMENTS, \"IMPLEMENTS_CLAUSE\"] typeList )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(IMPLEMENTS_CLAUSE, IMPLEMENTS31, "IMPLEMENTS_CLAUSE"), root_1)

                    self._adaptor.addChild(root_1, stream_typeList.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 10, implementsClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "implementsClause"

    class genericTypeParameterList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeParameterList_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeParameterList"
    # Java.g:277:1: genericTypeParameterList : LESS_THAN genericTypeParameter ( COMMA genericTypeParameter )* genericTypeListClosing -> ^( GENERIC_TYPE_PARAM_LIST[$LESS_THAN, \"GENERIC_TYPE_PARAM_LIST\"] ( genericTypeParameter )+ ) ;
    def genericTypeParameterList(self, ):

        retval = self.genericTypeParameterList_return()
        retval.start = self.input.LT(1)
        genericTypeParameterList_StartIndex = self.input.index()
        root_0 = None

        LESS_THAN33 = None
        COMMA35 = None
        genericTypeParameter34 = None

        genericTypeParameter36 = None

        genericTypeListClosing37 = None


        LESS_THAN33_tree = None
        COMMA35_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_LESS_THAN = RewriteRuleTokenStream(self._adaptor, "token LESS_THAN")
        stream_genericTypeParameter = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeParameter")
        stream_genericTypeListClosing = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeListClosing")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 11):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:278:5: ( LESS_THAN genericTypeParameter ( COMMA genericTypeParameter )* genericTypeListClosing -> ^( GENERIC_TYPE_PARAM_LIST[$LESS_THAN, \"GENERIC_TYPE_PARAM_LIST\"] ( genericTypeParameter )+ ) )
                # Java.g:278:9: LESS_THAN genericTypeParameter ( COMMA genericTypeParameter )* genericTypeListClosing
                pass 
                LESS_THAN33=self.match(self.input, LESS_THAN, self.FOLLOW_LESS_THAN_in_genericTypeParameterList4919) 
                if self._state.backtracking == 0:
                    stream_LESS_THAN.add(LESS_THAN33)
                self._state.following.append(self.FOLLOW_genericTypeParameter_in_genericTypeParameterList4921)
                genericTypeParameter34 = self.genericTypeParameter()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeParameter.add(genericTypeParameter34.tree)
                # Java.g:278:40: ( COMMA genericTypeParameter )*
                while True: #loop11
                    alt11 = 2
                    LA11_0 = self.input.LA(1)

                    if (LA11_0 == COMMA) :
                        alt11 = 1


                    if alt11 == 1:
                        # Java.g:278:41: COMMA genericTypeParameter
                        pass 
                        COMMA35=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_genericTypeParameterList4924) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA35)
                        self._state.following.append(self.FOLLOW_genericTypeParameter_in_genericTypeParameterList4926)
                        genericTypeParameter36 = self.genericTypeParameter()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericTypeParameter.add(genericTypeParameter36.tree)


                    else:
                        break #loop11
                self._state.following.append(self.FOLLOW_genericTypeListClosing_in_genericTypeParameterList4930)
                genericTypeListClosing37 = self.genericTypeListClosing()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeListClosing.add(genericTypeListClosing37.tree)

                # AST Rewrite
                # elements: genericTypeParameter
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 279:9: -> ^( GENERIC_TYPE_PARAM_LIST[$LESS_THAN, \"GENERIC_TYPE_PARAM_LIST\"] ( genericTypeParameter )+ )
                    # Java.g:279:13: ^( GENERIC_TYPE_PARAM_LIST[$LESS_THAN, \"GENERIC_TYPE_PARAM_LIST\"] ( genericTypeParameter )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(GENERIC_TYPE_PARAM_LIST, LESS_THAN33, "GENERIC_TYPE_PARAM_LIST"), root_1)

                    # Java.g:279:78: ( genericTypeParameter )+
                    if not (stream_genericTypeParameter.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_genericTypeParameter.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeParameter.nextTree())


                    stream_genericTypeParameter.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 11, genericTypeParameterList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeParameterList"

    class genericTypeListClosing_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeListClosing_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeListClosing"
    # Java.g:282:1: genericTypeListClosing : ( GREATER_THAN | SHIFT_RIGHT | BIT_SHIFT_RIGHT | );
    def genericTypeListClosing(self, ):

        retval = self.genericTypeListClosing_return()
        retval.start = self.input.LT(1)
        genericTypeListClosing_StartIndex = self.input.index()
        root_0 = None

        GREATER_THAN38 = None
        SHIFT_RIGHT39 = None
        BIT_SHIFT_RIGHT40 = None

        GREATER_THAN38_tree = None
        SHIFT_RIGHT39_tree = None
        BIT_SHIFT_RIGHT40_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 12):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:286:5: ( GREATER_THAN | SHIFT_RIGHT | BIT_SHIFT_RIGHT | )
                alt12 = 4
                LA12 = self.input.LA(1)
                if LA12 == GREATER_THAN:
                    LA12_1 = self.input.LA(2)

                    if (self.synpred14_Java()) :
                        alt12 = 1
                    elif (True) :
                        alt12 = 4
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 12, 1, self.input)

                        raise nvae

                elif LA12 == SHIFT_RIGHT:
                    LA12_2 = self.input.LA(2)

                    if (self.synpred15_Java()) :
                        alt12 = 2
                    elif (True) :
                        alt12 = 4
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 12, 2, self.input)

                        raise nvae

                elif LA12 == BIT_SHIFT_RIGHT:
                    LA12_3 = self.input.LA(2)

                    if (self.synpred16_Java()) :
                        alt12 = 3
                    elif (True) :
                        alt12 = 4
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 12, 3, self.input)

                        raise nvae

                elif LA12 == EOF or LA12 == AND or LA12 == AND_ASSIGN or LA12 == ASSIGN or LA12 == BIT_SHIFT_RIGHT_ASSIGN or LA12 == COLON or LA12 == COMMA or LA12 == DIV_ASSIGN or LA12 == DOT or LA12 == ELLIPSIS or LA12 == EQUAL or LA12 == LBRACK or LA12 == LCURLY or LA12 == LOGICAL_AND or LA12 == LOGICAL_OR or LA12 == LPAREN or LA12 == MINUS_ASSIGN or LA12 == MOD_ASSIGN or LA12 == NOT_EQUAL or LA12 == OR or LA12 == OR_ASSIGN or LA12 == PLUS_ASSIGN or LA12 == QUESTION or LA12 == RBRACK or LA12 == RCURLY or LA12 == RPAREN or LA12 == SEMI or LA12 == SHIFT_LEFT_ASSIGN or LA12 == SHIFT_RIGHT_ASSIGN or LA12 == STAR_ASSIGN or LA12 == XOR or LA12 == XOR_ASSIGN or LA12 == BOOLEAN or LA12 == BYTE or LA12 == CHAR or LA12 == DOUBLE or LA12 == EXTENDS or LA12 == FLOAT or LA12 == IMPLEMENTS or LA12 == INT or LA12 == LONG or LA12 == SHORT or LA12 == SUPER or LA12 == THIS or LA12 == VOID or LA12 == IDENT:
                    alt12 = 4
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 12, 0, self.input)

                    raise nvae

                if alt12 == 1:
                    # Java.g:286:9: GREATER_THAN
                    pass 
                    root_0 = self._adaptor.nil()

                    GREATER_THAN38=self.match(self.input, GREATER_THAN, self.FOLLOW_GREATER_THAN_in_genericTypeListClosing5045)
                    if self._state.backtracking == 0:

                        GREATER_THAN38_tree = self._adaptor.createWithPayload(GREATER_THAN38)
                        self._adaptor.addChild(root_0, GREATER_THAN38_tree)



                elif alt12 == 2:
                    # Java.g:287:9: SHIFT_RIGHT
                    pass 
                    root_0 = self._adaptor.nil()

                    SHIFT_RIGHT39=self.match(self.input, SHIFT_RIGHT, self.FOLLOW_SHIFT_RIGHT_in_genericTypeListClosing5055)
                    if self._state.backtracking == 0:

                        SHIFT_RIGHT39_tree = self._adaptor.createWithPayload(SHIFT_RIGHT39)
                        self._adaptor.addChild(root_0, SHIFT_RIGHT39_tree)



                elif alt12 == 3:
                    # Java.g:288:9: BIT_SHIFT_RIGHT
                    pass 
                    root_0 = self._adaptor.nil()

                    BIT_SHIFT_RIGHT40=self.match(self.input, BIT_SHIFT_RIGHT, self.FOLLOW_BIT_SHIFT_RIGHT_in_genericTypeListClosing5065)
                    if self._state.backtracking == 0:

                        BIT_SHIFT_RIGHT40_tree = self._adaptor.createWithPayload(BIT_SHIFT_RIGHT40)
                        self._adaptor.addChild(root_0, BIT_SHIFT_RIGHT40_tree)



                elif alt12 == 4:
                    # Java.g:290:5: 
                    pass 
                    root_0 = self._adaptor.nil()


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 12, genericTypeListClosing_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeListClosing"

    class genericTypeParameter_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeParameter_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeParameter"
    # Java.g:292:1: genericTypeParameter : IDENT ( bound )? -> ^( IDENT ( bound )? ) ;
    def genericTypeParameter(self, ):

        retval = self.genericTypeParameter_return()
        retval.start = self.input.LT(1)
        genericTypeParameter_StartIndex = self.input.index()
        root_0 = None

        IDENT41 = None
        bound42 = None


        IDENT41_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_bound = RewriteRuleSubtreeStream(self._adaptor, "rule bound")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 13):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:293:5: ( IDENT ( bound )? -> ^( IDENT ( bound )? ) )
                # Java.g:293:9: IDENT ( bound )?
                pass 
                IDENT41=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_genericTypeParameter5093) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT41)
                # Java.g:293:15: ( bound )?
                alt13 = 2
                LA13_0 = self.input.LA(1)

                if (LA13_0 == EXTENDS) :
                    LA13_1 = self.input.LA(2)

                    if (LA13_1 == BOOLEAN or LA13_1 == BYTE or LA13_1 == CHAR or LA13_1 == DOUBLE or LA13_1 == FLOAT or (INT <= LA13_1 <= LONG) or LA13_1 == SHORT) :
                        LA13_3 = self.input.LA(3)

                        if (self.synpred17_Java()) :
                            alt13 = 1
                    elif (LA13_1 == IDENT) :
                        LA13_4 = self.input.LA(3)

                        if (self.synpred17_Java()) :
                            alt13 = 1
                if alt13 == 1:
                    # Java.g:0:0: bound
                    pass 
                    self._state.following.append(self.FOLLOW_bound_in_genericTypeParameter5095)
                    bound42 = self.bound()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_bound.add(bound42.tree)




                # AST Rewrite
                # elements: IDENT, bound
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 294:9: -> ^( IDENT ( bound )? )
                    # Java.g:294:13: ^( IDENT ( bound )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_IDENT.nextNode(), root_1)

                    # Java.g:294:21: ( bound )?
                    if stream_bound.hasNext():
                        self._adaptor.addChild(root_1, stream_bound.nextTree())


                    stream_bound.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 13, genericTypeParameter_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeParameter"

    class bound_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.bound_return, self).__init__()

            self.tree = None




    # $ANTLR start "bound"
    # Java.g:297:1: bound : EXTENDS type ( AND type )* -> ^( EXTENDS_BOUND_LIST[$EXTENDS, \"EXTENDS_BOUND_LIST\"] ( type )+ ) ;
    def bound(self, ):

        retval = self.bound_return()
        retval.start = self.input.LT(1)
        bound_StartIndex = self.input.index()
        root_0 = None

        EXTENDS43 = None
        AND45 = None
        type44 = None

        type46 = None


        EXTENDS43_tree = None
        AND45_tree = None
        stream_AND = RewriteRuleTokenStream(self._adaptor, "token AND")
        stream_EXTENDS = RewriteRuleTokenStream(self._adaptor, "token EXTENDS")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 14):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:298:5: ( EXTENDS type ( AND type )* -> ^( EXTENDS_BOUND_LIST[$EXTENDS, \"EXTENDS_BOUND_LIST\"] ( type )+ ) )
                # Java.g:298:9: EXTENDS type ( AND type )*
                pass 
                EXTENDS43=self.match(self.input, EXTENDS, self.FOLLOW_EXTENDS_in_bound5133) 
                if self._state.backtracking == 0:
                    stream_EXTENDS.add(EXTENDS43)
                self._state.following.append(self.FOLLOW_type_in_bound5135)
                type44 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_type.add(type44.tree)
                # Java.g:298:22: ( AND type )*
                while True: #loop14
                    alt14 = 2
                    LA14_0 = self.input.LA(1)

                    if (LA14_0 == AND) :
                        alt14 = 1


                    if alt14 == 1:
                        # Java.g:298:23: AND type
                        pass 
                        AND45=self.match(self.input, AND, self.FOLLOW_AND_in_bound5138) 
                        if self._state.backtracking == 0:
                            stream_AND.add(AND45)
                        self._state.following.append(self.FOLLOW_type_in_bound5140)
                        type46 = self.type()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_type.add(type46.tree)


                    else:
                        break #loop14

                # AST Rewrite
                # elements: type
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 299:9: -> ^( EXTENDS_BOUND_LIST[$EXTENDS, \"EXTENDS_BOUND_LIST\"] ( type )+ )
                    # Java.g:299:13: ^( EXTENDS_BOUND_LIST[$EXTENDS, \"EXTENDS_BOUND_LIST\"] ( type )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(EXTENDS_BOUND_LIST, EXTENDS43, "EXTENDS_BOUND_LIST"), root_1)

                    # Java.g:299:66: ( type )+
                    if not (stream_type.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_type.hasNext():
                        self._adaptor.addChild(root_1, stream_type.nextTree())


                    stream_type.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 14, bound_StartIndex, success)

            pass
        return retval

    # $ANTLR end "bound"

    class enumTypeDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumTypeDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumTypeDeclaration"
    # Java.g:302:1: enumTypeDeclaration[modifiers] : ENUM IDENT ( implementsClause )? enumBody -> ^( ENUM IDENT ( implementsClause )? enumBody ) ;
    def enumTypeDeclaration(self, modifiers):

        retval = self.enumTypeDeclaration_return()
        retval.start = self.input.LT(1)
        enumTypeDeclaration_StartIndex = self.input.index()
        root_0 = None

        ENUM47 = None
        IDENT48 = None
        implementsClause49 = None

        enumBody50 = None


        ENUM47_tree = None
        IDENT48_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_ENUM = RewriteRuleTokenStream(self._adaptor, "token ENUM")
        stream_implementsClause = RewriteRuleSubtreeStream(self._adaptor, "rule implementsClause")
        stream_enumBody = RewriteRuleSubtreeStream(self._adaptor, "rule enumBody")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 15):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:303:5: ( ENUM IDENT ( implementsClause )? enumBody -> ^( ENUM IDENT ( implementsClause )? enumBody ) )
                # Java.g:303:9: ENUM IDENT ( implementsClause )? enumBody
                pass 
                ENUM47=self.match(self.input, ENUM, self.FOLLOW_ENUM_in_enumTypeDeclaration5181) 
                if self._state.backtracking == 0:
                    stream_ENUM.add(ENUM47)
                IDENT48=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_enumTypeDeclaration5183) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT48)
                # Java.g:303:20: ( implementsClause )?
                alt15 = 2
                LA15_0 = self.input.LA(1)

                if (LA15_0 == IMPLEMENTS) :
                    alt15 = 1
                if alt15 == 1:
                    # Java.g:0:0: implementsClause
                    pass 
                    self._state.following.append(self.FOLLOW_implementsClause_in_enumTypeDeclaration5185)
                    implementsClause49 = self.implementsClause()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_implementsClause.add(implementsClause49.tree)



                self._state.following.append(self.FOLLOW_enumBody_in_enumTypeDeclaration5188)
                enumBody50 = self.enumBody()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_enumBody.add(enumBody50.tree)

                # AST Rewrite
                # elements: ENUM, IDENT, enumBody, implementsClause
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 304:9: -> ^( ENUM IDENT ( implementsClause )? enumBody )
                    # Java.g:304:13: ^( ENUM IDENT ( implementsClause )? enumBody )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_ENUM.nextNode(), root_1)

                    self._adaptor.addChild(root_1, modifiers)
                    self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                    # Java.g:304:39: ( implementsClause )?
                    if stream_implementsClause.hasNext():
                        self._adaptor.addChild(root_1, stream_implementsClause.nextTree())


                    stream_implementsClause.reset();
                    self._adaptor.addChild(root_1, stream_enumBody.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 15, enumTypeDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumTypeDeclaration"

    class enumBody_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumBody_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumBody"
    # Java.g:307:1: enumBody : LCURLY enumScopeDeclarations RCURLY -> ^( ENUM_TOP_LEVEL_SCOPE[$LCURLY, \"ENUM_TOP_LEVEL_SCOPE\"] enumScopeDeclarations ) ;
    def enumBody(self, ):

        retval = self.enumBody_return()
        retval.start = self.input.LT(1)
        enumBody_StartIndex = self.input.index()
        root_0 = None

        LCURLY51 = None
        RCURLY53 = None
        enumScopeDeclarations52 = None


        LCURLY51_tree = None
        RCURLY53_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_enumScopeDeclarations = RewriteRuleSubtreeStream(self._adaptor, "rule enumScopeDeclarations")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 16):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:308:5: ( LCURLY enumScopeDeclarations RCURLY -> ^( ENUM_TOP_LEVEL_SCOPE[$LCURLY, \"ENUM_TOP_LEVEL_SCOPE\"] enumScopeDeclarations ) )
                # Java.g:308:9: LCURLY enumScopeDeclarations RCURLY
                pass 
                LCURLY51=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_enumBody5231) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY51)
                self._state.following.append(self.FOLLOW_enumScopeDeclarations_in_enumBody5233)
                enumScopeDeclarations52 = self.enumScopeDeclarations()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_enumScopeDeclarations.add(enumScopeDeclarations52.tree)
                RCURLY53=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_enumBody5235) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY53)

                # AST Rewrite
                # elements: enumScopeDeclarations
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 309:9: -> ^( ENUM_TOP_LEVEL_SCOPE[$LCURLY, \"ENUM_TOP_LEVEL_SCOPE\"] enumScopeDeclarations )
                    # Java.g:309:13: ^( ENUM_TOP_LEVEL_SCOPE[$LCURLY, \"ENUM_TOP_LEVEL_SCOPE\"] enumScopeDeclarations )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ENUM_TOP_LEVEL_SCOPE, LCURLY51, "ENUM_TOP_LEVEL_SCOPE"), root_1)

                    self._adaptor.addChild(root_1, stream_enumScopeDeclarations.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 16, enumBody_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumBody"

    class enumScopeDeclarations_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumScopeDeclarations_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumScopeDeclarations"
    # Java.g:312:1: enumScopeDeclarations : enumConstants ( COMMA )? ( enumClassScopeDeclarations )? ;
    def enumScopeDeclarations(self, ):

        retval = self.enumScopeDeclarations_return()
        retval.start = self.input.LT(1)
        enumScopeDeclarations_StartIndex = self.input.index()
        root_0 = None

        COMMA55 = None
        enumConstants54 = None

        enumClassScopeDeclarations56 = None


        COMMA55_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 17):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:313:5: ( enumConstants ( COMMA )? ( enumClassScopeDeclarations )? )
                # Java.g:313:9: enumConstants ( COMMA )? ( enumClassScopeDeclarations )?
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_enumConstants_in_enumScopeDeclarations5272)
                enumConstants54 = self.enumConstants()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, enumConstants54.tree)
                # Java.g:313:23: ( COMMA )?
                alt16 = 2
                LA16_0 = self.input.LA(1)

                if (LA16_0 == COMMA) :
                    alt16 = 1
                if alt16 == 1:
                    # Java.g:313:24: COMMA
                    pass 
                    COMMA55=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_enumScopeDeclarations5275)



                # Java.g:313:33: ( enumClassScopeDeclarations )?
                alt17 = 2
                LA17_0 = self.input.LA(1)

                if (LA17_0 == SEMI) :
                    alt17 = 1
                if alt17 == 1:
                    # Java.g:0:0: enumClassScopeDeclarations
                    pass 
                    self._state.following.append(self.FOLLOW_enumClassScopeDeclarations_in_enumScopeDeclarations5280)
                    enumClassScopeDeclarations56 = self.enumClassScopeDeclarations()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, enumClassScopeDeclarations56.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 17, enumScopeDeclarations_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumScopeDeclarations"

    class enumClassScopeDeclarations_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumClassScopeDeclarations_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumClassScopeDeclarations"
    # Java.g:316:1: enumClassScopeDeclarations : SEMI ( classScopeDeclarations )* -> ^( CLASS_TOP_LEVEL_SCOPE[$SEMI, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* ) ;
    def enumClassScopeDeclarations(self, ):

        retval = self.enumClassScopeDeclarations_return()
        retval.start = self.input.LT(1)
        enumClassScopeDeclarations_StartIndex = self.input.index()
        root_0 = None

        SEMI57 = None
        classScopeDeclarations58 = None


        SEMI57_tree = None
        stream_SEMI = RewriteRuleTokenStream(self._adaptor, "token SEMI")
        stream_classScopeDeclarations = RewriteRuleSubtreeStream(self._adaptor, "rule classScopeDeclarations")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 18):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:317:5: ( SEMI ( classScopeDeclarations )* -> ^( CLASS_TOP_LEVEL_SCOPE[$SEMI, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* ) )
                # Java.g:317:9: SEMI ( classScopeDeclarations )*
                pass 
                SEMI57=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_enumClassScopeDeclarations5300) 
                if self._state.backtracking == 0:
                    stream_SEMI.add(SEMI57)
                # Java.g:317:14: ( classScopeDeclarations )*
                while True: #loop18
                    alt18 = 2
                    LA18_0 = self.input.LA(1)

                    if (LA18_0 == AT or LA18_0 == LCURLY or LA18_0 == LESS_THAN or LA18_0 == SEMI or LA18_0 == ABSTRACT or LA18_0 == BOOLEAN or LA18_0 == BYTE or (CHAR <= LA18_0 <= CLASS) or LA18_0 == DOUBLE or LA18_0 == ENUM or LA18_0 == FINAL or LA18_0 == FLOAT or LA18_0 == INTERFACE or (INT <= LA18_0 <= NATIVE) or (PRIVATE <= LA18_0 <= PUBLIC) or (SHORT <= LA18_0 <= STRICTFP) or LA18_0 == SYNCHRONIZED or LA18_0 == TRANSIENT or (VOID <= LA18_0 <= VOLATILE) or LA18_0 == IDENT) :
                        alt18 = 1


                    if alt18 == 1:
                        # Java.g:0:0: classScopeDeclarations
                        pass 
                        self._state.following.append(self.FOLLOW_classScopeDeclarations_in_enumClassScopeDeclarations5302)
                        classScopeDeclarations58 = self.classScopeDeclarations()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_classScopeDeclarations.add(classScopeDeclarations58.tree)


                    else:
                        break #loop18

                # AST Rewrite
                # elements: classScopeDeclarations
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 318:9: -> ^( CLASS_TOP_LEVEL_SCOPE[$SEMI, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* )
                    # Java.g:318:13: ^( CLASS_TOP_LEVEL_SCOPE[$SEMI, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(CLASS_TOP_LEVEL_SCOPE, SEMI57, "CLASS_TOP_LEVEL_SCOPE"), root_1)

                    # Java.g:318:69: ( classScopeDeclarations )*
                    while stream_classScopeDeclarations.hasNext():
                        self._adaptor.addChild(root_1, stream_classScopeDeclarations.nextTree())


                    stream_classScopeDeclarations.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 18, enumClassScopeDeclarations_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumClassScopeDeclarations"

    class enumConstants_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumConstants_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumConstants"
    # Java.g:321:1: enumConstants : enumConstant ( COMMA enumConstant )* ;
    def enumConstants(self, ):

        retval = self.enumConstants_return()
        retval.start = self.input.LT(1)
        enumConstants_StartIndex = self.input.index()
        root_0 = None

        COMMA60 = None
        enumConstant59 = None

        enumConstant61 = None


        COMMA60_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 19):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:322:5: ( enumConstant ( COMMA enumConstant )* )
                # Java.g:322:9: enumConstant ( COMMA enumConstant )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_enumConstant_in_enumConstants5341)
                enumConstant59 = self.enumConstant()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, enumConstant59.tree)
                # Java.g:322:22: ( COMMA enumConstant )*
                while True: #loop19
                    alt19 = 2
                    LA19_0 = self.input.LA(1)

                    if (LA19_0 == COMMA) :
                        LA19_1 = self.input.LA(2)

                        if (LA19_1 == AT or LA19_1 == IDENT) :
                            alt19 = 1




                    if alt19 == 1:
                        # Java.g:322:23: COMMA enumConstant
                        pass 
                        COMMA60=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_enumConstants5344)
                        self._state.following.append(self.FOLLOW_enumConstant_in_enumConstants5347)
                        enumConstant61 = self.enumConstant()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, enumConstant61.tree)


                    else:
                        break #loop19



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 19, enumConstants_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumConstants"

    class enumConstant_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.enumConstant_return, self).__init__()

            self.tree = None




    # $ANTLR start "enumConstant"
    # Java.g:325:1: enumConstant : annotationList IDENT ( arguments )? ( classBody )? ;
    def enumConstant(self, ):

        retval = self.enumConstant_return()
        retval.start = self.input.LT(1)
        enumConstant_StartIndex = self.input.index()
        root_0 = None

        IDENT63 = None
        annotationList62 = None

        arguments64 = None

        classBody65 = None


        IDENT63_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 20):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:326:5: ( annotationList IDENT ( arguments )? ( classBody )? )
                # Java.g:326:9: annotationList IDENT ( arguments )? ( classBody )?
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_annotationList_in_enumConstant5368)
                annotationList62 = self.annotationList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, annotationList62.tree)
                IDENT63=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_enumConstant5370)
                if self._state.backtracking == 0:

                    IDENT63_tree = self._adaptor.createWithPayload(IDENT63)
                    root_0 = self._adaptor.becomeRoot(IDENT63_tree, root_0)

                # Java.g:326:31: ( arguments )?
                alt20 = 2
                LA20_0 = self.input.LA(1)

                if (LA20_0 == LPAREN) :
                    alt20 = 1
                if alt20 == 1:
                    # Java.g:0:0: arguments
                    pass 
                    self._state.following.append(self.FOLLOW_arguments_in_enumConstant5373)
                    arguments64 = self.arguments()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, arguments64.tree)



                # Java.g:326:42: ( classBody )?
                alt21 = 2
                LA21_0 = self.input.LA(1)

                if (LA21_0 == LCURLY) :
                    alt21 = 1
                if alt21 == 1:
                    # Java.g:0:0: classBody
                    pass 
                    self._state.following.append(self.FOLLOW_classBody_in_enumConstant5376)
                    classBody65 = self.classBody()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, classBody65.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 20, enumConstant_StartIndex, success)

            pass
        return retval

    # $ANTLR end "enumConstant"

    class interfaceTypeDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceTypeDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceTypeDeclaration"
    # Java.g:329:1: interfaceTypeDeclaration[modifiers] : INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody -> ^( INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody ) ;
    def interfaceTypeDeclaration(self, modifiers):

        retval = self.interfaceTypeDeclaration_return()
        retval.start = self.input.LT(1)
        interfaceTypeDeclaration_StartIndex = self.input.index()
        root_0 = None

        INTERFACE66 = None
        IDENT67 = None
        genericTypeParameterList68 = None

        interfaceExtendsClause69 = None

        interfaceBody70 = None


        INTERFACE66_tree = None
        IDENT67_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_INTERFACE = RewriteRuleTokenStream(self._adaptor, "token INTERFACE")
        stream_genericTypeParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeParameterList")
        stream_interfaceBody = RewriteRuleSubtreeStream(self._adaptor, "rule interfaceBody")
        stream_interfaceExtendsClause = RewriteRuleSubtreeStream(self._adaptor, "rule interfaceExtendsClause")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 21):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:330:5: ( INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody -> ^( INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody ) )
                # Java.g:330:9: INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody
                pass 
                INTERFACE66=self.match(self.input, INTERFACE, self.FOLLOW_INTERFACE_in_interfaceTypeDeclaration5397) 
                if self._state.backtracking == 0:
                    stream_INTERFACE.add(INTERFACE66)
                IDENT67=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_interfaceTypeDeclaration5399) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT67)
                # Java.g:330:25: ( genericTypeParameterList )?
                alt22 = 2
                LA22_0 = self.input.LA(1)

                if (LA22_0 == LESS_THAN) :
                    alt22 = 1
                if alt22 == 1:
                    # Java.g:0:0: genericTypeParameterList
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeParameterList_in_interfaceTypeDeclaration5401)
                    genericTypeParameterList68 = self.genericTypeParameterList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_genericTypeParameterList.add(genericTypeParameterList68.tree)



                # Java.g:330:51: ( interfaceExtendsClause )?
                alt23 = 2
                LA23_0 = self.input.LA(1)

                if (LA23_0 == EXTENDS) :
                    alt23 = 1
                if alt23 == 1:
                    # Java.g:0:0: interfaceExtendsClause
                    pass 
                    self._state.following.append(self.FOLLOW_interfaceExtendsClause_in_interfaceTypeDeclaration5404)
                    interfaceExtendsClause69 = self.interfaceExtendsClause()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_interfaceExtendsClause.add(interfaceExtendsClause69.tree)



                self._state.following.append(self.FOLLOW_interfaceBody_in_interfaceTypeDeclaration5407)
                interfaceBody70 = self.interfaceBody()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_interfaceBody.add(interfaceBody70.tree)

                # AST Rewrite
                # elements: INTERFACE, genericTypeParameterList, IDENT, interfaceExtendsClause, interfaceBody
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 331:9: -> ^( INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody )
                    # Java.g:331:13: ^( INTERFACE IDENT ( genericTypeParameterList )? ( interfaceExtendsClause )? interfaceBody )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_INTERFACE.nextNode(), root_1)

                    self._adaptor.addChild(root_1, modifiers)
                    self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                    # Java.g:331:44: ( genericTypeParameterList )?
                    if stream_genericTypeParameterList.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                    stream_genericTypeParameterList.reset();
                    # Java.g:331:70: ( interfaceExtendsClause )?
                    if stream_interfaceExtendsClause.hasNext():
                        self._adaptor.addChild(root_1, stream_interfaceExtendsClause.nextTree())


                    stream_interfaceExtendsClause.reset();
                    self._adaptor.addChild(root_1, stream_interfaceBody.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 21, interfaceTypeDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceTypeDeclaration"

    class typeList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.typeList_return, self).__init__()

            self.tree = None




    # $ANTLR start "typeList"
    # Java.g:334:1: typeList : type ( COMMA type )* ;
    def typeList(self, ):

        retval = self.typeList_return()
        retval.start = self.input.LT(1)
        typeList_StartIndex = self.input.index()
        root_0 = None

        COMMA72 = None
        type71 = None

        type73 = None


        COMMA72_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 22):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:335:5: ( type ( COMMA type )* )
                # Java.g:335:9: type ( COMMA type )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_type_in_typeList5453)
                type71 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, type71.tree)
                # Java.g:335:14: ( COMMA type )*
                while True: #loop24
                    alt24 = 2
                    LA24_0 = self.input.LA(1)

                    if (LA24_0 == COMMA) :
                        alt24 = 1


                    if alt24 == 1:
                        # Java.g:335:15: COMMA type
                        pass 
                        COMMA72=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_typeList5456)
                        self._state.following.append(self.FOLLOW_type_in_typeList5459)
                        type73 = self.type()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, type73.tree)


                    else:
                        break #loop24



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 22, typeList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "typeList"

    class classBody_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classBody_return, self).__init__()

            self.tree = None




    # $ANTLR start "classBody"
    # Java.g:338:1: classBody : LCURLY ( classScopeDeclarations )* RCURLY -> ^( CLASS_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* ) ;
    def classBody(self, ):

        retval = self.classBody_return()
        retval.start = self.input.LT(1)
        classBody_StartIndex = self.input.index()
        root_0 = None

        LCURLY74 = None
        RCURLY76 = None
        classScopeDeclarations75 = None


        LCURLY74_tree = None
        RCURLY76_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_classScopeDeclarations = RewriteRuleSubtreeStream(self._adaptor, "rule classScopeDeclarations")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 23):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:339:5: ( LCURLY ( classScopeDeclarations )* RCURLY -> ^( CLASS_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* ) )
                # Java.g:339:9: LCURLY ( classScopeDeclarations )* RCURLY
                pass 
                LCURLY74=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_classBody5480) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY74)
                # Java.g:339:16: ( classScopeDeclarations )*
                while True: #loop25
                    alt25 = 2
                    LA25_0 = self.input.LA(1)

                    if (LA25_0 == AT or LA25_0 == LCURLY or LA25_0 == LESS_THAN or LA25_0 == SEMI or LA25_0 == ABSTRACT or LA25_0 == BOOLEAN or LA25_0 == BYTE or (CHAR <= LA25_0 <= CLASS) or LA25_0 == DOUBLE or LA25_0 == ENUM or LA25_0 == FINAL or LA25_0 == FLOAT or LA25_0 == INTERFACE or (INT <= LA25_0 <= NATIVE) or (PRIVATE <= LA25_0 <= PUBLIC) or (SHORT <= LA25_0 <= STRICTFP) or LA25_0 == SYNCHRONIZED or LA25_0 == TRANSIENT or (VOID <= LA25_0 <= VOLATILE) or LA25_0 == IDENT) :
                        alt25 = 1


                    if alt25 == 1:
                        # Java.g:0:0: classScopeDeclarations
                        pass 
                        self._state.following.append(self.FOLLOW_classScopeDeclarations_in_classBody5482)
                        classScopeDeclarations75 = self.classScopeDeclarations()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_classScopeDeclarations.add(classScopeDeclarations75.tree)


                    else:
                        break #loop25
                RCURLY76=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_classBody5485) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY76)

                # AST Rewrite
                # elements: classScopeDeclarations
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 340:9: -> ^( CLASS_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* )
                    # Java.g:340:13: ^( CLASS_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( classScopeDeclarations )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(CLASS_TOP_LEVEL_SCOPE, LCURLY74, "CLASS_TOP_LEVEL_SCOPE"), root_1)

                    # Java.g:340:71: ( classScopeDeclarations )*
                    while stream_classScopeDeclarations.hasNext():
                        self._adaptor.addChild(root_1, stream_classScopeDeclarations.nextTree())


                    stream_classScopeDeclarations.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 23, classBody_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classBody"

    class interfaceBody_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceBody_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceBody"
    # Java.g:343:1: interfaceBody : LCURLY ( interfaceScopeDeclarations )* RCURLY -> ^( INTERFACE_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( interfaceScopeDeclarations )* ) ;
    def interfaceBody(self, ):

        retval = self.interfaceBody_return()
        retval.start = self.input.LT(1)
        interfaceBody_StartIndex = self.input.index()
        root_0 = None

        LCURLY77 = None
        RCURLY79 = None
        interfaceScopeDeclarations78 = None


        LCURLY77_tree = None
        RCURLY79_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_interfaceScopeDeclarations = RewriteRuleSubtreeStream(self._adaptor, "rule interfaceScopeDeclarations")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 24):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:344:5: ( LCURLY ( interfaceScopeDeclarations )* RCURLY -> ^( INTERFACE_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( interfaceScopeDeclarations )* ) )
                # Java.g:344:9: LCURLY ( interfaceScopeDeclarations )* RCURLY
                pass 
                LCURLY77=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_interfaceBody5523) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY77)
                # Java.g:344:16: ( interfaceScopeDeclarations )*
                while True: #loop26
                    alt26 = 2
                    LA26_0 = self.input.LA(1)

                    if (LA26_0 == AT or LA26_0 == LESS_THAN or LA26_0 == SEMI or LA26_0 == ABSTRACT or LA26_0 == BOOLEAN or LA26_0 == BYTE or (CHAR <= LA26_0 <= CLASS) or LA26_0 == DOUBLE or LA26_0 == ENUM or LA26_0 == FINAL or LA26_0 == FLOAT or LA26_0 == INTERFACE or (INT <= LA26_0 <= NATIVE) or (PRIVATE <= LA26_0 <= PUBLIC) or (SHORT <= LA26_0 <= STRICTFP) or LA26_0 == SYNCHRONIZED or LA26_0 == TRANSIENT or (VOID <= LA26_0 <= VOLATILE) or LA26_0 == IDENT) :
                        alt26 = 1


                    if alt26 == 1:
                        # Java.g:0:0: interfaceScopeDeclarations
                        pass 
                        self._state.following.append(self.FOLLOW_interfaceScopeDeclarations_in_interfaceBody5525)
                        interfaceScopeDeclarations78 = self.interfaceScopeDeclarations()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_interfaceScopeDeclarations.add(interfaceScopeDeclarations78.tree)


                    else:
                        break #loop26
                RCURLY79=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_interfaceBody5528) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY79)

                # AST Rewrite
                # elements: interfaceScopeDeclarations
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 345:9: -> ^( INTERFACE_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( interfaceScopeDeclarations )* )
                    # Java.g:345:13: ^( INTERFACE_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( interfaceScopeDeclarations )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(INTERFACE_TOP_LEVEL_SCOPE, LCURLY77, "CLASS_TOP_LEVEL_SCOPE"), root_1)

                    # Java.g:345:75: ( interfaceScopeDeclarations )*
                    while stream_interfaceScopeDeclarations.hasNext():
                        self._adaptor.addChild(root_1, stream_interfaceScopeDeclarations.nextTree())


                    stream_interfaceScopeDeclarations.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 24, interfaceBody_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceBody"

    class classScopeDeclarations_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classScopeDeclarations_return, self).__init__()

            self.tree = None




    # $ANTLR start "classScopeDeclarations"
    # Java.g:348:1: classScopeDeclarations : ( block -> ^( CLASS_INSTANCE_INITIALIZER block ) | STATIC block -> ^( CLASS_STATIC_INITIALIZER[$STATIC, \"CLASS_STATIC_INITIALIZER\"] block ) | modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) ) | type classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) ) | typeDeclaration | SEMI );
    def classScopeDeclarations(self, ):

        retval = self.classScopeDeclarations_return()
        retval.start = self.input.LT(1)
        classScopeDeclarations_StartIndex = self.input.index()
        root_0 = None

        ident = None
        STATIC81 = None
        IDENT86 = None
        SEMI91 = None
        VOID92 = None
        IDENT93 = None
        SEMI97 = None
        SEMI103 = None
        SEMI105 = None
        block80 = None

        block82 = None

        modifierList83 = None

        genericTypeParameterList84 = None

        type85 = None

        formalParameterList87 = None

        arrayDeclaratorList88 = None

        throwsClause89 = None

        block90 = None

        formalParameterList94 = None

        throwsClause95 = None

        block96 = None

        formalParameterList98 = None

        throwsClause99 = None

        block100 = None

        type101 = None

        classFieldDeclaratorList102 = None

        typeDeclaration104 = None


        ident_tree = None
        STATIC81_tree = None
        IDENT86_tree = None
        SEMI91_tree = None
        VOID92_tree = None
        IDENT93_tree = None
        SEMI97_tree = None
        SEMI103_tree = None
        SEMI105_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_VOID = RewriteRuleTokenStream(self._adaptor, "token VOID")
        stream_SEMI = RewriteRuleTokenStream(self._adaptor, "token SEMI")
        stream_STATIC = RewriteRuleTokenStream(self._adaptor, "token STATIC")
        stream_arrayDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclaratorList")
        stream_throwsClause = RewriteRuleSubtreeStream(self._adaptor, "rule throwsClause")
        stream_modifierList = RewriteRuleSubtreeStream(self._adaptor, "rule modifierList")
        stream_genericTypeParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeParameterList")
        stream_block = RewriteRuleSubtreeStream(self._adaptor, "rule block")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        stream_classFieldDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule classFieldDeclaratorList")
        stream_formalParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule formalParameterList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 25):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:349:5: ( block -> ^( CLASS_INSTANCE_INITIALIZER block ) | STATIC block -> ^( CLASS_STATIC_INITIALIZER[$STATIC, \"CLASS_STATIC_INITIALIZER\"] block ) | modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) ) | type classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) ) | typeDeclaration | SEMI )
                alt36 = 5
                alt36 = self.dfa36.predict(self.input)
                if alt36 == 1:
                    # Java.g:349:9: block
                    pass 
                    self._state.following.append(self.FOLLOW_block_in_classScopeDeclarations5566)
                    block80 = self.block()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_block.add(block80.tree)

                    # AST Rewrite
                    # elements: block
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 349:25: -> ^( CLASS_INSTANCE_INITIALIZER block )
                        # Java.g:349:29: ^( CLASS_INSTANCE_INITIALIZER block )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(CLASS_INSTANCE_INITIALIZER, "CLASS_INSTANCE_INITIALIZER"), root_1)

                        self._adaptor.addChild(root_1, stream_block.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt36 == 2:
                    # Java.g:350:9: STATIC block
                    pass 
                    STATIC81=self.match(self.input, STATIC, self.FOLLOW_STATIC_in_classScopeDeclarations5595) 
                    if self._state.backtracking == 0:
                        stream_STATIC.add(STATIC81)
                    self._state.following.append(self.FOLLOW_block_in_classScopeDeclarations5597)
                    block82 = self.block()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_block.add(block82.tree)

                    # AST Rewrite
                    # elements: block
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 350:25: -> ^( CLASS_STATIC_INITIALIZER[$STATIC, \"CLASS_STATIC_INITIALIZER\"] block )
                        # Java.g:350:29: ^( CLASS_STATIC_INITIALIZER[$STATIC, \"CLASS_STATIC_INITIALIZER\"] block )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(CLASS_STATIC_INITIALIZER, STATIC81, "CLASS_STATIC_INITIALIZER"), root_1)

                        self._adaptor.addChild(root_1, stream_block.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt36 == 3:
                    # Java.g:351:9: modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) ) | type classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) )
                    pass 
                    self._state.following.append(self.FOLLOW_modifierList_in_classScopeDeclarations5620)
                    modifierList83 = self.modifierList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_modifierList.add(modifierList83.tree)
                    # Java.g:352:9: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) ) | type classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) )
                    alt35 = 2
                    LA35 = self.input.LA(1)
                    if LA35 == LESS_THAN or LA35 == VOID:
                        alt35 = 1
                    elif LA35 == BOOLEAN or LA35 == BYTE or LA35 == CHAR or LA35 == DOUBLE or LA35 == FLOAT or LA35 == INT or LA35 == LONG or LA35 == SHORT:
                        LA35_2 = self.input.LA(2)

                        if (self.synpred42_Java()) :
                            alt35 = 1
                        elif (True) :
                            alt35 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 35, 2, self.input)

                            raise nvae

                    elif LA35 == IDENT:
                        LA35_3 = self.input.LA(2)

                        if (self.synpred42_Java()) :
                            alt35 = 1
                        elif (True) :
                            alt35 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 35, 3, self.input)

                            raise nvae

                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 35, 0, self.input)

                        raise nvae

                    if alt35 == 1:
                        # Java.g:352:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) )
                        pass 
                        # Java.g:352:13: ( genericTypeParameterList )?
                        alt27 = 2
                        LA27_0 = self.input.LA(1)

                        if (LA27_0 == LESS_THAN) :
                            alt27 = 1
                        if alt27 == 1:
                            # Java.g:0:0: genericTypeParameterList
                            pass 
                            self._state.following.append(self.FOLLOW_genericTypeParameterList_in_classScopeDeclarations5634)
                            genericTypeParameterList84 = self.genericTypeParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_genericTypeParameterList.add(genericTypeParameterList84.tree)



                        # Java.g:353:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? ) | ident= IDENT formalParameterList ( throwsClause )? block -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block ) )
                        alt34 = 3
                        LA34 = self.input.LA(1)
                        if LA34 == BOOLEAN or LA34 == BYTE or LA34 == CHAR or LA34 == DOUBLE or LA34 == FLOAT or LA34 == INT or LA34 == LONG or LA34 == SHORT:
                            alt34 = 1
                        elif LA34 == IDENT:
                            LA34_2 = self.input.LA(2)

                            if (LA34_2 == LPAREN) :
                                alt34 = 3
                            elif (LA34_2 == DOT or LA34_2 == LBRACK or LA34_2 == LESS_THAN or LA34_2 == IDENT) :
                                alt34 = 1
                            else:
                                if self._state.backtracking > 0:
                                    raise BacktrackingFailed

                                nvae = NoViableAltException("", 34, 2, self.input)

                                raise nvae

                        elif LA34 == VOID:
                            alt34 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 34, 0, self.input)

                            raise nvae

                        if alt34 == 1:
                            # Java.g:353:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI )
                            pass 
                            self._state.following.append(self.FOLLOW_type_in_classScopeDeclarations5653)
                            type85 = self.type()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_type.add(type85.tree)
                            IDENT86=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_classScopeDeclarations5655) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT86)
                            self._state.following.append(self.FOLLOW_formalParameterList_in_classScopeDeclarations5657)
                            formalParameterList87 = self.formalParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterList.add(formalParameterList87.tree)
                            # Java.g:353:48: ( arrayDeclaratorList )?
                            alt28 = 2
                            LA28_0 = self.input.LA(1)

                            if (LA28_0 == LBRACK) :
                                alt28 = 1
                            if alt28 == 1:
                                # Java.g:0:0: arrayDeclaratorList
                                pass 
                                self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_classScopeDeclarations5659)
                                arrayDeclaratorList88 = self.arrayDeclaratorList()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_arrayDeclaratorList.add(arrayDeclaratorList88.tree)



                            # Java.g:353:69: ( throwsClause )?
                            alt29 = 2
                            LA29_0 = self.input.LA(1)

                            if (LA29_0 == THROWS) :
                                alt29 = 1
                            if alt29 == 1:
                                # Java.g:0:0: throwsClause
                                pass 
                                self._state.following.append(self.FOLLOW_throwsClause_in_classScopeDeclarations5662)
                                throwsClause89 = self.throwsClause()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_throwsClause.add(throwsClause89.tree)



                            # Java.g:353:83: ( block | SEMI )
                            alt30 = 2
                            LA30_0 = self.input.LA(1)

                            if (LA30_0 == LCURLY) :
                                alt30 = 1
                            elif (LA30_0 == SEMI) :
                                alt30 = 2
                            else:
                                if self._state.backtracking > 0:
                                    raise BacktrackingFailed

                                nvae = NoViableAltException("", 30, 0, self.input)

                                raise nvae

                            if alt30 == 1:
                                # Java.g:353:84: block
                                pass 
                                self._state.following.append(self.FOLLOW_block_in_classScopeDeclarations5666)
                                block90 = self.block()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_block.add(block90.tree)


                            elif alt30 == 2:
                                # Java.g:353:92: SEMI
                                pass 
                                SEMI91=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_classScopeDeclarations5670) 
                                if self._state.backtracking == 0:
                                    stream_SEMI.add(SEMI91)




                            # AST Rewrite
                            # elements: arrayDeclaratorList, type, modifierList, block, throwsClause, formalParameterList, IDENT, genericTypeParameterList
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 354:17: -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? )
                                # Java.g:354:21: ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block )? )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL"), root_1)

                                self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                                # Java.g:354:57: ( genericTypeParameterList )?
                                if stream_genericTypeParameterList.hasNext():
                                    self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                                stream_genericTypeParameterList.reset();
                                self._adaptor.addChild(root_1, stream_type.nextTree())
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                                self._adaptor.addChild(root_1, stream_formalParameterList.nextTree())
                                # Java.g:354:114: ( arrayDeclaratorList )?
                                if stream_arrayDeclaratorList.hasNext():
                                    self._adaptor.addChild(root_1, stream_arrayDeclaratorList.nextTree())


                                stream_arrayDeclaratorList.reset();
                                # Java.g:354:135: ( throwsClause )?
                                if stream_throwsClause.hasNext():
                                    self._adaptor.addChild(root_1, stream_throwsClause.nextTree())


                                stream_throwsClause.reset();
                                # Java.g:354:149: ( block )?
                                if stream_block.hasNext():
                                    self._adaptor.addChild(root_1, stream_block.nextTree())


                                stream_block.reset();

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt34 == 2:
                            # Java.g:355:17: VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI )
                            pass 
                            VOID92=self.match(self.input, VOID, self.FOLLOW_VOID_in_classScopeDeclarations5732) 
                            if self._state.backtracking == 0:
                                stream_VOID.add(VOID92)
                            IDENT93=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_classScopeDeclarations5734) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT93)
                            self._state.following.append(self.FOLLOW_formalParameterList_in_classScopeDeclarations5736)
                            formalParameterList94 = self.formalParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterList.add(formalParameterList94.tree)
                            # Java.g:355:48: ( throwsClause )?
                            alt31 = 2
                            LA31_0 = self.input.LA(1)

                            if (LA31_0 == THROWS) :
                                alt31 = 1
                            if alt31 == 1:
                                # Java.g:0:0: throwsClause
                                pass 
                                self._state.following.append(self.FOLLOW_throwsClause_in_classScopeDeclarations5738)
                                throwsClause95 = self.throwsClause()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_throwsClause.add(throwsClause95.tree)



                            # Java.g:355:62: ( block | SEMI )
                            alt32 = 2
                            LA32_0 = self.input.LA(1)

                            if (LA32_0 == LCURLY) :
                                alt32 = 1
                            elif (LA32_0 == SEMI) :
                                alt32 = 2
                            else:
                                if self._state.backtracking > 0:
                                    raise BacktrackingFailed

                                nvae = NoViableAltException("", 32, 0, self.input)

                                raise nvae

                            if alt32 == 1:
                                # Java.g:355:63: block
                                pass 
                                self._state.following.append(self.FOLLOW_block_in_classScopeDeclarations5742)
                                block96 = self.block()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_block.add(block96.tree)


                            elif alt32 == 2:
                                # Java.g:355:71: SEMI
                                pass 
                                SEMI97=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_classScopeDeclarations5746) 
                                if self._state.backtracking == 0:
                                    stream_SEMI.add(SEMI97)




                            # AST Rewrite
                            # elements: block, genericTypeParameterList, modifierList, formalParameterList, throwsClause, IDENT
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 356:17: -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? )
                                # Java.g:356:21: ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ( block )? )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VOID_METHOD_DECL, "VOID_METHOD_DECL"), root_1)

                                self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                                # Java.g:356:53: ( genericTypeParameterList )?
                                if stream_genericTypeParameterList.hasNext():
                                    self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                                stream_genericTypeParameterList.reset();
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                                self._adaptor.addChild(root_1, stream_formalParameterList.nextTree())
                                # Java.g:356:105: ( throwsClause )?
                                if stream_throwsClause.hasNext():
                                    self._adaptor.addChild(root_1, stream_throwsClause.nextTree())


                                stream_throwsClause.reset();
                                # Java.g:356:119: ( block )?
                                if stream_block.hasNext():
                                    self._adaptor.addChild(root_1, stream_block.nextTree())


                                stream_block.reset();

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt34 == 3:
                            # Java.g:357:17: ident= IDENT formalParameterList ( throwsClause )? block
                            pass 
                            ident=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_classScopeDeclarations5805) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(ident)
                            self._state.following.append(self.FOLLOW_formalParameterList_in_classScopeDeclarations5807)
                            formalParameterList98 = self.formalParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterList.add(formalParameterList98.tree)
                            # Java.g:357:49: ( throwsClause )?
                            alt33 = 2
                            LA33_0 = self.input.LA(1)

                            if (LA33_0 == THROWS) :
                                alt33 = 1
                            if alt33 == 1:
                                # Java.g:0:0: throwsClause
                                pass 
                                self._state.following.append(self.FOLLOW_throwsClause_in_classScopeDeclarations5809)
                                throwsClause99 = self.throwsClause()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_throwsClause.add(throwsClause99.tree)



                            self._state.following.append(self.FOLLOW_block_in_classScopeDeclarations5812)
                            block100 = self.block()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_block.add(block100.tree)

                            # AST Rewrite
                            # elements: formalParameterList, modifierList, block, throwsClause, genericTypeParameterList
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 358:17: -> ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block )
                                # Java.g:358:21: ^( CONSTRUCTOR_DECL[$ident, \"CONSTRUCTOR_DECL\"] modifierList ( genericTypeParameterList )? formalParameterList ( throwsClause )? block )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.create(CONSTRUCTOR_DECL, ident, "CONSTRUCTOR_DECL"), root_1)

                                self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                                # Java.g:358:81: ( genericTypeParameterList )?
                                if stream_genericTypeParameterList.hasNext():
                                    self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                                stream_genericTypeParameterList.reset();
                                self._adaptor.addChild(root_1, stream_formalParameterList.nextTree())
                                # Java.g:358:127: ( throwsClause )?
                                if stream_throwsClause.hasNext():
                                    self._adaptor.addChild(root_1, stream_throwsClause.nextTree())


                                stream_throwsClause.reset();
                                self._adaptor.addChild(root_1, stream_block.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0





                    elif alt35 == 2:
                        # Java.g:360:13: type classFieldDeclaratorList SEMI
                        pass 
                        self._state.following.append(self.FOLLOW_type_in_classScopeDeclarations5876)
                        type101 = self.type()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_type.add(type101.tree)
                        self._state.following.append(self.FOLLOW_classFieldDeclaratorList_in_classScopeDeclarations5878)
                        classFieldDeclaratorList102 = self.classFieldDeclaratorList()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_classFieldDeclaratorList.add(classFieldDeclaratorList102.tree)
                        SEMI103=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_classScopeDeclarations5880) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI103)

                        # AST Rewrite
                        # elements: type, modifierList, classFieldDeclaratorList
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 361:13: -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList )
                            # Java.g:361:17: ^( VAR_DECLARATION modifierList type classFieldDeclaratorList )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATION, "VAR_DECLARATION"), root_1)

                            self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                            self._adaptor.addChild(root_1, stream_type.nextTree())
                            self._adaptor.addChild(root_1, stream_classFieldDeclaratorList.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt36 == 4:
                    # Java.g:363:9: typeDeclaration
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_typeDeclaration_in_classScopeDeclarations5925)
                    typeDeclaration104 = self.typeDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, typeDeclaration104.tree)


                elif alt36 == 5:
                    # Java.g:364:9: SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    SEMI105=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_classScopeDeclarations5935)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 25, classScopeDeclarations_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classScopeDeclarations"

    class interfaceScopeDeclarations_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceScopeDeclarations_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceScopeDeclarations"
    # Java.g:367:1: interfaceScopeDeclarations : ( modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) ) | type interfaceFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList ) ) | typeDeclaration | SEMI );
    def interfaceScopeDeclarations(self, ):

        retval = self.interfaceScopeDeclarations_return()
        retval.start = self.input.LT(1)
        interfaceScopeDeclarations_StartIndex = self.input.index()
        root_0 = None

        IDENT109 = None
        SEMI113 = None
        VOID114 = None
        IDENT115 = None
        SEMI118 = None
        SEMI121 = None
        SEMI123 = None
        modifierList106 = None

        genericTypeParameterList107 = None

        type108 = None

        formalParameterList110 = None

        arrayDeclaratorList111 = None

        throwsClause112 = None

        formalParameterList116 = None

        throwsClause117 = None

        type119 = None

        interfaceFieldDeclaratorList120 = None

        typeDeclaration122 = None


        IDENT109_tree = None
        SEMI113_tree = None
        VOID114_tree = None
        IDENT115_tree = None
        SEMI118_tree = None
        SEMI121_tree = None
        SEMI123_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_VOID = RewriteRuleTokenStream(self._adaptor, "token VOID")
        stream_SEMI = RewriteRuleTokenStream(self._adaptor, "token SEMI")
        stream_arrayDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclaratorList")
        stream_throwsClause = RewriteRuleSubtreeStream(self._adaptor, "rule throwsClause")
        stream_modifierList = RewriteRuleSubtreeStream(self._adaptor, "rule modifierList")
        stream_genericTypeParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeParameterList")
        stream_interfaceFieldDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule interfaceFieldDeclaratorList")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        stream_formalParameterList = RewriteRuleSubtreeStream(self._adaptor, "rule formalParameterList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 26):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:368:5: ( modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) ) | type interfaceFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList ) ) | typeDeclaration | SEMI )
                alt43 = 3
                alt43 = self.dfa43.predict(self.input)
                if alt43 == 1:
                    # Java.g:368:9: modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) ) | type interfaceFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList ) )
                    pass 
                    self._state.following.append(self.FOLLOW_modifierList_in_interfaceScopeDeclarations5955)
                    modifierList106 = self.modifierList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_modifierList.add(modifierList106.tree)
                    # Java.g:369:9: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) ) | type interfaceFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList ) )
                    alt42 = 2
                    LA42 = self.input.LA(1)
                    if LA42 == LESS_THAN or LA42 == VOID:
                        alt42 = 1
                    elif LA42 == BOOLEAN or LA42 == BYTE or LA42 == CHAR or LA42 == DOUBLE or LA42 == FLOAT or LA42 == INT or LA42 == LONG or LA42 == SHORT:
                        LA42_2 = self.input.LA(2)

                        if (self.synpred50_Java()) :
                            alt42 = 1
                        elif (True) :
                            alt42 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 42, 2, self.input)

                            raise nvae

                    elif LA42 == IDENT:
                        LA42_3 = self.input.LA(2)

                        if (self.synpred50_Java()) :
                            alt42 = 1
                        elif (True) :
                            alt42 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 42, 3, self.input)

                            raise nvae

                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 42, 0, self.input)

                        raise nvae

                    if alt42 == 1:
                        # Java.g:369:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) )
                        pass 
                        # Java.g:369:13: ( genericTypeParameterList )?
                        alt37 = 2
                        LA37_0 = self.input.LA(1)

                        if (LA37_0 == LESS_THAN) :
                            alt37 = 1
                        if alt37 == 1:
                            # Java.g:0:0: genericTypeParameterList
                            pass 
                            self._state.following.append(self.FOLLOW_genericTypeParameterList_in_interfaceScopeDeclarations5969)
                            genericTypeParameterList107 = self.genericTypeParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_genericTypeParameterList.add(genericTypeParameterList107.tree)



                        # Java.g:370:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ) | VOID IDENT formalParameterList ( throwsClause )? SEMI -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? ) )
                        alt41 = 2
                        LA41_0 = self.input.LA(1)

                        if (LA41_0 == BOOLEAN or LA41_0 == BYTE or LA41_0 == CHAR or LA41_0 == DOUBLE or LA41_0 == FLOAT or (INT <= LA41_0 <= LONG) or LA41_0 == SHORT or LA41_0 == IDENT) :
                            alt41 = 1
                        elif (LA41_0 == VOID) :
                            alt41 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 41, 0, self.input)

                            raise nvae

                        if alt41 == 1:
                            # Java.g:370:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI
                            pass 
                            self._state.following.append(self.FOLLOW_type_in_interfaceScopeDeclarations5988)
                            type108 = self.type()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_type.add(type108.tree)
                            IDENT109=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_interfaceScopeDeclarations5990) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT109)
                            self._state.following.append(self.FOLLOW_formalParameterList_in_interfaceScopeDeclarations5992)
                            formalParameterList110 = self.formalParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterList.add(formalParameterList110.tree)
                            # Java.g:370:48: ( arrayDeclaratorList )?
                            alt38 = 2
                            LA38_0 = self.input.LA(1)

                            if (LA38_0 == LBRACK) :
                                alt38 = 1
                            if alt38 == 1:
                                # Java.g:0:0: arrayDeclaratorList
                                pass 
                                self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_interfaceScopeDeclarations5994)
                                arrayDeclaratorList111 = self.arrayDeclaratorList()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_arrayDeclaratorList.add(arrayDeclaratorList111.tree)



                            # Java.g:370:69: ( throwsClause )?
                            alt39 = 2
                            LA39_0 = self.input.LA(1)

                            if (LA39_0 == THROWS) :
                                alt39 = 1
                            if alt39 == 1:
                                # Java.g:0:0: throwsClause
                                pass 
                                self._state.following.append(self.FOLLOW_throwsClause_in_interfaceScopeDeclarations5997)
                                throwsClause112 = self.throwsClause()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_throwsClause.add(throwsClause112.tree)



                            SEMI113=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_interfaceScopeDeclarations6000) 
                            if self._state.backtracking == 0:
                                stream_SEMI.add(SEMI113)

                            # AST Rewrite
                            # elements: throwsClause, IDENT, arrayDeclaratorList, modifierList, genericTypeParameterList, formalParameterList, type
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 371:17: -> ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? )
                                # Java.g:371:21: ^( FUNCTION_METHOD_DECL modifierList ( genericTypeParameterList )? type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL"), root_1)

                                self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                                # Java.g:371:57: ( genericTypeParameterList )?
                                if stream_genericTypeParameterList.hasNext():
                                    self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                                stream_genericTypeParameterList.reset();
                                self._adaptor.addChild(root_1, stream_type.nextTree())
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                                self._adaptor.addChild(root_1, stream_formalParameterList.nextTree())
                                # Java.g:371:114: ( arrayDeclaratorList )?
                                if stream_arrayDeclaratorList.hasNext():
                                    self._adaptor.addChild(root_1, stream_arrayDeclaratorList.nextTree())


                                stream_arrayDeclaratorList.reset();
                                # Java.g:371:135: ( throwsClause )?
                                if stream_throwsClause.hasNext():
                                    self._adaptor.addChild(root_1, stream_throwsClause.nextTree())


                                stream_throwsClause.reset();

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt41 == 2:
                            # Java.g:372:17: VOID IDENT formalParameterList ( throwsClause )? SEMI
                            pass 
                            VOID114=self.match(self.input, VOID, self.FOLLOW_VOID_in_interfaceScopeDeclarations6058) 
                            if self._state.backtracking == 0:
                                stream_VOID.add(VOID114)
                            IDENT115=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_interfaceScopeDeclarations6060) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT115)
                            self._state.following.append(self.FOLLOW_formalParameterList_in_interfaceScopeDeclarations6062)
                            formalParameterList116 = self.formalParameterList()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterList.add(formalParameterList116.tree)
                            # Java.g:372:48: ( throwsClause )?
                            alt40 = 2
                            LA40_0 = self.input.LA(1)

                            if (LA40_0 == THROWS) :
                                alt40 = 1
                            if alt40 == 1:
                                # Java.g:0:0: throwsClause
                                pass 
                                self._state.following.append(self.FOLLOW_throwsClause_in_interfaceScopeDeclarations6064)
                                throwsClause117 = self.throwsClause()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_throwsClause.add(throwsClause117.tree)



                            SEMI118=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_interfaceScopeDeclarations6067) 
                            if self._state.backtracking == 0:
                                stream_SEMI.add(SEMI118)

                            # AST Rewrite
                            # elements: throwsClause, formalParameterList, IDENT, genericTypeParameterList, modifierList
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 373:17: -> ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? )
                                # Java.g:373:21: ^( VOID_METHOD_DECL modifierList ( genericTypeParameterList )? IDENT formalParameterList ( throwsClause )? )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VOID_METHOD_DECL, "VOID_METHOD_DECL"), root_1)

                                self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                                # Java.g:373:53: ( genericTypeParameterList )?
                                if stream_genericTypeParameterList.hasNext():
                                    self._adaptor.addChild(root_1, stream_genericTypeParameterList.nextTree())


                                stream_genericTypeParameterList.reset();
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                                self._adaptor.addChild(root_1, stream_formalParameterList.nextTree())
                                # Java.g:373:105: ( throwsClause )?
                                if stream_throwsClause.hasNext():
                                    self._adaptor.addChild(root_1, stream_throwsClause.nextTree())


                                stream_throwsClause.reset();

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0





                    elif alt42 == 2:
                        # Java.g:375:13: type interfaceFieldDeclaratorList SEMI
                        pass 
                        self._state.following.append(self.FOLLOW_type_in_interfaceScopeDeclarations6130)
                        type119 = self.type()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_type.add(type119.tree)
                        self._state.following.append(self.FOLLOW_interfaceFieldDeclaratorList_in_interfaceScopeDeclarations6132)
                        interfaceFieldDeclaratorList120 = self.interfaceFieldDeclaratorList()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_interfaceFieldDeclaratorList.add(interfaceFieldDeclaratorList120.tree)
                        SEMI121=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_interfaceScopeDeclarations6134) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI121)

                        # AST Rewrite
                        # elements: interfaceFieldDeclaratorList, type, modifierList
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 376:13: -> ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList )
                            # Java.g:376:17: ^( VAR_DECLARATION modifierList type interfaceFieldDeclaratorList )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATION, "VAR_DECLARATION"), root_1)

                            self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                            self._adaptor.addChild(root_1, stream_type.nextTree())
                            self._adaptor.addChild(root_1, stream_interfaceFieldDeclaratorList.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt43 == 2:
                    # Java.g:378:9: typeDeclaration
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_typeDeclaration_in_interfaceScopeDeclarations6179)
                    typeDeclaration122 = self.typeDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, typeDeclaration122.tree)


                elif alt43 == 3:
                    # Java.g:379:9: SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    SEMI123=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_interfaceScopeDeclarations6189)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 26, interfaceScopeDeclarations_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceScopeDeclarations"

    class classFieldDeclaratorList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classFieldDeclaratorList_return, self).__init__()

            self.tree = None




    # $ANTLR start "classFieldDeclaratorList"
    # Java.g:382:1: classFieldDeclaratorList : classFieldDeclarator ( COMMA classFieldDeclarator )* -> ^( VAR_DECLARATOR_LIST ( classFieldDeclarator )+ ) ;
    def classFieldDeclaratorList(self, ):

        retval = self.classFieldDeclaratorList_return()
        retval.start = self.input.LT(1)
        classFieldDeclaratorList_StartIndex = self.input.index()
        root_0 = None

        COMMA125 = None
        classFieldDeclarator124 = None

        classFieldDeclarator126 = None


        COMMA125_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_classFieldDeclarator = RewriteRuleSubtreeStream(self._adaptor, "rule classFieldDeclarator")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 27):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:383:5: ( classFieldDeclarator ( COMMA classFieldDeclarator )* -> ^( VAR_DECLARATOR_LIST ( classFieldDeclarator )+ ) )
                # Java.g:383:9: classFieldDeclarator ( COMMA classFieldDeclarator )*
                pass 
                self._state.following.append(self.FOLLOW_classFieldDeclarator_in_classFieldDeclaratorList6209)
                classFieldDeclarator124 = self.classFieldDeclarator()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_classFieldDeclarator.add(classFieldDeclarator124.tree)
                # Java.g:383:30: ( COMMA classFieldDeclarator )*
                while True: #loop44
                    alt44 = 2
                    LA44_0 = self.input.LA(1)

                    if (LA44_0 == COMMA) :
                        alt44 = 1


                    if alt44 == 1:
                        # Java.g:383:31: COMMA classFieldDeclarator
                        pass 
                        COMMA125=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_classFieldDeclaratorList6212) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA125)
                        self._state.following.append(self.FOLLOW_classFieldDeclarator_in_classFieldDeclaratorList6214)
                        classFieldDeclarator126 = self.classFieldDeclarator()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_classFieldDeclarator.add(classFieldDeclarator126.tree)


                    else:
                        break #loop44

                # AST Rewrite
                # elements: classFieldDeclarator
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 384:9: -> ^( VAR_DECLARATOR_LIST ( classFieldDeclarator )+ )
                    # Java.g:384:13: ^( VAR_DECLARATOR_LIST ( classFieldDeclarator )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATOR_LIST, "VAR_DECLARATOR_LIST"), root_1)

                    # Java.g:384:35: ( classFieldDeclarator )+
                    if not (stream_classFieldDeclarator.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_classFieldDeclarator.hasNext():
                        self._adaptor.addChild(root_1, stream_classFieldDeclarator.nextTree())


                    stream_classFieldDeclarator.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 27, classFieldDeclaratorList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classFieldDeclaratorList"

    class classFieldDeclarator_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.classFieldDeclarator_return, self).__init__()

            self.tree = None




    # $ANTLR start "classFieldDeclarator"
    # Java.g:387:1: classFieldDeclarator : variableDeclaratorId ( ASSIGN variableInitializer )? -> ^( VAR_DECLARATOR variableDeclaratorId ( variableInitializer )? ) ;
    def classFieldDeclarator(self, ):

        retval = self.classFieldDeclarator_return()
        retval.start = self.input.LT(1)
        classFieldDeclarator_StartIndex = self.input.index()
        root_0 = None

        ASSIGN128 = None
        variableDeclaratorId127 = None

        variableInitializer129 = None


        ASSIGN128_tree = None
        stream_ASSIGN = RewriteRuleTokenStream(self._adaptor, "token ASSIGN")
        stream_variableDeclaratorId = RewriteRuleSubtreeStream(self._adaptor, "rule variableDeclaratorId")
        stream_variableInitializer = RewriteRuleSubtreeStream(self._adaptor, "rule variableInitializer")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 28):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:388:5: ( variableDeclaratorId ( ASSIGN variableInitializer )? -> ^( VAR_DECLARATOR variableDeclaratorId ( variableInitializer )? ) )
                # Java.g:388:9: variableDeclaratorId ( ASSIGN variableInitializer )?
                pass 
                self._state.following.append(self.FOLLOW_variableDeclaratorId_in_classFieldDeclarator6253)
                variableDeclaratorId127 = self.variableDeclaratorId()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_variableDeclaratorId.add(variableDeclaratorId127.tree)
                # Java.g:388:30: ( ASSIGN variableInitializer )?
                alt45 = 2
                LA45_0 = self.input.LA(1)

                if (LA45_0 == ASSIGN) :
                    alt45 = 1
                if alt45 == 1:
                    # Java.g:388:31: ASSIGN variableInitializer
                    pass 
                    ASSIGN128=self.match(self.input, ASSIGN, self.FOLLOW_ASSIGN_in_classFieldDeclarator6256) 
                    if self._state.backtracking == 0:
                        stream_ASSIGN.add(ASSIGN128)
                    self._state.following.append(self.FOLLOW_variableInitializer_in_classFieldDeclarator6258)
                    variableInitializer129 = self.variableInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_variableInitializer.add(variableInitializer129.tree)




                # AST Rewrite
                # elements: variableDeclaratorId, variableInitializer
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 389:9: -> ^( VAR_DECLARATOR variableDeclaratorId ( variableInitializer )? )
                    # Java.g:389:13: ^( VAR_DECLARATOR variableDeclaratorId ( variableInitializer )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATOR, "VAR_DECLARATOR"), root_1)

                    self._adaptor.addChild(root_1, stream_variableDeclaratorId.nextTree())
                    # Java.g:389:51: ( variableInitializer )?
                    if stream_variableInitializer.hasNext():
                        self._adaptor.addChild(root_1, stream_variableInitializer.nextTree())


                    stream_variableInitializer.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 28, classFieldDeclarator_StartIndex, success)

            pass
        return retval

    # $ANTLR end "classFieldDeclarator"

    class interfaceFieldDeclaratorList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceFieldDeclaratorList_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceFieldDeclaratorList"
    # Java.g:392:1: interfaceFieldDeclaratorList : interfaceFieldDeclarator ( COMMA interfaceFieldDeclarator )* -> ^( VAR_DECLARATOR_LIST ( interfaceFieldDeclarator )+ ) ;
    def interfaceFieldDeclaratorList(self, ):

        retval = self.interfaceFieldDeclaratorList_return()
        retval.start = self.input.LT(1)
        interfaceFieldDeclaratorList_StartIndex = self.input.index()
        root_0 = None

        COMMA131 = None
        interfaceFieldDeclarator130 = None

        interfaceFieldDeclarator132 = None


        COMMA131_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_interfaceFieldDeclarator = RewriteRuleSubtreeStream(self._adaptor, "rule interfaceFieldDeclarator")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 29):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:393:5: ( interfaceFieldDeclarator ( COMMA interfaceFieldDeclarator )* -> ^( VAR_DECLARATOR_LIST ( interfaceFieldDeclarator )+ ) )
                # Java.g:393:9: interfaceFieldDeclarator ( COMMA interfaceFieldDeclarator )*
                pass 
                self._state.following.append(self.FOLLOW_interfaceFieldDeclarator_in_interfaceFieldDeclaratorList6299)
                interfaceFieldDeclarator130 = self.interfaceFieldDeclarator()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_interfaceFieldDeclarator.add(interfaceFieldDeclarator130.tree)
                # Java.g:393:34: ( COMMA interfaceFieldDeclarator )*
                while True: #loop46
                    alt46 = 2
                    LA46_0 = self.input.LA(1)

                    if (LA46_0 == COMMA) :
                        alt46 = 1


                    if alt46 == 1:
                        # Java.g:393:35: COMMA interfaceFieldDeclarator
                        pass 
                        COMMA131=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_interfaceFieldDeclaratorList6302) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA131)
                        self._state.following.append(self.FOLLOW_interfaceFieldDeclarator_in_interfaceFieldDeclaratorList6304)
                        interfaceFieldDeclarator132 = self.interfaceFieldDeclarator()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_interfaceFieldDeclarator.add(interfaceFieldDeclarator132.tree)


                    else:
                        break #loop46

                # AST Rewrite
                # elements: interfaceFieldDeclarator
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 394:9: -> ^( VAR_DECLARATOR_LIST ( interfaceFieldDeclarator )+ )
                    # Java.g:394:13: ^( VAR_DECLARATOR_LIST ( interfaceFieldDeclarator )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATOR_LIST, "VAR_DECLARATOR_LIST"), root_1)

                    # Java.g:394:35: ( interfaceFieldDeclarator )+
                    if not (stream_interfaceFieldDeclarator.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_interfaceFieldDeclarator.hasNext():
                        self._adaptor.addChild(root_1, stream_interfaceFieldDeclarator.nextTree())


                    stream_interfaceFieldDeclarator.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 29, interfaceFieldDeclaratorList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceFieldDeclaratorList"

    class interfaceFieldDeclarator_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.interfaceFieldDeclarator_return, self).__init__()

            self.tree = None




    # $ANTLR start "interfaceFieldDeclarator"
    # Java.g:397:1: interfaceFieldDeclarator : variableDeclaratorId ASSIGN variableInitializer -> ^( VAR_DECLARATOR variableDeclaratorId variableInitializer ) ;
    def interfaceFieldDeclarator(self, ):

        retval = self.interfaceFieldDeclarator_return()
        retval.start = self.input.LT(1)
        interfaceFieldDeclarator_StartIndex = self.input.index()
        root_0 = None

        ASSIGN134 = None
        variableDeclaratorId133 = None

        variableInitializer135 = None


        ASSIGN134_tree = None
        stream_ASSIGN = RewriteRuleTokenStream(self._adaptor, "token ASSIGN")
        stream_variableDeclaratorId = RewriteRuleSubtreeStream(self._adaptor, "rule variableDeclaratorId")
        stream_variableInitializer = RewriteRuleSubtreeStream(self._adaptor, "rule variableInitializer")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 30):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:398:5: ( variableDeclaratorId ASSIGN variableInitializer -> ^( VAR_DECLARATOR variableDeclaratorId variableInitializer ) )
                # Java.g:398:9: variableDeclaratorId ASSIGN variableInitializer
                pass 
                self._state.following.append(self.FOLLOW_variableDeclaratorId_in_interfaceFieldDeclarator6343)
                variableDeclaratorId133 = self.variableDeclaratorId()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_variableDeclaratorId.add(variableDeclaratorId133.tree)
                ASSIGN134=self.match(self.input, ASSIGN, self.FOLLOW_ASSIGN_in_interfaceFieldDeclarator6345) 
                if self._state.backtracking == 0:
                    stream_ASSIGN.add(ASSIGN134)
                self._state.following.append(self.FOLLOW_variableInitializer_in_interfaceFieldDeclarator6347)
                variableInitializer135 = self.variableInitializer()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_variableInitializer.add(variableInitializer135.tree)

                # AST Rewrite
                # elements: variableInitializer, variableDeclaratorId
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 399:9: -> ^( VAR_DECLARATOR variableDeclaratorId variableInitializer )
                    # Java.g:399:13: ^( VAR_DECLARATOR variableDeclaratorId variableInitializer )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATOR, "VAR_DECLARATOR"), root_1)

                    self._adaptor.addChild(root_1, stream_variableDeclaratorId.nextTree())
                    self._adaptor.addChild(root_1, stream_variableInitializer.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 30, interfaceFieldDeclarator_StartIndex, success)

            pass
        return retval

    # $ANTLR end "interfaceFieldDeclarator"

    class variableDeclaratorId_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.variableDeclaratorId_return, self).__init__()

            self.tree = None




    # $ANTLR start "variableDeclaratorId"
    # Java.g:402:1: variableDeclaratorId : IDENT ( arrayDeclaratorList )? ;
    def variableDeclaratorId(self, ):

        retval = self.variableDeclaratorId_return()
        retval.start = self.input.LT(1)
        variableDeclaratorId_StartIndex = self.input.index()
        root_0 = None

        IDENT136 = None
        arrayDeclaratorList137 = None


        IDENT136_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 31):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:403:5: ( IDENT ( arrayDeclaratorList )? )
                # Java.g:403:9: IDENT ( arrayDeclaratorList )?
                pass 
                root_0 = self._adaptor.nil()

                IDENT136=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_variableDeclaratorId6385)
                if self._state.backtracking == 0:

                    IDENT136_tree = self._adaptor.createWithPayload(IDENT136)
                    root_0 = self._adaptor.becomeRoot(IDENT136_tree, root_0)

                # Java.g:403:16: ( arrayDeclaratorList )?
                alt47 = 2
                LA47_0 = self.input.LA(1)

                if (LA47_0 == LBRACK) :
                    alt47 = 1
                if alt47 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_variableDeclaratorId6388)
                    arrayDeclaratorList137 = self.arrayDeclaratorList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, arrayDeclaratorList137.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 31, variableDeclaratorId_StartIndex, success)

            pass
        return retval

    # $ANTLR end "variableDeclaratorId"

    class variableInitializer_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.variableInitializer_return, self).__init__()

            self.tree = None




    # $ANTLR start "variableInitializer"
    # Java.g:406:1: variableInitializer : ( arrayInitializer | expression );
    def variableInitializer(self, ):

        retval = self.variableInitializer_return()
        retval.start = self.input.LT(1)
        variableInitializer_StartIndex = self.input.index()
        root_0 = None

        arrayInitializer138 = None

        expression139 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 32):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:407:5: ( arrayInitializer | expression )
                alt48 = 2
                LA48_0 = self.input.LA(1)

                if (LA48_0 == LCURLY) :
                    alt48 = 1
                elif (LA48_0 == DEC or LA48_0 == INC or LA48_0 == LESS_THAN or LA48_0 == LOGICAL_NOT or (LPAREN <= LA48_0 <= MINUS) or LA48_0 == NOT or LA48_0 == PLUS or LA48_0 == BOOLEAN or LA48_0 == BYTE or LA48_0 == CHAR or LA48_0 == DOUBLE or LA48_0 == FALSE or LA48_0 == FLOAT or (INT <= LA48_0 <= LONG) or (NEW <= LA48_0 <= NULL) or LA48_0 == SHORT or LA48_0 == SUPER or LA48_0 == THIS or LA48_0 == TRUE or LA48_0 == VOID or (IDENT <= LA48_0 <= STRING_LITERAL)) :
                    alt48 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 48, 0, self.input)

                    raise nvae

                if alt48 == 1:
                    # Java.g:407:9: arrayInitializer
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_arrayInitializer_in_variableInitializer6408)
                    arrayInitializer138 = self.arrayInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, arrayInitializer138.tree)


                elif alt48 == 2:
                    # Java.g:408:9: expression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_expression_in_variableInitializer6418)
                    expression139 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, expression139.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 32, variableInitializer_StartIndex, success)

            pass
        return retval

    # $ANTLR end "variableInitializer"

    class arrayDeclarator_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.arrayDeclarator_return, self).__init__()

            self.tree = None




    # $ANTLR start "arrayDeclarator"
    # Java.g:411:1: arrayDeclarator : LBRACK RBRACK -> ^( ARRAY_DECLARATOR ) ;
    def arrayDeclarator(self, ):

        retval = self.arrayDeclarator_return()
        retval.start = self.input.LT(1)
        arrayDeclarator_StartIndex = self.input.index()
        root_0 = None

        LBRACK140 = None
        RBRACK141 = None

        LBRACK140_tree = None
        RBRACK141_tree = None
        stream_RBRACK = RewriteRuleTokenStream(self._adaptor, "token RBRACK")
        stream_LBRACK = RewriteRuleTokenStream(self._adaptor, "token LBRACK")

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 33):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:412:5: ( LBRACK RBRACK -> ^( ARRAY_DECLARATOR ) )
                # Java.g:412:9: LBRACK RBRACK
                pass 
                LBRACK140=self.match(self.input, LBRACK, self.FOLLOW_LBRACK_in_arrayDeclarator6437) 
                if self._state.backtracking == 0:
                    stream_LBRACK.add(LBRACK140)
                RBRACK141=self.match(self.input, RBRACK, self.FOLLOW_RBRACK_in_arrayDeclarator6439) 
                if self._state.backtracking == 0:
                    stream_RBRACK.add(RBRACK141)

                # AST Rewrite
                # elements: 
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 413:9: -> ^( ARRAY_DECLARATOR )
                    # Java.g:413:13: ^( ARRAY_DECLARATOR )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ARRAY_DECLARATOR, "ARRAY_DECLARATOR"), root_1)

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 33, arrayDeclarator_StartIndex, success)

            pass
        return retval

    # $ANTLR end "arrayDeclarator"

    class arrayDeclaratorList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.arrayDeclaratorList_return, self).__init__()

            self.tree = None




    # $ANTLR start "arrayDeclaratorList"
    # Java.g:416:1: arrayDeclaratorList : ( arrayDeclarator )+ -> ^( ARRAY_DECLARATOR_LIST ( arrayDeclarator )+ ) ;
    def arrayDeclaratorList(self, ):

        retval = self.arrayDeclaratorList_return()
        retval.start = self.input.LT(1)
        arrayDeclaratorList_StartIndex = self.input.index()
        root_0 = None

        arrayDeclarator142 = None


        stream_arrayDeclarator = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclarator")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 34):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:417:5: ( ( arrayDeclarator )+ -> ^( ARRAY_DECLARATOR_LIST ( arrayDeclarator )+ ) )
                # Java.g:417:9: ( arrayDeclarator )+
                pass 
                # Java.g:417:9: ( arrayDeclarator )+
                cnt49 = 0
                while True: #loop49
                    alt49 = 2
                    LA49_0 = self.input.LA(1)

                    if (LA49_0 == LBRACK) :
                        LA49_2 = self.input.LA(2)

                        if (self.synpred58_Java()) :
                            alt49 = 1




                    if alt49 == 1:
                        # Java.g:0:0: arrayDeclarator
                        pass 
                        self._state.following.append(self.FOLLOW_arrayDeclarator_in_arrayDeclaratorList6473)
                        arrayDeclarator142 = self.arrayDeclarator()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arrayDeclarator.add(arrayDeclarator142.tree)


                    else:
                        if cnt49 >= 1:
                            break #loop49

                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        eee = EarlyExitException(49, self.input)
                        raise eee

                    cnt49 += 1

                # AST Rewrite
                # elements: arrayDeclarator
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 418:9: -> ^( ARRAY_DECLARATOR_LIST ( arrayDeclarator )+ )
                    # Java.g:418:13: ^( ARRAY_DECLARATOR_LIST ( arrayDeclarator )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ARRAY_DECLARATOR_LIST, "ARRAY_DECLARATOR_LIST"), root_1)

                    # Java.g:418:37: ( arrayDeclarator )+
                    if not (stream_arrayDeclarator.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_arrayDeclarator.hasNext():
                        self._adaptor.addChild(root_1, stream_arrayDeclarator.nextTree())


                    stream_arrayDeclarator.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 34, arrayDeclaratorList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "arrayDeclaratorList"

    class arrayInitializer_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.arrayInitializer_return, self).__init__()

            self.tree = None




    # $ANTLR start "arrayInitializer"
    # Java.g:421:1: arrayInitializer : LCURLY ( variableInitializer ( COMMA variableInitializer )* ( COMMA )? )? RCURLY -> ^( ARRAY_INITIALIZER[$LCURLY, \"ARRAY_INITIALIZER\"] ( variableInitializer )* ) ;
    def arrayInitializer(self, ):

        retval = self.arrayInitializer_return()
        retval.start = self.input.LT(1)
        arrayInitializer_StartIndex = self.input.index()
        root_0 = None

        LCURLY143 = None
        COMMA145 = None
        COMMA147 = None
        RCURLY148 = None
        variableInitializer144 = None

        variableInitializer146 = None


        LCURLY143_tree = None
        COMMA145_tree = None
        COMMA147_tree = None
        RCURLY148_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_variableInitializer = RewriteRuleSubtreeStream(self._adaptor, "rule variableInitializer")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 35):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:422:5: ( LCURLY ( variableInitializer ( COMMA variableInitializer )* ( COMMA )? )? RCURLY -> ^( ARRAY_INITIALIZER[$LCURLY, \"ARRAY_INITIALIZER\"] ( variableInitializer )* ) )
                # Java.g:422:9: LCURLY ( variableInitializer ( COMMA variableInitializer )* ( COMMA )? )? RCURLY
                pass 
                LCURLY143=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_arrayInitializer6511) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY143)
                # Java.g:422:16: ( variableInitializer ( COMMA variableInitializer )* ( COMMA )? )?
                alt52 = 2
                LA52_0 = self.input.LA(1)

                if (LA52_0 == DEC or LA52_0 == INC or LA52_0 == LCURLY or LA52_0 == LESS_THAN or LA52_0 == LOGICAL_NOT or (LPAREN <= LA52_0 <= MINUS) or LA52_0 == NOT or LA52_0 == PLUS or LA52_0 == BOOLEAN or LA52_0 == BYTE or LA52_0 == CHAR or LA52_0 == DOUBLE or LA52_0 == FALSE or LA52_0 == FLOAT or (INT <= LA52_0 <= LONG) or (NEW <= LA52_0 <= NULL) or LA52_0 == SHORT or LA52_0 == SUPER or LA52_0 == THIS or LA52_0 == TRUE or LA52_0 == VOID or (IDENT <= LA52_0 <= STRING_LITERAL)) :
                    alt52 = 1
                if alt52 == 1:
                    # Java.g:422:17: variableInitializer ( COMMA variableInitializer )* ( COMMA )?
                    pass 
                    self._state.following.append(self.FOLLOW_variableInitializer_in_arrayInitializer6514)
                    variableInitializer144 = self.variableInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_variableInitializer.add(variableInitializer144.tree)
                    # Java.g:422:37: ( COMMA variableInitializer )*
                    while True: #loop50
                        alt50 = 2
                        LA50_0 = self.input.LA(1)

                        if (LA50_0 == COMMA) :
                            LA50_1 = self.input.LA(2)

                            if (LA50_1 == DEC or LA50_1 == INC or LA50_1 == LCURLY or LA50_1 == LESS_THAN or LA50_1 == LOGICAL_NOT or (LPAREN <= LA50_1 <= MINUS) or LA50_1 == NOT or LA50_1 == PLUS or LA50_1 == BOOLEAN or LA50_1 == BYTE or LA50_1 == CHAR or LA50_1 == DOUBLE or LA50_1 == FALSE or LA50_1 == FLOAT or (INT <= LA50_1 <= LONG) or (NEW <= LA50_1 <= NULL) or LA50_1 == SHORT or LA50_1 == SUPER or LA50_1 == THIS or LA50_1 == TRUE or LA50_1 == VOID or (IDENT <= LA50_1 <= STRING_LITERAL)) :
                                alt50 = 1




                        if alt50 == 1:
                            # Java.g:422:38: COMMA variableInitializer
                            pass 
                            COMMA145=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_arrayInitializer6517) 
                            if self._state.backtracking == 0:
                                stream_COMMA.add(COMMA145)
                            self._state.following.append(self.FOLLOW_variableInitializer_in_arrayInitializer6519)
                            variableInitializer146 = self.variableInitializer()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_variableInitializer.add(variableInitializer146.tree)


                        else:
                            break #loop50
                    # Java.g:422:66: ( COMMA )?
                    alt51 = 2
                    LA51_0 = self.input.LA(1)

                    if (LA51_0 == COMMA) :
                        alt51 = 1
                    if alt51 == 1:
                        # Java.g:0:0: COMMA
                        pass 
                        COMMA147=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_arrayInitializer6523) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA147)






                RCURLY148=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_arrayInitializer6528) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY148)

                # AST Rewrite
                # elements: variableInitializer
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 423:9: -> ^( ARRAY_INITIALIZER[$LCURLY, \"ARRAY_INITIALIZER\"] ( variableInitializer )* )
                    # Java.g:423:13: ^( ARRAY_INITIALIZER[$LCURLY, \"ARRAY_INITIALIZER\"] ( variableInitializer )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ARRAY_INITIALIZER, LCURLY143, "ARRAY_INITIALIZER"), root_1)

                    # Java.g:423:63: ( variableInitializer )*
                    while stream_variableInitializer.hasNext():
                        self._adaptor.addChild(root_1, stream_variableInitializer.nextTree())


                    stream_variableInitializer.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 35, arrayInitializer_StartIndex, success)

            pass
        return retval

    # $ANTLR end "arrayInitializer"

    class throwsClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.throwsClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "throwsClause"
    # Java.g:426:1: throwsClause : THROWS qualifiedIdentList -> ^( THROWS_CLAUSE[$THROWS, \"THROWS_CLAUSE\"] qualifiedIdentList ) ;
    def throwsClause(self, ):

        retval = self.throwsClause_return()
        retval.start = self.input.LT(1)
        throwsClause_StartIndex = self.input.index()
        root_0 = None

        THROWS149 = None
        qualifiedIdentList150 = None


        THROWS149_tree = None
        stream_THROWS = RewriteRuleTokenStream(self._adaptor, "token THROWS")
        stream_qualifiedIdentList = RewriteRuleSubtreeStream(self._adaptor, "rule qualifiedIdentList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 36):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:427:5: ( THROWS qualifiedIdentList -> ^( THROWS_CLAUSE[$THROWS, \"THROWS_CLAUSE\"] qualifiedIdentList ) )
                # Java.g:427:9: THROWS qualifiedIdentList
                pass 
                THROWS149=self.match(self.input, THROWS, self.FOLLOW_THROWS_in_throwsClause6566) 
                if self._state.backtracking == 0:
                    stream_THROWS.add(THROWS149)
                self._state.following.append(self.FOLLOW_qualifiedIdentList_in_throwsClause6568)
                qualifiedIdentList150 = self.qualifiedIdentList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_qualifiedIdentList.add(qualifiedIdentList150.tree)

                # AST Rewrite
                # elements: qualifiedIdentList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 428:9: -> ^( THROWS_CLAUSE[$THROWS, \"THROWS_CLAUSE\"] qualifiedIdentList )
                    # Java.g:428:13: ^( THROWS_CLAUSE[$THROWS, \"THROWS_CLAUSE\"] qualifiedIdentList )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(THROWS_CLAUSE, THROWS149, "THROWS_CLAUSE"), root_1)

                    self._adaptor.addChild(root_1, stream_qualifiedIdentList.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 36, throwsClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "throwsClause"

    class modifierList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.modifierList_return, self).__init__()

            self.tree = None




    # $ANTLR start "modifierList"
    # Java.g:431:1: modifierList : ( modifier )* -> ^( MODIFIER_LIST ( modifier )* ) ;
    def modifierList(self, ):

        retval = self.modifierList_return()
        retval.start = self.input.LT(1)
        modifierList_StartIndex = self.input.index()
        root_0 = None

        modifier151 = None


        stream_modifier = RewriteRuleSubtreeStream(self._adaptor, "rule modifier")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 37):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:432:5: ( ( modifier )* -> ^( MODIFIER_LIST ( modifier )* ) )
                # Java.g:432:9: ( modifier )*
                pass 
                # Java.g:432:9: ( modifier )*
                while True: #loop53
                    alt53 = 2
                    LA53_0 = self.input.LA(1)

                    if (LA53_0 == AT) :
                        LA53_2 = self.input.LA(2)

                        if (LA53_2 == IDENT) :
                            alt53 = 1


                    elif (LA53_0 == ABSTRACT or LA53_0 == FINAL or LA53_0 == NATIVE or (PRIVATE <= LA53_0 <= PUBLIC) or (STATIC <= LA53_0 <= STRICTFP) or LA53_0 == SYNCHRONIZED or LA53_0 == TRANSIENT or LA53_0 == VOLATILE) :
                        alt53 = 1


                    if alt53 == 1:
                        # Java.g:0:0: modifier
                        pass 
                        self._state.following.append(self.FOLLOW_modifier_in_modifierList6605)
                        modifier151 = self.modifier()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_modifier.add(modifier151.tree)


                    else:
                        break #loop53

                # AST Rewrite
                # elements: modifier
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 433:9: -> ^( MODIFIER_LIST ( modifier )* )
                    # Java.g:433:13: ^( MODIFIER_LIST ( modifier )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(MODIFIER_LIST, "MODIFIER_LIST"), root_1)

                    # Java.g:433:29: ( modifier )*
                    while stream_modifier.hasNext():
                        self._adaptor.addChild(root_1, stream_modifier.nextTree())


                    stream_modifier.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 37, modifierList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "modifierList"

    class modifier_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.modifier_return, self).__init__()

            self.tree = None




    # $ANTLR start "modifier"
    # Java.g:436:1: modifier : ( PUBLIC | PROTECTED | PRIVATE | STATIC | ABSTRACT | NATIVE | SYNCHRONIZED | TRANSIENT | VOLATILE | STRICTFP | localModifier );
    def modifier(self, ):

        retval = self.modifier_return()
        retval.start = self.input.LT(1)
        modifier_StartIndex = self.input.index()
        root_0 = None

        PUBLIC152 = None
        PROTECTED153 = None
        PRIVATE154 = None
        STATIC155 = None
        ABSTRACT156 = None
        NATIVE157 = None
        SYNCHRONIZED158 = None
        TRANSIENT159 = None
        VOLATILE160 = None
        STRICTFP161 = None
        localModifier162 = None


        PUBLIC152_tree = None
        PROTECTED153_tree = None
        PRIVATE154_tree = None
        STATIC155_tree = None
        ABSTRACT156_tree = None
        NATIVE157_tree = None
        SYNCHRONIZED158_tree = None
        TRANSIENT159_tree = None
        VOLATILE160_tree = None
        STRICTFP161_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 38):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:437:5: ( PUBLIC | PROTECTED | PRIVATE | STATIC | ABSTRACT | NATIVE | SYNCHRONIZED | TRANSIENT | VOLATILE | STRICTFP | localModifier )
                alt54 = 11
                LA54 = self.input.LA(1)
                if LA54 == PUBLIC:
                    alt54 = 1
                elif LA54 == PROTECTED:
                    alt54 = 2
                elif LA54 == PRIVATE:
                    alt54 = 3
                elif LA54 == STATIC:
                    alt54 = 4
                elif LA54 == ABSTRACT:
                    alt54 = 5
                elif LA54 == NATIVE:
                    alt54 = 6
                elif LA54 == SYNCHRONIZED:
                    alt54 = 7
                elif LA54 == TRANSIENT:
                    alt54 = 8
                elif LA54 == VOLATILE:
                    alt54 = 9
                elif LA54 == STRICTFP:
                    alt54 = 10
                elif LA54 == AT or LA54 == FINAL:
                    alt54 = 11
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 54, 0, self.input)

                    raise nvae

                if alt54 == 1:
                    # Java.g:437:9: PUBLIC
                    pass 
                    root_0 = self._adaptor.nil()

                    PUBLIC152=self.match(self.input, PUBLIC, self.FOLLOW_PUBLIC_in_modifier6643)
                    if self._state.backtracking == 0:

                        PUBLIC152_tree = self._adaptor.createWithPayload(PUBLIC152)
                        self._adaptor.addChild(root_0, PUBLIC152_tree)



                elif alt54 == 2:
                    # Java.g:438:9: PROTECTED
                    pass 
                    root_0 = self._adaptor.nil()

                    PROTECTED153=self.match(self.input, PROTECTED, self.FOLLOW_PROTECTED_in_modifier6653)
                    if self._state.backtracking == 0:

                        PROTECTED153_tree = self._adaptor.createWithPayload(PROTECTED153)
                        self._adaptor.addChild(root_0, PROTECTED153_tree)



                elif alt54 == 3:
                    # Java.g:439:9: PRIVATE
                    pass 
                    root_0 = self._adaptor.nil()

                    PRIVATE154=self.match(self.input, PRIVATE, self.FOLLOW_PRIVATE_in_modifier6663)
                    if self._state.backtracking == 0:

                        PRIVATE154_tree = self._adaptor.createWithPayload(PRIVATE154)
                        self._adaptor.addChild(root_0, PRIVATE154_tree)



                elif alt54 == 4:
                    # Java.g:440:9: STATIC
                    pass 
                    root_0 = self._adaptor.nil()

                    STATIC155=self.match(self.input, STATIC, self.FOLLOW_STATIC_in_modifier6673)
                    if self._state.backtracking == 0:

                        STATIC155_tree = self._adaptor.createWithPayload(STATIC155)
                        self._adaptor.addChild(root_0, STATIC155_tree)



                elif alt54 == 5:
                    # Java.g:441:9: ABSTRACT
                    pass 
                    root_0 = self._adaptor.nil()

                    ABSTRACT156=self.match(self.input, ABSTRACT, self.FOLLOW_ABSTRACT_in_modifier6683)
                    if self._state.backtracking == 0:

                        ABSTRACT156_tree = self._adaptor.createWithPayload(ABSTRACT156)
                        self._adaptor.addChild(root_0, ABSTRACT156_tree)



                elif alt54 == 6:
                    # Java.g:442:9: NATIVE
                    pass 
                    root_0 = self._adaptor.nil()

                    NATIVE157=self.match(self.input, NATIVE, self.FOLLOW_NATIVE_in_modifier6693)
                    if self._state.backtracking == 0:

                        NATIVE157_tree = self._adaptor.createWithPayload(NATIVE157)
                        self._adaptor.addChild(root_0, NATIVE157_tree)



                elif alt54 == 7:
                    # Java.g:443:9: SYNCHRONIZED
                    pass 
                    root_0 = self._adaptor.nil()

                    SYNCHRONIZED158=self.match(self.input, SYNCHRONIZED, self.FOLLOW_SYNCHRONIZED_in_modifier6703)
                    if self._state.backtracking == 0:

                        SYNCHRONIZED158_tree = self._adaptor.createWithPayload(SYNCHRONIZED158)
                        self._adaptor.addChild(root_0, SYNCHRONIZED158_tree)



                elif alt54 == 8:
                    # Java.g:444:9: TRANSIENT
                    pass 
                    root_0 = self._adaptor.nil()

                    TRANSIENT159=self.match(self.input, TRANSIENT, self.FOLLOW_TRANSIENT_in_modifier6713)
                    if self._state.backtracking == 0:

                        TRANSIENT159_tree = self._adaptor.createWithPayload(TRANSIENT159)
                        self._adaptor.addChild(root_0, TRANSIENT159_tree)



                elif alt54 == 9:
                    # Java.g:445:9: VOLATILE
                    pass 
                    root_0 = self._adaptor.nil()

                    VOLATILE160=self.match(self.input, VOLATILE, self.FOLLOW_VOLATILE_in_modifier6723)
                    if self._state.backtracking == 0:

                        VOLATILE160_tree = self._adaptor.createWithPayload(VOLATILE160)
                        self._adaptor.addChild(root_0, VOLATILE160_tree)



                elif alt54 == 10:
                    # Java.g:446:9: STRICTFP
                    pass 
                    root_0 = self._adaptor.nil()

                    STRICTFP161=self.match(self.input, STRICTFP, self.FOLLOW_STRICTFP_in_modifier6733)
                    if self._state.backtracking == 0:

                        STRICTFP161_tree = self._adaptor.createWithPayload(STRICTFP161)
                        self._adaptor.addChild(root_0, STRICTFP161_tree)



                elif alt54 == 11:
                    # Java.g:447:9: localModifier
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_localModifier_in_modifier6743)
                    localModifier162 = self.localModifier()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, localModifier162.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 38, modifier_StartIndex, success)

            pass
        return retval

    # $ANTLR end "modifier"

    class localModifierList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.localModifierList_return, self).__init__()

            self.tree = None




    # $ANTLR start "localModifierList"
    # Java.g:450:1: localModifierList : ( localModifier )* -> ^( LOCAL_MODIFIER_LIST ( localModifier )* ) ;
    def localModifierList(self, ):

        retval = self.localModifierList_return()
        retval.start = self.input.LT(1)
        localModifierList_StartIndex = self.input.index()
        root_0 = None

        localModifier163 = None


        stream_localModifier = RewriteRuleSubtreeStream(self._adaptor, "rule localModifier")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 39):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:451:5: ( ( localModifier )* -> ^( LOCAL_MODIFIER_LIST ( localModifier )* ) )
                # Java.g:451:9: ( localModifier )*
                pass 
                # Java.g:451:9: ( localModifier )*
                while True: #loop55
                    alt55 = 2
                    LA55_0 = self.input.LA(1)

                    if (LA55_0 == AT or LA55_0 == FINAL) :
                        alt55 = 1


                    if alt55 == 1:
                        # Java.g:0:0: localModifier
                        pass 
                        self._state.following.append(self.FOLLOW_localModifier_in_localModifierList6762)
                        localModifier163 = self.localModifier()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_localModifier.add(localModifier163.tree)


                    else:
                        break #loop55

                # AST Rewrite
                # elements: localModifier
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 452:9: -> ^( LOCAL_MODIFIER_LIST ( localModifier )* )
                    # Java.g:452:12: ^( LOCAL_MODIFIER_LIST ( localModifier )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(LOCAL_MODIFIER_LIST, "LOCAL_MODIFIER_LIST"), root_1)

                    # Java.g:452:34: ( localModifier )*
                    while stream_localModifier.hasNext():
                        self._adaptor.addChild(root_1, stream_localModifier.nextTree())


                    stream_localModifier.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 39, localModifierList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "localModifierList"

    class localModifier_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.localModifier_return, self).__init__()

            self.tree = None




    # $ANTLR start "localModifier"
    # Java.g:455:1: localModifier : ( FINAL | annotation );
    def localModifier(self, ):

        retval = self.localModifier_return()
        retval.start = self.input.LT(1)
        localModifier_StartIndex = self.input.index()
        root_0 = None

        FINAL164 = None
        annotation165 = None


        FINAL164_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 40):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:456:5: ( FINAL | annotation )
                alt56 = 2
                LA56_0 = self.input.LA(1)

                if (LA56_0 == FINAL) :
                    alt56 = 1
                elif (LA56_0 == AT) :
                    alt56 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 56, 0, self.input)

                    raise nvae

                if alt56 == 1:
                    # Java.g:456:9: FINAL
                    pass 
                    root_0 = self._adaptor.nil()

                    FINAL164=self.match(self.input, FINAL, self.FOLLOW_FINAL_in_localModifier6799)
                    if self._state.backtracking == 0:

                        FINAL164_tree = self._adaptor.createWithPayload(FINAL164)
                        self._adaptor.addChild(root_0, FINAL164_tree)



                elif alt56 == 2:
                    # Java.g:457:9: annotation
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_annotation_in_localModifier6809)
                    annotation165 = self.annotation()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotation165.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 40, localModifier_StartIndex, success)

            pass
        return retval

    # $ANTLR end "localModifier"

    class type_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.type_return, self).__init__()

            self.tree = None




    # $ANTLR start "type"
    # Java.g:460:1: type : ( simpleType | objectType );
    def type(self, ):

        retval = self.type_return()
        retval.start = self.input.LT(1)
        type_StartIndex = self.input.index()
        root_0 = None

        simpleType166 = None

        objectType167 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 41):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:461:5: ( simpleType | objectType )
                alt57 = 2
                LA57_0 = self.input.LA(1)

                if (LA57_0 == BOOLEAN or LA57_0 == BYTE or LA57_0 == CHAR or LA57_0 == DOUBLE or LA57_0 == FLOAT or (INT <= LA57_0 <= LONG) or LA57_0 == SHORT) :
                    alt57 = 1
                elif (LA57_0 == IDENT) :
                    alt57 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 57, 0, self.input)

                    raise nvae

                if alt57 == 1:
                    # Java.g:461:9: simpleType
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_simpleType_in_type6828)
                    simpleType166 = self.simpleType()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, simpleType166.tree)


                elif alt57 == 2:
                    # Java.g:462:9: objectType
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_objectType_in_type6838)
                    objectType167 = self.objectType()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, objectType167.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 41, type_StartIndex, success)

            pass
        return retval

    # $ANTLR end "type"

    class simpleType_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.simpleType_return, self).__init__()

            self.tree = None




    # $ANTLR start "simpleType"
    # Java.g:465:1: simpleType : primitiveType ( arrayDeclaratorList )? -> ^( TYPE primitiveType ( arrayDeclaratorList )? ) ;
    def simpleType(self, ):

        retval = self.simpleType_return()
        retval.start = self.input.LT(1)
        simpleType_StartIndex = self.input.index()
        root_0 = None

        primitiveType168 = None

        arrayDeclaratorList169 = None


        stream_arrayDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclaratorList")
        stream_primitiveType = RewriteRuleSubtreeStream(self._adaptor, "rule primitiveType")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 42):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:466:5: ( primitiveType ( arrayDeclaratorList )? -> ^( TYPE primitiveType ( arrayDeclaratorList )? ) )
                # Java.g:466:9: primitiveType ( arrayDeclaratorList )?
                pass 
                self._state.following.append(self.FOLLOW_primitiveType_in_simpleType6858)
                primitiveType168 = self.primitiveType()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_primitiveType.add(primitiveType168.tree)
                # Java.g:466:23: ( arrayDeclaratorList )?
                alt58 = 2
                LA58_0 = self.input.LA(1)

                if (LA58_0 == LBRACK) :
                    LA58_1 = self.input.LA(2)

                    if (LA58_1 == RBRACK) :
                        LA58_3 = self.input.LA(3)

                        if (self.synpred76_Java()) :
                            alt58 = 1
                if alt58 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_simpleType6860)
                    arrayDeclaratorList169 = self.arrayDeclaratorList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_arrayDeclaratorList.add(arrayDeclaratorList169.tree)




                # AST Rewrite
                # elements: primitiveType, arrayDeclaratorList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 467:9: -> ^( TYPE primitiveType ( arrayDeclaratorList )? )
                    # Java.g:467:13: ^( TYPE primitiveType ( arrayDeclaratorList )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(TYPE, "TYPE"), root_1)

                    self._adaptor.addChild(root_1, stream_primitiveType.nextTree())
                    # Java.g:467:34: ( arrayDeclaratorList )?
                    if stream_arrayDeclaratorList.hasNext():
                        self._adaptor.addChild(root_1, stream_arrayDeclaratorList.nextTree())


                    stream_arrayDeclaratorList.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 42, simpleType_StartIndex, success)

            pass
        return retval

    # $ANTLR end "simpleType"

    class objectType_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.objectType_return, self).__init__()

            self.tree = None




    # $ANTLR start "objectType"
    # Java.g:470:1: objectType : qualifiedTypeIdent ( arrayDeclaratorList )? -> ^( TYPE qualifiedTypeIdent ( arrayDeclaratorList )? ) ;
    def objectType(self, ):

        retval = self.objectType_return()
        retval.start = self.input.LT(1)
        objectType_StartIndex = self.input.index()
        root_0 = None

        qualifiedTypeIdent170 = None

        arrayDeclaratorList171 = None


        stream_arrayDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclaratorList")
        stream_qualifiedTypeIdent = RewriteRuleSubtreeStream(self._adaptor, "rule qualifiedTypeIdent")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 43):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:471:5: ( qualifiedTypeIdent ( arrayDeclaratorList )? -> ^( TYPE qualifiedTypeIdent ( arrayDeclaratorList )? ) )
                # Java.g:471:9: qualifiedTypeIdent ( arrayDeclaratorList )?
                pass 
                self._state.following.append(self.FOLLOW_qualifiedTypeIdent_in_objectType6901)
                qualifiedTypeIdent170 = self.qualifiedTypeIdent()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_qualifiedTypeIdent.add(qualifiedTypeIdent170.tree)
                # Java.g:471:28: ( arrayDeclaratorList )?
                alt59 = 2
                LA59_0 = self.input.LA(1)

                if (LA59_0 == LBRACK) :
                    LA59_1 = self.input.LA(2)

                    if (self.synpred77_Java()) :
                        alt59 = 1
                if alt59 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_objectType6903)
                    arrayDeclaratorList171 = self.arrayDeclaratorList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_arrayDeclaratorList.add(arrayDeclaratorList171.tree)




                # AST Rewrite
                # elements: arrayDeclaratorList, qualifiedTypeIdent
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 472:9: -> ^( TYPE qualifiedTypeIdent ( arrayDeclaratorList )? )
                    # Java.g:472:13: ^( TYPE qualifiedTypeIdent ( arrayDeclaratorList )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(TYPE, "TYPE"), root_1)

                    self._adaptor.addChild(root_1, stream_qualifiedTypeIdent.nextTree())
                    # Java.g:472:39: ( arrayDeclaratorList )?
                    if stream_arrayDeclaratorList.hasNext():
                        self._adaptor.addChild(root_1, stream_arrayDeclaratorList.nextTree())


                    stream_arrayDeclaratorList.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 43, objectType_StartIndex, success)

            pass
        return retval

    # $ANTLR end "objectType"

    class objectTypeSimplified_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.objectTypeSimplified_return, self).__init__()

            self.tree = None




    # $ANTLR start "objectTypeSimplified"
    # Java.g:475:1: objectTypeSimplified : qualifiedTypeIdentSimplified ( arrayDeclaratorList )? -> ^( TYPE qualifiedTypeIdentSimplified ( arrayDeclaratorList )? ) ;
    def objectTypeSimplified(self, ):

        retval = self.objectTypeSimplified_return()
        retval.start = self.input.LT(1)
        objectTypeSimplified_StartIndex = self.input.index()
        root_0 = None

        qualifiedTypeIdentSimplified172 = None

        arrayDeclaratorList173 = None


        stream_arrayDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclaratorList")
        stream_qualifiedTypeIdentSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule qualifiedTypeIdentSimplified")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 44):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:476:5: ( qualifiedTypeIdentSimplified ( arrayDeclaratorList )? -> ^( TYPE qualifiedTypeIdentSimplified ( arrayDeclaratorList )? ) )
                # Java.g:476:9: qualifiedTypeIdentSimplified ( arrayDeclaratorList )?
                pass 
                self._state.following.append(self.FOLLOW_qualifiedTypeIdentSimplified_in_objectTypeSimplified6943)
                qualifiedTypeIdentSimplified172 = self.qualifiedTypeIdentSimplified()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_qualifiedTypeIdentSimplified.add(qualifiedTypeIdentSimplified172.tree)
                # Java.g:476:38: ( arrayDeclaratorList )?
                alt60 = 2
                LA60_0 = self.input.LA(1)

                if (LA60_0 == LBRACK) :
                    alt60 = 1
                if alt60 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_objectTypeSimplified6945)
                    arrayDeclaratorList173 = self.arrayDeclaratorList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_arrayDeclaratorList.add(arrayDeclaratorList173.tree)




                # AST Rewrite
                # elements: qualifiedTypeIdentSimplified, arrayDeclaratorList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 477:9: -> ^( TYPE qualifiedTypeIdentSimplified ( arrayDeclaratorList )? )
                    # Java.g:477:13: ^( TYPE qualifiedTypeIdentSimplified ( arrayDeclaratorList )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(TYPE, "TYPE"), root_1)

                    self._adaptor.addChild(root_1, stream_qualifiedTypeIdentSimplified.nextTree())
                    # Java.g:477:49: ( arrayDeclaratorList )?
                    if stream_arrayDeclaratorList.hasNext():
                        self._adaptor.addChild(root_1, stream_arrayDeclaratorList.nextTree())


                    stream_arrayDeclaratorList.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 44, objectTypeSimplified_StartIndex, success)

            pass
        return retval

    # $ANTLR end "objectTypeSimplified"

    class qualifiedTypeIdent_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.qualifiedTypeIdent_return, self).__init__()

            self.tree = None




    # $ANTLR start "qualifiedTypeIdent"
    # Java.g:480:1: qualifiedTypeIdent : typeIdent ( DOT typeIdent )* -> ^( QUALIFIED_TYPE_IDENT ( typeIdent )+ ) ;
    def qualifiedTypeIdent(self, ):

        retval = self.qualifiedTypeIdent_return()
        retval.start = self.input.LT(1)
        qualifiedTypeIdent_StartIndex = self.input.index()
        root_0 = None

        DOT175 = None
        typeIdent174 = None

        typeIdent176 = None


        DOT175_tree = None
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")
        stream_typeIdent = RewriteRuleSubtreeStream(self._adaptor, "rule typeIdent")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 45):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:481:5: ( typeIdent ( DOT typeIdent )* -> ^( QUALIFIED_TYPE_IDENT ( typeIdent )+ ) )
                # Java.g:481:9: typeIdent ( DOT typeIdent )*
                pass 
                self._state.following.append(self.FOLLOW_typeIdent_in_qualifiedTypeIdent6985)
                typeIdent174 = self.typeIdent()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_typeIdent.add(typeIdent174.tree)
                # Java.g:481:19: ( DOT typeIdent )*
                while True: #loop61
                    alt61 = 2
                    LA61_0 = self.input.LA(1)

                    if (LA61_0 == DOT) :
                        LA61_2 = self.input.LA(2)

                        if (self.synpred79_Java()) :
                            alt61 = 1




                    if alt61 == 1:
                        # Java.g:481:20: DOT typeIdent
                        pass 
                        DOT175=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedTypeIdent6988) 
                        if self._state.backtracking == 0:
                            stream_DOT.add(DOT175)
                        self._state.following.append(self.FOLLOW_typeIdent_in_qualifiedTypeIdent6990)
                        typeIdent176 = self.typeIdent()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_typeIdent.add(typeIdent176.tree)


                    else:
                        break #loop61

                # AST Rewrite
                # elements: typeIdent
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 482:9: -> ^( QUALIFIED_TYPE_IDENT ( typeIdent )+ )
                    # Java.g:482:13: ^( QUALIFIED_TYPE_IDENT ( typeIdent )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"), root_1)

                    # Java.g:482:36: ( typeIdent )+
                    if not (stream_typeIdent.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_typeIdent.hasNext():
                        self._adaptor.addChild(root_1, stream_typeIdent.nextTree())


                    stream_typeIdent.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 45, qualifiedTypeIdent_StartIndex, success)

            pass
        return retval

    # $ANTLR end "qualifiedTypeIdent"

    class qualifiedTypeIdentSimplified_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.qualifiedTypeIdentSimplified_return, self).__init__()

            self.tree = None




    # $ANTLR start "qualifiedTypeIdentSimplified"
    # Java.g:485:1: qualifiedTypeIdentSimplified : typeIdentSimplified ( DOT typeIdentSimplified )* -> ^( QUALIFIED_TYPE_IDENT ( typeIdentSimplified )+ ) ;
    def qualifiedTypeIdentSimplified(self, ):

        retval = self.qualifiedTypeIdentSimplified_return()
        retval.start = self.input.LT(1)
        qualifiedTypeIdentSimplified_StartIndex = self.input.index()
        root_0 = None

        DOT178 = None
        typeIdentSimplified177 = None

        typeIdentSimplified179 = None


        DOT178_tree = None
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")
        stream_typeIdentSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule typeIdentSimplified")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 46):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:486:5: ( typeIdentSimplified ( DOT typeIdentSimplified )* -> ^( QUALIFIED_TYPE_IDENT ( typeIdentSimplified )+ ) )
                # Java.g:486:9: typeIdentSimplified ( DOT typeIdentSimplified )*
                pass 
                self._state.following.append(self.FOLLOW_typeIdentSimplified_in_qualifiedTypeIdentSimplified7029)
                typeIdentSimplified177 = self.typeIdentSimplified()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_typeIdentSimplified.add(typeIdentSimplified177.tree)
                # Java.g:486:29: ( DOT typeIdentSimplified )*
                while True: #loop62
                    alt62 = 2
                    LA62_0 = self.input.LA(1)

                    if (LA62_0 == DOT) :
                        alt62 = 1


                    if alt62 == 1:
                        # Java.g:486:30: DOT typeIdentSimplified
                        pass 
                        DOT178=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedTypeIdentSimplified7032) 
                        if self._state.backtracking == 0:
                            stream_DOT.add(DOT178)
                        self._state.following.append(self.FOLLOW_typeIdentSimplified_in_qualifiedTypeIdentSimplified7034)
                        typeIdentSimplified179 = self.typeIdentSimplified()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_typeIdentSimplified.add(typeIdentSimplified179.tree)


                    else:
                        break #loop62

                # AST Rewrite
                # elements: typeIdentSimplified
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 487:9: -> ^( QUALIFIED_TYPE_IDENT ( typeIdentSimplified )+ )
                    # Java.g:487:13: ^( QUALIFIED_TYPE_IDENT ( typeIdentSimplified )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"), root_1)

                    # Java.g:487:36: ( typeIdentSimplified )+
                    if not (stream_typeIdentSimplified.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_typeIdentSimplified.hasNext():
                        self._adaptor.addChild(root_1, stream_typeIdentSimplified.nextTree())


                    stream_typeIdentSimplified.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 46, qualifiedTypeIdentSimplified_StartIndex, success)

            pass
        return retval

    # $ANTLR end "qualifiedTypeIdentSimplified"

    class typeIdent_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.typeIdent_return, self).__init__()

            self.tree = None




    # $ANTLR start "typeIdent"
    # Java.g:490:1: typeIdent : IDENT ( genericTypeArgumentList )? ;
    def typeIdent(self, ):

        retval = self.typeIdent_return()
        retval.start = self.input.LT(1)
        typeIdent_StartIndex = self.input.index()
        root_0 = None

        IDENT180 = None
        genericTypeArgumentList181 = None


        IDENT180_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 47):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:491:5: ( IDENT ( genericTypeArgumentList )? )
                # Java.g:491:9: IDENT ( genericTypeArgumentList )?
                pass 
                root_0 = self._adaptor.nil()

                IDENT180=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_typeIdent7073)
                if self._state.backtracking == 0:

                    IDENT180_tree = self._adaptor.createWithPayload(IDENT180)
                    root_0 = self._adaptor.becomeRoot(IDENT180_tree, root_0)

                # Java.g:491:16: ( genericTypeArgumentList )?
                alt63 = 2
                LA63_0 = self.input.LA(1)

                if (LA63_0 == LESS_THAN) :
                    alt63 = 1
                if alt63 == 1:
                    # Java.g:0:0: genericTypeArgumentList
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeArgumentList_in_typeIdent7076)
                    genericTypeArgumentList181 = self.genericTypeArgumentList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, genericTypeArgumentList181.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 47, typeIdent_StartIndex, success)

            pass
        return retval

    # $ANTLR end "typeIdent"

    class typeIdentSimplified_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.typeIdentSimplified_return, self).__init__()

            self.tree = None




    # $ANTLR start "typeIdentSimplified"
    # Java.g:494:1: typeIdentSimplified : IDENT ( genericTypeArgumentListSimplified )? ;
    def typeIdentSimplified(self, ):

        retval = self.typeIdentSimplified_return()
        retval.start = self.input.LT(1)
        typeIdentSimplified_StartIndex = self.input.index()
        root_0 = None

        IDENT182 = None
        genericTypeArgumentListSimplified183 = None


        IDENT182_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 48):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:495:5: ( IDENT ( genericTypeArgumentListSimplified )? )
                # Java.g:495:9: IDENT ( genericTypeArgumentListSimplified )?
                pass 
                root_0 = self._adaptor.nil()

                IDENT182=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_typeIdentSimplified7096)
                if self._state.backtracking == 0:

                    IDENT182_tree = self._adaptor.createWithPayload(IDENT182)
                    root_0 = self._adaptor.becomeRoot(IDENT182_tree, root_0)

                # Java.g:495:16: ( genericTypeArgumentListSimplified )?
                alt64 = 2
                LA64_0 = self.input.LA(1)

                if (LA64_0 == LESS_THAN) :
                    alt64 = 1
                if alt64 == 1:
                    # Java.g:0:0: genericTypeArgumentListSimplified
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_typeIdentSimplified7099)
                    genericTypeArgumentListSimplified183 = self.genericTypeArgumentListSimplified()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, genericTypeArgumentListSimplified183.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 48, typeIdentSimplified_StartIndex, success)

            pass
        return retval

    # $ANTLR end "typeIdentSimplified"

    class primitiveType_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.primitiveType_return, self).__init__()

            self.tree = None




    # $ANTLR start "primitiveType"
    # Java.g:498:1: primitiveType : ( BOOLEAN | CHAR | BYTE | SHORT | INT | LONG | FLOAT | DOUBLE );
    def primitiveType(self, ):

        retval = self.primitiveType_return()
        retval.start = self.input.LT(1)
        primitiveType_StartIndex = self.input.index()
        root_0 = None

        set184 = None

        set184_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 49):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:499:5: ( BOOLEAN | CHAR | BYTE | SHORT | INT | LONG | FLOAT | DOUBLE )
                # Java.g:
                pass 
                root_0 = self._adaptor.nil()

                set184 = self.input.LT(1)
                if self.input.LA(1) == BOOLEAN or self.input.LA(1) == BYTE or self.input.LA(1) == CHAR or self.input.LA(1) == DOUBLE or self.input.LA(1) == FLOAT or (INT <= self.input.LA(1) <= LONG) or self.input.LA(1) == SHORT:
                    self.input.consume()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, self._adaptor.createWithPayload(set184))
                    self._state.errorRecovery = False

                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    mse = MismatchedSetException(None, self.input)
                    raise mse





                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 49, primitiveType_StartIndex, success)

            pass
        return retval

    # $ANTLR end "primitiveType"

    class genericTypeArgumentList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeArgumentList_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeArgumentList"
    # Java.g:509:1: genericTypeArgumentList : LESS_THAN genericTypeArgument ( COMMA genericTypeArgument )* genericTypeListClosing -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgument )+ ) ;
    def genericTypeArgumentList(self, ):

        retval = self.genericTypeArgumentList_return()
        retval.start = self.input.LT(1)
        genericTypeArgumentList_StartIndex = self.input.index()
        root_0 = None

        LESS_THAN185 = None
        COMMA187 = None
        genericTypeArgument186 = None

        genericTypeArgument188 = None

        genericTypeListClosing189 = None


        LESS_THAN185_tree = None
        COMMA187_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_LESS_THAN = RewriteRuleTokenStream(self._adaptor, "token LESS_THAN")
        stream_genericTypeArgument = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgument")
        stream_genericTypeListClosing = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeListClosing")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 50):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:510:5: ( LESS_THAN genericTypeArgument ( COMMA genericTypeArgument )* genericTypeListClosing -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgument )+ ) )
                # Java.g:510:9: LESS_THAN genericTypeArgument ( COMMA genericTypeArgument )* genericTypeListClosing
                pass 
                LESS_THAN185=self.match(self.input, LESS_THAN, self.FOLLOW_LESS_THAN_in_genericTypeArgumentList7208) 
                if self._state.backtracking == 0:
                    stream_LESS_THAN.add(LESS_THAN185)
                self._state.following.append(self.FOLLOW_genericTypeArgument_in_genericTypeArgumentList7210)
                genericTypeArgument186 = self.genericTypeArgument()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeArgument.add(genericTypeArgument186.tree)
                # Java.g:510:39: ( COMMA genericTypeArgument )*
                while True: #loop65
                    alt65 = 2
                    LA65_0 = self.input.LA(1)

                    if (LA65_0 == COMMA) :
                        LA65_2 = self.input.LA(2)

                        if (self.synpred90_Java()) :
                            alt65 = 1




                    if alt65 == 1:
                        # Java.g:510:40: COMMA genericTypeArgument
                        pass 
                        COMMA187=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_genericTypeArgumentList7213) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA187)
                        self._state.following.append(self.FOLLOW_genericTypeArgument_in_genericTypeArgumentList7215)
                        genericTypeArgument188 = self.genericTypeArgument()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericTypeArgument.add(genericTypeArgument188.tree)


                    else:
                        break #loop65
                self._state.following.append(self.FOLLOW_genericTypeListClosing_in_genericTypeArgumentList7219)
                genericTypeListClosing189 = self.genericTypeListClosing()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeListClosing.add(genericTypeListClosing189.tree)

                # AST Rewrite
                # elements: genericTypeArgument
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 511:9: -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgument )+ )
                    # Java.g:511:13: ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgument )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(GENERIC_TYPE_ARG_LIST, LESS_THAN185, "GENERIC_TYPE_ARG_LIST"), root_1)

                    # Java.g:511:74: ( genericTypeArgument )+
                    if not (stream_genericTypeArgument.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_genericTypeArgument.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeArgument.nextTree())


                    stream_genericTypeArgument.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 50, genericTypeArgumentList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeArgumentList"

    class genericTypeArgument_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeArgument_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeArgument"
    # Java.g:514:1: genericTypeArgument : ( type | QUESTION ( genericWildcardBoundType )? -> ^( QUESTION ( genericWildcardBoundType )? ) );
    def genericTypeArgument(self, ):

        retval = self.genericTypeArgument_return()
        retval.start = self.input.LT(1)
        genericTypeArgument_StartIndex = self.input.index()
        root_0 = None

        QUESTION191 = None
        type190 = None

        genericWildcardBoundType192 = None


        QUESTION191_tree = None
        stream_QUESTION = RewriteRuleTokenStream(self._adaptor, "token QUESTION")
        stream_genericWildcardBoundType = RewriteRuleSubtreeStream(self._adaptor, "rule genericWildcardBoundType")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 51):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:515:5: ( type | QUESTION ( genericWildcardBoundType )? -> ^( QUESTION ( genericWildcardBoundType )? ) )
                alt67 = 2
                LA67_0 = self.input.LA(1)

                if (LA67_0 == BOOLEAN or LA67_0 == BYTE or LA67_0 == CHAR or LA67_0 == DOUBLE or LA67_0 == FLOAT or (INT <= LA67_0 <= LONG) or LA67_0 == SHORT or LA67_0 == IDENT) :
                    alt67 = 1
                elif (LA67_0 == QUESTION) :
                    alt67 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 67, 0, self.input)

                    raise nvae

                if alt67 == 1:
                    # Java.g:515:9: type
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_type_in_genericTypeArgument7257)
                    type190 = self.type()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, type190.tree)


                elif alt67 == 2:
                    # Java.g:516:9: QUESTION ( genericWildcardBoundType )?
                    pass 
                    QUESTION191=self.match(self.input, QUESTION, self.FOLLOW_QUESTION_in_genericTypeArgument7267) 
                    if self._state.backtracking == 0:
                        stream_QUESTION.add(QUESTION191)
                    # Java.g:516:18: ( genericWildcardBoundType )?
                    alt66 = 2
                    LA66_0 = self.input.LA(1)

                    if (LA66_0 == EXTENDS) :
                        LA66_1 = self.input.LA(2)

                        if (LA66_1 == BOOLEAN or LA66_1 == BYTE or LA66_1 == CHAR or LA66_1 == DOUBLE or LA66_1 == FLOAT or (INT <= LA66_1 <= LONG) or LA66_1 == SHORT) :
                            LA66_4 = self.input.LA(3)

                            if (self.synpred92_Java()) :
                                alt66 = 1
                        elif (LA66_1 == IDENT) :
                            LA66_5 = self.input.LA(3)

                            if (self.synpred92_Java()) :
                                alt66 = 1
                    elif (LA66_0 == SUPER) :
                        LA66_3 = self.input.LA(2)

                        if (LA66_3 == BOOLEAN or LA66_3 == BYTE or LA66_3 == CHAR or LA66_3 == DOUBLE or LA66_3 == FLOAT or (INT <= LA66_3 <= LONG) or LA66_3 == SHORT or LA66_3 == IDENT) :
                            alt66 = 1
                    if alt66 == 1:
                        # Java.g:0:0: genericWildcardBoundType
                        pass 
                        self._state.following.append(self.FOLLOW_genericWildcardBoundType_in_genericTypeArgument7269)
                        genericWildcardBoundType192 = self.genericWildcardBoundType()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericWildcardBoundType.add(genericWildcardBoundType192.tree)




                    # AST Rewrite
                    # elements: QUESTION, genericWildcardBoundType
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 517:9: -> ^( QUESTION ( genericWildcardBoundType )? )
                        # Java.g:517:13: ^( QUESTION ( genericWildcardBoundType )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_QUESTION.nextNode(), root_1)

                        # Java.g:517:24: ( genericWildcardBoundType )?
                        if stream_genericWildcardBoundType.hasNext():
                            self._adaptor.addChild(root_1, stream_genericWildcardBoundType.nextTree())


                        stream_genericWildcardBoundType.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 51, genericTypeArgument_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeArgument"

    class genericWildcardBoundType_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericWildcardBoundType_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericWildcardBoundType"
    # Java.g:520:1: genericWildcardBoundType : ( EXTENDS | SUPER ) type ;
    def genericWildcardBoundType(self, ):

        retval = self.genericWildcardBoundType_return()
        retval.start = self.input.LT(1)
        genericWildcardBoundType_StartIndex = self.input.index()
        root_0 = None

        set193 = None
        type194 = None


        set193_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 52):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:521:5: ( ( EXTENDS | SUPER ) type )
                # Java.g:521:9: ( EXTENDS | SUPER ) type
                pass 
                root_0 = self._adaptor.nil()

                set193 = self.input.LT(1)
                set193 = self.input.LT(1)
                if self.input.LA(1) == EXTENDS or self.input.LA(1) == SUPER:
                    self.input.consume()
                    if self._state.backtracking == 0:
                        root_0 = self._adaptor.becomeRoot(self._adaptor.createWithPayload(set193), root_0)
                    self._state.errorRecovery = False

                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    mse = MismatchedSetException(None, self.input)
                    raise mse


                self._state.following.append(self.FOLLOW_type_in_genericWildcardBoundType7316)
                type194 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, type194.tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 52, genericWildcardBoundType_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericWildcardBoundType"

    class genericTypeArgumentListSimplified_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeArgumentListSimplified_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeArgumentListSimplified"
    # Java.g:524:1: genericTypeArgumentListSimplified : LESS_THAN genericTypeArgumentSimplified ( COMMA genericTypeArgumentSimplified )* genericTypeListClosing -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgumentSimplified )+ ) ;
    def genericTypeArgumentListSimplified(self, ):

        retval = self.genericTypeArgumentListSimplified_return()
        retval.start = self.input.LT(1)
        genericTypeArgumentListSimplified_StartIndex = self.input.index()
        root_0 = None

        LESS_THAN195 = None
        COMMA197 = None
        genericTypeArgumentSimplified196 = None

        genericTypeArgumentSimplified198 = None

        genericTypeListClosing199 = None


        LESS_THAN195_tree = None
        COMMA197_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_LESS_THAN = RewriteRuleTokenStream(self._adaptor, "token LESS_THAN")
        stream_genericTypeArgumentSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentSimplified")
        stream_genericTypeListClosing = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeListClosing")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 53):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:525:5: ( LESS_THAN genericTypeArgumentSimplified ( COMMA genericTypeArgumentSimplified )* genericTypeListClosing -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgumentSimplified )+ ) )
                # Java.g:525:9: LESS_THAN genericTypeArgumentSimplified ( COMMA genericTypeArgumentSimplified )* genericTypeListClosing
                pass 
                LESS_THAN195=self.match(self.input, LESS_THAN, self.FOLLOW_LESS_THAN_in_genericTypeArgumentListSimplified7335) 
                if self._state.backtracking == 0:
                    stream_LESS_THAN.add(LESS_THAN195)
                self._state.following.append(self.FOLLOW_genericTypeArgumentSimplified_in_genericTypeArgumentListSimplified7337)
                genericTypeArgumentSimplified196 = self.genericTypeArgumentSimplified()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeArgumentSimplified.add(genericTypeArgumentSimplified196.tree)
                # Java.g:525:49: ( COMMA genericTypeArgumentSimplified )*
                while True: #loop68
                    alt68 = 2
                    LA68_0 = self.input.LA(1)

                    if (LA68_0 == COMMA) :
                        alt68 = 1


                    if alt68 == 1:
                        # Java.g:525:50: COMMA genericTypeArgumentSimplified
                        pass 
                        COMMA197=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_genericTypeArgumentListSimplified7340) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA197)
                        self._state.following.append(self.FOLLOW_genericTypeArgumentSimplified_in_genericTypeArgumentListSimplified7342)
                        genericTypeArgumentSimplified198 = self.genericTypeArgumentSimplified()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericTypeArgumentSimplified.add(genericTypeArgumentSimplified198.tree)


                    else:
                        break #loop68
                self._state.following.append(self.FOLLOW_genericTypeListClosing_in_genericTypeArgumentListSimplified7346)
                genericTypeListClosing199 = self.genericTypeListClosing()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_genericTypeListClosing.add(genericTypeListClosing199.tree)

                # AST Rewrite
                # elements: genericTypeArgumentSimplified
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 526:9: -> ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgumentSimplified )+ )
                    # Java.g:526:13: ^( GENERIC_TYPE_ARG_LIST[$LESS_THAN, \"GENERIC_TYPE_ARG_LIST\"] ( genericTypeArgumentSimplified )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(GENERIC_TYPE_ARG_LIST, LESS_THAN195, "GENERIC_TYPE_ARG_LIST"), root_1)

                    # Java.g:526:74: ( genericTypeArgumentSimplified )+
                    if not (stream_genericTypeArgumentSimplified.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_genericTypeArgumentSimplified.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeArgumentSimplified.nextTree())


                    stream_genericTypeArgumentSimplified.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 53, genericTypeArgumentListSimplified_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeArgumentListSimplified"

    class genericTypeArgumentSimplified_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.genericTypeArgumentSimplified_return, self).__init__()

            self.tree = None




    # $ANTLR start "genericTypeArgumentSimplified"
    # Java.g:529:1: genericTypeArgumentSimplified : ( type | QUESTION );
    def genericTypeArgumentSimplified(self, ):

        retval = self.genericTypeArgumentSimplified_return()
        retval.start = self.input.LT(1)
        genericTypeArgumentSimplified_StartIndex = self.input.index()
        root_0 = None

        QUESTION201 = None
        type200 = None


        QUESTION201_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 54):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:530:5: ( type | QUESTION )
                alt69 = 2
                LA69_0 = self.input.LA(1)

                if (LA69_0 == BOOLEAN or LA69_0 == BYTE or LA69_0 == CHAR or LA69_0 == DOUBLE or LA69_0 == FLOAT or (INT <= LA69_0 <= LONG) or LA69_0 == SHORT or LA69_0 == IDENT) :
                    alt69 = 1
                elif (LA69_0 == QUESTION) :
                    alt69 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 69, 0, self.input)

                    raise nvae

                if alt69 == 1:
                    # Java.g:530:9: type
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_type_in_genericTypeArgumentSimplified7384)
                    type200 = self.type()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, type200.tree)


                elif alt69 == 2:
                    # Java.g:531:9: QUESTION
                    pass 
                    root_0 = self._adaptor.nil()

                    QUESTION201=self.match(self.input, QUESTION, self.FOLLOW_QUESTION_in_genericTypeArgumentSimplified7394)
                    if self._state.backtracking == 0:

                        QUESTION201_tree = self._adaptor.createWithPayload(QUESTION201)
                        self._adaptor.addChild(root_0, QUESTION201_tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 54, genericTypeArgumentSimplified_StartIndex, success)

            pass
        return retval

    # $ANTLR end "genericTypeArgumentSimplified"

    class qualifiedIdentList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.qualifiedIdentList_return, self).__init__()

            self.tree = None




    # $ANTLR start "qualifiedIdentList"
    # Java.g:534:1: qualifiedIdentList : qualifiedIdentifier ( COMMA qualifiedIdentifier )* ;
    def qualifiedIdentList(self, ):

        retval = self.qualifiedIdentList_return()
        retval.start = self.input.LT(1)
        qualifiedIdentList_StartIndex = self.input.index()
        root_0 = None

        COMMA203 = None
        qualifiedIdentifier202 = None

        qualifiedIdentifier204 = None


        COMMA203_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 55):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:535:5: ( qualifiedIdentifier ( COMMA qualifiedIdentifier )* )
                # Java.g:535:9: qualifiedIdentifier ( COMMA qualifiedIdentifier )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_qualifiedIdentList7413)
                qualifiedIdentifier202 = self.qualifiedIdentifier()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, qualifiedIdentifier202.tree)
                # Java.g:535:29: ( COMMA qualifiedIdentifier )*
                while True: #loop70
                    alt70 = 2
                    LA70_0 = self.input.LA(1)

                    if (LA70_0 == COMMA) :
                        alt70 = 1


                    if alt70 == 1:
                        # Java.g:535:30: COMMA qualifiedIdentifier
                        pass 
                        COMMA203=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_qualifiedIdentList7416)
                        self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_qualifiedIdentList7419)
                        qualifiedIdentifier204 = self.qualifiedIdentifier()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, qualifiedIdentifier204.tree)


                    else:
                        break #loop70



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 55, qualifiedIdentList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "qualifiedIdentList"

    class formalParameterList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.formalParameterList_return, self).__init__()

            self.tree = None




    # $ANTLR start "formalParameterList"
    # Java.g:538:1: formalParameterList : LPAREN ( formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )? -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? ) | formalParameterVarArgDecl -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl ) | -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ) ) RPAREN ;
    def formalParameterList(self, ):

        retval = self.formalParameterList_return()
        retval.start = self.input.LT(1)
        formalParameterList_StartIndex = self.input.index()
        root_0 = None

        LPAREN205 = None
        COMMA207 = None
        COMMA209 = None
        RPAREN212 = None
        formalParameterStandardDecl206 = None

        formalParameterStandardDecl208 = None

        formalParameterVarArgDecl210 = None

        formalParameterVarArgDecl211 = None


        LPAREN205_tree = None
        COMMA207_tree = None
        COMMA209_tree = None
        RPAREN212_tree = None
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_formalParameterVarArgDecl = RewriteRuleSubtreeStream(self._adaptor, "rule formalParameterVarArgDecl")
        stream_formalParameterStandardDecl = RewriteRuleSubtreeStream(self._adaptor, "rule formalParameterStandardDecl")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 56):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:539:5: ( LPAREN ( formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )? -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? ) | formalParameterVarArgDecl -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl ) | -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ) ) RPAREN )
                # Java.g:539:9: LPAREN ( formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )? -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? ) | formalParameterVarArgDecl -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl ) | -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ) ) RPAREN
                pass 
                LPAREN205=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_formalParameterList7440) 
                if self._state.backtracking == 0:
                    stream_LPAREN.add(LPAREN205)
                # Java.g:540:9: ( formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )? -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? ) | formalParameterVarArgDecl -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl ) | -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ) )
                alt73 = 3
                LA73 = self.input.LA(1)
                if LA73 == FINAL:
                    LA73_1 = self.input.LA(2)

                    if (self.synpred99_Java()) :
                        alt73 = 1
                    elif (self.synpred100_Java()) :
                        alt73 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 73, 1, self.input)

                        raise nvae

                elif LA73 == AT:
                    LA73_2 = self.input.LA(2)

                    if (self.synpred99_Java()) :
                        alt73 = 1
                    elif (self.synpred100_Java()) :
                        alt73 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 73, 2, self.input)

                        raise nvae

                elif LA73 == BOOLEAN or LA73 == BYTE or LA73 == CHAR or LA73 == DOUBLE or LA73 == FLOAT or LA73 == INT or LA73 == LONG or LA73 == SHORT:
                    LA73_3 = self.input.LA(2)

                    if (self.synpred99_Java()) :
                        alt73 = 1
                    elif (self.synpred100_Java()) :
                        alt73 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 73, 3, self.input)

                        raise nvae

                elif LA73 == IDENT:
                    LA73_4 = self.input.LA(2)

                    if (self.synpred99_Java()) :
                        alt73 = 1
                    elif (self.synpred100_Java()) :
                        alt73 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 73, 4, self.input)

                        raise nvae

                elif LA73 == RPAREN:
                    alt73 = 3
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 73, 0, self.input)

                    raise nvae

                if alt73 == 1:
                    # Java.g:541:13: formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )?
                    pass 
                    self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_formalParameterList7467)
                    formalParameterStandardDecl206 = self.formalParameterStandardDecl()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_formalParameterStandardDecl.add(formalParameterStandardDecl206.tree)
                    # Java.g:541:41: ( COMMA formalParameterStandardDecl )*
                    while True: #loop71
                        alt71 = 2
                        LA71_0 = self.input.LA(1)

                        if (LA71_0 == COMMA) :
                            LA71_1 = self.input.LA(2)

                            if (self.synpred97_Java()) :
                                alt71 = 1




                        if alt71 == 1:
                            # Java.g:541:42: COMMA formalParameterStandardDecl
                            pass 
                            COMMA207=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_formalParameterList7470) 
                            if self._state.backtracking == 0:
                                stream_COMMA.add(COMMA207)
                            self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_formalParameterList7472)
                            formalParameterStandardDecl208 = self.formalParameterStandardDecl()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_formalParameterStandardDecl.add(formalParameterStandardDecl208.tree)


                        else:
                            break #loop71
                    # Java.g:541:78: ( COMMA formalParameterVarArgDecl )?
                    alt72 = 2
                    LA72_0 = self.input.LA(1)

                    if (LA72_0 == COMMA) :
                        alt72 = 1
                    if alt72 == 1:
                        # Java.g:541:79: COMMA formalParameterVarArgDecl
                        pass 
                        COMMA209=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_formalParameterList7477) 
                        if self._state.backtracking == 0:
                            stream_COMMA.add(COMMA209)
                        self._state.following.append(self.FOLLOW_formalParameterVarArgDecl_in_formalParameterList7479)
                        formalParameterVarArgDecl210 = self.formalParameterVarArgDecl()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_formalParameterVarArgDecl.add(formalParameterVarArgDecl210.tree)




                    # AST Rewrite
                    # elements: formalParameterVarArgDecl, formalParameterStandardDecl
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 542:13: -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? )
                        # Java.g:542:17: ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] ( formalParameterStandardDecl )+ ( formalParameterVarArgDecl )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(FORMAL_PARAM_LIST, LPAREN205, "FORMAL_PARAM_LIST"), root_1)

                        # Java.g:542:67: ( formalParameterStandardDecl )+
                        if not (stream_formalParameterStandardDecl.hasNext()):
                            raise RewriteEarlyExitException()

                        while stream_formalParameterStandardDecl.hasNext():
                            self._adaptor.addChild(root_1, stream_formalParameterStandardDecl.nextTree())


                        stream_formalParameterStandardDecl.reset()
                        # Java.g:542:96: ( formalParameterVarArgDecl )?
                        if stream_formalParameterVarArgDecl.hasNext():
                            self._adaptor.addChild(root_1, stream_formalParameterVarArgDecl.nextTree())


                        stream_formalParameterVarArgDecl.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt73 == 2:
                    # Java.g:544:13: formalParameterVarArgDecl
                    pass 
                    self._state.following.append(self.FOLLOW_formalParameterVarArgDecl_in_formalParameterList7534)
                    formalParameterVarArgDecl211 = self.formalParameterVarArgDecl()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_formalParameterVarArgDecl.add(formalParameterVarArgDecl211.tree)

                    # AST Rewrite
                    # elements: formalParameterVarArgDecl
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 545:13: -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl )
                        # Java.g:545:17: ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] formalParameterVarArgDecl )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(FORMAL_PARAM_LIST, LPAREN205, "FORMAL_PARAM_LIST"), root_1)

                        self._adaptor.addChild(root_1, stream_formalParameterVarArgDecl.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt73 == 3:
                    # Java.g:547:13: 
                    pass 
                    # AST Rewrite
                    # elements: 
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 547:13: -> ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] )
                        # Java.g:547:17: ^( FORMAL_PARAM_LIST[$LPAREN, \"FORMAL_PARAM_LIST\"] )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(FORMAL_PARAM_LIST, LPAREN205, "FORMAL_PARAM_LIST"), root_1)

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0



                RPAREN212=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_formalParameterList7609) 
                if self._state.backtracking == 0:
                    stream_RPAREN.add(RPAREN212)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 56, formalParameterList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "formalParameterList"

    class formalParameterStandardDecl_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.formalParameterStandardDecl_return, self).__init__()

            self.tree = None




    # $ANTLR start "formalParameterStandardDecl"
    # Java.g:552:1: formalParameterStandardDecl : localModifierList type variableDeclaratorId -> ^( FORMAL_PARAM_STD_DECL localModifierList type variableDeclaratorId ) ;
    def formalParameterStandardDecl(self, ):

        retval = self.formalParameterStandardDecl_return()
        retval.start = self.input.LT(1)
        formalParameterStandardDecl_StartIndex = self.input.index()
        root_0 = None

        localModifierList213 = None

        type214 = None

        variableDeclaratorId215 = None


        stream_variableDeclaratorId = RewriteRuleSubtreeStream(self._adaptor, "rule variableDeclaratorId")
        stream_localModifierList = RewriteRuleSubtreeStream(self._adaptor, "rule localModifierList")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 57):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:553:5: ( localModifierList type variableDeclaratorId -> ^( FORMAL_PARAM_STD_DECL localModifierList type variableDeclaratorId ) )
                # Java.g:553:9: localModifierList type variableDeclaratorId
                pass 
                self._state.following.append(self.FOLLOW_localModifierList_in_formalParameterStandardDecl7628)
                localModifierList213 = self.localModifierList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_localModifierList.add(localModifierList213.tree)
                self._state.following.append(self.FOLLOW_type_in_formalParameterStandardDecl7630)
                type214 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_type.add(type214.tree)
                self._state.following.append(self.FOLLOW_variableDeclaratorId_in_formalParameterStandardDecl7632)
                variableDeclaratorId215 = self.variableDeclaratorId()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_variableDeclaratorId.add(variableDeclaratorId215.tree)

                # AST Rewrite
                # elements: localModifierList, type, variableDeclaratorId
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 554:9: -> ^( FORMAL_PARAM_STD_DECL localModifierList type variableDeclaratorId )
                    # Java.g:554:13: ^( FORMAL_PARAM_STD_DECL localModifierList type variableDeclaratorId )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"), root_1)

                    self._adaptor.addChild(root_1, stream_localModifierList.nextTree())
                    self._adaptor.addChild(root_1, stream_type.nextTree())
                    self._adaptor.addChild(root_1, stream_variableDeclaratorId.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 57, formalParameterStandardDecl_StartIndex, success)

            pass
        return retval

    # $ANTLR end "formalParameterStandardDecl"

    class formalParameterVarArgDecl_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.formalParameterVarArgDecl_return, self).__init__()

            self.tree = None




    # $ANTLR start "formalParameterVarArgDecl"
    # Java.g:557:1: formalParameterVarArgDecl : localModifierList type ELLIPSIS variableDeclaratorId -> ^( FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId ) ;
    def formalParameterVarArgDecl(self, ):

        retval = self.formalParameterVarArgDecl_return()
        retval.start = self.input.LT(1)
        formalParameterVarArgDecl_StartIndex = self.input.index()
        root_0 = None

        ELLIPSIS218 = None
        localModifierList216 = None

        type217 = None

        variableDeclaratorId219 = None


        ELLIPSIS218_tree = None
        stream_ELLIPSIS = RewriteRuleTokenStream(self._adaptor, "token ELLIPSIS")
        stream_variableDeclaratorId = RewriteRuleSubtreeStream(self._adaptor, "rule variableDeclaratorId")
        stream_localModifierList = RewriteRuleSubtreeStream(self._adaptor, "rule localModifierList")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 58):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:558:5: ( localModifierList type ELLIPSIS variableDeclaratorId -> ^( FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId ) )
                # Java.g:558:9: localModifierList type ELLIPSIS variableDeclaratorId
                pass 
                self._state.following.append(self.FOLLOW_localModifierList_in_formalParameterVarArgDecl7672)
                localModifierList216 = self.localModifierList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_localModifierList.add(localModifierList216.tree)
                self._state.following.append(self.FOLLOW_type_in_formalParameterVarArgDecl7674)
                type217 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_type.add(type217.tree)
                ELLIPSIS218=self.match(self.input, ELLIPSIS, self.FOLLOW_ELLIPSIS_in_formalParameterVarArgDecl7676) 
                if self._state.backtracking == 0:
                    stream_ELLIPSIS.add(ELLIPSIS218)
                self._state.following.append(self.FOLLOW_variableDeclaratorId_in_formalParameterVarArgDecl7678)
                variableDeclaratorId219 = self.variableDeclaratorId()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_variableDeclaratorId.add(variableDeclaratorId219.tree)

                # AST Rewrite
                # elements: variableDeclaratorId, localModifierList, type
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 559:9: -> ^( FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId )
                    # Java.g:559:13: ^( FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FORMAL_PARAM_VARARG_DECL, "FORMAL_PARAM_VARARG_DECL"), root_1)

                    self._adaptor.addChild(root_1, stream_localModifierList.nextTree())
                    self._adaptor.addChild(root_1, stream_type.nextTree())
                    self._adaptor.addChild(root_1, stream_variableDeclaratorId.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 58, formalParameterVarArgDecl_StartIndex, success)

            pass
        return retval

    # $ANTLR end "formalParameterVarArgDecl"

    class qualifiedIdentifier_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.qualifiedIdentifier_return, self).__init__()

            self.tree = None




    # $ANTLR start "qualifiedIdentifier"
    # Java.g:562:1: qualifiedIdentifier : ( IDENT -> IDENT ) ( DOT ident= IDENT -> ^( DOT $qualifiedIdentifier $ident) )* ;
    def qualifiedIdentifier(self, ):

        retval = self.qualifiedIdentifier_return()
        retval.start = self.input.LT(1)
        qualifiedIdentifier_StartIndex = self.input.index()
        root_0 = None

        ident = None
        IDENT220 = None
        DOT221 = None

        ident_tree = None
        IDENT220_tree = None
        DOT221_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 59):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:563:5: ( ( IDENT -> IDENT ) ( DOT ident= IDENT -> ^( DOT $qualifiedIdentifier $ident) )* )
                # Java.g:563:9: ( IDENT -> IDENT ) ( DOT ident= IDENT -> ^( DOT $qualifiedIdentifier $ident) )*
                pass 
                # Java.g:563:9: ( IDENT -> IDENT )
                # Java.g:563:13: IDENT
                pass 
                IDENT220=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_qualifiedIdentifier7722) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT220)

                # AST Rewrite
                # elements: IDENT
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 563:33: -> IDENT
                    self._adaptor.addChild(root_0, stream_IDENT.nextNode())



                    retval.tree = root_0



                # Java.g:565:9: ( DOT ident= IDENT -> ^( DOT $qualifiedIdentifier $ident) )*
                while True: #loop74
                    alt74 = 2
                    LA74_0 = self.input.LA(1)

                    if (LA74_0 == DOT) :
                        LA74_2 = self.input.LA(2)

                        if (LA74_2 == IDENT) :
                            LA74_3 = self.input.LA(3)

                            if (self.synpred101_Java()) :
                                alt74 = 1






                    if alt74 == 1:
                        # Java.g:565:13: DOT ident= IDENT
                        pass 
                        DOT221=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedIdentifier7765) 
                        if self._state.backtracking == 0:
                            stream_DOT.add(DOT221)
                        ident=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_qualifiedIdentifier7769) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(ident)

                        # AST Rewrite
                        # elements: DOT, qualifiedIdentifier, ident
                        # token labels: ident
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0
                            stream_ident = RewriteRuleTokenStream(self._adaptor, "token ident", ident)

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 565:33: -> ^( DOT $qualifiedIdentifier $ident)
                            # Java.g:565:37: ^( DOT $qualifiedIdentifier $ident)
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_retval.nextTree())
                            self._adaptor.addChild(root_1, stream_ident.nextNode())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    else:
                        break #loop74



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 59, qualifiedIdentifier_StartIndex, success)

            pass
        return retval

    # $ANTLR end "qualifiedIdentifier"

    class annotationList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationList_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationList"
    # Java.g:571:1: annotationList : ( annotation )* -> ^( ANNOTATION_LIST ( annotation )* ) ;
    def annotationList(self, ):

        retval = self.annotationList_return()
        retval.start = self.input.LT(1)
        annotationList_StartIndex = self.input.index()
        root_0 = None

        annotation222 = None


        stream_annotation = RewriteRuleSubtreeStream(self._adaptor, "rule annotation")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 60):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:572:5: ( ( annotation )* -> ^( ANNOTATION_LIST ( annotation )* ) )
                # Java.g:572:9: ( annotation )*
                pass 
                # Java.g:572:9: ( annotation )*
                while True: #loop75
                    alt75 = 2
                    LA75_0 = self.input.LA(1)

                    if (LA75_0 == AT) :
                        LA75_2 = self.input.LA(2)

                        if (LA75_2 == IDENT) :
                            LA75_3 = self.input.LA(3)

                            if (self.synpred102_Java()) :
                                alt75 = 1






                    if alt75 == 1:
                        # Java.g:0:0: annotation
                        pass 
                        self._state.following.append(self.FOLLOW_annotation_in_annotationList7818)
                        annotation222 = self.annotation()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_annotation.add(annotation222.tree)


                    else:
                        break #loop75

                # AST Rewrite
                # elements: annotation
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 573:9: -> ^( ANNOTATION_LIST ( annotation )* )
                    # Java.g:573:13: ^( ANNOTATION_LIST ( annotation )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ANNOTATION_LIST, "ANNOTATION_LIST"), root_1)

                    # Java.g:573:31: ( annotation )*
                    while stream_annotation.hasNext():
                        self._adaptor.addChild(root_1, stream_annotation.nextTree())


                    stream_annotation.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 60, annotationList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationList"

    class annotation_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotation_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotation"
    # Java.g:576:1: annotation : AT qualifiedIdentifier ( annotationInit )? ;
    def annotation(self, ):

        retval = self.annotation_return()
        retval.start = self.input.LT(1)
        annotation_StartIndex = self.input.index()
        root_0 = None

        AT223 = None
        qualifiedIdentifier224 = None

        annotationInit225 = None


        AT223_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 61):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:577:5: ( AT qualifiedIdentifier ( annotationInit )? )
                # Java.g:577:9: AT qualifiedIdentifier ( annotationInit )?
                pass 
                root_0 = self._adaptor.nil()

                AT223=self.match(self.input, AT, self.FOLLOW_AT_in_annotation7856)
                if self._state.backtracking == 0:

                    AT223_tree = self._adaptor.createWithPayload(AT223)
                    root_0 = self._adaptor.becomeRoot(AT223_tree, root_0)

                self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_annotation7859)
                qualifiedIdentifier224 = self.qualifiedIdentifier()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, qualifiedIdentifier224.tree)
                # Java.g:577:33: ( annotationInit )?
                alt76 = 2
                LA76_0 = self.input.LA(1)

                if (LA76_0 == LPAREN) :
                    alt76 = 1
                if alt76 == 1:
                    # Java.g:0:0: annotationInit
                    pass 
                    self._state.following.append(self.FOLLOW_annotationInit_in_annotation7861)
                    annotationInit225 = self.annotationInit()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotationInit225.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 61, annotation_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotation"

    class annotationInit_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationInit_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationInit"
    # Java.g:580:1: annotationInit : LPAREN annotationInitializers RPAREN -> ^( ANNOTATION_INIT_BLOCK[$LPAREN, \"ANNOTATION_INIT_BLOCK\"] annotationInitializers ) ;
    def annotationInit(self, ):

        retval = self.annotationInit_return()
        retval.start = self.input.LT(1)
        annotationInit_StartIndex = self.input.index()
        root_0 = None

        LPAREN226 = None
        RPAREN228 = None
        annotationInitializers227 = None


        LPAREN226_tree = None
        RPAREN228_tree = None
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_annotationInitializers = RewriteRuleSubtreeStream(self._adaptor, "rule annotationInitializers")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 62):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:581:5: ( LPAREN annotationInitializers RPAREN -> ^( ANNOTATION_INIT_BLOCK[$LPAREN, \"ANNOTATION_INIT_BLOCK\"] annotationInitializers ) )
                # Java.g:581:9: LPAREN annotationInitializers RPAREN
                pass 
                LPAREN226=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_annotationInit7881) 
                if self._state.backtracking == 0:
                    stream_LPAREN.add(LPAREN226)
                self._state.following.append(self.FOLLOW_annotationInitializers_in_annotationInit7883)
                annotationInitializers227 = self.annotationInitializers()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_annotationInitializers.add(annotationInitializers227.tree)
                RPAREN228=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_annotationInit7885) 
                if self._state.backtracking == 0:
                    stream_RPAREN.add(RPAREN228)

                # AST Rewrite
                # elements: annotationInitializers
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 582:9: -> ^( ANNOTATION_INIT_BLOCK[$LPAREN, \"ANNOTATION_INIT_BLOCK\"] annotationInitializers )
                    # Java.g:582:13: ^( ANNOTATION_INIT_BLOCK[$LPAREN, \"ANNOTATION_INIT_BLOCK\"] annotationInitializers )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ANNOTATION_INIT_BLOCK, LPAREN226, "ANNOTATION_INIT_BLOCK"), root_1)

                    self._adaptor.addChild(root_1, stream_annotationInitializers.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 62, annotationInit_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationInit"

    class annotationInitializers_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationInitializers_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationInitializers"
    # Java.g:585:1: annotationInitializers : ( annotationInitializer ( COMMA annotationInitializer )* -> ^( ANNOTATION_INIT_KEY_LIST ( annotationInitializer )+ ) | annotationElementValue -> ^( ANNOTATION_INIT_DEFAULT_KEY annotationElementValue ) );
    def annotationInitializers(self, ):

        retval = self.annotationInitializers_return()
        retval.start = self.input.LT(1)
        annotationInitializers_StartIndex = self.input.index()
        root_0 = None

        COMMA230 = None
        annotationInitializer229 = None

        annotationInitializer231 = None

        annotationElementValue232 = None


        COMMA230_tree = None
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_annotationElementValue = RewriteRuleSubtreeStream(self._adaptor, "rule annotationElementValue")
        stream_annotationInitializer = RewriteRuleSubtreeStream(self._adaptor, "rule annotationInitializer")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 63):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:586:5: ( annotationInitializer ( COMMA annotationInitializer )* -> ^( ANNOTATION_INIT_KEY_LIST ( annotationInitializer )+ ) | annotationElementValue -> ^( ANNOTATION_INIT_DEFAULT_KEY annotationElementValue ) )
                alt78 = 2
                LA78_0 = self.input.LA(1)

                if (LA78_0 == IDENT) :
                    LA78_1 = self.input.LA(2)

                    if (LA78_1 == ASSIGN) :
                        alt78 = 1
                    elif (LA78_1 == AND or LA78_1 == BIT_SHIFT_RIGHT or (DEC <= LA78_1 <= DIV) or LA78_1 == DOT or (EQUAL <= LA78_1 <= LBRACK) or (LESS_OR_EQUAL <= LA78_1 <= LOGICAL_AND) or (LOGICAL_OR <= LA78_1 <= MINUS) or LA78_1 == MOD or (NOT_EQUAL <= LA78_1 <= OR) or LA78_1 == PLUS or LA78_1 == QUESTION or LA78_1 == RPAREN or LA78_1 == SHIFT_LEFT or LA78_1 == SHIFT_RIGHT or LA78_1 == STAR or LA78_1 == XOR or LA78_1 == INSTANCEOF) :
                        alt78 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 78, 1, self.input)

                        raise nvae

                elif (LA78_0 == AT or LA78_0 == DEC or LA78_0 == INC or LA78_0 == LCURLY or LA78_0 == LESS_THAN or LA78_0 == LOGICAL_NOT or (LPAREN <= LA78_0 <= MINUS) or LA78_0 == NOT or LA78_0 == PLUS or LA78_0 == BOOLEAN or LA78_0 == BYTE or LA78_0 == CHAR or LA78_0 == DOUBLE or LA78_0 == FALSE or LA78_0 == FLOAT or (INT <= LA78_0 <= LONG) or (NEW <= LA78_0 <= NULL) or LA78_0 == SHORT or LA78_0 == SUPER or LA78_0 == THIS or LA78_0 == TRUE or LA78_0 == VOID or (HEX_LITERAL <= LA78_0 <= STRING_LITERAL)) :
                    alt78 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 78, 0, self.input)

                    raise nvae

                if alt78 == 1:
                    # Java.g:586:9: annotationInitializer ( COMMA annotationInitializer )*
                    pass 
                    self._state.following.append(self.FOLLOW_annotationInitializer_in_annotationInitializers7922)
                    annotationInitializer229 = self.annotationInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_annotationInitializer.add(annotationInitializer229.tree)
                    # Java.g:586:31: ( COMMA annotationInitializer )*
                    while True: #loop77
                        alt77 = 2
                        LA77_0 = self.input.LA(1)

                        if (LA77_0 == COMMA) :
                            alt77 = 1


                        if alt77 == 1:
                            # Java.g:586:32: COMMA annotationInitializer
                            pass 
                            COMMA230=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_annotationInitializers7925) 
                            if self._state.backtracking == 0:
                                stream_COMMA.add(COMMA230)
                            self._state.following.append(self.FOLLOW_annotationInitializer_in_annotationInitializers7927)
                            annotationInitializer231 = self.annotationInitializer()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_annotationInitializer.add(annotationInitializer231.tree)


                        else:
                            break #loop77

                    # AST Rewrite
                    # elements: annotationInitializer
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 587:9: -> ^( ANNOTATION_INIT_KEY_LIST ( annotationInitializer )+ )
                        # Java.g:587:13: ^( ANNOTATION_INIT_KEY_LIST ( annotationInitializer )+ )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ANNOTATION_INIT_KEY_LIST, "ANNOTATION_INIT_KEY_LIST"), root_1)

                        # Java.g:587:40: ( annotationInitializer )+
                        if not (stream_annotationInitializer.hasNext()):
                            raise RewriteEarlyExitException()

                        while stream_annotationInitializer.hasNext():
                            self._adaptor.addChild(root_1, stream_annotationInitializer.nextTree())


                        stream_annotationInitializer.reset()

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt78 == 2:
                    # Java.g:588:9: annotationElementValue
                    pass 
                    self._state.following.append(self.FOLLOW_annotationElementValue_in_annotationInitializers7957)
                    annotationElementValue232 = self.annotationElementValue()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_annotationElementValue.add(annotationElementValue232.tree)

                    # AST Rewrite
                    # elements: annotationElementValue
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 589:9: -> ^( ANNOTATION_INIT_DEFAULT_KEY annotationElementValue )
                        # Java.g:589:13: ^( ANNOTATION_INIT_DEFAULT_KEY annotationElementValue )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ANNOTATION_INIT_DEFAULT_KEY, "ANNOTATION_INIT_DEFAULT_KEY"), root_1)

                        self._adaptor.addChild(root_1, stream_annotationElementValue.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 63, annotationInitializers_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationInitializers"

    class annotationInitializer_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationInitializer_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationInitializer"
    # Java.g:592:1: annotationInitializer : IDENT ASSIGN annotationElementValue ;
    def annotationInitializer(self, ):

        retval = self.annotationInitializer_return()
        retval.start = self.input.LT(1)
        annotationInitializer_StartIndex = self.input.index()
        root_0 = None

        IDENT233 = None
        ASSIGN234 = None
        annotationElementValue235 = None


        IDENT233_tree = None
        ASSIGN234_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 64):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:593:5: ( IDENT ASSIGN annotationElementValue )
                # Java.g:593:9: IDENT ASSIGN annotationElementValue
                pass 
                root_0 = self._adaptor.nil()

                IDENT233=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_annotationInitializer7994)
                if self._state.backtracking == 0:

                    IDENT233_tree = self._adaptor.createWithPayload(IDENT233)
                    root_0 = self._adaptor.becomeRoot(IDENT233_tree, root_0)

                ASSIGN234=self.match(self.input, ASSIGN, self.FOLLOW_ASSIGN_in_annotationInitializer7997)
                self._state.following.append(self.FOLLOW_annotationElementValue_in_annotationInitializer8000)
                annotationElementValue235 = self.annotationElementValue()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, annotationElementValue235.tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 64, annotationInitializer_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationInitializer"

    class annotationElementValue_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationElementValue_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationElementValue"
    # Java.g:596:1: annotationElementValue : ( annotationElementValueExpression | annotation | annotationElementValueArrayInitializer );
    def annotationElementValue(self, ):

        retval = self.annotationElementValue_return()
        retval.start = self.input.LT(1)
        annotationElementValue_StartIndex = self.input.index()
        root_0 = None

        annotationElementValueExpression236 = None

        annotation237 = None

        annotationElementValueArrayInitializer238 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 65):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:597:5: ( annotationElementValueExpression | annotation | annotationElementValueArrayInitializer )
                alt79 = 3
                LA79 = self.input.LA(1)
                if LA79 == DEC or LA79 == INC or LA79 == LESS_THAN or LA79 == LOGICAL_NOT or LA79 == LPAREN or LA79 == MINUS or LA79 == NOT or LA79 == PLUS or LA79 == BOOLEAN or LA79 == BYTE or LA79 == CHAR or LA79 == DOUBLE or LA79 == FALSE or LA79 == FLOAT or LA79 == INT or LA79 == LONG or LA79 == NEW or LA79 == NULL or LA79 == SHORT or LA79 == SUPER or LA79 == THIS or LA79 == TRUE or LA79 == VOID or LA79 == IDENT or LA79 == HEX_LITERAL or LA79 == OCTAL_LITERAL or LA79 == DECIMAL_LITERAL or LA79 == FLOATING_POINT_LITERAL or LA79 == CHARACTER_LITERAL or LA79 == STRING_LITERAL:
                    alt79 = 1
                elif LA79 == AT:
                    alt79 = 2
                elif LA79 == LCURLY:
                    alt79 = 3
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 79, 0, self.input)

                    raise nvae

                if alt79 == 1:
                    # Java.g:597:9: annotationElementValueExpression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_annotationElementValueExpression_in_annotationElementValue8019)
                    annotationElementValueExpression236 = self.annotationElementValueExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotationElementValueExpression236.tree)


                elif alt79 == 2:
                    # Java.g:598:9: annotation
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_annotation_in_annotationElementValue8029)
                    annotation237 = self.annotation()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotation237.tree)


                elif alt79 == 3:
                    # Java.g:599:9: annotationElementValueArrayInitializer
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_annotationElementValueArrayInitializer_in_annotationElementValue8039)
                    annotationElementValueArrayInitializer238 = self.annotationElementValueArrayInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, annotationElementValueArrayInitializer238.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 65, annotationElementValue_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationElementValue"

    class annotationElementValueExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationElementValueExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationElementValueExpression"
    # Java.g:602:1: annotationElementValueExpression : conditionalExpression -> ^( EXPR conditionalExpression ) ;
    def annotationElementValueExpression(self, ):

        retval = self.annotationElementValueExpression_return()
        retval.start = self.input.LT(1)
        annotationElementValueExpression_StartIndex = self.input.index()
        root_0 = None

        conditionalExpression239 = None


        stream_conditionalExpression = RewriteRuleSubtreeStream(self._adaptor, "rule conditionalExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 66):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:603:5: ( conditionalExpression -> ^( EXPR conditionalExpression ) )
                # Java.g:603:9: conditionalExpression
                pass 
                self._state.following.append(self.FOLLOW_conditionalExpression_in_annotationElementValueExpression8058)
                conditionalExpression239 = self.conditionalExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_conditionalExpression.add(conditionalExpression239.tree)

                # AST Rewrite
                # elements: conditionalExpression
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 604:9: -> ^( EXPR conditionalExpression )
                    # Java.g:604:13: ^( EXPR conditionalExpression )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(EXPR, "EXPR"), root_1)

                    self._adaptor.addChild(root_1, stream_conditionalExpression.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 66, annotationElementValueExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationElementValueExpression"

    class annotationElementValueArrayInitializer_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationElementValueArrayInitializer_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationElementValueArrayInitializer"
    # Java.g:607:1: annotationElementValueArrayInitializer : LCURLY ( annotationElementValue ( COMMA annotationElementValue )* )? ( COMMA )? RCURLY -> ^( ANNOTATION_INIT_ARRAY_ELEMENT[$LCURLY, \"ANNOTATION_ELEM_VALUE_ARRAY_INIT\"] ( annotationElementValue )* ) ;
    def annotationElementValueArrayInitializer(self, ):

        retval = self.annotationElementValueArrayInitializer_return()
        retval.start = self.input.LT(1)
        annotationElementValueArrayInitializer_StartIndex = self.input.index()
        root_0 = None

        LCURLY240 = None
        COMMA242 = None
        COMMA244 = None
        RCURLY245 = None
        annotationElementValue241 = None

        annotationElementValue243 = None


        LCURLY240_tree = None
        COMMA242_tree = None
        COMMA244_tree = None
        RCURLY245_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_COMMA = RewriteRuleTokenStream(self._adaptor, "token COMMA")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_annotationElementValue = RewriteRuleSubtreeStream(self._adaptor, "rule annotationElementValue")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 67):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:608:5: ( LCURLY ( annotationElementValue ( COMMA annotationElementValue )* )? ( COMMA )? RCURLY -> ^( ANNOTATION_INIT_ARRAY_ELEMENT[$LCURLY, \"ANNOTATION_ELEM_VALUE_ARRAY_INIT\"] ( annotationElementValue )* ) )
                # Java.g:608:9: LCURLY ( annotationElementValue ( COMMA annotationElementValue )* )? ( COMMA )? RCURLY
                pass 
                LCURLY240=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_annotationElementValueArrayInitializer8094) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY240)
                # Java.g:608:16: ( annotationElementValue ( COMMA annotationElementValue )* )?
                alt81 = 2
                LA81_0 = self.input.LA(1)

                if (LA81_0 == AT or LA81_0 == DEC or LA81_0 == INC or LA81_0 == LCURLY or LA81_0 == LESS_THAN or LA81_0 == LOGICAL_NOT or (LPAREN <= LA81_0 <= MINUS) or LA81_0 == NOT or LA81_0 == PLUS or LA81_0 == BOOLEAN or LA81_0 == BYTE or LA81_0 == CHAR or LA81_0 == DOUBLE or LA81_0 == FALSE or LA81_0 == FLOAT or (INT <= LA81_0 <= LONG) or (NEW <= LA81_0 <= NULL) or LA81_0 == SHORT or LA81_0 == SUPER or LA81_0 == THIS or LA81_0 == TRUE or LA81_0 == VOID or (IDENT <= LA81_0 <= STRING_LITERAL)) :
                    alt81 = 1
                if alt81 == 1:
                    # Java.g:608:17: annotationElementValue ( COMMA annotationElementValue )*
                    pass 
                    self._state.following.append(self.FOLLOW_annotationElementValue_in_annotationElementValueArrayInitializer8097)
                    annotationElementValue241 = self.annotationElementValue()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_annotationElementValue.add(annotationElementValue241.tree)
                    # Java.g:608:40: ( COMMA annotationElementValue )*
                    while True: #loop80
                        alt80 = 2
                        LA80_0 = self.input.LA(1)

                        if (LA80_0 == COMMA) :
                            LA80_1 = self.input.LA(2)

                            if (LA80_1 == AT or LA80_1 == DEC or LA80_1 == INC or LA80_1 == LCURLY or LA80_1 == LESS_THAN or LA80_1 == LOGICAL_NOT or (LPAREN <= LA80_1 <= MINUS) or LA80_1 == NOT or LA80_1 == PLUS or LA80_1 == BOOLEAN or LA80_1 == BYTE or LA80_1 == CHAR or LA80_1 == DOUBLE or LA80_1 == FALSE or LA80_1 == FLOAT or (INT <= LA80_1 <= LONG) or (NEW <= LA80_1 <= NULL) or LA80_1 == SHORT or LA80_1 == SUPER or LA80_1 == THIS or LA80_1 == TRUE or LA80_1 == VOID or (IDENT <= LA80_1 <= STRING_LITERAL)) :
                                alt80 = 1




                        if alt80 == 1:
                            # Java.g:608:41: COMMA annotationElementValue
                            pass 
                            COMMA242=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_annotationElementValueArrayInitializer8100) 
                            if self._state.backtracking == 0:
                                stream_COMMA.add(COMMA242)
                            self._state.following.append(self.FOLLOW_annotationElementValue_in_annotationElementValueArrayInitializer8102)
                            annotationElementValue243 = self.annotationElementValue()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_annotationElementValue.add(annotationElementValue243.tree)


                        else:
                            break #loop80



                # Java.g:608:74: ( COMMA )?
                alt82 = 2
                LA82_0 = self.input.LA(1)

                if (LA82_0 == COMMA) :
                    alt82 = 1
                if alt82 == 1:
                    # Java.g:608:75: COMMA
                    pass 
                    COMMA244=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_annotationElementValueArrayInitializer8109) 
                    if self._state.backtracking == 0:
                        stream_COMMA.add(COMMA244)



                RCURLY245=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_annotationElementValueArrayInitializer8113) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY245)

                # AST Rewrite
                # elements: annotationElementValue
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 609:9: -> ^( ANNOTATION_INIT_ARRAY_ELEMENT[$LCURLY, \"ANNOTATION_ELEM_VALUE_ARRAY_INIT\"] ( annotationElementValue )* )
                    # Java.g:609:13: ^( ANNOTATION_INIT_ARRAY_ELEMENT[$LCURLY, \"ANNOTATION_ELEM_VALUE_ARRAY_INIT\"] ( annotationElementValue )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ANNOTATION_INIT_ARRAY_ELEMENT, LCURLY240, "ANNOTATION_ELEM_VALUE_ARRAY_INIT"), root_1)

                    # Java.g:609:90: ( annotationElementValue )*
                    while stream_annotationElementValue.hasNext():
                        self._adaptor.addChild(root_1, stream_annotationElementValue.nextTree())


                    stream_annotationElementValue.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 67, annotationElementValueArrayInitializer_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationElementValueArrayInitializer"

    class annotationTypeDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationTypeDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationTypeDeclaration"
    # Java.g:612:1: annotationTypeDeclaration[modifiers] : AT INTERFACE IDENT annotationBody -> ^( AT IDENT annotationBody ) ;
    def annotationTypeDeclaration(self, modifiers):

        retval = self.annotationTypeDeclaration_return()
        retval.start = self.input.LT(1)
        annotationTypeDeclaration_StartIndex = self.input.index()
        root_0 = None

        AT246 = None
        INTERFACE247 = None
        IDENT248 = None
        annotationBody249 = None


        AT246_tree = None
        INTERFACE247_tree = None
        IDENT248_tree = None
        stream_AT = RewriteRuleTokenStream(self._adaptor, "token AT")
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_INTERFACE = RewriteRuleTokenStream(self._adaptor, "token INTERFACE")
        stream_annotationBody = RewriteRuleSubtreeStream(self._adaptor, "rule annotationBody")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 68):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:613:5: ( AT INTERFACE IDENT annotationBody -> ^( AT IDENT annotationBody ) )
                # Java.g:613:9: AT INTERFACE IDENT annotationBody
                pass 
                AT246=self.match(self.input, AT, self.FOLLOW_AT_in_annotationTypeDeclaration8152) 
                if self._state.backtracking == 0:
                    stream_AT.add(AT246)
                INTERFACE247=self.match(self.input, INTERFACE, self.FOLLOW_INTERFACE_in_annotationTypeDeclaration8154) 
                if self._state.backtracking == 0:
                    stream_INTERFACE.add(INTERFACE247)
                IDENT248=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_annotationTypeDeclaration8156) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT248)
                self._state.following.append(self.FOLLOW_annotationBody_in_annotationTypeDeclaration8158)
                annotationBody249 = self.annotationBody()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_annotationBody.add(annotationBody249.tree)

                # AST Rewrite
                # elements: AT, IDENT, annotationBody
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 614:9: -> ^( AT IDENT annotationBody )
                    # Java.g:614:12: ^( AT IDENT annotationBody )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(stream_AT.nextNode(), root_1)

                    self._adaptor.addChild(root_1, modifiers)
                    self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                    self._adaptor.addChild(root_1, stream_annotationBody.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 68, annotationTypeDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationTypeDeclaration"

    class annotationBody_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationBody_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationBody"
    # Java.g:617:1: annotationBody : LCURLY ( annotationScopeDeclarations )* RCURLY -> ^( ANNOTATION_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( annotationScopeDeclarations )* ) ;
    def annotationBody(self, ):

        retval = self.annotationBody_return()
        retval.start = self.input.LT(1)
        annotationBody_StartIndex = self.input.index()
        root_0 = None

        LCURLY250 = None
        RCURLY252 = None
        annotationScopeDeclarations251 = None


        LCURLY250_tree = None
        RCURLY252_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_annotationScopeDeclarations = RewriteRuleSubtreeStream(self._adaptor, "rule annotationScopeDeclarations")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 69):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:618:5: ( LCURLY ( annotationScopeDeclarations )* RCURLY -> ^( ANNOTATION_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( annotationScopeDeclarations )* ) )
                # Java.g:618:9: LCURLY ( annotationScopeDeclarations )* RCURLY
                pass 
                LCURLY250=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_annotationBody8197) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY250)
                # Java.g:618:16: ( annotationScopeDeclarations )*
                while True: #loop83
                    alt83 = 2
                    LA83_0 = self.input.LA(1)

                    if (LA83_0 == AT or LA83_0 == LESS_THAN or LA83_0 == ABSTRACT or LA83_0 == BOOLEAN or LA83_0 == BYTE or (CHAR <= LA83_0 <= CLASS) or LA83_0 == DOUBLE or LA83_0 == ENUM or LA83_0 == FINAL or LA83_0 == FLOAT or LA83_0 == INTERFACE or (INT <= LA83_0 <= NATIVE) or (PRIVATE <= LA83_0 <= PUBLIC) or (SHORT <= LA83_0 <= STRICTFP) or LA83_0 == SYNCHRONIZED or LA83_0 == TRANSIENT or (VOID <= LA83_0 <= VOLATILE) or LA83_0 == IDENT) :
                        alt83 = 1


                    if alt83 == 1:
                        # Java.g:0:0: annotationScopeDeclarations
                        pass 
                        self._state.following.append(self.FOLLOW_annotationScopeDeclarations_in_annotationBody8199)
                        annotationScopeDeclarations251 = self.annotationScopeDeclarations()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_annotationScopeDeclarations.add(annotationScopeDeclarations251.tree)


                    else:
                        break #loop83
                RCURLY252=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_annotationBody8202) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY252)

                # AST Rewrite
                # elements: annotationScopeDeclarations
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 619:9: -> ^( ANNOTATION_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( annotationScopeDeclarations )* )
                    # Java.g:619:13: ^( ANNOTATION_TOP_LEVEL_SCOPE[$LCURLY, \"CLASS_TOP_LEVEL_SCOPE\"] ( annotationScopeDeclarations )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ANNOTATION_TOP_LEVEL_SCOPE, LCURLY250, "CLASS_TOP_LEVEL_SCOPE"), root_1)

                    # Java.g:619:76: ( annotationScopeDeclarations )*
                    while stream_annotationScopeDeclarations.hasNext():
                        self._adaptor.addChild(root_1, stream_annotationScopeDeclarations.nextTree())


                    stream_annotationScopeDeclarations.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 69, annotationBody_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationBody"

    class annotationScopeDeclarations_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationScopeDeclarations_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationScopeDeclarations"
    # Java.g:622:1: annotationScopeDeclarations : ( modifierList type ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI -> ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? ) | classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) ) | typeDeclaration );
    def annotationScopeDeclarations(self, ):

        retval = self.annotationScopeDeclarations_return()
        retval.start = self.input.LT(1)
        annotationScopeDeclarations_StartIndex = self.input.index()
        root_0 = None

        IDENT255 = None
        LPAREN256 = None
        RPAREN257 = None
        SEMI259 = None
        SEMI261 = None
        modifierList253 = None

        type254 = None

        annotationDefaultValue258 = None

        classFieldDeclaratorList260 = None

        typeDeclaration262 = None


        IDENT255_tree = None
        LPAREN256_tree = None
        RPAREN257_tree = None
        SEMI259_tree = None
        SEMI261_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_SEMI = RewriteRuleTokenStream(self._adaptor, "token SEMI")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_modifierList = RewriteRuleSubtreeStream(self._adaptor, "rule modifierList")
        stream_annotationDefaultValue = RewriteRuleSubtreeStream(self._adaptor, "rule annotationDefaultValue")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        stream_classFieldDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule classFieldDeclaratorList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 70):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:623:5: ( modifierList type ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI -> ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? ) | classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) ) | typeDeclaration )
                alt86 = 2
                alt86 = self.dfa86.predict(self.input)
                if alt86 == 1:
                    # Java.g:623:9: modifierList type ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI -> ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? ) | classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) )
                    pass 
                    self._state.following.append(self.FOLLOW_modifierList_in_annotationScopeDeclarations8240)
                    modifierList253 = self.modifierList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_modifierList.add(modifierList253.tree)
                    self._state.following.append(self.FOLLOW_type_in_annotationScopeDeclarations8242)
                    type254 = self.type()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_type.add(type254.tree)
                    # Java.g:624:9: ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI -> ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? ) | classFieldDeclaratorList SEMI -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList ) )
                    alt85 = 2
                    LA85_0 = self.input.LA(1)

                    if (LA85_0 == IDENT) :
                        LA85_1 = self.input.LA(2)

                        if (LA85_1 == LPAREN) :
                            alt85 = 1
                        elif (LA85_1 == ASSIGN or LA85_1 == COMMA or LA85_1 == LBRACK or LA85_1 == SEMI) :
                            alt85 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 85, 1, self.input)

                            raise nvae

                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 85, 0, self.input)

                        raise nvae

                    if alt85 == 1:
                        # Java.g:624:13: IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI
                        pass 
                        IDENT255=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_annotationScopeDeclarations8256) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(IDENT255)
                        LPAREN256=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_annotationScopeDeclarations8258) 
                        if self._state.backtracking == 0:
                            stream_LPAREN.add(LPAREN256)
                        RPAREN257=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_annotationScopeDeclarations8260) 
                        if self._state.backtracking == 0:
                            stream_RPAREN.add(RPAREN257)
                        # Java.g:624:33: ( annotationDefaultValue )?
                        alt84 = 2
                        LA84_0 = self.input.LA(1)

                        if (LA84_0 == DEFAULT) :
                            alt84 = 1
                        if alt84 == 1:
                            # Java.g:0:0: annotationDefaultValue
                            pass 
                            self._state.following.append(self.FOLLOW_annotationDefaultValue_in_annotationScopeDeclarations8262)
                            annotationDefaultValue258 = self.annotationDefaultValue()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_annotationDefaultValue.add(annotationDefaultValue258.tree)



                        SEMI259=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_annotationScopeDeclarations8265) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI259)

                        # AST Rewrite
                        # elements: annotationDefaultValue, IDENT, modifierList, type
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 625:13: -> ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? )
                            # Java.g:625:17: ^( ANNOTATION_METHOD_DECL modifierList type IDENT ( annotationDefaultValue )? )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ANNOTATION_METHOD_DECL, "ANNOTATION_METHOD_DECL"), root_1)

                            self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                            self._adaptor.addChild(root_1, stream_type.nextTree())
                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                            # Java.g:625:66: ( annotationDefaultValue )?
                            if stream_annotationDefaultValue.hasNext():
                                self._adaptor.addChild(root_1, stream_annotationDefaultValue.nextTree())


                            stream_annotationDefaultValue.reset();

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt85 == 2:
                        # Java.g:626:13: classFieldDeclaratorList SEMI
                        pass 
                        self._state.following.append(self.FOLLOW_classFieldDeclaratorList_in_annotationScopeDeclarations8307)
                        classFieldDeclaratorList260 = self.classFieldDeclaratorList()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_classFieldDeclaratorList.add(classFieldDeclaratorList260.tree)
                        SEMI261=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_annotationScopeDeclarations8309) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI261)

                        # AST Rewrite
                        # elements: classFieldDeclaratorList, type, modifierList
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 627:13: -> ^( VAR_DECLARATION modifierList type classFieldDeclaratorList )
                            # Java.g:627:17: ^( VAR_DECLARATION modifierList type classFieldDeclaratorList )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATION, "VAR_DECLARATION"), root_1)

                            self._adaptor.addChild(root_1, stream_modifierList.nextTree())
                            self._adaptor.addChild(root_1, stream_type.nextTree())
                            self._adaptor.addChild(root_1, stream_classFieldDeclaratorList.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt86 == 2:
                    # Java.g:629:9: typeDeclaration
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_typeDeclaration_in_annotationScopeDeclarations8354)
                    typeDeclaration262 = self.typeDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, typeDeclaration262.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 70, annotationScopeDeclarations_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationScopeDeclarations"

    class annotationDefaultValue_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.annotationDefaultValue_return, self).__init__()

            self.tree = None




    # $ANTLR start "annotationDefaultValue"
    # Java.g:632:1: annotationDefaultValue : DEFAULT annotationElementValue ;
    def annotationDefaultValue(self, ):

        retval = self.annotationDefaultValue_return()
        retval.start = self.input.LT(1)
        annotationDefaultValue_StartIndex = self.input.index()
        root_0 = None

        DEFAULT263 = None
        annotationElementValue264 = None


        DEFAULT263_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 71):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:633:5: ( DEFAULT annotationElementValue )
                # Java.g:633:9: DEFAULT annotationElementValue
                pass 
                root_0 = self._adaptor.nil()

                DEFAULT263=self.match(self.input, DEFAULT, self.FOLLOW_DEFAULT_in_annotationDefaultValue8373)
                if self._state.backtracking == 0:

                    DEFAULT263_tree = self._adaptor.createWithPayload(DEFAULT263)
                    root_0 = self._adaptor.becomeRoot(DEFAULT263_tree, root_0)

                self._state.following.append(self.FOLLOW_annotationElementValue_in_annotationDefaultValue8376)
                annotationElementValue264 = self.annotationElementValue()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, annotationElementValue264.tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 71, annotationDefaultValue_StartIndex, success)

            pass
        return retval

    # $ANTLR end "annotationDefaultValue"

    class block_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.block_return, self).__init__()

            self.tree = None




    # $ANTLR start "block"
    # Java.g:638:1: block : LCURLY ( blockStatement )* RCURLY -> ^( BLOCK_SCOPE[$LCURLY, \"BLOCK_SCOPE\"] ( blockStatement )* ) ;
    def block(self, ):

        retval = self.block_return()
        retval.start = self.input.LT(1)
        block_StartIndex = self.input.index()
        root_0 = None

        LCURLY265 = None
        RCURLY267 = None
        blockStatement266 = None


        LCURLY265_tree = None
        RCURLY267_tree = None
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_blockStatement = RewriteRuleSubtreeStream(self._adaptor, "rule blockStatement")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 72):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:639:5: ( LCURLY ( blockStatement )* RCURLY -> ^( BLOCK_SCOPE[$LCURLY, \"BLOCK_SCOPE\"] ( blockStatement )* ) )
                # Java.g:639:9: LCURLY ( blockStatement )* RCURLY
                pass 
                LCURLY265=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_block8397) 
                if self._state.backtracking == 0:
                    stream_LCURLY.add(LCURLY265)
                # Java.g:639:16: ( blockStatement )*
                while True: #loop87
                    alt87 = 2
                    LA87_0 = self.input.LA(1)

                    if (LA87_0 == AT or LA87_0 == DEC or LA87_0 == INC or LA87_0 == LCURLY or LA87_0 == LESS_THAN or LA87_0 == LOGICAL_NOT or (LPAREN <= LA87_0 <= MINUS) or LA87_0 == NOT or LA87_0 == PLUS or LA87_0 == SEMI or (ABSTRACT <= LA87_0 <= BYTE) or (CHAR <= LA87_0 <= CONTINUE) or (DO <= LA87_0 <= DOUBLE) or LA87_0 == ENUM or (FALSE <= LA87_0 <= FINAL) or (FLOAT <= LA87_0 <= IF) or LA87_0 == INTERFACE or (INT <= LA87_0 <= NULL) or (PRIVATE <= LA87_0 <= THROW) or (TRANSIENT <= LA87_0 <= WHILE) or (IDENT <= LA87_0 <= STRING_LITERAL)) :
                        alt87 = 1


                    if alt87 == 1:
                        # Java.g:0:0: blockStatement
                        pass 
                        self._state.following.append(self.FOLLOW_blockStatement_in_block8399)
                        blockStatement266 = self.blockStatement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_blockStatement.add(blockStatement266.tree)


                    else:
                        break #loop87
                RCURLY267=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_block8402) 
                if self._state.backtracking == 0:
                    stream_RCURLY.add(RCURLY267)

                # AST Rewrite
                # elements: blockStatement
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 640:9: -> ^( BLOCK_SCOPE[$LCURLY, \"BLOCK_SCOPE\"] ( blockStatement )* )
                    # Java.g:640:13: ^( BLOCK_SCOPE[$LCURLY, \"BLOCK_SCOPE\"] ( blockStatement )* )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(BLOCK_SCOPE, LCURLY265, "BLOCK_SCOPE"), root_1)

                    # Java.g:640:51: ( blockStatement )*
                    while stream_blockStatement.hasNext():
                        self._adaptor.addChild(root_1, stream_blockStatement.nextTree())


                    stream_blockStatement.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 72, block_StartIndex, success)

            pass
        return retval

    # $ANTLR end "block"

    class blockStatement_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.blockStatement_return, self).__init__()

            self.tree = None




    # $ANTLR start "blockStatement"
    # Java.g:643:1: blockStatement : ( localVariableDeclaration SEMI | typeDeclaration | statement );
    def blockStatement(self, ):

        retval = self.blockStatement_return()
        retval.start = self.input.LT(1)
        blockStatement_StartIndex = self.input.index()
        root_0 = None

        SEMI269 = None
        localVariableDeclaration268 = None

        typeDeclaration270 = None

        statement271 = None


        SEMI269_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 73):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:644:5: ( localVariableDeclaration SEMI | typeDeclaration | statement )
                alt88 = 3
                alt88 = self.dfa88.predict(self.input)
                if alt88 == 1:
                    # Java.g:644:9: localVariableDeclaration SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_localVariableDeclaration_in_blockStatement8440)
                    localVariableDeclaration268 = self.localVariableDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, localVariableDeclaration268.tree)
                    SEMI269=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_blockStatement8442)


                elif alt88 == 2:
                    # Java.g:645:9: typeDeclaration
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_typeDeclaration_in_blockStatement8453)
                    typeDeclaration270 = self.typeDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, typeDeclaration270.tree)


                elif alt88 == 3:
                    # Java.g:646:9: statement
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_statement_in_blockStatement8463)
                    statement271 = self.statement()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, statement271.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 73, blockStatement_StartIndex, success)

            pass
        return retval

    # $ANTLR end "blockStatement"

    class localVariableDeclaration_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.localVariableDeclaration_return, self).__init__()

            self.tree = None




    # $ANTLR start "localVariableDeclaration"
    # Java.g:649:1: localVariableDeclaration : localModifierList type classFieldDeclaratorList -> ^( VAR_DECLARATION localModifierList type classFieldDeclaratorList ) ;
    def localVariableDeclaration(self, ):

        retval = self.localVariableDeclaration_return()
        retval.start = self.input.LT(1)
        localVariableDeclaration_StartIndex = self.input.index()
        root_0 = None

        localModifierList272 = None

        type273 = None

        classFieldDeclaratorList274 = None


        stream_localModifierList = RewriteRuleSubtreeStream(self._adaptor, "rule localModifierList")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        stream_classFieldDeclaratorList = RewriteRuleSubtreeStream(self._adaptor, "rule classFieldDeclaratorList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 74):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:650:5: ( localModifierList type classFieldDeclaratorList -> ^( VAR_DECLARATION localModifierList type classFieldDeclaratorList ) )
                # Java.g:650:9: localModifierList type classFieldDeclaratorList
                pass 
                self._state.following.append(self.FOLLOW_localModifierList_in_localVariableDeclaration8482)
                localModifierList272 = self.localModifierList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_localModifierList.add(localModifierList272.tree)
                self._state.following.append(self.FOLLOW_type_in_localVariableDeclaration8484)
                type273 = self.type()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_type.add(type273.tree)
                self._state.following.append(self.FOLLOW_classFieldDeclaratorList_in_localVariableDeclaration8486)
                classFieldDeclaratorList274 = self.classFieldDeclaratorList()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_classFieldDeclaratorList.add(classFieldDeclaratorList274.tree)

                # AST Rewrite
                # elements: type, classFieldDeclaratorList, localModifierList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 651:9: -> ^( VAR_DECLARATION localModifierList type classFieldDeclaratorList )
                    # Java.g:651:13: ^( VAR_DECLARATION localModifierList type classFieldDeclaratorList )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(VAR_DECLARATION, "VAR_DECLARATION"), root_1)

                    self._adaptor.addChild(root_1, stream_localModifierList.nextTree())
                    self._adaptor.addChild(root_1, stream_type.nextTree())
                    self._adaptor.addChild(root_1, stream_classFieldDeclaratorList.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 74, localVariableDeclaration_StartIndex, success)

            pass
        return retval

    # $ANTLR end "localVariableDeclaration"

    class statement_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.statement_return, self).__init__()

            self.tree = None




    # $ANTLR start "statement"
    # Java.g:655:1: statement : ( block | ASSERT expr1= expression ( COLON expr2= expression SEMI -> ^( ASSERT $expr1 $expr2) | SEMI -> ^( ASSERT $expr1) ) | IF parenthesizedExpression ifStat= statement ( ELSE elseStat= statement -> ^( IF parenthesizedExpression $ifStat $elseStat) | -> ^( IF parenthesizedExpression $ifStat) ) | FOR LPAREN ( forInit SEMI forCondition SEMI forUpdater RPAREN statement -> ^( FOR forInit forCondition forUpdater statement ) | localModifierList type IDENT COLON expression RPAREN statement -> ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement ) ) | WHILE parenthesizedExpression statement -> ^( WHILE parenthesizedExpression statement ) | DO statement WHILE parenthesizedExpression SEMI -> ^( DO statement parenthesizedExpression ) | TRY block ( catches ( finallyClause )? | finallyClause ) -> ^( TRY block ( catches )? ( finallyClause )? ) | SWITCH parenthesizedExpression LCURLY ( switchBlockLabels )? RCURLY -> ^( SWITCH parenthesizedExpression ( switchBlockLabels )? ) | SYNCHRONIZED parenthesizedExpression block -> ^( SYNCHRONIZED parenthesizedExpression block ) | RETURN ( expression )? SEMI -> ^( RETURN ( expression )? ) | THROW expression SEMI -> ^( THROW expression ) | BREAK ( IDENT )? SEMI -> ^( BREAK ( IDENT )? ) | CONTINUE ( IDENT )? SEMI -> ^( CONTINUE ( IDENT )? ) | IDENT COLON statement -> ^( LABELED_STATEMENT IDENT statement ) | expression SEMI | SEMI );
    def statement(self, ):

        retval = self.statement_return()
        retval.start = self.input.LT(1)
        statement_StartIndex = self.input.index()
        root_0 = None

        ASSERT276 = None
        COLON277 = None
        SEMI278 = None
        SEMI279 = None
        IF280 = None
        ELSE282 = None
        FOR283 = None
        LPAREN284 = None
        SEMI286 = None
        SEMI288 = None
        RPAREN290 = None
        IDENT294 = None
        COLON295 = None
        RPAREN297 = None
        WHILE299 = None
        DO302 = None
        WHILE304 = None
        SEMI306 = None
        TRY307 = None
        SWITCH312 = None
        LCURLY314 = None
        RCURLY316 = None
        SYNCHRONIZED317 = None
        RETURN320 = None
        SEMI322 = None
        THROW323 = None
        SEMI325 = None
        BREAK326 = None
        IDENT327 = None
        SEMI328 = None
        CONTINUE329 = None
        IDENT330 = None
        SEMI331 = None
        IDENT332 = None
        COLON333 = None
        SEMI336 = None
        SEMI337 = None
        expr1 = None

        expr2 = None

        ifStat = None

        elseStat = None

        block275 = None

        parenthesizedExpression281 = None

        forInit285 = None

        forCondition287 = None

        forUpdater289 = None

        statement291 = None

        localModifierList292 = None

        type293 = None

        expression296 = None

        statement298 = None

        parenthesizedExpression300 = None

        statement301 = None

        statement303 = None

        parenthesizedExpression305 = None

        block308 = None

        catches309 = None

        finallyClause310 = None

        finallyClause311 = None

        parenthesizedExpression313 = None

        switchBlockLabels315 = None

        parenthesizedExpression318 = None

        block319 = None

        expression321 = None

        expression324 = None

        statement334 = None

        expression335 = None


        ASSERT276_tree = None
        COLON277_tree = None
        SEMI278_tree = None
        SEMI279_tree = None
        IF280_tree = None
        ELSE282_tree = None
        FOR283_tree = None
        LPAREN284_tree = None
        SEMI286_tree = None
        SEMI288_tree = None
        RPAREN290_tree = None
        IDENT294_tree = None
        COLON295_tree = None
        RPAREN297_tree = None
        WHILE299_tree = None
        DO302_tree = None
        WHILE304_tree = None
        SEMI306_tree = None
        TRY307_tree = None
        SWITCH312_tree = None
        LCURLY314_tree = None
        RCURLY316_tree = None
        SYNCHRONIZED317_tree = None
        RETURN320_tree = None
        SEMI322_tree = None
        THROW323_tree = None
        SEMI325_tree = None
        BREAK326_tree = None
        IDENT327_tree = None
        SEMI328_tree = None
        CONTINUE329_tree = None
        IDENT330_tree = None
        SEMI331_tree = None
        IDENT332_tree = None
        COLON333_tree = None
        SEMI336_tree = None
        SEMI337_tree = None
        stream_COLON = RewriteRuleTokenStream(self._adaptor, "token COLON")
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_SYNCHRONIZED = RewriteRuleTokenStream(self._adaptor, "token SYNCHRONIZED")
        stream_WHILE = RewriteRuleTokenStream(self._adaptor, "token WHILE")
        stream_CONTINUE = RewriteRuleTokenStream(self._adaptor, "token CONTINUE")
        stream_SWITCH = RewriteRuleTokenStream(self._adaptor, "token SWITCH")
        stream_RCURLY = RewriteRuleTokenStream(self._adaptor, "token RCURLY")
        stream_ELSE = RewriteRuleTokenStream(self._adaptor, "token ELSE")
        stream_RETURN = RewriteRuleTokenStream(self._adaptor, "token RETURN")
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_FOR = RewriteRuleTokenStream(self._adaptor, "token FOR")
        stream_DO = RewriteRuleTokenStream(self._adaptor, "token DO")
        stream_LCURLY = RewriteRuleTokenStream(self._adaptor, "token LCURLY")
        stream_SEMI = RewriteRuleTokenStream(self._adaptor, "token SEMI")
        stream_ASSERT = RewriteRuleTokenStream(self._adaptor, "token ASSERT")
        stream_BREAK = RewriteRuleTokenStream(self._adaptor, "token BREAK")
        stream_THROW = RewriteRuleTokenStream(self._adaptor, "token THROW")
        stream_TRY = RewriteRuleTokenStream(self._adaptor, "token TRY")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_IF = RewriteRuleTokenStream(self._adaptor, "token IF")
        stream_statement = RewriteRuleSubtreeStream(self._adaptor, "rule statement")
        stream_expression = RewriteRuleSubtreeStream(self._adaptor, "rule expression")
        stream_finallyClause = RewriteRuleSubtreeStream(self._adaptor, "rule finallyClause")
        stream_catches = RewriteRuleSubtreeStream(self._adaptor, "rule catches")
        stream_forUpdater = RewriteRuleSubtreeStream(self._adaptor, "rule forUpdater")
        stream_block = RewriteRuleSubtreeStream(self._adaptor, "rule block")
        stream_forCondition = RewriteRuleSubtreeStream(self._adaptor, "rule forCondition")
        stream_localModifierList = RewriteRuleSubtreeStream(self._adaptor, "rule localModifierList")
        stream_forInit = RewriteRuleSubtreeStream(self._adaptor, "rule forInit")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        stream_switchBlockLabels = RewriteRuleSubtreeStream(self._adaptor, "rule switchBlockLabels")
        stream_parenthesizedExpression = RewriteRuleSubtreeStream(self._adaptor, "rule parenthesizedExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 75):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:656:5: ( block | ASSERT expr1= expression ( COLON expr2= expression SEMI -> ^( ASSERT $expr1 $expr2) | SEMI -> ^( ASSERT $expr1) ) | IF parenthesizedExpression ifStat= statement ( ELSE elseStat= statement -> ^( IF parenthesizedExpression $ifStat $elseStat) | -> ^( IF parenthesizedExpression $ifStat) ) | FOR LPAREN ( forInit SEMI forCondition SEMI forUpdater RPAREN statement -> ^( FOR forInit forCondition forUpdater statement ) | localModifierList type IDENT COLON expression RPAREN statement -> ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement ) ) | WHILE parenthesizedExpression statement -> ^( WHILE parenthesizedExpression statement ) | DO statement WHILE parenthesizedExpression SEMI -> ^( DO statement parenthesizedExpression ) | TRY block ( catches ( finallyClause )? | finallyClause ) -> ^( TRY block ( catches )? ( finallyClause )? ) | SWITCH parenthesizedExpression LCURLY ( switchBlockLabels )? RCURLY -> ^( SWITCH parenthesizedExpression ( switchBlockLabels )? ) | SYNCHRONIZED parenthesizedExpression block -> ^( SYNCHRONIZED parenthesizedExpression block ) | RETURN ( expression )? SEMI -> ^( RETURN ( expression )? ) | THROW expression SEMI -> ^( THROW expression ) | BREAK ( IDENT )? SEMI -> ^( BREAK ( IDENT )? ) | CONTINUE ( IDENT )? SEMI -> ^( CONTINUE ( IDENT )? ) | IDENT COLON statement -> ^( LABELED_STATEMENT IDENT statement ) | expression SEMI | SEMI )
                alt98 = 16
                alt98 = self.dfa98.predict(self.input)
                if alt98 == 1:
                    # Java.g:656:9: block
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_block_in_statement8527)
                    block275 = self.block()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, block275.tree)


                elif alt98 == 2:
                    # Java.g:657:9: ASSERT expr1= expression ( COLON expr2= expression SEMI -> ^( ASSERT $expr1 $expr2) | SEMI -> ^( ASSERT $expr1) )
                    pass 
                    ASSERT276=self.match(self.input, ASSERT, self.FOLLOW_ASSERT_in_statement8537) 
                    if self._state.backtracking == 0:
                        stream_ASSERT.add(ASSERT276)
                    self._state.following.append(self.FOLLOW_expression_in_statement8541)
                    expr1 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expression.add(expr1.tree)
                    # Java.g:658:9: ( COLON expr2= expression SEMI -> ^( ASSERT $expr1 $expr2) | SEMI -> ^( ASSERT $expr1) )
                    alt89 = 2
                    LA89_0 = self.input.LA(1)

                    if (LA89_0 == COLON) :
                        alt89 = 1
                    elif (LA89_0 == SEMI) :
                        alt89 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 89, 0, self.input)

                        raise nvae

                    if alt89 == 1:
                        # Java.g:658:13: COLON expr2= expression SEMI
                        pass 
                        COLON277=self.match(self.input, COLON, self.FOLLOW_COLON_in_statement8555) 
                        if self._state.backtracking == 0:
                            stream_COLON.add(COLON277)
                        self._state.following.append(self.FOLLOW_expression_in_statement8559)
                        expr2 = self.expression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_expression.add(expr2.tree)
                        SEMI278=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement8561) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI278)

                        # AST Rewrite
                        # elements: ASSERT, expr1, expr2
                        # token labels: 
                        # rule labels: retval, expr1, expr2
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            if expr1 is not None:
                                stream_expr1 = RewriteRuleSubtreeStream(self._adaptor, "rule expr1", expr1.tree)
                            else:
                                stream_expr1 = RewriteRuleSubtreeStream(self._adaptor, "token expr1", None)


                            if expr2 is not None:
                                stream_expr2 = RewriteRuleSubtreeStream(self._adaptor, "rule expr2", expr2.tree)
                            else:
                                stream_expr2 = RewriteRuleSubtreeStream(self._adaptor, "token expr2", None)


                            root_0 = self._adaptor.nil()
                            # 658:77: -> ^( ASSERT $expr1 $expr2)
                            # Java.g:658:81: ^( ASSERT $expr1 $expr2)
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_ASSERT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_expr1.nextTree())
                            self._adaptor.addChild(root_1, stream_expr2.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt89 == 2:
                        # Java.g:659:13: SEMI
                        pass 
                        SEMI279=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement8624) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI279)

                        # AST Rewrite
                        # elements: expr1, ASSERT
                        # token labels: 
                        # rule labels: retval, expr1
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            if expr1 is not None:
                                stream_expr1 = RewriteRuleSubtreeStream(self._adaptor, "rule expr1", expr1.tree)
                            else:
                                stream_expr1 = RewriteRuleSubtreeStream(self._adaptor, "token expr1", None)


                            root_0 = self._adaptor.nil()
                            # 659:77: -> ^( ASSERT $expr1)
                            # Java.g:659:81: ^( ASSERT $expr1)
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_ASSERT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_expr1.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt98 == 3:
                    # Java.g:661:9: IF parenthesizedExpression ifStat= statement ( ELSE elseStat= statement -> ^( IF parenthesizedExpression $ifStat $elseStat) | -> ^( IF parenthesizedExpression $ifStat) )
                    pass 
                    IF280=self.match(self.input, IF, self.FOLLOW_IF_in_statement8713) 
                    if self._state.backtracking == 0:
                        stream_IF.add(IF280)
                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_statement8715)
                    parenthesizedExpression281 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_parenthesizedExpression.add(parenthesizedExpression281.tree)
                    self._state.following.append(self.FOLLOW_statement_in_statement8719)
                    ifStat = self.statement()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_statement.add(ifStat.tree)
                    # Java.g:662:9: ( ELSE elseStat= statement -> ^( IF parenthesizedExpression $ifStat $elseStat) | -> ^( IF parenthesizedExpression $ifStat) )
                    alt90 = 2
                    LA90_0 = self.input.LA(1)

                    if (LA90_0 == ELSE) :
                        LA90_1 = self.input.LA(2)

                        if (self.synpred121_Java()) :
                            alt90 = 1
                        elif (True) :
                            alt90 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 90, 1, self.input)

                            raise nvae

                    elif (LA90_0 == EOF or LA90_0 == AT or LA90_0 == DEC or LA90_0 == INC or LA90_0 == LCURLY or LA90_0 == LESS_THAN or LA90_0 == LOGICAL_NOT or (LPAREN <= LA90_0 <= MINUS) or LA90_0 == NOT or LA90_0 == PLUS or LA90_0 == RCURLY or LA90_0 == SEMI or (ABSTRACT <= LA90_0 <= CASE) or (CHAR <= LA90_0 <= DOUBLE) or LA90_0 == ENUM or (FALSE <= LA90_0 <= FINAL) or (FLOAT <= LA90_0 <= IF) or LA90_0 == INTERFACE or (INT <= LA90_0 <= NULL) or (PRIVATE <= LA90_0 <= THROW) or (TRANSIENT <= LA90_0 <= WHILE) or (IDENT <= LA90_0 <= STRING_LITERAL)) :
                        alt90 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 90, 0, self.input)

                        raise nvae

                    if alt90 == 1:
                        # Java.g:662:13: ELSE elseStat= statement
                        pass 
                        ELSE282=self.match(self.input, ELSE, self.FOLLOW_ELSE_in_statement8733) 
                        if self._state.backtracking == 0:
                            stream_ELSE.add(ELSE282)
                        self._state.following.append(self.FOLLOW_statement_in_statement8737)
                        elseStat = self.statement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_statement.add(elseStat.tree)

                        # AST Rewrite
                        # elements: elseStat, ifStat, parenthesizedExpression, IF
                        # token labels: 
                        # rule labels: retval, ifStat, elseStat
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            if ifStat is not None:
                                stream_ifStat = RewriteRuleSubtreeStream(self._adaptor, "rule ifStat", ifStat.tree)
                            else:
                                stream_ifStat = RewriteRuleSubtreeStream(self._adaptor, "token ifStat", None)


                            if elseStat is not None:
                                stream_elseStat = RewriteRuleSubtreeStream(self._adaptor, "rule elseStat", elseStat.tree)
                            else:
                                stream_elseStat = RewriteRuleSubtreeStream(self._adaptor, "token elseStat", None)


                            root_0 = self._adaptor.nil()
                            # 662:77: -> ^( IF parenthesizedExpression $ifStat $elseStat)
                            # Java.g:662:81: ^( IF parenthesizedExpression $ifStat $elseStat)
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_IF.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())
                            self._adaptor.addChild(root_1, stream_ifStat.nextTree())
                            self._adaptor.addChild(root_1, stream_elseStat.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt90 == 2:
                        # Java.g:663:77: 
                        pass 
                        # AST Rewrite
                        # elements: parenthesizedExpression, ifStat, IF
                        # token labels: 
                        # rule labels: retval, ifStat
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            if ifStat is not None:
                                stream_ifStat = RewriteRuleSubtreeStream(self._adaptor, "rule ifStat", ifStat.tree)
                            else:
                                stream_ifStat = RewriteRuleSubtreeStream(self._adaptor, "token ifStat", None)


                            root_0 = self._adaptor.nil()
                            # 663:77: -> ^( IF parenthesizedExpression $ifStat)
                            # Java.g:663:81: ^( IF parenthesizedExpression $ifStat)
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_IF.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())
                            self._adaptor.addChild(root_1, stream_ifStat.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt98 == 4:
                    # Java.g:665:9: FOR LPAREN ( forInit SEMI forCondition SEMI forUpdater RPAREN statement -> ^( FOR forInit forCondition forUpdater statement ) | localModifierList type IDENT COLON expression RPAREN statement -> ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement ) )
                    pass 
                    FOR283=self.match(self.input, FOR, self.FOLLOW_FOR_in_statement8900) 
                    if self._state.backtracking == 0:
                        stream_FOR.add(FOR283)
                    LPAREN284=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_statement8902) 
                    if self._state.backtracking == 0:
                        stream_LPAREN.add(LPAREN284)
                    # Java.g:666:9: ( forInit SEMI forCondition SEMI forUpdater RPAREN statement -> ^( FOR forInit forCondition forUpdater statement ) | localModifierList type IDENT COLON expression RPAREN statement -> ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement ) )
                    alt91 = 2
                    alt91 = self.dfa91.predict(self.input)
                    if alt91 == 1:
                        # Java.g:666:13: forInit SEMI forCondition SEMI forUpdater RPAREN statement
                        pass 
                        self._state.following.append(self.FOLLOW_forInit_in_statement8916)
                        forInit285 = self.forInit()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_forInit.add(forInit285.tree)
                        SEMI286=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement8918) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI286)
                        self._state.following.append(self.FOLLOW_forCondition_in_statement8920)
                        forCondition287 = self.forCondition()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_forCondition.add(forCondition287.tree)
                        SEMI288=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement8922) 
                        if self._state.backtracking == 0:
                            stream_SEMI.add(SEMI288)
                        self._state.following.append(self.FOLLOW_forUpdater_in_statement8924)
                        forUpdater289 = self.forUpdater()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_forUpdater.add(forUpdater289.tree)
                        RPAREN290=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_statement8926) 
                        if self._state.backtracking == 0:
                            stream_RPAREN.add(RPAREN290)
                        self._state.following.append(self.FOLLOW_statement_in_statement8928)
                        statement291 = self.statement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_statement.add(statement291.tree)

                        # AST Rewrite
                        # elements: forInit, forUpdater, FOR, statement, forCondition
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 666:77: -> ^( FOR forInit forCondition forUpdater statement )
                            # Java.g:666:81: ^( FOR forInit forCondition forUpdater statement )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_FOR.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_forInit.nextTree())
                            self._adaptor.addChild(root_1, stream_forCondition.nextTree())
                            self._adaptor.addChild(root_1, stream_forUpdater.nextTree())
                            self._adaptor.addChild(root_1, stream_statement.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt91 == 2:
                        # Java.g:667:13: localModifierList type IDENT COLON expression RPAREN statement
                        pass 
                        self._state.following.append(self.FOLLOW_localModifierList_in_statement8962)
                        localModifierList292 = self.localModifierList()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_localModifierList.add(localModifierList292.tree)
                        self._state.following.append(self.FOLLOW_type_in_statement8964)
                        type293 = self.type()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_type.add(type293.tree)
                        IDENT294=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_statement8966) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(IDENT294)
                        COLON295=self.match(self.input, COLON, self.FOLLOW_COLON_in_statement8968) 
                        if self._state.backtracking == 0:
                            stream_COLON.add(COLON295)
                        self._state.following.append(self.FOLLOW_expression_in_statement8970)
                        expression296 = self.expression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_expression.add(expression296.tree)
                        RPAREN297=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_statement8972) 
                        if self._state.backtracking == 0:
                            stream_RPAREN.add(RPAREN297)
                        self._state.following.append(self.FOLLOW_statement_in_statement8974)
                        statement298 = self.statement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_statement.add(statement298.tree)

                        # AST Rewrite
                        # elements: IDENT, expression, localModifierList, type, statement
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 668:77: -> ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement )
                            # Java.g:668:81: ^( FOR_EACH[$FOR, \"FOR_EACH\"] localModifierList type IDENT expression statement )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(FOR_EACH, FOR283, "FOR_EACH"), root_1)

                            self._adaptor.addChild(root_1, stream_localModifierList.nextTree())
                            self._adaptor.addChild(root_1, stream_type.nextTree())
                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                            self._adaptor.addChild(root_1, stream_expression.nextTree())
                            self._adaptor.addChild(root_1, stream_statement.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt98 == 5:
                    # Java.g:670:9: WHILE parenthesizedExpression statement
                    pass 
                    WHILE299=self.match(self.input, WHILE, self.FOLLOW_WHILE_in_statement9088) 
                    if self._state.backtracking == 0:
                        stream_WHILE.add(WHILE299)
                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_statement9090)
                    parenthesizedExpression300 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_parenthesizedExpression.add(parenthesizedExpression300.tree)
                    self._state.following.append(self.FOLLOW_statement_in_statement9092)
                    statement301 = self.statement()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_statement.add(statement301.tree)

                    # AST Rewrite
                    # elements: statement, WHILE, parenthesizedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 670:77: -> ^( WHILE parenthesizedExpression statement )
                        # Java.g:670:81: ^( WHILE parenthesizedExpression statement )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_WHILE.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())
                        self._adaptor.addChild(root_1, stream_statement.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 6:
                    # Java.g:671:9: DO statement WHILE parenthesizedExpression SEMI
                    pass 
                    DO302=self.match(self.input, DO, self.FOLLOW_DO_in_statement9141) 
                    if self._state.backtracking == 0:
                        stream_DO.add(DO302)
                    self._state.following.append(self.FOLLOW_statement_in_statement9143)
                    statement303 = self.statement()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_statement.add(statement303.tree)
                    WHILE304=self.match(self.input, WHILE, self.FOLLOW_WHILE_in_statement9145) 
                    if self._state.backtracking == 0:
                        stream_WHILE.add(WHILE304)
                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_statement9147)
                    parenthesizedExpression305 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_parenthesizedExpression.add(parenthesizedExpression305.tree)
                    SEMI306=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9149) 
                    if self._state.backtracking == 0:
                        stream_SEMI.add(SEMI306)

                    # AST Rewrite
                    # elements: DO, parenthesizedExpression, statement
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 671:77: -> ^( DO statement parenthesizedExpression )
                        # Java.g:671:81: ^( DO statement parenthesizedExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_DO.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_statement.nextTree())
                        self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 7:
                    # Java.g:672:9: TRY block ( catches ( finallyClause )? | finallyClause )
                    pass 
                    TRY307=self.match(self.input, TRY, self.FOLLOW_TRY_in_statement9190) 
                    if self._state.backtracking == 0:
                        stream_TRY.add(TRY307)
                    self._state.following.append(self.FOLLOW_block_in_statement9192)
                    block308 = self.block()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_block.add(block308.tree)
                    # Java.g:672:19: ( catches ( finallyClause )? | finallyClause )
                    alt93 = 2
                    LA93_0 = self.input.LA(1)

                    if (LA93_0 == CATCH) :
                        alt93 = 1
                    elif (LA93_0 == FINALLY) :
                        alt93 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 93, 0, self.input)

                        raise nvae

                    if alt93 == 1:
                        # Java.g:672:20: catches ( finallyClause )?
                        pass 
                        self._state.following.append(self.FOLLOW_catches_in_statement9195)
                        catches309 = self.catches()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_catches.add(catches309.tree)
                        # Java.g:672:28: ( finallyClause )?
                        alt92 = 2
                        LA92_0 = self.input.LA(1)

                        if (LA92_0 == FINALLY) :
                            alt92 = 1
                        if alt92 == 1:
                            # Java.g:0:0: finallyClause
                            pass 
                            self._state.following.append(self.FOLLOW_finallyClause_in_statement9197)
                            finallyClause310 = self.finallyClause()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_finallyClause.add(finallyClause310.tree)





                    elif alt93 == 2:
                        # Java.g:672:45: finallyClause
                        pass 
                        self._state.following.append(self.FOLLOW_finallyClause_in_statement9202)
                        finallyClause311 = self.finallyClause()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_finallyClause.add(finallyClause311.tree)




                    # AST Rewrite
                    # elements: block, finallyClause, catches, TRY
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 672:77: -> ^( TRY block ( catches )? ( finallyClause )? )
                        # Java.g:672:81: ^( TRY block ( catches )? ( finallyClause )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_TRY.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_block.nextTree())
                        # Java.g:672:93: ( catches )?
                        if stream_catches.hasNext():
                            self._adaptor.addChild(root_1, stream_catches.nextTree())


                        stream_catches.reset();
                        # Java.g:672:102: ( finallyClause )?
                        if stream_finallyClause.hasNext():
                            self._adaptor.addChild(root_1, stream_finallyClause.nextTree())


                        stream_finallyClause.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 8:
                    # Java.g:673:9: SWITCH parenthesizedExpression LCURLY ( switchBlockLabels )? RCURLY
                    pass 
                    SWITCH312=self.match(self.input, SWITCH, self.FOLLOW_SWITCH_in_statement9245) 
                    if self._state.backtracking == 0:
                        stream_SWITCH.add(SWITCH312)
                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_statement9247)
                    parenthesizedExpression313 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_parenthesizedExpression.add(parenthesizedExpression313.tree)
                    LCURLY314=self.match(self.input, LCURLY, self.FOLLOW_LCURLY_in_statement9249) 
                    if self._state.backtracking == 0:
                        stream_LCURLY.add(LCURLY314)
                    # Java.g:673:47: ( switchBlockLabels )?
                    alt94 = 2
                    LA94_0 = self.input.LA(1)

                    if (LA94_0 == CASE or LA94_0 == DEFAULT) :
                        alt94 = 1
                    elif (LA94_0 == RCURLY) :
                        LA94_2 = self.input.LA(2)

                        if (self.synpred130_Java()) :
                            alt94 = 1
                    if alt94 == 1:
                        # Java.g:0:0: switchBlockLabels
                        pass 
                        self._state.following.append(self.FOLLOW_switchBlockLabels_in_statement9251)
                        switchBlockLabels315 = self.switchBlockLabels()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_switchBlockLabels.add(switchBlockLabels315.tree)



                    RCURLY316=self.match(self.input, RCURLY, self.FOLLOW_RCURLY_in_statement9254) 
                    if self._state.backtracking == 0:
                        stream_RCURLY.add(RCURLY316)

                    # AST Rewrite
                    # elements: switchBlockLabels, parenthesizedExpression, SWITCH
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 673:78: -> ^( SWITCH parenthesizedExpression ( switchBlockLabels )? )
                        # Java.g:673:82: ^( SWITCH parenthesizedExpression ( switchBlockLabels )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_SWITCH.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())
                        # Java.g:673:115: ( switchBlockLabels )?
                        if stream_switchBlockLabels.hasNext():
                            self._adaptor.addChild(root_1, stream_switchBlockLabels.nextTree())


                        stream_switchBlockLabels.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 9:
                    # Java.g:674:9: SYNCHRONIZED parenthesizedExpression block
                    pass 
                    SYNCHRONIZED317=self.match(self.input, SYNCHRONIZED, self.FOLLOW_SYNCHRONIZED_in_statement9281) 
                    if self._state.backtracking == 0:
                        stream_SYNCHRONIZED.add(SYNCHRONIZED317)
                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_statement9283)
                    parenthesizedExpression318 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_parenthesizedExpression.add(parenthesizedExpression318.tree)
                    self._state.following.append(self.FOLLOW_block_in_statement9285)
                    block319 = self.block()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_block.add(block319.tree)

                    # AST Rewrite
                    # elements: SYNCHRONIZED, block, parenthesizedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 674:77: -> ^( SYNCHRONIZED parenthesizedExpression block )
                        # Java.g:674:81: ^( SYNCHRONIZED parenthesizedExpression block )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_SYNCHRONIZED.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_parenthesizedExpression.nextTree())
                        self._adaptor.addChild(root_1, stream_block.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 10:
                    # Java.g:675:9: RETURN ( expression )? SEMI
                    pass 
                    RETURN320=self.match(self.input, RETURN, self.FOLLOW_RETURN_in_statement9331) 
                    if self._state.backtracking == 0:
                        stream_RETURN.add(RETURN320)
                    # Java.g:675:16: ( expression )?
                    alt95 = 2
                    LA95_0 = self.input.LA(1)

                    if (LA95_0 == DEC or LA95_0 == INC or LA95_0 == LESS_THAN or LA95_0 == LOGICAL_NOT or (LPAREN <= LA95_0 <= MINUS) or LA95_0 == NOT or LA95_0 == PLUS or LA95_0 == BOOLEAN or LA95_0 == BYTE or LA95_0 == CHAR or LA95_0 == DOUBLE or LA95_0 == FALSE or LA95_0 == FLOAT or (INT <= LA95_0 <= LONG) or (NEW <= LA95_0 <= NULL) or LA95_0 == SHORT or LA95_0 == SUPER or LA95_0 == THIS or LA95_0 == TRUE or LA95_0 == VOID or (IDENT <= LA95_0 <= STRING_LITERAL)) :
                        alt95 = 1
                    if alt95 == 1:
                        # Java.g:0:0: expression
                        pass 
                        self._state.following.append(self.FOLLOW_expression_in_statement9333)
                        expression321 = self.expression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_expression.add(expression321.tree)



                    SEMI322=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9336) 
                    if self._state.backtracking == 0:
                        stream_SEMI.add(SEMI322)

                    # AST Rewrite
                    # elements: RETURN, expression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 675:77: -> ^( RETURN ( expression )? )
                        # Java.g:675:81: ^( RETURN ( expression )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_RETURN.nextNode(), root_1)

                        # Java.g:675:90: ( expression )?
                        if stream_expression.hasNext():
                            self._adaptor.addChild(root_1, stream_expression.nextTree())


                        stream_expression.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 11:
                    # Java.g:676:9: THROW expression SEMI
                    pass 
                    THROW323=self.match(self.input, THROW, self.FOLLOW_THROW_in_statement9400) 
                    if self._state.backtracking == 0:
                        stream_THROW.add(THROW323)
                    self._state.following.append(self.FOLLOW_expression_in_statement9402)
                    expression324 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expression.add(expression324.tree)
                    SEMI325=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9404) 
                    if self._state.backtracking == 0:
                        stream_SEMI.add(SEMI325)

                    # AST Rewrite
                    # elements: expression, THROW
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 676:77: -> ^( THROW expression )
                        # Java.g:676:81: ^( THROW expression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_THROW.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_expression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 12:
                    # Java.g:677:9: BREAK ( IDENT )? SEMI
                    pass 
                    BREAK326=self.match(self.input, BREAK, self.FOLLOW_BREAK_in_statement9469) 
                    if self._state.backtracking == 0:
                        stream_BREAK.add(BREAK326)
                    # Java.g:677:15: ( IDENT )?
                    alt96 = 2
                    LA96_0 = self.input.LA(1)

                    if (LA96_0 == IDENT) :
                        alt96 = 1
                    if alt96 == 1:
                        # Java.g:0:0: IDENT
                        pass 
                        IDENT327=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_statement9471) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(IDENT327)



                    SEMI328=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9474) 
                    if self._state.backtracking == 0:
                        stream_SEMI.add(SEMI328)

                    # AST Rewrite
                    # elements: BREAK, IDENT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 677:77: -> ^( BREAK ( IDENT )? )
                        # Java.g:677:81: ^( BREAK ( IDENT )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_BREAK.nextNode(), root_1)

                        # Java.g:677:89: ( IDENT )?
                        if stream_IDENT.hasNext():
                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())


                        stream_IDENT.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 13:
                    # Java.g:678:9: CONTINUE ( IDENT )? SEMI
                    pass 
                    CONTINUE329=self.match(self.input, CONTINUE, self.FOLLOW_CONTINUE_in_statement9544) 
                    if self._state.backtracking == 0:
                        stream_CONTINUE.add(CONTINUE329)
                    # Java.g:678:18: ( IDENT )?
                    alt97 = 2
                    LA97_0 = self.input.LA(1)

                    if (LA97_0 == IDENT) :
                        alt97 = 1
                    if alt97 == 1:
                        # Java.g:0:0: IDENT
                        pass 
                        IDENT330=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_statement9546) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(IDENT330)



                    SEMI331=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9549) 
                    if self._state.backtracking == 0:
                        stream_SEMI.add(SEMI331)

                    # AST Rewrite
                    # elements: CONTINUE, IDENT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 678:77: -> ^( CONTINUE ( IDENT )? )
                        # Java.g:678:81: ^( CONTINUE ( IDENT )? )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_CONTINUE.nextNode(), root_1)

                        # Java.g:678:92: ( IDENT )?
                        if stream_IDENT.hasNext():
                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())


                        stream_IDENT.reset();

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 14:
                    # Java.g:679:9: IDENT COLON statement
                    pass 
                    IDENT332=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_statement9616) 
                    if self._state.backtracking == 0:
                        stream_IDENT.add(IDENT332)
                    COLON333=self.match(self.input, COLON, self.FOLLOW_COLON_in_statement9618) 
                    if self._state.backtracking == 0:
                        stream_COLON.add(COLON333)
                    self._state.following.append(self.FOLLOW_statement_in_statement9620)
                    statement334 = self.statement()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_statement.add(statement334.tree)

                    # AST Rewrite
                    # elements: statement, IDENT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 679:77: -> ^( LABELED_STATEMENT IDENT statement )
                        # Java.g:679:81: ^( LABELED_STATEMENT IDENT statement )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(LABELED_STATEMENT, "LABELED_STATEMENT"), root_1)

                        self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                        self._adaptor.addChild(root_1, stream_statement.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt98 == 15:
                    # Java.g:680:9: expression SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_expression_in_statement9687)
                    expression335 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, expression335.tree)
                    SEMI336=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9689)


                elif alt98 == 16:
                    # Java.g:681:9: SEMI
                    pass 
                    root_0 = self._adaptor.nil()

                    SEMI337=self.match(self.input, SEMI, self.FOLLOW_SEMI_in_statement9700)
                    if self._state.backtracking == 0:

                        SEMI337_tree = self._adaptor.createWithPayload(SEMI337)
                        self._adaptor.addChild(root_0, SEMI337_tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 75, statement_StartIndex, success)

            pass
        return retval

    # $ANTLR end "statement"

    class catches_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.catches_return, self).__init__()

            self.tree = None




    # $ANTLR start "catches"
    # Java.g:684:1: catches : ( catchClause )+ -> ^( CATCH_CLAUSE_LIST ( catchClause )+ ) ;
    def catches(self, ):

        retval = self.catches_return()
        retval.start = self.input.LT(1)
        catches_StartIndex = self.input.index()
        root_0 = None

        catchClause338 = None


        stream_catchClause = RewriteRuleSubtreeStream(self._adaptor, "rule catchClause")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 76):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:685:5: ( ( catchClause )+ -> ^( CATCH_CLAUSE_LIST ( catchClause )+ ) )
                # Java.g:685:9: ( catchClause )+
                pass 
                # Java.g:685:9: ( catchClause )+
                cnt99 = 0
                while True: #loop99
                    alt99 = 2
                    LA99_0 = self.input.LA(1)

                    if (LA99_0 == CATCH) :
                        alt99 = 1


                    if alt99 == 1:
                        # Java.g:0:0: catchClause
                        pass 
                        self._state.following.append(self.FOLLOW_catchClause_in_catches9720)
                        catchClause338 = self.catchClause()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_catchClause.add(catchClause338.tree)


                    else:
                        if cnt99 >= 1:
                            break #loop99

                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        eee = EarlyExitException(99, self.input)
                        raise eee

                    cnt99 += 1

                # AST Rewrite
                # elements: catchClause
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 686:9: -> ^( CATCH_CLAUSE_LIST ( catchClause )+ )
                    # Java.g:686:13: ^( CATCH_CLAUSE_LIST ( catchClause )+ )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(CATCH_CLAUSE_LIST, "CATCH_CLAUSE_LIST"), root_1)

                    # Java.g:686:33: ( catchClause )+
                    if not (stream_catchClause.hasNext()):
                        raise RewriteEarlyExitException()

                    while stream_catchClause.hasNext():
                        self._adaptor.addChild(root_1, stream_catchClause.nextTree())


                    stream_catchClause.reset()

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 76, catches_StartIndex, success)

            pass
        return retval

    # $ANTLR end "catches"

    class catchClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.catchClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "catchClause"
    # Java.g:689:1: catchClause : CATCH LPAREN formalParameterStandardDecl RPAREN block ;
    def catchClause(self, ):

        retval = self.catchClause_return()
        retval.start = self.input.LT(1)
        catchClause_StartIndex = self.input.index()
        root_0 = None

        CATCH339 = None
        LPAREN340 = None
        RPAREN342 = None
        formalParameterStandardDecl341 = None

        block343 = None


        CATCH339_tree = None
        LPAREN340_tree = None
        RPAREN342_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 77):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:690:5: ( CATCH LPAREN formalParameterStandardDecl RPAREN block )
                # Java.g:690:9: CATCH LPAREN formalParameterStandardDecl RPAREN block
                pass 
                root_0 = self._adaptor.nil()

                CATCH339=self.match(self.input, CATCH, self.FOLLOW_CATCH_in_catchClause9758)
                if self._state.backtracking == 0:

                    CATCH339_tree = self._adaptor.createWithPayload(CATCH339)
                    root_0 = self._adaptor.becomeRoot(CATCH339_tree, root_0)

                LPAREN340=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_catchClause9761)
                self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_catchClause9764)
                formalParameterStandardDecl341 = self.formalParameterStandardDecl()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, formalParameterStandardDecl341.tree)
                RPAREN342=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_catchClause9766)
                self._state.following.append(self.FOLLOW_block_in_catchClause9769)
                block343 = self.block()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, block343.tree)



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 77, catchClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "catchClause"

    class finallyClause_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.finallyClause_return, self).__init__()

            self.tree = None




    # $ANTLR start "finallyClause"
    # Java.g:693:1: finallyClause : FINALLY block -> block ;
    def finallyClause(self, ):

        retval = self.finallyClause_return()
        retval.start = self.input.LT(1)
        finallyClause_StartIndex = self.input.index()
        root_0 = None

        FINALLY344 = None
        block345 = None


        FINALLY344_tree = None
        stream_FINALLY = RewriteRuleTokenStream(self._adaptor, "token FINALLY")
        stream_block = RewriteRuleSubtreeStream(self._adaptor, "rule block")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 78):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:694:5: ( FINALLY block -> block )
                # Java.g:694:9: FINALLY block
                pass 
                FINALLY344=self.match(self.input, FINALLY, self.FOLLOW_FINALLY_in_finallyClause9788) 
                if self._state.backtracking == 0:
                    stream_FINALLY.add(FINALLY344)
                self._state.following.append(self.FOLLOW_block_in_finallyClause9790)
                block345 = self.block()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_block.add(block345.tree)

                # AST Rewrite
                # elements: block
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 695:9: -> block
                    self._adaptor.addChild(root_0, stream_block.nextTree())



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 78, finallyClause_StartIndex, success)

            pass
        return retval

    # $ANTLR end "finallyClause"

    class switchBlockLabels_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.switchBlockLabels_return, self).__init__()

            self.tree = None




    # $ANTLR start "switchBlockLabels"
    # Java.g:698:1: switchBlockLabels : (c0= switchCaseLabels )? ( switchDefaultLabel )? (c1= switchCaseLabels )? -> ^( SWITCH_BLOCK_LABEL_LIST ( $c0)? ( switchDefaultLabel )? ( $c1)? ) ;
    def switchBlockLabels(self, ):

        retval = self.switchBlockLabels_return()
        retval.start = self.input.LT(1)
        switchBlockLabels_StartIndex = self.input.index()
        root_0 = None

        c0 = None

        c1 = None

        switchDefaultLabel346 = None


        stream_switchDefaultLabel = RewriteRuleSubtreeStream(self._adaptor, "rule switchDefaultLabel")
        stream_switchCaseLabels = RewriteRuleSubtreeStream(self._adaptor, "rule switchCaseLabels")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 79):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:702:5: ( (c0= switchCaseLabels )? ( switchDefaultLabel )? (c1= switchCaseLabels )? -> ^( SWITCH_BLOCK_LABEL_LIST ( $c0)? ( switchDefaultLabel )? ( $c1)? ) )
                # Java.g:702:9: (c0= switchCaseLabels )? ( switchDefaultLabel )? (c1= switchCaseLabels )?
                pass 
                # Java.g:702:11: (c0= switchCaseLabels )?
                alt100 = 2
                LA100 = self.input.LA(1)
                if LA100 == CASE:
                    LA100_1 = self.input.LA(2)

                    if (self.synpred143_Java()) :
                        alt100 = 1
                elif LA100 == DEFAULT:
                    LA100_2 = self.input.LA(2)

                    if (self.synpred143_Java()) :
                        alt100 = 1
                elif LA100 == RCURLY:
                    LA100_3 = self.input.LA(2)

                    if (self.synpred143_Java()) :
                        alt100 = 1
                elif LA100 == EOF:
                    LA100_4 = self.input.LA(2)

                    if (self.synpred143_Java()) :
                        alt100 = 1
                if alt100 == 1:
                    # Java.g:0:0: c0= switchCaseLabels
                    pass 
                    self._state.following.append(self.FOLLOW_switchCaseLabels_in_switchBlockLabels9839)
                    c0 = self.switchCaseLabels()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_switchCaseLabels.add(c0.tree)



                # Java.g:702:30: ( switchDefaultLabel )?
                alt101 = 2
                LA101_0 = self.input.LA(1)

                if (LA101_0 == DEFAULT) :
                    alt101 = 1
                if alt101 == 1:
                    # Java.g:0:0: switchDefaultLabel
                    pass 
                    self._state.following.append(self.FOLLOW_switchDefaultLabel_in_switchBlockLabels9842)
                    switchDefaultLabel346 = self.switchDefaultLabel()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_switchDefaultLabel.add(switchDefaultLabel346.tree)



                # Java.g:702:52: (c1= switchCaseLabels )?
                alt102 = 2
                LA102 = self.input.LA(1)
                if LA102 == CASE:
                    alt102 = 1
                elif LA102 == RCURLY:
                    LA102_2 = self.input.LA(2)

                    if (self.synpred145_Java()) :
                        alt102 = 1
                elif LA102 == EOF:
                    LA102_3 = self.input.LA(2)

                    if (self.synpred145_Java()) :
                        alt102 = 1
                if alt102 == 1:
                    # Java.g:0:0: c1= switchCaseLabels
                    pass 
                    self._state.following.append(self.FOLLOW_switchCaseLabels_in_switchBlockLabels9847)
                    c1 = self.switchCaseLabels()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_switchCaseLabels.add(c1.tree)




                # AST Rewrite
                # elements: c0, switchDefaultLabel, c1
                # token labels: 
                # rule labels: retval, c1, c0
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    if c1 is not None:
                        stream_c1 = RewriteRuleSubtreeStream(self._adaptor, "rule c1", c1.tree)
                    else:
                        stream_c1 = RewriteRuleSubtreeStream(self._adaptor, "token c1", None)


                    if c0 is not None:
                        stream_c0 = RewriteRuleSubtreeStream(self._adaptor, "rule c0", c0.tree)
                    else:
                        stream_c0 = RewriteRuleSubtreeStream(self._adaptor, "token c0", None)


                    root_0 = self._adaptor.nil()
                    # 703:9: -> ^( SWITCH_BLOCK_LABEL_LIST ( $c0)? ( switchDefaultLabel )? ( $c1)? )
                    # Java.g:703:13: ^( SWITCH_BLOCK_LABEL_LIST ( $c0)? ( switchDefaultLabel )? ( $c1)? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(SWITCH_BLOCK_LABEL_LIST, "SWITCH_BLOCK_LABEL_LIST"), root_1)

                    # Java.g:703:39: ( $c0)?
                    if stream_c0.hasNext():
                        self._adaptor.addChild(root_1, stream_c0.nextTree())


                    stream_c0.reset();
                    # Java.g:703:44: ( switchDefaultLabel )?
                    if stream_switchDefaultLabel.hasNext():
                        self._adaptor.addChild(root_1, stream_switchDefaultLabel.nextTree())


                    stream_switchDefaultLabel.reset();
                    # Java.g:703:64: ( $c1)?
                    if stream_c1.hasNext():
                        self._adaptor.addChild(root_1, stream_c1.nextTree())


                    stream_c1.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 79, switchBlockLabels_StartIndex, success)

            pass
        return retval

    # $ANTLR end "switchBlockLabels"

    class switchCaseLabels_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.switchCaseLabels_return, self).__init__()

            self.tree = None




    # $ANTLR start "switchCaseLabels"
    # Java.g:706:1: switchCaseLabels : ( switchCaseLabel )* ;
    def switchCaseLabels(self, ):

        retval = self.switchCaseLabels_return()
        retval.start = self.input.LT(1)
        switchCaseLabels_StartIndex = self.input.index()
        root_0 = None

        switchCaseLabel347 = None



        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 80):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:707:5: ( ( switchCaseLabel )* )
                # Java.g:707:9: ( switchCaseLabel )*
                pass 
                root_0 = self._adaptor.nil()

                # Java.g:707:9: ( switchCaseLabel )*
                while True: #loop103
                    alt103 = 2
                    LA103_0 = self.input.LA(1)

                    if (LA103_0 == CASE) :
                        LA103_2 = self.input.LA(2)

                        if (self.synpred146_Java()) :
                            alt103 = 1




                    if alt103 == 1:
                        # Java.g:0:0: switchCaseLabel
                        pass 
                        self._state.following.append(self.FOLLOW_switchCaseLabel_in_switchCaseLabels9893)
                        switchCaseLabel347 = self.switchCaseLabel()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, switchCaseLabel347.tree)


                    else:
                        break #loop103



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 80, switchCaseLabels_StartIndex, success)

            pass
        return retval

    # $ANTLR end "switchCaseLabels"

    class switchCaseLabel_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.switchCaseLabel_return, self).__init__()

            self.tree = None




    # $ANTLR start "switchCaseLabel"
    # Java.g:710:1: switchCaseLabel : CASE expression COLON ( blockStatement )* ;
    def switchCaseLabel(self, ):

        retval = self.switchCaseLabel_return()
        retval.start = self.input.LT(1)
        switchCaseLabel_StartIndex = self.input.index()
        root_0 = None

        CASE348 = None
        COLON350 = None
        expression349 = None

        blockStatement351 = None


        CASE348_tree = None
        COLON350_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 81):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:711:5: ( CASE expression COLON ( blockStatement )* )
                # Java.g:711:9: CASE expression COLON ( blockStatement )*
                pass 
                root_0 = self._adaptor.nil()

                CASE348=self.match(self.input, CASE, self.FOLLOW_CASE_in_switchCaseLabel9913)
                if self._state.backtracking == 0:

                    CASE348_tree = self._adaptor.createWithPayload(CASE348)
                    root_0 = self._adaptor.becomeRoot(CASE348_tree, root_0)

                self._state.following.append(self.FOLLOW_expression_in_switchCaseLabel9916)
                expression349 = self.expression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, expression349.tree)
                COLON350=self.match(self.input, COLON, self.FOLLOW_COLON_in_switchCaseLabel9918)
                # Java.g:711:33: ( blockStatement )*
                while True: #loop104
                    alt104 = 2
                    LA104_0 = self.input.LA(1)

                    if (LA104_0 == AT or LA104_0 == DEC or LA104_0 == INC or LA104_0 == LCURLY or LA104_0 == LESS_THAN or LA104_0 == LOGICAL_NOT or (LPAREN <= LA104_0 <= MINUS) or LA104_0 == NOT or LA104_0 == PLUS or LA104_0 == SEMI or (ABSTRACT <= LA104_0 <= BYTE) or (CHAR <= LA104_0 <= CONTINUE) or (DO <= LA104_0 <= DOUBLE) or LA104_0 == ENUM or (FALSE <= LA104_0 <= FINAL) or (FLOAT <= LA104_0 <= IF) or LA104_0 == INTERFACE or (INT <= LA104_0 <= NULL) or (PRIVATE <= LA104_0 <= THROW) or (TRANSIENT <= LA104_0 <= WHILE) or (IDENT <= LA104_0 <= STRING_LITERAL)) :
                        alt104 = 1


                    if alt104 == 1:
                        # Java.g:0:0: blockStatement
                        pass 
                        self._state.following.append(self.FOLLOW_blockStatement_in_switchCaseLabel9921)
                        blockStatement351 = self.blockStatement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, blockStatement351.tree)


                    else:
                        break #loop104



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 81, switchCaseLabel_StartIndex, success)

            pass
        return retval

    # $ANTLR end "switchCaseLabel"

    class switchDefaultLabel_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.switchDefaultLabel_return, self).__init__()

            self.tree = None




    # $ANTLR start "switchDefaultLabel"
    # Java.g:714:1: switchDefaultLabel : DEFAULT COLON ( blockStatement )* ;
    def switchDefaultLabel(self, ):

        retval = self.switchDefaultLabel_return()
        retval.start = self.input.LT(1)
        switchDefaultLabel_StartIndex = self.input.index()
        root_0 = None

        DEFAULT352 = None
        COLON353 = None
        blockStatement354 = None


        DEFAULT352_tree = None
        COLON353_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 82):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:715:5: ( DEFAULT COLON ( blockStatement )* )
                # Java.g:715:9: DEFAULT COLON ( blockStatement )*
                pass 
                root_0 = self._adaptor.nil()

                DEFAULT352=self.match(self.input, DEFAULT, self.FOLLOW_DEFAULT_in_switchDefaultLabel9941)
                if self._state.backtracking == 0:

                    DEFAULT352_tree = self._adaptor.createWithPayload(DEFAULT352)
                    root_0 = self._adaptor.becomeRoot(DEFAULT352_tree, root_0)

                COLON353=self.match(self.input, COLON, self.FOLLOW_COLON_in_switchDefaultLabel9944)
                # Java.g:715:25: ( blockStatement )*
                while True: #loop105
                    alt105 = 2
                    LA105_0 = self.input.LA(1)

                    if (LA105_0 == AT or LA105_0 == DEC or LA105_0 == INC or LA105_0 == LCURLY or LA105_0 == LESS_THAN or LA105_0 == LOGICAL_NOT or (LPAREN <= LA105_0 <= MINUS) or LA105_0 == NOT or LA105_0 == PLUS or LA105_0 == SEMI or (ABSTRACT <= LA105_0 <= BYTE) or (CHAR <= LA105_0 <= CONTINUE) or (DO <= LA105_0 <= DOUBLE) or LA105_0 == ENUM or (FALSE <= LA105_0 <= FINAL) or (FLOAT <= LA105_0 <= IF) or LA105_0 == INTERFACE or (INT <= LA105_0 <= NULL) or (PRIVATE <= LA105_0 <= THROW) or (TRANSIENT <= LA105_0 <= WHILE) or (IDENT <= LA105_0 <= STRING_LITERAL)) :
                        alt105 = 1


                    if alt105 == 1:
                        # Java.g:0:0: blockStatement
                        pass 
                        self._state.following.append(self.FOLLOW_blockStatement_in_switchDefaultLabel9947)
                        blockStatement354 = self.blockStatement()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, blockStatement354.tree)


                    else:
                        break #loop105



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 82, switchDefaultLabel_StartIndex, success)

            pass
        return retval

    # $ANTLR end "switchDefaultLabel"

    class forInit_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.forInit_return, self).__init__()

            self.tree = None




    # $ANTLR start "forInit"
    # Java.g:718:1: forInit : ( localVariableDeclaration -> ^( FOR_INIT localVariableDeclaration ) | expressionList -> ^( FOR_INIT expressionList ) | -> ^( FOR_INIT ) );
    def forInit(self, ):

        retval = self.forInit_return()
        retval.start = self.input.LT(1)
        forInit_StartIndex = self.input.index()
        root_0 = None

        localVariableDeclaration355 = None

        expressionList356 = None


        stream_expressionList = RewriteRuleSubtreeStream(self._adaptor, "rule expressionList")
        stream_localVariableDeclaration = RewriteRuleSubtreeStream(self._adaptor, "rule localVariableDeclaration")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 83):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:719:5: ( localVariableDeclaration -> ^( FOR_INIT localVariableDeclaration ) | expressionList -> ^( FOR_INIT expressionList ) | -> ^( FOR_INIT ) )
                alt106 = 3
                alt106 = self.dfa106.predict(self.input)
                if alt106 == 1:
                    # Java.g:719:9: localVariableDeclaration
                    pass 
                    self._state.following.append(self.FOLLOW_localVariableDeclaration_in_forInit9967)
                    localVariableDeclaration355 = self.localVariableDeclaration()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_localVariableDeclaration.add(localVariableDeclaration355.tree)

                    # AST Rewrite
                    # elements: localVariableDeclaration
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 719:37: -> ^( FOR_INIT localVariableDeclaration )
                        # Java.g:719:41: ^( FOR_INIT localVariableDeclaration )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FOR_INIT, "FOR_INIT"), root_1)

                        self._adaptor.addChild(root_1, stream_localVariableDeclaration.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt106 == 2:
                    # Java.g:720:9: expressionList
                    pass 
                    self._state.following.append(self.FOLLOW_expressionList_in_forInit9989)
                    expressionList356 = self.expressionList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expressionList.add(expressionList356.tree)

                    # AST Rewrite
                    # elements: expressionList
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 720:37: -> ^( FOR_INIT expressionList )
                        # Java.g:720:41: ^( FOR_INIT expressionList )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FOR_INIT, "FOR_INIT"), root_1)

                        self._adaptor.addChild(root_1, stream_expressionList.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt106 == 3:
                    # Java.g:721:37: 
                    pass 
                    # AST Rewrite
                    # elements: 
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 721:37: -> ^( FOR_INIT )
                        # Java.g:721:41: ^( FOR_INIT )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FOR_INIT, "FOR_INIT"), root_1)

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 83, forInit_StartIndex, success)

            pass
        return retval

    # $ANTLR end "forInit"

    class forCondition_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.forCondition_return, self).__init__()

            self.tree = None




    # $ANTLR start "forCondition"
    # Java.g:724:1: forCondition : ( expression )? -> ^( FOR_CONDITION ( expression )? ) ;
    def forCondition(self, ):

        retval = self.forCondition_return()
        retval.start = self.input.LT(1)
        forCondition_StartIndex = self.input.index()
        root_0 = None

        expression357 = None


        stream_expression = RewriteRuleSubtreeStream(self._adaptor, "rule expression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 84):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:725:5: ( ( expression )? -> ^( FOR_CONDITION ( expression )? ) )
                # Java.g:725:9: ( expression )?
                pass 
                # Java.g:725:9: ( expression )?
                alt107 = 2
                LA107_0 = self.input.LA(1)

                if (LA107_0 == DEC or LA107_0 == INC or LA107_0 == LESS_THAN or LA107_0 == LOGICAL_NOT or (LPAREN <= LA107_0 <= MINUS) or LA107_0 == NOT or LA107_0 == PLUS or LA107_0 == BOOLEAN or LA107_0 == BYTE or LA107_0 == CHAR or LA107_0 == DOUBLE or LA107_0 == FALSE or LA107_0 == FLOAT or (INT <= LA107_0 <= LONG) or (NEW <= LA107_0 <= NULL) or LA107_0 == SHORT or LA107_0 == SUPER or LA107_0 == THIS or LA107_0 == TRUE or LA107_0 == VOID or (IDENT <= LA107_0 <= STRING_LITERAL)) :
                    alt107 = 1
                if alt107 == 1:
                    # Java.g:0:0: expression
                    pass 
                    self._state.following.append(self.FOLLOW_expression_in_forCondition10073)
                    expression357 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expression.add(expression357.tree)




                # AST Rewrite
                # elements: expression
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 726:9: -> ^( FOR_CONDITION ( expression )? )
                    # Java.g:726:13: ^( FOR_CONDITION ( expression )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FOR_CONDITION, "FOR_CONDITION"), root_1)

                    # Java.g:726:29: ( expression )?
                    if stream_expression.hasNext():
                        self._adaptor.addChild(root_1, stream_expression.nextTree())


                    stream_expression.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 84, forCondition_StartIndex, success)

            pass
        return retval

    # $ANTLR end "forCondition"

    class forUpdater_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.forUpdater_return, self).__init__()

            self.tree = None




    # $ANTLR start "forUpdater"
    # Java.g:729:1: forUpdater : ( expressionList )? -> ^( FOR_UPDATE ( expressionList )? ) ;
    def forUpdater(self, ):

        retval = self.forUpdater_return()
        retval.start = self.input.LT(1)
        forUpdater_StartIndex = self.input.index()
        root_0 = None

        expressionList358 = None


        stream_expressionList = RewriteRuleSubtreeStream(self._adaptor, "rule expressionList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 85):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:730:5: ( ( expressionList )? -> ^( FOR_UPDATE ( expressionList )? ) )
                # Java.g:730:9: ( expressionList )?
                pass 
                # Java.g:730:9: ( expressionList )?
                alt108 = 2
                LA108_0 = self.input.LA(1)

                if (LA108_0 == DEC or LA108_0 == INC or LA108_0 == LESS_THAN or LA108_0 == LOGICAL_NOT or (LPAREN <= LA108_0 <= MINUS) or LA108_0 == NOT or LA108_0 == PLUS or LA108_0 == BOOLEAN or LA108_0 == BYTE or LA108_0 == CHAR or LA108_0 == DOUBLE or LA108_0 == FALSE or LA108_0 == FLOAT or (INT <= LA108_0 <= LONG) or (NEW <= LA108_0 <= NULL) or LA108_0 == SHORT or LA108_0 == SUPER or LA108_0 == THIS or LA108_0 == TRUE or LA108_0 == VOID or (IDENT <= LA108_0 <= STRING_LITERAL)) :
                    alt108 = 1
                if alt108 == 1:
                    # Java.g:0:0: expressionList
                    pass 
                    self._state.following.append(self.FOLLOW_expressionList_in_forUpdater10111)
                    expressionList358 = self.expressionList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expressionList.add(expressionList358.tree)




                # AST Rewrite
                # elements: expressionList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 731:9: -> ^( FOR_UPDATE ( expressionList )? )
                    # Java.g:731:13: ^( FOR_UPDATE ( expressionList )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(FOR_UPDATE, "FOR_UPDATE"), root_1)

                    # Java.g:731:26: ( expressionList )?
                    if stream_expressionList.hasNext():
                        self._adaptor.addChild(root_1, stream_expressionList.nextTree())


                    stream_expressionList.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 85, forUpdater_StartIndex, success)

            pass
        return retval

    # $ANTLR end "forUpdater"

    class parenthesizedExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.parenthesizedExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "parenthesizedExpression"
    # Java.g:736:1: parenthesizedExpression : LPAREN expression RPAREN -> ^( PARENTESIZED_EXPR[$LPAREN, \"PARENTESIZED_EXPR\"] expression ) ;
    def parenthesizedExpression(self, ):

        retval = self.parenthesizedExpression_return()
        retval.start = self.input.LT(1)
        parenthesizedExpression_StartIndex = self.input.index()
        root_0 = None

        LPAREN359 = None
        RPAREN361 = None
        expression360 = None


        LPAREN359_tree = None
        RPAREN361_tree = None
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_expression = RewriteRuleSubtreeStream(self._adaptor, "rule expression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 86):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:737:5: ( LPAREN expression RPAREN -> ^( PARENTESIZED_EXPR[$LPAREN, \"PARENTESIZED_EXPR\"] expression ) )
                # Java.g:737:9: LPAREN expression RPAREN
                pass 
                LPAREN359=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_parenthesizedExpression10151) 
                if self._state.backtracking == 0:
                    stream_LPAREN.add(LPAREN359)
                self._state.following.append(self.FOLLOW_expression_in_parenthesizedExpression10153)
                expression360 = self.expression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_expression.add(expression360.tree)
                RPAREN361=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_parenthesizedExpression10155) 
                if self._state.backtracking == 0:
                    stream_RPAREN.add(RPAREN361)

                # AST Rewrite
                # elements: expression
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 738:9: -> ^( PARENTESIZED_EXPR[$LPAREN, \"PARENTESIZED_EXPR\"] expression )
                    # Java.g:738:13: ^( PARENTESIZED_EXPR[$LPAREN, \"PARENTESIZED_EXPR\"] expression )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(PARENTESIZED_EXPR, LPAREN359, "PARENTESIZED_EXPR"), root_1)

                    self._adaptor.addChild(root_1, stream_expression.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 86, parenthesizedExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "parenthesizedExpression"

    class expressionList_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.expressionList_return, self).__init__()

            self.tree = None




    # $ANTLR start "expressionList"
    # Java.g:741:1: expressionList : expression ( COMMA expression )* ;
    def expressionList(self, ):

        retval = self.expressionList_return()
        retval.start = self.input.LT(1)
        expressionList_StartIndex = self.input.index()
        root_0 = None

        COMMA363 = None
        expression362 = None

        expression364 = None


        COMMA363_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 87):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:742:5: ( expression ( COMMA expression )* )
                # Java.g:742:9: expression ( COMMA expression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_expression_in_expressionList10192)
                expression362 = self.expression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, expression362.tree)
                # Java.g:742:20: ( COMMA expression )*
                while True: #loop109
                    alt109 = 2
                    LA109_0 = self.input.LA(1)

                    if (LA109_0 == COMMA) :
                        alt109 = 1


                    if alt109 == 1:
                        # Java.g:742:21: COMMA expression
                        pass 
                        COMMA363=self.match(self.input, COMMA, self.FOLLOW_COMMA_in_expressionList10195)
                        self._state.following.append(self.FOLLOW_expression_in_expressionList10198)
                        expression364 = self.expression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, expression364.tree)


                    else:
                        break #loop109



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 87, expressionList_StartIndex, success)

            pass
        return retval

    # $ANTLR end "expressionList"

    class expression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.expression_return, self).__init__()

            self.tree = None




    # $ANTLR start "expression"
    # Java.g:745:1: expression : assignmentExpression -> ^( EXPR assignmentExpression ) ;
    def expression(self, ):

        retval = self.expression_return()
        retval.start = self.input.LT(1)
        expression_StartIndex = self.input.index()
        root_0 = None

        assignmentExpression365 = None


        stream_assignmentExpression = RewriteRuleSubtreeStream(self._adaptor, "rule assignmentExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 88):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:746:5: ( assignmentExpression -> ^( EXPR assignmentExpression ) )
                # Java.g:746:9: assignmentExpression
                pass 
                self._state.following.append(self.FOLLOW_assignmentExpression_in_expression10219)
                assignmentExpression365 = self.assignmentExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_assignmentExpression.add(assignmentExpression365.tree)

                # AST Rewrite
                # elements: assignmentExpression
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 747:9: -> ^( EXPR assignmentExpression )
                    # Java.g:747:13: ^( EXPR assignmentExpression )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(EXPR, "EXPR"), root_1)

                    self._adaptor.addChild(root_1, stream_assignmentExpression.nextTree())

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 88, expression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "expression"

    class assignmentExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.assignmentExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "assignmentExpression"
    # Java.g:750:1: assignmentExpression : conditionalExpression ( ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN ) assignmentExpression )? ;
    def assignmentExpression(self, ):

        retval = self.assignmentExpression_return()
        retval.start = self.input.LT(1)
        assignmentExpression_StartIndex = self.input.index()
        root_0 = None

        ASSIGN367 = None
        PLUS_ASSIGN368 = None
        MINUS_ASSIGN369 = None
        STAR_ASSIGN370 = None
        DIV_ASSIGN371 = None
        AND_ASSIGN372 = None
        OR_ASSIGN373 = None
        XOR_ASSIGN374 = None
        MOD_ASSIGN375 = None
        SHIFT_LEFT_ASSIGN376 = None
        SHIFT_RIGHT_ASSIGN377 = None
        BIT_SHIFT_RIGHT_ASSIGN378 = None
        conditionalExpression366 = None

        assignmentExpression379 = None


        ASSIGN367_tree = None
        PLUS_ASSIGN368_tree = None
        MINUS_ASSIGN369_tree = None
        STAR_ASSIGN370_tree = None
        DIV_ASSIGN371_tree = None
        AND_ASSIGN372_tree = None
        OR_ASSIGN373_tree = None
        XOR_ASSIGN374_tree = None
        MOD_ASSIGN375_tree = None
        SHIFT_LEFT_ASSIGN376_tree = None
        SHIFT_RIGHT_ASSIGN377_tree = None
        BIT_SHIFT_RIGHT_ASSIGN378_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 89):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:751:5: ( conditionalExpression ( ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN ) assignmentExpression )? )
                # Java.g:751:9: conditionalExpression ( ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN ) assignmentExpression )?
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_conditionalExpression_in_assignmentExpression10255)
                conditionalExpression366 = self.conditionalExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, conditionalExpression366.tree)
                # Java.g:752:9: ( ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN ) assignmentExpression )?
                alt111 = 2
                LA111_0 = self.input.LA(1)

                if ((AND_ASSIGN <= LA111_0 <= ASSIGN) or LA111_0 == BIT_SHIFT_RIGHT_ASSIGN or LA111_0 == DIV_ASSIGN or LA111_0 == MINUS_ASSIGN or LA111_0 == MOD_ASSIGN or LA111_0 == OR_ASSIGN or LA111_0 == PLUS_ASSIGN or LA111_0 == SHIFT_LEFT_ASSIGN or LA111_0 == SHIFT_RIGHT_ASSIGN or LA111_0 == STAR_ASSIGN or LA111_0 == XOR_ASSIGN) :
                    alt111 = 1
                if alt111 == 1:
                    # Java.g:752:13: ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN ) assignmentExpression
                    pass 
                    # Java.g:752:13: ( ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN | STAR_ASSIGN | DIV_ASSIGN | AND_ASSIGN | OR_ASSIGN | XOR_ASSIGN | MOD_ASSIGN | SHIFT_LEFT_ASSIGN | SHIFT_RIGHT_ASSIGN | BIT_SHIFT_RIGHT_ASSIGN )
                    alt110 = 12
                    LA110 = self.input.LA(1)
                    if LA110 == ASSIGN:
                        alt110 = 1
                    elif LA110 == PLUS_ASSIGN:
                        alt110 = 2
                    elif LA110 == MINUS_ASSIGN:
                        alt110 = 3
                    elif LA110 == STAR_ASSIGN:
                        alt110 = 4
                    elif LA110 == DIV_ASSIGN:
                        alt110 = 5
                    elif LA110 == AND_ASSIGN:
                        alt110 = 6
                    elif LA110 == OR_ASSIGN:
                        alt110 = 7
                    elif LA110 == XOR_ASSIGN:
                        alt110 = 8
                    elif LA110 == MOD_ASSIGN:
                        alt110 = 9
                    elif LA110 == SHIFT_LEFT_ASSIGN:
                        alt110 = 10
                    elif LA110 == SHIFT_RIGHT_ASSIGN:
                        alt110 = 11
                    elif LA110 == BIT_SHIFT_RIGHT_ASSIGN:
                        alt110 = 12
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 110, 0, self.input)

                        raise nvae

                    if alt110 == 1:
                        # Java.g:752:17: ASSIGN
                        pass 
                        ASSIGN367=self.match(self.input, ASSIGN, self.FOLLOW_ASSIGN_in_assignmentExpression10273)
                        if self._state.backtracking == 0:

                            ASSIGN367_tree = self._adaptor.createWithPayload(ASSIGN367)
                            root_0 = self._adaptor.becomeRoot(ASSIGN367_tree, root_0)



                    elif alt110 == 2:
                        # Java.g:753:17: PLUS_ASSIGN
                        pass 
                        PLUS_ASSIGN368=self.match(self.input, PLUS_ASSIGN, self.FOLLOW_PLUS_ASSIGN_in_assignmentExpression10292)
                        if self._state.backtracking == 0:

                            PLUS_ASSIGN368_tree = self._adaptor.createWithPayload(PLUS_ASSIGN368)
                            root_0 = self._adaptor.becomeRoot(PLUS_ASSIGN368_tree, root_0)



                    elif alt110 == 3:
                        # Java.g:754:17: MINUS_ASSIGN
                        pass 
                        MINUS_ASSIGN369=self.match(self.input, MINUS_ASSIGN, self.FOLLOW_MINUS_ASSIGN_in_assignmentExpression10311)
                        if self._state.backtracking == 0:

                            MINUS_ASSIGN369_tree = self._adaptor.createWithPayload(MINUS_ASSIGN369)
                            root_0 = self._adaptor.becomeRoot(MINUS_ASSIGN369_tree, root_0)



                    elif alt110 == 4:
                        # Java.g:755:17: STAR_ASSIGN
                        pass 
                        STAR_ASSIGN370=self.match(self.input, STAR_ASSIGN, self.FOLLOW_STAR_ASSIGN_in_assignmentExpression10330)
                        if self._state.backtracking == 0:

                            STAR_ASSIGN370_tree = self._adaptor.createWithPayload(STAR_ASSIGN370)
                            root_0 = self._adaptor.becomeRoot(STAR_ASSIGN370_tree, root_0)



                    elif alt110 == 5:
                        # Java.g:756:17: DIV_ASSIGN
                        pass 
                        DIV_ASSIGN371=self.match(self.input, DIV_ASSIGN, self.FOLLOW_DIV_ASSIGN_in_assignmentExpression10349)
                        if self._state.backtracking == 0:

                            DIV_ASSIGN371_tree = self._adaptor.createWithPayload(DIV_ASSIGN371)
                            root_0 = self._adaptor.becomeRoot(DIV_ASSIGN371_tree, root_0)



                    elif alt110 == 6:
                        # Java.g:757:17: AND_ASSIGN
                        pass 
                        AND_ASSIGN372=self.match(self.input, AND_ASSIGN, self.FOLLOW_AND_ASSIGN_in_assignmentExpression10368)
                        if self._state.backtracking == 0:

                            AND_ASSIGN372_tree = self._adaptor.createWithPayload(AND_ASSIGN372)
                            root_0 = self._adaptor.becomeRoot(AND_ASSIGN372_tree, root_0)



                    elif alt110 == 7:
                        # Java.g:758:17: OR_ASSIGN
                        pass 
                        OR_ASSIGN373=self.match(self.input, OR_ASSIGN, self.FOLLOW_OR_ASSIGN_in_assignmentExpression10387)
                        if self._state.backtracking == 0:

                            OR_ASSIGN373_tree = self._adaptor.createWithPayload(OR_ASSIGN373)
                            root_0 = self._adaptor.becomeRoot(OR_ASSIGN373_tree, root_0)



                    elif alt110 == 8:
                        # Java.g:759:17: XOR_ASSIGN
                        pass 
                        XOR_ASSIGN374=self.match(self.input, XOR_ASSIGN, self.FOLLOW_XOR_ASSIGN_in_assignmentExpression10406)
                        if self._state.backtracking == 0:

                            XOR_ASSIGN374_tree = self._adaptor.createWithPayload(XOR_ASSIGN374)
                            root_0 = self._adaptor.becomeRoot(XOR_ASSIGN374_tree, root_0)



                    elif alt110 == 9:
                        # Java.g:760:17: MOD_ASSIGN
                        pass 
                        MOD_ASSIGN375=self.match(self.input, MOD_ASSIGN, self.FOLLOW_MOD_ASSIGN_in_assignmentExpression10425)
                        if self._state.backtracking == 0:

                            MOD_ASSIGN375_tree = self._adaptor.createWithPayload(MOD_ASSIGN375)
                            root_0 = self._adaptor.becomeRoot(MOD_ASSIGN375_tree, root_0)



                    elif alt110 == 10:
                        # Java.g:761:17: SHIFT_LEFT_ASSIGN
                        pass 
                        SHIFT_LEFT_ASSIGN376=self.match(self.input, SHIFT_LEFT_ASSIGN, self.FOLLOW_SHIFT_LEFT_ASSIGN_in_assignmentExpression10444)
                        if self._state.backtracking == 0:

                            SHIFT_LEFT_ASSIGN376_tree = self._adaptor.createWithPayload(SHIFT_LEFT_ASSIGN376)
                            root_0 = self._adaptor.becomeRoot(SHIFT_LEFT_ASSIGN376_tree, root_0)



                    elif alt110 == 11:
                        # Java.g:762:17: SHIFT_RIGHT_ASSIGN
                        pass 
                        SHIFT_RIGHT_ASSIGN377=self.match(self.input, SHIFT_RIGHT_ASSIGN, self.FOLLOW_SHIFT_RIGHT_ASSIGN_in_assignmentExpression10463)
                        if self._state.backtracking == 0:

                            SHIFT_RIGHT_ASSIGN377_tree = self._adaptor.createWithPayload(SHIFT_RIGHT_ASSIGN377)
                            root_0 = self._adaptor.becomeRoot(SHIFT_RIGHT_ASSIGN377_tree, root_0)



                    elif alt110 == 12:
                        # Java.g:763:17: BIT_SHIFT_RIGHT_ASSIGN
                        pass 
                        BIT_SHIFT_RIGHT_ASSIGN378=self.match(self.input, BIT_SHIFT_RIGHT_ASSIGN, self.FOLLOW_BIT_SHIFT_RIGHT_ASSIGN_in_assignmentExpression10482)
                        if self._state.backtracking == 0:

                            BIT_SHIFT_RIGHT_ASSIGN378_tree = self._adaptor.createWithPayload(BIT_SHIFT_RIGHT_ASSIGN378)
                            root_0 = self._adaptor.becomeRoot(BIT_SHIFT_RIGHT_ASSIGN378_tree, root_0)




                    self._state.following.append(self.FOLLOW_assignmentExpression_in_assignmentExpression10503)
                    assignmentExpression379 = self.assignmentExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, assignmentExpression379.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 89, assignmentExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "assignmentExpression"

    class conditionalExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.conditionalExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "conditionalExpression"
    # Java.g:768:1: conditionalExpression : logicalOrExpression ( QUESTION assignmentExpression COLON conditionalExpression )? ;
    def conditionalExpression(self, ):

        retval = self.conditionalExpression_return()
        retval.start = self.input.LT(1)
        conditionalExpression_StartIndex = self.input.index()
        root_0 = None

        QUESTION381 = None
        COLON383 = None
        logicalOrExpression380 = None

        assignmentExpression382 = None

        conditionalExpression384 = None


        QUESTION381_tree = None
        COLON383_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 90):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:769:5: ( logicalOrExpression ( QUESTION assignmentExpression COLON conditionalExpression )? )
                # Java.g:769:9: logicalOrExpression ( QUESTION assignmentExpression COLON conditionalExpression )?
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_logicalOrExpression_in_conditionalExpression10524)
                logicalOrExpression380 = self.logicalOrExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, logicalOrExpression380.tree)
                # Java.g:769:29: ( QUESTION assignmentExpression COLON conditionalExpression )?
                alt112 = 2
                LA112_0 = self.input.LA(1)

                if (LA112_0 == QUESTION) :
                    alt112 = 1
                if alt112 == 1:
                    # Java.g:769:30: QUESTION assignmentExpression COLON conditionalExpression
                    pass 
                    QUESTION381=self.match(self.input, QUESTION, self.FOLLOW_QUESTION_in_conditionalExpression10527)
                    if self._state.backtracking == 0:

                        QUESTION381_tree = self._adaptor.createWithPayload(QUESTION381)
                        root_0 = self._adaptor.becomeRoot(QUESTION381_tree, root_0)

                    self._state.following.append(self.FOLLOW_assignmentExpression_in_conditionalExpression10530)
                    assignmentExpression382 = self.assignmentExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, assignmentExpression382.tree)
                    COLON383=self.match(self.input, COLON, self.FOLLOW_COLON_in_conditionalExpression10532)
                    self._state.following.append(self.FOLLOW_conditionalExpression_in_conditionalExpression10535)
                    conditionalExpression384 = self.conditionalExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, conditionalExpression384.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 90, conditionalExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "conditionalExpression"

    class logicalOrExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.logicalOrExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "logicalOrExpression"
    # Java.g:772:1: logicalOrExpression : logicalAndExpression ( LOGICAL_OR logicalAndExpression )* ;
    def logicalOrExpression(self, ):

        retval = self.logicalOrExpression_return()
        retval.start = self.input.LT(1)
        logicalOrExpression_StartIndex = self.input.index()
        root_0 = None

        LOGICAL_OR386 = None
        logicalAndExpression385 = None

        logicalAndExpression387 = None


        LOGICAL_OR386_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 91):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:773:5: ( logicalAndExpression ( LOGICAL_OR logicalAndExpression )* )
                # Java.g:773:9: logicalAndExpression ( LOGICAL_OR logicalAndExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_logicalAndExpression_in_logicalOrExpression10556)
                logicalAndExpression385 = self.logicalAndExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, logicalAndExpression385.tree)
                # Java.g:773:30: ( LOGICAL_OR logicalAndExpression )*
                while True: #loop113
                    alt113 = 2
                    LA113_0 = self.input.LA(1)

                    if (LA113_0 == LOGICAL_OR) :
                        alt113 = 1


                    if alt113 == 1:
                        # Java.g:773:31: LOGICAL_OR logicalAndExpression
                        pass 
                        LOGICAL_OR386=self.match(self.input, LOGICAL_OR, self.FOLLOW_LOGICAL_OR_in_logicalOrExpression10559)
                        if self._state.backtracking == 0:

                            LOGICAL_OR386_tree = self._adaptor.createWithPayload(LOGICAL_OR386)
                            root_0 = self._adaptor.becomeRoot(LOGICAL_OR386_tree, root_0)

                        self._state.following.append(self.FOLLOW_logicalAndExpression_in_logicalOrExpression10562)
                        logicalAndExpression387 = self.logicalAndExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, logicalAndExpression387.tree)


                    else:
                        break #loop113



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 91, logicalOrExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "logicalOrExpression"

    class logicalAndExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.logicalAndExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "logicalAndExpression"
    # Java.g:776:1: logicalAndExpression : inclusiveOrExpression ( LOGICAL_AND inclusiveOrExpression )* ;
    def logicalAndExpression(self, ):

        retval = self.logicalAndExpression_return()
        retval.start = self.input.LT(1)
        logicalAndExpression_StartIndex = self.input.index()
        root_0 = None

        LOGICAL_AND389 = None
        inclusiveOrExpression388 = None

        inclusiveOrExpression390 = None


        LOGICAL_AND389_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 92):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:777:5: ( inclusiveOrExpression ( LOGICAL_AND inclusiveOrExpression )* )
                # Java.g:777:9: inclusiveOrExpression ( LOGICAL_AND inclusiveOrExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_inclusiveOrExpression_in_logicalAndExpression10583)
                inclusiveOrExpression388 = self.inclusiveOrExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, inclusiveOrExpression388.tree)
                # Java.g:777:31: ( LOGICAL_AND inclusiveOrExpression )*
                while True: #loop114
                    alt114 = 2
                    LA114_0 = self.input.LA(1)

                    if (LA114_0 == LOGICAL_AND) :
                        alt114 = 1


                    if alt114 == 1:
                        # Java.g:777:32: LOGICAL_AND inclusiveOrExpression
                        pass 
                        LOGICAL_AND389=self.match(self.input, LOGICAL_AND, self.FOLLOW_LOGICAL_AND_in_logicalAndExpression10586)
                        if self._state.backtracking == 0:

                            LOGICAL_AND389_tree = self._adaptor.createWithPayload(LOGICAL_AND389)
                            root_0 = self._adaptor.becomeRoot(LOGICAL_AND389_tree, root_0)

                        self._state.following.append(self.FOLLOW_inclusiveOrExpression_in_logicalAndExpression10589)
                        inclusiveOrExpression390 = self.inclusiveOrExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, inclusiveOrExpression390.tree)


                    else:
                        break #loop114



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 92, logicalAndExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "logicalAndExpression"

    class inclusiveOrExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.inclusiveOrExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "inclusiveOrExpression"
    # Java.g:780:1: inclusiveOrExpression : exclusiveOrExpression ( OR exclusiveOrExpression )* ;
    def inclusiveOrExpression(self, ):

        retval = self.inclusiveOrExpression_return()
        retval.start = self.input.LT(1)
        inclusiveOrExpression_StartIndex = self.input.index()
        root_0 = None

        OR392 = None
        exclusiveOrExpression391 = None

        exclusiveOrExpression393 = None


        OR392_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 93):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:781:5: ( exclusiveOrExpression ( OR exclusiveOrExpression )* )
                # Java.g:781:9: exclusiveOrExpression ( OR exclusiveOrExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_exclusiveOrExpression_in_inclusiveOrExpression10610)
                exclusiveOrExpression391 = self.exclusiveOrExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, exclusiveOrExpression391.tree)
                # Java.g:781:31: ( OR exclusiveOrExpression )*
                while True: #loop115
                    alt115 = 2
                    LA115_0 = self.input.LA(1)

                    if (LA115_0 == OR) :
                        alt115 = 1


                    if alt115 == 1:
                        # Java.g:781:32: OR exclusiveOrExpression
                        pass 
                        OR392=self.match(self.input, OR, self.FOLLOW_OR_in_inclusiveOrExpression10613)
                        if self._state.backtracking == 0:

                            OR392_tree = self._adaptor.createWithPayload(OR392)
                            root_0 = self._adaptor.becomeRoot(OR392_tree, root_0)

                        self._state.following.append(self.FOLLOW_exclusiveOrExpression_in_inclusiveOrExpression10616)
                        exclusiveOrExpression393 = self.exclusiveOrExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, exclusiveOrExpression393.tree)


                    else:
                        break #loop115



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 93, inclusiveOrExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "inclusiveOrExpression"

    class exclusiveOrExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.exclusiveOrExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "exclusiveOrExpression"
    # Java.g:784:1: exclusiveOrExpression : andExpression ( XOR andExpression )* ;
    def exclusiveOrExpression(self, ):

        retval = self.exclusiveOrExpression_return()
        retval.start = self.input.LT(1)
        exclusiveOrExpression_StartIndex = self.input.index()
        root_0 = None

        XOR395 = None
        andExpression394 = None

        andExpression396 = None


        XOR395_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 94):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:785:5: ( andExpression ( XOR andExpression )* )
                # Java.g:785:9: andExpression ( XOR andExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_andExpression_in_exclusiveOrExpression10637)
                andExpression394 = self.andExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, andExpression394.tree)
                # Java.g:785:23: ( XOR andExpression )*
                while True: #loop116
                    alt116 = 2
                    LA116_0 = self.input.LA(1)

                    if (LA116_0 == XOR) :
                        alt116 = 1


                    if alt116 == 1:
                        # Java.g:785:24: XOR andExpression
                        pass 
                        XOR395=self.match(self.input, XOR, self.FOLLOW_XOR_in_exclusiveOrExpression10640)
                        if self._state.backtracking == 0:

                            XOR395_tree = self._adaptor.createWithPayload(XOR395)
                            root_0 = self._adaptor.becomeRoot(XOR395_tree, root_0)

                        self._state.following.append(self.FOLLOW_andExpression_in_exclusiveOrExpression10643)
                        andExpression396 = self.andExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, andExpression396.tree)


                    else:
                        break #loop116



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 94, exclusiveOrExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "exclusiveOrExpression"

    class andExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.andExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "andExpression"
    # Java.g:788:1: andExpression : equalityExpression ( AND equalityExpression )* ;
    def andExpression(self, ):

        retval = self.andExpression_return()
        retval.start = self.input.LT(1)
        andExpression_StartIndex = self.input.index()
        root_0 = None

        AND398 = None
        equalityExpression397 = None

        equalityExpression399 = None


        AND398_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 95):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:789:5: ( equalityExpression ( AND equalityExpression )* )
                # Java.g:789:9: equalityExpression ( AND equalityExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_equalityExpression_in_andExpression10664)
                equalityExpression397 = self.equalityExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, equalityExpression397.tree)
                # Java.g:789:28: ( AND equalityExpression )*
                while True: #loop117
                    alt117 = 2
                    LA117_0 = self.input.LA(1)

                    if (LA117_0 == AND) :
                        alt117 = 1


                    if alt117 == 1:
                        # Java.g:789:29: AND equalityExpression
                        pass 
                        AND398=self.match(self.input, AND, self.FOLLOW_AND_in_andExpression10667)
                        if self._state.backtracking == 0:

                            AND398_tree = self._adaptor.createWithPayload(AND398)
                            root_0 = self._adaptor.becomeRoot(AND398_tree, root_0)

                        self._state.following.append(self.FOLLOW_equalityExpression_in_andExpression10670)
                        equalityExpression399 = self.equalityExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, equalityExpression399.tree)


                    else:
                        break #loop117



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 95, andExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "andExpression"

    class equalityExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.equalityExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "equalityExpression"
    # Java.g:792:1: equalityExpression : instanceOfExpression ( ( EQUAL | NOT_EQUAL ) instanceOfExpression )* ;
    def equalityExpression(self, ):

        retval = self.equalityExpression_return()
        retval.start = self.input.LT(1)
        equalityExpression_StartIndex = self.input.index()
        root_0 = None

        EQUAL401 = None
        NOT_EQUAL402 = None
        instanceOfExpression400 = None

        instanceOfExpression403 = None


        EQUAL401_tree = None
        NOT_EQUAL402_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 96):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:793:5: ( instanceOfExpression ( ( EQUAL | NOT_EQUAL ) instanceOfExpression )* )
                # Java.g:793:9: instanceOfExpression ( ( EQUAL | NOT_EQUAL ) instanceOfExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_instanceOfExpression_in_equalityExpression10691)
                instanceOfExpression400 = self.instanceOfExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, instanceOfExpression400.tree)
                # Java.g:794:9: ( ( EQUAL | NOT_EQUAL ) instanceOfExpression )*
                while True: #loop119
                    alt119 = 2
                    LA119_0 = self.input.LA(1)

                    if (LA119_0 == EQUAL or LA119_0 == NOT_EQUAL) :
                        alt119 = 1


                    if alt119 == 1:
                        # Java.g:794:13: ( EQUAL | NOT_EQUAL ) instanceOfExpression
                        pass 
                        # Java.g:794:13: ( EQUAL | NOT_EQUAL )
                        alt118 = 2
                        LA118_0 = self.input.LA(1)

                        if (LA118_0 == EQUAL) :
                            alt118 = 1
                        elif (LA118_0 == NOT_EQUAL) :
                            alt118 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 118, 0, self.input)

                            raise nvae

                        if alt118 == 1:
                            # Java.g:794:17: EQUAL
                            pass 
                            EQUAL401=self.match(self.input, EQUAL, self.FOLLOW_EQUAL_in_equalityExpression10709)
                            if self._state.backtracking == 0:

                                EQUAL401_tree = self._adaptor.createWithPayload(EQUAL401)
                                root_0 = self._adaptor.becomeRoot(EQUAL401_tree, root_0)



                        elif alt118 == 2:
                            # Java.g:795:17: NOT_EQUAL
                            pass 
                            NOT_EQUAL402=self.match(self.input, NOT_EQUAL, self.FOLLOW_NOT_EQUAL_in_equalityExpression10728)
                            if self._state.backtracking == 0:

                                NOT_EQUAL402_tree = self._adaptor.createWithPayload(NOT_EQUAL402)
                                root_0 = self._adaptor.becomeRoot(NOT_EQUAL402_tree, root_0)




                        self._state.following.append(self.FOLLOW_instanceOfExpression_in_equalityExpression10757)
                        instanceOfExpression403 = self.instanceOfExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, instanceOfExpression403.tree)


                    else:
                        break #loop119



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 96, equalityExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "equalityExpression"

    class instanceOfExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.instanceOfExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "instanceOfExpression"
    # Java.g:801:1: instanceOfExpression : relationalExpression ( INSTANCEOF type )? ;
    def instanceOfExpression(self, ):

        retval = self.instanceOfExpression_return()
        retval.start = self.input.LT(1)
        instanceOfExpression_StartIndex = self.input.index()
        root_0 = None

        INSTANCEOF405 = None
        relationalExpression404 = None

        type406 = None


        INSTANCEOF405_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 97):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:802:5: ( relationalExpression ( INSTANCEOF type )? )
                # Java.g:802:9: relationalExpression ( INSTANCEOF type )?
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_relationalExpression_in_instanceOfExpression10787)
                relationalExpression404 = self.relationalExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, relationalExpression404.tree)
                # Java.g:802:30: ( INSTANCEOF type )?
                alt120 = 2
                LA120_0 = self.input.LA(1)

                if (LA120_0 == INSTANCEOF) :
                    alt120 = 1
                if alt120 == 1:
                    # Java.g:802:31: INSTANCEOF type
                    pass 
                    INSTANCEOF405=self.match(self.input, INSTANCEOF, self.FOLLOW_INSTANCEOF_in_instanceOfExpression10790)
                    if self._state.backtracking == 0:

                        INSTANCEOF405_tree = self._adaptor.createWithPayload(INSTANCEOF405)
                        root_0 = self._adaptor.becomeRoot(INSTANCEOF405_tree, root_0)

                    self._state.following.append(self.FOLLOW_type_in_instanceOfExpression10793)
                    type406 = self.type()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, type406.tree)






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 97, instanceOfExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "instanceOfExpression"

    class relationalExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.relationalExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "relationalExpression"
    # Java.g:805:1: relationalExpression : shiftExpression ( ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN ) shiftExpression )* ;
    def relationalExpression(self, ):

        retval = self.relationalExpression_return()
        retval.start = self.input.LT(1)
        relationalExpression_StartIndex = self.input.index()
        root_0 = None

        LESS_OR_EQUAL408 = None
        GREATER_OR_EQUAL409 = None
        LESS_THAN410 = None
        GREATER_THAN411 = None
        shiftExpression407 = None

        shiftExpression412 = None


        LESS_OR_EQUAL408_tree = None
        GREATER_OR_EQUAL409_tree = None
        LESS_THAN410_tree = None
        GREATER_THAN411_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 98):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:806:5: ( shiftExpression ( ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN ) shiftExpression )* )
                # Java.g:806:9: shiftExpression ( ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN ) shiftExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_shiftExpression_in_relationalExpression10814)
                shiftExpression407 = self.shiftExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, shiftExpression407.tree)
                # Java.g:807:9: ( ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN ) shiftExpression )*
                while True: #loop122
                    alt122 = 2
                    LA122_0 = self.input.LA(1)

                    if ((GREATER_OR_EQUAL <= LA122_0 <= GREATER_THAN) or (LESS_OR_EQUAL <= LA122_0 <= LESS_THAN)) :
                        alt122 = 1


                    if alt122 == 1:
                        # Java.g:807:13: ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN ) shiftExpression
                        pass 
                        # Java.g:807:13: ( LESS_OR_EQUAL | GREATER_OR_EQUAL | LESS_THAN | GREATER_THAN )
                        alt121 = 4
                        LA121 = self.input.LA(1)
                        if LA121 == LESS_OR_EQUAL:
                            alt121 = 1
                        elif LA121 == GREATER_OR_EQUAL:
                            alt121 = 2
                        elif LA121 == LESS_THAN:
                            alt121 = 3
                        elif LA121 == GREATER_THAN:
                            alt121 = 4
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 121, 0, self.input)

                            raise nvae

                        if alt121 == 1:
                            # Java.g:807:17: LESS_OR_EQUAL
                            pass 
                            LESS_OR_EQUAL408=self.match(self.input, LESS_OR_EQUAL, self.FOLLOW_LESS_OR_EQUAL_in_relationalExpression10832)
                            if self._state.backtracking == 0:

                                LESS_OR_EQUAL408_tree = self._adaptor.createWithPayload(LESS_OR_EQUAL408)
                                root_0 = self._adaptor.becomeRoot(LESS_OR_EQUAL408_tree, root_0)



                        elif alt121 == 2:
                            # Java.g:808:17: GREATER_OR_EQUAL
                            pass 
                            GREATER_OR_EQUAL409=self.match(self.input, GREATER_OR_EQUAL, self.FOLLOW_GREATER_OR_EQUAL_in_relationalExpression10851)
                            if self._state.backtracking == 0:

                                GREATER_OR_EQUAL409_tree = self._adaptor.createWithPayload(GREATER_OR_EQUAL409)
                                root_0 = self._adaptor.becomeRoot(GREATER_OR_EQUAL409_tree, root_0)



                        elif alt121 == 3:
                            # Java.g:809:17: LESS_THAN
                            pass 
                            LESS_THAN410=self.match(self.input, LESS_THAN, self.FOLLOW_LESS_THAN_in_relationalExpression10870)
                            if self._state.backtracking == 0:

                                LESS_THAN410_tree = self._adaptor.createWithPayload(LESS_THAN410)
                                root_0 = self._adaptor.becomeRoot(LESS_THAN410_tree, root_0)



                        elif alt121 == 4:
                            # Java.g:810:17: GREATER_THAN
                            pass 
                            GREATER_THAN411=self.match(self.input, GREATER_THAN, self.FOLLOW_GREATER_THAN_in_relationalExpression10889)
                            if self._state.backtracking == 0:

                                GREATER_THAN411_tree = self._adaptor.createWithPayload(GREATER_THAN411)
                                root_0 = self._adaptor.becomeRoot(GREATER_THAN411_tree, root_0)




                        self._state.following.append(self.FOLLOW_shiftExpression_in_relationalExpression10918)
                        shiftExpression412 = self.shiftExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, shiftExpression412.tree)


                    else:
                        break #loop122



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 98, relationalExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "relationalExpression"

    class shiftExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.shiftExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "shiftExpression"
    # Java.g:816:1: shiftExpression : additiveExpression ( ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT ) additiveExpression )* ;
    def shiftExpression(self, ):

        retval = self.shiftExpression_return()
        retval.start = self.input.LT(1)
        shiftExpression_StartIndex = self.input.index()
        root_0 = None

        BIT_SHIFT_RIGHT414 = None
        SHIFT_RIGHT415 = None
        SHIFT_LEFT416 = None
        additiveExpression413 = None

        additiveExpression417 = None


        BIT_SHIFT_RIGHT414_tree = None
        SHIFT_RIGHT415_tree = None
        SHIFT_LEFT416_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 99):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:817:5: ( additiveExpression ( ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT ) additiveExpression )* )
                # Java.g:817:9: additiveExpression ( ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT ) additiveExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_additiveExpression_in_shiftExpression10948)
                additiveExpression413 = self.additiveExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, additiveExpression413.tree)
                # Java.g:818:9: ( ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT ) additiveExpression )*
                while True: #loop124
                    alt124 = 2
                    LA124_0 = self.input.LA(1)

                    if (LA124_0 == BIT_SHIFT_RIGHT or LA124_0 == SHIFT_LEFT or LA124_0 == SHIFT_RIGHT) :
                        alt124 = 1


                    if alt124 == 1:
                        # Java.g:818:13: ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT ) additiveExpression
                        pass 
                        # Java.g:818:13: ( BIT_SHIFT_RIGHT | SHIFT_RIGHT | SHIFT_LEFT )
                        alt123 = 3
                        LA123 = self.input.LA(1)
                        if LA123 == BIT_SHIFT_RIGHT:
                            alt123 = 1
                        elif LA123 == SHIFT_RIGHT:
                            alt123 = 2
                        elif LA123 == SHIFT_LEFT:
                            alt123 = 3
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 123, 0, self.input)

                            raise nvae

                        if alt123 == 1:
                            # Java.g:818:17: BIT_SHIFT_RIGHT
                            pass 
                            BIT_SHIFT_RIGHT414=self.match(self.input, BIT_SHIFT_RIGHT, self.FOLLOW_BIT_SHIFT_RIGHT_in_shiftExpression10966)
                            if self._state.backtracking == 0:

                                BIT_SHIFT_RIGHT414_tree = self._adaptor.createWithPayload(BIT_SHIFT_RIGHT414)
                                root_0 = self._adaptor.becomeRoot(BIT_SHIFT_RIGHT414_tree, root_0)



                        elif alt123 == 2:
                            # Java.g:819:17: SHIFT_RIGHT
                            pass 
                            SHIFT_RIGHT415=self.match(self.input, SHIFT_RIGHT, self.FOLLOW_SHIFT_RIGHT_in_shiftExpression10985)
                            if self._state.backtracking == 0:

                                SHIFT_RIGHT415_tree = self._adaptor.createWithPayload(SHIFT_RIGHT415)
                                root_0 = self._adaptor.becomeRoot(SHIFT_RIGHT415_tree, root_0)



                        elif alt123 == 3:
                            # Java.g:820:17: SHIFT_LEFT
                            pass 
                            SHIFT_LEFT416=self.match(self.input, SHIFT_LEFT, self.FOLLOW_SHIFT_LEFT_in_shiftExpression11004)
                            if self._state.backtracking == 0:

                                SHIFT_LEFT416_tree = self._adaptor.createWithPayload(SHIFT_LEFT416)
                                root_0 = self._adaptor.becomeRoot(SHIFT_LEFT416_tree, root_0)




                        self._state.following.append(self.FOLLOW_additiveExpression_in_shiftExpression11033)
                        additiveExpression417 = self.additiveExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, additiveExpression417.tree)


                    else:
                        break #loop124



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 99, shiftExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "shiftExpression"

    class additiveExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.additiveExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "additiveExpression"
    # Java.g:826:1: additiveExpression : multiplicativeExpression ( ( PLUS | MINUS ) multiplicativeExpression )* ;
    def additiveExpression(self, ):

        retval = self.additiveExpression_return()
        retval.start = self.input.LT(1)
        additiveExpression_StartIndex = self.input.index()
        root_0 = None

        PLUS419 = None
        MINUS420 = None
        multiplicativeExpression418 = None

        multiplicativeExpression421 = None


        PLUS419_tree = None
        MINUS420_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 100):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:827:5: ( multiplicativeExpression ( ( PLUS | MINUS ) multiplicativeExpression )* )
                # Java.g:827:9: multiplicativeExpression ( ( PLUS | MINUS ) multiplicativeExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_multiplicativeExpression_in_additiveExpression11063)
                multiplicativeExpression418 = self.multiplicativeExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, multiplicativeExpression418.tree)
                # Java.g:828:9: ( ( PLUS | MINUS ) multiplicativeExpression )*
                while True: #loop126
                    alt126 = 2
                    LA126_0 = self.input.LA(1)

                    if (LA126_0 == MINUS or LA126_0 == PLUS) :
                        alt126 = 1


                    if alt126 == 1:
                        # Java.g:828:13: ( PLUS | MINUS ) multiplicativeExpression
                        pass 
                        # Java.g:828:13: ( PLUS | MINUS )
                        alt125 = 2
                        LA125_0 = self.input.LA(1)

                        if (LA125_0 == PLUS) :
                            alt125 = 1
                        elif (LA125_0 == MINUS) :
                            alt125 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 125, 0, self.input)

                            raise nvae

                        if alt125 == 1:
                            # Java.g:828:17: PLUS
                            pass 
                            PLUS419=self.match(self.input, PLUS, self.FOLLOW_PLUS_in_additiveExpression11081)
                            if self._state.backtracking == 0:

                                PLUS419_tree = self._adaptor.createWithPayload(PLUS419)
                                root_0 = self._adaptor.becomeRoot(PLUS419_tree, root_0)



                        elif alt125 == 2:
                            # Java.g:829:17: MINUS
                            pass 
                            MINUS420=self.match(self.input, MINUS, self.FOLLOW_MINUS_in_additiveExpression11100)
                            if self._state.backtracking == 0:

                                MINUS420_tree = self._adaptor.createWithPayload(MINUS420)
                                root_0 = self._adaptor.becomeRoot(MINUS420_tree, root_0)




                        self._state.following.append(self.FOLLOW_multiplicativeExpression_in_additiveExpression11129)
                        multiplicativeExpression421 = self.multiplicativeExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, multiplicativeExpression421.tree)


                    else:
                        break #loop126



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 100, additiveExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "additiveExpression"

    class multiplicativeExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.multiplicativeExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "multiplicativeExpression"
    # Java.g:835:1: multiplicativeExpression : unaryExpression ( ( STAR | DIV | MOD ) unaryExpression )* ;
    def multiplicativeExpression(self, ):

        retval = self.multiplicativeExpression_return()
        retval.start = self.input.LT(1)
        multiplicativeExpression_StartIndex = self.input.index()
        root_0 = None

        STAR423 = None
        DIV424 = None
        MOD425 = None
        unaryExpression422 = None

        unaryExpression426 = None


        STAR423_tree = None
        DIV424_tree = None
        MOD425_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 101):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:836:5: ( unaryExpression ( ( STAR | DIV | MOD ) unaryExpression )* )
                # Java.g:836:9: unaryExpression ( ( STAR | DIV | MOD ) unaryExpression )*
                pass 
                root_0 = self._adaptor.nil()

                self._state.following.append(self.FOLLOW_unaryExpression_in_multiplicativeExpression11159)
                unaryExpression422 = self.unaryExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    self._adaptor.addChild(root_0, unaryExpression422.tree)
                # Java.g:837:9: ( ( STAR | DIV | MOD ) unaryExpression )*
                while True: #loop128
                    alt128 = 2
                    LA128_0 = self.input.LA(1)

                    if (LA128_0 == DIV or LA128_0 == MOD or LA128_0 == STAR) :
                        alt128 = 1


                    if alt128 == 1:
                        # Java.g:837:13: ( STAR | DIV | MOD ) unaryExpression
                        pass 
                        # Java.g:837:13: ( STAR | DIV | MOD )
                        alt127 = 3
                        LA127 = self.input.LA(1)
                        if LA127 == STAR:
                            alt127 = 1
                        elif LA127 == DIV:
                            alt127 = 2
                        elif LA127 == MOD:
                            alt127 = 3
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 127, 0, self.input)

                            raise nvae

                        if alt127 == 1:
                            # Java.g:837:17: STAR
                            pass 
                            STAR423=self.match(self.input, STAR, self.FOLLOW_STAR_in_multiplicativeExpression11177)
                            if self._state.backtracking == 0:

                                STAR423_tree = self._adaptor.createWithPayload(STAR423)
                                root_0 = self._adaptor.becomeRoot(STAR423_tree, root_0)



                        elif alt127 == 2:
                            # Java.g:838:17: DIV
                            pass 
                            DIV424=self.match(self.input, DIV, self.FOLLOW_DIV_in_multiplicativeExpression11196)
                            if self._state.backtracking == 0:

                                DIV424_tree = self._adaptor.createWithPayload(DIV424)
                                root_0 = self._adaptor.becomeRoot(DIV424_tree, root_0)



                        elif alt127 == 3:
                            # Java.g:839:17: MOD
                            pass 
                            MOD425=self.match(self.input, MOD, self.FOLLOW_MOD_in_multiplicativeExpression11215)
                            if self._state.backtracking == 0:

                                MOD425_tree = self._adaptor.createWithPayload(MOD425)
                                root_0 = self._adaptor.becomeRoot(MOD425_tree, root_0)




                        self._state.following.append(self.FOLLOW_unaryExpression_in_multiplicativeExpression11244)
                        unaryExpression426 = self.unaryExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, unaryExpression426.tree)


                    else:
                        break #loop128



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 101, multiplicativeExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "multiplicativeExpression"

    class unaryExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.unaryExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "unaryExpression"
    # Java.g:845:1: unaryExpression : ( PLUS unaryExpression -> ^( UNARY_PLUS[$PLUS, \"UNARY_PLUS\"] unaryExpression ) | MINUS unaryExpression -> ^( UNARY_MINUS[$MINUS, \"UNARY_MINUS\"] unaryExpression ) | INC postfixedExpression -> ^( PRE_INC[$INC, \"PRE_INC\"] postfixedExpression ) | DEC postfixedExpression -> ^( PRE_DEC[$DEC, \"PRE_DEC\"] postfixedExpression ) | unaryExpressionNotPlusMinus );
    def unaryExpression(self, ):

        retval = self.unaryExpression_return()
        retval.start = self.input.LT(1)
        unaryExpression_StartIndex = self.input.index()
        root_0 = None

        PLUS427 = None
        MINUS429 = None
        INC431 = None
        DEC433 = None
        unaryExpression428 = None

        unaryExpression430 = None

        postfixedExpression432 = None

        postfixedExpression434 = None

        unaryExpressionNotPlusMinus435 = None


        PLUS427_tree = None
        MINUS429_tree = None
        INC431_tree = None
        DEC433_tree = None
        stream_DEC = RewriteRuleTokenStream(self._adaptor, "token DEC")
        stream_INC = RewriteRuleTokenStream(self._adaptor, "token INC")
        stream_PLUS = RewriteRuleTokenStream(self._adaptor, "token PLUS")
        stream_MINUS = RewriteRuleTokenStream(self._adaptor, "token MINUS")
        stream_postfixedExpression = RewriteRuleSubtreeStream(self._adaptor, "rule postfixedExpression")
        stream_unaryExpression = RewriteRuleSubtreeStream(self._adaptor, "rule unaryExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 102):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:846:5: ( PLUS unaryExpression -> ^( UNARY_PLUS[$PLUS, \"UNARY_PLUS\"] unaryExpression ) | MINUS unaryExpression -> ^( UNARY_MINUS[$MINUS, \"UNARY_MINUS\"] unaryExpression ) | INC postfixedExpression -> ^( PRE_INC[$INC, \"PRE_INC\"] postfixedExpression ) | DEC postfixedExpression -> ^( PRE_DEC[$DEC, \"PRE_DEC\"] postfixedExpression ) | unaryExpressionNotPlusMinus )
                alt129 = 5
                LA129 = self.input.LA(1)
                if LA129 == PLUS:
                    alt129 = 1
                elif LA129 == MINUS:
                    alt129 = 2
                elif LA129 == INC:
                    alt129 = 3
                elif LA129 == DEC:
                    alt129 = 4
                elif LA129 == LESS_THAN or LA129 == LOGICAL_NOT or LA129 == LPAREN or LA129 == NOT or LA129 == BOOLEAN or LA129 == BYTE or LA129 == CHAR or LA129 == DOUBLE or LA129 == FALSE or LA129 == FLOAT or LA129 == INT or LA129 == LONG or LA129 == NEW or LA129 == NULL or LA129 == SHORT or LA129 == SUPER or LA129 == THIS or LA129 == TRUE or LA129 == VOID or LA129 == IDENT or LA129 == HEX_LITERAL or LA129 == OCTAL_LITERAL or LA129 == DECIMAL_LITERAL or LA129 == FLOATING_POINT_LITERAL or LA129 == CHARACTER_LITERAL or LA129 == STRING_LITERAL:
                    alt129 = 5
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 129, 0, self.input)

                    raise nvae

                if alt129 == 1:
                    # Java.g:846:9: PLUS unaryExpression
                    pass 
                    PLUS427=self.match(self.input, PLUS, self.FOLLOW_PLUS_in_unaryExpression11274) 
                    if self._state.backtracking == 0:
                        stream_PLUS.add(PLUS427)
                    self._state.following.append(self.FOLLOW_unaryExpression_in_unaryExpression11276)
                    unaryExpression428 = self.unaryExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_unaryExpression.add(unaryExpression428.tree)

                    # AST Rewrite
                    # elements: unaryExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 846:37: -> ^( UNARY_PLUS[$PLUS, \"UNARY_PLUS\"] unaryExpression )
                        # Java.g:846:41: ^( UNARY_PLUS[$PLUS, \"UNARY_PLUS\"] unaryExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(UNARY_PLUS, PLUS427, "UNARY_PLUS"), root_1)

                        self._adaptor.addChild(root_1, stream_unaryExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt129 == 2:
                    # Java.g:847:9: MINUS unaryExpression
                    pass 
                    MINUS429=self.match(self.input, MINUS, self.FOLLOW_MINUS_in_unaryExpression11303) 
                    if self._state.backtracking == 0:
                        stream_MINUS.add(MINUS429)
                    self._state.following.append(self.FOLLOW_unaryExpression_in_unaryExpression11305)
                    unaryExpression430 = self.unaryExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_unaryExpression.add(unaryExpression430.tree)

                    # AST Rewrite
                    # elements: unaryExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 847:37: -> ^( UNARY_MINUS[$MINUS, \"UNARY_MINUS\"] unaryExpression )
                        # Java.g:847:41: ^( UNARY_MINUS[$MINUS, \"UNARY_MINUS\"] unaryExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(UNARY_MINUS, MINUS429, "UNARY_MINUS"), root_1)

                        self._adaptor.addChild(root_1, stream_unaryExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt129 == 3:
                    # Java.g:848:9: INC postfixedExpression
                    pass 
                    INC431=self.match(self.input, INC, self.FOLLOW_INC_in_unaryExpression11331) 
                    if self._state.backtracking == 0:
                        stream_INC.add(INC431)
                    self._state.following.append(self.FOLLOW_postfixedExpression_in_unaryExpression11333)
                    postfixedExpression432 = self.postfixedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_postfixedExpression.add(postfixedExpression432.tree)

                    # AST Rewrite
                    # elements: postfixedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 848:37: -> ^( PRE_INC[$INC, \"PRE_INC\"] postfixedExpression )
                        # Java.g:848:41: ^( PRE_INC[$INC, \"PRE_INC\"] postfixedExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(PRE_INC, INC431, "PRE_INC"), root_1)

                        self._adaptor.addChild(root_1, stream_postfixedExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt129 == 4:
                    # Java.g:849:9: DEC postfixedExpression
                    pass 
                    DEC433=self.match(self.input, DEC, self.FOLLOW_DEC_in_unaryExpression11357) 
                    if self._state.backtracking == 0:
                        stream_DEC.add(DEC433)
                    self._state.following.append(self.FOLLOW_postfixedExpression_in_unaryExpression11359)
                    postfixedExpression434 = self.postfixedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_postfixedExpression.add(postfixedExpression434.tree)

                    # AST Rewrite
                    # elements: postfixedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 849:37: -> ^( PRE_DEC[$DEC, \"PRE_DEC\"] postfixedExpression )
                        # Java.g:849:41: ^( PRE_DEC[$DEC, \"PRE_DEC\"] postfixedExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(PRE_DEC, DEC433, "PRE_DEC"), root_1)

                        self._adaptor.addChild(root_1, stream_postfixedExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt129 == 5:
                    # Java.g:850:9: unaryExpressionNotPlusMinus
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_unaryExpressionNotPlusMinus_in_unaryExpression11383)
                    unaryExpressionNotPlusMinus435 = self.unaryExpressionNotPlusMinus()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, unaryExpressionNotPlusMinus435.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 102, unaryExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "unaryExpression"

    class unaryExpressionNotPlusMinus_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.unaryExpressionNotPlusMinus_return, self).__init__()

            self.tree = None




    # $ANTLR start "unaryExpressionNotPlusMinus"
    # Java.g:853:1: unaryExpressionNotPlusMinus : ( NOT unaryExpression -> ^( NOT unaryExpression ) | LOGICAL_NOT unaryExpression -> ^( LOGICAL_NOT unaryExpression ) | LPAREN type RPAREN unaryExpression -> ^( CAST_EXPR[$LPAREN, \"CAST_EXPR\"] type unaryExpression ) | postfixedExpression );
    def unaryExpressionNotPlusMinus(self, ):

        retval = self.unaryExpressionNotPlusMinus_return()
        retval.start = self.input.LT(1)
        unaryExpressionNotPlusMinus_StartIndex = self.input.index()
        root_0 = None

        NOT436 = None
        LOGICAL_NOT438 = None
        LPAREN440 = None
        RPAREN442 = None
        unaryExpression437 = None

        unaryExpression439 = None

        type441 = None

        unaryExpression443 = None

        postfixedExpression444 = None


        NOT436_tree = None
        LOGICAL_NOT438_tree = None
        LPAREN440_tree = None
        RPAREN442_tree = None
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_LOGICAL_NOT = RewriteRuleTokenStream(self._adaptor, "token LOGICAL_NOT")
        stream_NOT = RewriteRuleTokenStream(self._adaptor, "token NOT")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_unaryExpression = RewriteRuleSubtreeStream(self._adaptor, "rule unaryExpression")
        stream_type = RewriteRuleSubtreeStream(self._adaptor, "rule type")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 103):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:854:5: ( NOT unaryExpression -> ^( NOT unaryExpression ) | LOGICAL_NOT unaryExpression -> ^( LOGICAL_NOT unaryExpression ) | LPAREN type RPAREN unaryExpression -> ^( CAST_EXPR[$LPAREN, \"CAST_EXPR\"] type unaryExpression ) | postfixedExpression )
                alt130 = 4
                alt130 = self.dfa130.predict(self.input)
                if alt130 == 1:
                    # Java.g:854:9: NOT unaryExpression
                    pass 
                    NOT436=self.match(self.input, NOT, self.FOLLOW_NOT_in_unaryExpressionNotPlusMinus11402) 
                    if self._state.backtracking == 0:
                        stream_NOT.add(NOT436)
                    self._state.following.append(self.FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11404)
                    unaryExpression437 = self.unaryExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_unaryExpression.add(unaryExpression437.tree)

                    # AST Rewrite
                    # elements: unaryExpression, NOT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 854:57: -> ^( NOT unaryExpression )
                        # Java.g:854:61: ^( NOT unaryExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_NOT.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_unaryExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt130 == 2:
                    # Java.g:855:9: LOGICAL_NOT unaryExpression
                    pass 
                    LOGICAL_NOT438=self.match(self.input, LOGICAL_NOT, self.FOLLOW_LOGICAL_NOT_in_unaryExpressionNotPlusMinus11451) 
                    if self._state.backtracking == 0:
                        stream_LOGICAL_NOT.add(LOGICAL_NOT438)
                    self._state.following.append(self.FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11453)
                    unaryExpression439 = self.unaryExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_unaryExpression.add(unaryExpression439.tree)

                    # AST Rewrite
                    # elements: unaryExpression, LOGICAL_NOT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 855:57: -> ^( LOGICAL_NOT unaryExpression )
                        # Java.g:855:61: ^( LOGICAL_NOT unaryExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_LOGICAL_NOT.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_unaryExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt130 == 3:
                    # Java.g:856:9: LPAREN type RPAREN unaryExpression
                    pass 
                    LPAREN440=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_unaryExpressionNotPlusMinus11492) 
                    if self._state.backtracking == 0:
                        stream_LPAREN.add(LPAREN440)
                    self._state.following.append(self.FOLLOW_type_in_unaryExpressionNotPlusMinus11494)
                    type441 = self.type()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_type.add(type441.tree)
                    RPAREN442=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_unaryExpressionNotPlusMinus11496) 
                    if self._state.backtracking == 0:
                        stream_RPAREN.add(RPAREN442)
                    self._state.following.append(self.FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11498)
                    unaryExpression443 = self.unaryExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_unaryExpression.add(unaryExpression443.tree)

                    # AST Rewrite
                    # elements: type, unaryExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 856:57: -> ^( CAST_EXPR[$LPAREN, \"CAST_EXPR\"] type unaryExpression )
                        # Java.g:856:61: ^( CAST_EXPR[$LPAREN, \"CAST_EXPR\"] type unaryExpression )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(CAST_EXPR, LPAREN440, "CAST_EXPR"), root_1)

                        self._adaptor.addChild(root_1, stream_type.nextTree())
                        self._adaptor.addChild(root_1, stream_unaryExpression.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt130 == 4:
                    # Java.g:857:9: postfixedExpression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_postfixedExpression_in_unaryExpressionNotPlusMinus11533)
                    postfixedExpression444 = self.postfixedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, postfixedExpression444.tree)


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 103, unaryExpressionNotPlusMinus_StartIndex, success)

            pass
        return retval

    # $ANTLR end "unaryExpressionNotPlusMinus"

    class postfixedExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.postfixedExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "postfixedExpression"
    # Java.g:860:1: postfixedExpression : ( primaryExpression -> primaryExpression ) (outerDot= DOT ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) ) | LBRACK expression RBRACK -> ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression ) )* ( INC -> ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression) | DEC -> ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression) )? ;
    def postfixedExpression(self, ):

        retval = self.postfixedExpression_return()
        retval.start = self.input.LT(1)
        postfixedExpression_StartIndex = self.input.index()
        root_0 = None

        outerDot = None
        Super = None
        innerDot = None
        IDENT447 = None
        THIS449 = None
        SUPER451 = None
        IDENT452 = None
        LBRACK455 = None
        RBRACK457 = None
        INC458 = None
        DEC459 = None
        primaryExpression445 = None

        genericTypeArgumentListSimplified446 = None

        arguments448 = None

        arguments450 = None

        arguments453 = None

        innerNewExpression454 = None

        expression456 = None


        outerDot_tree = None
        Super_tree = None
        innerDot_tree = None
        IDENT447_tree = None
        THIS449_tree = None
        SUPER451_tree = None
        IDENT452_tree = None
        LBRACK455_tree = None
        RBRACK457_tree = None
        INC458_tree = None
        DEC459_tree = None
        stream_RBRACK = RewriteRuleTokenStream(self._adaptor, "token RBRACK")
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_INC = RewriteRuleTokenStream(self._adaptor, "token INC")
        stream_DEC = RewriteRuleTokenStream(self._adaptor, "token DEC")
        stream_LBRACK = RewriteRuleTokenStream(self._adaptor, "token LBRACK")
        stream_SUPER = RewriteRuleTokenStream(self._adaptor, "token SUPER")
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")
        stream_THIS = RewriteRuleTokenStream(self._adaptor, "token THIS")
        stream_expression = RewriteRuleSubtreeStream(self._adaptor, "rule expression")
        stream_arguments = RewriteRuleSubtreeStream(self._adaptor, "rule arguments")
        stream_primaryExpression = RewriteRuleSubtreeStream(self._adaptor, "rule primaryExpression")
        stream_genericTypeArgumentListSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentListSimplified")
        stream_innerNewExpression = RewriteRuleSubtreeStream(self._adaptor, "rule innerNewExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 104):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:862:5: ( ( primaryExpression -> primaryExpression ) (outerDot= DOT ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) ) | LBRACK expression RBRACK -> ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression ) )* ( INC -> ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression) | DEC -> ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression) )? )
                # Java.g:862:9: ( primaryExpression -> primaryExpression ) (outerDot= DOT ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) ) | LBRACK expression RBRACK -> ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression ) )* ( INC -> ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression) | DEC -> ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression) )?
                pass 
                # Java.g:862:9: ( primaryExpression -> primaryExpression )
                # Java.g:862:13: primaryExpression
                pass 
                self._state.following.append(self.FOLLOW_primaryExpression_in_postfixedExpression11565)
                primaryExpression445 = self.primaryExpression()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_primaryExpression.add(primaryExpression445.tree)

                # AST Rewrite
                # elements: primaryExpression
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 862:53: -> primaryExpression
                    self._adaptor.addChild(root_0, stream_primaryExpression.nextTree())



                    retval.tree = root_0



                # Java.g:865:9: (outerDot= DOT ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) ) | LBRACK expression RBRACK -> ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression ) )*
                while True: #loop135
                    alt135 = 3
                    LA135_0 = self.input.LA(1)

                    if (LA135_0 == DOT) :
                        alt135 = 1
                    elif (LA135_0 == LBRACK) :
                        alt135 = 2


                    if alt135 == 1:
                        # Java.g:865:13: outerDot= DOT ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) )
                        pass 
                        outerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_postfixedExpression11627) 
                        if self._state.backtracking == 0:
                            stream_DOT.add(outerDot)
                        # Java.g:866:13: ( ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )? | THIS -> ^( DOT $postfixedExpression THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments ) | ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )? | innerNewExpression -> ^( DOT $postfixedExpression innerNewExpression ) )
                        alt134 = 5
                        LA134 = self.input.LA(1)
                        if LA134 == LESS_THAN or LA134 == IDENT:
                            alt134 = 1
                        elif LA134 == THIS:
                            alt134 = 2
                        elif LA134 == SUPER:
                            LA134_3 = self.input.LA(2)

                            if (LA134_3 == DOT) :
                                alt134 = 4
                            elif (LA134_3 == LPAREN) :
                                alt134 = 3
                            else:
                                if self._state.backtracking > 0:
                                    raise BacktrackingFailed

                                nvae = NoViableAltException("", 134, 3, self.input)

                                raise nvae

                        elif LA134 == NEW:
                            alt134 = 5
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 134, 0, self.input)

                            raise nvae

                        if alt134 == 1:
                            # Java.g:866:17: ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )?
                            pass 
                            # Java.g:866:17: ( ( genericTypeArgumentListSimplified )? IDENT -> ^( DOT $postfixedExpression IDENT ) )
                            # Java.g:866:21: ( genericTypeArgumentListSimplified )? IDENT
                            pass 
                            # Java.g:866:21: ( genericTypeArgumentListSimplified )?
                            alt131 = 2
                            LA131_0 = self.input.LA(1)

                            if (LA131_0 == LESS_THAN) :
                                alt131 = 1
                            if alt131 == 1:
                                # Java.g:0:0: genericTypeArgumentListSimplified
                                pass 
                                self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_postfixedExpression11649)
                                genericTypeArgumentListSimplified446 = self.genericTypeArgumentListSimplified()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_genericTypeArgumentListSimplified.add(genericTypeArgumentListSimplified446.tree)



                            IDENT447=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_postfixedExpression11731) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT447)

                            # AST Rewrite
                            # elements: IDENT, postfixedExpression, DOT
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 868:53: -> ^( DOT $postfixedExpression IDENT )
                                # Java.g:868:57: ^( DOT $postfixedExpression IDENT )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0



                            # Java.g:870:17: ( arguments -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments ) )?
                            alt132 = 2
                            LA132_0 = self.input.LA(1)

                            if (LA132_0 == LPAREN) :
                                alt132 = 1
                            if alt132 == 1:
                                # Java.g:870:21: arguments
                                pass 
                                self._state.following.append(self.FOLLOW_arguments_in_postfixedExpression11809)
                                arguments448 = self.arguments()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_arguments.add(arguments448.tree)

                                # AST Rewrite
                                # elements: arguments, genericTypeArgumentListSimplified, postfixedExpression
                                # token labels: 
                                # rule labels: retval
                                # token list labels: 
                                # rule list labels: 
                                # wildcard labels: 
                                if self._state.backtracking == 0:

                                    retval.tree = root_0

                                    if retval is not None:
                                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                    else:
                                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                    root_0 = self._adaptor.nil()
                                    # 870:53: -> ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments )
                                    # Java.g:870:57: ^( METHOD_CALL $postfixedExpression ( genericTypeArgumentListSimplified )? arguments )
                                    root_1 = self._adaptor.nil()
                                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                                    self._adaptor.addChild(root_1, stream_retval.nextTree())
                                    # Java.g:870:92: ( genericTypeArgumentListSimplified )?
                                    if stream_genericTypeArgumentListSimplified.hasNext():
                                        self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())


                                    stream_genericTypeArgumentListSimplified.reset();
                                    self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                    self._adaptor.addChild(root_0, root_1)



                                    retval.tree = root_0





                        elif alt134 == 2:
                            # Java.g:872:17: THIS
                            pass 
                            THIS449=self.match(self.input, THIS, self.FOLLOW_THIS_in_postfixedExpression11883) 
                            if self._state.backtracking == 0:
                                stream_THIS.add(THIS449)

                            # AST Rewrite
                            # elements: THIS, postfixedExpression, DOT
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 872:53: -> ^( DOT $postfixedExpression THIS )
                                # Java.g:872:57: ^( DOT $postfixedExpression THIS )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())
                                self._adaptor.addChild(root_1, stream_THIS.nextNode())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt134 == 3:
                            # Java.g:873:17: Super= SUPER arguments
                            pass 
                            Super=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_postfixedExpression11946) 
                            if self._state.backtracking == 0:
                                stream_SUPER.add(Super)
                            self._state.following.append(self.FOLLOW_arguments_in_postfixedExpression11948)
                            arguments450 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments450.tree)

                            # AST Rewrite
                            # elements: postfixedExpression, arguments
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 873:57: -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments )
                                # Java.g:873:61: ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] $postfixedExpression arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.create(SUPER_CONSTRUCTOR_CALL, Super, "SUPER_CONSTRUCTOR_CALL"), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt134 == 4:
                            # Java.g:874:17: ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) ) ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )?
                            pass 
                            # Java.g:874:17: ( SUPER innerDot= DOT IDENT -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT ) )
                            # Java.g:874:21: SUPER innerDot= DOT IDENT
                            pass 
                            SUPER451=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_postfixedExpression12001) 
                            if self._state.backtracking == 0:
                                stream_SUPER.add(SUPER451)
                            innerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_postfixedExpression12005) 
                            if self._state.backtracking == 0:
                                stream_DOT.add(innerDot)
                            IDENT452=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_postfixedExpression12007) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT452)

                            # AST Rewrite
                            # elements: SUPER, IDENT, postfixedExpression, innerDot, outerDot
                            # token labels: outerDot, innerDot
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0
                                stream_outerDot = RewriteRuleTokenStream(self._adaptor, "token outerDot", outerDot)
                                stream_innerDot = RewriteRuleTokenStream(self._adaptor, "token innerDot", innerDot)

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 874:53: -> ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT )
                                # Java.g:874:57: ^( $innerDot ^( $outerDot $postfixedExpression SUPER ) IDENT )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_innerDot.nextNode(), root_1)

                                # Java.g:874:69: ^( $outerDot $postfixedExpression SUPER )
                                root_2 = self._adaptor.nil()
                                root_2 = self._adaptor.becomeRoot(stream_outerDot.nextNode(), root_2)

                                self._adaptor.addChild(root_2, stream_retval.nextTree())
                                self._adaptor.addChild(root_2, stream_SUPER.nextNode())

                                self._adaptor.addChild(root_1, root_2)
                                self._adaptor.addChild(root_1, stream_IDENT.nextNode())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0



                            # Java.g:876:17: ( arguments -> ^( METHOD_CALL $postfixedExpression arguments ) )?
                            alt133 = 2
                            LA133_0 = self.input.LA(1)

                            if (LA133_0 == LPAREN) :
                                alt133 = 1
                            if alt133 == 1:
                                # Java.g:876:21: arguments
                                pass 
                                self._state.following.append(self.FOLLOW_arguments_in_postfixedExpression12074)
                                arguments453 = self.arguments()

                                self._state.following.pop()
                                if self._state.backtracking == 0:
                                    stream_arguments.add(arguments453.tree)

                                # AST Rewrite
                                # elements: postfixedExpression, arguments
                                # token labels: 
                                # rule labels: retval
                                # token list labels: 
                                # rule list labels: 
                                # wildcard labels: 
                                if self._state.backtracking == 0:

                                    retval.tree = root_0

                                    if retval is not None:
                                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                    else:
                                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                    root_0 = self._adaptor.nil()
                                    # 876:53: -> ^( METHOD_CALL $postfixedExpression arguments )
                                    # Java.g:876:57: ^( METHOD_CALL $postfixedExpression arguments )
                                    root_1 = self._adaptor.nil()
                                    root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                                    self._adaptor.addChild(root_1, stream_retval.nextTree())
                                    self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                    self._adaptor.addChild(root_0, root_1)



                                    retval.tree = root_0





                        elif alt134 == 5:
                            # Java.g:878:17: innerNewExpression
                            pass 
                            self._state.following.append(self.FOLLOW_innerNewExpression_in_postfixedExpression12145)
                            innerNewExpression454 = self.innerNewExpression()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_innerNewExpression.add(innerNewExpression454.tree)

                            # AST Rewrite
                            # elements: innerNewExpression, DOT, postfixedExpression
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 878:53: -> ^( DOT $postfixedExpression innerNewExpression )
                                # Java.g:878:57: ^( DOT $postfixedExpression innerNewExpression )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())
                                self._adaptor.addChild(root_1, stream_innerNewExpression.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0





                    elif alt135 == 2:
                        # Java.g:880:13: LBRACK expression RBRACK
                        pass 
                        LBRACK455=self.match(self.input, LBRACK, self.FOLLOW_LBRACK_in_postfixedExpression12202) 
                        if self._state.backtracking == 0:
                            stream_LBRACK.add(LBRACK455)
                        self._state.following.append(self.FOLLOW_expression_in_postfixedExpression12204)
                        expression456 = self.expression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_expression.add(expression456.tree)
                        RBRACK457=self.match(self.input, RBRACK, self.FOLLOW_RBRACK_in_postfixedExpression12206) 
                        if self._state.backtracking == 0:
                            stream_RBRACK.add(RBRACK457)

                        # AST Rewrite
                        # elements: expression, postfixedExpression
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 880:53: -> ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression )
                            # Java.g:880:57: ^( ARRAY_ELEMENT_ACCESS $postfixedExpression expression )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(ARRAY_ELEMENT_ACCESS, "ARRAY_ELEMENT_ACCESS"), root_1)

                            self._adaptor.addChild(root_1, stream_retval.nextTree())
                            self._adaptor.addChild(root_1, stream_expression.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    else:
                        break #loop135
                # Java.g:883:9: ( INC -> ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression) | DEC -> ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression) )?
                alt136 = 3
                LA136_0 = self.input.LA(1)

                if (LA136_0 == INC) :
                    alt136 = 1
                elif (LA136_0 == DEC) :
                    alt136 = 2
                if alt136 == 1:
                    # Java.g:883:13: INC
                    pass 
                    INC458=self.match(self.input, INC, self.FOLLOW_INC_in_postfixedExpression12267) 
                    if self._state.backtracking == 0:
                        stream_INC.add(INC458)

                    # AST Rewrite
                    # elements: postfixedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 883:17: -> ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression)
                        # Java.g:883:20: ^( POST_INC[$INC, \"POST_INC\"] $postfixedExpression)
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(POST_INC, INC458, "POST_INC"), root_1)

                        self._adaptor.addChild(root_1, stream_retval.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt136 == 2:
                    # Java.g:884:13: DEC
                    pass 
                    DEC459=self.match(self.input, DEC, self.FOLLOW_DEC_in_postfixedExpression12291) 
                    if self._state.backtracking == 0:
                        stream_DEC.add(DEC459)

                    # AST Rewrite
                    # elements: postfixedExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 884:17: -> ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression)
                        # Java.g:884:20: ^( POST_DEC[$DEC, \"POST_DEC\"] $postfixedExpression)
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(POST_DEC, DEC459, "POST_DEC"), root_1)

                        self._adaptor.addChild(root_1, stream_retval.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0






                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 104, postfixedExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "postfixedExpression"

    class primaryExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.primaryExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "primaryExpression"
    # Java.g:888:1: primaryExpression : ( parenthesizedExpression | literal | newExpression | qualifiedIdentExpression | genericTypeArgumentListSimplified ( SUPER ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) ) | IDENT arguments -> ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments ) | THIS arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) ) | ( THIS -> THIS ) ( arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments ) )? | SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] arguments ) | ( SUPER DOT IDENT ) ( arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments ) | -> ^( DOT SUPER IDENT ) ) | ( primitiveType -> primitiveType ) ( arrayDeclarator -> ^( arrayDeclarator $primaryExpression) )* DOT CLASS -> ^( DOT $primaryExpression CLASS ) | VOID DOT CLASS -> ^( DOT VOID CLASS ) );
    def primaryExpression(self, ):

        retval = self.primaryExpression_return()
        retval.start = self.input.LT(1)
        primaryExpression_StartIndex = self.input.index()
        root_0 = None

        SUPER465 = None
        DOT467 = None
        IDENT468 = None
        IDENT470 = None
        THIS472 = None
        THIS474 = None
        SUPER476 = None
        SUPER478 = None
        DOT479 = None
        IDENT480 = None
        DOT484 = None
        CLASS485 = None
        VOID486 = None
        DOT487 = None
        CLASS488 = None
        parenthesizedExpression460 = None

        literal461 = None

        newExpression462 = None

        qualifiedIdentExpression463 = None

        genericTypeArgumentListSimplified464 = None

        arguments466 = None

        arguments469 = None

        arguments471 = None

        arguments473 = None

        arguments475 = None

        arguments477 = None

        arguments481 = None

        primitiveType482 = None

        arrayDeclarator483 = None


        SUPER465_tree = None
        DOT467_tree = None
        IDENT468_tree = None
        IDENT470_tree = None
        THIS472_tree = None
        THIS474_tree = None
        SUPER476_tree = None
        SUPER478_tree = None
        DOT479_tree = None
        IDENT480_tree = None
        DOT484_tree = None
        CLASS485_tree = None
        VOID486_tree = None
        DOT487_tree = None
        CLASS488_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_CLASS = RewriteRuleTokenStream(self._adaptor, "token CLASS")
        stream_VOID = RewriteRuleTokenStream(self._adaptor, "token VOID")
        stream_SUPER = RewriteRuleTokenStream(self._adaptor, "token SUPER")
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")
        stream_THIS = RewriteRuleTokenStream(self._adaptor, "token THIS")
        stream_arrayDeclarator = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclarator")
        stream_arguments = RewriteRuleSubtreeStream(self._adaptor, "rule arguments")
        stream_primitiveType = RewriteRuleSubtreeStream(self._adaptor, "rule primitiveType")
        stream_genericTypeArgumentListSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentListSimplified")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 105):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:889:5: ( parenthesizedExpression | literal | newExpression | qualifiedIdentExpression | genericTypeArgumentListSimplified ( SUPER ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) ) | IDENT arguments -> ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments ) | THIS arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) ) | ( THIS -> THIS ) ( arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments ) )? | SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] arguments ) | ( SUPER DOT IDENT ) ( arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments ) | -> ^( DOT SUPER IDENT ) ) | ( primitiveType -> primitiveType ) ( arrayDeclarator -> ^( arrayDeclarator $primaryExpression) )* DOT CLASS -> ^( DOT $primaryExpression CLASS ) | VOID DOT CLASS -> ^( DOT VOID CLASS ) )
                alt142 = 10
                alt142 = self.dfa142.predict(self.input)
                if alt142 == 1:
                    # Java.g:889:9: parenthesizedExpression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_parenthesizedExpression_in_primaryExpression12331)
                    parenthesizedExpression460 = self.parenthesizedExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, parenthesizedExpression460.tree)


                elif alt142 == 2:
                    # Java.g:890:9: literal
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_literal_in_primaryExpression12341)
                    literal461 = self.literal()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, literal461.tree)


                elif alt142 == 3:
                    # Java.g:891:9: newExpression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_newExpression_in_primaryExpression12351)
                    newExpression462 = self.newExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, newExpression462.tree)


                elif alt142 == 4:
                    # Java.g:892:9: qualifiedIdentExpression
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_qualifiedIdentExpression_in_primaryExpression12361)
                    qualifiedIdentExpression463 = self.qualifiedIdentExpression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, qualifiedIdentExpression463.tree)


                elif alt142 == 5:
                    # Java.g:893:9: genericTypeArgumentListSimplified ( SUPER ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) ) | IDENT arguments -> ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments ) | THIS arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) )
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_primaryExpression12371)
                    genericTypeArgumentListSimplified464 = self.genericTypeArgumentListSimplified()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_genericTypeArgumentListSimplified.add(genericTypeArgumentListSimplified464.tree)
                    # Java.g:894:9: ( SUPER ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) ) | IDENT arguments -> ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments ) | THIS arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) )
                    alt138 = 3
                    LA138 = self.input.LA(1)
                    if LA138 == SUPER:
                        alt138 = 1
                    elif LA138 == IDENT:
                        alt138 = 2
                    elif LA138 == THIS:
                        alt138 = 3
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 138, 0, self.input)

                        raise nvae

                    if alt138 == 1:
                        # Java.g:894:13: SUPER ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) )
                        pass 
                        SUPER465=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_primaryExpression12385) 
                        if self._state.backtracking == 0:
                            stream_SUPER.add(SUPER465)
                        # Java.g:895:13: ( arguments -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments ) | DOT IDENT arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments ) )
                        alt137 = 2
                        LA137_0 = self.input.LA(1)

                        if (LA137_0 == LPAREN) :
                            alt137 = 1
                        elif (LA137_0 == DOT) :
                            alt137 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 137, 0, self.input)

                            raise nvae

                        if alt137 == 1:
                            # Java.g:895:17: arguments
                            pass 
                            self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12403)
                            arguments466 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments466.tree)

                            # AST Rewrite
                            # elements: arguments, genericTypeArgumentListSimplified
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 895:57: -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments )
                                # Java.g:895:61: ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.create(SUPER_CONSTRUCTOR_CALL, SUPER465, "SUPER_CONSTRUCTOR_CALL"), root_1)

                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt137 == 2:
                            # Java.g:896:17: DOT IDENT arguments
                            pass 
                            DOT467=self.match(self.input, DOT, self.FOLLOW_DOT_in_primaryExpression12463) 
                            if self._state.backtracking == 0:
                                stream_DOT.add(DOT467)
                            IDENT468=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_primaryExpression12465) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT468)
                            self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12467)
                            arguments469 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments469.tree)

                            # AST Rewrite
                            # elements: genericTypeArgumentListSimplified, arguments, DOT, IDENT, SUPER
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 896:57: -> ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments )
                                # Java.g:896:61: ^( METHOD_CALL ^( DOT SUPER IDENT ) genericTypeArgumentListSimplified arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                                # Java.g:896:75: ^( DOT SUPER IDENT )
                                root_2 = self._adaptor.nil()
                                root_2 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_2)

                                self._adaptor.addChild(root_2, stream_SUPER.nextNode())
                                self._adaptor.addChild(root_2, stream_IDENT.nextNode())

                                self._adaptor.addChild(root_1, root_2)
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0





                    elif alt138 == 2:
                        # Java.g:898:13: IDENT arguments
                        pass 
                        IDENT470=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_primaryExpression12534) 
                        if self._state.backtracking == 0:
                            stream_IDENT.add(IDENT470)
                        self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12536)
                        arguments471 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments471.tree)

                        # AST Rewrite
                        # elements: arguments, IDENT, genericTypeArgumentListSimplified
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 898:57: -> ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments )
                            # Java.g:898:61: ^( METHOD_CALL IDENT genericTypeArgumentListSimplified arguments )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                            self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                            self._adaptor.addChild(root_1, stream_arguments.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt138 == 3:
                        # Java.g:899:13: THIS arguments
                        pass 
                        THIS472=self.match(self.input, THIS, self.FOLLOW_THIS_in_primaryExpression12591) 
                        if self._state.backtracking == 0:
                            stream_THIS.add(THIS472)
                        self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12593)
                        arguments473 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments473.tree)

                        # AST Rewrite
                        # elements: genericTypeArgumentListSimplified, arguments
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 899:57: -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments )
                            # Java.g:899:61: ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] genericTypeArgumentListSimplified arguments )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(THIS_CONSTRUCTOR_CALL, THIS472, "THIS_CONSTRUCTOR_CALL"), root_1)

                            self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                            self._adaptor.addChild(root_1, stream_arguments.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt142 == 6:
                    # Java.g:901:9: ( THIS -> THIS ) ( arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments ) )?
                    pass 
                    # Java.g:901:9: ( THIS -> THIS )
                    # Java.g:901:13: THIS
                    pass 
                    THIS474=self.match(self.input, THIS, self.FOLLOW_THIS_in_primaryExpression12658) 
                    if self._state.backtracking == 0:
                        stream_THIS.add(THIS474)

                    # AST Rewrite
                    # elements: THIS
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 901:57: -> THIS
                        self._adaptor.addChild(root_0, stream_THIS.nextNode())



                        retval.tree = root_0



                    # Java.g:903:9: ( arguments -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments ) )?
                    alt139 = 2
                    LA139_0 = self.input.LA(1)

                    if (LA139_0 == LPAREN) :
                        alt139 = 1
                    if alt139 == 1:
                        # Java.g:903:13: arguments
                        pass 
                        self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12726)
                        arguments475 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments475.tree)

                        # AST Rewrite
                        # elements: arguments
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 903:57: -> ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments )
                            # Java.g:903:61: ^( THIS_CONSTRUCTOR_CALL[$THIS, \"THIS_CONSTRUCTOR_CALL\"] arguments )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(THIS_CONSTRUCTOR_CALL, THIS474, "THIS_CONSTRUCTOR_CALL"), root_1)

                            self._adaptor.addChild(root_1, stream_arguments.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt142 == 7:
                    # Java.g:905:9: SUPER arguments
                    pass 
                    SUPER476=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_primaryExpression12791) 
                    if self._state.backtracking == 0:
                        stream_SUPER.add(SUPER476)
                    self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12793)
                    arguments477 = self.arguments()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_arguments.add(arguments477.tree)

                    # AST Rewrite
                    # elements: arguments
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 905:57: -> ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] arguments )
                        # Java.g:905:61: ^( SUPER_CONSTRUCTOR_CALL[$SUPER, \"SUPER_CONSTRUCTOR_CALL\"] arguments )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(SUPER_CONSTRUCTOR_CALL, SUPER476, "SUPER_CONSTRUCTOR_CALL"), root_1)

                        self._adaptor.addChild(root_1, stream_arguments.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt142 == 8:
                    # Java.g:906:9: ( SUPER DOT IDENT ) ( arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments ) | -> ^( DOT SUPER IDENT ) )
                    pass 
                    # Java.g:906:9: ( SUPER DOT IDENT )
                    # Java.g:906:13: SUPER DOT IDENT
                    pass 
                    SUPER478=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_primaryExpression12849) 
                    if self._state.backtracking == 0:
                        stream_SUPER.add(SUPER478)
                    DOT479=self.match(self.input, DOT, self.FOLLOW_DOT_in_primaryExpression12851) 
                    if self._state.backtracking == 0:
                        stream_DOT.add(DOT479)
                    IDENT480=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_primaryExpression12853) 
                    if self._state.backtracking == 0:
                        stream_IDENT.add(IDENT480)



                    # Java.g:908:9: ( arguments -> ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments ) | -> ^( DOT SUPER IDENT ) )
                    alt140 = 2
                    LA140_0 = self.input.LA(1)

                    if (LA140_0 == LPAREN) :
                        alt140 = 1
                    elif (LA140_0 == EOF or (AND <= LA140_0 <= ASSIGN) or (BIT_SHIFT_RIGHT <= LA140_0 <= DOT) or (EQUAL <= LA140_0 <= LBRACK) or (LESS_OR_EQUAL <= LA140_0 <= LOGICAL_AND) or LA140_0 == LOGICAL_OR or (MINUS <= LA140_0 <= MOD_ASSIGN) or (NOT_EQUAL <= LA140_0 <= XOR_ASSIGN) or LA140_0 == INSTANCEOF) :
                        alt140 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 140, 0, self.input)

                        raise nvae

                    if alt140 == 1:
                        # Java.g:908:13: arguments
                        pass 
                        self._state.following.append(self.FOLLOW_arguments_in_primaryExpression12877)
                        arguments481 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments481.tree)

                        # AST Rewrite
                        # elements: arguments, SUPER, DOT, IDENT
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 908:57: -> ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments )
                            # Java.g:908:61: ^( METHOD_CALL ^( DOT SUPER IDENT ) arguments )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                            # Java.g:908:75: ^( DOT SUPER IDENT )
                            root_2 = self._adaptor.nil()
                            root_2 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_2)

                            self._adaptor.addChild(root_2, stream_SUPER.nextNode())
                            self._adaptor.addChild(root_2, stream_IDENT.nextNode())

                            self._adaptor.addChild(root_1, root_2)
                            self._adaptor.addChild(root_1, stream_arguments.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt140 == 2:
                        # Java.g:909:57: 
                        pass 
                        # AST Rewrite
                        # elements: SUPER, IDENT, DOT
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 909:57: -> ^( DOT SUPER IDENT )
                            # Java.g:909:61: ^( DOT SUPER IDENT )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_SUPER.nextNode())
                            self._adaptor.addChild(root_1, stream_IDENT.nextNode())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0





                elif alt142 == 9:
                    # Java.g:911:9: ( primitiveType -> primitiveType ) ( arrayDeclarator -> ^( arrayDeclarator $primaryExpression) )* DOT CLASS
                    pass 
                    # Java.g:911:9: ( primitiveType -> primitiveType )
                    # Java.g:911:13: primitiveType
                    pass 
                    self._state.following.append(self.FOLLOW_primitiveType_in_primaryExpression13019)
                    primitiveType482 = self.primitiveType()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_primitiveType.add(primitiveType482.tree)

                    # AST Rewrite
                    # elements: primitiveType
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 911:57: -> primitiveType
                        self._adaptor.addChild(root_0, stream_primitiveType.nextTree())



                        retval.tree = root_0



                    # Java.g:913:9: ( arrayDeclarator -> ^( arrayDeclarator $primaryExpression) )*
                    while True: #loop141
                        alt141 = 2
                        LA141_0 = self.input.LA(1)

                        if (LA141_0 == LBRACK) :
                            alt141 = 1


                        if alt141 == 1:
                            # Java.g:913:13: arrayDeclarator
                            pass 
                            self._state.following.append(self.FOLLOW_arrayDeclarator_in_primaryExpression13078)
                            arrayDeclarator483 = self.arrayDeclarator()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arrayDeclarator.add(arrayDeclarator483.tree)

                            # AST Rewrite
                            # elements: primaryExpression, arrayDeclarator
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 913:57: -> ^( arrayDeclarator $primaryExpression)
                                # Java.g:913:61: ^( arrayDeclarator $primaryExpression)
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_arrayDeclarator.nextNode(), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        else:
                            break #loop141
                    DOT484=self.match(self.input, DOT, self.FOLLOW_DOT_in_primaryExpression13137) 
                    if self._state.backtracking == 0:
                        stream_DOT.add(DOT484)
                    CLASS485=self.match(self.input, CLASS, self.FOLLOW_CLASS_in_primaryExpression13139) 
                    if self._state.backtracking == 0:
                        stream_CLASS.add(CLASS485)

                    # AST Rewrite
                    # elements: primaryExpression, CLASS, DOT
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 915:57: -> ^( DOT $primaryExpression CLASS )
                        # Java.g:915:61: ^( DOT $primaryExpression CLASS )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_retval.nextTree())
                        self._adaptor.addChild(root_1, stream_CLASS.nextNode())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt142 == 10:
                    # Java.g:916:9: VOID DOT CLASS
                    pass 
                    VOID486=self.match(self.input, VOID, self.FOLLOW_VOID_in_primaryExpression13199) 
                    if self._state.backtracking == 0:
                        stream_VOID.add(VOID486)
                    DOT487=self.match(self.input, DOT, self.FOLLOW_DOT_in_primaryExpression13201) 
                    if self._state.backtracking == 0:
                        stream_DOT.add(DOT487)
                    CLASS488=self.match(self.input, CLASS, self.FOLLOW_CLASS_in_primaryExpression13203) 
                    if self._state.backtracking == 0:
                        stream_CLASS.add(CLASS488)

                    # AST Rewrite
                    # elements: DOT, VOID, CLASS
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 916:57: -> ^( DOT VOID CLASS )
                        # Java.g:916:61: ^( DOT VOID CLASS )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_VOID.nextNode())
                        self._adaptor.addChild(root_1, stream_CLASS.nextNode())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 105, primaryExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "primaryExpression"

    class qualifiedIdentExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.qualifiedIdentExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "qualifiedIdentExpression"
    # Java.g:919:1: qualifiedIdentExpression : ( qualifiedIdentifier -> qualifiedIdentifier ) ( ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+ ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) ) | arguments -> ^( METHOD_CALL qualifiedIdentifier arguments ) | outerDot= DOT ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) ) )? ;
    def qualifiedIdentExpression(self, ):

        retval = self.qualifiedIdentExpression_return()
        retval.start = self.input.LT(1)
        qualifiedIdentExpression_StartIndex = self.input.index()
        root_0 = None

        outerDot = None
        Super = None
        innerDot = None
        DOT491 = None
        CLASS492 = None
        CLASS494 = None
        SUPER497 = None
        IDENT498 = None
        IDENT500 = None
        THIS502 = None
        qualifiedIdentifier489 = None

        arrayDeclarator490 = None

        arguments493 = None

        genericTypeArgumentListSimplified495 = None

        arguments496 = None

        arguments499 = None

        arguments501 = None

        arguments503 = None

        innerNewExpression504 = None


        outerDot_tree = None
        Super_tree = None
        innerDot_tree = None
        DOT491_tree = None
        CLASS492_tree = None
        CLASS494_tree = None
        SUPER497_tree = None
        IDENT498_tree = None
        IDENT500_tree = None
        THIS502_tree = None
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_CLASS = RewriteRuleTokenStream(self._adaptor, "token CLASS")
        stream_SUPER = RewriteRuleTokenStream(self._adaptor, "token SUPER")
        stream_DOT = RewriteRuleTokenStream(self._adaptor, "token DOT")
        stream_THIS = RewriteRuleTokenStream(self._adaptor, "token THIS")
        stream_arrayDeclarator = RewriteRuleSubtreeStream(self._adaptor, "rule arrayDeclarator")
        stream_arguments = RewriteRuleSubtreeStream(self._adaptor, "rule arguments")
        stream_qualifiedIdentifier = RewriteRuleSubtreeStream(self._adaptor, "rule qualifiedIdentifier")
        stream_genericTypeArgumentListSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentListSimplified")
        stream_innerNewExpression = RewriteRuleSubtreeStream(self._adaptor, "rule innerNewExpression")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 106):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:921:5: ( ( qualifiedIdentifier -> qualifiedIdentifier ) ( ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+ ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) ) | arguments -> ^( METHOD_CALL qualifiedIdentifier arguments ) | outerDot= DOT ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) ) )? )
                # Java.g:921:9: ( qualifiedIdentifier -> qualifiedIdentifier ) ( ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+ ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) ) | arguments -> ^( METHOD_CALL qualifiedIdentifier arguments ) | outerDot= DOT ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) ) )?
                pass 
                # Java.g:921:9: ( qualifiedIdentifier -> qualifiedIdentifier )
                # Java.g:921:13: qualifiedIdentifier
                pass 
                self._state.following.append(self.FOLLOW_qualifiedIdentifier_in_qualifiedIdentExpression13279)
                qualifiedIdentifier489 = self.qualifiedIdentifier()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_qualifiedIdentifier.add(qualifiedIdentifier489.tree)

                # AST Rewrite
                # elements: qualifiedIdentifier
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 921:61: -> qualifiedIdentifier
                    self._adaptor.addChild(root_0, stream_qualifiedIdentifier.nextTree())



                    retval.tree = root_0



                # Java.g:924:9: ( ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+ ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) ) | arguments -> ^( METHOD_CALL qualifiedIdentifier arguments ) | outerDot= DOT ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) ) )?
                alt146 = 4
                alt146 = self.dfa146.predict(self.input)
                if alt146 == 1:
                    # Java.g:924:13: ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+ ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) )
                    pass 
                    # Java.g:924:13: ( arrayDeclarator -> ^( arrayDeclarator $qualifiedIdentExpression) )+
                    cnt143 = 0
                    while True: #loop143
                        alt143 = 2
                        LA143_0 = self.input.LA(1)

                        if (LA143_0 == LBRACK) :
                            alt143 = 1


                        if alt143 == 1:
                            # Java.g:924:17: arrayDeclarator
                            pass 
                            self._state.following.append(self.FOLLOW_arrayDeclarator_in_qualifiedIdentExpression13349)
                            arrayDeclarator490 = self.arrayDeclarator()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arrayDeclarator.add(arrayDeclarator490.tree)

                            # AST Rewrite
                            # elements: qualifiedIdentExpression, arrayDeclarator
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 924:57: -> ^( arrayDeclarator $qualifiedIdentExpression)
                                # Java.g:924:61: ^( arrayDeclarator $qualifiedIdentExpression)
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(stream_arrayDeclarator.nextNode(), root_1)

                                self._adaptor.addChild(root_1, stream_retval.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        else:
                            if cnt143 >= 1:
                                break #loop143

                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            eee = EarlyExitException(143, self.input)
                            raise eee

                        cnt143 += 1
                    # Java.g:926:13: ( DOT CLASS -> ^( DOT $qualifiedIdentExpression CLASS ) )
                    # Java.g:926:17: DOT CLASS
                    pass 
                    DOT491=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedIdentExpression13416) 
                    if self._state.backtracking == 0:
                        stream_DOT.add(DOT491)
                    CLASS492=self.match(self.input, CLASS, self.FOLLOW_CLASS_in_qualifiedIdentExpression13418) 
                    if self._state.backtracking == 0:
                        stream_CLASS.add(CLASS492)

                    # AST Rewrite
                    # elements: CLASS, DOT, qualifiedIdentExpression
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 926:57: -> ^( DOT $qualifiedIdentExpression CLASS )
                        # Java.g:926:61: ^( DOT $qualifiedIdentExpression CLASS )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                        self._adaptor.addChild(root_1, stream_retval.nextTree())
                        self._adaptor.addChild(root_1, stream_CLASS.nextNode())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0





                elif alt146 == 2:
                    # Java.g:928:13: arguments
                    pass 
                    self._state.following.append(self.FOLLOW_arguments_in_qualifiedIdentExpression13488)
                    arguments493 = self.arguments()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_arguments.add(arguments493.tree)

                    # AST Rewrite
                    # elements: arguments, qualifiedIdentifier
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 928:57: -> ^( METHOD_CALL qualifiedIdentifier arguments )
                        # Java.g:928:61: ^( METHOD_CALL qualifiedIdentifier arguments )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                        self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                        self._adaptor.addChild(root_1, stream_arguments.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt146 == 3:
                    # Java.g:929:13: outerDot= DOT ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) )
                    pass 
                    outerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedIdentExpression13549) 
                    if self._state.backtracking == 0:
                        stream_DOT.add(outerDot)
                    # Java.g:930:13: ( CLASS -> ^( DOT qualifiedIdentifier CLASS ) | genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) ) | THIS -> ^( DOT qualifiedIdentifier THIS ) | Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments ) | innerNewExpression -> ^( DOT qualifiedIdentifier innerNewExpression ) )
                    alt145 = 5
                    LA145 = self.input.LA(1)
                    if LA145 == CLASS:
                        alt145 = 1
                    elif LA145 == LESS_THAN:
                        alt145 = 2
                    elif LA145 == THIS:
                        alt145 = 3
                    elif LA145 == SUPER:
                        alt145 = 4
                    elif LA145 == NEW:
                        alt145 = 5
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 145, 0, self.input)

                        raise nvae

                    if alt145 == 1:
                        # Java.g:930:17: CLASS
                        pass 
                        CLASS494=self.match(self.input, CLASS, self.FOLLOW_CLASS_in_qualifiedIdentExpression13567) 
                        if self._state.backtracking == 0:
                            stream_CLASS.add(CLASS494)

                        # AST Rewrite
                        # elements: DOT, CLASS, qualifiedIdentifier
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 930:57: -> ^( DOT qualifiedIdentifier CLASS )
                            # Java.g:930:61: ^( DOT qualifiedIdentifier CLASS )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                            self._adaptor.addChild(root_1, stream_CLASS.nextNode())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt145 == 2:
                        # Java.g:931:17: genericTypeArgumentListSimplified (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) )
                        pass 
                        self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_qualifiedIdentExpression13630)
                        genericTypeArgumentListSimplified495 = self.genericTypeArgumentListSimplified()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericTypeArgumentListSimplified.add(genericTypeArgumentListSimplified495.tree)
                        # Java.g:932:17: (Super= SUPER arguments -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments ) | SUPER innerDot= DOT IDENT arguments -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments ) | IDENT arguments -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments ) )
                        alt144 = 3
                        LA144_0 = self.input.LA(1)

                        if (LA144_0 == SUPER) :
                            LA144_1 = self.input.LA(2)

                            if (LA144_1 == DOT) :
                                alt144 = 2
                            elif (LA144_1 == LPAREN) :
                                alt144 = 1
                            else:
                                if self._state.backtracking > 0:
                                    raise BacktrackingFailed

                                nvae = NoViableAltException("", 144, 1, self.input)

                                raise nvae

                        elif (LA144_0 == IDENT) :
                            alt144 = 3
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed

                            nvae = NoViableAltException("", 144, 0, self.input)

                            raise nvae

                        if alt144 == 1:
                            # Java.g:932:21: Super= SUPER arguments
                            pass 
                            Super=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_qualifiedIdentExpression13654) 
                            if self._state.backtracking == 0:
                                stream_SUPER.add(Super)
                            self._state.following.append(self.FOLLOW_arguments_in_qualifiedIdentExpression13656)
                            arguments496 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments496.tree)

                            # AST Rewrite
                            # elements: qualifiedIdentifier, arguments, genericTypeArgumentListSimplified
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 932:57: -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments )
                                # Java.g:932:61: ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier genericTypeArgumentListSimplified arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.create(SUPER_CONSTRUCTOR_CALL, Super, "SUPER_CONSTRUCTOR_CALL"), root_1)

                                self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt144 == 2:
                            # Java.g:933:21: SUPER innerDot= DOT IDENT arguments
                            pass 
                            SUPER497=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_qualifiedIdentExpression13706) 
                            if self._state.backtracking == 0:
                                stream_SUPER.add(SUPER497)
                            innerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_qualifiedIdentExpression13710) 
                            if self._state.backtracking == 0:
                                stream_DOT.add(innerDot)
                            IDENT498=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_qualifiedIdentExpression13712) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT498)
                            self._state.following.append(self.FOLLOW_arguments_in_qualifiedIdentExpression13714)
                            arguments499 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments499.tree)

                            # AST Rewrite
                            # elements: IDENT, qualifiedIdentifier, outerDot, SUPER, innerDot, genericTypeArgumentListSimplified, arguments
                            # token labels: outerDot, innerDot
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0
                                stream_outerDot = RewriteRuleTokenStream(self._adaptor, "token outerDot", outerDot)
                                stream_innerDot = RewriteRuleTokenStream(self._adaptor, "token innerDot", innerDot)

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 933:57: -> ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments )
                                # Java.g:933:61: ^( METHOD_CALL ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT ) genericTypeArgumentListSimplified arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                                # Java.g:933:75: ^( $innerDot ^( $outerDot qualifiedIdentifier SUPER ) IDENT )
                                root_2 = self._adaptor.nil()
                                root_2 = self._adaptor.becomeRoot(stream_innerDot.nextNode(), root_2)

                                # Java.g:933:87: ^( $outerDot qualifiedIdentifier SUPER )
                                root_3 = self._adaptor.nil()
                                root_3 = self._adaptor.becomeRoot(stream_outerDot.nextNode(), root_3)

                                self._adaptor.addChild(root_3, stream_qualifiedIdentifier.nextTree())
                                self._adaptor.addChild(root_3, stream_SUPER.nextNode())

                                self._adaptor.addChild(root_2, root_3)
                                self._adaptor.addChild(root_2, stream_IDENT.nextNode())

                                self._adaptor.addChild(root_1, root_2)
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0


                        elif alt144 == 3:
                            # Java.g:934:21: IDENT arguments
                            pass 
                            IDENT500=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_qualifiedIdentExpression13764) 
                            if self._state.backtracking == 0:
                                stream_IDENT.add(IDENT500)
                            self._state.following.append(self.FOLLOW_arguments_in_qualifiedIdentExpression13766)
                            arguments501 = self.arguments()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_arguments.add(arguments501.tree)

                            # AST Rewrite
                            # elements: IDENT, arguments, genericTypeArgumentListSimplified, qualifiedIdentifier, DOT
                            # token labels: 
                            # rule labels: retval
                            # token list labels: 
                            # rule list labels: 
                            # wildcard labels: 
                            if self._state.backtracking == 0:

                                retval.tree = root_0

                                if retval is not None:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                                else:
                                    stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                                root_0 = self._adaptor.nil()
                                # 934:57: -> ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments )
                                # Java.g:934:61: ^( METHOD_CALL ^( DOT qualifiedIdentifier IDENT ) genericTypeArgumentListSimplified arguments )
                                root_1 = self._adaptor.nil()
                                root_1 = self._adaptor.becomeRoot(self._adaptor.createFromType(METHOD_CALL, "METHOD_CALL"), root_1)

                                # Java.g:934:75: ^( DOT qualifiedIdentifier IDENT )
                                root_2 = self._adaptor.nil()
                                root_2 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_2)

                                self._adaptor.addChild(root_2, stream_qualifiedIdentifier.nextTree())
                                self._adaptor.addChild(root_2, stream_IDENT.nextNode())

                                self._adaptor.addChild(root_1, root_2)
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())
                                self._adaptor.addChild(root_1, stream_arguments.nextTree())

                                self._adaptor.addChild(root_0, root_1)



                                retval.tree = root_0





                    elif alt145 == 3:
                        # Java.g:936:17: THIS
                        pass 
                        THIS502=self.match(self.input, THIS, self.FOLLOW_THIS_in_qualifiedIdentExpression13841) 
                        if self._state.backtracking == 0:
                            stream_THIS.add(THIS502)

                        # AST Rewrite
                        # elements: DOT, THIS, qualifiedIdentifier
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 936:57: -> ^( DOT qualifiedIdentifier THIS )
                            # Java.g:936:61: ^( DOT qualifiedIdentifier THIS )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                            self._adaptor.addChild(root_1, stream_THIS.nextNode())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt145 == 4:
                        # Java.g:937:17: Super= SUPER arguments
                        pass 
                        Super=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_qualifiedIdentExpression13907) 
                        if self._state.backtracking == 0:
                            stream_SUPER.add(Super)
                        self._state.following.append(self.FOLLOW_arguments_in_qualifiedIdentExpression13909)
                        arguments503 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments503.tree)

                        # AST Rewrite
                        # elements: qualifiedIdentifier, arguments
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 937:57: -> ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments )
                            # Java.g:937:61: ^( SUPER_CONSTRUCTOR_CALL[$Super, \"SUPER_CONSTRUCTOR_CALL\"] qualifiedIdentifier arguments )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(SUPER_CONSTRUCTOR_CALL, Super, "SUPER_CONSTRUCTOR_CALL"), root_1)

                            self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                            self._adaptor.addChild(root_1, stream_arguments.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt145 == 5:
                        # Java.g:938:17: innerNewExpression
                        pass 
                        self._state.following.append(self.FOLLOW_innerNewExpression_in_qualifiedIdentExpression13957)
                        innerNewExpression504 = self.innerNewExpression()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_innerNewExpression.add(innerNewExpression504.tree)

                        # AST Rewrite
                        # elements: DOT, innerNewExpression, qualifiedIdentifier
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 938:57: -> ^( DOT qualifiedIdentifier innerNewExpression )
                            # Java.g:938:61: ^( DOT qualifiedIdentifier innerNewExpression )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(stream_DOT.nextNode(), root_1)

                            self._adaptor.addChild(root_1, stream_qualifiedIdentifier.nextTree())
                            self._adaptor.addChild(root_1, stream_innerNewExpression.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0









                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 106, qualifiedIdentExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "qualifiedIdentExpression"

    class newExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.newExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "newExpression"
    # Java.g:943:1: newExpression : NEW ( primitiveType newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction ) | ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) ) ) ;
    def newExpression(self, ):

        retval = self.newExpression_return()
        retval.start = self.input.LT(1)
        newExpression_StartIndex = self.input.index()
        root_0 = None

        NEW505 = None
        primitiveType506 = None

        newArrayConstruction507 = None

        genericTypeArgumentListSimplified508 = None

        qualifiedTypeIdentSimplified509 = None

        newArrayConstruction510 = None

        arguments511 = None

        classBody512 = None


        NEW505_tree = None
        stream_NEW = RewriteRuleTokenStream(self._adaptor, "token NEW")
        stream_newArrayConstruction = RewriteRuleSubtreeStream(self._adaptor, "rule newArrayConstruction")
        stream_arguments = RewriteRuleSubtreeStream(self._adaptor, "rule arguments")
        stream_qualifiedTypeIdentSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule qualifiedTypeIdentSimplified")
        stream_primitiveType = RewriteRuleSubtreeStream(self._adaptor, "rule primitiveType")
        stream_classBody = RewriteRuleSubtreeStream(self._adaptor, "rule classBody")
        stream_genericTypeArgumentListSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentListSimplified")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 107):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:944:5: ( NEW ( primitiveType newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction ) | ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) ) ) )
                # Java.g:944:9: NEW ( primitiveType newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction ) | ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) ) )
                pass 
                NEW505=self.match(self.input, NEW, self.FOLLOW_NEW_in_newExpression14033) 
                if self._state.backtracking == 0:
                    stream_NEW.add(NEW505)
                # Java.g:945:9: ( primitiveType newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction ) | ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) ) )
                alt150 = 2
                LA150_0 = self.input.LA(1)

                if (LA150_0 == BOOLEAN or LA150_0 == BYTE or LA150_0 == CHAR or LA150_0 == DOUBLE or LA150_0 == FLOAT or (INT <= LA150_0 <= LONG) or LA150_0 == SHORT) :
                    alt150 = 1
                elif (LA150_0 == LESS_THAN or LA150_0 == IDENT) :
                    alt150 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 150, 0, self.input)

                    raise nvae

                if alt150 == 1:
                    # Java.g:945:13: primitiveType newArrayConstruction
                    pass 
                    self._state.following.append(self.FOLLOW_primitiveType_in_newExpression14047)
                    primitiveType506 = self.primitiveType()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_primitiveType.add(primitiveType506.tree)
                    self._state.following.append(self.FOLLOW_newArrayConstruction_in_newExpression14049)
                    newArrayConstruction507 = self.newArrayConstruction()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_newArrayConstruction.add(newArrayConstruction507.tree)

                    # AST Rewrite
                    # elements: primitiveType, newArrayConstruction
                    # token labels: 
                    # rule labels: retval
                    # token list labels: 
                    # rule list labels: 
                    # wildcard labels: 
                    if self._state.backtracking == 0:

                        retval.tree = root_0

                        if retval is not None:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                        else:
                            stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                        root_0 = self._adaptor.nil()
                        # 946:13: -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction )
                        # Java.g:946:17: ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] primitiveType newArrayConstruction )
                        root_1 = self._adaptor.nil()
                        root_1 = self._adaptor.becomeRoot(self._adaptor.create(STATIC_ARRAY_CREATOR, NEW505, "STATIC_ARRAY_CREATOR"), root_1)

                        self._adaptor.addChild(root_1, stream_primitiveType.nextTree())
                        self._adaptor.addChild(root_1, stream_newArrayConstruction.nextTree())

                        self._adaptor.addChild(root_0, root_1)



                        retval.tree = root_0


                elif alt150 == 2:
                    # Java.g:947:13: ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) )
                    pass 
                    # Java.g:947:13: ( genericTypeArgumentListSimplified )?
                    alt147 = 2
                    LA147_0 = self.input.LA(1)

                    if (LA147_0 == LESS_THAN) :
                        alt147 = 1
                    if alt147 == 1:
                        # Java.g:0:0: genericTypeArgumentListSimplified
                        pass 
                        self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_newExpression14093)
                        genericTypeArgumentListSimplified508 = self.genericTypeArgumentListSimplified()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_genericTypeArgumentListSimplified.add(genericTypeArgumentListSimplified508.tree)



                    self._state.following.append(self.FOLLOW_qualifiedTypeIdentSimplified_in_newExpression14096)
                    qualifiedTypeIdentSimplified509 = self.qualifiedTypeIdentSimplified()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_qualifiedTypeIdentSimplified.add(qualifiedTypeIdentSimplified509.tree)
                    # Java.g:948:13: ( newArrayConstruction -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction ) | arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? ) )
                    alt149 = 2
                    LA149_0 = self.input.LA(1)

                    if (LA149_0 == LBRACK) :
                        alt149 = 1
                    elif (LA149_0 == LPAREN) :
                        alt149 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 149, 0, self.input)

                        raise nvae

                    if alt149 == 1:
                        # Java.g:948:17: newArrayConstruction
                        pass 
                        self._state.following.append(self.FOLLOW_newArrayConstruction_in_newExpression14114)
                        newArrayConstruction510 = self.newArrayConstruction()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_newArrayConstruction.add(newArrayConstruction510.tree)

                        # AST Rewrite
                        # elements: newArrayConstruction, qualifiedTypeIdentSimplified, genericTypeArgumentListSimplified
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 949:17: -> ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction )
                            # Java.g:949:21: ^( STATIC_ARRAY_CREATOR[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified newArrayConstruction )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(STATIC_ARRAY_CREATOR, NEW505, "STATIC_ARRAY_CREATOR"), root_1)

                            # Java.g:949:74: ( genericTypeArgumentListSimplified )?
                            if stream_genericTypeArgumentListSimplified.hasNext():
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())


                            stream_genericTypeArgumentListSimplified.reset();
                            self._adaptor.addChild(root_1, stream_qualifiedTypeIdentSimplified.nextTree())
                            self._adaptor.addChild(root_1, stream_newArrayConstruction.nextTree())

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0


                    elif alt149 == 2:
                        # Java.g:950:17: arguments ( classBody )?
                        pass 
                        self._state.following.append(self.FOLLOW_arguments_in_newExpression14179)
                        arguments511 = self.arguments()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            stream_arguments.add(arguments511.tree)
                        # Java.g:950:27: ( classBody )?
                        alt148 = 2
                        LA148_0 = self.input.LA(1)

                        if (LA148_0 == LCURLY) :
                            alt148 = 1
                        if alt148 == 1:
                            # Java.g:0:0: classBody
                            pass 
                            self._state.following.append(self.FOLLOW_classBody_in_newExpression14181)
                            classBody512 = self.classBody()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                stream_classBody.add(classBody512.tree)




                        # AST Rewrite
                        # elements: classBody, arguments, genericTypeArgumentListSimplified, qualifiedTypeIdentSimplified
                        # token labels: 
                        # rule labels: retval
                        # token list labels: 
                        # rule list labels: 
                        # wildcard labels: 
                        if self._state.backtracking == 0:

                            retval.tree = root_0

                            if retval is not None:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                            else:
                                stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                            root_0 = self._adaptor.nil()
                            # 951:17: -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? )
                            # Java.g:951:21: ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? qualifiedTypeIdentSimplified arguments ( classBody )? )
                            root_1 = self._adaptor.nil()
                            root_1 = self._adaptor.becomeRoot(self._adaptor.create(CLASS_CONSTRUCTOR_CALL, NEW505, "STATIC_ARRAY_CREATOR"), root_1)

                            # Java.g:951:76: ( genericTypeArgumentListSimplified )?
                            if stream_genericTypeArgumentListSimplified.hasNext():
                                self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())


                            stream_genericTypeArgumentListSimplified.reset();
                            self._adaptor.addChild(root_1, stream_qualifiedTypeIdentSimplified.nextTree())
                            self._adaptor.addChild(root_1, stream_arguments.nextTree())
                            # Java.g:951:150: ( classBody )?
                            if stream_classBody.hasNext():
                                self._adaptor.addChild(root_1, stream_classBody.nextTree())


                            stream_classBody.reset();

                            self._adaptor.addChild(root_0, root_1)



                            retval.tree = root_0









                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 107, newExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "newExpression"

    class innerNewExpression_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.innerNewExpression_return, self).__init__()

            self.tree = None




    # $ANTLR start "innerNewExpression"
    # Java.g:956:1: innerNewExpression : NEW ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? ) ;
    def innerNewExpression(self, ):

        retval = self.innerNewExpression_return()
        retval.start = self.input.LT(1)
        innerNewExpression_StartIndex = self.input.index()
        root_0 = None

        NEW513 = None
        IDENT515 = None
        genericTypeArgumentListSimplified514 = None

        arguments516 = None

        classBody517 = None


        NEW513_tree = None
        IDENT515_tree = None
        stream_NEW = RewriteRuleTokenStream(self._adaptor, "token NEW")
        stream_IDENT = RewriteRuleTokenStream(self._adaptor, "token IDENT")
        stream_arguments = RewriteRuleSubtreeStream(self._adaptor, "rule arguments")
        stream_classBody = RewriteRuleSubtreeStream(self._adaptor, "rule classBody")
        stream_genericTypeArgumentListSimplified = RewriteRuleSubtreeStream(self._adaptor, "rule genericTypeArgumentListSimplified")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 108):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:957:5: ( NEW ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? ) )
                # Java.g:957:9: NEW ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )?
                pass 
                NEW513=self.match(self.input, NEW, self.FOLLOW_NEW_in_innerNewExpression14276) 
                if self._state.backtracking == 0:
                    stream_NEW.add(NEW513)
                # Java.g:957:13: ( genericTypeArgumentListSimplified )?
                alt151 = 2
                LA151_0 = self.input.LA(1)

                if (LA151_0 == LESS_THAN) :
                    alt151 = 1
                if alt151 == 1:
                    # Java.g:0:0: genericTypeArgumentListSimplified
                    pass 
                    self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_innerNewExpression14278)
                    genericTypeArgumentListSimplified514 = self.genericTypeArgumentListSimplified()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_genericTypeArgumentListSimplified.add(genericTypeArgumentListSimplified514.tree)



                IDENT515=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_innerNewExpression14281) 
                if self._state.backtracking == 0:
                    stream_IDENT.add(IDENT515)
                self._state.following.append(self.FOLLOW_arguments_in_innerNewExpression14283)
                arguments516 = self.arguments()

                self._state.following.pop()
                if self._state.backtracking == 0:
                    stream_arguments.add(arguments516.tree)
                # Java.g:957:64: ( classBody )?
                alt152 = 2
                LA152_0 = self.input.LA(1)

                if (LA152_0 == LCURLY) :
                    alt152 = 1
                if alt152 == 1:
                    # Java.g:0:0: classBody
                    pass 
                    self._state.following.append(self.FOLLOW_classBody_in_innerNewExpression14285)
                    classBody517 = self.classBody()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_classBody.add(classBody517.tree)




                # AST Rewrite
                # elements: arguments, genericTypeArgumentListSimplified, IDENT, classBody
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 958:9: -> ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? )
                    # Java.g:958:13: ^( CLASS_CONSTRUCTOR_CALL[$NEW, \"STATIC_ARRAY_CREATOR\"] ( genericTypeArgumentListSimplified )? IDENT arguments ( classBody )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(CLASS_CONSTRUCTOR_CALL, NEW513, "STATIC_ARRAY_CREATOR"), root_1)

                    # Java.g:958:68: ( genericTypeArgumentListSimplified )?
                    if stream_genericTypeArgumentListSimplified.hasNext():
                        self._adaptor.addChild(root_1, stream_genericTypeArgumentListSimplified.nextTree())


                    stream_genericTypeArgumentListSimplified.reset();
                    self._adaptor.addChild(root_1, stream_IDENT.nextNode())
                    self._adaptor.addChild(root_1, stream_arguments.nextTree())
                    # Java.g:958:119: ( classBody )?
                    if stream_classBody.hasNext():
                        self._adaptor.addChild(root_1, stream_classBody.nextTree())


                    stream_classBody.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 108, innerNewExpression_StartIndex, success)

            pass
        return retval

    # $ANTLR end "innerNewExpression"

    class newArrayConstruction_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.newArrayConstruction_return, self).__init__()

            self.tree = None




    # $ANTLR start "newArrayConstruction"
    # Java.g:961:1: newArrayConstruction : ( arrayDeclaratorList arrayInitializer | LBRACK expression RBRACK ( LBRACK expression RBRACK )* ( arrayDeclaratorList )? );
    def newArrayConstruction(self, ):

        retval = self.newArrayConstruction_return()
        retval.start = self.input.LT(1)
        newArrayConstruction_StartIndex = self.input.index()
        root_0 = None

        LBRACK520 = None
        RBRACK522 = None
        LBRACK523 = None
        RBRACK525 = None
        arrayDeclaratorList518 = None

        arrayInitializer519 = None

        expression521 = None

        expression524 = None

        arrayDeclaratorList526 = None


        LBRACK520_tree = None
        RBRACK522_tree = None
        LBRACK523_tree = None
        RBRACK525_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 109):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:962:5: ( arrayDeclaratorList arrayInitializer | LBRACK expression RBRACK ( LBRACK expression RBRACK )* ( arrayDeclaratorList )? )
                alt155 = 2
                LA155_0 = self.input.LA(1)

                if (LA155_0 == LBRACK) :
                    LA155_1 = self.input.LA(2)

                    if (LA155_1 == RBRACK) :
                        alt155 = 1
                    elif (LA155_1 == DEC or LA155_1 == INC or LA155_1 == LESS_THAN or LA155_1 == LOGICAL_NOT or (LPAREN <= LA155_1 <= MINUS) or LA155_1 == NOT or LA155_1 == PLUS or LA155_1 == BOOLEAN or LA155_1 == BYTE or LA155_1 == CHAR or LA155_1 == DOUBLE or LA155_1 == FALSE or LA155_1 == FLOAT or (INT <= LA155_1 <= LONG) or (NEW <= LA155_1 <= NULL) or LA155_1 == SHORT or LA155_1 == SUPER or LA155_1 == THIS or LA155_1 == TRUE or LA155_1 == VOID or (IDENT <= LA155_1 <= STRING_LITERAL)) :
                        alt155 = 2
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed

                        nvae = NoViableAltException("", 155, 1, self.input)

                        raise nvae

                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 155, 0, self.input)

                    raise nvae

                if alt155 == 1:
                    # Java.g:962:9: arrayDeclaratorList arrayInitializer
                    pass 
                    root_0 = self._adaptor.nil()

                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_newArrayConstruction14331)
                    arrayDeclaratorList518 = self.arrayDeclaratorList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, arrayDeclaratorList518.tree)
                    self._state.following.append(self.FOLLOW_arrayInitializer_in_newArrayConstruction14333)
                    arrayInitializer519 = self.arrayInitializer()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, arrayInitializer519.tree)


                elif alt155 == 2:
                    # Java.g:963:9: LBRACK expression RBRACK ( LBRACK expression RBRACK )* ( arrayDeclaratorList )?
                    pass 
                    root_0 = self._adaptor.nil()

                    LBRACK520=self.match(self.input, LBRACK, self.FOLLOW_LBRACK_in_newArrayConstruction14343)
                    self._state.following.append(self.FOLLOW_expression_in_newArrayConstruction14346)
                    expression521 = self.expression()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, expression521.tree)
                    RBRACK522=self.match(self.input, RBRACK, self.FOLLOW_RBRACK_in_newArrayConstruction14348)
                    # Java.g:963:36: ( LBRACK expression RBRACK )*
                    while True: #loop153
                        alt153 = 2
                        alt153 = self.dfa153.predict(self.input)
                        if alt153 == 1:
                            # Java.g:963:37: LBRACK expression RBRACK
                            pass 
                            LBRACK523=self.match(self.input, LBRACK, self.FOLLOW_LBRACK_in_newArrayConstruction14352)
                            self._state.following.append(self.FOLLOW_expression_in_newArrayConstruction14355)
                            expression524 = self.expression()

                            self._state.following.pop()
                            if self._state.backtracking == 0:
                                self._adaptor.addChild(root_0, expression524.tree)
                            RBRACK525=self.match(self.input, RBRACK, self.FOLLOW_RBRACK_in_newArrayConstruction14357)


                        else:
                            break #loop153
                    # Java.g:963:66: ( arrayDeclaratorList )?
                    alt154 = 2
                    LA154_0 = self.input.LA(1)

                    if (LA154_0 == LBRACK) :
                        LA154_1 = self.input.LA(2)

                        if (LA154_1 == RBRACK) :
                            alt154 = 1
                    if alt154 == 1:
                        # Java.g:0:0: arrayDeclaratorList
                        pass 
                        self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_newArrayConstruction14362)
                        arrayDeclaratorList526 = self.arrayDeclaratorList()

                        self._state.following.pop()
                        if self._state.backtracking == 0:
                            self._adaptor.addChild(root_0, arrayDeclaratorList526.tree)





                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 109, newArrayConstruction_StartIndex, success)

            pass
        return retval

    # $ANTLR end "newArrayConstruction"

    class arguments_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.arguments_return, self).__init__()

            self.tree = None




    # $ANTLR start "arguments"
    # Java.g:966:1: arguments : LPAREN ( expressionList )? RPAREN -> ^( ARGUMENT_LIST[$LPAREN, \"ARGUMENT_LIST\"] ( expressionList )? ) ;
    def arguments(self, ):

        retval = self.arguments_return()
        retval.start = self.input.LT(1)
        arguments_StartIndex = self.input.index()
        root_0 = None

        LPAREN527 = None
        RPAREN529 = None
        expressionList528 = None


        LPAREN527_tree = None
        RPAREN529_tree = None
        stream_RPAREN = RewriteRuleTokenStream(self._adaptor, "token RPAREN")
        stream_LPAREN = RewriteRuleTokenStream(self._adaptor, "token LPAREN")
        stream_expressionList = RewriteRuleSubtreeStream(self._adaptor, "rule expressionList")
        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 110):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:967:5: ( LPAREN ( expressionList )? RPAREN -> ^( ARGUMENT_LIST[$LPAREN, \"ARGUMENT_LIST\"] ( expressionList )? ) )
                # Java.g:967:9: LPAREN ( expressionList )? RPAREN
                pass 
                LPAREN527=self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_arguments14382) 
                if self._state.backtracking == 0:
                    stream_LPAREN.add(LPAREN527)
                # Java.g:967:16: ( expressionList )?
                alt156 = 2
                LA156_0 = self.input.LA(1)

                if (LA156_0 == DEC or LA156_0 == INC or LA156_0 == LESS_THAN or LA156_0 == LOGICAL_NOT or (LPAREN <= LA156_0 <= MINUS) or LA156_0 == NOT or LA156_0 == PLUS or LA156_0 == BOOLEAN or LA156_0 == BYTE or LA156_0 == CHAR or LA156_0 == DOUBLE or LA156_0 == FALSE or LA156_0 == FLOAT or (INT <= LA156_0 <= LONG) or (NEW <= LA156_0 <= NULL) or LA156_0 == SHORT or LA156_0 == SUPER or LA156_0 == THIS or LA156_0 == TRUE or LA156_0 == VOID or (IDENT <= LA156_0 <= STRING_LITERAL)) :
                    alt156 = 1
                if alt156 == 1:
                    # Java.g:0:0: expressionList
                    pass 
                    self._state.following.append(self.FOLLOW_expressionList_in_arguments14384)
                    expressionList528 = self.expressionList()

                    self._state.following.pop()
                    if self._state.backtracking == 0:
                        stream_expressionList.add(expressionList528.tree)



                RPAREN529=self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_arguments14387) 
                if self._state.backtracking == 0:
                    stream_RPAREN.add(RPAREN529)

                # AST Rewrite
                # elements: expressionList
                # token labels: 
                # rule labels: retval
                # token list labels: 
                # rule list labels: 
                # wildcard labels: 
                if self._state.backtracking == 0:

                    retval.tree = root_0

                    if retval is not None:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "rule retval", retval.tree)
                    else:
                        stream_retval = RewriteRuleSubtreeStream(self._adaptor, "token retval", None)


                    root_0 = self._adaptor.nil()
                    # 968:9: -> ^( ARGUMENT_LIST[$LPAREN, \"ARGUMENT_LIST\"] ( expressionList )? )
                    # Java.g:968:13: ^( ARGUMENT_LIST[$LPAREN, \"ARGUMENT_LIST\"] ( expressionList )? )
                    root_1 = self._adaptor.nil()
                    root_1 = self._adaptor.becomeRoot(self._adaptor.create(ARGUMENT_LIST, LPAREN527, "ARGUMENT_LIST"), root_1)

                    # Java.g:968:55: ( expressionList )?
                    if stream_expressionList.hasNext():
                        self._adaptor.addChild(root_1, stream_expressionList.nextTree())


                    stream_expressionList.reset();

                    self._adaptor.addChild(root_0, root_1)



                    retval.tree = root_0



                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 110, arguments_StartIndex, success)

            pass
        return retval

    # $ANTLR end "arguments"

    class literal_return(ParserRuleReturnScope):
        def __init__(self):
            super(JavaParser.literal_return, self).__init__()

            self.tree = None




    # $ANTLR start "literal"
    # Java.g:971:1: literal : ( HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL | FLOATING_POINT_LITERAL | CHARACTER_LITERAL | STRING_LITERAL | TRUE | FALSE | NULL );
    def literal(self, ):

        retval = self.literal_return()
        retval.start = self.input.LT(1)
        literal_StartIndex = self.input.index()
        root_0 = None

        set530 = None

        set530_tree = None

        success = False
        try:
            try:
                if self._state.backtracking > 0 and self.alreadyParsedRule(self.input, 111):
                    # for cached failed rules, alreadyParsedRule will raise an exception
                    success = True
                    return retval

                # Java.g:972:5: ( HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL | FLOATING_POINT_LITERAL | CHARACTER_LITERAL | STRING_LITERAL | TRUE | FALSE | NULL )
                # Java.g:
                pass 
                root_0 = self._adaptor.nil()

                set530 = self.input.LT(1)
                if self.input.LA(1) == FALSE or self.input.LA(1) == NULL or self.input.LA(1) == TRUE or (HEX_LITERAL <= self.input.LA(1) <= STRING_LITERAL):
                    self.input.consume()
                    if self._state.backtracking == 0:
                        self._adaptor.addChild(root_0, self._adaptor.createWithPayload(set530))
                    self._state.errorRecovery = False

                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    mse = MismatchedSetException(None, self.input)
                    raise mse





                retval.stop = self.input.LT(-1)

                if self._state.backtracking == 0:

                    retval.tree = self._adaptor.rulePostProcessing(root_0)
                    self._adaptor.setTokenBoundaries(retval.tree, retval.start, retval.stop)


                success = True
            except RecognitionException, re:
                self.reportError(re)
                self.recover(self.input, re)
                retval.tree = self._adaptor.errorNode(self.input, retval.start, self.input.LT(-1), re)
        finally:
            if self._state.backtracking > 0:
                self.memoize(self.input, 111, literal_StartIndex, success)

            pass
        return retval

    # $ANTLR end "literal"

    # $ANTLR start "synpred14_Java"
    def synpred14_Java_fragment(self, ):
        # Java.g:286:9: ( GREATER_THAN )
        # Java.g:286:9: GREATER_THAN
        pass 
        self.match(self.input, GREATER_THAN, self.FOLLOW_GREATER_THAN_in_synpred14_Java5045)


    # $ANTLR end "synpred14_Java"



    # $ANTLR start "synpred15_Java"
    def synpred15_Java_fragment(self, ):
        # Java.g:287:9: ( SHIFT_RIGHT )
        # Java.g:287:9: SHIFT_RIGHT
        pass 
        self.match(self.input, SHIFT_RIGHT, self.FOLLOW_SHIFT_RIGHT_in_synpred15_Java5055)


    # $ANTLR end "synpred15_Java"



    # $ANTLR start "synpred16_Java"
    def synpred16_Java_fragment(self, ):
        # Java.g:288:9: ( BIT_SHIFT_RIGHT )
        # Java.g:288:9: BIT_SHIFT_RIGHT
        pass 
        self.match(self.input, BIT_SHIFT_RIGHT, self.FOLLOW_BIT_SHIFT_RIGHT_in_synpred16_Java5065)


    # $ANTLR end "synpred16_Java"



    # $ANTLR start "synpred17_Java"
    def synpred17_Java_fragment(self, ):
        # Java.g:293:15: ( bound )
        # Java.g:293:15: bound
        pass 
        self._state.following.append(self.FOLLOW_bound_in_synpred17_Java5095)
        self.bound()

        self._state.following.pop()


    # $ANTLR end "synpred17_Java"



    # $ANTLR start "synpred32_Java"
    def synpred32_Java_fragment(self, ):
        # Java.g:350:9: ( STATIC block )
        # Java.g:350:9: STATIC block
        pass 
        self.match(self.input, STATIC, self.FOLLOW_STATIC_in_synpred32_Java5595)
        self._state.following.append(self.FOLLOW_block_in_synpred32_Java5597)
        self.block()

        self._state.following.pop()


    # $ANTLR end "synpred32_Java"



    # $ANTLR start "synpred42_Java"
    def synpred42_Java_fragment(self, ):
        # Java.g:352:13: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block ) )
        # Java.g:352:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block )
        pass 
        # Java.g:352:13: ( genericTypeParameterList )?
        alt162 = 2
        LA162_0 = self.input.LA(1)

        if (LA162_0 == LESS_THAN) :
            alt162 = 1
        if alt162 == 1:
            # Java.g:0:0: genericTypeParameterList
            pass 
            self._state.following.append(self.FOLLOW_genericTypeParameterList_in_synpred42_Java5634)
            self.genericTypeParameterList()

            self._state.following.pop()



        # Java.g:353:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block )
        alt169 = 3
        LA169 = self.input.LA(1)
        if LA169 == BOOLEAN or LA169 == BYTE or LA169 == CHAR or LA169 == DOUBLE or LA169 == FLOAT or LA169 == INT or LA169 == LONG or LA169 == SHORT:
            alt169 = 1
        elif LA169 == IDENT:
            LA169_2 = self.input.LA(2)

            if (LA169_2 == DOT or LA169_2 == LBRACK or LA169_2 == LESS_THAN or LA169_2 == IDENT) :
                alt169 = 1
            elif (LA169_2 == LPAREN) :
                alt169 = 3
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 169, 2, self.input)

                raise nvae

        elif LA169 == VOID:
            alt169 = 2
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 169, 0, self.input)

            raise nvae

        if alt169 == 1:
            # Java.g:353:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI )
            pass 
            self._state.following.append(self.FOLLOW_type_in_synpred42_Java5653)
            self.type()

            self._state.following.pop()
            self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred42_Java5655)
            self._state.following.append(self.FOLLOW_formalParameterList_in_synpred42_Java5657)
            self.formalParameterList()

            self._state.following.pop()
            # Java.g:353:48: ( arrayDeclaratorList )?
            alt163 = 2
            LA163_0 = self.input.LA(1)

            if (LA163_0 == LBRACK) :
                alt163 = 1
            if alt163 == 1:
                # Java.g:0:0: arrayDeclaratorList
                pass 
                self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred42_Java5659)
                self.arrayDeclaratorList()

                self._state.following.pop()



            # Java.g:353:69: ( throwsClause )?
            alt164 = 2
            LA164_0 = self.input.LA(1)

            if (LA164_0 == THROWS) :
                alt164 = 1
            if alt164 == 1:
                # Java.g:0:0: throwsClause
                pass 
                self._state.following.append(self.FOLLOW_throwsClause_in_synpred42_Java5662)
                self.throwsClause()

                self._state.following.pop()



            # Java.g:353:83: ( block | SEMI )
            alt165 = 2
            LA165_0 = self.input.LA(1)

            if (LA165_0 == LCURLY) :
                alt165 = 1
            elif (LA165_0 == SEMI) :
                alt165 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 165, 0, self.input)

                raise nvae

            if alt165 == 1:
                # Java.g:353:84: block
                pass 
                self._state.following.append(self.FOLLOW_block_in_synpred42_Java5666)
                self.block()

                self._state.following.pop()


            elif alt165 == 2:
                # Java.g:353:92: SEMI
                pass 
                self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred42_Java5670)





        elif alt169 == 2:
            # Java.g:355:17: VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI )
            pass 
            self.match(self.input, VOID, self.FOLLOW_VOID_in_synpred42_Java5732)
            self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred42_Java5734)
            self._state.following.append(self.FOLLOW_formalParameterList_in_synpred42_Java5736)
            self.formalParameterList()

            self._state.following.pop()
            # Java.g:355:48: ( throwsClause )?
            alt166 = 2
            LA166_0 = self.input.LA(1)

            if (LA166_0 == THROWS) :
                alt166 = 1
            if alt166 == 1:
                # Java.g:0:0: throwsClause
                pass 
                self._state.following.append(self.FOLLOW_throwsClause_in_synpred42_Java5738)
                self.throwsClause()

                self._state.following.pop()



            # Java.g:355:62: ( block | SEMI )
            alt167 = 2
            LA167_0 = self.input.LA(1)

            if (LA167_0 == LCURLY) :
                alt167 = 1
            elif (LA167_0 == SEMI) :
                alt167 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 167, 0, self.input)

                raise nvae

            if alt167 == 1:
                # Java.g:355:63: block
                pass 
                self._state.following.append(self.FOLLOW_block_in_synpred42_Java5742)
                self.block()

                self._state.following.pop()


            elif alt167 == 2:
                # Java.g:355:71: SEMI
                pass 
                self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred42_Java5746)





        elif alt169 == 3:
            # Java.g:357:17: ident= IDENT formalParameterList ( throwsClause )? block
            pass 
            ident=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred42_Java5805)
            self._state.following.append(self.FOLLOW_formalParameterList_in_synpred42_Java5807)
            self.formalParameterList()

            self._state.following.pop()
            # Java.g:357:49: ( throwsClause )?
            alt168 = 2
            LA168_0 = self.input.LA(1)

            if (LA168_0 == THROWS) :
                alt168 = 1
            if alt168 == 1:
                # Java.g:0:0: throwsClause
                pass 
                self._state.following.append(self.FOLLOW_throwsClause_in_synpred42_Java5809)
                self.throwsClause()

                self._state.following.pop()



            self._state.following.append(self.FOLLOW_block_in_synpred42_Java5812)
            self.block()

            self._state.following.pop()





    # $ANTLR end "synpred42_Java"



    # $ANTLR start "synpred43_Java"
    def synpred43_Java_fragment(self, ):
        # Java.g:351:9: ( modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block ) | type classFieldDeclaratorList SEMI ) )
        # Java.g:351:9: modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block ) | type classFieldDeclaratorList SEMI )
        pass 
        self._state.following.append(self.FOLLOW_modifierList_in_synpred43_Java5620)
        self.modifierList()

        self._state.following.pop()
        # Java.g:352:9: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block ) | type classFieldDeclaratorList SEMI )
        alt178 = 2
        LA178 = self.input.LA(1)
        if LA178 == LESS_THAN or LA178 == VOID:
            alt178 = 1
        elif LA178 == BOOLEAN or LA178 == BYTE or LA178 == CHAR or LA178 == DOUBLE or LA178 == FLOAT or LA178 == INT or LA178 == LONG or LA178 == SHORT:
            LA178_2 = self.input.LA(2)

            if (self.synpred42_Java()) :
                alt178 = 1
            elif (True) :
                alt178 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 178, 2, self.input)

                raise nvae

        elif LA178 == IDENT:
            LA178_3 = self.input.LA(2)

            if (self.synpred42_Java()) :
                alt178 = 1
            elif (True) :
                alt178 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 178, 3, self.input)

                raise nvae

        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 178, 0, self.input)

            raise nvae

        if alt178 == 1:
            # Java.g:352:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block )
            pass 
            # Java.g:352:13: ( genericTypeParameterList )?
            alt170 = 2
            LA170_0 = self.input.LA(1)

            if (LA170_0 == LESS_THAN) :
                alt170 = 1
            if alt170 == 1:
                # Java.g:0:0: genericTypeParameterList
                pass 
                self._state.following.append(self.FOLLOW_genericTypeParameterList_in_synpred43_Java5634)
                self.genericTypeParameterList()

                self._state.following.pop()



            # Java.g:353:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI ) | VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI ) | ident= IDENT formalParameterList ( throwsClause )? block )
            alt177 = 3
            LA177 = self.input.LA(1)
            if LA177 == BOOLEAN or LA177 == BYTE or LA177 == CHAR or LA177 == DOUBLE or LA177 == FLOAT or LA177 == INT or LA177 == LONG or LA177 == SHORT:
                alt177 = 1
            elif LA177 == IDENT:
                LA177_2 = self.input.LA(2)

                if (LA177_2 == LPAREN) :
                    alt177 = 3
                elif (LA177_2 == DOT or LA177_2 == LBRACK or LA177_2 == LESS_THAN or LA177_2 == IDENT) :
                    alt177 = 1
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 177, 2, self.input)

                    raise nvae

            elif LA177 == VOID:
                alt177 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 177, 0, self.input)

                raise nvae

            if alt177 == 1:
                # Java.g:353:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? ( block | SEMI )
                pass 
                self._state.following.append(self.FOLLOW_type_in_synpred43_Java5653)
                self.type()

                self._state.following.pop()
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred43_Java5655)
                self._state.following.append(self.FOLLOW_formalParameterList_in_synpred43_Java5657)
                self.formalParameterList()

                self._state.following.pop()
                # Java.g:353:48: ( arrayDeclaratorList )?
                alt171 = 2
                LA171_0 = self.input.LA(1)

                if (LA171_0 == LBRACK) :
                    alt171 = 1
                if alt171 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred43_Java5659)
                    self.arrayDeclaratorList()

                    self._state.following.pop()



                # Java.g:353:69: ( throwsClause )?
                alt172 = 2
                LA172_0 = self.input.LA(1)

                if (LA172_0 == THROWS) :
                    alt172 = 1
                if alt172 == 1:
                    # Java.g:0:0: throwsClause
                    pass 
                    self._state.following.append(self.FOLLOW_throwsClause_in_synpred43_Java5662)
                    self.throwsClause()

                    self._state.following.pop()



                # Java.g:353:83: ( block | SEMI )
                alt173 = 2
                LA173_0 = self.input.LA(1)

                if (LA173_0 == LCURLY) :
                    alt173 = 1
                elif (LA173_0 == SEMI) :
                    alt173 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 173, 0, self.input)

                    raise nvae

                if alt173 == 1:
                    # Java.g:353:84: block
                    pass 
                    self._state.following.append(self.FOLLOW_block_in_synpred43_Java5666)
                    self.block()

                    self._state.following.pop()


                elif alt173 == 2:
                    # Java.g:353:92: SEMI
                    pass 
                    self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred43_Java5670)





            elif alt177 == 2:
                # Java.g:355:17: VOID IDENT formalParameterList ( throwsClause )? ( block | SEMI )
                pass 
                self.match(self.input, VOID, self.FOLLOW_VOID_in_synpred43_Java5732)
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred43_Java5734)
                self._state.following.append(self.FOLLOW_formalParameterList_in_synpred43_Java5736)
                self.formalParameterList()

                self._state.following.pop()
                # Java.g:355:48: ( throwsClause )?
                alt174 = 2
                LA174_0 = self.input.LA(1)

                if (LA174_0 == THROWS) :
                    alt174 = 1
                if alt174 == 1:
                    # Java.g:0:0: throwsClause
                    pass 
                    self._state.following.append(self.FOLLOW_throwsClause_in_synpred43_Java5738)
                    self.throwsClause()

                    self._state.following.pop()



                # Java.g:355:62: ( block | SEMI )
                alt175 = 2
                LA175_0 = self.input.LA(1)

                if (LA175_0 == LCURLY) :
                    alt175 = 1
                elif (LA175_0 == SEMI) :
                    alt175 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 175, 0, self.input)

                    raise nvae

                if alt175 == 1:
                    # Java.g:355:63: block
                    pass 
                    self._state.following.append(self.FOLLOW_block_in_synpred43_Java5742)
                    self.block()

                    self._state.following.pop()


                elif alt175 == 2:
                    # Java.g:355:71: SEMI
                    pass 
                    self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred43_Java5746)





            elif alt177 == 3:
                # Java.g:357:17: ident= IDENT formalParameterList ( throwsClause )? block
                pass 
                ident=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred43_Java5805)
                self._state.following.append(self.FOLLOW_formalParameterList_in_synpred43_Java5807)
                self.formalParameterList()

                self._state.following.pop()
                # Java.g:357:49: ( throwsClause )?
                alt176 = 2
                LA176_0 = self.input.LA(1)

                if (LA176_0 == THROWS) :
                    alt176 = 1
                if alt176 == 1:
                    # Java.g:0:0: throwsClause
                    pass 
                    self._state.following.append(self.FOLLOW_throwsClause_in_synpred43_Java5809)
                    self.throwsClause()

                    self._state.following.pop()



                self._state.following.append(self.FOLLOW_block_in_synpred43_Java5812)
                self.block()

                self._state.following.pop()





        elif alt178 == 2:
            # Java.g:360:13: type classFieldDeclaratorList SEMI
            pass 
            self._state.following.append(self.FOLLOW_type_in_synpred43_Java5876)
            self.type()

            self._state.following.pop()
            self._state.following.append(self.FOLLOW_classFieldDeclaratorList_in_synpred43_Java5878)
            self.classFieldDeclaratorList()

            self._state.following.pop()
            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred43_Java5880)





    # $ANTLR end "synpred43_Java"



    # $ANTLR start "synpred44_Java"
    def synpred44_Java_fragment(self, ):
        # Java.g:363:9: ( typeDeclaration )
        # Java.g:363:9: typeDeclaration
        pass 
        self._state.following.append(self.FOLLOW_typeDeclaration_in_synpred44_Java5925)
        self.typeDeclaration()

        self._state.following.pop()


    # $ANTLR end "synpred44_Java"



    # $ANTLR start "synpred50_Java"
    def synpred50_Java_fragment(self, ):
        # Java.g:369:13: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI ) )
        # Java.g:369:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI )
        pass 
        # Java.g:369:13: ( genericTypeParameterList )?
        alt181 = 2
        LA181_0 = self.input.LA(1)

        if (LA181_0 == LESS_THAN) :
            alt181 = 1
        if alt181 == 1:
            # Java.g:0:0: genericTypeParameterList
            pass 
            self._state.following.append(self.FOLLOW_genericTypeParameterList_in_synpred50_Java5969)
            self.genericTypeParameterList()

            self._state.following.pop()



        # Java.g:370:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI )
        alt185 = 2
        LA185_0 = self.input.LA(1)

        if (LA185_0 == BOOLEAN or LA185_0 == BYTE or LA185_0 == CHAR or LA185_0 == DOUBLE or LA185_0 == FLOAT or (INT <= LA185_0 <= LONG) or LA185_0 == SHORT or LA185_0 == IDENT) :
            alt185 = 1
        elif (LA185_0 == VOID) :
            alt185 = 2
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 185, 0, self.input)

            raise nvae

        if alt185 == 1:
            # Java.g:370:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI
            pass 
            self._state.following.append(self.FOLLOW_type_in_synpred50_Java5988)
            self.type()

            self._state.following.pop()
            self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred50_Java5990)
            self._state.following.append(self.FOLLOW_formalParameterList_in_synpred50_Java5992)
            self.formalParameterList()

            self._state.following.pop()
            # Java.g:370:48: ( arrayDeclaratorList )?
            alt182 = 2
            LA182_0 = self.input.LA(1)

            if (LA182_0 == LBRACK) :
                alt182 = 1
            if alt182 == 1:
                # Java.g:0:0: arrayDeclaratorList
                pass 
                self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred50_Java5994)
                self.arrayDeclaratorList()

                self._state.following.pop()



            # Java.g:370:69: ( throwsClause )?
            alt183 = 2
            LA183_0 = self.input.LA(1)

            if (LA183_0 == THROWS) :
                alt183 = 1
            if alt183 == 1:
                # Java.g:0:0: throwsClause
                pass 
                self._state.following.append(self.FOLLOW_throwsClause_in_synpred50_Java5997)
                self.throwsClause()

                self._state.following.pop()



            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred50_Java6000)


        elif alt185 == 2:
            # Java.g:372:17: VOID IDENT formalParameterList ( throwsClause )? SEMI
            pass 
            self.match(self.input, VOID, self.FOLLOW_VOID_in_synpred50_Java6058)
            self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred50_Java6060)
            self._state.following.append(self.FOLLOW_formalParameterList_in_synpred50_Java6062)
            self.formalParameterList()

            self._state.following.pop()
            # Java.g:372:48: ( throwsClause )?
            alt184 = 2
            LA184_0 = self.input.LA(1)

            if (LA184_0 == THROWS) :
                alt184 = 1
            if alt184 == 1:
                # Java.g:0:0: throwsClause
                pass 
                self._state.following.append(self.FOLLOW_throwsClause_in_synpred50_Java6064)
                self.throwsClause()

                self._state.following.pop()



            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred50_Java6067)





    # $ANTLR end "synpred50_Java"



    # $ANTLR start "synpred51_Java"
    def synpred51_Java_fragment(self, ):
        # Java.g:368:9: ( modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI ) | type interfaceFieldDeclaratorList SEMI ) )
        # Java.g:368:9: modifierList ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI ) | type interfaceFieldDeclaratorList SEMI )
        pass 
        self._state.following.append(self.FOLLOW_modifierList_in_synpred51_Java5955)
        self.modifierList()

        self._state.following.pop()
        # Java.g:369:9: ( ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI ) | type interfaceFieldDeclaratorList SEMI )
        alt191 = 2
        LA191 = self.input.LA(1)
        if LA191 == LESS_THAN or LA191 == VOID:
            alt191 = 1
        elif LA191 == BOOLEAN or LA191 == BYTE or LA191 == CHAR or LA191 == DOUBLE or LA191 == FLOAT or LA191 == INT or LA191 == LONG or LA191 == SHORT:
            LA191_2 = self.input.LA(2)

            if (self.synpred50_Java()) :
                alt191 = 1
            elif (True) :
                alt191 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 191, 2, self.input)

                raise nvae

        elif LA191 == IDENT:
            LA191_3 = self.input.LA(2)

            if (self.synpred50_Java()) :
                alt191 = 1
            elif (True) :
                alt191 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 191, 3, self.input)

                raise nvae

        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 191, 0, self.input)

            raise nvae

        if alt191 == 1:
            # Java.g:369:13: ( genericTypeParameterList )? ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI )
            pass 
            # Java.g:369:13: ( genericTypeParameterList )?
            alt186 = 2
            LA186_0 = self.input.LA(1)

            if (LA186_0 == LESS_THAN) :
                alt186 = 1
            if alt186 == 1:
                # Java.g:0:0: genericTypeParameterList
                pass 
                self._state.following.append(self.FOLLOW_genericTypeParameterList_in_synpred51_Java5969)
                self.genericTypeParameterList()

                self._state.following.pop()



            # Java.g:370:13: ( type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI | VOID IDENT formalParameterList ( throwsClause )? SEMI )
            alt190 = 2
            LA190_0 = self.input.LA(1)

            if (LA190_0 == BOOLEAN or LA190_0 == BYTE or LA190_0 == CHAR or LA190_0 == DOUBLE or LA190_0 == FLOAT or (INT <= LA190_0 <= LONG) or LA190_0 == SHORT or LA190_0 == IDENT) :
                alt190 = 1
            elif (LA190_0 == VOID) :
                alt190 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 190, 0, self.input)

                raise nvae

            if alt190 == 1:
                # Java.g:370:17: type IDENT formalParameterList ( arrayDeclaratorList )? ( throwsClause )? SEMI
                pass 
                self._state.following.append(self.FOLLOW_type_in_synpred51_Java5988)
                self.type()

                self._state.following.pop()
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred51_Java5990)
                self._state.following.append(self.FOLLOW_formalParameterList_in_synpred51_Java5992)
                self.formalParameterList()

                self._state.following.pop()
                # Java.g:370:48: ( arrayDeclaratorList )?
                alt187 = 2
                LA187_0 = self.input.LA(1)

                if (LA187_0 == LBRACK) :
                    alt187 = 1
                if alt187 == 1:
                    # Java.g:0:0: arrayDeclaratorList
                    pass 
                    self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred51_Java5994)
                    self.arrayDeclaratorList()

                    self._state.following.pop()



                # Java.g:370:69: ( throwsClause )?
                alt188 = 2
                LA188_0 = self.input.LA(1)

                if (LA188_0 == THROWS) :
                    alt188 = 1
                if alt188 == 1:
                    # Java.g:0:0: throwsClause
                    pass 
                    self._state.following.append(self.FOLLOW_throwsClause_in_synpred51_Java5997)
                    self.throwsClause()

                    self._state.following.pop()



                self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred51_Java6000)


            elif alt190 == 2:
                # Java.g:372:17: VOID IDENT formalParameterList ( throwsClause )? SEMI
                pass 
                self.match(self.input, VOID, self.FOLLOW_VOID_in_synpred51_Java6058)
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred51_Java6060)
                self._state.following.append(self.FOLLOW_formalParameterList_in_synpred51_Java6062)
                self.formalParameterList()

                self._state.following.pop()
                # Java.g:372:48: ( throwsClause )?
                alt189 = 2
                LA189_0 = self.input.LA(1)

                if (LA189_0 == THROWS) :
                    alt189 = 1
                if alt189 == 1:
                    # Java.g:0:0: throwsClause
                    pass 
                    self._state.following.append(self.FOLLOW_throwsClause_in_synpred51_Java6064)
                    self.throwsClause()

                    self._state.following.pop()



                self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred51_Java6067)





        elif alt191 == 2:
            # Java.g:375:13: type interfaceFieldDeclaratorList SEMI
            pass 
            self._state.following.append(self.FOLLOW_type_in_synpred51_Java6130)
            self.type()

            self._state.following.pop()
            self._state.following.append(self.FOLLOW_interfaceFieldDeclaratorList_in_synpred51_Java6132)
            self.interfaceFieldDeclaratorList()

            self._state.following.pop()
            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred51_Java6134)





    # $ANTLR end "synpred51_Java"



    # $ANTLR start "synpred52_Java"
    def synpred52_Java_fragment(self, ):
        # Java.g:378:9: ( typeDeclaration )
        # Java.g:378:9: typeDeclaration
        pass 
        self._state.following.append(self.FOLLOW_typeDeclaration_in_synpred52_Java6179)
        self.typeDeclaration()

        self._state.following.pop()


    # $ANTLR end "synpred52_Java"



    # $ANTLR start "synpred58_Java"
    def synpred58_Java_fragment(self, ):
        # Java.g:417:9: ( arrayDeclarator )
        # Java.g:417:9: arrayDeclarator
        pass 
        self._state.following.append(self.FOLLOW_arrayDeclarator_in_synpred58_Java6473)
        self.arrayDeclarator()

        self._state.following.pop()


    # $ANTLR end "synpred58_Java"



    # $ANTLR start "synpred76_Java"
    def synpred76_Java_fragment(self, ):
        # Java.g:466:23: ( arrayDeclaratorList )
        # Java.g:466:23: arrayDeclaratorList
        pass 
        self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred76_Java6860)
        self.arrayDeclaratorList()

        self._state.following.pop()


    # $ANTLR end "synpred76_Java"



    # $ANTLR start "synpred77_Java"
    def synpred77_Java_fragment(self, ):
        # Java.g:471:28: ( arrayDeclaratorList )
        # Java.g:471:28: arrayDeclaratorList
        pass 
        self._state.following.append(self.FOLLOW_arrayDeclaratorList_in_synpred77_Java6903)
        self.arrayDeclaratorList()

        self._state.following.pop()


    # $ANTLR end "synpred77_Java"



    # $ANTLR start "synpred79_Java"
    def synpred79_Java_fragment(self, ):
        # Java.g:481:20: ( DOT typeIdent )
        # Java.g:481:20: DOT typeIdent
        pass 
        self.match(self.input, DOT, self.FOLLOW_DOT_in_synpred79_Java6988)
        self._state.following.append(self.FOLLOW_typeIdent_in_synpred79_Java6990)
        self.typeIdent()

        self._state.following.pop()


    # $ANTLR end "synpred79_Java"



    # $ANTLR start "synpred90_Java"
    def synpred90_Java_fragment(self, ):
        # Java.g:510:40: ( COMMA genericTypeArgument )
        # Java.g:510:40: COMMA genericTypeArgument
        pass 
        self.match(self.input, COMMA, self.FOLLOW_COMMA_in_synpred90_Java7213)
        self._state.following.append(self.FOLLOW_genericTypeArgument_in_synpred90_Java7215)
        self.genericTypeArgument()

        self._state.following.pop()


    # $ANTLR end "synpred90_Java"



    # $ANTLR start "synpred92_Java"
    def synpred92_Java_fragment(self, ):
        # Java.g:516:18: ( genericWildcardBoundType )
        # Java.g:516:18: genericWildcardBoundType
        pass 
        self._state.following.append(self.FOLLOW_genericWildcardBoundType_in_synpred92_Java7269)
        self.genericWildcardBoundType()

        self._state.following.pop()


    # $ANTLR end "synpred92_Java"



    # $ANTLR start "synpred97_Java"
    def synpred97_Java_fragment(self, ):
        # Java.g:541:42: ( COMMA formalParameterStandardDecl )
        # Java.g:541:42: COMMA formalParameterStandardDecl
        pass 
        self.match(self.input, COMMA, self.FOLLOW_COMMA_in_synpred97_Java7470)
        self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_synpred97_Java7472)
        self.formalParameterStandardDecl()

        self._state.following.pop()


    # $ANTLR end "synpred97_Java"



    # $ANTLR start "synpred99_Java"
    def synpred99_Java_fragment(self, ):
        # Java.g:541:13: ( formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )? )
        # Java.g:541:13: formalParameterStandardDecl ( COMMA formalParameterStandardDecl )* ( COMMA formalParameterVarArgDecl )?
        pass 
        self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_synpred99_Java7467)
        self.formalParameterStandardDecl()

        self._state.following.pop()
        # Java.g:541:41: ( COMMA formalParameterStandardDecl )*
        while True: #loop194
            alt194 = 2
            LA194_0 = self.input.LA(1)

            if (LA194_0 == COMMA) :
                LA194_1 = self.input.LA(2)

                if (self.synpred97_Java()) :
                    alt194 = 1




            if alt194 == 1:
                # Java.g:541:42: COMMA formalParameterStandardDecl
                pass 
                self.match(self.input, COMMA, self.FOLLOW_COMMA_in_synpred99_Java7470)
                self._state.following.append(self.FOLLOW_formalParameterStandardDecl_in_synpred99_Java7472)
                self.formalParameterStandardDecl()

                self._state.following.pop()


            else:
                break #loop194
        # Java.g:541:78: ( COMMA formalParameterVarArgDecl )?
        alt195 = 2
        LA195_0 = self.input.LA(1)

        if (LA195_0 == COMMA) :
            alt195 = 1
        if alt195 == 1:
            # Java.g:541:79: COMMA formalParameterVarArgDecl
            pass 
            self.match(self.input, COMMA, self.FOLLOW_COMMA_in_synpred99_Java7477)
            self._state.following.append(self.FOLLOW_formalParameterVarArgDecl_in_synpred99_Java7479)
            self.formalParameterVarArgDecl()

            self._state.following.pop()





    # $ANTLR end "synpred99_Java"



    # $ANTLR start "synpred100_Java"
    def synpred100_Java_fragment(self, ):
        # Java.g:544:13: ( formalParameterVarArgDecl )
        # Java.g:544:13: formalParameterVarArgDecl
        pass 
        self._state.following.append(self.FOLLOW_formalParameterVarArgDecl_in_synpred100_Java7534)
        self.formalParameterVarArgDecl()

        self._state.following.pop()


    # $ANTLR end "synpred100_Java"



    # $ANTLR start "synpred101_Java"
    def synpred101_Java_fragment(self, ):
        # Java.g:565:13: ( DOT ident= IDENT )
        # Java.g:565:13: DOT ident= IDENT
        pass 
        self.match(self.input, DOT, self.FOLLOW_DOT_in_synpred101_Java7765)
        ident=self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred101_Java7769)


    # $ANTLR end "synpred101_Java"



    # $ANTLR start "synpred102_Java"
    def synpred102_Java_fragment(self, ):
        # Java.g:572:9: ( annotation )
        # Java.g:572:9: annotation
        pass 
        self._state.following.append(self.FOLLOW_annotation_in_synpred102_Java7818)
        self.annotation()

        self._state.following.pop()


    # $ANTLR end "synpred102_Java"



    # $ANTLR start "synpred114_Java"
    def synpred114_Java_fragment(self, ):
        # Java.g:623:9: ( modifierList type ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI | classFieldDeclaratorList SEMI ) )
        # Java.g:623:9: modifierList type ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI | classFieldDeclaratorList SEMI )
        pass 
        self._state.following.append(self.FOLLOW_modifierList_in_synpred114_Java8240)
        self.modifierList()

        self._state.following.pop()
        self._state.following.append(self.FOLLOW_type_in_synpred114_Java8242)
        self.type()

        self._state.following.pop()
        # Java.g:624:9: ( IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI | classFieldDeclaratorList SEMI )
        alt200 = 2
        LA200_0 = self.input.LA(1)

        if (LA200_0 == IDENT) :
            LA200_1 = self.input.LA(2)

            if (LA200_1 == LPAREN) :
                alt200 = 1
            elif (LA200_1 == ASSIGN or LA200_1 == COMMA or LA200_1 == LBRACK or LA200_1 == SEMI) :
                alt200 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 200, 1, self.input)

                raise nvae

        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 200, 0, self.input)

            raise nvae

        if alt200 == 1:
            # Java.g:624:13: IDENT LPAREN RPAREN ( annotationDefaultValue )? SEMI
            pass 
            self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred114_Java8256)
            self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_synpred114_Java8258)
            self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_synpred114_Java8260)
            # Java.g:624:33: ( annotationDefaultValue )?
            alt199 = 2
            LA199_0 = self.input.LA(1)

            if (LA199_0 == DEFAULT) :
                alt199 = 1
            if alt199 == 1:
                # Java.g:0:0: annotationDefaultValue
                pass 
                self._state.following.append(self.FOLLOW_annotationDefaultValue_in_synpred114_Java8262)
                self.annotationDefaultValue()

                self._state.following.pop()



            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred114_Java8265)


        elif alt200 == 2:
            # Java.g:626:13: classFieldDeclaratorList SEMI
            pass 
            self._state.following.append(self.FOLLOW_classFieldDeclaratorList_in_synpred114_Java8307)
            self.classFieldDeclaratorList()

            self._state.following.pop()
            self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred114_Java8309)





    # $ANTLR end "synpred114_Java"



    # $ANTLR start "synpred116_Java"
    def synpred116_Java_fragment(self, ):
        # Java.g:644:9: ( localVariableDeclaration SEMI )
        # Java.g:644:9: localVariableDeclaration SEMI
        pass 
        self._state.following.append(self.FOLLOW_localVariableDeclaration_in_synpred116_Java8440)
        self.localVariableDeclaration()

        self._state.following.pop()
        self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred116_Java8442)


    # $ANTLR end "synpred116_Java"



    # $ANTLR start "synpred117_Java"
    def synpred117_Java_fragment(self, ):
        # Java.g:645:9: ( typeDeclaration )
        # Java.g:645:9: typeDeclaration
        pass 
        self._state.following.append(self.FOLLOW_typeDeclaration_in_synpred117_Java8453)
        self.typeDeclaration()

        self._state.following.pop()


    # $ANTLR end "synpred117_Java"



    # $ANTLR start "synpred121_Java"
    def synpred121_Java_fragment(self, ):
        # Java.g:662:13: ( ELSE elseStat= statement )
        # Java.g:662:13: ELSE elseStat= statement
        pass 
        self.match(self.input, ELSE, self.FOLLOW_ELSE_in_synpred121_Java8733)
        self._state.following.append(self.FOLLOW_statement_in_synpred121_Java8737)
        elseStat = self.statement()

        self._state.following.pop()


    # $ANTLR end "synpred121_Java"



    # $ANTLR start "synpred123_Java"
    def synpred123_Java_fragment(self, ):
        # Java.g:666:13: ( forInit SEMI forCondition SEMI forUpdater RPAREN statement )
        # Java.g:666:13: forInit SEMI forCondition SEMI forUpdater RPAREN statement
        pass 
        self._state.following.append(self.FOLLOW_forInit_in_synpred123_Java8916)
        self.forInit()

        self._state.following.pop()
        self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred123_Java8918)
        self._state.following.append(self.FOLLOW_forCondition_in_synpred123_Java8920)
        self.forCondition()

        self._state.following.pop()
        self.match(self.input, SEMI, self.FOLLOW_SEMI_in_synpred123_Java8922)
        self._state.following.append(self.FOLLOW_forUpdater_in_synpred123_Java8924)
        self.forUpdater()

        self._state.following.pop()
        self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_synpred123_Java8926)
        self._state.following.append(self.FOLLOW_statement_in_synpred123_Java8928)
        self.statement()

        self._state.following.pop()


    # $ANTLR end "synpred123_Java"



    # $ANTLR start "synpred130_Java"
    def synpred130_Java_fragment(self, ):
        # Java.g:673:47: ( switchBlockLabels )
        # Java.g:673:47: switchBlockLabels
        pass 
        self._state.following.append(self.FOLLOW_switchBlockLabels_in_synpred130_Java9251)
        self.switchBlockLabels()

        self._state.following.pop()


    # $ANTLR end "synpred130_Java"



    # $ANTLR start "synpred143_Java"
    def synpred143_Java_fragment(self, ):
        # Java.g:702:11: (c0= switchCaseLabels )
        # Java.g:702:11: c0= switchCaseLabels
        pass 
        self._state.following.append(self.FOLLOW_switchCaseLabels_in_synpred143_Java9839)
        c0 = self.switchCaseLabels()

        self._state.following.pop()


    # $ANTLR end "synpred143_Java"



    # $ANTLR start "synpred145_Java"
    def synpred145_Java_fragment(self, ):
        # Java.g:702:52: (c1= switchCaseLabels )
        # Java.g:702:52: c1= switchCaseLabels
        pass 
        self._state.following.append(self.FOLLOW_switchCaseLabels_in_synpred145_Java9847)
        c1 = self.switchCaseLabels()

        self._state.following.pop()


    # $ANTLR end "synpred145_Java"



    # $ANTLR start "synpred146_Java"
    def synpred146_Java_fragment(self, ):
        # Java.g:707:9: ( switchCaseLabel )
        # Java.g:707:9: switchCaseLabel
        pass 
        self._state.following.append(self.FOLLOW_switchCaseLabel_in_synpred146_Java9893)
        self.switchCaseLabel()

        self._state.following.pop()


    # $ANTLR end "synpred146_Java"



    # $ANTLR start "synpred149_Java"
    def synpred149_Java_fragment(self, ):
        # Java.g:719:9: ( localVariableDeclaration )
        # Java.g:719:9: localVariableDeclaration
        pass 
        self._state.following.append(self.FOLLOW_localVariableDeclaration_in_synpred149_Java9967)
        self.localVariableDeclaration()

        self._state.following.pop()


    # $ANTLR end "synpred149_Java"



    # $ANTLR start "synpred150_Java"
    def synpred150_Java_fragment(self, ):
        # Java.g:720:9: ( expressionList )
        # Java.g:720:9: expressionList
        pass 
        self._state.following.append(self.FOLLOW_expressionList_in_synpred150_Java9989)
        self.expressionList()

        self._state.following.pop()


    # $ANTLR end "synpred150_Java"



    # $ANTLR start "synpred193_Java"
    def synpred193_Java_fragment(self, ):
        # Java.g:856:9: ( LPAREN type RPAREN unaryExpression )
        # Java.g:856:9: LPAREN type RPAREN unaryExpression
        pass 
        self.match(self.input, LPAREN, self.FOLLOW_LPAREN_in_synpred193_Java11492)
        self._state.following.append(self.FOLLOW_type_in_synpred193_Java11494)
        self.type()

        self._state.following.pop()
        self.match(self.input, RPAREN, self.FOLLOW_RPAREN_in_synpred193_Java11496)
        self._state.following.append(self.FOLLOW_unaryExpression_in_synpred193_Java11498)
        self.unaryExpression()

        self._state.following.pop()


    # $ANTLR end "synpred193_Java"



    # $ANTLR start "synpred221_Java"
    def synpred221_Java_fragment(self, ):
        # Java.g:924:13: ( ( arrayDeclarator )+ ( DOT CLASS ) )
        # Java.g:924:13: ( arrayDeclarator )+ ( DOT CLASS )
        pass 
        # Java.g:924:13: ( arrayDeclarator )+
        cnt224 = 0
        while True: #loop224
            alt224 = 2
            LA224_0 = self.input.LA(1)

            if (LA224_0 == LBRACK) :
                alt224 = 1


            if alt224 == 1:
                # Java.g:924:17: arrayDeclarator
                pass 
                self._state.following.append(self.FOLLOW_arrayDeclarator_in_synpred221_Java13349)
                self.arrayDeclarator()

                self._state.following.pop()


            else:
                if cnt224 >= 1:
                    break #loop224

                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                eee = EarlyExitException(224, self.input)
                raise eee

            cnt224 += 1
        # Java.g:926:13: ( DOT CLASS )
        # Java.g:926:17: DOT CLASS
        pass 
        self.match(self.input, DOT, self.FOLLOW_DOT_in_synpred221_Java13416)
        self.match(self.input, CLASS, self.FOLLOW_CLASS_in_synpred221_Java13418)





    # $ANTLR end "synpred221_Java"



    # $ANTLR start "synpred229_Java"
    def synpred229_Java_fragment(self, ):
        # Java.g:929:13: (outerDot= DOT ( CLASS | genericTypeArgumentListSimplified (Super= SUPER arguments | SUPER innerDot= DOT IDENT arguments | IDENT arguments ) | THIS | Super= SUPER arguments | innerNewExpression ) )
        # Java.g:929:13: outerDot= DOT ( CLASS | genericTypeArgumentListSimplified (Super= SUPER arguments | SUPER innerDot= DOT IDENT arguments | IDENT arguments ) | THIS | Super= SUPER arguments | innerNewExpression )
        pass 
        outerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_synpred229_Java13549)
        # Java.g:930:13: ( CLASS | genericTypeArgumentListSimplified (Super= SUPER arguments | SUPER innerDot= DOT IDENT arguments | IDENT arguments ) | THIS | Super= SUPER arguments | innerNewExpression )
        alt227 = 5
        LA227 = self.input.LA(1)
        if LA227 == CLASS:
            alt227 = 1
        elif LA227 == LESS_THAN:
            alt227 = 2
        elif LA227 == THIS:
            alt227 = 3
        elif LA227 == SUPER:
            alt227 = 4
        elif LA227 == NEW:
            alt227 = 5
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed

            nvae = NoViableAltException("", 227, 0, self.input)

            raise nvae

        if alt227 == 1:
            # Java.g:930:17: CLASS
            pass 
            self.match(self.input, CLASS, self.FOLLOW_CLASS_in_synpred229_Java13567)


        elif alt227 == 2:
            # Java.g:931:17: genericTypeArgumentListSimplified (Super= SUPER arguments | SUPER innerDot= DOT IDENT arguments | IDENT arguments )
            pass 
            self._state.following.append(self.FOLLOW_genericTypeArgumentListSimplified_in_synpred229_Java13630)
            self.genericTypeArgumentListSimplified()

            self._state.following.pop()
            # Java.g:932:17: (Super= SUPER arguments | SUPER innerDot= DOT IDENT arguments | IDENT arguments )
            alt226 = 3
            LA226_0 = self.input.LA(1)

            if (LA226_0 == SUPER) :
                LA226_1 = self.input.LA(2)

                if (LA226_1 == DOT) :
                    alt226 = 2
                elif (LA226_1 == LPAREN) :
                    alt226 = 1
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed

                    nvae = NoViableAltException("", 226, 1, self.input)

                    raise nvae

            elif (LA226_0 == IDENT) :
                alt226 = 3
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed

                nvae = NoViableAltException("", 226, 0, self.input)

                raise nvae

            if alt226 == 1:
                # Java.g:932:21: Super= SUPER arguments
                pass 
                Super=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_synpred229_Java13654)
                self._state.following.append(self.FOLLOW_arguments_in_synpred229_Java13656)
                self.arguments()

                self._state.following.pop()


            elif alt226 == 2:
                # Java.g:933:21: SUPER innerDot= DOT IDENT arguments
                pass 
                self.match(self.input, SUPER, self.FOLLOW_SUPER_in_synpred229_Java13706)
                innerDot=self.match(self.input, DOT, self.FOLLOW_DOT_in_synpred229_Java13710)
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred229_Java13712)
                self._state.following.append(self.FOLLOW_arguments_in_synpred229_Java13714)
                self.arguments()

                self._state.following.pop()


            elif alt226 == 3:
                # Java.g:934:21: IDENT arguments
                pass 
                self.match(self.input, IDENT, self.FOLLOW_IDENT_in_synpred229_Java13764)
                self._state.following.append(self.FOLLOW_arguments_in_synpred229_Java13766)
                self.arguments()

                self._state.following.pop()





        elif alt227 == 3:
            # Java.g:936:17: THIS
            pass 
            self.match(self.input, THIS, self.FOLLOW_THIS_in_synpred229_Java13841)


        elif alt227 == 4:
            # Java.g:937:17: Super= SUPER arguments
            pass 
            Super=self.match(self.input, SUPER, self.FOLLOW_SUPER_in_synpred229_Java13907)
            self._state.following.append(self.FOLLOW_arguments_in_synpred229_Java13909)
            self.arguments()

            self._state.following.pop()


        elif alt227 == 5:
            # Java.g:938:17: innerNewExpression
            pass 
            self._state.following.append(self.FOLLOW_innerNewExpression_in_synpred229_Java13957)
            self.innerNewExpression()

            self._state.following.pop()





    # $ANTLR end "synpred229_Java"



    # $ANTLR start "synpred237_Java"
    def synpred237_Java_fragment(self, ):
        # Java.g:963:37: ( LBRACK expression RBRACK )
        # Java.g:963:37: LBRACK expression RBRACK
        pass 
        self.match(self.input, LBRACK, self.FOLLOW_LBRACK_in_synpred237_Java14352)
        self._state.following.append(self.FOLLOW_expression_in_synpred237_Java14355)
        self.expression()

        self._state.following.pop()
        self.match(self.input, RBRACK, self.FOLLOW_RBRACK_in_synpred237_Java14357)


    # $ANTLR end "synpred237_Java"




    # Delegated rules

    def synpred193_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred193_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred43_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred43_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred121_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred121_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred76_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred76_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred221_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred221_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred229_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred229_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred114_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred114_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred116_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred116_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred97_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred97_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred102_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred102_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred117_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred117_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred79_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred79_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred101_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred101_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred130_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred130_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred16_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred16_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred42_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred42_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred145_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred145_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred143_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred143_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred77_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred77_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred51_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred51_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred100_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred100_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred52_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred52_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred15_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred15_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred123_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred123_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred32_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred32_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred149_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred149_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred17_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred17_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred92_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred92_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred90_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred90_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred150_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred150_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred58_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred58_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred50_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred50_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred14_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred14_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred99_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred99_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred146_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred146_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred237_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred237_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred44_Java(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred44_Java_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success



    # lookup tables for DFA #36

    DFA36_eot = DFA.unpack(
        u"\27\uffff"
        )

    DFA36_eof = DFA.unpack(
        u"\27\uffff"
        )

    DFA36_min = DFA.unpack(
        u"\1\7\1\uffff\14\0\11\uffff"
        )

    DFA36_max = DFA.unpack(
        u"\1\u00a4\1\uffff\14\0\11\uffff"
        )

    DFA36_accept = DFA.unpack(
        u"\1\uffff\1\1\14\uffff\1\3\3\uffff\1\4\2\uffff\1\5\1\2"
        )

    DFA36_special = DFA.unpack(
        u"\2\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\1\13\11"
        u"\uffff"
        )

            
    DFA36_transition = [
        DFA.unpack(u"\1\15\17\uffff\1\1\1\uffff\1\16\22\uffff\1\25\10\uffff"
        u"\1\6\1\uffff\1\16\1\uffff\1\16\2\uffff\1\16\1\22\3\uffff\1\16\1"
        u"\uffff\1\22\2\uffff\1\14\1\uffff\1\16\4\uffff\1\22\1\uffff\2\16"
        u"\1\7\3\uffff\1\5\1\4\1\3\1\uffff\1\16\1\2\1\13\2\uffff\1\10\3\uffff"
        u"\1\11\2\uffff\1\16\1\12\75\uffff\1\16"),
        DFA.unpack(u""),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #36

    class DFA36(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA36_2 = input.LA(1)

                 
                index36_2 = input.index()
                input.rewind()
                s = -1
                if (self.synpred32_Java()):
                    s = 22

                elif (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_2)
                if s >= 0:
                    return s
            elif s == 1: 
                LA36_3 = input.LA(1)

                 
                index36_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_3)
                if s >= 0:
                    return s
            elif s == 2: 
                LA36_4 = input.LA(1)

                 
                index36_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_4)
                if s >= 0:
                    return s
            elif s == 3: 
                LA36_5 = input.LA(1)

                 
                index36_5 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_5)
                if s >= 0:
                    return s
            elif s == 4: 
                LA36_6 = input.LA(1)

                 
                index36_6 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_6)
                if s >= 0:
                    return s
            elif s == 5: 
                LA36_7 = input.LA(1)

                 
                index36_7 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_7)
                if s >= 0:
                    return s
            elif s == 6: 
                LA36_8 = input.LA(1)

                 
                index36_8 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_8)
                if s >= 0:
                    return s
            elif s == 7: 
                LA36_9 = input.LA(1)

                 
                index36_9 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_9)
                if s >= 0:
                    return s
            elif s == 8: 
                LA36_10 = input.LA(1)

                 
                index36_10 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_10)
                if s >= 0:
                    return s
            elif s == 9: 
                LA36_11 = input.LA(1)

                 
                index36_11 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_11)
                if s >= 0:
                    return s
            elif s == 10: 
                LA36_12 = input.LA(1)

                 
                index36_12 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_12)
                if s >= 0:
                    return s
            elif s == 11: 
                LA36_13 = input.LA(1)

                 
                index36_13 = input.index()
                input.rewind()
                s = -1
                if (self.synpred43_Java()):
                    s = 14

                elif (self.synpred44_Java()):
                    s = 18

                 
                input.seek(index36_13)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 36, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #43

    DFA43_eot = DFA.unpack(
        u"\25\uffff"
        )

    DFA43_eof = DFA.unpack(
        u"\25\uffff"
        )

    DFA43_min = DFA.unpack(
        u"\1\7\14\0\10\uffff"
        )

    DFA43_max = DFA.unpack(
        u"\1\u00a4\14\0\10\uffff"
        )

    DFA43_accept = DFA.unpack(
        u"\15\uffff\1\1\3\uffff\1\2\2\uffff\1\3"
        )

    DFA43_special = DFA.unpack(
        u"\1\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\1\13\10"
        u"\uffff"
        )

            
    DFA43_transition = [
        DFA.unpack(u"\1\14\21\uffff\1\15\22\uffff\1\24\10\uffff\1\5\1\uffff"
        u"\1\15\1\uffff\1\15\2\uffff\1\15\1\21\3\uffff\1\15\1\uffff\1\21"
        u"\2\uffff\1\13\1\uffff\1\15\4\uffff\1\21\1\uffff\2\15\1\6\3\uffff"
        u"\1\3\1\2\1\1\1\uffff\1\15\1\4\1\12\2\uffff\1\7\3\uffff\1\10\2\uffff"
        u"\1\15\1\11\75\uffff\1\15"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #43

    class DFA43(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA43_1 = input.LA(1)

                 
                index43_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_1)
                if s >= 0:
                    return s
            elif s == 1: 
                LA43_2 = input.LA(1)

                 
                index43_2 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_2)
                if s >= 0:
                    return s
            elif s == 2: 
                LA43_3 = input.LA(1)

                 
                index43_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_3)
                if s >= 0:
                    return s
            elif s == 3: 
                LA43_4 = input.LA(1)

                 
                index43_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_4)
                if s >= 0:
                    return s
            elif s == 4: 
                LA43_5 = input.LA(1)

                 
                index43_5 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_5)
                if s >= 0:
                    return s
            elif s == 5: 
                LA43_6 = input.LA(1)

                 
                index43_6 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_6)
                if s >= 0:
                    return s
            elif s == 6: 
                LA43_7 = input.LA(1)

                 
                index43_7 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_7)
                if s >= 0:
                    return s
            elif s == 7: 
                LA43_8 = input.LA(1)

                 
                index43_8 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_8)
                if s >= 0:
                    return s
            elif s == 8: 
                LA43_9 = input.LA(1)

                 
                index43_9 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_9)
                if s >= 0:
                    return s
            elif s == 9: 
                LA43_10 = input.LA(1)

                 
                index43_10 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_10)
                if s >= 0:
                    return s
            elif s == 10: 
                LA43_11 = input.LA(1)

                 
                index43_11 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_11)
                if s >= 0:
                    return s
            elif s == 11: 
                LA43_12 = input.LA(1)

                 
                index43_12 = input.index()
                input.rewind()
                s = -1
                if (self.synpred51_Java()):
                    s = 13

                elif (self.synpred52_Java()):
                    s = 17

                 
                input.seek(index43_12)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 43, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #86

    DFA86_eot = DFA.unpack(
        u"\22\uffff"
        )

    DFA86_eof = DFA.unpack(
        u"\22\uffff"
        )

    DFA86_min = DFA.unpack(
        u"\1\7\14\0\5\uffff"
        )

    DFA86_max = DFA.unpack(
        u"\1\u00a4\14\0\5\uffff"
        )

    DFA86_accept = DFA.unpack(
        u"\15\uffff\1\1\1\uffff\1\2\2\uffff"
        )

    DFA86_special = DFA.unpack(
        u"\1\uffff\1\0\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\1\13\5"
        u"\uffff"
        )

            
    DFA86_transition = [
        DFA.unpack(u"\1\14\55\uffff\1\5\1\uffff\1\15\1\uffff\1\15\2\uffff"
        u"\1\15\1\17\3\uffff\1\15\1\uffff\1\17\2\uffff\1\13\1\uffff\1\15"
        u"\4\uffff\1\17\1\uffff\2\15\1\6\3\uffff\1\3\1\2\1\1\1\uffff\1\15"
        u"\1\4\1\12\2\uffff\1\7\3\uffff\1\10\3\uffff\1\11\75\uffff\1\15"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #86

    class DFA86(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA86_1 = input.LA(1)

                 
                index86_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_1)
                if s >= 0:
                    return s
            elif s == 1: 
                LA86_2 = input.LA(1)

                 
                index86_2 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_2)
                if s >= 0:
                    return s
            elif s == 2: 
                LA86_3 = input.LA(1)

                 
                index86_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_3)
                if s >= 0:
                    return s
            elif s == 3: 
                LA86_4 = input.LA(1)

                 
                index86_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_4)
                if s >= 0:
                    return s
            elif s == 4: 
                LA86_5 = input.LA(1)

                 
                index86_5 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_5)
                if s >= 0:
                    return s
            elif s == 5: 
                LA86_6 = input.LA(1)

                 
                index86_6 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_6)
                if s >= 0:
                    return s
            elif s == 6: 
                LA86_7 = input.LA(1)

                 
                index86_7 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_7)
                if s >= 0:
                    return s
            elif s == 7: 
                LA86_8 = input.LA(1)

                 
                index86_8 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_8)
                if s >= 0:
                    return s
            elif s == 8: 
                LA86_9 = input.LA(1)

                 
                index86_9 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_9)
                if s >= 0:
                    return s
            elif s == 9: 
                LA86_10 = input.LA(1)

                 
                index86_10 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_10)
                if s >= 0:
                    return s
            elif s == 10: 
                LA86_11 = input.LA(1)

                 
                index86_11 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_11)
                if s >= 0:
                    return s
            elif s == 11: 
                LA86_12 = input.LA(1)

                 
                index86_12 = input.index()
                input.rewind()
                s = -1
                if (self.synpred114_Java()):
                    s = 13

                elif (True):
                    s = 15

                 
                input.seek(index86_12)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 86, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #88

    DFA88_eot = DFA.unpack(
        u"\55\uffff"
        )

    DFA88_eof = DFA.unpack(
        u"\55\uffff"
        )

    DFA88_min = DFA.unpack(
        u"\1\7\4\0\6\uffff\1\0\41\uffff"
        )

    DFA88_max = DFA.unpack(
        u"\1\u00aa\4\0\6\uffff\1\0\41\uffff"
        )

    DFA88_accept = DFA.unpack(
        u"\5\uffff\1\2\14\uffff\1\3\31\uffff\1\1"
        )

    DFA88_special = DFA.unpack(
        u"\1\uffff\1\0\1\1\1\2\1\3\6\uffff\1\4\41\uffff"
        )

            
    DFA88_transition = [
        DFA.unpack(u"\1\2\4\uffff\1\22\10\uffff\1\22\1\uffff\1\22\1\uffff"
        u"\1\22\1\uffff\1\22\1\uffff\2\22\3\uffff\1\22\3\uffff\1\22\5\uffff"
        u"\1\22\10\uffff\1\5\1\22\1\3\1\22\1\3\2\uffff\1\3\1\5\1\22\1\uffff"
        u"\1\22\1\3\1\uffff\1\5\1\uffff\1\22\1\1\1\uffff\1\3\2\22\2\uffff"
        u"\1\5\1\uffff\2\3\1\5\2\22\1\uffff\3\5\1\22\1\3\2\5\2\22\1\13\2"
        u"\22\1\uffff\1\5\3\22\1\5\1\22\74\uffff\1\4\6\22"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #88

    class DFA88(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA88_1 = input.LA(1)

                 
                index88_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred116_Java()):
                    s = 44

                elif (self.synpred117_Java()):
                    s = 5

                 
                input.seek(index88_1)
                if s >= 0:
                    return s
            elif s == 1: 
                LA88_2 = input.LA(1)

                 
                index88_2 = input.index()
                input.rewind()
                s = -1
                if (self.synpred116_Java()):
                    s = 44

                elif (self.synpred117_Java()):
                    s = 5

                 
                input.seek(index88_2)
                if s >= 0:
                    return s
            elif s == 2: 
                LA88_3 = input.LA(1)

                 
                index88_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred116_Java()):
                    s = 44

                elif (True):
                    s = 18

                 
                input.seek(index88_3)
                if s >= 0:
                    return s
            elif s == 3: 
                LA88_4 = input.LA(1)

                 
                index88_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred116_Java()):
                    s = 44

                elif (True):
                    s = 18

                 
                input.seek(index88_4)
                if s >= 0:
                    return s
            elif s == 4: 
                LA88_11 = input.LA(1)

                 
                index88_11 = input.index()
                input.rewind()
                s = -1
                if (self.synpred117_Java()):
                    s = 5

                elif (True):
                    s = 18

                 
                input.seek(index88_11)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 88, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #98

    DFA98_eot = DFA.unpack(
        u"\22\uffff"
        )

    DFA98_eof = DFA.unpack(
        u"\22\uffff"
        )

    DFA98_min = DFA.unpack(
        u"\1\14\15\uffff\1\4\3\uffff"
        )

    DFA98_max = DFA.unpack(
        u"\1\u00aa\15\uffff\1\114\3\uffff"
        )

    DFA98_accept = DFA.unpack(
        u"\1\uffff\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12\1\13\1\14\1"
        u"\15\1\uffff\1\17\1\20\1\16"
        )

    DFA98_special = DFA.unpack(
        u"\22\uffff"
        )

            
    DFA98_transition = [
        DFA.unpack(u"\1\17\10\uffff\1\17\1\uffff\1\1\1\uffff\1\17\1\uffff"
        u"\1\17\1\uffff\2\17\3\uffff\1\17\3\uffff\1\17\5\uffff\1\20\11\uffff"
        u"\1\2\1\17\1\14\1\17\2\uffff\1\17\1\uffff\1\15\1\uffff\1\6\1\17"
        u"\3\uffff\1\17\2\uffff\1\17\1\4\1\3\4\uffff\2\17\1\uffff\2\17\4"
        u"\uffff\1\12\1\17\2\uffff\1\17\1\10\1\11\1\17\1\13\2\uffff\1\17"
        u"\1\7\1\17\1\uffff\1\5\74\uffff\1\16\6\17"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\3\17\1\uffff\2\17\1\21\1\uffff\4\17\2\uffff\5\17\1"
        u"\uffff\3\17\1\uffff\6\17\1\uffff\6\17\3\uffff\11\17\27\uffff\1"
        u"\17"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #98

    class DFA98(DFA):
        pass


    # lookup tables for DFA #91

    DFA91_eot = DFA.unpack(
        u"\24\uffff"
        )

    DFA91_eof = DFA.unpack(
        u"\24\uffff"
        )

    DFA91_min = DFA.unpack(
        u"\1\7\4\0\17\uffff"
        )

    DFA91_max = DFA.unpack(
        u"\1\u00aa\4\0\17\uffff"
        )

    DFA91_accept = DFA.unpack(
        u"\5\uffff\1\1\15\uffff\1\2"
        )

    DFA91_special = DFA.unpack(
        u"\1\uffff\1\0\1\1\1\2\1\3\17\uffff"
        )

            
    DFA91_transition = [
        DFA.unpack(u"\1\2\4\uffff\1\5\10\uffff\1\5\3\uffff\1\5\1\uffff\1"
        u"\5\1\uffff\2\5\3\uffff\1\5\3\uffff\1\5\5\uffff\1\5\12\uffff\1\3"
        u"\1\uffff\1\3\2\uffff\1\3\4\uffff\1\3\3\uffff\1\5\1\1\1\uffff\1"
        u"\3\6\uffff\2\3\1\uffff\2\5\5\uffff\1\3\2\uffff\1\5\2\uffff\1\5"
        u"\3\uffff\1\5\1\uffff\1\5\76\uffff\1\4\6\5"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #91

    class DFA91(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA91_1 = input.LA(1)

                 
                index91_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred123_Java()):
                    s = 5

                elif (True):
                    s = 19

                 
                input.seek(index91_1)
                if s >= 0:
                    return s
            elif s == 1: 
                LA91_2 = input.LA(1)

                 
                index91_2 = input.index()
                input.rewind()
                s = -1
                if (self.synpred123_Java()):
                    s = 5

                elif (True):
                    s = 19

                 
                input.seek(index91_2)
                if s >= 0:
                    return s
            elif s == 2: 
                LA91_3 = input.LA(1)

                 
                index91_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred123_Java()):
                    s = 5

                elif (True):
                    s = 19

                 
                input.seek(index91_3)
                if s >= 0:
                    return s
            elif s == 3: 
                LA91_4 = input.LA(1)

                 
                index91_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred123_Java()):
                    s = 5

                elif (True):
                    s = 19

                 
                input.seek(index91_4)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 91, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #106

    DFA106_eot = DFA.unpack(
        u"\23\uffff"
        )

    DFA106_eof = DFA.unpack(
        u"\23\uffff"
        )

    DFA106_min = DFA.unpack(
        u"\1\7\2\uffff\2\0\16\uffff"
        )

    DFA106_max = DFA.unpack(
        u"\1\u00aa\2\uffff\2\0\16\uffff"
        )

    DFA106_accept = DFA.unpack(
        u"\1\uffff\1\1\3\uffff\1\2\14\uffff\1\3"
        )

    DFA106_special = DFA.unpack(
        u"\3\uffff\1\0\1\1\16\uffff"
        )

            
    DFA106_transition = [
        DFA.unpack(u"\1\1\4\uffff\1\5\10\uffff\1\5\3\uffff\1\5\1\uffff\1"
        u"\5\1\uffff\2\5\3\uffff\1\5\3\uffff\1\5\5\uffff\1\22\12\uffff\1"
        u"\3\1\uffff\1\3\2\uffff\1\3\4\uffff\1\3\3\uffff\1\5\1\1\1\uffff"
        u"\1\3\6\uffff\2\3\1\uffff\2\5\5\uffff\1\3\2\uffff\1\5\2\uffff\1"
        u"\5\3\uffff\1\5\1\uffff\1\5\76\uffff\1\4\6\5"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #106

    class DFA106(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA106_3 = input.LA(1)

                 
                index106_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred149_Java()):
                    s = 1

                elif (self.synpred150_Java()):
                    s = 5

                 
                input.seek(index106_3)
                if s >= 0:
                    return s
            elif s == 1: 
                LA106_4 = input.LA(1)

                 
                index106_4 = input.index()
                input.rewind()
                s = -1
                if (self.synpred149_Java()):
                    s = 1

                elif (self.synpred150_Java()):
                    s = 5

                 
                input.seek(index106_4)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 106, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #130

    DFA130_eot = DFA.unpack(
        u"\15\uffff"
        )

    DFA130_eof = DFA.unpack(
        u"\15\uffff"
        )

    DFA130_min = DFA.unpack(
        u"\1\31\2\uffff\1\0\11\uffff"
        )

    DFA130_max = DFA.unpack(
        u"\1\u00aa\2\uffff\1\0\11\uffff"
        )

    DFA130_accept = DFA.unpack(
        u"\1\uffff\1\1\1\2\1\uffff\1\4\7\uffff\1\3"
        )

    DFA130_special = DFA.unpack(
        u"\3\uffff\1\0\11\uffff"
        )

            
    DFA130_transition = [
        DFA.unpack(u"\1\4\1\uffff\1\2\1\uffff\1\3\4\uffff\1\1\24\uffff\1"
        u"\4\1\uffff\1\4\2\uffff\1\4\4\uffff\1\4\3\uffff\1\4\2\uffff\1\4"
        u"\6\uffff\2\4\1\uffff\2\4\5\uffff\1\4\2\uffff\1\4\2\uffff\1\4\3"
        u"\uffff\1\4\1\uffff\1\4\76\uffff\7\4"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #130

    class DFA130(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA130_3 = input.LA(1)

                 
                index130_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred193_Java()):
                    s = 12

                elif (True):
                    s = 4

                 
                input.seek(index130_3)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 130, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #142

    DFA142_eot = DFA.unpack(
        u"\14\uffff"
        )

    DFA142_eof = DFA.unpack(
        u"\14\uffff"
        )

    DFA142_min = DFA.unpack(
        u"\1\31\6\uffff\1\17\4\uffff"
        )

    DFA142_max = DFA.unpack(
        u"\1\u00aa\6\uffff\1\35\4\uffff"
        )

    DFA142_accept = DFA.unpack(
        u"\1\uffff\1\1\1\2\1\3\1\4\1\5\1\6\1\uffff\1\11\1\12\1\10\1\7"
        )

    DFA142_special = DFA.unpack(
        u"\14\uffff"
        )

            
    DFA142_transition = [
        DFA.unpack(u"\1\5\3\uffff\1\1\31\uffff\1\10\1\uffff\1\10\2\uffff"
        u"\1\10\4\uffff\1\10\3\uffff\1\2\2\uffff\1\10\6\uffff\2\10\1\uffff"
        u"\1\3\1\2\5\uffff\1\10\2\uffff\1\7\2\uffff\1\6\3\uffff\1\2\1\uffff"
        u"\1\11\76\uffff\1\4\6\2"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"\1\12\15\uffff\1\13"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #142

    class DFA142(DFA):
        pass


    # lookup tables for DFA #146

    DFA146_eot = DFA.unpack(
        u"\60\uffff"
        )

    DFA146_eof = DFA.unpack(
        u"\1\4\57\uffff"
        )

    DFA146_min = DFA.unpack(
        u"\1\4\1\0\1\uffff\1\0\54\uffff"
        )

    DFA146_max = DFA.unpack(
        u"\1\114\1\0\1\uffff\1\0\54\uffff"
        )

    DFA146_accept = DFA.unpack(
        u"\2\uffff\1\2\1\uffff\1\4\51\uffff\1\1\1\3"
        )

    DFA146_special = DFA.unpack(
        u"\1\uffff\1\0\1\uffff\1\1\54\uffff"
        )

            
    DFA146_transition = [
        DFA.unpack(u"\3\4\1\uffff\7\4\1\3\2\uffff\4\4\1\1\1\uffff\3\4\1\uffff"
        u"\1\4\1\2\4\4\1\uffff\22\4\27\uffff\1\4"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #146

    class DFA146(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA146_1 = input.LA(1)

                 
                index146_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred221_Java()):
                    s = 46

                elif (True):
                    s = 4

                 
                input.seek(index146_1)
                if s >= 0:
                    return s
            elif s == 1: 
                LA146_3 = input.LA(1)

                 
                index146_3 = input.index()
                input.rewind()
                s = -1
                if (self.synpred229_Java()):
                    s = 47

                elif (True):
                    s = 4

                 
                input.seek(index146_3)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 146, _s, input)
            self_.error(nvae)
            raise nvae
    # lookup tables for DFA #153

    DFA153_eot = DFA.unpack(
        u"\56\uffff"
        )

    DFA153_eof = DFA.unpack(
        u"\1\2\55\uffff"
        )

    DFA153_min = DFA.unpack(
        u"\1\4\1\0\54\uffff"
        )

    DFA153_max = DFA.unpack(
        u"\1\114\1\0\54\uffff"
        )

    DFA153_accept = DFA.unpack(
        u"\2\uffff\1\2\52\uffff\1\1"
        )

    DFA153_special = DFA.unpack(
        u"\1\uffff\1\0\54\uffff"
        )

            
    DFA153_transition = [
        DFA.unpack(u"\3\2\1\uffff\10\2\2\uffff\4\2\1\1\1\uffff\3\2\1\uffff"
        u"\1\2\1\uffff\4\2\1\uffff\22\2\27\uffff\1\2"),
        DFA.unpack(u"\1\uffff"),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u""),
        DFA.unpack(u"")
    ]

    # class definition for DFA #153

    class DFA153(DFA):
        pass


        def specialStateTransition(self_, s, input):
            # convince pylint that my self_ magic is ok ;)
            # pylint: disable-msg=E0213

            # pretend we are a member of the recognizer
            # thus semantic predicates can be evaluated
            self = self_.recognizer

            _s = s

            if s == 0: 
                LA153_1 = input.LA(1)

                 
                index153_1 = input.index()
                input.rewind()
                s = -1
                if (self.synpred237_Java()):
                    s = 45

                elif (True):
                    s = 2

                 
                input.seek(index153_1)
                if s >= 0:
                    return s

            if self._state.backtracking >0:
                raise BacktrackingFailed
            nvae = NoViableAltException(self_.getDescription(), 153, _s, input)
            self_.error(nvae)
            raise nvae
 

    FOLLOW_compilationUnit_in_javaSource4461 = frozenset([1])
    FOLLOW_annotationList_in_compilationUnit4497 = frozenset([1, 7, 44, 53, 61, 67, 70, 77, 78, 81, 84, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_packageDeclaration_in_compilationUnit4507 = frozenset([1, 7, 44, 53, 61, 67, 70, 77, 78, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_importDeclaration_in_compilationUnit4518 = frozenset([1, 7, 44, 53, 61, 67, 70, 77, 78, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_typeDecls_in_compilationUnit4529 = frozenset([1, 7, 44, 53, 61, 67, 70, 77, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_typeDeclaration_in_typeDecls4549 = frozenset([1])
    FOLLOW_SEMI_in_typeDecls4559 = frozenset([1])
    FOLLOW_PACKAGE_in_packageDeclaration4579 = frozenset([164])
    FOLLOW_qualifiedIdentifier_in_packageDeclaration4582 = frozenset([44])
    FOLLOW_SEMI_in_packageDeclaration4584 = frozenset([1])
    FOLLOW_IMPORT_in_importDeclaration4604 = frozenset([90, 164])
    FOLLOW_STATIC_in_importDeclaration4607 = frozenset([164])
    FOLLOW_qualifiedIdentifier_in_importDeclaration4610 = frozenset([16, 44])
    FOLLOW_DOTSTAR_in_importDeclaration4612 = frozenset([44])
    FOLLOW_SEMI_in_importDeclaration4615 = frozenset([1])
    FOLLOW_modifierList_in_typeDeclaration4635 = frozenset([7, 53, 61, 67, 70, 77, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_classTypeDeclaration_in_typeDeclaration4650 = frozenset([1])
    FOLLOW_interfaceTypeDeclaration_in_typeDeclaration4665 = frozenset([1])
    FOLLOW_enumTypeDeclaration_in_typeDeclaration4680 = frozenset([1])
    FOLLOW_annotationTypeDeclaration_in_typeDeclaration4695 = frozenset([1])
    FOLLOW_CLASS_in_classTypeDeclaration4726 = frozenset([164])
    FOLLOW_IDENT_in_classTypeDeclaration4728 = frozenset([23, 25, 68, 75])
    FOLLOW_genericTypeParameterList_in_classTypeDeclaration4730 = frozenset([23, 25, 68, 75])
    FOLLOW_classExtendsClause_in_classTypeDeclaration4733 = frozenset([23, 25, 68, 75])
    FOLLOW_implementsClause_in_classTypeDeclaration4736 = frozenset([23, 25, 68, 75])
    FOLLOW_classBody_in_classTypeDeclaration4739 = frozenset([1])
    FOLLOW_EXTENDS_in_classExtendsClause4802 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_classExtendsClause4804 = frozenset([1])
    FOLLOW_EXTENDS_in_interfaceExtendsClause4841 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_typeList_in_interfaceExtendsClause4843 = frozenset([1])
    FOLLOW_IMPLEMENTS_in_implementsClause4880 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_typeList_in_implementsClause4882 = frozenset([1])
    FOLLOW_LESS_THAN_in_genericTypeParameterList4919 = frozenset([164])
    FOLLOW_genericTypeParameter_in_genericTypeParameterList4921 = frozenset([8, 11, 20, 47])
    FOLLOW_COMMA_in_genericTypeParameterList4924 = frozenset([164])
    FOLLOW_genericTypeParameter_in_genericTypeParameterList4926 = frozenset([8, 11, 20, 47])
    FOLLOW_genericTypeListClosing_in_genericTypeParameterList4930 = frozenset([1])
    FOLLOW_GREATER_THAN_in_genericTypeListClosing5045 = frozenset([1])
    FOLLOW_SHIFT_RIGHT_in_genericTypeListClosing5055 = frozenset([1])
    FOLLOW_BIT_SHIFT_RIGHT_in_genericTypeListClosing5065 = frozenset([1])
    FOLLOW_IDENT_in_genericTypeParameter5093 = frozenset([1, 68])
    FOLLOW_bound_in_genericTypeParameter5095 = frozenset([1])
    FOLLOW_EXTENDS_in_bound5133 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_bound5135 = frozenset([1, 4])
    FOLLOW_AND_in_bound5138 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_bound5140 = frozenset([1, 4])
    FOLLOW_ENUM_in_enumTypeDeclaration5181 = frozenset([164])
    FOLLOW_IDENT_in_enumTypeDeclaration5183 = frozenset([23, 75])
    FOLLOW_implementsClause_in_enumTypeDeclaration5185 = frozenset([23, 75])
    FOLLOW_enumBody_in_enumTypeDeclaration5188 = frozenset([1])
    FOLLOW_LCURLY_in_enumBody5231 = frozenset([7, 53, 70, 81, 85, 86, 87, 90, 91, 94, 98, 102, 164])
    FOLLOW_enumScopeDeclarations_in_enumBody5233 = frozenset([42])
    FOLLOW_RCURLY_in_enumBody5235 = frozenset([1])
    FOLLOW_enumConstants_in_enumScopeDeclarations5272 = frozenset([1, 11, 44])
    FOLLOW_COMMA_in_enumScopeDeclarations5275 = frozenset([1, 44])
    FOLLOW_enumClassScopeDeclarations_in_enumScopeDeclarations5280 = frozenset([1])
    FOLLOW_SEMI_in_enumClassScopeDeclarations5300 = frozenset([1, 7, 23, 25, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_classScopeDeclarations_in_enumClassScopeDeclarations5302 = frozenset([1, 7, 23, 25, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_enumConstant_in_enumConstants5341 = frozenset([1, 11])
    FOLLOW_COMMA_in_enumConstants5344 = frozenset([7, 53, 70, 81, 85, 86, 87, 90, 91, 94, 98, 102, 164])
    FOLLOW_enumConstant_in_enumConstants5347 = frozenset([1, 11])
    FOLLOW_annotationList_in_enumConstant5368 = frozenset([164])
    FOLLOW_IDENT_in_enumConstant5370 = frozenset([1, 23, 25, 29, 68, 75])
    FOLLOW_arguments_in_enumConstant5373 = frozenset([1, 23, 25, 68, 75])
    FOLLOW_classBody_in_enumConstant5376 = frozenset([1])
    FOLLOW_INTERFACE_in_interfaceTypeDeclaration5397 = frozenset([164])
    FOLLOW_IDENT_in_interfaceTypeDeclaration5399 = frozenset([23, 25, 68])
    FOLLOW_genericTypeParameterList_in_interfaceTypeDeclaration5401 = frozenset([23, 25, 68])
    FOLLOW_interfaceExtendsClause_in_interfaceTypeDeclaration5404 = frozenset([23, 25, 68])
    FOLLOW_interfaceBody_in_interfaceTypeDeclaration5407 = frozenset([1])
    FOLLOW_type_in_typeList5453 = frozenset([1, 11])
    FOLLOW_COMMA_in_typeList5456 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_typeList5459 = frozenset([1, 11])
    FOLLOW_LCURLY_in_classBody5480 = frozenset([7, 23, 25, 42, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_classScopeDeclarations_in_classBody5482 = frozenset([7, 23, 25, 42, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_RCURLY_in_classBody5485 = frozenset([1])
    FOLLOW_LCURLY_in_interfaceBody5523 = frozenset([7, 25, 42, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_interfaceScopeDeclarations_in_interfaceBody5525 = frozenset([7, 25, 42, 44, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 101, 102, 164])
    FOLLOW_RCURLY_in_interfaceBody5528 = frozenset([1])
    FOLLOW_block_in_classScopeDeclarations5566 = frozenset([1])
    FOLLOW_STATIC_in_classScopeDeclarations5595 = frozenset([23])
    FOLLOW_block_in_classScopeDeclarations5597 = frozenset([1])
    FOLLOW_modifierList_in_classScopeDeclarations5620 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_genericTypeParameterList_in_classScopeDeclarations5634 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_classScopeDeclarations5653 = frozenset([164])
    FOLLOW_IDENT_in_classScopeDeclarations5655 = frozenset([29])
    FOLLOW_formalParameterList_in_classScopeDeclarations5657 = frozenset([22, 23, 44, 97])
    FOLLOW_arrayDeclaratorList_in_classScopeDeclarations5659 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_classScopeDeclarations5662 = frozenset([23, 44])
    FOLLOW_block_in_classScopeDeclarations5666 = frozenset([1])
    FOLLOW_SEMI_in_classScopeDeclarations5670 = frozenset([1])
    FOLLOW_VOID_in_classScopeDeclarations5732 = frozenset([164])
    FOLLOW_IDENT_in_classScopeDeclarations5734 = frozenset([29])
    FOLLOW_formalParameterList_in_classScopeDeclarations5736 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_classScopeDeclarations5738 = frozenset([23, 44])
    FOLLOW_block_in_classScopeDeclarations5742 = frozenset([1])
    FOLLOW_SEMI_in_classScopeDeclarations5746 = frozenset([1])
    FOLLOW_IDENT_in_classScopeDeclarations5805 = frozenset([29])
    FOLLOW_formalParameterList_in_classScopeDeclarations5807 = frozenset([23, 97])
    FOLLOW_throwsClause_in_classScopeDeclarations5809 = frozenset([23])
    FOLLOW_block_in_classScopeDeclarations5812 = frozenset([1])
    FOLLOW_type_in_classScopeDeclarations5876 = frozenset([164])
    FOLLOW_classFieldDeclaratorList_in_classScopeDeclarations5878 = frozenset([44])
    FOLLOW_SEMI_in_classScopeDeclarations5880 = frozenset([1])
    FOLLOW_typeDeclaration_in_classScopeDeclarations5925 = frozenset([1])
    FOLLOW_SEMI_in_classScopeDeclarations5935 = frozenset([1])
    FOLLOW_modifierList_in_interfaceScopeDeclarations5955 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_genericTypeParameterList_in_interfaceScopeDeclarations5969 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_interfaceScopeDeclarations5988 = frozenset([164])
    FOLLOW_IDENT_in_interfaceScopeDeclarations5990 = frozenset([29])
    FOLLOW_formalParameterList_in_interfaceScopeDeclarations5992 = frozenset([22, 44, 97])
    FOLLOW_arrayDeclaratorList_in_interfaceScopeDeclarations5994 = frozenset([44, 97])
    FOLLOW_throwsClause_in_interfaceScopeDeclarations5997 = frozenset([44])
    FOLLOW_SEMI_in_interfaceScopeDeclarations6000 = frozenset([1])
    FOLLOW_VOID_in_interfaceScopeDeclarations6058 = frozenset([164])
    FOLLOW_IDENT_in_interfaceScopeDeclarations6060 = frozenset([29])
    FOLLOW_formalParameterList_in_interfaceScopeDeclarations6062 = frozenset([44, 97])
    FOLLOW_throwsClause_in_interfaceScopeDeclarations6064 = frozenset([44])
    FOLLOW_SEMI_in_interfaceScopeDeclarations6067 = frozenset([1])
    FOLLOW_type_in_interfaceScopeDeclarations6130 = frozenset([164])
    FOLLOW_interfaceFieldDeclaratorList_in_interfaceScopeDeclarations6132 = frozenset([44])
    FOLLOW_SEMI_in_interfaceScopeDeclarations6134 = frozenset([1])
    FOLLOW_typeDeclaration_in_interfaceScopeDeclarations6179 = frozenset([1])
    FOLLOW_SEMI_in_interfaceScopeDeclarations6189 = frozenset([1])
    FOLLOW_classFieldDeclarator_in_classFieldDeclaratorList6209 = frozenset([1, 11])
    FOLLOW_COMMA_in_classFieldDeclaratorList6212 = frozenset([164])
    FOLLOW_classFieldDeclarator_in_classFieldDeclaratorList6214 = frozenset([1, 11])
    FOLLOW_variableDeclaratorId_in_classFieldDeclarator6253 = frozenset([1, 6])
    FOLLOW_ASSIGN_in_classFieldDeclarator6256 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_variableInitializer_in_classFieldDeclarator6258 = frozenset([1])
    FOLLOW_interfaceFieldDeclarator_in_interfaceFieldDeclaratorList6299 = frozenset([1, 11])
    FOLLOW_COMMA_in_interfaceFieldDeclaratorList6302 = frozenset([164])
    FOLLOW_interfaceFieldDeclarator_in_interfaceFieldDeclaratorList6304 = frozenset([1, 11])
    FOLLOW_variableDeclaratorId_in_interfaceFieldDeclarator6343 = frozenset([6])
    FOLLOW_ASSIGN_in_interfaceFieldDeclarator6345 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_variableInitializer_in_interfaceFieldDeclarator6347 = frozenset([1])
    FOLLOW_IDENT_in_variableDeclaratorId6385 = frozenset([1, 22])
    FOLLOW_arrayDeclaratorList_in_variableDeclaratorId6388 = frozenset([1])
    FOLLOW_arrayInitializer_in_variableInitializer6408 = frozenset([1])
    FOLLOW_expression_in_variableInitializer6418 = frozenset([1])
    FOLLOW_LBRACK_in_arrayDeclarator6437 = frozenset([41])
    FOLLOW_RBRACK_in_arrayDeclarator6439 = frozenset([1])
    FOLLOW_arrayDeclarator_in_arrayDeclaratorList6473 = frozenset([1, 22])
    FOLLOW_LCURLY_in_arrayInitializer6511 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 42, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_variableInitializer_in_arrayInitializer6514 = frozenset([11, 42])
    FOLLOW_COMMA_in_arrayInitializer6517 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_variableInitializer_in_arrayInitializer6519 = frozenset([11, 42])
    FOLLOW_COMMA_in_arrayInitializer6523 = frozenset([42])
    FOLLOW_RCURLY_in_arrayInitializer6528 = frozenset([1])
    FOLLOW_THROWS_in_throwsClause6566 = frozenset([164])
    FOLLOW_qualifiedIdentList_in_throwsClause6568 = frozenset([1])
    FOLLOW_modifier_in_modifierList6605 = frozenset([1, 7, 53, 70, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_PUBLIC_in_modifier6643 = frozenset([1])
    FOLLOW_PROTECTED_in_modifier6653 = frozenset([1])
    FOLLOW_PRIVATE_in_modifier6663 = frozenset([1])
    FOLLOW_STATIC_in_modifier6673 = frozenset([1])
    FOLLOW_ABSTRACT_in_modifier6683 = frozenset([1])
    FOLLOW_NATIVE_in_modifier6693 = frozenset([1])
    FOLLOW_SYNCHRONIZED_in_modifier6703 = frozenset([1])
    FOLLOW_TRANSIENT_in_modifier6713 = frozenset([1])
    FOLLOW_VOLATILE_in_modifier6723 = frozenset([1])
    FOLLOW_STRICTFP_in_modifier6733 = frozenset([1])
    FOLLOW_localModifier_in_modifier6743 = frozenset([1])
    FOLLOW_localModifier_in_localModifierList6762 = frozenset([1, 7, 53, 70, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_FINAL_in_localModifier6799 = frozenset([1])
    FOLLOW_annotation_in_localModifier6809 = frozenset([1])
    FOLLOW_simpleType_in_type6828 = frozenset([1])
    FOLLOW_objectType_in_type6838 = frozenset([1])
    FOLLOW_primitiveType_in_simpleType6858 = frozenset([1, 22])
    FOLLOW_arrayDeclaratorList_in_simpleType6860 = frozenset([1])
    FOLLOW_qualifiedTypeIdent_in_objectType6901 = frozenset([1, 22])
    FOLLOW_arrayDeclaratorList_in_objectType6903 = frozenset([1])
    FOLLOW_qualifiedTypeIdentSimplified_in_objectTypeSimplified6943 = frozenset([1, 22])
    FOLLOW_arrayDeclaratorList_in_objectTypeSimplified6945 = frozenset([1])
    FOLLOW_typeIdent_in_qualifiedTypeIdent6985 = frozenset([1, 15])
    FOLLOW_DOT_in_qualifiedTypeIdent6988 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_typeIdent_in_qualifiedTypeIdent6990 = frozenset([1, 15])
    FOLLOW_typeIdentSimplified_in_qualifiedTypeIdentSimplified7029 = frozenset([1, 15])
    FOLLOW_DOT_in_qualifiedTypeIdentSimplified7032 = frozenset([164])
    FOLLOW_typeIdentSimplified_in_qualifiedTypeIdentSimplified7034 = frozenset([1, 15])
    FOLLOW_IDENT_in_typeIdent7073 = frozenset([1, 25])
    FOLLOW_genericTypeArgumentList_in_typeIdent7076 = frozenset([1])
    FOLLOW_IDENT_in_typeIdentSimplified7096 = frozenset([1, 25])
    FOLLOW_genericTypeArgumentListSimplified_in_typeIdentSimplified7099 = frozenset([1])
    FOLLOW_set_in_primitiveType0 = frozenset([1])
    FOLLOW_LESS_THAN_in_genericTypeArgumentList7208 = frozenset([40, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_genericTypeArgument_in_genericTypeArgumentList7210 = frozenset([8, 11, 20, 47])
    FOLLOW_COMMA_in_genericTypeArgumentList7213 = frozenset([40, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_genericTypeArgument_in_genericTypeArgumentList7215 = frozenset([8, 11, 20, 47])
    FOLLOW_genericTypeListClosing_in_genericTypeArgumentList7219 = frozenset([1])
    FOLLOW_type_in_genericTypeArgument7257 = frozenset([1])
    FOLLOW_QUESTION_in_genericTypeArgument7267 = frozenset([1, 68, 92])
    FOLLOW_genericWildcardBoundType_in_genericTypeArgument7269 = frozenset([1])
    FOLLOW_set_in_genericWildcardBoundType7307 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_genericWildcardBoundType7316 = frozenset([1])
    FOLLOW_LESS_THAN_in_genericTypeArgumentListSimplified7335 = frozenset([40, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_genericTypeArgumentSimplified_in_genericTypeArgumentListSimplified7337 = frozenset([8, 11, 20, 47])
    FOLLOW_COMMA_in_genericTypeArgumentListSimplified7340 = frozenset([40, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_genericTypeArgumentSimplified_in_genericTypeArgumentListSimplified7342 = frozenset([8, 11, 20, 47])
    FOLLOW_genericTypeListClosing_in_genericTypeArgumentListSimplified7346 = frozenset([1])
    FOLLOW_type_in_genericTypeArgumentSimplified7384 = frozenset([1])
    FOLLOW_QUESTION_in_genericTypeArgumentSimplified7394 = frozenset([1])
    FOLLOW_qualifiedIdentifier_in_qualifiedIdentList7413 = frozenset([1, 11])
    FOLLOW_COMMA_in_qualifiedIdentList7416 = frozenset([164])
    FOLLOW_qualifiedIdentifier_in_qualifiedIdentList7419 = frozenset([1, 11])
    FOLLOW_LPAREN_in_formalParameterList7440 = frozenset([7, 43, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterStandardDecl_in_formalParameterList7467 = frozenset([11, 43])
    FOLLOW_COMMA_in_formalParameterList7470 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterStandardDecl_in_formalParameterList7472 = frozenset([11, 43])
    FOLLOW_COMMA_in_formalParameterList7477 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterVarArgDecl_in_formalParameterList7479 = frozenset([43])
    FOLLOW_formalParameterVarArgDecl_in_formalParameterList7534 = frozenset([43])
    FOLLOW_RPAREN_in_formalParameterList7609 = frozenset([1])
    FOLLOW_localModifierList_in_formalParameterStandardDecl7628 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_formalParameterStandardDecl7630 = frozenset([164])
    FOLLOW_variableDeclaratorId_in_formalParameterStandardDecl7632 = frozenset([1])
    FOLLOW_localModifierList_in_formalParameterVarArgDecl7672 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_formalParameterVarArgDecl7674 = frozenset([17])
    FOLLOW_ELLIPSIS_in_formalParameterVarArgDecl7676 = frozenset([164])
    FOLLOW_variableDeclaratorId_in_formalParameterVarArgDecl7678 = frozenset([1])
    FOLLOW_IDENT_in_qualifiedIdentifier7722 = frozenset([1, 15])
    FOLLOW_DOT_in_qualifiedIdentifier7765 = frozenset([164])
    FOLLOW_IDENT_in_qualifiedIdentifier7769 = frozenset([1, 15])
    FOLLOW_annotation_in_annotationList7818 = frozenset([1, 7, 53, 70, 81, 85, 86, 87, 90, 91, 94, 98, 102])
    FOLLOW_AT_in_annotation7856 = frozenset([164])
    FOLLOW_qualifiedIdentifier_in_annotation7859 = frozenset([1, 29])
    FOLLOW_annotationInit_in_annotation7861 = frozenset([1])
    FOLLOW_LPAREN_in_annotationInit7881 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_annotationInitializers_in_annotationInit7883 = frozenset([43])
    FOLLOW_RPAREN_in_annotationInit7885 = frozenset([1])
    FOLLOW_annotationInitializer_in_annotationInitializers7922 = frozenset([1, 11])
    FOLLOW_COMMA_in_annotationInitializers7925 = frozenset([164])
    FOLLOW_annotationInitializer_in_annotationInitializers7927 = frozenset([1, 11])
    FOLLOW_annotationElementValue_in_annotationInitializers7957 = frozenset([1])
    FOLLOW_IDENT_in_annotationInitializer7994 = frozenset([6])
    FOLLOW_ASSIGN_in_annotationInitializer7997 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_annotationElementValue_in_annotationInitializer8000 = frozenset([1])
    FOLLOW_annotationElementValueExpression_in_annotationElementValue8019 = frozenset([1])
    FOLLOW_annotation_in_annotationElementValue8029 = frozenset([1])
    FOLLOW_annotationElementValueArrayInitializer_in_annotationElementValue8039 = frozenset([1])
    FOLLOW_conditionalExpression_in_annotationElementValueExpression8058 = frozenset([1])
    FOLLOW_LCURLY_in_annotationElementValueArrayInitializer8094 = frozenset([7, 11, 12, 21, 23, 25, 27, 29, 30, 34, 38, 42, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_annotationElementValue_in_annotationElementValueArrayInitializer8097 = frozenset([11, 42])
    FOLLOW_COMMA_in_annotationElementValueArrayInitializer8100 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_annotationElementValue_in_annotationElementValueArrayInitializer8102 = frozenset([11, 42])
    FOLLOW_COMMA_in_annotationElementValueArrayInitializer8109 = frozenset([42])
    FOLLOW_RCURLY_in_annotationElementValueArrayInitializer8113 = frozenset([1])
    FOLLOW_AT_in_annotationTypeDeclaration8152 = frozenset([77])
    FOLLOW_INTERFACE_in_annotationTypeDeclaration8154 = frozenset([164])
    FOLLOW_IDENT_in_annotationTypeDeclaration8156 = frozenset([23])
    FOLLOW_annotationBody_in_annotationTypeDeclaration8158 = frozenset([1])
    FOLLOW_LCURLY_in_annotationBody8197 = frozenset([7, 42, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_annotationScopeDeclarations_in_annotationBody8199 = frozenset([7, 42, 53, 55, 57, 60, 61, 65, 67, 70, 72, 77, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_RCURLY_in_annotationBody8202 = frozenset([1])
    FOLLOW_modifierList_in_annotationScopeDeclarations8240 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_annotationScopeDeclarations8242 = frozenset([164])
    FOLLOW_IDENT_in_annotationScopeDeclarations8256 = frozenset([29])
    FOLLOW_LPAREN_in_annotationScopeDeclarations8258 = frozenset([43])
    FOLLOW_RPAREN_in_annotationScopeDeclarations8260 = frozenset([44, 63])
    FOLLOW_annotationDefaultValue_in_annotationScopeDeclarations8262 = frozenset([44])
    FOLLOW_SEMI_in_annotationScopeDeclarations8265 = frozenset([1])
    FOLLOW_classFieldDeclaratorList_in_annotationScopeDeclarations8307 = frozenset([44])
    FOLLOW_SEMI_in_annotationScopeDeclarations8309 = frozenset([1])
    FOLLOW_typeDeclaration_in_annotationScopeDeclarations8354 = frozenset([1])
    FOLLOW_DEFAULT_in_annotationDefaultValue8373 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_annotationElementValue_in_annotationDefaultValue8376 = frozenset([1])
    FOLLOW_LCURLY_in_block8397 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 42, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_blockStatement_in_block8399 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 42, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_RCURLY_in_block8402 = frozenset([1])
    FOLLOW_localVariableDeclaration_in_blockStatement8440 = frozenset([44])
    FOLLOW_SEMI_in_blockStatement8442 = frozenset([1])
    FOLLOW_typeDeclaration_in_blockStatement8453 = frozenset([1])
    FOLLOW_statement_in_blockStatement8463 = frozenset([1])
    FOLLOW_localModifierList_in_localVariableDeclaration8482 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_localVariableDeclaration8484 = frozenset([164])
    FOLLOW_classFieldDeclaratorList_in_localVariableDeclaration8486 = frozenset([1])
    FOLLOW_block_in_statement8527 = frozenset([1])
    FOLLOW_ASSERT_in_statement8537 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_statement8541 = frozenset([10, 44])
    FOLLOW_COLON_in_statement8555 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_statement8559 = frozenset([44])
    FOLLOW_SEMI_in_statement8561 = frozenset([1])
    FOLLOW_SEMI_in_statement8624 = frozenset([1])
    FOLLOW_IF_in_statement8713 = frozenset([29])
    FOLLOW_parenthesizedExpression_in_statement8715 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement8719 = frozenset([1, 66])
    FOLLOW_ELSE_in_statement8733 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement8737 = frozenset([1])
    FOLLOW_FOR_in_statement8900 = frozenset([29])
    FOLLOW_LPAREN_in_statement8902 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 55, 57, 60, 65, 69, 70, 72, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 98, 99, 101, 102, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_forInit_in_statement8916 = frozenset([44])
    FOLLOW_SEMI_in_statement8918 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_forCondition_in_statement8920 = frozenset([44])
    FOLLOW_SEMI_in_statement8922 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 43, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_forUpdater_in_statement8924 = frozenset([43])
    FOLLOW_RPAREN_in_statement8926 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement8928 = frozenset([1])
    FOLLOW_localModifierList_in_statement8962 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_statement8964 = frozenset([164])
    FOLLOW_IDENT_in_statement8966 = frozenset([10])
    FOLLOW_COLON_in_statement8968 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_statement8970 = frozenset([43])
    FOLLOW_RPAREN_in_statement8972 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement8974 = frozenset([1])
    FOLLOW_WHILE_in_statement9088 = frozenset([29])
    FOLLOW_parenthesizedExpression_in_statement9090 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement9092 = frozenset([1])
    FOLLOW_DO_in_statement9141 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement9143 = frozenset([103])
    FOLLOW_WHILE_in_statement9145 = frozenset([29])
    FOLLOW_parenthesizedExpression_in_statement9147 = frozenset([44])
    FOLLOW_SEMI_in_statement9149 = frozenset([1])
    FOLLOW_TRY_in_statement9190 = frozenset([23])
    FOLLOW_block_in_statement9192 = frozenset([59, 71])
    FOLLOW_catches_in_statement9195 = frozenset([1, 59, 71])
    FOLLOW_finallyClause_in_statement9197 = frozenset([1])
    FOLLOW_finallyClause_in_statement9202 = frozenset([1])
    FOLLOW_SWITCH_in_statement9245 = frozenset([29])
    FOLLOW_parenthesizedExpression_in_statement9247 = frozenset([23])
    FOLLOW_LCURLY_in_statement9249 = frozenset([42, 58, 63])
    FOLLOW_switchBlockLabels_in_statement9251 = frozenset([42])
    FOLLOW_RCURLY_in_statement9254 = frozenset([1])
    FOLLOW_SYNCHRONIZED_in_statement9281 = frozenset([29])
    FOLLOW_parenthesizedExpression_in_statement9283 = frozenset([23])
    FOLLOW_block_in_statement9285 = frozenset([1])
    FOLLOW_RETURN_in_statement9331 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_statement9333 = frozenset([44])
    FOLLOW_SEMI_in_statement9336 = frozenset([1])
    FOLLOW_THROW_in_statement9400 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_statement9402 = frozenset([44])
    FOLLOW_SEMI_in_statement9404 = frozenset([1])
    FOLLOW_BREAK_in_statement9469 = frozenset([44, 164])
    FOLLOW_IDENT_in_statement9471 = frozenset([44])
    FOLLOW_SEMI_in_statement9474 = frozenset([1])
    FOLLOW_CONTINUE_in_statement9544 = frozenset([44, 164])
    FOLLOW_IDENT_in_statement9546 = frozenset([44])
    FOLLOW_SEMI_in_statement9549 = frozenset([1])
    FOLLOW_IDENT_in_statement9616 = frozenset([10])
    FOLLOW_COLON_in_statement9618 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_statement9620 = frozenset([1])
    FOLLOW_expression_in_statement9687 = frozenset([44])
    FOLLOW_SEMI_in_statement9689 = frozenset([1])
    FOLLOW_SEMI_in_statement9700 = frozenset([1])
    FOLLOW_catchClause_in_catches9720 = frozenset([1, 59])
    FOLLOW_CATCH_in_catchClause9758 = frozenset([29])
    FOLLOW_LPAREN_in_catchClause9761 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterStandardDecl_in_catchClause9764 = frozenset([43])
    FOLLOW_RPAREN_in_catchClause9766 = frozenset([23])
    FOLLOW_block_in_catchClause9769 = frozenset([1])
    FOLLOW_FINALLY_in_finallyClause9788 = frozenset([23])
    FOLLOW_block_in_finallyClause9790 = frozenset([1])
    FOLLOW_switchCaseLabels_in_switchBlockLabels9839 = frozenset([58, 63])
    FOLLOW_switchDefaultLabel_in_switchBlockLabels9842 = frozenset([58])
    FOLLOW_switchCaseLabels_in_switchBlockLabels9847 = frozenset([1])
    FOLLOW_switchCaseLabel_in_switchCaseLabels9893 = frozenset([1, 58])
    FOLLOW_CASE_in_switchCaseLabel9913 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_switchCaseLabel9916 = frozenset([10])
    FOLLOW_COLON_in_switchCaseLabel9918 = frozenset([1, 7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_blockStatement_in_switchCaseLabel9921 = frozenset([1, 7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_DEFAULT_in_switchDefaultLabel9941 = frozenset([10])
    FOLLOW_COLON_in_switchDefaultLabel9944 = frozenset([1, 7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_blockStatement_in_switchDefaultLabel9947 = frozenset([1, 7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_localVariableDeclaration_in_forInit9967 = frozenset([1])
    FOLLOW_expressionList_in_forInit9989 = frozenset([1])
    FOLLOW_expression_in_forCondition10073 = frozenset([1])
    FOLLOW_expressionList_in_forUpdater10111 = frozenset([1])
    FOLLOW_LPAREN_in_parenthesizedExpression10151 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_parenthesizedExpression10153 = frozenset([43])
    FOLLOW_RPAREN_in_parenthesizedExpression10155 = frozenset([1])
    FOLLOW_expression_in_expressionList10192 = frozenset([1, 11])
    FOLLOW_COMMA_in_expressionList10195 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_expressionList10198 = frozenset([1, 11])
    FOLLOW_assignmentExpression_in_expression10219 = frozenset([1])
    FOLLOW_conditionalExpression_in_assignmentExpression10255 = frozenset([1, 5, 6, 9, 14, 31, 33, 37, 39, 46, 48, 50, 52])
    FOLLOW_ASSIGN_in_assignmentExpression10273 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_PLUS_ASSIGN_in_assignmentExpression10292 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_MINUS_ASSIGN_in_assignmentExpression10311 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_STAR_ASSIGN_in_assignmentExpression10330 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_DIV_ASSIGN_in_assignmentExpression10349 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_AND_ASSIGN_in_assignmentExpression10368 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_OR_ASSIGN_in_assignmentExpression10387 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_XOR_ASSIGN_in_assignmentExpression10406 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_MOD_ASSIGN_in_assignmentExpression10425 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_SHIFT_LEFT_ASSIGN_in_assignmentExpression10444 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_SHIFT_RIGHT_ASSIGN_in_assignmentExpression10463 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_BIT_SHIFT_RIGHT_ASSIGN_in_assignmentExpression10482 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_assignmentExpression_in_assignmentExpression10503 = frozenset([1])
    FOLLOW_logicalOrExpression_in_conditionalExpression10524 = frozenset([1, 40])
    FOLLOW_QUESTION_in_conditionalExpression10527 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_assignmentExpression_in_conditionalExpression10530 = frozenset([10])
    FOLLOW_COLON_in_conditionalExpression10532 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_conditionalExpression_in_conditionalExpression10535 = frozenset([1])
    FOLLOW_logicalAndExpression_in_logicalOrExpression10556 = frozenset([1, 28])
    FOLLOW_LOGICAL_OR_in_logicalOrExpression10559 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_logicalAndExpression_in_logicalOrExpression10562 = frozenset([1, 28])
    FOLLOW_inclusiveOrExpression_in_logicalAndExpression10583 = frozenset([1, 26])
    FOLLOW_LOGICAL_AND_in_logicalAndExpression10586 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_inclusiveOrExpression_in_logicalAndExpression10589 = frozenset([1, 26])
    FOLLOW_exclusiveOrExpression_in_inclusiveOrExpression10610 = frozenset([1, 36])
    FOLLOW_OR_in_inclusiveOrExpression10613 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_exclusiveOrExpression_in_inclusiveOrExpression10616 = frozenset([1, 36])
    FOLLOW_andExpression_in_exclusiveOrExpression10637 = frozenset([1, 51])
    FOLLOW_XOR_in_exclusiveOrExpression10640 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_andExpression_in_exclusiveOrExpression10643 = frozenset([1, 51])
    FOLLOW_equalityExpression_in_andExpression10664 = frozenset([1, 4])
    FOLLOW_AND_in_andExpression10667 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_equalityExpression_in_andExpression10670 = frozenset([1, 4])
    FOLLOW_instanceOfExpression_in_equalityExpression10691 = frozenset([1, 18, 35])
    FOLLOW_EQUAL_in_equalityExpression10709 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_NOT_EQUAL_in_equalityExpression10728 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_instanceOfExpression_in_equalityExpression10757 = frozenset([1, 18, 35])
    FOLLOW_relationalExpression_in_instanceOfExpression10787 = frozenset([1, 76])
    FOLLOW_INSTANCEOF_in_instanceOfExpression10790 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_instanceOfExpression10793 = frozenset([1])
    FOLLOW_shiftExpression_in_relationalExpression10814 = frozenset([1, 19, 20, 24, 25])
    FOLLOW_LESS_OR_EQUAL_in_relationalExpression10832 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_GREATER_OR_EQUAL_in_relationalExpression10851 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_LESS_THAN_in_relationalExpression10870 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_GREATER_THAN_in_relationalExpression10889 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_shiftExpression_in_relationalExpression10918 = frozenset([1, 19, 20, 24, 25])
    FOLLOW_additiveExpression_in_shiftExpression10948 = frozenset([1, 8, 45, 47])
    FOLLOW_BIT_SHIFT_RIGHT_in_shiftExpression10966 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_SHIFT_RIGHT_in_shiftExpression10985 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_SHIFT_LEFT_in_shiftExpression11004 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_additiveExpression_in_shiftExpression11033 = frozenset([1, 8, 45, 47])
    FOLLOW_multiplicativeExpression_in_additiveExpression11063 = frozenset([1, 30, 38])
    FOLLOW_PLUS_in_additiveExpression11081 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_MINUS_in_additiveExpression11100 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_multiplicativeExpression_in_additiveExpression11129 = frozenset([1, 30, 38])
    FOLLOW_unaryExpression_in_multiplicativeExpression11159 = frozenset([1, 13, 32, 49])
    FOLLOW_STAR_in_multiplicativeExpression11177 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_DIV_in_multiplicativeExpression11196 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_MOD_in_multiplicativeExpression11215 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_multiplicativeExpression11244 = frozenset([1, 13, 32, 49])
    FOLLOW_PLUS_in_unaryExpression11274 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_unaryExpression11276 = frozenset([1])
    FOLLOW_MINUS_in_unaryExpression11303 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_unaryExpression11305 = frozenset([1])
    FOLLOW_INC_in_unaryExpression11331 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_postfixedExpression_in_unaryExpression11333 = frozenset([1])
    FOLLOW_DEC_in_unaryExpression11357 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_postfixedExpression_in_unaryExpression11359 = frozenset([1])
    FOLLOW_unaryExpressionNotPlusMinus_in_unaryExpression11383 = frozenset([1])
    FOLLOW_NOT_in_unaryExpressionNotPlusMinus11402 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11404 = frozenset([1])
    FOLLOW_LOGICAL_NOT_in_unaryExpressionNotPlusMinus11451 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11453 = frozenset([1])
    FOLLOW_LPAREN_in_unaryExpressionNotPlusMinus11492 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_unaryExpressionNotPlusMinus11494 = frozenset([43])
    FOLLOW_RPAREN_in_unaryExpressionNotPlusMinus11496 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_unaryExpressionNotPlusMinus11498 = frozenset([1])
    FOLLOW_postfixedExpression_in_unaryExpressionNotPlusMinus11533 = frozenset([1])
    FOLLOW_primaryExpression_in_postfixedExpression11565 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_DOT_in_postfixedExpression11627 = frozenset([25, 82, 92, 95, 164])
    FOLLOW_genericTypeArgumentListSimplified_in_postfixedExpression11649 = frozenset([164])
    FOLLOW_IDENT_in_postfixedExpression11731 = frozenset([1, 12, 15, 21, 22, 29])
    FOLLOW_arguments_in_postfixedExpression11809 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_THIS_in_postfixedExpression11883 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_SUPER_in_postfixedExpression11946 = frozenset([29])
    FOLLOW_arguments_in_postfixedExpression11948 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_SUPER_in_postfixedExpression12001 = frozenset([15])
    FOLLOW_DOT_in_postfixedExpression12005 = frozenset([164])
    FOLLOW_IDENT_in_postfixedExpression12007 = frozenset([1, 12, 15, 21, 22, 29])
    FOLLOW_arguments_in_postfixedExpression12074 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_innerNewExpression_in_postfixedExpression12145 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_LBRACK_in_postfixedExpression12202 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_postfixedExpression12204 = frozenset([41])
    FOLLOW_RBRACK_in_postfixedExpression12206 = frozenset([1, 12, 15, 21, 22])
    FOLLOW_INC_in_postfixedExpression12267 = frozenset([1])
    FOLLOW_DEC_in_postfixedExpression12291 = frozenset([1])
    FOLLOW_parenthesizedExpression_in_primaryExpression12331 = frozenset([1])
    FOLLOW_literal_in_primaryExpression12341 = frozenset([1])
    FOLLOW_newExpression_in_primaryExpression12351 = frozenset([1])
    FOLLOW_qualifiedIdentExpression_in_primaryExpression12361 = frozenset([1])
    FOLLOW_genericTypeArgumentListSimplified_in_primaryExpression12371 = frozenset([92, 95, 164])
    FOLLOW_SUPER_in_primaryExpression12385 = frozenset([15, 29])
    FOLLOW_arguments_in_primaryExpression12403 = frozenset([1])
    FOLLOW_DOT_in_primaryExpression12463 = frozenset([164])
    FOLLOW_IDENT_in_primaryExpression12465 = frozenset([29])
    FOLLOW_arguments_in_primaryExpression12467 = frozenset([1])
    FOLLOW_IDENT_in_primaryExpression12534 = frozenset([29])
    FOLLOW_arguments_in_primaryExpression12536 = frozenset([1])
    FOLLOW_THIS_in_primaryExpression12591 = frozenset([29])
    FOLLOW_arguments_in_primaryExpression12593 = frozenset([1])
    FOLLOW_THIS_in_primaryExpression12658 = frozenset([1, 29])
    FOLLOW_arguments_in_primaryExpression12726 = frozenset([1])
    FOLLOW_SUPER_in_primaryExpression12791 = frozenset([29])
    FOLLOW_arguments_in_primaryExpression12793 = frozenset([1])
    FOLLOW_SUPER_in_primaryExpression12849 = frozenset([15])
    FOLLOW_DOT_in_primaryExpression12851 = frozenset([164])
    FOLLOW_IDENT_in_primaryExpression12853 = frozenset([1, 29])
    FOLLOW_arguments_in_primaryExpression12877 = frozenset([1])
    FOLLOW_primitiveType_in_primaryExpression13019 = frozenset([15, 22])
    FOLLOW_arrayDeclarator_in_primaryExpression13078 = frozenset([15, 22])
    FOLLOW_DOT_in_primaryExpression13137 = frozenset([61])
    FOLLOW_CLASS_in_primaryExpression13139 = frozenset([1])
    FOLLOW_VOID_in_primaryExpression13199 = frozenset([15])
    FOLLOW_DOT_in_primaryExpression13201 = frozenset([61])
    FOLLOW_CLASS_in_primaryExpression13203 = frozenset([1])
    FOLLOW_qualifiedIdentifier_in_qualifiedIdentExpression13279 = frozenset([1, 15, 22, 29])
    FOLLOW_arrayDeclarator_in_qualifiedIdentExpression13349 = frozenset([15, 22])
    FOLLOW_DOT_in_qualifiedIdentExpression13416 = frozenset([61])
    FOLLOW_CLASS_in_qualifiedIdentExpression13418 = frozenset([1])
    FOLLOW_arguments_in_qualifiedIdentExpression13488 = frozenset([1])
    FOLLOW_DOT_in_qualifiedIdentExpression13549 = frozenset([25, 61, 82, 92, 95, 164])
    FOLLOW_CLASS_in_qualifiedIdentExpression13567 = frozenset([1])
    FOLLOW_genericTypeArgumentListSimplified_in_qualifiedIdentExpression13630 = frozenset([92, 164])
    FOLLOW_SUPER_in_qualifiedIdentExpression13654 = frozenset([29])
    FOLLOW_arguments_in_qualifiedIdentExpression13656 = frozenset([1])
    FOLLOW_SUPER_in_qualifiedIdentExpression13706 = frozenset([15])
    FOLLOW_DOT_in_qualifiedIdentExpression13710 = frozenset([164])
    FOLLOW_IDENT_in_qualifiedIdentExpression13712 = frozenset([29])
    FOLLOW_arguments_in_qualifiedIdentExpression13714 = frozenset([1])
    FOLLOW_IDENT_in_qualifiedIdentExpression13764 = frozenset([29])
    FOLLOW_arguments_in_qualifiedIdentExpression13766 = frozenset([1])
    FOLLOW_THIS_in_qualifiedIdentExpression13841 = frozenset([1])
    FOLLOW_SUPER_in_qualifiedIdentExpression13907 = frozenset([29])
    FOLLOW_arguments_in_qualifiedIdentExpression13909 = frozenset([1])
    FOLLOW_innerNewExpression_in_qualifiedIdentExpression13957 = frozenset([1])
    FOLLOW_NEW_in_newExpression14033 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_primitiveType_in_newExpression14047 = frozenset([22])
    FOLLOW_newArrayConstruction_in_newExpression14049 = frozenset([1])
    FOLLOW_genericTypeArgumentListSimplified_in_newExpression14093 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_qualifiedTypeIdentSimplified_in_newExpression14096 = frozenset([22, 29])
    FOLLOW_newArrayConstruction_in_newExpression14114 = frozenset([1])
    FOLLOW_arguments_in_newExpression14179 = frozenset([1, 23, 25, 68, 75])
    FOLLOW_classBody_in_newExpression14181 = frozenset([1])
    FOLLOW_NEW_in_innerNewExpression14276 = frozenset([25, 164])
    FOLLOW_genericTypeArgumentListSimplified_in_innerNewExpression14278 = frozenset([164])
    FOLLOW_IDENT_in_innerNewExpression14281 = frozenset([29])
    FOLLOW_arguments_in_innerNewExpression14283 = frozenset([1, 23, 25, 68, 75])
    FOLLOW_classBody_in_innerNewExpression14285 = frozenset([1])
    FOLLOW_arrayDeclaratorList_in_newArrayConstruction14331 = frozenset([23])
    FOLLOW_arrayInitializer_in_newArrayConstruction14333 = frozenset([1])
    FOLLOW_LBRACK_in_newArrayConstruction14343 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_newArrayConstruction14346 = frozenset([41])
    FOLLOW_RBRACK_in_newArrayConstruction14348 = frozenset([1, 22])
    FOLLOW_LBRACK_in_newArrayConstruction14352 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_newArrayConstruction14355 = frozenset([41])
    FOLLOW_RBRACK_in_newArrayConstruction14357 = frozenset([1, 22])
    FOLLOW_arrayDeclaratorList_in_newArrayConstruction14362 = frozenset([1])
    FOLLOW_LPAREN_in_arguments14382 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 43, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expressionList_in_arguments14384 = frozenset([43])
    FOLLOW_RPAREN_in_arguments14387 = frozenset([1])
    FOLLOW_set_in_literal0 = frozenset([1])
    FOLLOW_GREATER_THAN_in_synpred14_Java5045 = frozenset([1])
    FOLLOW_SHIFT_RIGHT_in_synpred15_Java5055 = frozenset([1])
    FOLLOW_BIT_SHIFT_RIGHT_in_synpred16_Java5065 = frozenset([1])
    FOLLOW_bound_in_synpred17_Java5095 = frozenset([1])
    FOLLOW_STATIC_in_synpred32_Java5595 = frozenset([23])
    FOLLOW_block_in_synpred32_Java5597 = frozenset([1])
    FOLLOW_genericTypeParameterList_in_synpred42_Java5634 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_synpred42_Java5653 = frozenset([164])
    FOLLOW_IDENT_in_synpred42_Java5655 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred42_Java5657 = frozenset([22, 23, 44, 97])
    FOLLOW_arrayDeclaratorList_in_synpred42_Java5659 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_synpred42_Java5662 = frozenset([23, 44])
    FOLLOW_block_in_synpred42_Java5666 = frozenset([1])
    FOLLOW_SEMI_in_synpred42_Java5670 = frozenset([1])
    FOLLOW_VOID_in_synpred42_Java5732 = frozenset([164])
    FOLLOW_IDENT_in_synpred42_Java5734 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred42_Java5736 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_synpred42_Java5738 = frozenset([23, 44])
    FOLLOW_block_in_synpred42_Java5742 = frozenset([1])
    FOLLOW_SEMI_in_synpred42_Java5746 = frozenset([1])
    FOLLOW_IDENT_in_synpred42_Java5805 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred42_Java5807 = frozenset([23, 97])
    FOLLOW_throwsClause_in_synpred42_Java5809 = frozenset([23])
    FOLLOW_block_in_synpred42_Java5812 = frozenset([1])
    FOLLOW_modifierList_in_synpred43_Java5620 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_genericTypeParameterList_in_synpred43_Java5634 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_synpred43_Java5653 = frozenset([164])
    FOLLOW_IDENT_in_synpred43_Java5655 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred43_Java5657 = frozenset([22, 23, 44, 97])
    FOLLOW_arrayDeclaratorList_in_synpred43_Java5659 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_synpred43_Java5662 = frozenset([23, 44])
    FOLLOW_block_in_synpred43_Java5666 = frozenset([1])
    FOLLOW_SEMI_in_synpred43_Java5670 = frozenset([1])
    FOLLOW_VOID_in_synpred43_Java5732 = frozenset([164])
    FOLLOW_IDENT_in_synpred43_Java5734 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred43_Java5736 = frozenset([23, 44, 97])
    FOLLOW_throwsClause_in_synpred43_Java5738 = frozenset([23, 44])
    FOLLOW_block_in_synpred43_Java5742 = frozenset([1])
    FOLLOW_SEMI_in_synpred43_Java5746 = frozenset([1])
    FOLLOW_IDENT_in_synpred43_Java5805 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred43_Java5807 = frozenset([23, 97])
    FOLLOW_throwsClause_in_synpred43_Java5809 = frozenset([23])
    FOLLOW_block_in_synpred43_Java5812 = frozenset([1])
    FOLLOW_type_in_synpred43_Java5876 = frozenset([164])
    FOLLOW_classFieldDeclaratorList_in_synpred43_Java5878 = frozenset([44])
    FOLLOW_SEMI_in_synpred43_Java5880 = frozenset([1])
    FOLLOW_typeDeclaration_in_synpred44_Java5925 = frozenset([1])
    FOLLOW_genericTypeParameterList_in_synpred50_Java5969 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_synpred50_Java5988 = frozenset([164])
    FOLLOW_IDENT_in_synpred50_Java5990 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred50_Java5992 = frozenset([22, 44, 97])
    FOLLOW_arrayDeclaratorList_in_synpred50_Java5994 = frozenset([44, 97])
    FOLLOW_throwsClause_in_synpred50_Java5997 = frozenset([44])
    FOLLOW_SEMI_in_synpred50_Java6000 = frozenset([1])
    FOLLOW_VOID_in_synpred50_Java6058 = frozenset([164])
    FOLLOW_IDENT_in_synpred50_Java6060 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred50_Java6062 = frozenset([44, 97])
    FOLLOW_throwsClause_in_synpred50_Java6064 = frozenset([44])
    FOLLOW_SEMI_in_synpred50_Java6067 = frozenset([1])
    FOLLOW_modifierList_in_synpred51_Java5955 = frozenset([25, 55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_genericTypeParameterList_in_synpred51_Java5969 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 101, 164])
    FOLLOW_type_in_synpred51_Java5988 = frozenset([164])
    FOLLOW_IDENT_in_synpred51_Java5990 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred51_Java5992 = frozenset([22, 44, 97])
    FOLLOW_arrayDeclaratorList_in_synpred51_Java5994 = frozenset([44, 97])
    FOLLOW_throwsClause_in_synpred51_Java5997 = frozenset([44])
    FOLLOW_SEMI_in_synpred51_Java6000 = frozenset([1])
    FOLLOW_VOID_in_synpred51_Java6058 = frozenset([164])
    FOLLOW_IDENT_in_synpred51_Java6060 = frozenset([29])
    FOLLOW_formalParameterList_in_synpred51_Java6062 = frozenset([44, 97])
    FOLLOW_throwsClause_in_synpred51_Java6064 = frozenset([44])
    FOLLOW_SEMI_in_synpred51_Java6067 = frozenset([1])
    FOLLOW_type_in_synpred51_Java6130 = frozenset([164])
    FOLLOW_interfaceFieldDeclaratorList_in_synpred51_Java6132 = frozenset([44])
    FOLLOW_SEMI_in_synpred51_Java6134 = frozenset([1])
    FOLLOW_typeDeclaration_in_synpred52_Java6179 = frozenset([1])
    FOLLOW_arrayDeclarator_in_synpred58_Java6473 = frozenset([1])
    FOLLOW_arrayDeclaratorList_in_synpred76_Java6860 = frozenset([1])
    FOLLOW_arrayDeclaratorList_in_synpred77_Java6903 = frozenset([1])
    FOLLOW_DOT_in_synpred79_Java6988 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_typeIdent_in_synpred79_Java6990 = frozenset([1])
    FOLLOW_COMMA_in_synpred90_Java7213 = frozenset([40, 55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_genericTypeArgument_in_synpred90_Java7215 = frozenset([1])
    FOLLOW_genericWildcardBoundType_in_synpred92_Java7269 = frozenset([1])
    FOLLOW_COMMA_in_synpred97_Java7470 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterStandardDecl_in_synpred97_Java7472 = frozenset([1])
    FOLLOW_formalParameterStandardDecl_in_synpred99_Java7467 = frozenset([1, 11])
    FOLLOW_COMMA_in_synpred99_Java7470 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterStandardDecl_in_synpred99_Java7472 = frozenset([1, 11])
    FOLLOW_COMMA_in_synpred99_Java7477 = frozenset([7, 53, 55, 57, 60, 65, 70, 72, 79, 80, 81, 85, 86, 87, 89, 90, 91, 94, 98, 102, 164])
    FOLLOW_formalParameterVarArgDecl_in_synpred99_Java7479 = frozenset([1])
    FOLLOW_formalParameterVarArgDecl_in_synpred100_Java7534 = frozenset([1])
    FOLLOW_DOT_in_synpred101_Java7765 = frozenset([164])
    FOLLOW_IDENT_in_synpred101_Java7769 = frozenset([1])
    FOLLOW_annotation_in_synpred102_Java7818 = frozenset([1])
    FOLLOW_modifierList_in_synpred114_Java8240 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_synpred114_Java8242 = frozenset([164])
    FOLLOW_IDENT_in_synpred114_Java8256 = frozenset([29])
    FOLLOW_LPAREN_in_synpred114_Java8258 = frozenset([43])
    FOLLOW_RPAREN_in_synpred114_Java8260 = frozenset([44, 63])
    FOLLOW_annotationDefaultValue_in_synpred114_Java8262 = frozenset([44])
    FOLLOW_SEMI_in_synpred114_Java8265 = frozenset([1])
    FOLLOW_classFieldDeclaratorList_in_synpred114_Java8307 = frozenset([44])
    FOLLOW_SEMI_in_synpred114_Java8309 = frozenset([1])
    FOLLOW_localVariableDeclaration_in_synpred116_Java8440 = frozenset([44])
    FOLLOW_SEMI_in_synpred116_Java8442 = frozenset([1])
    FOLLOW_typeDeclaration_in_synpred117_Java8453 = frozenset([1])
    FOLLOW_ELSE_in_synpred121_Java8733 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_synpred121_Java8737 = frozenset([1])
    FOLLOW_forInit_in_synpred123_Java8916 = frozenset([44])
    FOLLOW_SEMI_in_synpred123_Java8918 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_forCondition_in_synpred123_Java8920 = frozenset([44])
    FOLLOW_SEMI_in_synpred123_Java8922 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 43, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_forUpdater_in_synpred123_Java8924 = frozenset([43])
    FOLLOW_RPAREN_in_synpred123_Java8926 = frozenset([7, 12, 21, 23, 25, 27, 29, 30, 34, 38, 44, 53, 54, 55, 56, 57, 60, 61, 62, 64, 65, 67, 69, 70, 72, 73, 74, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_statement_in_synpred123_Java8928 = frozenset([1])
    FOLLOW_switchBlockLabels_in_synpred130_Java9251 = frozenset([1])
    FOLLOW_switchCaseLabels_in_synpred143_Java9839 = frozenset([1])
    FOLLOW_switchCaseLabels_in_synpred145_Java9847 = frozenset([1])
    FOLLOW_switchCaseLabel_in_synpred146_Java9893 = frozenset([1])
    FOLLOW_localVariableDeclaration_in_synpred149_Java9967 = frozenset([1])
    FOLLOW_expressionList_in_synpred150_Java9989 = frozenset([1])
    FOLLOW_LPAREN_in_synpred193_Java11492 = frozenset([55, 57, 60, 65, 72, 79, 80, 89, 164])
    FOLLOW_type_in_synpred193_Java11494 = frozenset([43])
    FOLLOW_RPAREN_in_synpred193_Java11496 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_unaryExpression_in_synpred193_Java11498 = frozenset([1])
    FOLLOW_arrayDeclarator_in_synpred221_Java13349 = frozenset([15, 22])
    FOLLOW_DOT_in_synpred221_Java13416 = frozenset([61])
    FOLLOW_CLASS_in_synpred221_Java13418 = frozenset([1])
    FOLLOW_DOT_in_synpred229_Java13549 = frozenset([25, 61, 82, 92, 95, 164])
    FOLLOW_CLASS_in_synpred229_Java13567 = frozenset([1])
    FOLLOW_genericTypeArgumentListSimplified_in_synpred229_Java13630 = frozenset([92, 164])
    FOLLOW_SUPER_in_synpred229_Java13654 = frozenset([29])
    FOLLOW_arguments_in_synpred229_Java13656 = frozenset([1])
    FOLLOW_SUPER_in_synpred229_Java13706 = frozenset([15])
    FOLLOW_DOT_in_synpred229_Java13710 = frozenset([164])
    FOLLOW_IDENT_in_synpred229_Java13712 = frozenset([29])
    FOLLOW_arguments_in_synpred229_Java13714 = frozenset([1])
    FOLLOW_IDENT_in_synpred229_Java13764 = frozenset([29])
    FOLLOW_arguments_in_synpred229_Java13766 = frozenset([1])
    FOLLOW_THIS_in_synpred229_Java13841 = frozenset([1])
    FOLLOW_SUPER_in_synpred229_Java13907 = frozenset([29])
    FOLLOW_arguments_in_synpred229_Java13909 = frozenset([1])
    FOLLOW_innerNewExpression_in_synpred229_Java13957 = frozenset([1])
    FOLLOW_LBRACK_in_synpred237_Java14352 = frozenset([12, 21, 23, 25, 27, 29, 30, 34, 38, 55, 57, 60, 65, 69, 72, 79, 80, 82, 83, 89, 92, 95, 99, 101, 164, 165, 166, 167, 168, 169, 170])
    FOLLOW_expression_in_synpred237_Java14355 = frozenset([41])
    FOLLOW_RBRACK_in_synpred237_Java14357 = frozenset([1])



def main(argv, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    from antlr3.main import ParserMain
    main = ParserMain("JavaLexer", JavaParser)
    main.stdin = stdin
    main.stdout = stdout
    main.stderr = stderr
    main.execute(argv)


if __name__ == '__main__':
    main(sys.argv)
