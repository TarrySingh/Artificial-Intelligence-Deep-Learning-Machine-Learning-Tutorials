lexer grammar TreeRewrite;
options {
  language=ObjC;

}

// $ANTLR src "TreeRewrite.g" 15
INT	:	('0'..'9')+
	;

// $ANTLR src "TreeRewrite.g" 18
WS  :   ' ' {$channel=99;}
    ;
