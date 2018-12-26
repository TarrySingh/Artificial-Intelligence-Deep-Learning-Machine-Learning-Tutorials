grammar TreeRewrite;

options {
	output=AST;
	language=ObjC;
}

rule:	INT subrule -> ^(subrule INT)
	;
	
subrule
    :   INT
    ;
    
INT	:	('0'..'9')+
	;

WS  :   ' ' {$channel=99;}
    ;
