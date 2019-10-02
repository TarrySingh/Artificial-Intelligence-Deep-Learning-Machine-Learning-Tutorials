grammar SymbolTable;

/* Scope of symbol names.  Both globals and block rules need to push a new
 * symbol table upon entry and they must use the same stack.  So, I must
 * define a global scope and say that globals and block use this by saying
 * 'scope Symbols;' in those rule definitions.
 */

options {
	language=ObjC;
}

scope Symbols {
  NSMutableArray *names;
}

@ivars {
int level;
}

@init {
level = 0;
}

prog:   globals (method)*
    ;

globals
scope Symbols;
@init {
    level++;
    $Symbols::names = [[NSMutableArray alloc] init];
}
    :   (decl)*
        {
        NSLog(@"globals: \%@", $Symbols::names);
        level--;
        }
    ;

method
    :   'method' ID '(' ')' block
    ;

block
scope Symbols;
@init {
    level++;
    $Symbols::names = [[NSMutableArray alloc] init];
}
    :   '{' (decl)* (stat)* '}'
        {
        NSLog(@"level \%d symbols: \%@", level, $Symbols::names);
        level--;
        }
    ;

stat:   ID '=' INT ';'
    |   block
    ;

decl:   'int' ID ';'
        {[$Symbols::names addObject:$ID];} // add to current symbol table
    ;

ID  :   ('a'..'z')+
    ;

INT :   ('0'..'9')+
    ;

WS  :   (' '|'\n'|'\r')+ {$channel=99;}
    ;
