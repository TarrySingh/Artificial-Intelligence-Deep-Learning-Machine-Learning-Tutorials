// @@ANTLR Tool Options@@: -trace
tree grammar t049treeparsereWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : ID INT+ PERIOD {this.capture("alt 1");}
  | ID INT+ SEMI   {this.capture("alt 2");}
  ;
