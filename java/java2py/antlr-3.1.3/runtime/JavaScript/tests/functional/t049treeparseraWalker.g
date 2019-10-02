// @@ANTLR Tool Options@@: -trace
tree grammar t049treeparseraWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : ID INT
    {this.capture($ID+", "+$INT);}
  ;
