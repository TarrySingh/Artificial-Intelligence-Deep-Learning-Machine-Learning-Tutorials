// @@ANTLR Tool Options@@: -trace
tree grammar t049treeparsergWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : ^(ID INT?) SEMI
    {this.capture($ID);}
  ;
