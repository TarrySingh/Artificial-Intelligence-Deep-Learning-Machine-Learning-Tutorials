// @@ANTLR Tool Options@@: -trace
tree grammar t049treeparserfWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : ^(ID INT?)
    {this.capture($ID);}
  ;
