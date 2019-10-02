// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASToWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTo;
}
a : ^(ID ^(ID INT))
  ;
