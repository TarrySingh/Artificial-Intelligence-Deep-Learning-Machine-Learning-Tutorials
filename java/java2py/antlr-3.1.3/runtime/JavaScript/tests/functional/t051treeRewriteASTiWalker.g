// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTiWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTi;
}
a : ^(ID INT)
  ;
