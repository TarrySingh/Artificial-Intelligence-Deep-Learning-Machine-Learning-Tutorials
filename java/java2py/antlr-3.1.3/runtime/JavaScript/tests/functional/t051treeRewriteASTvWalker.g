// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTvWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTv;
    rewrite=true;
}
s : a ;
a : b ;
b : ID INT -> INT ID
  ;
