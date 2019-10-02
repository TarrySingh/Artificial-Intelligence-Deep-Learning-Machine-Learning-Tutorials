// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTzWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTz;
    rewrite=true;
}
tokens { X; }
s : ^('boo' a* b) ; // don't reset s.tree to b.tree due to 'boo'
a : X ;
b : ^(ID INT) -> INT
  ;
