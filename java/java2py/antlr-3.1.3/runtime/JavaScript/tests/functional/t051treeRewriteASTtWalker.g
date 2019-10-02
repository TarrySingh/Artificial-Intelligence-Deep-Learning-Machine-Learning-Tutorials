// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTtWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTt;
    rewrite=true;
}
a : ^(ID INT) -> ^(ID["ick"] INT)
  | INT // leaves it alone, returning $a.start
  ;
