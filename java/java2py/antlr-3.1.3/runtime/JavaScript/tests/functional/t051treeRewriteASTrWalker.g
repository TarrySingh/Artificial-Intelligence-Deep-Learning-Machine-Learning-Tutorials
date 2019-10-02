// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTrWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTr;
}
a : ^(ID (ID | INT) ) ;
