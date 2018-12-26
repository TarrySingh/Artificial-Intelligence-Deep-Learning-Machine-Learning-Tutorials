// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTmWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTm;
}
a : ^(x=b INT) ;
b : ID ;
