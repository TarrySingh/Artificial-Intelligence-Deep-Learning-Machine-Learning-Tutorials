// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTbWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTb;
}
a : ^(ID INT) -> ^(INT ID);
