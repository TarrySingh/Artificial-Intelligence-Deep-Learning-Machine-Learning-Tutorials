// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTdWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTd;
}

a : ID -> ^(ID ID);
