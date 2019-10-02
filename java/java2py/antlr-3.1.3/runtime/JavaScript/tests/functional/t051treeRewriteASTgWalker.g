// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTgWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTg;
}
a : b c ;
b : ID ;
c : INT ;
