// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTeWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTe;
}
a : (^(ID INT))+ -> INT+ ID+;
