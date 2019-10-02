// @@ANTLR Tool Options@@: -trace
tree grammar t051treeRewriteASTabWalker;
options {
    language=JavaScript;
    output=AST;
    ASTLabelType=CommonTree;
    tokenVocab=t051treeRewriteASTab;
    rewrite=true;
}
s : ^(ID a) { this.buf += $s.start.toStringTree() };
a : ^(ID INT) -> {true}? ^(ID["ick"] INT)
              -> INT
  ;
