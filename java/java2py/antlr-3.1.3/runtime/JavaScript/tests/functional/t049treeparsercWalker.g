tree grammar t049treeparsercWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : b b ;
b : ID INT    {this.capture($ID+" "+$INT+"\n");}
  | ^(ID INT) {this.capture("^("+$ID+" "+$INT+")");}
  ;
