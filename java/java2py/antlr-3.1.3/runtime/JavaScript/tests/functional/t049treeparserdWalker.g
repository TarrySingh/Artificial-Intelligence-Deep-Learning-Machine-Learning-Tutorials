tree grammar t049treeparserdWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a : b b ;
b : ID INT+    {this.capture($ID+" "+$INT+"\n");}
  | ^(x=ID (y=INT)+) {this.capture("^("+$x+' '+$y+")");}
  ;
