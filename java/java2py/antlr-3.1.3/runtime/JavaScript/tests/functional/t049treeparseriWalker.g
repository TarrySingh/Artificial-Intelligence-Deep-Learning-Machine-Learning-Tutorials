tree grammar t049treeparseriWalker;
options {
    language=JavaScript;
    ASTLabelType=CommonTree;
}
a @init {var x=0;} : ^(ID {x=1;} {x=2;} INT?)
    {this.capture($ID+", "+x);}
  ;
