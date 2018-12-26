grammar t052importM1;
options {
    language=JavaScript;
}
import t052importS1;
s : a ;
B : 'b' ; // defines B from inherited token space
WS : (' '|'\n') {this.skip();} ;
