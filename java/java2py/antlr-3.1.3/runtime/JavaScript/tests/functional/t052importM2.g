grammar t052importM2;
options {
    language=JavaScript;
}
import t052importS2;
s : label=a[3] {this.capture($label.y);} ;
B : 'b' ; // defines B from inherited token space
WS : (' '|'\n') {this.skip();} ;
