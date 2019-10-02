grammar t015calc;
options {
  language = Python;
}

@header {
import math
}

@parser::init {
self.reportedErrors = []
}

@parser::members {
def emitErrorMessage(self, msg):
    self.reportedErrors.append(msg)
}

evaluate returns [result]: r=expression {result = r};

expression returns [result]: r=mult (
    '+' r2=mult {r += r2}
  | '-' r2=mult {r -= r2}
  )* {result = r};

mult returns [result]: r=log (
    '*' r2=log {r *= r2}
  | '/' r2=log {r /= r2}
//  | '%' r2=log {r %= r2}
  )* {result = r};

log returns [result]: 'ln' r=exp {result = math.log(r)}
    | r=exp {result = r}
    ;

exp returns [result]: r=atom ('^' r2=atom {r = math.pow(r,r2)} )? {result = r}
    ;

atom returns [result]:
    n=INTEGER {result = int($n.text)}
  | n=DECIMAL {result = float($n.text)} 
  | '(' r=expression {result = r} ')'
  | 'PI' {result = math.pi}
  | 'E' {result = math.e}
  ;

INTEGER: DIGIT+;

DECIMAL: DIGIT+ '.' DIGIT+;

fragment
DIGIT: '0'..'9';

WS: (' ' | '\n' | '\t')+ {$channel = HIDDEN};
