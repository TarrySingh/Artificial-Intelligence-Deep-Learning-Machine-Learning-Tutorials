lexer grammar t029synpredgate;
options {
  language = Python;
}

FOO
    : ('ab')=> A
    | ('ac')=> B
    ;

fragment
A: 'a';

fragment
B: 'a';

