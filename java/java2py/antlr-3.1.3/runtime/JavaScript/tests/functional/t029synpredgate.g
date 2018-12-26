lexer grammar t029synpredgate;
options {
  language = JavaScript;
}

FOO
    : ('ab')=>A
    | ('ac')=>B
    ;

fragment
A: 'a';

fragment
B: 'a';

