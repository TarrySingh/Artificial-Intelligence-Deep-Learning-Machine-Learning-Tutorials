lexer grammar t007lexer;
options {
  language = JavaScript;
}

FOO: 'f' ('o' | 'a' 'b'+)*;
