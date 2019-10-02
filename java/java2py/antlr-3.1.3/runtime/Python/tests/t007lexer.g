lexer grammar t007lexer;
options {
  language = Python;
}

FOO: 'f' ('o' | 'a' 'b'+)*;
