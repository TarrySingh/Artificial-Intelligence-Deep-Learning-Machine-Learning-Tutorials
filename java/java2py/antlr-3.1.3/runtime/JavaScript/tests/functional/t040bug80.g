lexer grammar t040bug80; 
options {
  language = JavaScript;
}
 
ID_LIKE
    : 'defined' 
    | {false}? Identifier 
    | Identifier 
    ; 
 
fragment 
Identifier: 'a'..'z'+ ; // with just 'a', output compiles 
