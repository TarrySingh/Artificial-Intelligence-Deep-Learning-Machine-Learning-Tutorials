lexer grammar t056lexer14;
options {language=JavaScript;}
QUOTED_CONTENT 
        : 'q' (~'q')* (('x' 'q') )* 'q' ;
