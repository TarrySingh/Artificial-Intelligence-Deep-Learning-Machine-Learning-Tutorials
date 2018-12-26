grammar t032subrulePredict;
options {
  language = JavaScript;
}

a: 'BEGIN' b WS+ 'END';
b: ( WS+ 'A' )+;
WS: ' ';
