grammar t032subrulePredict;
options {
  language = Python;
}

a: 'BEGIN' b WS+ 'END';
b: ( WS+ 'A' )+;
WS: ' ';
