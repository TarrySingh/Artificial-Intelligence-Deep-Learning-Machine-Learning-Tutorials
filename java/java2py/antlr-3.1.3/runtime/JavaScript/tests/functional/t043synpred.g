grammar t043synpred;
options {
  language = JavaScript;
}

a: ((s+ P)=> s+ b)? E;
b: P 'foo';

s: S;


S: ' ';
P: '+';
E: '>';
