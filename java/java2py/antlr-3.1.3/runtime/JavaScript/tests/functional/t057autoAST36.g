grammar t057autoAST36;
options {language=JavaScript;output=AST;}
a returns [result] : ids+=ID ids+=ID {
    var p, buffer=[];
    for (p=0; p<$ids.length; p++) {
        buffer.push($ids[p]);
    }
    $result = "id list=["+buffer.join(",")+"],";
} ;
ID : 'a'..'z'+ ;
INT : '0'..'9'+;
WS : (' '|'\n') {$channel=HIDDEN;} ;
