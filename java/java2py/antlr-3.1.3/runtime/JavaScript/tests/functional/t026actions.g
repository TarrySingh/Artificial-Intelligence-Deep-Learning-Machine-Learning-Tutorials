grammar t026actions;
options {
  language = JavaScript;
}

@lexer::init {
    this.xlog = [];
    this.foobar = "attribute;";
}

prog
@init {
    this.xlog = [];
    this.xlog.push("init;");
}
@after {
    this.xlog.push("after;");
}
    :   IDENTIFIER EOF
    ;
    catch [ exc ] {
        this.xlog.push("catch;");
        throw new Error();
    }
    finally {
        this.xlog.push("finally;");
    }


IDENTIFIER
    : ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
        {
          // a comment
          this.xlog.push("action;");
          this.xlog.push([$text, $type, $line, $pos, $index, $channel, $start, $stop].join(" "));
          if (true)
              this.xlog.push(this.foobar);
        }
    ;

WS: (' ' | '\n')+;
