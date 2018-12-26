/** A lexer is recognizer that draws input symbols from a character stream.
 *  lexer grammars result in a subclass of this object. A Lexer object
 *  uses simplified match() and error recovery mechanisms in the interest
 *  of speed.
 */
org.antlr.runtime.Lexer = function(input, state) {
    if (state) {
        org.antlr.runtime.Lexer.superclass.constructor.call(this, state);
    }
    if (input) {
        this.input = input;
    }
};

org.antlr.lang.extend(org.antlr.runtime.Lexer, org.antlr.runtime.BaseRecognizer, {
    reset: function() {
        // reset all recognizer state variables
        org.antlr.runtime.Lexer.superclass.reset.call(this);
        if ( org.antlr.lang.isValue(this.input) ) {
            this.input.seek(0); // rewind the input
        }
        if ( !org.antlr.lang.isValue(this.state) ) {
            return; // no shared state work to do
        }
        this.state.token = null;
        this.state.type = org.antlr.runtime.Token.INVALID_TOKEN_TYPE;
        this.state.channel = org.antlr.runtime.Token.DEFAULT_CHANNEL;
        this.state.tokenStartCharIndex = -1;
        this.state.tokenStartCharPositionInLine = -1;
        this.state.tokenStartLine = -1;
        this.state.text = null;
    },

    /** Return a token from this source; i.e., match a token on the char
     *  stream.
     */
    nextToken: function() {
        while (true) {
            this.state.token = null;
            this.state.channel = org.antlr.runtime.Token.DEFAULT_CHANNEL;
            this.state.tokenStartCharIndex = this.input.index();
            this.state.tokenStartCharPositionInLine = this.input.getCharPositionInLine();
            this.state.tokenStartLine = this.input.getLine();
            this.state.text = null;
            if ( this.input.LA(1)===org.antlr.runtime.CharStream.EOF ) {
                return org.antlr.runtime.Token.EOF_TOKEN;
            }
            try {
                this.mTokens();
                if ( !org.antlr.lang.isValue(this.state.token) ) {
                    this.emit();
                }
                else if ( this.state.token==org.antlr.runtime.Token.SKIP_TOKEN ) {
                    continue;
                }
                return this.state.token;
            }
            catch (re) {
                if ( re instanceof org.antlr.runtime.RecognitionException ) {
                    this.reportError(re);
                } else if (re instanceof org.antlr.runtime.NoViableAltException) {
                    this.reportError(re);
                    this.recover(re);
                } else {
                    throw re;
                }
            }
        }
    },

    /** Instruct the lexer to skip creating a token for current lexer rule
     *  and look for another token.  nextToken() knows to keep looking when
     *  a lexer rule finishes with token set to SKIP_TOKEN.  Recall that
     *  if token==null at end of any token rule, it creates one for you
     *  and emits it.
     */
    skip: function() {
        this.state.token = org.antlr.runtime.Token.SKIP_TOKEN;
    },

    /** Set the char stream and reset the lexer */
    setCharStream: function(input) {
        this.input = null;
        this.reset();
        this.input = input;
    },

    getCharStream: function() {
        return this.input;
    },

    getSourceName: function() {
        return this.input.getSourceName();
    },

    /** Currently does not support multiple emits per nextToken invocation
     *  for efficiency reasons.  Subclass and override this method and
     *  nextToken (to push tokens into a list and pull from that list rather
     *  than a single variable as this implementation does).
     *
     *  The standard method called to automatically emit a token at the
     *  outermost lexical rule.  The token object should point into the
     *  char buffer start..stop.  If there is a text override in 'text',
     *  use that to set the token's text.  Override this method to emit
     *  custom Token objects.
     *
     *  If you are building trees, then you should also override
     *  Parser or TreeParser.getMissingSymbol().
     */
    emit: function() {
        if (arguments.length===0) {
            var t = new org.antlr.runtime.CommonToken(this.input, this.state.type, this.state.channel, this.state.tokenStartCharIndex, this.getCharIndex()-1);
            t.setLine(this.state.tokenStartLine);
            t.setText(this.state.text);
            t.setCharPositionInLine(this.state.tokenStartCharPositionInLine);
            this.state.token = t;
            return t;
        } else {
            this.state.token = arguments[0];
        }
    },

    match: function(s) {
        var i = 0,
            mte;

        if (org.antlr.lang.isString(s)) {
            while ( i<s.length ) {
                if ( this.input.LA(1)!=s.charAt(i) ) {
                    if ( this.state.backtracking>0 ) {
                        this.state.failed = true;
                        return;
                    }
                    mte = new org.antlr.runtime.MismatchedTokenException(s.charAt(i), this.input);
                    this.recover(mte);
                    throw mte;
                }
                i++;
                this.input.consume();
                this.state.failed = false;
            }
        } else if (org.antlr.lang.isNumber(s)) {
            if ( this.input.LA(1)!=s ) {
                if ( this.state.backtracking>0 ) {
                    this.state.failed = true;
                    return;
                }
                mte = new org.antlr.runtime.MismatchedTokenException(s, this.input);
                this.recover(mte);
                throw mte;
            }
            this.input.consume();
            this.state.failed = false;
        }
    },

    matchAny: function() {
        this.input.consume();
    },

    matchRange: function(a, b) {
        if ( this.input.LA(1)<a || this.input.LA(1)>b ) {
            if ( this.state.backtracking>0 ) {
                this.state.failed = true;
                return;
            }
            mre = new org.antlr.runtime.MismatchedRangeException(a,b,this.input);
            this.recover(mre);
            throw mre;
        }
        this.input.consume();
        this.state.failed = false;
    },

    getLine: function() {
        return this.input.getLine();
    },

    getCharPositionInLine: function() {
        return this.input.getCharPositionInLine();
    },

    /** What is the index of the current character of lookahead? */
    getCharIndex: function() {
        return this.input.index();
    },

    /** Return the text matched so far for the current token or any
     *  text override.
     */
    getText: function() {
        if ( org.antlr.lang.isString(this.state.text) ) {
            return this.state.text;
        }
        return this.input.substring(this.state.tokenStartCharIndex,this.getCharIndex()-1);
    },

    /** Set the complete text of this token; it wipes any previous
     *  changes to the text.
     */
    setText: function(text) {
        this.state.text = text;
    },

    reportError: function(e) {
        /** TODO: not thought about recovery in lexer yet.
         *
        // if we've already reported an error and have not matched a token
        // yet successfully, don't report any errors.
        if ( errorRecovery ) {
            //System.err.print("[SPURIOUS] ");
            return;
        }
        errorRecovery = true;
         */

        this.displayRecognitionError(this.getTokenNames(), e);
    },

    getErrorMessage: function(e, tokenNames) {
        var msg = null;
        if ( e instanceof org.antlr.runtime.MismatchedTokenException ) {
            msg = "mismatched character "+this.getCharErrorDisplay(e.c)+" expecting "+this.getCharErrorDisplay(e.expecting);
        }
        else if ( e instanceof org.antlr.runtime.NoViableAltException ) {
            msg = "no viable alternative at character "+this.getCharErrorDisplay(e.c);
        }
        else if ( e instanceof org.antlr.runtime.EarlyExitException ) {
            msg = "required (...)+ loop did not match anything at character "+this.getCharErrorDisplay(e.c);
        }
        else if ( e instanceof org.antlr.runtime.MismatchedNotSetException ) {
            msg = "mismatched character "+this.getCharErrorDisplay(e.c)+" expecting set "+e.expecting;
        }
        else if ( e instanceof org.antlr.runtime.MismatchedSetException ) {
            msg = "mismatched character "+this.getCharErrorDisplay(e.c)+" expecting set "+e.expecting;
        }
        else if ( e instanceof org.antlr.runtime.MismatchedRangeException ) {
            msg = "mismatched character "+this.getCharErrorDisplay(e.c)+" expecting set "+
                this.getCharErrorDisplay(e.a)+".."+this.getCharErrorDisplay(e.b);
        }
        else {
            msg = org.antlr.runtime.Lexer.superclass.getErrorMessage.call(this, e, tokenNames);
        }
        return msg;
    },

    getCharErrorDisplay: function(c) {
        var s = c; //String.fromCharCode(c);
        switch ( s ) {
            case org.antlr.runtime.Token.EOF :
                s = "<EOF>";
                break;
            case "\n" :
                s = "\\n";
                break;
            case "\t" :
                s = "\\t";
                break;
            case "\r" :
                s = "\\r";
                break;
        }
        return "'"+s+"'";
    },

    /** Lexers can normally match any char in it's vocabulary after matching
     *  a token, so do the easy thing and just kill a character and hope
     *  it all works out.  You can instead use the rule invocation stack
     *  to do sophisticated error recovery if you are in a fragment rule.
     */
    recover: function(re) {
        this.input.consume();
    },

    traceIn: function(ruleName, ruleIndex)  {
        var inputSymbol = String.fromCharCode(this.input.LT(1))+" line="+this.getLine()+":"+this.getCharPositionInLine();
        org.antlr.runtime.Lexer.superclass.traceIn.call(this, ruleName, ruleIndex, inputSymbol);
    },

    traceOut: function(ruleName, ruleIndex)  {
		var inputSymbol = String.fromCharCode(this.input.LT(1))+" line="+this.getLine()+":"+this.getCharPositionInLine();
		org.antlr.runtime.Lexer.superclass.traceOut.call(this, ruleName, ruleIndex, inputSymbol);
	}
});
