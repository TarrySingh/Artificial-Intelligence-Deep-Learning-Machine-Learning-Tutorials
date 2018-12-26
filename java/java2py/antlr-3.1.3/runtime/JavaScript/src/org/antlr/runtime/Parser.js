/** A parser for TokenStreams.  "parser grammars" result in a subclass
 *  of this.
 */
org.antlr.runtime.Parser = function(input, state) {
    org.antlr.runtime.Parser.superclass.constructor.call(this, state);
    this.setTokenStream(input);
};

org.antlr.lang.extend(org.antlr.runtime.Parser, org.antlr.runtime.BaseRecognizer, {
    reset: function() {
        // reset all recognizer state variables
		org.antlr.runtime.Parser.superclass.reset.call(this);
		if ( org.antlr.lang.isValue(this.input) ) {
			this.input.seek(0); // rewind the input
		}
	},

    getCurrentInputSymbol: function(input) {
        return input.LT(1);
    },

    getMissingSymbol: function(input,
                               e,
                               expectedTokenType,
                               follow)
    {
        var tokenText =
            "<missing "+this.getTokenNames()[expectedTokenType]+">";
        var t = new org.antlr.runtime.CommonToken(expectedTokenType, tokenText);
        var current = input.LT(1);
        var old_current;
        if ( current.getType() === org.antlr.runtime.Token.EOF ) {
            old_current = current;
            current = input.LT(-1);
            // handle edge case where there are no good tokens in the stream
            if (!current) {
                current = old_current;
            }
        }
        t.line = current.getLine();
        t.charPositionInLine = current.getCharPositionInLine();
        t.channel = org.antlr.runtime.BaseRecognizer.DEFAULT_TOKEN_CHANNEL;
        return t;
    },


	/** Set the token stream and reset the parser */
    setTokenStream: function(input) {
		this.input = null;
		this.reset();
		this.input = input;
	},

    getTokenStream: function() {
		return this.input;
	},

    getSourceName: function() {
        return this.input.getSourceName();
    },

    traceIn: function(ruleName, ruleIndex)  {
		org.antlr.runtime.Parser.superclass.traceIn.call(
                this, ruleName, ruleIndex, this.input.LT(1));
	},

    traceOut: function(ruleName, ruleIndex)  {
		org.antlr.runtime.Parser.superclass.traceOut.call(
                this, ruleName, ruleIndex, this.input.LT(1));
	}
});
