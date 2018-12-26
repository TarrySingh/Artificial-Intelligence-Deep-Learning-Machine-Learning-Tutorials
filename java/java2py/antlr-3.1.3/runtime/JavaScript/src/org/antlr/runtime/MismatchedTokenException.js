org.antlr.runtime.MismatchedTokenException = function(expecting, input) {
    if (arguments.length===0) {
        this.expecting = org.antlr.runtime.Token.INVALID_TOKEN_TYPE;
    } else {
        org.antlr.runtime.MismatchedTokenException.superclass.constructor.call(
                this, input);
        this.expecting = expecting;
    }
};

org.antlr.lang.extend(
    org.antlr.runtime.MismatchedTokenException,
    org.antlr.runtime.RecognitionException, {
    toString: function() {
        return "MismatchedTokenException(" +
                this.getUnexpectedType() + "!=" + this.expecting + ")";
    },
    name: "org.antlr.runtime.MismatchedTokenException"
});
