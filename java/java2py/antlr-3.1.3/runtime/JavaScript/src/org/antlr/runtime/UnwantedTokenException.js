/** An extra token while parsing a TokenStream */
org.antlr.runtime.UnwantedTokenException = function(expecting, input) {
    if (arguments.length>0) {
        org.antlr.runtime.UnwantedTokenException.superclass.constructor.call(
                this, expecting, input);
    }
};

org.antlr.lang.extend(
    org.antlr.runtime.UnwantedTokenException,
    org.antlr.runtime.MismatchedTokenException, {
    getUnexpectedToken: function() {
        return this.token;
    },
    toString: function() {
        var exp = ", expected "+this.expecting;
        if ( this.expecting===org.antlr.runtime.Token.INVALID_TOKEN_TYPE ) {
            exp = "";
        }
        if ( !org.antlr.lang.isValue(this.token) ) {
            return "UnwantedTokenException(found="+exp+")";
        }
        return "UnwantedTokenException(found="+this.token.getText()+exp+")";
    },
    name: "org.antlr.runtime.UnwantedTokenException"
});
