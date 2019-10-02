org.antlr.runtime.MissingTokenException = function(expecting, input, inserted) {
    if (arguments.length>0) {
        org.antlr.runtime.MissingTokenException.superclass.constructor.call(
                this, expecting, input);
        this.inserted = inserted;
    }
};

org.antlr.lang.extend(
    org.antlr.runtime.MissingTokenException,
    org.antlr.runtime.MismatchedTokenException, {
    getMissingType: function() {
        return this.expecting;
    },

    toString: function() {
        if (org.antlr.lang.isValue(this.inserted) &&
            org.antlr.lang.isValue(this.token))
        {
            return "MissingTokenException(inserted "+this.inserted+" at "+this.token.getText()+")";
        }
        if ( org.antlr.lang.isValue(this.token) ) {
            return "MissingTokenException(at "+this.token.getText()+")";
        }
        return "MissingTokenException";
    },
    name: "org.antlr.runtime.MissingTokenException"
});
