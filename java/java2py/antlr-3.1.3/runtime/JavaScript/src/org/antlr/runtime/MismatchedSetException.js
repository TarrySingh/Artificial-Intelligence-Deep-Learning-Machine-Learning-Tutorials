org.antlr.runtime.MismatchedSetException = function(expecting, input) {
    org.antlr.runtime.MismatchedSetException.superclass.constructor.call(
            this, input);
    this.expecting = expecting;
};

org.antlr.lang.extend(
    org.antlr.runtime.MismatchedSetException,
    org.antlr.runtime.RecognitionException, {
    toString: function() {
        return "MismatchedSetException(" +
                this.getUnexpectedType() + "!=" + this.expecting + ")";
    },
    name: "org.antlr.runtime.MismatchedSetException"
});
