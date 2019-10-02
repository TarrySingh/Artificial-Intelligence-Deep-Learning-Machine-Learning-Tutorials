org.antlr.runtime.MismatchedNotSetException = function(expecting, input) {
    org.antlr.runtime.MismatchedNotSetException.superclass.constructor.call(this, expecting, input);
};

org.antlr.lang.extend(
    org.antlr.runtime.MismatchedNotSetException,
    org.antlr.runtime.MismatchedSetException, {
    toString: function() {
        return "MismatchedNotSetException(" +
                this.getUnexpectedType() + "!=" + this.expecting + ")";
    },
    name: "org.antlr.runtime.MismatchedNotSetException"
});
