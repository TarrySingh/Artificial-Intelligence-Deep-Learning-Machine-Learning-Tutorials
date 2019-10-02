org.antlr.runtime.MismatchedRangeException = function(a, b, input) {
    if (arguments.length===0) {
        return this;
    }

    org.antlr.runtime.MismatchedRangeException.superclass.constructor.call(
            this, input);
    this.a = a;
    this.b = b;
};

org.antlr.lang.extend(
    org.antlr.runtime.MismatchedRangeException,
    org.antlr.runtime.RecognitionException, {
    toString: function() {
        return "MismatchedRangeException(" +
                this.getUnexpectedType()+" not in ["+this.a+","+this.b+"])";
    },
    name: "org.antlr.runtime.MismatchedRangeException"
});
