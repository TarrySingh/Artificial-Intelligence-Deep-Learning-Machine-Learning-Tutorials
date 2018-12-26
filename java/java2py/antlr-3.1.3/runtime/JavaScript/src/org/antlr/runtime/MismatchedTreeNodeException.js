org.antlr.runtime.MismatchedTreeNodeException = function(expecting, input) {
    if (expecting && input) {
        org.antlr.runtime.MismatchedTreeNodeException.superclass.constructor.call(
                this, input);
        this.expecting = expecting;
    }
};

org.antlr.lang.extend(
    org.antlr.runtime.MismatchedTreeNodeException,
    org.antlr.runtime.RecognitionException, {
    toString: function() {
        return "MismatchedTreeNodeException(" +
                this.getUnexpectedType() + "!=" + this.expecting + ")";
    },
    name: "org.antlr.runtime.MismatchedTreeNodeException"
});
