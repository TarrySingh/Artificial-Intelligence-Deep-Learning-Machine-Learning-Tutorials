org.antlr.runtime.NoViableAltException = function(grammarDecisionDescription,
                                            decisionNumber,
                                            stateNumber,
                                            input)
{
    org.antlr.runtime.NoViableAltException.superclass.constructor.call(this, input);
    this.grammarDecisionDescription = grammarDecisionDescription;
    this.decisionNumber = decisionNumber;
    this.stateNumber = stateNumber;
};

org.antlr.lang.extend(
    org.antlr.runtime.NoViableAltException,
    org.antlr.runtime.RecognitionException, {
    toString: function() {
        if ( this.input instanceof org.antlr.runtime.CharStream ) {
            return "NoViableAltException('"+this.getUnexpectedType()+"'@["+this.grammarDecisionDescription+"])";
        }
        else {
            return "NoViableAltException("+this.getUnexpectedType()+"@["+this.grammarDecisionDescription+"])";
        }
    },
    name: "org.antlr.runtime.NoViableAltException"
});
