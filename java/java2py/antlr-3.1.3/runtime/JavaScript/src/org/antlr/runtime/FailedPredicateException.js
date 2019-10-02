/** A semantic predicate failed during validation.  Validation of predicates
 *  occurs when normally parsing the alternative just like matching a token.
 *  Disambiguating predicate evaluation occurs when we hoist a predicate into
 *  a prediction decision.
 *
 *  @class
 *  @param {org.antlr.runtime.CommonTokenStream|org.antlr.runtime.tree.TreeNodeStream|org.antlr.runtime.ANTLRStringStream} input input stream that has an exception.
 *  @param {String} ruleName name of the rule in which the exception occurred.
 *  @param {String} predicateText the predicate that failed.
 *  @extends org.antlr.runtime.RecognitionException
 */
org.antlr.runtime.FailedPredicateException = function(input, ruleName, predicateText){
    org.antlr.runtime.FailedPredicateException.superclass.constructor.call(this, input);
    this.ruleName = ruleName;
    this.predicateText = predicateText;
};

org.antlr.lang.extend(
    org.antlr.runtime.FailedPredicateException,
    org.antlr.runtime.RecognitionException,
/** @lends org.antlr.runtime.FailedPredicateException.prototype */
{
    /** Create a string representation of this exception.
     *  @returns {String}
     */ 
    toString: function() {
        return "FailedPredicateException("+this.ruleName+",{"+this.predicateText+"}?)";
    },

    /** Name of this class.
     *  @type String
     */
    name: "org.antlr.runtime.FailedPredicateException"
});
