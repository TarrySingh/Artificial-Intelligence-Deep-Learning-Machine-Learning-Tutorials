/** The recognizer did not match anything for a ()+ loop.
 *
 *  @class
 *  @param {Number} decisionNumber
 *  @param {org.antlr.runtime.CommonTokenStream|org.antlr.runtime.tree.TreeNodeStream|org.antlr.runtime.ANTLRStringStream} input input stream that has an exception.
 *  @extends org.antlr.runtime.RecognitionException
 */
org.antlr.runtime.EarlyExitException = function(decisionNumber, input) {
    org.antlr.runtime.EarlyExitException.superclass.constructor.call(
            this, input);
    this.decisionNumber = decisionNumber;
};

org.antlr.lang.extend(
    org.antlr.runtime.EarlyExitException,
    org.antlr.runtime.RecognitionException,
/** @lends org.antlr.runtime.EarlyExitException.prototype */
{
    /** Name of this class.
     *  @type String
     */
    name: "org.antlr.runtime.EarlyExitException"
});
