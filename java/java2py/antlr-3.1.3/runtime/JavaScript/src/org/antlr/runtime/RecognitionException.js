/** The root of the ANTLR exception hierarchy.
 *
 *  <p>To avoid English-only error messages and to generally make things
 *  as flexible as possible, these exceptions are not created with strings,
 *  but rather the information necessary to generate an error.  Then
 *  the various reporting methods in Parser and Lexer can be overridden
 *  to generate a localized error message.  For example, MismatchedToken
 *  exceptions are built with the expected token type.
 *  So, don't expect getMessage() to return anything.</p>
 *
 *  <p>ANTLR generates code that throws exceptions upon recognition error and
 *  also generates code to catch these exceptions in each rule.  If you
 *  want to quit upon first error, you can turn off the automatic error
 *  handling mechanism using rulecatch action, but you still need to
 *  override methods mismatch and recoverFromMismatchSet.</p>
 *
 *  <p>In general, the recognition exceptions can track where in a grammar a
 *  problem occurred and/or what was the expected input.  While the parser
 *  knows its state (such as current input symbol and line info) that
 *  state can change before the exception is reported so current token index
 *  is computed and stored at exception time.  From this info, you can
 *  perhaps print an entire line of input not just a single token, for example.
 *  Better to just say the recognizer had a problem and then let the parser
 *  figure out a fancy report.</p>
 *
 *  @class
 *  @param {org.antlr.runtime.CommonTokenStream|org.antlr.runtime.tree.TreeNodeStream|org.antlr.runtime.ANTLRStringStream} input input stream that has an exception.
 *  @extends Error
 *
 */
org.antlr.runtime.RecognitionException = function(input) {
    org.antlr.runtime.RecognitionException.superclass.constructor.call(this);
    this.input = input;
    this.index = input.index();
    if ( input instanceof org.antlr.runtime.TokenStream ) {
        this.token = input.LT(1);
        this.line = this.token.getLine();
        this.charPositionInLine = this.token.getCharPositionInLine();
    }
    if ( input instanceof org.antlr.runtime.tree.TreeNodeStream ) {
        this.extractInformationFromTreeNodeStream(input);
    }
    else if ( input instanceof org.antlr.runtime.CharStream ) {
        // Note: removed CharStream from hierarchy in JS port so checking for
        // StringStream instead
        this.c = input.LA(1);
        this.line = input.getLine();
        this.charPositionInLine = input.getCharPositionInLine();
    }
    else {
        this.c = input.LA(1);
    }

    this.message = this.toString();
};

org.antlr.lang.extend(org.antlr.runtime.RecognitionException, Error,
/** @lends org.antlr.runtime.RecognitionException.prototype */
{
	/**
     * What input stream did the error occur in?
     */
    input: null,

    /** What is index of token/char were we looking at when the error occurred?
     *  @type Number
     */
	index: null,

	/** The current Token when an error occurred.  Since not all streams
	 *  can retrieve the ith Token, we have to track the Token object.
	 *  For parsers.  Even when it's a tree parser, token might be set.
     *  @type org.antlr.runtime.CommonToken
	 */
	token: null,

	/** If this is a tree parser exception, node is set to the node with
	 *  the problem.
     *  @type Object
	 */
	node: null,

	/** The current char when an error occurred. For lexers.
     *  @type Number
     */
	c: null,

	/** Track the line at which the error occurred in case this is
	 *  generated from a lexer.  We need to track this since the
	 *  unexpected char doesn't carry the line info.
     *  @type Number
	 */
	line: null,

    /** The exception's class name.
     *  @type String
     */
    name: "org.antlr.runtime.RecognitionException",

    /** Position in the line where exception occurred.
     *  @type Number
     */
	charPositionInLine: null,

	/** If you are parsing a tree node stream, you will encounter som
	 *  imaginary nodes w/o line/col info.  We now search backwards looking
	 *  for most recent token with line/col info, but notify getErrorHeader()
	 *  that info is approximate.
     *  @type Boolean
	 */
	approximateLineInfo: null,

    /** Gather exception information from input stream.
     *  @param {org.antlr.runtime.CommonTokenStream|org.antlr.runtime.tree.TreeNodeStream|org.antlr.runtime.ANTLRStringStream} input input stream that has an exception.
     */
	extractInformationFromTreeNodeStream: function(input) {
		var nodes = input,
            priorNode,
            priorPayLoad,
            type,
            text,
            i;

		this.node = nodes.LT(1);
		var adaptor = nodes.getTreeAdaptor(),
		    payload = adaptor.getToken(this.node);
		if ( payload ) {
			this.token = payload;
			if ( payload.getLine()<= 0 ) {
				// imaginary node; no line/pos info; scan backwards
				i = -1;
				priorNode = nodes.LT(i);
				while ( priorNode ) {
					priorPayload = adaptor.getToken(priorNode);
					if ( priorPayload && priorPayload.getLine()>0 ) {
						// we found the most recent real line / pos info
						this.line = priorPayload.getLine();
						this.charPositionInLine = priorPayload.getCharPositionInLine();
						this.approximateLineInfo = true;
						break;
					}
					--i;
					priorNode = nodes.LT(i);
				}
			}
			else { // node created from real token
				this.line = payload.getLine();
				this.charPositionInLine = payload.getCharPositionInLine();
			}
		}
		else if ( this.node instanceof org.antlr.runtime.tree.Tree) {
			this.line = this.node.getLine();
			this.charPositionInLine = this.node.getCharPositionInLine();
			if ( this.node instanceof org.antlr.runtime.tree.CommonTree) {
				this.token = this.node.token;
			}
		}
		else {
			type = adaptor.getType(this.node);
			text = adaptor.getText(this.node);
			this.token = new org.antlr.runtime.CommonToken(type, text);
		}
	},

	/** Return the token type or char of the unexpected input element
     *  @return {Number} type of the unexpected input element.
     */
    getUnexpectedType: function() {
		if ( this.input instanceof org.antlr.runtime.TokenStream ) {
			return this.token.getType();
		}
		else if ( this.input instanceof org.antlr.runtime.tree.TreeNodeStream ) {
			var nodes = this.input;
			var adaptor = nodes.getTreeAdaptor();
			return adaptor.getType(this.node);
		}
		else {
			return this.c;
		}
	}
});
