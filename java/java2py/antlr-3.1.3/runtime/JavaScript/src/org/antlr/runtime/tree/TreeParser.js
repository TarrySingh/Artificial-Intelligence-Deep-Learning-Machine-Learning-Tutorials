/** A parser for a stream of tree nodes.  "tree grammars" result in a subclass
 *  of this.  All the error reporting and recovery is shared with Parser via
 *  the BaseRecognizer superclass.
*/
org.antlr.runtime.tree.TreeParser = function(input) {
    org.antlr.runtime.tree.TreeParser.superclass.constructor.call(this, arguments[1]);
    this.setTreeNodeStream(input);
};

(function(){
var TP = org.antlr.runtime.tree.TreeParser;

org.antlr.lang.augmentObject(TP, {
    DOWN: org.antlr.runtime.Token.DOWN,
    UP: org.antlr.runtime.Token.UP
});

org.antlr.lang.extend(TP, org.antlr.runtime.BaseRecognizer, {
    reset: function() {
        TP.superclass.reset.call(this); // reset all recognizer state variables
        if ( this.input ) {
            this.input.seek(0); // rewind the input
        }
    },

    /** Set the input stream */
    setTreeNodeStream: function(input) {
        this.input = input;
    },

    getTreeNodeStream: function() {
        return this.input;
    },

    getSourceName: function() {
        return this.input.getSourceName();
    },

    getCurrentInputSymbol: function(input) {
        return input.LT(1);
    },

    getMissingSymbol: function(input, e, expectedTokenType, follow) {
        var tokenText =
            "<missing "+this.getTokenNames()[expectedTokenType]+">";
        return new org.antlr.runtime.tree.CommonTree(new org.antlr.runtime.CommonToken(expectedTokenType, tokenText));
    },

    /** Match '.' in tree parser has special meaning.  Skip node or
     *  entire tree if node has children.  If children, scan until
     *  corresponding UP node.
     */
    matchAny: function(ignore) { // ignore stream, copy of this.input
        this.state.errorRecovery = false;
        this.state.failed = false;
        var look = this.input.LT(1);
        if ( this.input.getTreeAdaptor().getChildCount(look)===0 ) {
            this.input.consume(); // not subtree, consume 1 node and return
            return;
        }
        // current node is a subtree, skip to corresponding UP.
        // must count nesting level to get right UP
        var level=0,
            tokenType = this.input.getTreeAdaptor().getType(look);
        while ( tokenType!==org.antlr.runtime.Token.EOF &&
                !(tokenType===TP.UP && level===0) )
        {
            this.input.consume();
            look = this.input.LT(1);
            tokenType = this.input.getTreeAdaptor().getType(look);
            if ( tokenType === TP.DOWN ) {
                level++;
            }
            else if ( tokenType === TP.UP ) {
                level--;
            }
        }
        this.input.consume(); // consume UP
    },

    /** We have DOWN/UP nodes in the stream that have no line info; override.
     *  plus we want to alter the exception type.  Don't try to recover
     *       *  from tree parser errors inline...
     */
    mismatch: function(input, ttype, follow) {
        throw new org.antlr.runtime.MismatchedTreeNodeException(ttype, input);
    },

    /** Prefix error message with the grammar name because message is
     *  always intended for the programmer because the parser built
     *  the input tree not the user.
     */
    getErrorHeader: function(e) {
        return this.getGrammarFileName()+": node from "+
               (e.approximateLineInfo?"after ":"")+"line "+e.line+":"+e.charPositionInLine;
    },

    /** Tree parsers parse nodes they usually have a token object as
     *  payload. Set the exception token and do the default behavior.
     */
    getErrorMessage: function(e, tokenNames) {
        var adaptor;
        if ( this instanceof TP ) {
            adaptor = e.input.getTreeAdaptor();
            e.token = adaptor.getToken(e.node);
            if ( !org.antlr.lang.isValue(e.token) ) { // could be an UP/DOWN node
                e.token = new org.antlr.runtime.CommonToken(
                        adaptor.getType(e.node),
                        adaptor.getText(e.node));
            }
        }
        return TP.superclass.getErrorMessage.call(this, e, tokenNames);
    },

    traceIn: function(ruleName, ruleIndex) {
        TP.superclass.traceIn.call(this, ruleName, ruleIndex, this.input.LT(1));
    },

    traceOut: function(ruleName, ruleIndex) {
        TP.superclass.traceOut.call(this, ruleName, ruleIndex, this.input.LT(1));
    }
});

})();
