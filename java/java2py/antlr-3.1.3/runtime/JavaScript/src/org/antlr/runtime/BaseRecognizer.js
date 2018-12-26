/** A generic recognizer that can handle recognizers generated from
 *  lexer, parser, and tree grammars.  This is all the parsing
 *  support code essentially; most of it is error recovery stuff and
 *  backtracking.
 *
 *  <p>This class should not be instantiated directly.  Instead, use one of its
 *  subclasses.</p>
 *
 *  @class
 *  @param {org.antlr.runtime.RecognizerSharedState} [state] state object with
 *      which to initialize this recognizer.
 */
org.antlr.runtime.BaseRecognizer = function(state) {
    /** State of a lexer, parser, or tree parser are collected into a state
     *  object so the state can be shared.  This sharing is needed to
     *  have one grammar import others and share same error variables
     *  and other state variables.  It's a kind of explicit multiple
     *  inheritance via delegation of methods and shared state.
     *  @type org.antlr.runtime.RecognizerSharedState
     */
    this.state = state || new org.antlr.runtime.RecognizerSharedState();
};

org.antlr.lang.augmentObject(org.antlr.runtime.BaseRecognizer, {
    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    MEMO_RULE_FAILED: -2,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    MEMO_RULE_UNKNOWN: -1,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    INITIAL_FOLLOW_STACK_SIZE: 100,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    MEMO_RULE_FAILED_I: -2,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    DEFAULT_TOKEN_CHANNEL: org.antlr.runtime.Token.DEFAULT_CHANNEL,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type Number
     */
    HIDDEN: org.antlr.runtime.Token.HIDDEN_CHANNEL,

    /**
     * @memberOf org.antlr.runtime.BaseRecognizer
     * @type String 
     */
    NEXT_TOKEN_RULE_NAME: "nextToken"
});

org.antlr.runtime.BaseRecognizer.prototype = {
    /** Reset the parser's state.  Subclasses must rewinds the input stream */
    reset: function() {
        var i, len;

        // wack everything related to error recovery
        if (!this.state) {
            return; // no shared state work to do
        }
        this.state._fsp = -1;
        this.state.errorRecovery = false;
        this.state.lastErrorIndex = -1;
        this.state.failed = false;
        this.state.syntaxErrors = 0;
        // wack everything related to backtracking and memoization
        this.state.backtracking = 0;
        // wipe cache
        if (this.state.ruleMemo) {
            for (i=0, len=this.state.ruleMemo.length; i<len; i++) {
                this.state.ruleMemo[i] = null;
            }
        }
    },

    /** Match current input symbol against ttype.  Attempt
     *  single token insertion or deletion error recovery.  If
     *  that fails, throw {@link org.antlr.runtime.MismatchedTokenException}.
     *
     *  <p>To turn off single token insertion or deletion error
     *  recovery, override {@link #mismatchRecover} and have it call
     *  plain {@link #mismatch}, which does not recover.  Then any error
     *  in a rule will cause an exception and immediate exit from
     *  rule.  Rule would recover by resynchronizing to the set of
     *  symbols that can follow rule ref.</p>
     *
     *  @param {org.antlr.runtime.IntStream} input input stream to match against.
     *  @param {Number} ttype  input type to match.
     *  @param {org.antlr.runtime.BitSet} [follow] set of tokens that can follow the
     *      matched token.
     *  @returns {Object} the matched symbol
     */
    match: function(input, ttype, follow) {
        var matchedSymbol = this.getCurrentInputSymbol(input);
        if ( input.LA(1)===ttype ) {
            input.consume();
            this.state.errorRecovery = false;
            this.state.failed = false;
            return matchedSymbol;
        }
        if ( this.state.backtracking>0 ) {
            this.state.failed = true;
            return matchedSymbol;
        }
        matchedSymbol = this.recoverFromMismatchedToken(input, ttype, follow);
        return matchedSymbol;
    },

    /**
     * Match any token.
     * @param {org.antlr.runtime.IntStream} input input stream to match against.
     */
    matchAny: function(input) {
        this.state.errorRecovery = false;
        this.state.failed = false;
        input.consume();
    },

    /**
     * Is the following token (LA(2)) the unwanted type (ttype)?
     * @param {org.antlr.runtime.IntStream} input input stream to match against.
     * @param {Number} ttype the undesired token type.
     * @returns {Boolean} true if and only if the following token is the
     *      unwanted type.
     */
    mismatchIsUnwantedToken: function(input, ttype) {
        return input.LA(2)===ttype;
    },

    /**
     * Does the stream appear to be missing a single token?
     * @param {org.antlr.runtime.IntStream} input input stream to match against.
     * @param {org.antlr.runtime.BitSet} [follow] set of tokens that can follow the
     *      matched token.
     * @returns {Boolean} true if and only if it appears that the stream is
     *      missing a single token.
     */
    mismatchIsMissingToken: function(input, follow) {
        if ( !follow ) {
            // we have no information about the follow; we can only consume
            // a single token and hope for the best
            return false;
        }
        // compute what can follow this grammar element reference
        if ( follow.member(org.antlr.runtime.Token.EOR_TOKEN_TYPE) ) {
            var viableTokensFollowingThisRule = this.computeContextSensitiveRuleFOLLOW();
            follow = follow.or(this.viableTokensFollowingThisRule);
            if ( this.state._fsp>=0 ) { // remove EOR if we're not the start symbol
                follow.remove(org.antlr.runtime.Token.EOR_TOKEN_TYPE);
            }
        }
        // if current token is consistent with what could come after set
        // then we know we're missing a token; error recovery is free to
        // "insert" the missing token

        // BitSet cannot handle negative numbers like -1 (EOF) so I leave EOR
        // in follow set to indicate that the fall of the start symbol is
        // in the set (EOF can follow).
        if ( follow.member(input.LA(1)) ||
             follow.member(org.antlr.runtime.Token.EOR_TOKEN_TYPE) )
        {
            return true;
        }
        return false;
    },

    /** Factor out what to do upon token mismatch so tree parsers can behave
     *  differently.  Override and call {@link #mismatchRecover}
     *  to get single token insertion and deletion.
     *
     *  @param {org.antlr.runtime.IntStream} input input stream to match against.
     *  @param {Number} ttype  input type to match.
     *  @param {org.antlr.runtime.BitSet} [follow] set of tokens that can follow the
     *      matched token.
     */
    mismatch: function(input, ttype, follow) {
        if ( this.mismatchIsUnwantedToken(input, ttype) ) {
            throw new org.antlr.runtime.UnwantedTokenException(ttype, input);
        } else if ( this.mismatchIsMissingToken(input, follow) ) {
            throw new org.antlr.runtime.MissingTokenException(ttype, input, null);
        }
        throw new org.antlr.runtime.MismatchedTokenException(ttype, input);
    },

    /** Report a recognition problem.
     *
     *  <p>This method sets errorRecovery to indicate the parser is recovering
     *  not parsing.  Once in recovery mode, no errors are generated.
     *  To get out of recovery mode, the parser must successfully match
     *  a token (after a resync).  So it will go:</p>
     *  <ol>
     *      <li>error occurs</li>
     *      <li>enter recovery mode, report error</li>
     *      <li>consume until token found in resynch set</li>
     *      <li>try to resume parsing</li>
     *      <li>next match() will reset errorRecovery mode</li>
     *  </ol>
     *
     *  <p>If you override, make sure to update this.state.syntaxErrors if you
     *  care about that.</p>
     *  @param {org.antlr.runtime.RecognitionException} e the error to be reported.
     */
    reportError: function(e) {
        // if we've already reported an error and have not matched a token
        // yet successfully, don't report any errors.
        if ( this.state.errorRecovery ) {
            return;
        }
        this.state.syntaxErrors++;
        this.state.errorRecovery = true;

        this.displayRecognitionError(this.getTokenNames(), e);
    },

    /**
     * Assemble recognition error message.
     * @param {Array} tokenNames array of token names (strings).
     * @param {org.antlr.runtime.RecognitionException} e the error to be reported.
     */
    displayRecognitionError: function(tokenNames, e) {
        var hdr = this.getErrorHeader(e),
            msg = this.getErrorMessage(e, tokenNames);
        this.emitErrorMessage(hdr+" "+msg);
    },

    /**
     * Create error header message.  Format is <q>line
     * lineNumber:positionInLine</q>.
     * @param {org.antlr.runtime.RecognitionException} e the error to be reported.
     * @returns {String} The error header.
     */
    getErrorHeader: function(e) {
        /* handle null input */
        if (!org.antlr.lang.isNumber(e.line)) {
            e.line = 0;
        }
        return "line "+e.line+":"+e.charPositionInLine;
    },

    /**
     * Override this method to change where error messages go.
     * Defaults to "alert"-ing the error in browsers and "print"-ing the error
     * in other environments (e.g. Rhino, SpiderMonkey).
     * @param {String} msg the error message to be displayed.
     */
    emitErrorMessage: function(msg) {
        if (typeof(window) != 'undefined' && window.alert) {
            alert(msg);
        } else {
            print(msg);
        }
    },

    /** What error message should be generated for the various
     *  exception types?
     *
     *  <p>Not very object-oriented code, but I like having all error message
     *  generation within one method rather than spread among all of the
     *  exception classes. This also makes it much easier for the exception
     *  handling because the exception classes do not have to have pointers back
     *  to this object to access utility routines and so on. Also, changing
     *  the message for an exception type would be difficult because you
     *  would have to be subclassing exceptions, but then somehow get ANTLR
     *  to make those kinds of exception objects instead of the default.</p>
     *
     *  <p>For grammar debugging, you will want to override this to add
     *  more information such as the stack frame and no viable alts.</p>
     *
     *  <p>Override this to change the message generated for one or more
     *  exception types.</p>
     *
     * @param {Array} tokenNames array of token names (strings).
     * @param {org.antlr.runtime.RecognitionException} e the error to be reported.
     * @returns {String} the error message to be emitted.
     */
    getErrorMessage: function(e, tokenNames) {
        var msg = (e && e.getMessage) ? e.getMessage() : null,
            mte,
            tokenName;
        if ( e instanceof org.antlr.runtime.UnwantedTokenException ) {
            var ute = e;
            tokenName="<unknown>";
            if ( ute.expecting== org.antlr.runtime.Token.EOF ) {
                tokenName = "EOF";
            } else {
                tokenName = tokenNames[ute.expecting];
            }
            msg = "extraneous input "+this.getTokenErrorDisplay(ute.getUnexpectedToken())+
                " expecting "+tokenName;
        }
        else if ( e instanceof org.antlr.runtime.MissingTokenException ) {
            mte = e;
            tokenName="<unknown>";
            if ( mte.expecting== org.antlr.runtime.Token.EOF ) {
                tokenName = "EOF";
            } else {
                tokenName = tokenNames[mte.expecting];
            }
            msg = "missing "+tokenName+" at "+this.getTokenErrorDisplay(e.token);
        }
        else if ( e instanceof org.antlr.runtime.MismatchedTokenException ) {
            mte = e;
            tokenName="<unknown>";
            if ( mte.expecting== org.antlr.runtime.Token.EOF ) {
                tokenName = "EOF";
            }
            else {
                tokenName = tokenNames[mte.expecting];
            }
            msg = "mismatched input "+this.getTokenErrorDisplay(e.token)+
                " expecting "+tokenName;
        }
        else if ( e instanceof org.antlr.runtime.NoViableAltException ) {
            msg = "no viable alternative at input "+this.getTokenErrorDisplay(e.token);
        }
        else if ( e instanceof org.antlr.runtime.EarlyExitException ) {
            msg = "required (...)+ loop did not match anything at input "+
                this.getTokenErrorDisplay(e.token);
        }
        else if ( e instanceof org.antlr.runtime.MismatchedSetException ) {
            msg = "mismatched input "+this.getTokenErrorDisplay(e.token)+
                " expecting set "+e.expecting;
        }
        else if ( e instanceof org.antlr.runtime.MismatchedNotSetException ) {
            msg = "mismatched input "+this.getTokenErrorDisplay(e.token)+
                " expecting set "+e.expecting;
        }
        else if ( e instanceof org.antlr.runtime.FailedPredicateException ) {
            msg = "rule "+e.ruleName+" failed predicate: {"+
                e.predicateText+"}?";
        }
        return msg;
    },

    /** <p>Get number of recognition errors (lexer, parser, tree parser).  Each
     *  recognizer tracks its own number.  So parser and lexer each have
     *  separate count.  Does not count the spurious errors found between
     *  an error and next valid token match.</p>
     *
     *  <p>See also {@link #reportError}()
     *  @returns {Number} number of syntax errors encountered
     */
    getNumberOfSyntaxErrors: function() {
        return this.state.syntaxErrors;
    },

    /** How should a token be displayed in an error message? The default
     *  is to display just the text, but during development you might
     *  want to have a lot of information spit out.  Override in that case
     *  to use t.toString() (which, for CommonToken, dumps everything about
     *  the token).
     * @param {org.antlr.runtime.Token} t token that will be displayed in an error message
     * @return {String} the string representation of the token
     */
    getTokenErrorDisplay: function(t) {
        var s = t.getText();
        if ( !org.antlr.lang.isValue(s) ) {
            if ( t.getType()==org.antlr.runtime.Token.EOF ) {
                s = "<EOF>";
            }
            else {
                s = "<"+t.getType()+">";
            }
        }
        s = s.replace(/\n/g,"\\n");
        s = s.replace(/\r/g,"\\r");
        s = s.replace(/\t/g,"\\t");
        return "'"+s+"'";
    },

    /** Recover from an error found on the input stream.  This is
     *  for NoViableAlt and mismatched symbol exceptions.  If you enable
     *  single token insertion and deletion, this will usually not
     *  handle mismatched symbol exceptions but there could be a mismatched
     *  token that the match() routine could not recover from.
     *  @param {org.antlr.runtime.IntStream} input the intput stream
     *  @param {org.antlr.runtime.RecogntionException} the error found on the input stream
     */
    recover: function(input, re) {
        if ( this.state.lastErrorIndex==input.index() ) {
            // uh oh, another error at same token index; must be a case
            // where LT(1) is in the recovery token set so nothing is
            // consumed; consume a single token so at least to prevent
            // an infinite loop; this is a failsafe.
            input.consume();
        }
        this.state.lastErrorIndex = input.index();
        var followSet = this.computeErrorRecoverySet();
        this.beginResync();
        this.consumeUntil(input, followSet);
        this.endResync();
    },

    /** A hook to listen in on the token consumption during error recovery.
     */
    beginResync: function() {
    },

    /** A hook to listen in on the token consumption during error recovery.
     */
    endResync: function() {
    },

    /** Compute the error recovery set for the current rule.
     *  <p>During rule invocation, the parser pushes the set of tokens that can
     *  follow that rule reference on the stack; this amounts to
     *  computing FIRST of what follows the rule reference in the
     *  enclosing rule. This local follow set only includes tokens
     *  from within the rule; i.e., the FIRST computation done by
     *  ANTLR stops at the end of a rule.</p>
     *
     *  <p>EXAMPLE</p>
     *
     *  <p>When you find a "no viable alt exception", the input is not
     *  consistent with any of the alternatives for rule r.  The best
     *  thing to do is to consume tokens until you see something that
     *  can legally follow a call to r *or* any rule that called r.
     *  You don't want the exact set of viable next tokens because the
     *  input might just be missing a token--you might consume the
     *  rest of the input looking for one of the missing tokens.</p>
     *
     *  <p>Consider grammar:</p>
     *  <code><pre>
     *  a : '[' b ']'
     *    | '(' b ')'
     *    ;
     *  b : c '^' INT ;
     *  c : ID
     *    | INT
     *    ;
     *  </pre></code>
     *
     *  <p>At each rule invocation, the set of tokens that could follow
     *  that rule is pushed on a stack.  Here are the various "local"
     *  follow sets:</p>
     *
     *  <code><pre>
     *  FOLLOW(b1_in_a) = FIRST(']') = ']'
     *  FOLLOW(b2_in_a) = FIRST(')') = ')'
     *  FOLLOW(c_in_b) = FIRST('^') = '^'
     *  </pre></code>
     *
     *  <p>Upon erroneous input "[]", the call chain is</p>
     *
     *  <code>a -> b -> c</code>
     *
     *  <p>and, hence, the follow context stack is:</p>
     *
     *  <code><pre>
     *  depth  local follow set     after call to rule
     *    0         <EOF>                    a (from main())
     *    1          ']'                     b
     *    3          '^'                     c
     *  </pre></code>
     *
     *  <p>Notice that ')' is not included, because b would have to have
     *  been called from a different context in rule a for ')' to be
     *  included.</p>
     *
     *  <p>For error recovery, we cannot consider FOLLOW(c)
     *  (context-sensitive or otherwise).  We need the combined set of
     *  all context-sensitive FOLLOW sets--the set of all tokens that
     *  could follow any reference in the call chain.  We need to
     *  resync to one of those tokens.  Note that FOLLOW(c)='^' and if
     *  we resync'd to that token, we'd consume until EOF.  We need to
     *  sync to context-sensitive FOLLOWs for a, b, and c: {']','^'}.
     *  In this case, for input "[]", LA(1) is in this set so we would
     *  not consume anything and after printing an error rule c would
     *  return normally.  It would not find the required '^' though.
     *  At this point, it gets a mismatched token error and throws an
     *  exception (since LA(1) is not in the viable following token
     *  set).  The rule exception handler tries to recover, but finds
     *  the same recovery set and doesn't consume anything.  Rule b
     *  exits normally returning to rule a.  Now it finds the ']' (and
     *  with the successful match exits errorRecovery mode).</p>
     *
     *  <p>So, you cna see that the parser walks up call chain looking
     *  for the token that was a member of the recovery set.</p>
     *
     *  <p>Errors are not generated in errorRecovery mode.</p>
     *
     *  <p>ANTLR's error recovery mechanism is based upon original ideas:</p>
     *
     *  <p>"Algorithms + Data Structures = Programs" by Niklaus Wirth</p>
     *
     *  <p>and</p>
     *
     *  <p>"A note on error recovery in recursive descent parsers":
     *  http://portal.acm.org/citation.cfm?id=947902.947905</p>
     *
     *  <p>Later, Josef Grosch had some good ideas:</p>
     *
     *  <p>"Efficient and Comfortable Error Recovery in Recursive Descent
     *  Parsers":
     *  ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip</p>
     *
     *  <p>Like Grosch I implemented local FOLLOW sets that are combined
     *  at run-time upon error to avoid overhead during parsing.</p>
     *  @returns {org.antlr.runtime.BitSet}
     */
    computeErrorRecoverySet: function() {
        return this.combineFollows(false);
    },


    /** Compute the context-sensitive FOLLOW set for current rule.
     *  <p>This is set of token types that can follow a specific rule
     *  reference given a specific call chain.  You get the set of
     *  viable tokens that can possibly come next (lookahead depth 1)
     *  given the current call chain.  Contrast this with the
     *  definition of plain FOLLOW for rule r:</p>
     *
     *   <code>FOLLOW(r)={x | S=>*alpha r beta in G and x in FIRST(beta)}</code>
     *
     *  <p>where x in T* and alpha, beta in V*; T is set of terminals and
     *  V is the set of terminals and nonterminals.  In other words,
     *  FOLLOW(r) is the set of all tokens that can possibly follow
     *  references to r in *any* sentential form (context).  At
     *  runtime, however, we know precisely which context applies as
     *  we have the call chain.  We may compute the exact (rather
     *  than covering superset) set of following tokens.</p>
     *
     *  <p>For example, consider grammar:</p>
     *
     *  <code><pre>
     *  stat : ID '=' expr ';'      // FOLLOW(stat)=={EOF}
     *       | "return" expr '.'
     *       ;
     *  expr : atom ('+' atom)* ;   // FOLLOW(expr)=={';','.',')'}
     *  atom : INT                  // FOLLOW(atom)=={'+',')',';','.'}
     *       | '(' expr ')'
     *       ;
     *  </pre></code>
     *
     *  <p>The FOLLOW sets are all inclusive whereas context-sensitive
     *  FOLLOW sets are precisely what could follow a rule reference.
     *  For input input "i=(3);", here is the derivation:</p>
     *
     *  <code><pre>
     *  stat => ID '=' expr ';'
     *       => ID '=' atom ('+' atom)* ';'
     *       => ID '=' '(' expr ')' ('+' atom)* ';'
     *       => ID '=' '(' atom ')' ('+' atom)* ';'
     *       => ID '=' '(' INT ')' ('+' atom)* ';'
     *       => ID '=' '(' INT ')' ';'
     *  </pre></code>
     *
     *  <p>At the "3" token, you'd have a call chain of</p>
     *
     *  <code>  stat -> expr -> atom -> expr -> atom</code>
     *
     *  <p>What can follow that specific nested ref to atom?  Exactly ')'
     *  as you can see by looking at the derivation of this specific
     *  input.  Contrast this with the FOLLOW(atom)={'+',')',';','.'}.</p>
     *
     *  <p>You want the exact viable token set when recovering from a
     *  token mismatch.  Upon token mismatch, if LA(1) is member of
     *  the viable next token set, then you know there is most likely
     *  a missing token in the input stream.  "Insert" one by just not
     *  throwing an exception.</p>
     *  @returns {org.antlr.runtime.BitSet}
     */
    computeContextSensitiveRuleFOLLOW: function() {
        return this.combineFollows(true);
    },

    /**
     * Helper method for {@link #computeErrorRecoverySet} and
     * {@link computeContextSensitiveRuleFOLLO}.
     * @param {Boolean} exact
     * @returns {org.antlr.runtime.BitSet}
     */
    combineFollows: function(exact) {
        var top = this.state._fsp,
            i,
            localFollowSet,
            followSet = new org.antlr.runtime.BitSet();
        for (i=top; i>=0; i--) {
            localFollowSet = this.state.following[i];
            followSet.orInPlace(localFollowSet);
            if ( exact ) {
                // can we see end of rule?
                if ( localFollowSet.member(org.antlr.runtime.Token.EOR_TOKEN_TYPE) )
                {
                    // Only leave EOR in set if at top (start rule); this lets
                    // us know if have to include follow(start rule); i.e., EOF
                    if ( i>0 ) {
                        followSet.remove(org.antlr.runtime.Token.EOR_TOKEN_TYPE);
                    }
                }
                else { // can't see end of rule, quit
                    break;
                }
            }
        }
        return followSet;
    },

    /** Attempt to recover from a single missing or extra token.
     *
     *  <p>EXTRA TOKEN</p>
     *
     *  <p>LA(1) is not what we are looking for.  If LA(2) has the right token,
     *  however, then assume LA(1) is some extra spurious token.  Delete it
     *  and LA(2) as if we were doing a normal match(), which advances the
     *  input.</p>
     *
     *  <p>MISSING TOKEN</p>
     *
     *  <p>If current token is consistent with what could come after
     *  ttype then it is ok to "insert" the missing token, else throw
     *  exception For example, Input "i=(3;" is clearly missing the
     *  ')'.  When the parser returns from the nested call to expr, it
     *  will have call chain:</p>
     *
     *  <pre><code>  stat -> expr -> atom</code></pre>
     *
     *  <p>and it will be trying to match the ')' at this point in the
     *  derivation:</p>
     *
     *  <pre><code>     => ID '=' '(' INT ')' ('+' atom)* ';'</code></pre>
     *                          ^
     *  <p>match() will see that ';' doesn't match ')' and report a
     *  mismatched token error.  To recover, it sees that LA(1)==';'
     *  is in the set of tokens that can follow the ')' token
     *  reference in rule atom.  It can assume that you forgot the ')'.</p>
     *
     *  @param {org.antlr.runtime.IntStream} input
     *  @param {Number} ttype
     *  @param {org.antlr.runtime.BitSet} follow
     *  @returns {Object}
     */
    recoverFromMismatchedToken: function(input,
                                         ttype,
                                         follow)
    {
        var e = null;
        // if next token is what we are looking for then "delete" this token
        if ( this.mismatchIsUnwantedToken(input, ttype) ) {
            e = new org.antlr.runtime.UnwantedTokenException(ttype, input);
            this.beginResync();
            input.consume(); // simply delete extra token
            this.endResync();
            this.reportError(e);  // report after consuming so AW sees the token in the exception
            // we want to return the token we're actually matching
            var matchedSymbol = this.getCurrentInputSymbol(input);
            input.consume(); // move past ttype token as if all were ok
            return matchedSymbol;
        }
        // can't recover with single token deletion, try insertion
        if ( this.mismatchIsMissingToken(input, follow) ) {
            var inserted = this.getMissingSymbol(input, e, ttype, follow);
            e = new org.antlr.runtime.MissingTokenException(ttype, input, inserted);
            this.reportError(e);  // report after inserting so AW sees the token in the exception
            return inserted;
        }
        // even that didn't work; must throw the exception
        e = new org.antlr.runtime.MismatchedTokenException(ttype, input);
        throw e;
    },

    /**
     * Recover from a mismatched set exception.
     * @param {org.antlr.runtime.IntStream} input
     * @param {org.antlr.runtime.RecognitionException} e
     * @param {org.antlr.runtime.BitSet} follow
     * @returns {Object}
     */
    recoverFromMismatchedSet: function(input,
                                       e,
                                       follow)
    {
        if ( this.mismatchIsMissingToken(input, follow) ) {
            // System.out.println("missing token");
            this.reportError(e);
            // we don't know how to conjure up a token for sets yet
            return this.getMissingSymbol(input, e, org.antlr.runtime.Token.INVALID_TOKEN_TYPE, follow);
        }
        throw e;
    },

    /** Match needs to return the current input symbol, which gets put
     *  into the label for the associated token ref; e.g., x=ID.  Token
     *  and tree parsers need to return different objects. Rather than test
     *  for input stream type or change the IntStream interface, I use
     *  a simple method to ask the recognizer to tell me what the current
     *  input symbol is.
     * 
     *  <p>This is ignored for lexers.</p>
     *  @param {org.antlr.runtime.IntStream} input
     *  @returns {Object}
     */
    getCurrentInputSymbol: function(input) { return null; },

    /** Conjure up a missing token during error recovery.
     *
     *  <p>The recognizer attempts to recover from single missing
     *  symbols. But, actions might refer to that missing symbol.
     *  For example, x=ID {f($x);}. The action clearly assumes
     *  that there has been an identifier matched previously and that
     *  $x points at that token. If that token is missing, but
     *  the next token in the stream is what we want we assume that
     *  this token is missing and we keep going. Because we
     *  have to return some token to replace the missing token,
     *  we have to conjure one up. This method gives the user control
     *  over the tokens returned for missing tokens. Mostly,
     *  you will want to create something special for identifier
     *  tokens. For literals such as '{' and ',', the default
     *  action in the parser or tree parser works. It simply creates
     *  a CommonToken of the appropriate type. The text will be the token.
     *  If you change what tokens must be created by the lexer,
     *  override this method to create the appropriate tokens.</p>
     *
     *  @param {org.antlr.runtime.IntStream} input
     *  @param {org.antlr.runtime.RecognitionException} e
     *  @param {Number} expectedTokenType
     *  @param {org.antlr.runtime.BitSet} follow
     *  @returns {Object}
     */
    getMissingSymbol: function(input,
                               e,
                               expectedTokenType,
                               follow)
    {
        return null;
    },


    /**
     * Consume tokens until one matches the given token or token set
     * @param {org.antlr.runtime.IntStream} input
     * @param {Number|org.antlr.runtime.BitSet} set
     */
    consumeUntil: function(input, set) {
        var ttype = input.LA(1);
        while (ttype != org.antlr.runtime.Token.EOF && !set.member(ttype) ) {
            input.consume();
            ttype = input.LA(1);
        }
    },

    /**
     * Push a rule's follow set using our own hardcoded stack.
     * @param {org.antlr.runtime.BitSet} fset
     */
    pushFollow: function(fset) {
        if ( (this.state._fsp +1)>=this.state.following.length ) {
            var f = [];
            var i;
            for (i=this.state.following.length-1; i>=0; i--) {
                f[i] = this.state.following[i];
            }
            this.state.following = f;
        }
        this.state._fsp++;
        this.state.following[this.state._fsp] = fset;
    },

    /**
     * Sadly JavaScript doesn't provide a robust mechanism for runtime stack reflection.
     * This makes implementing this function impossible without maintaining an auxillary
     * stack data structure, which would be crazy expensive, especially in Lexers.  As such,
     * this method remains unimplemented.
     * @deprecated
     */
    getRuleInvocationStack: function(e, recognizerClassName)
    {
        throw new Error("Not implemented.");
    },

    /**
     * Get this recognizer's backtracking level.
     * @returns {Number} backtracking level
     */
    getBacktrackingLevel: function() {
        return this.state.backtracking;
    },

    /** Used to print out token names like ID during debugging and
     *  error reporting.  The generated parsers implement a method
     *  that overrides this to point to their String[] tokenNames.
     *  @returns {Array} String array of token names.
     */
    getTokenNames: function() {
        return null;
    },

    /** For debugging and other purposes, might want the grammar name.
     *  Have ANTLR generate an implementation for this method.
     *  @returns {String} the grammar name.
     */
    getGrammarFileName: function() {
        return null;
    },

    /** A convenience method for use most often with template rewrites.
     *  Convert an array of Tokens to an array of Strings.
     *  @param {Array} array of org.antlr.runtime.Token objects.
     *  @returns {Array} array of string representations of the argument.
     */
    toStrings: function(tokens) {
        if ( !tokens ) {
            return null;
        }
        var strings = [];
        var i;
        for (i=0; i<tokens.length; i++) {
            strings.push(tokens[i].getText());
        }
        return strings;
    },

    /** Given a rule number and a start token index number, return
     *  MEMO_RULE_UNKNOWN if the rule has not parsed input starting from
     *  start index.  If this rule has parsed input starting from the
     *  start index before, then return where the rule stopped parsing.
     *  It returns the index of the last token matched by the rule.
     *
     *  <p>For now we use a hashtable and just the slow Object-based one.
     *  Later, we can make a special one for ints and also one that
     *  tosses out data after we commit past input position i.</p>
     *  @param {Number} ruleIndex
     *  @param {Number} ruleStartIndex
     *  @returns {Number}
     */
    getRuleMemoization: function(ruleIndex, ruleStartIndex) {
        if ( !this.state.ruleMemo[ruleIndex] ) {
            this.state.ruleMemo[ruleIndex] = {};
        }
        var stopIndexI =
            this.state.ruleMemo[ruleIndex][ruleStartIndex];
        if ( !org.antlr.lang.isNumber(stopIndexI) ) {
            return org.antlr.runtime.BaseRecognizer.MEMO_RULE_UNKNOWN;
        }
        return stopIndexI;
    },

    /** Has this rule already parsed input at the current index in the
     *  input stream?  Return the stop token index or MEMO_RULE_UNKNOWN.
     *  If we attempted but failed to parse properly before, return
     *  MEMO_RULE_FAILED.
     *
     *  <p>This method has a side-effect: if we have seen this input for
     *  this rule and successfully parsed before, then seek ahead to
     *  1 past the stop token matched for this rule last time.</p>
     *  @param {org.antlr.runtime.IntStream} input
     *  @param {Number} ruleIndex
     *  @returns {Boolean}
     */
    alreadyParsedRule: function(input, ruleIndex) {
        var stopIndex = this.getRuleMemoization(ruleIndex, input.index());
        if ( stopIndex==org.antlr.runtime.BaseRecognizer.MEMO_RULE_UNKNOWN ) {
            return false;
        }
        if ( stopIndex==org.antlr.runtime.BaseRecognizer.MEMO_RULE_FAILED ) {
            //System.out.println("rule "+ruleIndex+" will never succeed");
            this.state.failed=true;
        }
        else {
            input.seek(stopIndex+1); // jump to one past stop token
        }
        return true;
    },

    /** Record whether or not this rule parsed the input at this position
     *  successfully.  Use a standard java hashtable for now.
     *  @param {org.antlr.runtime.IntStream} input
     *  @param {Number} ruleIndex
     *  @param {Number} ruleStartIndex
     */
    memoize: function(input,
                      ruleIndex,
                      ruleStartIndex)
    {
        var stopTokenIndex = this.state.failed ? 
            org.antlr.runtime.BaseRecognizer.MEMO_RULE_FAILED : input.index()-1;
        if ( !org.antlr.lang.isValue(this.state.ruleMemo) ) {
            throw new Error("!!!!!!!!! memo array is null for "+ this.getGrammarFileName());
        }
        if ( ruleIndex >= this.state.ruleMemo.length ) {
            throw new Error("!!!!!!!!! memo size is "+this.state.ruleMemo.length+", but rule index is "+ruleIndex);
        }
        if ( org.antlr.lang.isValue(this.state.ruleMemo[ruleIndex]) ) {
            this.state.ruleMemo[ruleIndex][ruleStartIndex] = stopTokenIndex;
        }
    },

    /** return how many rule/input-index pairs there are in total.
     *  TODO: this includes synpreds.
     *  @returns {Number}
     */
    getRuleMemoizationCacheSize: function() {
        var n = 0, i;
        for (i = 0; this.state.ruleMemo && i < this.state.ruleMemo.length; i++) {
            var ruleMap = this.state.ruleMemo[i];
            if ( ruleMap ) {
                // @todo need to get size of rulemap?
                n += ruleMap.length; // how many input indexes are recorded?
            }
        }
        return n;
    },

    /**
     * When a grammar is compiled with the tracing flag enabled, this method is invoked
     * at the start of each rule.
     * @param {String} ruleName the current ruleName
     * @param {Number} ruleIndex
     * @param {Object} inputSymbol
     */
    traceIn: function(ruleName, ruleIndex, inputSymbol)  {
        this.emitErrorMessage("enter "+ruleName+" "+inputSymbol);
        if ( this.state.failed ) {
            this.emitErrorMessage(" failed="+this.failed);
        }
        if ( this.state.backtracking>0 ) {
            this.emitErrorMessage(" backtracking="+this.state.backtracking);
        }
        // System.out.println();
    },

    /**
     * When a grammar is compiled with the tracing flag enabled, this method is invoked
     * at the end of each rule.
     * @param {String} ruleName the current ruleName
     * @param {Number} ruleIndex
     * @param {Object} inputSymbol
     */
    traceOut: function(ruleName, ruleIndex, inputSymbol) {
        this.emitErrorMessage("exit "+ruleName+" "+inputSymbol);
        if ( this.state.failed ) {
            this.emitErrorMessage(" failed="+this.state.failed);
        }
        if ( this.state.backtracking>0 ) {
            this.emitErrorMessage(" backtracking="+this.state.backtracking);
        }
    }
};
