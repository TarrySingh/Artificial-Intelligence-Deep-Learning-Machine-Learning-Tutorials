/** The set of fields needed by an abstract recognizer to recognize input
 *  and recover from errors etc...  As a separate state object, it can be
 *  shared among multiple grammars; e.g., when one grammar imports another.
 *
 *  These fields are publically visible but the actual state pointer per
 *  parser is protected.
 */
org.antlr.runtime.RecognizerSharedState = function() {
    /** Track the set of token types that can follow any rule invocation.
     *  Stack grows upwards.  When it hits the max, it grows 2x in size
     *  and keeps going.
     */
    this.following = [];

    this._fsp = -1;

    /** This is true when we see an error and before having successfully
     *  matched a token.  Prevents generation of more than one error message
     *  per error.
     */
    this.errorRecovery = false;

    /** The index into the input stream where the last error occurred.
     *  This is used to prevent infinite loops where an error is found
     *  but no token is consumed during recovery...another error is found,
     *  ad naseum.  This is a failsafe mechanism to guarantee that at least
     *  one token/tree node is consumed for two errors.
     */
    this.lastErrorIndex = -1;

    /** In lieu of a return value, this indicates that a rule or token
     *  has failed to match.  Reset to false upon valid token match.
     */
    this.failed = false;

    /** Did the recognizer encounter a syntax error?  Track how many. */
    this.syntaxErrors = 0;

    /** If 0, no backtracking is going on.  Safe to exec actions etc...
     *  If >0 then it's the level of backtracking.
     */
    this.backtracking = 0;

    /** An array[size num rules] of Map<Integer,Integer> that tracks
     *  the stop token index for each rule.  ruleMemo[ruleIndex] is
     *  the memoization table for ruleIndex.  For key ruleStartIndex, you
     *  get back the stop token for associated rule or MEMO_RULE_FAILED.
     *
     *  This is only used if rule memoization is on (which it is by default).
     */
    this.ruleMemo = null;


    // LEXER FIELDS (must be in same state object to avoid casting
    //               constantly in generated code and Lexer object) :(


    /** The goal of all lexer rules/methods is to create a token object.
     *  This is an instance variable as multiple rules may collaborate to
     *  create a single token.  nextToken will return this object after
     *  matching lexer rule(s).  If you subclass to allow multiple token
     *  emissions, then set this to the last token to be matched or
     *  something nonnull so that the auto token emit mechanism will not
     *  emit another token.
     */
    this.token = null;

    /** What character index in the stream did the current token start at?
     *  Needed, for example, to get the text for current token.  Set at
     *  the start of nextToken.
     */
    this.tokenStartCharIndex = -1;

    /** The line on which the first character of the token resides */
    // this.tokenStartLine;

    /** The character position of first character within the line */
    // this.tokenStartCharPositionInLine;

    /** The channel number for the current token */
    // this.channel;

    /** The token type for the current token */
    // this.type;

    /** You can set the text for the current token to override what is in
     *  the input char buffer.  Use setText() or can set this instance var.
     */
    this.text = null;
};
