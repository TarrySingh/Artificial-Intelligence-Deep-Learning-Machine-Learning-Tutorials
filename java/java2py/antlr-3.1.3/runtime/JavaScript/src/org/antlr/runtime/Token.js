// NB: Because Token has static members of type CommonToken, the Token dummy
// constructor is defined in CommonToken.  All methods and vars of Token are
// defined here.  Token is an interface, not a subclass in the Java runtime.

/**
 * @class Abstract base class of all token types.
 * @name Token
 * @memberOf org.antlr.runtime
 */
org.antlr.runtime.Token = function() {};
org.antlr.lang.augmentObject(org.antlr.runtime.Token, /** @lends Token */ {
    EOR_TOKEN_TYPE: 1,

    /** imaginary tree navigation type; traverse "get child" link */
    DOWN: 2,
    /** imaginary tree navigation type; finish with a child list */
    UP: 3,

    MIN_TOKEN_TYPE: 4, // UP+1,

    EOF: org.antlr.runtime.CharStream.EOF,
    EOF_TOKEN: null,

    INVALID_TOKEN_TYPE: 0,
    INVALID_TOKEN: null,

    /** In an action, a lexer rule can set token to this SKIP_TOKEN and ANTLR
     *  will avoid creating a token for this symbol and try to fetch another.
     */
    SKIP_TOKEN: null,

    /** All tokens go to the parser (unless skip() is called in that rule)
     *  on a particular "channel".  The parser tunes to a particular channel
     *  so that whitespace etc... can go to the parser on a "hidden" channel.
     */
    DEFAULT_CHANNEL: 0,

    /** Anything on different channel than DEFAULT_CHANNEL is not parsed
     *  by parser.
     */
    HIDDEN_CHANNEL: 99
});
