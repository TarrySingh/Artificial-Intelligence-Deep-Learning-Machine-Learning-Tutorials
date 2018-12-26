/**
 * A source of characters for an ANTLR lexer.
 * This class should not be instantiated directly.  Instead, use one of its subclasses.
 * @class
 * @extends org.antlr.runtime.IntStream
 */
org.antlr.runtime.CharStream = function() {};

org.antlr.lang.extend(org.antlr.runtime.CharStream,
                      org.antlr.runtime.IntStream);  

org.antlr.lang.augmentObject(org.antlr.runtime.CharStream,
/** @lends org.antlr.runtime.CharStream */
{
    /**
     * Token type of the EOF character.
     * @type Number
     */
    EOF: -1
});
