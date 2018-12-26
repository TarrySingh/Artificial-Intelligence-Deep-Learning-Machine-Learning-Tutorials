/**
 * A stream of characters created from a JavaScript string that in turn gets
 * fed to a lexer.
 * @class
 * @extends org.antlr.runtime.CharStream
 * @param {String} data the string from which this stream will be created.
 */
org.antlr.runtime.ANTLRStringStream = function(data) {
    /**
     * Location in the stream.
     * Ranges from 0 to (stream length - 1).
     * @private
     * @type Number
     */
    this.p = 0;

    /**
     * The current line in the input.
     * Ranges from 1 to (number of lines).
     * @private
     * @type Number
     */
    this.line = 1;

    /**
     * The index of the character relative to the beginning of the line.
     * Ranges from 0 to (length of line - 1).
     * @private
     * @type Number
     */
    this.charPositionInLine = 0;

    /**
     * Tracks how deep mark() calls are nested
     * @private
     * @type Number
     */
    this.markDepth = 0;

    /**
     * An Array of objects that tracks the stream state
     * values line, charPositionInLine, and p that can change as you
     * move through the input stream.  Indexed from 1..markDepth.
     * A null is kept at index 0.  Created upon first call to mark().
     * @private
     * @type Array
     */
    this.markers = null;

    /**
     * Track the last mark() call result value for use in rewind().
     * @private
     * @type Number
     */
    this.lastMarker = null;

    /**
     * The data being scanned.
     * @private
     * @type String
     */
    this.data = data;

    /**
     * The number of characters in the stream.
     * @private
     * @type Number
     */
    this.n = data.length;
};

org.antlr.lang.extend(org.antlr.runtime.ANTLRStringStream,
                      org.antlr.runtime.CharStream,
/** @lends org.antlr.runtime.ANTLRStringStream.prototype */
{
    /**
     * Reset the stream so that it's in the same state it was
     * when the object was created *except* the data array is not
     * touched.
     */
    reset: function() {
       this.p = 0;
       this.line = 1;
       this.charPositionInLine = 0;
       this.markDepth = 0;
    },

    /**
     * Consume the next character of data in the stream.
     */
    consume: function() {
        if ( this.p < this.n ) {
            this.charPositionInLine++;
            if ( this.data.charAt(this.p)==="\n" ) {
                this.line++;
                this.charPositionInLine=0;
            }
            this.p++;
        }
    },

    /**
     * Get character at current input pointer + i ahead where i=1 is next int.
     * Negative indexes are allowed.  LA(-1) is previous token (token
     * just matched).  LA(-i) where i is before first token should
     * yield -1, invalid char / EOF.
     * @param {Number} i non-zero amount of lookahead or lookback
     * @returns {String|Number} The charcter at the specified position or -1 if
     *      you fell off either end of the stream.
     */
    LA: function(i) {
        if ( i<0 ) {
            i++; // e.g., translate LA(-1) to use offset i=0; then data[p+0-1]
        }

        var new_pos = this.p+i-1;
        if (new_pos>=this.n || new_pos<0) {
            return org.antlr.runtime.CharStream.EOF;
        }
        return this.data.charAt(new_pos);
    },


    /**
     * Return the current input symbol index 0..n where n indicates the
     * last symbol has been read.  The index is the index of char to
     * be returned from LA(1) (i.e. the one about to be consumed).
     * @returns {Number} the index of the current input symbol
     */
    index: function() {
        return this.p;
    },

    /**
     * The length of this stream.
     * @returns {Number} the length of this stream.
     */
    size: function() {
        return this.n;
    },

    /**
     * Tell the stream to start buffering if it hasn't already.  Return
     * current input position, index(), or some other marker so that
     * when passed to rewind() you get back to the same spot.
     * rewind(mark()) should not affect the input cursor.  The Lexer
     * tracks line/col info as well as input index so its markers are
     * not pure input indexes.  Same for tree node streams.
     *
     * <p>Marking is a mechanism for storing the current position of a stream
     * in a stack.  This corresponds with the predictive look-ahead mechanism
     * used in Lexers.</p>
     * @returns {Number} the current size of the mark stack.
     */
    mark: function() {
        if ( !this.markers ) {
            this.markers = [];
            this.markers.push(null); // depth 0 means no backtracking, leave blank
        }
        this.markDepth++;
        var state = null;
        if ( this.markDepth>=this.markers.length ) {
            state = {};
            this.markers.push(state);
        }
        else {
            state = this.markers[this.markDepth];
        }
        state.p = this.p;
        state.line = this.line;
        state.charPositionInLine = this.charPositionInLine;
        this.lastMarker = this.markDepth;
        return this.markDepth;
    },

    /**
     * Rewind to the input position of the last marker.
     * Used currently only after a cyclic DFA and just
     * before starting a sem/syn predicate to get the
     * input position back to the start of the decision.
     * Do not "pop" the marker off the state.  mark(i)
     * and rewind(i) should balance still. It is
     * like invoking rewind(last marker) but it should not "pop"
     * the marker off.  It's like seek(last marker's input position).
     * @param {Number} [m] the index in the mark stack to load instead of the
     *      last.
     */
    rewind: function(m) {
        if (!org.antlr.lang.isNumber(m)) {
            m = this.lastMarker;
        }

        var state = this.markers[m];
        // restore stream state
        this.seek(state.p);
        this.line = state.line;
        this.charPositionInLine = state.charPositionInLine;
        this.release(m);
    },

    /**
     * You may want to commit to a backtrack but don't want to force the
     * stream to keep bookkeeping objects around for a marker that is
     * no longer necessary.  This will have the same behavior as
     * rewind() except it releases resources without the backward seek.
     * This must throw away resources for all markers back to the marker
     * argument.  So if you're nested 5 levels of mark(), and then release(2)
     * you have to release resources for depths 2..5.
     * @param {Number} marker the mark depth above which all mark states will
     *      be released.
     */
    release: function(marker) {
        // unwind any other markers made after m and release m
        this.markDepth = marker;
        // release this marker
        this.markDepth--;
    },

    /**
     * Set the input cursor to the position indicated by index.  This is
     * normally used to seek ahead in the input stream.  No buffering is
     * required to do this unless you know your stream will use seek to
     * move backwards such as when backtracking.
     *
     * <p>This is different from rewind in its multi-directional
     * requirement and in that its argument is strictly an input cursor
     * (index).</p>
     *
     * <p>For char streams, seeking forward must update the stream state such
     * as line number.  For seeking backwards, you will be presumably
     * backtracking using the mark/rewind mechanism that restores state and
     * so this method does not need to update state when seeking backwards.</p>
     *
     * <p>Currently, this method is only used for efficient backtracking using
     * memoization, but in the future it may be used for incremental
     * parsing.</p>
     *
     * <p>The index is 0..n-1.  A seek to position i means that LA(1) will
     * return the ith symbol.  So, seeking to 0 means LA(1) will return the
     * first element in the stream.</p>
     *
     * <p>Esentially this method method moves the input position,
     * {@link #consume}-ing data if necessary.</p>
     *
     * @param {Number} index the position to seek to.
     */
    seek: function(index) {
        if ( index<=this.p ) {
            this.p = index; // just jump; don't update stream state (line, ...)
            return;
        }
        // seek forward, consume until p hits index
        while ( this.p<index ) {
            this.consume();
        }
    },

    /**
     * Retrieve a substring from this stream.
     * @param {Number} start the starting index of the substring (inclusive).
     * @param {Number} stop the last index of the substring (inclusive).
     * @returns {String}
     */
    substring: function(start, stop) {
        return this.data.substr(start,stop-start+1);
    },

    /**
     * Return the current line position in the stream.
     * @returns {Number} the current line position in the stream (1..numlines).
     */
    getLine: function() {
        return this.line;
    },

    /**
     * Get the index of the character relative to the beginning of the line.
     * Ranges from 0 to (length of line - 1).
     * @returns {Number}
     */
    getCharPositionInLine: function() {
        return this.charPositionInLine;
    },

    /**
     * Set the current line in the input stream.
     * This is used internally when performing rewinds.
     * @param {Number} line
     * @private
     */
    setLine: function(line) {
        this.line = line;
    },

    /**
     * Set the index of the character relative to the beginning of the line.
     * Ranges from 0 to (length of line - 1).
     * @param {Number} pos
     * @private
     */
    setCharPositionInLine: function(pos) {
        this.charPositionInLine = pos;
    },

    /** Where are you getting symbols from? Normally, implementations will
     *  pass the buck all the way to the lexer who can ask its input stream
     *  for the file name or whatever.
     */
    getSourceName: function() {
        return null;
    }
});

/**
 * Alias for {@link #LA}.
 * @methodOf org.antlr.runtime.ANTLRStringStream.prototype
 */
org.antlr.runtime.ANTLRStringStream.prototype.LT = org.antlr.runtime.ANTLRStringStream.prototype.LA;
