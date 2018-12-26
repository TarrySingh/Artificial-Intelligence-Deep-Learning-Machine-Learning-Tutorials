/** The most common stream of tokens is one where every token is buffered up
 *  and tokens are prefiltered for a certain channel (the parser will only
 *  see these tokens and cannot change the filter channel number during the
 *  parse).
 *
 *  TODO: how to access the full token stream?  How to track all tokens matched per rule?
 */
org.antlr.runtime.CommonTokenStream = function(tokenSource, channel) {
    this.p = -1;
    this.channel = org.antlr.runtime.Token.DEFAULT_CHANNEL;
    this.v_discardOffChannelTokens = false;

    this.tokens = [];
    if (arguments.length >= 2) {
        this.channel = channel;
    } else if (arguments.length === 1) {
        this.tokenSource = tokenSource;
    }
};

org.antlr.lang.extend(org.antlr.runtime.CommonTokenStream,
                      org.antlr.runtime.TokenStream,       
{
    /** Reset this token stream by setting its token source. */
    setTokenSource: function(tokenSource) {
        this.tokenSource = tokenSource;
        this.tokens = [];
        this.p = -1;
        this.channel = org.antlr.runtime.Token.DEFAULT_CHANNEL;
    },

    /** Load all tokens from the token source and put in tokens.
     *  This is done upon first LT request because you might want to
     *  set some token type / channel overrides before filling buffer.
     */
    fillBuffer: function() {
        var index = 0,
            t = this.tokenSource.nextToken(),
            discard,
            channelI;
        while ( org.antlr.lang.isValue(t) && 
                t.getType()!=org.antlr.runtime.CharStream.EOF )
        {
            discard = false;
            // is there a channel override for token type?
            if ( this.channelOverrideMap ) {
                channelI = this.channelOverrideMap[t.getType()];
                if ( org.antlr.lang.isValue(channelI) ) {
                    t.setChannel(channelI);
                }
            }
            if ( this.discardSet && this.discardSet[t.getType()] )
            {
                discard = true;
            }
            else if ( this.v_discardOffChannelTokens &&
                    t.getChannel()!=this.channel )
            {
                discard = true;
            }
            if ( !discard )    {
                t.setTokenIndex(index);
                this.tokens.push(t);
                index++;
            }
            t = this.tokenSource.nextToken();
        }
        // leave p pointing at first token on channel
        this.p = 0;
        this.p = this.skipOffTokenChannels(this.p);
    },

    /** Move the input pointer to the next incoming token.  The stream
     *  must become active with LT(1) available.  consume() simply
     *  moves the input pointer so that LT(1) points at the next
     *  input symbol. Consume at least one token.
     *
     *  Walk past any token not on the channel the parser is listening to.
     */
    consume: function() {
        if ( this.p<this.tokens.length ) {
            this.p++;
            this.p = this.skipOffTokenChannels(this.p); // leave p on valid token
        }
    },

    /** Given a starting index, return the index of the first on-channel
     *  token.
     */
    skipOffTokenChannels: function(i) {
        var n = this.tokens.length;
        while ( i<n && (this.tokens[i]).getChannel()!=this.channel ) {
            i++;
        }
        return i;
    },

    skipOffTokenChannelsReverse: function(i) {
        while ( i>=0 && (this.tokens[i]).getChannel()!=this.channel ) {
            i--;
        }
        return i;
    },

    /** A simple filter mechanism whereby you can tell this token stream
     *  to force all tokens of type ttype to be on channel.  For example,
     *  when interpreting, we cannot exec actions so we need to tell
     *  the stream to force all WS and NEWLINE to be a different, ignored
     *  channel.
     */
    setTokenTypeChannel: function(ttype, channel) {
        if ( !this.channelOverrideMap ) {
            this.channelOverrideMap = {};
        }
        this.channelOverrideMap[ttype] = channel;
    },

    discardTokenType: function(ttype) {
        if ( !this.discardSet ) {
            this.discardSet = {};
        }
        this.discardSet[ttype] = true;
    },

    discardOffChannelTokens: function(b) {
        this.v_discardOffChannelTokens = b;
    },

    /** Given a start and stop index, return a List of all tokens in
     *  the token type BitSet.  Return null if no tokens were found.  This
     *  method looks at both on and off channel tokens.
     */
    getTokens: function(start, stop, types) {
        if ( this.p === -1 ) {
            this.fillBuffer();
        }

        if (arguments.length===0) {
            return this.tokens;
        }

        if (org.antlr.lang.isArray(types)) {
            types = new org.antlr.runtime.BitSet(types);
        } else if (org.antlr.lang.isNumber(types)) {
            types = org.antlr.runtime.BitSet.of(types);
        }

        if ( stop>=this.tokens.length ) {
            stop=this.tokens.length-1;
        }
        if ( start<0 ) {
            start=0;
        }
        if ( start>stop ) {
            return null;
        }

        // list = tokens[start:stop]:{Token t, t.getType() in types}
        var filteredTokens = [],
            i,
            t;
        for (i=start; i<=stop; i++) {
            t = this.tokens[i];
            if ( !this.types || types.member(t.getType()) ) {
                filteredTokens.push(t);
            }
        }
        if ( filteredTokens.length===0 ) {
            filteredTokens = null;
        }
        return filteredTokens;
    },

    /** Get the ith token from the current position 1..n where k=1 is the
     *  first symbol of lookahead.
     */
    LT: function(k) {
        if ( this.p === -1 ) {
            this.fillBuffer();
        }
        if ( k===0 ) {
            return null;
        }
        if ( k<0 ) {
            return this.LB(-1*k);
        }
        if ( (this.p+k-1) >= this.tokens.length ) {
            return org.antlr.runtime.Token.EOF_TOKEN;
        }
        var i = this.p,
            n = 1;
        // find k good tokens
        while ( n<k ) {
            // skip off-channel tokens
            i = this.skipOffTokenChannels(i+1); // leave p on valid token
            n++;
        }
        if ( i>=this.tokens.length ) {
            return org.antlr.runtime.Token.EOF_TOKEN;
        }
        return this.tokens[i];
    },

    /** Look backwards k tokens on-channel tokens */
    LB: function(k) {
        if ( this.p === -1 ) {
            this.fillBuffer();
        }
        if ( k===0 ) {
            return null;
        }
        if ( (this.p-k)<0 ) {
            return null;
        }

        var i = this.p,
            n = 1;
        // find k good tokens looking backwards
        while ( n<=k ) {
            // skip off-channel tokens
            i = this.skipOffTokenChannelsReverse(i-1); // leave p on valid token
            n++;
        }
        if ( i<0 ) {
            return null;
        }
        return this.tokens[i];
    },

    /** Return absolute token i; ignore which channel the tokens are on;
     *  that is, count all tokens not just on-channel tokens.
     */
    get: function(i) {
        return this.tokens[i];
    },

    LA: function(i) {
        return this.LT(i).getType();
    },

    mark: function() {
        if ( this.p === -1 ) {
            this.fillBuffer();
        }
        this.lastMarker = this.index();
        return this.lastMarker;
    },

    release: function(marker) {
        // no resources to release
    },

    size: function() {
        return this.tokens.length;
    },

    index: function() {
        return this.p;
    },

    rewind: function(marker) {
        if (!org.antlr.lang.isNumber(marker)) {
            marker = this.lastMarker;
        }
        this.seek(marker);
    },

    reset: function() {
        this.p = 0;
        this.lastMarker = 0;
    },

    seek: function(index) {
        this.p = index;
    },

    getTokenSource: function() {
        return this.tokenSource;
    },

    getSourceName: function() {
        return this.getTokenSource().getSourceName();
    },

    toString: function(start, stop) {
        if (arguments.length===0) {
            if ( this.p === -1 ) {
                this.fillBuffer();
            }
            start = 0;
            stop = this.tokens.length-1;
        }

        if (!org.antlr.lang.isNumber(start) && !org.antlr.lang.isNumber(stop)) {
            if ( org.antlr.lang.isValue(start) && org.antlr.lang.isValue(stop) ) {
                start = start.getTokenIndex();
                stop = stop.getTokenIndex();
            } else {
                return null;
            }
        }

        var buf = "",
            i;
 
        if ( start<0 || stop<0 ) {
            return null;
        }
        if ( this.p == -1 ) {
            this.fillBuffer();
        }
        if ( stop>=this.tokens.length ) {
            stop = this.tokens.length-1;
        }
        for (i = start; i <= stop; i++) {
            t = this.tokens[i];
            buf = buf + this.tokens[i].getText();
        }
        return buf;
    }
});
