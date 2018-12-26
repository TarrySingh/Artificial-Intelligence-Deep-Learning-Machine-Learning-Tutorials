org.antlr.runtime.CommonToken = function() {
    var oldToken;

    this.charPositionInLine = -1; // set to invalid position
    this.channel = 0; // org.antlr.runtime.CommonToken.DEFAULT_CHANNEL
    this.index = -1;

    if (arguments.length == 1) {
        if (org.antlr.lang.isNumber(arguments[0])) {
            this.type = arguments[0];
        } else {
            oldToken = arguments[0];
            this.text = oldToken.getText();
            this.type = oldToken.getType();
            this.line = oldToken.getLine();
            this.index = oldToken.getTokenIndex();
            this.charPositionInLine = oldToken.getCharPositionInLine();
            this.channel = oldToken.getChannel();
            if ( oldToken instanceof org.antlr.runtime.CommonToken ) {
                this.start = oldToken.start;
                this.stop = oldToken.stop;
            }
        }
    } else if (arguments.length == 2) {
        this.type = arguments[0];
        this.text = arguments[1];
        this.channel = 0; // org.antlr.runtime.CommonToken.DEFAULT_CHANNEL
    } else if (arguments.length == 5) {
        this.input = arguments[0];
        this.type = arguments[1];
        this.channel = arguments[2];
        this.start = arguments[3];
        this.stop = arguments[4];
    }
};

org.antlr.lang.extend(org.antlr.runtime.CommonToken,
                      org.antlr.runtime.Token,
{
    getType: function() {
        return this.type;
    },

    setLine: function(line) {
        this.line = line;
    },

    getText: function() {
        if ( org.antlr.lang.isString(this.text) ) {
            return this.text;
        }
        if ( !this.input ) {
            return null;
        }
        this.text = this.input.substring(this.start,this.stop);
        return this.text;
    },

    /** Override the text for this token.  getText() will return this text
     *  rather than pulling from the buffer.  Note that this does not mean
     *  that start/stop indexes are not valid.  It means that that input
     *  was converted to a new string in the token object.
     */
    setText: function(text) {
        this.text = text;
    },

    getLine: function() {
        return this.line;
    },

    getCharPositionInLine: function() {
        return this.charPositionInLine;
    },

    setCharPositionInLine: function(charPositionInLine) {
        this.charPositionInLine = charPositionInLine;
    },

    getChannel: function() {
        return this.channel;
    },

    setChannel: function(channel) {
        this.channel = channel;
    },

    setType: function(type) {
        this.type = type;
    },

    getStartIndex: function() {
        return this.start;
    },

    setStartIndex: function(start) {
        this.start = start;
    },

    getStopIndex: function() {
        return this.stop;
    },

    setStopIndex: function(stop) {
        this.stop = stop;
    },

    getTokenIndex: function() {
        return this.index;
    },

    setTokenIndex: function(index) {
        this.index = index;
    },

    getInputStream: function() {
        return this.input;
    },

    setInputStream: function(input) {
        this.input = input;
    },

    toString: function() {
        var channelStr = "";
        if ( this.channel>0 ) {
            channelStr=",channel="+this.channel;
        }
        var txt = this.getText();
        if ( !org.antlr.lang.isNull(txt) ) {
            txt = txt.replace(/\n/g,"\\\\n");
            txt = txt.replace(/\r/g,"\\\\r");
            txt = txt.replace(/\t/g,"\\\\t");
        }
        else {
            txt = "<no text>";
        }
        return "[@"+this.getTokenIndex()+","+this.start+":"+this.stop+"='"+txt+"',<"+this.type+">"+channelStr+","+this.line+":"+this.getCharPositionInLine()+"]";
    }
});

/* Monkey patch Token static vars that depend on CommonToken. */
org.antlr.lang.augmentObject(org.antlr.runtime.Token, {
    EOF_TOKEN: new org.antlr.runtime.CommonToken(org.antlr.runtime.CharStream.EOF),
    INVALID_TOKEN: new org.antlr.runtime.CommonToken(0),
    SKIP_TOKEN: new org.antlr.runtime.CommonToken(0)
}, true);
