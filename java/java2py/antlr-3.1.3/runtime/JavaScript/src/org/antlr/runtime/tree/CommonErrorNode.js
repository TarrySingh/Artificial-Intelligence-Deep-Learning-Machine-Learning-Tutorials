org.antlr.runtime.tree.CommonErrorNode = function(input, start, stop, e) {
    if ( !stop ||
            (stop.getTokenIndex() < start.getTokenIndex() &&
             stop.getType()!=org.antlr.runtime.Token.EOF) )
    {
        // sometimes resync does not consume a token (when LT(1) is
        // in follow set.  So, stop will be 1 to left to start. adjust.
        // Also handle case where start is the first token and no token
        // is consumed during recovery; LT(-1) will return null.
        stop = start;
    }
    this.input = input;
    this.start = start;
    this.stop = stop;
    this.trappedException = e;
};

org.antlr.lang.extend(org.antlr.runtime.tree.CommonErrorNode, org.antlr.runtime.tree.CommonTree, {
    isNil: function() {
        return false;
    },

    getType: function() {
        return org.antlr.runtime.Token.INVALID_TOKEN_TYPE;
    },

    getText: function() {
        var badText = null;
        if ( this.start instanceof org.antlr.runtime.Token ) {
            var i = this.start.getTokenIndex();
            var j = this.stop.getTokenIndex();
            if ( this.stop.getType() === org.antlr.runtime.Token.EOF ) {
                j = this.input.size();
            }
            badText = this.input.toString(i, j);
        }
        else if ( this.start instanceof org.antlr.runtime.tree.Tree ) {
            badText = this.input.toString(this.start, this.stop);
        }
        else {
            // people should subclass if they alter the tree type so this
            // next one is for sure correct.
            badText = "<unknown>";
        }
        return badText;
    },

    toString: function() {
        if ( this.trappedException instanceof org.antlr.runtime.MissingTokenException ) {
            return "<missing type: "+
                   this.trappedException.getMissingType()+
                   ">";
        }
        else if ( this.trappedException instanceof org.antlr.runtime.UnwantedTokenException ) {
            return "<extraneous: "+
                   this.trappedException.getUnexpectedToken()+
                   ", resync="+this.getText()+">";
        }
        else if ( this.trappedException instanceof org.antlr.runtime.MismatchedTokenException ) {
            return "<mismatched token: "+this.trappedException.token+", resync="+this.getText()+">";
        }
        else if ( this.trappedException instanceof org.antlr.runtime.NoViableAltException ) {
            return "<unexpected: "+this.trappedException.token+
                   ", resync="+this.getText()+">";
        }
        return "<error: "+this.getText()+">";
    }
});
